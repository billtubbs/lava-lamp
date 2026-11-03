"""
lava_lamp/sph.py
================

Python port of the particle physics in the browser "Lava Lamp Simulator"
(https://lava-lamp-simulation.web.app/, JavaScript function ``updateParticles``).

Credit: the original simulator, its physics model and its rendering are the
work of the author of the r/Simulated post "[OC] Lava lamp simulation using
SPH fluid dynamics":
https://www.reddit.com/r/Simulated/comments/1t11plc/oc_lava_lamp_simulation_using_sph_fluid_dynamics/
The original has no licence notice; this port is for personal study.

The simulation is a type of simplified smoothed particle hydrodynamics (SPH)
particle model:

* pairwise soft repulsion  ``stiffness * (1 - q)^2``          (q = r / h)
* same-material cohesion   ``- 0.04 * st_factor``             ("surface tension")
* same-material viscosity  radial impulse damping separating pairs
* temperature stored as "mass": gravity acts as ``vy += g * m``, the heater at
  the bottom drives wax towards ``m = goal_hot`` (< 0, buoyant) and ambient
  cooling drives it back towards ``m = goal_cold`` (> 0, sinks)
* symplectic Euler with dt = 1 frame, velocity clamp, lamp-shaped polygon
  boundary (snap-back + SPH-style wall repulsion)

Implementation
--------------
Vectorised NumPy over a pair list: ``cKDTree.query_pairs`` finds every pair
closer than ``h`` once, the pair force is computed once per pair, and it is
added to both particles with opposite signs using ``np.bincount``.  The lamp
outline has only a few edges, so the boundary code loops over edges and
vectorises over particles.

State layout: ``[x, y, vx, vy, m, mat]`` with ``mat = 0`` for water and
``mat = 1`` for wax.  :class:`LavaLampSim` exposes it as a (6, N) array; the
functions below work on the (N, 6) array.  Units are canvas pixels and
frames, as in the original.

Differences from the JavaScript
-------------------------------
The JS loop updates particles in place (Gauss-Seidel-like: a particle sees
velocities/positions already modified earlier in the same frame).  This port
evaluates all interactions from the start-of-step state (Jacobi).  Every
formula and constant is otherwise taken directly from the JS.  Mouse
interaction, the debugging "balls" and the deprecated source/drain modes are
not ported.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------
# Constants hard-coded in the JavaScript
# --------------------------------------------------------------------------
STATE_NAMES = ("x", "y", "vx", "vy", "m", "mat")
NX = len(STATE_NAMES)
WATER, WAX = 0.0, 1.0

COHESION = 0.04  # pressure -= 0.04 * stFactor
WATER_ST_NUMERATOR = 2.5  # water stFactor = 2.5 / ST_RATIO
VISC_EPS = 0.01  # impulse denominator: distSq + 0.01 * H^2
HEAT_BASE = 0.002  # heater rate at the centre line
HEAT_FALLOFF = 0.0015  # rate reduction towards the side walls
WALL_WAX_FACTOR = 1.25  # wax is pushed harder by the walls
NORMAL_EPS = 1e-7  # wall-distance threshold for the fallback normal
BIG = 1e30


@dataclass
class LavaParams:
    """Run-time parameters (can be changed between steps)."""

    gravity: float = 0.1
    h: float = 17.0  # interaction (smoothing) radius [px]
    stiffness: float = 2.5  # pair repulsion strength
    visc_water: float = 0.3  # VISCOSITY (water-water pairs)
    visc_wax: float = 0.3  # particle.viscosity of wax (= VISCOSITY at init)
    st_ratio: float = 2.5  # "Surface Tension" slider
    temp_speed: float = 1.0  # "Heat Transfer" slider
    max_vel: float = 4.0  # velocity clamp [px/frame]
    goal_hot: float = -6.0  # "mass" wax relaxes to on the heater
    goal_cold: float = 1.0  # "mass" wax relaxes to through ambient cooling
    ambient_rate: float = 0.25
    cond_rate: float = 1e-5  # wax-wax conduction rate
    heater_height: float = 10.0  # heater zone thickness above the bottom [px]

    def vector(self) -> np.ndarray:
        return np.array([float(getattr(self, f.name)) for f in fields(self)])


PARAM_NAMES = tuple(f.name for f in fields(LavaParams))
NP = len(PARAM_NAMES)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def lamp_polygon(width: float = 400.0, height: float = 600.0) -> np.ndarray:
    """The lamp outline used in the original (BORDER_POLY), shape (n, 2)."""
    top, mid, bot = 175 / 1.25, 340 / 1.25, 280 / 1.25
    cx, ym = width / 2, 0.8 * height
    return np.array(
        [
            [cx - top / 2, 0.0],
            [cx + top / 2, 0.0],
            [cx + mid / 2, ym],
            [cx + bot / 2, height],
            [cx - bot / 2, height],
            [cx - mid / 2, ym],
        ]
    )


def square_polygon(size: float) -> np.ndarray:
    """A square domain ``[0, size] x [0, size]``, shape (4, 2)."""
    return np.array([[0.0, 0.0], [size, 0.0], [size, size], [0.0, size]])


# Particles per px^2 inside the original lamp: 3,000 particles over the lamp's
# area of about 128,600 px^2 (the initial grid fills the 400 x 600 canvas and
# the particles outside the lamp are snapped inside on the first step).
LAMP_DENSITY = 3000 / 128_640


def initial_state(
    n: int = 3000,
    wax_ratio: float = 0.2,
    width: float = 400.0,
    height: float = 600.0,
    seed: int | None = None,
    wax_at: str = "top",
) -> np.ndarray:
    """Replicates ``initParticles``: jittered grid, ``wax_ratio`` of it wax.

    The grid fills the canvas row by row from the top. ``wax_at="top"`` makes
    the first rows wax, as in the original (it then falls through the water);
    ``"bottom"`` the last rows, so the wax starts resting on the heater.
    """
    rng = np.random.default_rng(seed)
    R = height / width
    c_ideal = np.sqrt(n / R)
    c1, c2 = int(np.floor(c_ideal)), int(np.ceil(c_ideal))
    r1, r2 = int(np.ceil(n / c1)), int(np.ceil(n / c2))
    # (sic) the JS compares row counts with the aspect ratio
    if abs(r1 - R) < abs(r2 - R):
        per_row, per_col = r1, c1
    else:
        per_row, per_col = r2, c2
    i = np.arange(n)
    x = ((i % per_row) + (rng.random(n) - 0.5) / 5) * (width / per_row)
    y = ((i // per_row) + (rng.random(n) - 0.5) / 5) * (height / per_col)
    if wax_at == "top":
        wax = i < n * wax_ratio
    elif wax_at == "bottom":
        wax = i >= n * (1 - wax_ratio)
    else:
        raise ValueError(f"unknown wax_at {wax_at!r}")
    s = np.zeros((NX, n))
    s[0], s[1] = x, y
    s[4] = np.where(wax, 1.5, 0.0)
    s[5] = np.where(wax, WAX, WATER)
    return s


# --------------------------------------------------------------------------
# Physics (all functions take and return the (N, 6) state array)
# --------------------------------------------------------------------------
def heat_and_gravity(
    S: np.ndarray, P: LavaParams, width: float = 400.0, height: float = 600.0
) -> np.ndarray:
    """Gravity, then the heater and ambient cooling of the wax."""
    x, y, vx, vy, m, mat = S.T
    out = S.copy()
    out[:, 3] = vy + P.gravity * m  # gravity uses the old "mass"
    rate = (
        HEAT_BASE - np.abs(x - width / 2) / width * HEAT_FALLOFF
    ) * P.temp_speed
    goal = np.where(y > height - P.heater_height, P.goal_hot, m)
    m1 = m + (goal - m) * rate
    m1 = np.where(m1 < P.goal_cold, m1 + P.ambient_rate * rate, m1)
    out[:, 4] = np.where(mat > 0.5, m1, m)
    return out


def apply_boundary(
    px, py, vx, vy, is_wax, polygon: np.ndarray, h: float, stiffness: float
):
    """Port of ``applyBorderForce``: snap back inside, then wall repulsion.

    ``px, py`` are the positions after the move and ``vx, vy`` the velocities
    used for the move.  Returns the new ``(px, py, vx, vy)``.
    """
    poly = np.asarray(polygon, dtype=float)
    nv = len(poly)
    area = sum(
        poly[i, 0] * poly[(i + 1) % nv, 1] - poly[(i + 1) % nv, 0] * poly[i, 1]
        for i in range(nv)
    )
    is_ccw = area > 0

    # 1. inside test (ray casting, crossing parity)
    crossings = np.zeros(px.shape, dtype=int)
    for i in range(nv):
        (xi, yi), (xj, yj) = (
            poly[i],
            poly[i - 1],
        )  # previous vertex, as in the JS
        if yi == yj:
            continue  # horizontal edge never straddles
        x_int = (xj - xi) * (py - yi) / (yj - yi) + xi
        crossings += ((yi > py) != (yj > py)) & (px < x_int)
    inside = crossings % 2 == 1

    # 2. outside: snap to the first wall crossing along the step p0 -> p,
    #    or else to the nearest point on the outline
    p0x, p0y = px - vx, py - vy
    best_t = np.full(px.shape, BIG)
    found = np.zeros(px.shape, dtype=bool)
    best_d = np.full(px.shape, BIG)
    snap_x, snap_y = px.copy(), py.copy()
    for i in range(nv):
        a, b = poly[i], poly[(i + 1) % nv]
        ex, ey = b - a
        denom = -ex * vy + vx * ey
        nz = denom != 0
        d_safe = np.where(nz, denom, 1.0)
        s = (-vy * (p0x - a[0]) + vx * (p0y - a[1])) / d_safe
        t = (ex * (p0y - a[1]) - ey * (p0x - a[0])) / d_safe
        valid = nz & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
        best_t = np.where(valid & (t < best_t), t, best_t)
        found |= valid

        l2 = ex * ex + ey * ey
        if l2 == 0:
            continue
        t = np.clip(((px - a[0]) * ex + (py - a[1]) * ey) / l2, 0, 1)
        qx, qy = a[0] + t * ex, a[1] + t * ey
        d = np.hypot(px - qx, py - qy)
        better = d < best_d
        best_d = np.where(better, d, best_d)
        snap_x = np.where(better, qx, snap_x)
        snap_y = np.where(better, qy, snap_y)
    t_hit = np.where(found, best_t, 0.0)
    xs = np.where(inside, px, np.where(found, p0x + t_hit * vx, snap_x))
    ys = np.where(inside, py, np.where(found, p0y + t_hit * vy, snap_y))

    # 3. SPH-style wall repulsion (changes velocity for the next step)
    wall_k = stiffness * np.where(is_wax, WALL_WAX_FACTOR, 1.0)
    for i in range(nv):
        a, b = poly[i], poly[(i + 1) % nv]
        ex, ey = b - a
        l2 = ex * ex + ey * ey
        if l2 == 0:
            continue
        el = np.sqrt(l2)
        n_in_x, n_in_y = (-ey / el, ex / el) if is_ccw else (ey / el, -ex / el)
        t = np.clip(((xs - a[0]) * ex + (ys - a[1]) * ey) / l2, 0, 1)
        dx, dy = xs - (a[0] + t * ex), ys - (a[1] + t * ey)
        d = np.hypot(dx, dy)
        f = np.where(d < h, wall_k * (1 - d / h) ** 2, 0.0)
        far = d > NORMAL_EPS
        d_safe = np.maximum(d, NORMAL_EPS)
        vx = vx + f * np.where(far, dx / d_safe, n_in_x)
        vy = vy + f * np.where(far, dy / d_safe, n_in_y)
    return xs, ys, vx, vy


def interact(
    S: np.ndarray, P: LavaParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair forces and wax-wax heat conduction.

    Returns the new ``(vx, vy, m)`` before the velocity clamp.
    """
    x, y, vx, vy, m, mat = S.T
    n, h = len(S), P.h
    ij = cKDTree(S[:, :2]).query_pairs(h, output_type="ndarray")
    i, j = ij[:, 0], ij[:, 1]  # i < j
    dx, dy = x[i] - x[j], y[i] - y[j]
    r2 = dx * dx + dy * dy
    keep = (r2 < h * h) & (r2 > 0)
    i, j, dx, dy, r2 = i[keep], j[keep], dx[keep], dy[keep], r2[keep]
    r = np.sqrt(r2)
    q = r / h
    wax_i, wax_j = mat[i] > 0.5, mat[j] > 0.5
    same = wax_i == wax_j

    # pressure + cohesion ("surface tension"), symmetric within one material
    st = same * np.where(wax_i, P.st_ratio, WATER_ST_NUMERATOR / P.st_ratio)
    pressure = P.stiffness * (1 - q) ** 2 - COHESION * st

    # viscosity (only for separating pairs, only within one material)
    visc = same * np.where(wax_i, P.visc_wax, P.visc_water)
    dot = (vx[i] - vx[j]) * dx + (vy[i] - vy[j]) * dy
    impulse = (dot > 0) * dot / (r2 + VISC_EPS * h * h) * visc * (1 - q)

    # equal and opposite: +f on i, -f on j (the unit vector flips sign)
    f = pressure - impulse
    fx, fy = f * dx / r, f * dy / r
    vx1 = vx + np.bincount(i, fx, n) - np.bincount(j, fx, n)
    vy1 = vy + np.bincount(i, fy, n) - np.bincount(j, fy, n)

    # wax-wax conduction is one-sided in the JS: only the higher index changes
    ww = wax_i & wax_j
    m1 = m + P.cond_rate * np.bincount(j[ww], m[i[ww]] - m[j[ww]], n)
    return vx1, vy1, m1


def step(
    S: np.ndarray,
    P: LavaParams,
    polygon: np.ndarray,
    width: float = 400.0,
    height: float = 600.0,
) -> np.ndarray:
    """Advance the (N, 6) state by one frame and return the new state."""
    S = heat_and_gravity(S, P, width, height)
    vx1, vy1, m1 = interact(S, P)

    # velocity clamp
    speed = np.hypot(vx1, vy1)
    fac = np.where(
        speed > P.max_vel, P.max_vel / np.maximum(speed, 1e-12), 1.0
    )
    vx2, vy2 = vx1 * fac, vy1 * fac

    # integrate (dt = 1 frame) and apply the lamp boundary
    x, y, mat = S[:, 0], S[:, 1], S[:, 5]
    xs, ys, vx3, vy3 = apply_boundary(
        x + vx2, y + vy2, vx2, vy2, mat > 0.5, polygon, P.h, P.stiffness
    )
    return np.column_stack([xs, ys, vx3, vy3, m1, mat])


# --------------------------------------------------------------------------
# Simulation driver
# --------------------------------------------------------------------------
class LavaLampSim:
    """Holds the particle state and steps it.

    Parameters
    ----------
    n_particles, wax_ratio : initial population (defaults as in the original)
    params      : :class:`LavaParams` (can be changed between steps)
    width, height, polygon : canvas size and lamp outline
    seed        : random seed for the initial jitter
    state       : optional initial (6, N) state instead of ``initial_state``
    wax_at      : "top" (as in the original) or "bottom", see :func:`initial_state`
    backend     : "numpy" (:func:`step`) or "numba" (:func:`lava_lamp.fast.step`,
                  needs numba)
    """

    def __init__(
        self,
        n_particles: int = 3000,
        wax_ratio: float = 0.2,
        params: LavaParams | None = None,
        width: float = 400.0,
        height: float = 600.0,
        polygon: np.ndarray | None = None,
        seed: int | None = None,
        state: np.ndarray | None = None,
        backend: str = "numpy",
        wax_at: str = "top",
    ):
        self.params = params or LavaParams()
        if backend == "numba":
            from .fast import step as step_fn
        elif backend == "numpy":
            step_fn = step
        else:
            raise ValueError(f"unknown backend {backend!r}")
        self.backend, self._step_fn = backend, step_fn
        self.width, self.height = width, height
        self.polygon = (
            lamp_polygon(width, height)
            if polygon is None
            else np.asarray(polygon)
        )
        s0 = (
            initial_state(n_particles, wax_ratio, width, height, seed, wax_at)
            if state is None
            else np.asarray(state, dtype=float)
        )
        self.n = s0.shape[1]
        self._s = np.ascontiguousarray(s0.T)  # (N, 6)
        self.frame = 0

    @property
    def state(self) -> np.ndarray:
        """State as a (6, N) array (rows x, y, vx, vy, m, mat). A view."""
        return self._s.T

    @state.setter
    def state(self, value: np.ndarray):
        self._s[...] = np.asarray(value, dtype=float).T

    def step(self, n_steps: int = 1) -> np.ndarray:
        for _ in range(n_steps):
            self._s = self._step_fn(
                self._s, self.params, self.polygon, self.width, self.height
            )
            self.frame += 1
        return self.state

    # convenience views
    @property
    def positions(self) -> np.ndarray:
        return self.state[:2].T

    @property
    def is_wax(self) -> np.ndarray:
        return self.state[5] > 0.5

    def temperature(self) -> np.ndarray:
        """Normalised temperature used for colouring: 0 cold, 1 hot (tempData)."""
        P = self.params
        return 1.0 - (self.state[4] - P.goal_hot) / (P.goal_cold - P.goal_hot)
