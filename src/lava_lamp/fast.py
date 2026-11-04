"""Numba versions of the simulation step and the point splat.

Same physics as :func:`lava_lamp.sph.step` (Jacobi order: every particle
reads the start-of-step state), but each pass is one compiled loop over the
particles, run in parallel. Neighbours are found with a cell list (cells
``h`` wide, so each particle checks its own and the 8 surrounding cells)
instead of a KD-tree.

Each particle sums the forces from all its neighbours itself, so every pair
is computed twice (once from each end), but no two threads write to the
same particle. Results match the NumPy version to rounding error; the sums
are in a different order, so they're not bit-identical.

Functions are compiled on first use and cached on disk (``cache=True``), so
only the first run after a change pays the compile time. They release the
GIL (``nogil=True``), so other Python threads, such as the display driver's
serial workers, can run while they do.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .render import INFLUENCE
from .sph import (
    BIG,
    COHESION,
    HEAT_BASE,
    HEAT_FALLOFF,
    NORMAL_EPS,
    PARAM_NAMES,
    VISC_EPS,
    WALL_WAX_FACTOR,
    WATER_ST_NUMERATOR,
    LavaParams,
)

# parameter vector indices (LavaParams.vector() order)
(
    GRAVITY,
    H,
    STIFFNESS,
    VISC_WATER,
    VISC_WAX,
    ST_RATIO,
    TEMP_SPEED,
    MAX_VEL,
    GOAL_HOT,
    GOAL_COLD,
    AMBIENT_RATE,
    COND_RATE,
    HEATER_HEIGHT,
    DT,
) = range(14)
assert PARAM_NAMES == (
    "gravity",
    "h",
    "stiffness",
    "visc_water",
    "visc_wax",
    "st_ratio",
    "temp_speed",
    "max_vel",
    "goal_hot",
    "goal_cold",
    "ambient_rate",
    "cond_rate",
    "heater_height",
    "dt",
), "update the indices in fast.py to match LavaParams"


# --------------------------------------------------------------------------
# Cell list
# --------------------------------------------------------------------------
@njit(cache=True, nogil=True)
def build_cells(x, y, x0, y0, cell, nx, ny):
    """Sort points into a grid of cells.

    Returns ``(start, order)``: the points in cell ``c = iy * nx + ix`` are
    ``order[start[c]:start[c + 1]]``. Points outside the grid go in the
    nearest edge cell.
    """
    n = x.shape[0]
    cidx = np.empty(n, np.int64)
    start = np.zeros(nx * ny + 1, np.int64)
    for i in range(n):
        ix = min(max(int(np.floor((x[i] - x0) / cell)), 0), nx - 1)
        iy = min(max(int(np.floor((y[i] - y0) / cell)), 0), ny - 1)
        c = iy * nx + ix
        cidx[i] = c
        start[c + 1] += 1
    for c in range(nx * ny):
        start[c + 1] += start[c]
    fill = start[:-1].copy()
    order = np.empty(n, np.int64)
    for i in range(n):
        c = cidx[i]
        order[fill[c]] = i
        fill[c] += 1
    return start, order


# --------------------------------------------------------------------------
# Simulation step
# --------------------------------------------------------------------------
@njit(parallel=True, cache=True, nogil=True)
def _heat_and_gravity(S, p, width, height):
    n = S.shape[0]
    out = np.empty_like(S)
    for i in prange(n):
        x, y, vx, vy, m, mat = (
            S[i, 0],
            S[i, 1],
            S[i, 2],
            S[i, 3],
            S[i, 4],
            S[i, 5],
        )
        out[i, 0], out[i, 1], out[i, 2], out[i, 5] = x, y, vx, mat
        out[i, 3] = vy + p[GRAVITY] * m * p[DT]  # gravity uses the old "mass"
        if mat > 0.5:
            rate = (
                (HEAT_BASE - abs(x - width / 2) / width * HEAT_FALLOFF)
                * p[TEMP_SPEED]
                * p[DT]
            )
            goal = p[GOAL_HOT] if y > height - p[HEATER_HEIGHT] else m
            m1 = m + (goal - m) * rate
            if m1 < p[GOAL_COLD]:
                m1 += p[AMBIENT_RATE] * rate
            out[i, 4] = m1
        else:
            out[i, 4] = m
    return out


@njit(cache=True, nogil=True)
def _boundary(px, py, vx, vy, is_wax, poly, is_ccw, h, stiffness, dt):
    """Scalar version of :func:`lava_lamp.sph.apply_boundary`."""
    nv = poly.shape[0]
    mx, my = vx * dt, vy * dt  # the move

    # 1. inside test (ray casting, crossing parity)
    inside = False
    for i in range(nv):
        j = i - 1 if i > 0 else nv - 1  # previous vertex, as in the JS
        xi, yi, xj, yj = poly[i, 0], poly[i, 1], poly[j, 0], poly[j, 1]
        if yi == yj:
            continue
        if (yi > py) != (yj > py):
            if px < (xj - xi) * (py - yi) / (yj - yi) + xi:
                inside = not inside

    xs, ys = px, py
    if not inside:
        # 2. snap to the first wall crossing along the step p0 -> p,
        #    or else to the nearest point on the outline
        p0x, p0y = px - mx, py - my
        best_t, found = BIG, False
        best_d, snap_x, snap_y = BIG, px, py
        for i in range(nv):
            ax, ay = poly[i, 0], poly[i, 1]
            k = (i + 1) % nv
            ex, ey = poly[k, 0] - ax, poly[k, 1] - ay
            denom = -ex * my + mx * ey
            if denom != 0:
                s = (-my * (p0x - ax) + mx * (p0y - ay)) / denom
                t = (ex * (p0y - ay) - ey * (p0x - ax)) / denom
                if 0 <= s <= 1 and 0 <= t <= 1:
                    found = True
                    best_t = min(best_t, t)
            l2 = ex * ex + ey * ey
            if l2 == 0:
                continue
            t = min(1.0, max(0.0, ((px - ax) * ex + (py - ay) * ey) / l2))
            qx, qy = ax + t * ex, ay + t * ey
            d = np.hypot(px - qx, py - qy)
            if d < best_d:
                best_d, snap_x, snap_y = d, qx, qy
        if found:
            xs, ys = p0x + best_t * mx, p0y + best_t * my
        else:
            xs, ys = snap_x, snap_y

    # 3. SPH-style wall repulsion (changes velocity for the next step)
    wall_k = stiffness * (WALL_WAX_FACTOR if is_wax else 1.0)
    for i in range(nv):
        ax, ay = poly[i, 0], poly[i, 1]
        k = (i + 1) % nv
        ex, ey = poly[k, 0] - ax, poly[k, 1] - ay
        l2 = ex * ex + ey * ey
        if l2 == 0:
            continue
        t = min(1.0, max(0.0, ((xs - ax) * ex + (ys - ay) * ey) / l2))
        dx, dy = xs - (ax + t * ex), ys - (ay + t * ey)
        d = np.hypot(dx, dy)
        if d < h:
            f = wall_k * (1 - d / h) ** 2 * dt
            if d > NORMAL_EPS:
                vx += f * dx / d
                vy += f * dy / d
            else:
                el = np.sqrt(l2)
                if is_ccw:
                    vx += f * -ey / el
                    vy += f * ex / el
                else:
                    vx += f * ey / el
                    vy += f * -ex / el
    return xs, ys, vx, vy


@njit(parallel=True, cache=True, nogil=True)
def _interact_and_move(S, p, poly, is_ccw, start, order, x0, y0, nx, ny):
    n = S.shape[0]
    h = p[H]
    h2 = h * h
    out = np.empty_like(S)
    for i in prange(n):
        xi, yi, vxi, vyi, mi, mati = (
            S[i, 0],
            S[i, 1],
            S[i, 2],
            S[i, 3],
            S[i, 4],
            S[i, 5],
        )
        wax_i = mati > 0.5
        if wax_i:
            st_i, visc_i = p[ST_RATIO], p[VISC_WAX]
        else:
            st_i, visc_i = WATER_ST_NUMERATOR / p[ST_RATIO], p[VISC_WATER]
        fx = 0.0
        fy = 0.0
        dm = 0.0
        cx = min(max(int(np.floor((xi - x0) / h)), 0), nx - 1)
        cy = min(max(int(np.floor((yi - y0) / h)), 0), ny - 1)
        for gy in range(max(cy - 1, 0), min(cy + 2, ny)):
            for gx in range(max(cx - 1, 0), min(cx + 2, nx)):
                c = gy * nx + gx
                for kk in range(start[c], start[c + 1]):
                    j = order[kk]
                    dx = xi - S[j, 0]
                    dy = yi - S[j, 1]
                    r2 = dx * dx + dy * dy
                    if r2 >= h2 or r2 <= 0:
                        continue
                    r = np.sqrt(r2)
                    q = r / h
                    wax_j = S[j, 5] > 0.5
                    pressure = p[STIFFNESS] * (1 - q) ** 2
                    f = pressure
                    if wax_i == wax_j:
                        f -= COHESION * st_i
                        dot = (vxi - S[j, 2]) * dx + (vyi - S[j, 3]) * dy
                        if dot > 0:
                            f -= dot / (r2 + VISC_EPS * h2) * visc_i * (1 - q)
                        # wax-wax conduction is one-sided in the JS
                        if wax_i and j < i:
                            dm += S[j, 4] - mi
                    fx += f * dx / r
                    fy += f * dy / r
        dt = p[DT]
        vx1 = vxi + fx * dt
        vy1 = vyi + fy * dt
        speed = np.hypot(vx1, vy1)
        if speed > p[MAX_VEL]:
            fac = p[MAX_VEL] / max(speed, 1e-12)
            vx1 *= fac
            vy1 *= fac
        xs, ys, vx3, vy3 = _boundary(
            xi + vx1 * dt,
            yi + vy1 * dt,
            vx1,
            vy1,
            wax_i,
            poly,
            is_ccw,
            h,
            p[STIFFNESS],
            dt,
        )
        out[i, 0], out[i, 1], out[i, 2], out[i, 3] = xs, ys, vx3, vy3
        out[i, 4] = mi + p[COND_RATE] * dt * dm
        out[i, 5] = mati
    return out


def _is_ccw(poly):
    nv = len(poly)
    area = sum(
        poly[i, 0] * poly[(i + 1) % nv, 1] - poly[(i + 1) % nv, 0] * poly[i, 1]
        for i in range(nv)
    )
    return area > 0


def step(
    S: np.ndarray,
    P: LavaParams,
    polygon: np.ndarray,
    width: float = 400.0,
    height: float = 600.0,
) -> np.ndarray:
    """Numba version of :func:`lava_lamp.sph.step`."""
    p = P.vector()
    poly = np.ascontiguousarray(polygon, dtype=float)
    Sp = _heat_and_gravity(np.ascontiguousarray(S), p, width, height)
    # cell grid over the polygon's bounding box, cells h wide
    x0, y0 = poly.min(axis=0)
    x1, y1 = poly.max(axis=0)
    nx = max(int(np.ceil((x1 - x0) / P.h)), 1)
    ny = max(int(np.ceil((y1 - y0) / P.h)), 1)
    start, order = build_cells(Sp[:, 0], Sp[:, 1], x0, y0, P.h, nx, ny)
    return _interact_and_move(
        Sp, p, poly, _is_ccw(poly), start, order, x0, y0, nx, ny
    )


# --------------------------------------------------------------------------
# Point splat (renderer)
# --------------------------------------------------------------------------
@njit(parallel=True, cache=True, nogil=True)
def splat(points, pos, values, radius, start, order, x0, y0, nx, ny):
    """Kernel-weighted sums of particle ``values`` (N, k) at ``points``.

    The particles must be sorted into cells ``radius`` wide (``start``,
    ``order`` from :func:`build_cells`).
    """
    m = points.shape[0]
    k = values.shape[1]
    out = np.zeros((m, k))
    r2max = radius * radius
    for i in prange(m):
        px, py = points[i, 0], points[i, 1]
        cx = min(max(int(np.floor((px - x0) / radius)), 0), nx - 1)
        cy = min(max(int(np.floor((py - y0) / radius)), 0), ny - 1)
        for gy in range(max(cy - 1, 0), min(cy + 2, ny)):
            for gx in range(max(cx - 1, 0), min(cx + 2, nx)):
                c = gy * nx + gx
                for kk in range(start[c], start[c + 1]):
                    j = order[kk]
                    dx = px - pos[j, 0]
                    dy = py - pos[j, 1]
                    d2 = dx * dx + dy * dy
                    if d2 > r2max:
                        continue
                    w = INFLUENCE * (1 - np.sqrt(d2) / radius) ** 3
                    for c2 in range(k):
                        out[i, c2] += w * values[j, c2]
    return out
