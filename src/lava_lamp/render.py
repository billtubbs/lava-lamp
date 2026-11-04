"""Colour a set of points (e.g. LED centres) from the particle state.

A point-sampled version of the original's two WebGL shader passes:

1. Splat: each particle adds ``INFLUENCE * (1 - d/R)^3`` to the water or wax
   density at every point within ``R`` of it, and the same weight times its
   temperature to a temperature sum.
2. Colour: where the wax makes up enough of the local density, the point is
   coloured from the wax gradient, indexed by the average temperature;
   elsewhere it is the water colour.

The original switches between water and wax at a wax fraction of 0.2. On a
low-resolution display that gives jagged blob edges, so here the switch is
blended over ``wax_fraction +/- wax_soft``.

The original averages the temperature over all nearby particles, so water
(whose temperature never changes) dilutes it, and indexes the gradient with
twice that average. With ``wax_only_temp=True`` the average is over the wax
particles only, and ``temp_scale`` sets which temperature reaches the end of
the gradient (``1 / temp_scale``), so the colours can span the range the wax
actually covers.

The original also fades to the water colour where the total density is low
(the empty top of the lamp), but it fades to the same water colour, so that
step makes no difference and is left out.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

SPLAT_RADIUS = 34.0  # CONFIG.PARTICLE_RADIUS [px]
INFLUENCE = 0.5  # CONFIG.INFLUENCE_WEIGHT
WATER_RGB = np.array([0x19, 0xA5, 0xCC]) / 255
# wax gradient: (temperature, (R, G, B)) stops, the original's default
WAX_STOPS = [(0.0, (180, 125, 40)), (0.8, (180, 75, 40)), (1.0, (180, 75, 40))]


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class PointRenderer:
    """Colours fixed points from the particles around them.

    The splat is linear in the particles, so it is done in two stages:
    each particle is spread bilinearly over the 4 nearest nodes of a grid
    with spacing ``cell``, then a fixed sparse matrix, precomputed from the
    point positions, applies the splat kernel from the grid nodes to the
    points. Per frame this costs one ``bincount`` and one sparse product.
    The grid moves each particle's weight by less than ``cell``, which is
    small next to the kernel radius. ``cell=None`` uses the exact (slower)
    particle-to-point distances instead, and ``cell="numba"`` the exact
    distances in a compiled loop (:func:`lava_lamp.fast.splat`, needs numba),
    which is the fastest.

    Parameters
    ----------
    points : (M, 2) point positions in simulation coordinates (px)
    width, height : extent of the simulation domain (px), for the grid
    radius : splat radius (px)
    cell : grid spacing (px), None for the exact calculation, or "numba"
    wax_fraction, wax_soft : wax/water switch and its blending half-width
    water_rgb : (3,) water colour, 0-1
    wax_stops : wax gradient stops, ``[(t, (R, G, B)), ...]`` with 0-255 colours
    wax_only_temp : average the temperature over wax particles only (default
                    False: over all particles, as in the original)
    temp_scale : gradient position = ``temp_scale`` x average temperature
                 (default 2, as in the original)
    """

    def __init__(
        self,
        points,
        width,
        height,
        radius=SPLAT_RADIUS,
        cell=4.0,
        wax_fraction=0.2,
        wax_soft=0.1,
        water_rgb=WATER_RGB,
        wax_stops=WAX_STOPS,
        wax_only_temp=False,
        temp_scale=2.0,
    ):
        self.points = np.asarray(points, dtype=float)
        self.radius, self.cell = radius, cell
        self.wax_fraction, self.wax_soft = wax_fraction, wax_soft
        self.water_rgb = np.asarray(water_rgb, dtype=float)
        self.wax_only_temp, self.temp_scale = wax_only_temp, temp_scale
        self._stop_t = np.array([s[0] for s in wax_stops])
        self._stop_rgb = np.array([s[1] for s in wax_stops], dtype=float) / 255
        self._tree = cKDTree(self.points)
        if cell == "numba":
            from . import fast

            self._fast = fast
            self._x0, self._y0 = -radius, -radius  # margin around the domain
            self._nx = int(np.ceil((width + 2 * radius) / radius))
            self._ny = int(np.ceil((height + 2 * radius) / radius))
        elif cell is not None:
            # grid nodes cover the domain plus one cell on each side, so
            # particles slightly outside still have 4 nodes
            self._nx = int(np.ceil(width / cell)) + 3
            self._ny = int(np.ceil(height / cell)) + 3
            gx, gy = np.meshgrid(
                (np.arange(self._nx) - 1) * cell,
                (np.arange(self._ny) - 1) * cell,
            )
            nodes = np.c_[gx.ravel(), gy.ravel()]  # node k = iy * nx + ix
            self._kernel = self._splat_matrix(cKDTree(nodes))

    def _splat_matrix(self, other_tree):
        """Sparse (M, len(other)) matrix of kernel weights."""
        W = self._tree.sparse_distance_matrix(
            other_tree, self.radius, output_type="coo_matrix"
        )
        W.data = INFLUENCE * (1 - W.data / self.radius) ** 3
        return W.tocsr()

    def _splat(self, positions, values):
        """Splat sums at the points: (M, k) for (N, k) particle values."""
        if self.cell is None:
            return self._splat_matrix(cKDTree(positions)) @ values
        if self.cell == "numba":
            pos = np.ascontiguousarray(positions)
            start, order = self._fast.build_cells(
                pos[:, 0],
                pos[:, 1],
                self._x0,
                self._y0,
                self.radius,
                self._nx,
                self._ny,
            )
            return self._fast.splat(
                self.points,
                pos,
                values,
                self.radius,
                start,
                order,
                self._x0,
                self._y0,
                self._nx,
                self._ny,
            )
        u = positions / self.cell + 1.0  # grid coordinates
        u[:, 0] = np.clip(u[:, 0], 0, self._nx - 1.001)
        u[:, 1] = np.clip(u[:, 1], 0, self._ny - 1.001)
        i0 = np.floor(u).astype(int)
        f = u - i0
        n_nodes = self._nx * self._ny
        grid = np.zeros((n_nodes, values.shape[1]))
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            w = (f[:, 0] if dx else 1 - f[:, 0]) * (
                f[:, 1] if dy else 1 - f[:, 1]
            )
            k = (i0[:, 1] + dy) * self._nx + i0[:, 0] + dx
            for c in range(values.shape[1]):
                grid[:, c] += np.bincount(k, w * values[:, c], n_nodes)
        return self._kernel @ grid

    def wax_colour(self, t):
        t = np.clip(t, 0, 1)
        return np.stack(
            [
                np.interp(t, self._stop_t, self._stop_rgb[:, c])
                for c in range(3)
            ],
            axis=-1,
        )

    def rgb(self, positions, is_wax, temperature):
        """(M, 3) colours in 0-1 for the given particles."""
        if self.wax_only_temp:
            temperature = temperature * is_wax
        values = np.c_[np.ones(len(is_wax)), is_wax, temperature]
        total, d_wax, t_sum = self._splat(
            np.asarray(positions, float), values
        ).T
        t_weight = d_wax if self.wax_only_temp else total
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(total > 0, d_wax / total, 0.0)
            avg_t = np.where(t_weight > 0, t_sum / t_weight, 0.0)
        a = smoothstep(
            self.wax_fraction - self.wax_soft,
            self.wax_fraction + self.wax_soft,
            frac,
        )[:, None]
        return (1 - a) * self.water_rgb + a * self.wax_colour(
            self.temp_scale * avg_t
        )


def to_led_values(rgb, brightness=1.0, gamma=2.2):
    """Scale 0-1 colours to 0-255 LED values with gamma correction.

    The LEDs' light output is non-linear in the programmed value, so the
    colours are treated as intended (perceptually linear) brightness and
    raised to ``gamma`` (as in display1593's fluidsim). ``brightness`` then
    scales the output linearly, like ``show_image``'s ``1 / dimness``.
    """
    v = np.clip(rgb, 0.0, 1.0) ** gamma * np.clip(brightness, 0.0, 1.0)
    return np.round(v * 255).astype(np.uint8)
