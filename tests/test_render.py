"""Tests for the point renderer and the square domain (no display needed)."""

import numpy as np
import pytest

from lava_lamp import sph
from lava_lamp.render import PointRenderer, to_led_values

SIZE = 200.0


def square_sim(n_steps, seed=0):
    n = int(round(sph.LAMP_DENSITY * SIZE * SIZE))
    sim = sph.LavaLampSim(n_particles=n, width=SIZE, height=SIZE,
                          polygon=sph.square_polygon(SIZE), seed=seed)
    sim.step(n_steps)
    return sim


def test_square_stays_inside_and_finite():
    sim = square_sim(500)
    assert np.isfinite(sim.state).all()
    x, y = sim.state[:2]
    assert x.min() >= -1e-9 and x.max() <= SIZE + 1e-9
    assert y.min() >= -1e-9 and y.max() <= SIZE + 1e-9


def test_grid_splat_matches_exact():
    sim = square_sim(1000)
    rng = np.random.default_rng(0)
    points = rng.uniform(0, SIZE, (500, 2))
    args = (sim.positions, sim.is_wax, sim.temperature())
    exact = PointRenderer(points, SIZE, SIZE, cell=None).rgb(*args)
    grid = PointRenderer(points, SIZE, SIZE, cell=4.0).rgb(*args)
    # blob edges are where the colour changes fastest; 4 px cells keep the
    # error within about 10/255 there
    assert np.abs(grid - exact).max() < 12 / 255
    assert np.abs(grid - exact).mean() < 1 / 255


def test_to_led_values():
    rgb = np.array([[0.0, 0.5, 1.0]])
    v = to_led_values(rgb, brightness=1.0, gamma=1.0)
    assert v.dtype == np.uint8
    assert v.tolist() == [[0, 128, 255]]
    assert to_led_values(rgb, brightness=0.5, gamma=2.0).tolist() == [[0, 32, 128]]


def test_numba_matches_numpy():
    pytest.importorskip("numba")
    a = square_sim(300)
    b = sph.LavaLampSim(state=a.state, width=SIZE, height=SIZE,
                        polygon=sph.square_polygon(SIZE), backend="numba")
    a.step(3)
    b.step(3)
    assert np.abs(a.state - b.state).max() < 1e-9
    rng = np.random.default_rng(1)
    points = rng.uniform(0, SIZE, (300, 2))
    args = (a.positions, a.is_wax, a.temperature())
    exact = PointRenderer(points, SIZE, SIZE, cell=None).rgb(*args)
    fast = PointRenderer(points, SIZE, SIZE, cell="numba").rgb(*args)
    assert np.abs(fast - exact).max() < 1e-9
