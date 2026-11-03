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


def test_settle_stops_motion_and_restores_heat():
    n = int(round(sph.LAMP_DENSITY * SIZE * SIZE))
    sim = sph.LavaLampSim(n_particles=n, width=SIZE, height=SIZE,
                          polygon=sph.square_polygon(SIZE), seed=0,
                          wax_at="bottom")
    m0 = sim.state[4].copy()
    sim.settle(100)
    assert sim.params.temp_speed == 1.0
    assert sim.frame == 0
    assert np.array_equal(sim.state[4], m0)        # no heating while settling
    # at rest: particles barely move in a step (the stored velocities aren't
    # a good measure, as they include the walls' push for the next step)
    xy = sim.state[:2].copy()
    sim.step(1)
    assert np.abs(sim.state[:2] - xy).max() < 0.2


def test_numba_matches_numpy_with_smaller_dt():
    pytest.importorskip("numba")
    P = sph.LavaParams(dt=0.5)
    a = square_sim(300)
    a.params = P
    b = sph.LavaLampSim(state=a.state, params=P, width=SIZE, height=SIZE,
                        polygon=sph.square_polygon(SIZE), backend="numba")
    a.step(4)
    b.step(4)
    assert np.abs(a.state - b.state).max() < 1e-9


def test_dt_one_is_the_default():
    # dt = 1 must reproduce the original (checked against the JS translation
    # in test_reference.py); smaller dt moves particles about half as far
    a, b = square_sim(300), square_sim(300)
    b.params = sph.LavaParams(dt=0.5)
    x0 = a.state[:2].copy()
    a.step(1)
    b.step(1)
    move_a = np.hypot(*(a.state[:2] - x0)).mean()
    move_b = np.hypot(*(b.state[:2] - x0)).mean()
    assert 0.3 < move_b / move_a < 0.7
