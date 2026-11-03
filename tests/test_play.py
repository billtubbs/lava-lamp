"""Tests for play_lava_lamp.LavaLampLeds (needs display1593 for the LED layout)."""

import numpy as np
import pytest

pytest.importorskip("display1593")
from play_lava_lamp import LavaLampLeds  # noqa: E402


def make(steps_per_frame):
    return LavaLampLeds(steps_per_frame=steps_per_frame, seed=0, backend="numpy")


def test_whole_steps_show_the_current_state():
    lamp = make(1)
    for _ in range(3):
        lamp.advance()
    pos, is_wax, temp = lamp.shown()
    assert lamp.sim.frame == 3
    assert np.array_equal(pos, lamp.sim.positions)
    assert np.allclose(temp, lamp.sim.temperature())


def test_half_steps_interpolate():
    lamp = make(0.5)
    lamp.advance()                       # time 0.5: one step run, half shown
    assert lamp.sim.frame == 1
    prev = lamp._prev.copy()
    pos, _, _ = lamp.shown()
    assert np.allclose(pos, 0.5 * (prev[:2] + lamp.sim.state[:2]).T)
    lamp.advance()                       # time 1.0: same step, fully shown
    assert lamp.sim.frame == 1
    assert np.allclose(lamp.shown()[0], lamp.sim.positions)
    lamp.advance()                       # time 1.5: second step run
    assert lamp.sim.frame == 2


def test_rate_is_steps_per_frame():
    lamp = make(1 / 3)
    for _ in range(30):
        lamp.advance()
    assert lamp.sim.frame == 10
