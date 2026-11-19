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


def test_driver_warnings_counted_by_kind(capsys):
    import logging

    from play_lava_lamp import DriverWarningCounter

    counter = DriverWarningCounter()
    log = logging.getLogger("display1593.test_play")
    log.addHandler(counter)
    log.propagate = False
    try:
        log.warning("Frame shown %.1f ms late: %s not ready.", 3.0, "TEENSY1")
        log.warning("Show time missed by %.1f ms.", 1.0)
        for i in range(5):
            log.warning("TEENSY1: reply %s matches no command", i)
        log.warning("TEENSY2 sent a message: %s", "My name is TEENSY2")
    finally:
        log.removeHandler(counter)

    assert (counter.late, counter.other) == (2, 6)
    assert counter.summary() == "2 late frames, 6 other driver warnings"
    printed = capsys.readouterr().out.splitlines()
    assert printed == [
        "Display driver warning: TEENSY1: reply 0 matches no command",
        "Display driver warning: TEENSY2 sent a message: My name is TEENSY2",
    ]
