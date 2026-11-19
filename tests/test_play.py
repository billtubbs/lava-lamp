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


class _FakeLamp:
    steps_per_frame = 1

    class params:
        dt = 0.5

    def advance(self):
        pass

    def led_values(self):
        return np.zeros((1593, 3), dtype=np.uint8)


def _fake_display_class(fail_after, stop_after, fail_connect=()):
    """A Display1593 stand-in. Connection k (from 0) fails in show() after
    fail_after[k] frames, or raises KeyboardInterrupt after stop_after
    frames in total; connecting fails on the attempts in fail_connect."""
    events = []

    class FakeDisplay:
        n_connect = 0
        total = 0

        def __init__(self, pipelined=True):
            self.show_slack = []
            self.k = FakeDisplay.n_connect
            self.shown = 0
            self._lock = self

        def connect(self):
            FakeDisplay.n_connect += 1
            events.append(("connect", self.k))
            if self.k in fail_connect:
                raise Exception("No microcontroller found")

        def release(self):
            events.append(("release", self.k))

        def disconnect(self):
            events.append(("disconnect", self.k))

        def clear_all(self):
            pass

        def set_all_leds(self, values):
            pass

        def show(self, t):
            if self.shown >= fail_after.get(self.k, 10**9):
                raise RuntimeError("TEENSY1: timed out waiting for response")
            if FakeDisplay.total >= stop_after:
                raise KeyboardInterrupt
            self.shown += 1
            FakeDisplay.total += 1

    return FakeDisplay, events


def _patch_display(monkeypatch, display_class):
    import display1593
    import display1593.logging_utils

    monkeypatch.setattr(display1593, "Display1593", display_class, raising=False)
    monkeypatch.setattr(
        display1593.logging_utils, "configure_root_logging", lambda path: None
    )


def test_reconnects_after_display_failure(monkeypatch, capsys):
    from play_lava_lamp import run_display

    display_class, events = _fake_display_class({0: 5, 1: 3}, stop_after=20)
    _patch_display(monkeypatch, display_class)
    run_display(_FakeLamp(), fps=1000, reconnect_wait=0)

    connects = [k for e, k in events if e == "connect"]
    assert connects == [0, 1, 2]
    # every connection was closed and its lock released
    assert [k for e, k in events if e == "disconnect"] == [0, 1, 2]
    assert [k for e, k in events if e == "release"] == [0, 1, 2]
    out = capsys.readouterr().out
    assert out.count("Display failed: TEENSY1: timed out") == 2
    assert "20 frames in" in out and "2 reconnection(s)" in out


def test_failure_to_reconnect_stops_the_script(monkeypatch):
    from play_lava_lamp import run_display

    display_class, events = _fake_display_class(
        {0: 5}, stop_after=20, fail_connect={1}
    )
    _patch_display(monkeypatch, display_class)
    with pytest.raises(Exception, match="No microcontroller found"):
        run_display(_FakeLamp(), fps=1000, reconnect_wait=0)
    assert [e for e, _ in events].count("connect") == 2
