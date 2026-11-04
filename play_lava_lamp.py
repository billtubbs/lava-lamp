"""Run the lava lamp simulation live on the 1593-LED display.

The simulation runs in a square the shape of the display, ``--size`` px
across (simulation units, see ``lava_lamp.sph``), with the particle density
of the original lamp. Each displayed frame advances it ``--steps-per-frame``
steps, then colours every LED from the particles around its centre (see
``lava_lamp.render``). The original runs one step of time 1 per screen
refresh, about 60 time units/s. The lamp's speed here is fps x steps per
frame x dt; the default, 24 fps x 1 step/frame x dt 0.5, is 12 time units/s,
a fifth of the original's. A fast blob moves at most 2 px between frames.
With a fractional number of steps per frame, each frame shows the particles
part-way between two steps (linear interpolation). The wax starts as a cold layer at rest on the heater (see
``LavaLampSim.settle``).

Modes:

    python play_lava_lamp.py                    # run on the display (Pi)
    python play_lava_lamp.py --benchmark        # time it, no display needed
    python play_lava_lamp.py --preview          # render media/preview.gif (needs matplotlib)

Only the default mode needs the hardware. The others use display1593 only for
the LED positions, so they run on a desktop.

Frames are scheduled with ``show(t)`` in the driver's pipelined mode (as in
display1593's ``playback.py``), so the frame rate stays fixed; frames that
aren't ready in time are logged as late (``play_lava_lamp.log``).
"""

import argparse
import logging
import os
import time

import numpy as np
from display1593.data.ledArray_data_1593 import centres_x, centres_y

from lava_lamp import sph
from lava_lamp.render import PointRenderer, to_led_values

try:
    import numba
except ImportError:
    numba = None

# Colours (hex RRGGBB). The original's water is 19a5cc; on the LEDs a black
# background (LEDs off) looks better (Bill's choice). The original's wax
# gradient (b47d28 to b44b28) only changes green by 50/255, which can't be
# seen on the LEDs. The default wax colours, cold to hot and evenly spaced,
# are red, orange and yellow, sampled from a photo of a classic lava lamp
# (whose liquid is purple, about 9b46a5).
WATER_COLOUR = "000000"
WAX_COLOURS = ("e63c32", "f59632", "ffd728")
# Wax temperature (0 = cold, 1 = the heater's maximum) that gets the full hot
# colour. Averaged over the wax around each LED, the wax spans about 0.01-0.40
# (1st-99th percentile; median 0.12, 90th percentile 0.27), so 0.3 uses
# nearly the whole gradient.
HOT_TEMP = 0.3
# Ambient cooling of the wax. The original's 0.25 suits its taller lamp; in
# the shorter square, hot wax reaches the top before it has cooled, and too
# much wax collects there. With the other defaults here (viscosity 0.5,
# surface tension 3.0, dt 0.5), the share of the wax in the top / bottom
# third is 0.34 / 0.62 at 0.5, and 0.24 / 0.71 at 0.75 (measured over 27,000
# time units).
COOLING = 0.75

LOG_PATH = "play_lava_lamp.log"
DISPLAY_SIZE = 2000.0  # LED centres are in [0, 2000] x [0, 2000]
START_DELAY = 0.1  # s before the first frame is shown

logger = logging.getLogger("play_lava_lamp")


class LavaLampLeds:
    """The simulation in a square, coloured at the LED centres."""

    def __init__(
        self,
        size=200.0,
        n_particles=None,
        wax_ratio=0.2,
        params=None,
        steps_per_frame=1,
        brightness=0.125,
        gamma=2.2,
        seed=None,
        backend="numba",
        water_colour=WATER_COLOUR,
        wax_colours=WAX_COLOURS,
        hot_temp=HOT_TEMP,
    ):
        self.size = size
        self.backend = backend
        if n_particles is None:
            n_particles = int(round(sph.LAMP_DENSITY * size * size))
        self.n_particles, self.wax_ratio, self.seed = (
            n_particles,
            wax_ratio,
            seed,
        )
        self.params = params or sph.LavaParams()
        if steps_per_frame <= 0:
            raise ValueError("steps_per_frame must be positive")
        self.steps_per_frame = steps_per_frame
        self.brightness, self.gamma = brightness, gamma
        self.leds = np.c_[centres_x, centres_y] * (size / DISPLAY_SIZE)
        colours = [parse_colour(c) for c in wax_colours]
        stops = [
            (k / max(len(colours) - 1, 1), c) for k, c in enumerate(colours)
        ]
        self.renderer = PointRenderer(
            self.leds,
            size,
            size,
            cell="numba" if backend == "numba" else 4.0,
            water_rgb=parse_colour(water_colour) / 255,
            wax_stops=stops,
            wax_only_temp=True,
            temp_scale=1 / hot_temp,
        )
        self.reset()

    def reset(self):
        self.sim = sph.LavaLampSim(
            n_particles=self.n_particles,
            wax_ratio=self.wax_ratio,
            params=self.params,
            width=self.size,
            height=self.size,
            polygon=sph.square_polygon(self.size),
            seed=self.seed,
            backend=self.backend,
            wax_at="bottom",
        )
        # start from rest: otherwise the wax layer collapses under its own
        # weight, then bounces (grows and shrinks) every ~4 s
        self.sim.settle()
        self.time = 0.0  # simulation time shown, in steps
        self._prev = self.sim.state.copy()  # state one step before sim.state

    def advance(self):
        """Advance the shown time by ``steps_per_frame`` steps.

        Runs whole simulation steps until the simulation is at or past the
        shown time. With a fractional ``steps_per_frame`` the frame shows the
        particles part-way between the last two steps (see :meth:`shown`),
        so the motion stays smooth. Resets the simulation if its state is no
        longer finite.
        """
        self.time += self.steps_per_frame
        while self.sim.frame < self.time - 1e-9:
            self._prev = self.sim.state.copy()
            self.sim.step(1)
        if not np.isfinite(self.sim.state).all():
            logger.warning(
                "Non-finite state at frame %d, resetting", self.sim.frame
            )
            print("WARNING: simulation diverged, resetting")
            self.reset()

    def shown(self):
        """(positions, is_wax, temperature) at the shown time.

        Linear interpolation between the last two steps, by the fraction of
        the step reached (1 with a whole number of steps per frame).
        """
        s = self.sim
        a = 1.0 - (s.frame - self.time)
        cur = s.state
        if a < 1.0 - 1e-9:
            st = self._prev + a * (cur - self._prev)
        else:
            st = cur
        P = self.params
        temperature = 1.0 - (st[4] - P.goal_hot) / (P.goal_cold - P.goal_hot)
        return st[:2].T, cur[5] > 0.5, temperature

    def rgb(self):
        """(1593, 3) colours in 0-1."""
        return self.renderer.rgb(*self.shown())

    def led_values(self):
        """(1593, 3) uint8 values for ``set_all_leds``."""
        return to_led_values(self.rgb(), self.brightness, self.gamma)


def run_display(lamp, fps, sync=False, report_interval=60.0):
    from display1593 import Display1593
    from display1593.logging_utils import configure_root_logging

    configure_root_logging(LOG_PATH)
    period = 1 / fps
    late = LateFrameCounter()
    logging.getLogger("display1593").addHandler(late)
    # run one frame before the clock starts: with numba, the first call loads
    # (or compiles) the compiled code, which would make the first frames late
    lamp.advance()
    lamp.led_values()
    with Display1593(pipelined=not sync) as dis:
        print(
            f"Running at {fps:g} fps, {lamp.steps_per_frame:g} steps/frame "
            f"of dt {lamp.params.dt:g} "
            f"({'synchronous' if sync else 'pipelined'} mode). "
            "Ctrl+C to stop."
        )
        dis.clear_all()
        dis.show_slack.clear()
        compute = []
        n_shown = 0
        start = report_start = time.monotonic()
        t_show = start + START_DELAY
        try:
            while True:
                t0 = time.perf_counter()
                lamp.advance()
                values = lamp.led_values()
                compute.append(time.perf_counter() - t0)
                dis.set_all_leds(values)
                # pipelined: shows the previous frame at t_show and starts
                # sending this one; synchronous: waits until t_show, then
                # shows this one
                dis.show(t_show)
                n_shown += 1
                t_show += period
                now = time.monotonic()
                if now - report_start >= report_interval:
                    report(compute, dis.show_slack, late.count, period)
                    compute.clear()
                    dis.show_slack.clear()
                    report_start = now
        except KeyboardInterrupt:
            print("Stopped.")
        finally:
            logging.getLogger("display1593").removeHandler(late)
        elapsed = time.monotonic() - start
        summary = (
            f"{n_shown} frames in {elapsed:.1f} s "
            f"({n_shown / elapsed:.1f} fps, target {fps:g}); "
            f"{late.count} late-frame warnings"
        )
        print(summary)
        logger.info(summary)


def parse_colour(text):
    """'RRGGBB', '#RRGGBB' or 'R,G,B' (0-255) to an array of 3 floats."""
    text = str(text).strip()
    if "," in text:
        rgb = [float(v) for v in text.split(",")]
    else:
        h = text.lstrip("#")
        if len(h) != 6:
            raise ValueError(f"expected RRGGBB or R,G,B, got {text!r}")
        rgb = [int(h[i : i + 2], 16) for i in (0, 2, 4)]
    if len(rgb) != 3 or not all(0 <= v <= 255 for v in rgb):
        raise ValueError(f"expected 3 values from 0 to 255, got {text!r}")
    return np.array(rgb, dtype=float)


class LateFrameCounter(logging.Handler):
    """Counts the display driver's warnings (e.g. late frames)."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record):
        self.count += 1


def report(compute, show_slack, n_late, period):
    c = 1000 * np.array(compute)
    msg = (
        f"compute per frame: mean {c.mean():.1f} ms, max {c.max():.1f} ms "
        f"(of {1000 * period:.0f} ms); {n_late} late frames so far"
    )
    if len(show_slack):
        s = 1000 * np.array(show_slack)
        msg += f"; spare time mean {s.mean():.1f} ms, min {s.min():.1f} ms"
    print(msg)
    logger.info(msg)


def benchmark(lamp, n_frames=100, fps=24.0):
    """Time the simulation and rendering per frame (no display needed)."""
    t0 = time.perf_counter()
    lamp.advance()  # warm up (and compile, with numba)
    lamp.led_values()
    print(f"first frame: {time.perf_counter() - t0:.1f} s")
    t_step, t_render = [], []
    for _ in range(n_frames):
        t0 = time.perf_counter()
        lamp.advance()
        t1 = time.perf_counter()
        lamp.led_values()
        t2 = time.perf_counter()
        t_step.append(t1 - t0)
        t_render.append(t2 - t1)
    step_ms = 1000 * np.mean(t_step)
    render_ms = 1000 * np.mean(t_render)
    total = step_ms + render_ms
    threads = (
        f", {numba.get_num_threads()} threads"
        if lamp.backend == "numba"
        else ""
    )
    print(
        f"{lamp.backend}{threads}: "
        f"{lamp.n_particles} particles, {lamp.size:g} px square, "
        f"{lamp.steps_per_frame:g} steps/frame of dt {lamp.params.dt:g}, "
        f"{n_frames} frames"
    )
    print(
        f"  simulation: {step_ms:6.1f} ms/frame "
        f"({step_ms / lamp.steps_per_frame:.1f} ms/step)"
    )
    print(f"  rendering:  {render_ms:6.1f} ms/frame")
    print(
        f"  total:      {total:6.1f} ms/frame -> at most {1000 / total:.1f} fps "
        f"(target {fps:g} fps = {1000 / fps:.0f} ms/frame, before the "
        "serial I/O to the display)"
    )


def preview(lamp, path, n_frames=300, fps=24.0):
    """Save a GIF of the LED colours, drawn at the LED positions."""
    if not os.path.splitext(path)[1]:
        path += ".gif"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    frames = []
    for k in range(n_frames):
        lamp.advance()
        # show what the LEDs will show: gamma-corrected values, but at full
        # brightness so the preview isn't dark
        frames.append(to_led_values(lamp.rgb(), 1.0, 1.0))
        if (k + 1) % 100 == 0:
            print(f"frame {k + 1}/{n_frames}")
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_facecolor("black")
    ax.set_xlim(0, lamp.size)
    ax.set_ylim(lamp.size, 0)  # LED y increases downwards
    ax.set_axis_off()
    sc = ax.scatter(lamp.leds[:, 0], lamp.leds[:, 1], c=frames[0] / 255, s=12)

    def update(i):
        sc.set_facecolor(frames[i] / 255)
        return (sc,)

    anim = FuncAnimation(fig, update, frames=len(frames))
    anim.save(
        path,
        writer=PillowWriter(fps=fps),
        savefig_kwargs={"facecolor": "black"},
    )
    print(f"Saved {path} ({len(frames)} frames at {fps:g} fps)")


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="display frame rate (default 24)",
    )
    ap.add_argument(
        "--steps-per-frame",
        type=float,
        default=1,
        help="simulation steps per displayed frame (default 1). "
        "The lamp's speed is fps x steps per frame x dt time "
        "units/s. Fractions are shown smoothly, by interpolating "
        "between steps",
    )
    ap.add_argument(
        "--dt",
        type=float,
        default=0.5,
        help="simulation time step (default 0.5; the original's "
        "is 1). Smaller is more accurate but needs more steps for "
        "the same speed",
    )
    ap.add_argument(
        "--size",
        type=float,
        default=200.0,
        help="side of the square in simulation px; the particle "
        "count scales with its area, and the blobs look bigger on "
        "the display when it's smaller (default 200)",
    )
    ap.add_argument(
        "--n-particles",
        type=int,
        default=None,
        help="default: the original lamp's particle density",
    )
    ap.add_argument(
        "--wax-ratio",
        type=float,
        default=0.2,
        help="fraction of the particles that are wax (default 0.2)",
    )
    ap.add_argument(
        "--heat",
        type=float,
        default=1.0,
        help="heat transfer rate (the original's slider, default 1)",
    )
    ap.add_argument(
        "--surface-tension",
        type=float,
        default=3.0,
        help="the original's slider (default 3.0; the original's is 2.5)",
    )
    ap.add_argument(
        "--viscosity",
        type=float,
        default=0.5,
        help="wax viscosity (default 0.5; the original's is 0.3). "
        "Lower makes fewer, bigger blobs: on average 3.4 blobs "
        "at 0.3, 2.2 at 0.15, 1 at 0.1",
    )
    ap.add_argument(
        "--cooling",
        type=float,
        default=COOLING,
        help="ambient cooling rate of the wax (default "
        f"{COOLING:g}; the original's is 0.25). Higher: more wax "
        "pools at the bottom, less collects at the top",
    )
    ap.add_argument(
        "--water-colour",
        default=WATER_COLOUR,
        help=f"background colour, RRGGBB or R,G,B (default "
        f"{WATER_COLOUR}, i.e. off; the original's is 19a5cc, a "
        "classic purple lava lamp's about 9b46a5)",
    )
    ap.add_argument(
        "--wax-colours",
        nargs="+",
        default=list(WAX_COLOURS),
        metavar="COLOUR",
        help="wax colours from cold to hot, evenly spaced over "
        "the temperature range (default: "
        f"{' '.join(WAX_COLOURS)}, red, orange, yellow). One "
        "colour gives wax of a single colour",
    )
    ap.add_argument(
        "--hot-temp",
        type=float,
        default=HOT_TEMP,
        help="wax temperature (0-1) that gets the full hot colour "
        f"(default {HOT_TEMP:g}; most wax is between 0 and 0.4). "
        "Lower: more of the wax looks hot",
    )
    ap.add_argument(
        "--brightness",
        type=float,
        default=0.125,
        help="output scale after gamma correction, 0-1 "
        "(default 0.125, the same as display1593's show_image; tune by eye)",
    )
    ap.add_argument(
        "--gamma",
        type=float,
        default=2.2,
        help="gamma correction for the LEDs' response (default 2.2)",
    )
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--backend",
        choices=("numba", "numpy"),
        default="numba" if numba is not None else "numpy",
        help="numba (default if installed; compiled, parallel) or "
        "numpy. Numba compiles on the first run (can take a "
        "minute on the Pi) and caches the result",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=None,
        help="numba threads (default: all cores); try one less "
        "than the cores to leave one for the display's serial I/O",
    )
    ap.add_argument(
        "--sync",
        action="store_true",
        help="use the driver's synchronous mode, not pipelined",
    )
    ap.add_argument(
        "--report-interval",
        type=float,
        default=60.0,
        help="seconds between timing reports (default 60)",
    )
    ap.add_argument(
        "--benchmark",
        action="store_true",
        help="time the simulation and rendering, no display",
    )
    ap.add_argument(
        "--preview",
        nargs="?",
        const="media/preview.gif",
        metavar="FILE",
        help="save a GIF of the LED colours, no display (default "
        "file media/preview.gif)",
    )
    ap.add_argument(
        "--frames",
        type=int,
        default=None,
        help="frames for --benchmark (100) or --preview (300)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    if args.threads is not None and args.backend == "numba":
        numba.set_num_threads(args.threads)
    params = sph.LavaParams(
        temp_speed=args.heat,
        st_ratio=args.surface_tension,
        ambient_rate=args.cooling,
        visc_wax=args.viscosity,
        dt=args.dt,
    )
    lamp = LavaLampLeds(
        size=args.size,
        n_particles=args.n_particles,
        wax_ratio=args.wax_ratio,
        params=params,
        steps_per_frame=args.steps_per_frame,
        brightness=args.brightness,
        gamma=args.gamma,
        seed=args.seed,
        backend=args.backend,
        water_colour=args.water_colour,
        wax_colours=args.wax_colours,
        hot_temp=args.hot_temp,
    )
    if args.benchmark:
        benchmark(lamp, args.frames or 100, args.fps)
    elif args.preview:
        preview(lamp, args.preview, args.frames or 300, args.fps)
    else:
        run_display(lamp, args.fps, args.sync, args.report_interval)


if __name__ == "__main__":
    main()
