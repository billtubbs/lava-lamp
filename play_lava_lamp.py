"""Run the lava lamp simulation live on the 1593-LED display.

The simulation runs in a square the shape of the display, ``--size`` px
across (simulation units, see ``lava_lamp.sph``), with the particle density
of the original lamp. Each displayed frame advances it ``--steps-per-frame``
steps, then colours every LED from the particles around its centre (see
``lava_lamp.render``). The original runs one step per screen refresh, about
60 steps/s, so 10 fps x 4 steps/frame runs at 2/3 of its speed.

Modes:

    python play_lava_lamp.py                    # run on the display (Pi)
    python play_lava_lamp.py --benchmark        # time it, no display needed
    python play_lava_lamp.py --preview out.gif  # render a GIF (needs matplotlib)

Only the default mode needs the hardware. The others use display1593 only for
the LED positions, so they run on a desktop.

Frames are scheduled with ``show(t)`` in the driver's pipelined mode (as in
display1593's ``playback.py``), so the frame rate stays fixed; frames that
aren't ready in time are logged as late (``play_lava_lamp.log``).
"""

import argparse
import logging
import time

import numpy as np
from display1593.data.ledArray_data_1593 import centres_x, centres_y

from lava_lamp import sph
from lava_lamp.render import PointRenderer, to_led_values

LOG_PATH = "play_lava_lamp.log"
DISPLAY_SIZE = 2000.0  # LED centres are in [0, 2000] x [0, 2000]
START_DELAY = 0.1  # s before the first frame is shown

logger = logging.getLogger("play_lava_lamp")


class LavaLampLeds:
    """The simulation in a square, coloured at the LED centres."""

    def __init__(self, size=200.0, n_particles=None, wax_ratio=0.2,
                 params=None, steps_per_frame=4, brightness=0.125, gamma=2.2,
                 seed=None):
        self.size = size
        if n_particles is None:
            n_particles = int(round(sph.LAMP_DENSITY * size * size))
        self.n_particles, self.wax_ratio, self.seed = n_particles, wax_ratio, seed
        self.params = params or sph.LavaParams()
        self.steps_per_frame = steps_per_frame
        self.brightness, self.gamma = brightness, gamma
        self.leds = np.c_[centres_x, centres_y] * (size / DISPLAY_SIZE)
        self.renderer = PointRenderer(self.leds, size, size)
        self.reset()

    def reset(self):
        self.sim = sph.LavaLampSim(
            n_particles=self.n_particles, wax_ratio=self.wax_ratio,
            params=self.params, width=self.size, height=self.size,
            polygon=sph.square_polygon(self.size), seed=self.seed)

    def advance(self):
        """Step the simulation; resets it if the state is no longer finite."""
        self.sim.step(self.steps_per_frame)
        if not np.isfinite(self.sim.state).all():
            logger.warning("Non-finite state at frame %d, resetting",
                           self.sim.frame)
            print("WARNING: simulation diverged, resetting")
            self.reset()

    def rgb(self):
        """(1593, 3) colours in 0-1."""
        s = self.sim
        return self.renderer.rgb(s.positions, s.is_wax, s.temperature())

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
    with Display1593(pipelined=not sync) as dis:
        print(f"Running at {fps:g} fps, {lamp.steps_per_frame} steps/frame "
              f"({'synchronous' if sync else 'pipelined'} mode). "
              "Ctrl+C to stop.")
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
        summary = (f"{n_shown} frames in {elapsed:.1f} s "
                   f"({n_shown / elapsed:.1f} fps, target {fps:g}); "
                   f"{late.count} late-frame warnings")
        print(summary)
        logger.info(summary)


class LateFrameCounter(logging.Handler):
    """Counts the display driver's warnings (e.g. late frames)."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record):
        self.count += 1


def report(compute, show_slack, n_late, period):
    c = 1000 * np.array(compute)
    msg = (f"compute per frame: mean {c.mean():.1f} ms, max {c.max():.1f} ms "
           f"(of {1000 * period:.0f} ms); {n_late} late frames so far")
    if len(show_slack):
        s = 1000 * np.array(show_slack)
        msg += f"; spare time mean {s.mean():.1f} ms, min {s.min():.1f} ms"
    print(msg)
    logger.info(msg)


def benchmark(lamp, n_frames=100, fps=10.0):
    """Time the simulation and rendering per frame (no display needed)."""
    lamp.advance()  # warm up
    lamp.led_values()
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
    print(f"{lamp.n_particles} particles, {lamp.size:g} px square, "
          f"{lamp.steps_per_frame} steps/frame, {n_frames} frames")
    print(f"  simulation: {step_ms:6.1f} ms/frame "
          f"({step_ms / lamp.steps_per_frame:.1f} ms/step)")
    print(f"  rendering:  {render_ms:6.1f} ms/frame")
    print(f"  total:      {total:6.1f} ms/frame -> at most {1000 / total:.1f} fps "
          f"(target {fps:g} fps = {1000 / fps:.0f} ms/frame, before the "
          "serial I/O to the display)")


def preview(lamp, path, n_frames=300, fps=10.0):
    """Save a GIF of the LED colours, drawn at the LED positions."""
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
    anim.save(path, writer=PillowWriter(fps=fps),
              savefig_kwargs={"facecolor": "black"})
    print(f"Saved {path} ({len(frames)} frames at {fps:g} fps)")


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=float, default=10.0,
                    help="display frame rate (default 10)")
    ap.add_argument("--steps-per-frame", type=int, default=4,
                    help="simulation steps per displayed frame: sets how fast "
                    "the lamp moves (default 4)")
    ap.add_argument("--size", type=float, default=200.0,
                    help="side of the square in simulation px; the particle "
                    "count scales with its area, and the blobs look bigger on "
                    "the display when it's smaller (default 200)")
    ap.add_argument("--n-particles", type=int, default=None,
                    help="default: the original lamp's particle density")
    ap.add_argument("--wax-ratio", type=float, default=0.2,
                    help="fraction of the particles that are wax (default 0.2)")
    ap.add_argument("--heat", type=float, default=1.0,
                    help="heat transfer rate (the original's slider, default 1)")
    ap.add_argument("--surface-tension", type=float, default=2.5,
                    help="the original's slider (default 2.5)")
    ap.add_argument("--brightness", type=float, default=0.125,
                    help="output scale after gamma correction, 0-1 "
                    "(default 0.125, the same as display1593's show_image; tune by eye)")
    ap.add_argument("--gamma", type=float, default=2.2,
                    help="gamma correction for the LEDs' response (default 2.2)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sync", action="store_true",
                    help="use the driver's synchronous mode, not pipelined")
    ap.add_argument("--report-interval", type=float, default=60.0,
                    help="seconds between timing reports (default 60)")
    ap.add_argument("--benchmark", action="store_true",
                    help="time the simulation and rendering, no display")
    ap.add_argument("--preview", metavar="GIF",
                    help="save a GIF of the LED colours, no display")
    ap.add_argument("--frames", type=int, default=None,
                    help="frames for --benchmark (100) or --preview (300)")
    return ap.parse_args()


def main():
    args = parse_args()
    params = sph.LavaParams(temp_speed=args.heat, st_ratio=args.surface_tension)
    lamp = LavaLampLeds(size=args.size, n_particles=args.n_particles,
                        wax_ratio=args.wax_ratio, params=params,
                        steps_per_frame=args.steps_per_frame,
                        brightness=args.brightness, gamma=args.gamma,
                        seed=args.seed)
    if args.benchmark:
        benchmark(lamp, args.frames or 100, args.fps)
    elif args.preview:
        preview(lamp, args.preview, args.frames or 300, args.fps)
    else:
        run_display(lamp, args.fps, args.sync, args.report_interval)


if __name__ == "__main__":
    main()
