"""Run the CasADi lava lamp and render frames like the original WebGL shader.

Usage:  python demo.py [--frames 6000] [--every 20] [--jit] [--gif out.gif]
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.path import Path as MplPath
import numpy as np
from scipy.signal import fftconvolve

import lava_lamp_sph as L

# --- renderer: port of the two shader passes --------------------------------
SPLAT_RADIUS = 34.0      # CONFIG.PARTICLE_RADIUS
INFLUENCE = 0.5          # CONFIG.INFLUENCE_WEIGHT
THRESHOLD = 0.3          # CONFIG.DENSITY_THRESHOLD
SOFT = 0.025             # CONFIG.EDGE_SOFTNESS
WATER_RGB = np.array([0x19, 0xa5, 0xcc]) / 255
WAX_STOPS = [(0.0, (180, 125, 40)), (0.8, (180, 75, 40)), (1.0, (180, 75, 40))]


def _kernel(radius):
    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    d = np.hypot(xx, yy) / radius
    return np.where(d <= 1, (1 - d) ** 3, 0.0) * INFLUENCE


def _wax_colour(t):
    t = np.clip(t, 0, 1)
    xs = [s[0] for s in WAX_STOPS]
    return np.stack([np.interp(t, xs, [s[1][c] for s in WAX_STOPS]) for c in range(3)], -1) / 255


class Renderer:
    def __init__(self, sim, scale=1.0):
        self.sim, self.scale = sim, scale
        self.w, self.h = int(sim.width * scale), int(sim.height * scale)
        self.k = _kernel(SPLAT_RADIUS * scale)
        yy, xx = np.mgrid[0:self.h, 0:self.w]
        pts = np.c_[(xx.ravel() + 0.5) / scale, (yy.ravel() + 0.5) / scale]
        self.inside = MplPath(sim.polygon).contains_points(pts).reshape(self.h, self.w)

    def _splat(self, pos, weights):
        img, _, _ = np.histogram2d(pos[:, 1] * self.scale, pos[:, 0] * self.scale,
                                   bins=(self.h, self.w),
                                   range=((0, self.h), (0, self.w)), weights=weights)
        return fftconvolve(img, self.k, mode="same")

    def image(self):
        s = self.sim
        pos, wax, temp = s.positions, s.is_wax, s.temperature()
        d0 = self._splat(pos, (~wax).astype(float))
        d1 = self._splat(pos, wax.astype(float))
        tacc = self._splat(pos, temp)       # every particle carries tempData
        total = d0 + d1
        with np.errstate(invalid="ignore", divide="ignore"):
            avg_t = np.where(total > 0, tacc / total, 0)
            frac = np.where(total > 0, d1 / total, 0)
        inside = np.clip((total - (THRESHOLD - SOFT)) / (2 * SOFT), 0, 1)
        inside = inside * inside * (3 - 2 * inside)            # smoothstep
        vol = np.where((frac > 0.2)[..., None], _wax_colour(avg_t * 2), WATER_RGB)
        rgb = WATER_RGB * (1 - inside[..., None]) + vol * inside[..., None]
        rgb[~self.inside] = 0
        return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=6000)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--jit", action="store_true")
    ap.add_argument("--gif", default="lava_lamp.gif")
    ap.add_argument("--png", default="lava_lamp_frames.png")
    args = ap.parse_args()

    t0 = time.perf_counter()
    sim = L.LavaLampSim(n_particles=3000, jit=args.jit, seed=0)
    print(f"build: {time.perf_counter() - t0:.1f} s")
    ren = Renderer(sim, scale=0.75)

    frames, stats = [], []
    t_sim = 0.0
    for f in range(args.frames):
        t0 = time.perf_counter()
        sim.step()
        t_sim += time.perf_counter() - t0
        if not np.all(np.isfinite(sim.state)):
            raise RuntimeError(f"non-finite state at frame {f}")
        if f % args.every == 0:
            frames.append(ren.image())
            wax = sim.is_wax
            stats.append((f, sim.state[1, wax].mean(), sim.temperature()[wax].mean(),
                          np.abs(sim.state[2:4]).max(), sim.truncated))
        if f % 1000 == 0:
            print(f"frame {f}: wax mean y={stats[-1][1]:.0f}, mean T={stats[-1][2]:.2f}, "
                  f"truncated={sim.truncated}, {1e3 * t_sim / (f + 1):.1f} ms/step")
    print(f"average {1e3 * t_sim / args.frames:.1f} ms/step over {args.frames} steps")

    # contact sheet
    picks = np.linspace(0, len(frames) - 1, 8).astype(int)
    fig, axes = plt.subplots(1, 8, figsize=(16, 3.4))
    for ax, i in zip(axes, picks):
        ax.imshow(frames[i]); ax.set_axis_off()
        ax.set_title(f"frame {stats[i][0]}", fontsize=9)
    fig.tight_layout(); fig.savefig(args.png, dpi=110, facecolor="black")

    # animation
    fig = plt.figure(figsize=(ren.w / 100, ren.h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    im = ax.imshow(frames[0])
    anim = FuncAnimation(fig, lambda i: (im.set_data(frames[i]),), frames=len(frames))
    anim.save(args.gif, writer=PillowWriter(fps=25))
    np.save("stats.npy", np.array(stats))


if __name__ == "__main__":
    main()
