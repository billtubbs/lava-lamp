# Lava lamp SPH (Python port)

NumPy/SciPy port of the physics in the lava lamp simulator at
<http://lava-lamp-simulation.web.app/> (from r/Simulated, "[OC] Lava lamp simulation
using SPH fluid dynamics"). The original is a single unminified HTML file
(~2,000 lines of plain JavaScript and WebGL). It's not in this repo.

**Credit:** the original is by the author of the Reddit post
<https://www.reddit.com/r/Simulated/comments/1t11plc/oc_lava_lamp_simulation_using_sph_fluid_dynamics/>
(username unknown). Keep this credit in the module docstrings and README.

**Licence:** the original has no licence or author notice, so it isn't open
source. Ask the author before publishing or redistributing this port.

## Files

- `src/lava_lamp/sph.py`: the module (package `lava_lamp`, which re-exports
  its public names). `LavaParams`, `initial_state`, `lamp_polygon`,
  the step functions (`heat_and_gravity`, `interact`, `apply_boundary`, `step`)
  and the `LavaLampSim` driver.
- `tests/test_reference.py`: checks the vectorised step against a line-by-line Python
  translation of the JS `updateParticles`/`applyBorderForce` (agree to ~1e-13).
  Run with `pytest`, or directly to print the per-step errors.
- `src/lava_lamp/render.py`: `PointRenderer`, the original's shader splat
  sampled at fixed points (the LED centres), and `to_led_values` (gamma, then
  brightness).
- `play_lava_lamp.py`: the display app (see below).
- `tests/test_render.py`: renderer and square-domain tests.
- `demo.py`: runs the simulation in the lamp shape and renders frames like the
  original's two WebGL shader passes. `python demo.py [--frames 6000] [--every 20] [--gif out.gif]`

Dependencies are in `pyproject.toml`; `requirements.txt` just installs the
package editable with the `demo` and `test` extras. The venv is `.venv/`
(git-ignored), made with `/opt/local/bin/python3.12` (Apple's `python3` is 3.9,
too old).

## Physics model (from the original JS)

Simplified SPH-like, not textbook SPH:

- 3,000 particles, 20% wax. Smoothing radius H = 17 px.
- Pressure: pairwise soft-sphere repulsion `STIFFNESS*(1-q)^2`, no density
  summation or equation of state.
- Surface tension (constant attraction) and viscosity act only between particles
  of the same material. That's what keeps wax and water separate.
- **Temperature is stored as `mass`** and gravity is `vy += GRAVITY*mass`. Wax near
  the bottom (heater, strongest at the centre line) relaxes to mass -6 and rises;
  elsewhere it relaxes back to +1 and sinks. Keep this trick.
- Symplectic Euler, dt = 1 per frame, velocity clamp MAX_VEL = 4, repulsion from the
  lamp-shaped boundary polygon.

## Design decisions

- Pair list from `cKDTree.query_pairs`: each pair's force is computed once and
  applied to both particles with opposite signs via `np.bincount`.
- Jacobi update: every particle uses the start-of-step state. The JS updates
  particles in place (Gauss-Seidel-like). The motion looks the same.
- Not ported: mouse interaction and the original's dead code (balls,
  source/drain/vortex modes, CPU canvas renderer).

## LED display app (`play_lava_lamp.py`)

This repo is a standalone display app for the 1593-LED display. Bill wants each
display app in its own repo, **not inside display1593** (earlier ones, fluidsim
and fireplace, were; don't follow that). It imports display1593 (sibling repo
`~/display1593`, plus its serial library `~/ser-comm-py`, both installed
editable in `.venv`; neither is on PyPI, so they're not in `pyproject.toml`).

- **Can't test the display from the Mac.** Only `--benchmark` and
  `--preview out.gif` run here; they use display1593 just for the LED centres
  (`display1593.data.ledArray_data_1593`). Bill pushes and clones on the Pi
  (Raspberry Pi Zero 2 W, 4 cores) to test on the hardware.
  On the Pi it's installed into the system Python with `pip install -e .`
  (no venv), alongside display1593.
- Defaults chosen by Bill on the display: `--fps 24 --viscosity 0.5
  --surface-tension 3.0` (0 late frames over 5 min, ~16 ms compute per frame),
  at half the speed he tried, so `--steps-per-frame 0.5` = 12 steps/s.
  Fractional steps per frame are shown by linear interpolation of positions
  and temperature between the last two steps (`LavaLampLeds.shown`), so the
  physics is unchanged (a smaller dt inside the physics would have changed the
  tuned look).
- Saved initial state: Bill asked why not save a settled state instead of
  settling each start. Not done: it depends on size, particle count, wax ratio
  and physics parameters, settling takes ~0.2 s on the Pi, and each start gets
  a random jitter. Offered a per-settings cache if he wants one.
- Earlier speed: 20 fps x 1 step/frame = 20 steps/s (the original runs ~60).
  Bill found 10 fps x 4 steps too fast and jumpy (up to 16 px, ~3 LEDs, per
  frame at the speed cap of 4 px/step) and asked for half speed. Slow the lamp
  by running fewer steps per second, not by changing the physics (dt = 1 is
  built into the model). 20 fps worked on the Pi (1590 frames in 79.5 s; the
  late frames were probably start-up, now avoided by running one frame before
  the display clock starts).
- Start: wax starts as a cold layer on the heater (`wax_at="bottom"`), not at
  the top (the original's "waterfall"; Bill's request), then
  `LavaLampSim.settle()` (300 steps, heater off, velocities x0.8 per step).
  The first plume rises after ~50 s at 20 steps/s (a lamp warming up).
- Why settle: the fluid is compressible (soft repulsion, no density
  constraint) and only wax has weight (water mass 0), so the layer is a weight
  on a spring. Unsettled, it collapsed 46 -> 31 px in 12 steps, then
  "breathed" with a ~80-step (4 s) period, +/-6 px, ~25% of its LED area
  peak-to-peak, barely damped (viscosity only acts on pairs moving apart, and
  not between wax and water). Bill saw it as the wax volume growing and
  shrinking. Settled: breathing 12 -> 1 px peak-to-peak. Doubling stiffness
  didn't help. The layer still swells slowly (~30 -> 40 px) as it heats,
  since lighter wax is squeezed less.
- Blob count: ~3 blobs at a time (2-5). Measured over 30,000 steps (mean blobs
  of >= 15 wax particles): visc_wax 0.3 -> 3.4, 0.2 -> 3.2, 0.15 -> 2.2,
  0.1 -> 1.0 (one merged mass); 0.45-0.6 and st_ratio 3.0-3.5 -> ~3.2-3.5;
  st_ratio 2.0 -> 1.6. Size 250 -> 3.6, 300 (wax 0.15) -> 4.1. So lower
  viscosity makes fewer, bigger blobs, not more. `--viscosity` is an option
  (default 0.3).
- Domain: a square `--size` px across (default 200) mapped to the display's
  2000 x 2000 LED coordinates. y increases downwards in both, so no flips.
  Particle count defaults to the original lamp's density (`LAMP_DENSITY`,
  ~930 for 200 px). Smaller squares make bigger blobs relative to the display,
  which suits its low resolution (~40 x 40 LEDs; fine shapes don't render).
- Colours are options (`--water-colour`, `--wax-colours` cold to hot, hex or
  R,G,B). Background (water) is black (LEDs off) by default at Bill's request;
  the original's is 19a5cc. Wax default red -> orange -> yellow (e63c32 f59632
  ffd728), sampled from a classic purple (9b46a5) lava lamp photo Bill liked.
- Wax colour is by **temperature** (Bill: more physically relevant). A
  colour-by-height option (lamp lit from below, as in real lamps, where wax
  colour doesn't actually vary) was tried and dropped at his request. Hot wax
  rises, so the top blobs look hottest (yellow).
- The original's wax gradient changed only green by 50/255 and was indexed by
  2 x the all-particle average temperature, so on the LEDs no variation showed.
  Now the temperature is averaged over wax particles only, and `--hot-temp`
  (default 0.3) gets the full hot colour; wax at the LEDs spans ~0.01-0.40
  (1st-99th percentile, median 0.12).
- Cooling: `--cooling` (`ambient_rate`) defaults to 0.5, not the original 0.25.
  Water has mass 0 (no gravity); wax stops rising only once ambient cooling
  brings its mass back above 0, which takes ~5,000 steps. The square is ~3x
  shorter than the lamp, so at 0.25 wax waited at the top (43% of wax in the
  top third, Bill noticed on the display). Wax in top / middle / bottom third
  over 27,000 steps: 0.25 -> .43/.10/.47, 0.5 -> .23/.08/.69,
  0.75 -> .18/.06/.76, 1.0 -> .11/.04/.85.
- First Pi run on the display (NumPy backend, --fps 6): ~134 ms compute per
  frame, 5 late frames in the first minute and none after.
- Next idea: a circular domain (wax pools at the bottom and merges at the top),
  probably via a signed distance function instead of the polygon boundary.
- Rendering: the splat is linear, so particles are spread bilinearly onto a
  4 px grid and a precomputed sparse matrix maps grid to LEDs (~1.5 ms vs 12 ms
  exact on the Mac; error < 10/255 before gamma). display1593's
  `convert_image` (mask averaging of a 256 x 256 image) was considered but
  would need a full image drawn first, and the splat kernel (~7 LED spacings)
  already smooths more than averaging over an LED's area.
- Brightness: `to_led_values` applies gamma, then scales the output, default
  0.125, matching `show_image` (`z**2 / (256 * dimness)`, dimness 8).
- Display driving follows display1593's `playback.py`: pipelined mode,
  `show(t)` at a fixed rate, late-frame warning counter, `show_slack` report.
- Speed (933 particles, 4 steps/frame), per frame:
  - NumPy: Mac 9.2 ms; Pi 118.5 ms (93.8 sim + 24.8 render, 2026-11-03),
    13.6x slower than the Mac, too slow for 10 fps.
  - Numba (`src/lava_lamp/fast.py`, default backend when installed; numba is on
    the Pi for display1593): Mac 2.6 ms (4 threads 0.34 ms/step vs 0.81 ms on 1
    thread). Pi: 19.5 ms/frame on 4 threads (9.7 sim + 9.9 render, 4 steps),
    52.5 ms on 1 thread, vs 112.9 ms NumPy. The display's serial I/O runs in
    background threads (Bill thinks it uses 2-3 cores).
  - Numba version: cell list (cells h wide), each particle sums its own
    neighbours (pairs computed twice, no write conflicts), `parallel=True`,
    `nogil=True` (so the driver's serial threads can run), `cache=True`.
    Matches NumPy to ~3e-14. The LED splat is exact (no grid) in Numba.

## History and performance

An earlier version used CasADi to generate per-particle step functions (removed
on 2026-11-03, see git history). It was ~3.4x slower: 18 ms/step for 3,000
particles against ~5 ms for this NumPy version. CasADi's own calculation took
only ~4 ms. Most of the time went on the fixed-size input: a k-nearest-neighbour
query padded to 49 slots (average neighbours ~20) and copying neighbour states
into that array. Consider CasADi again only if derivatives of the step or
generated C code are needed. A Numba cell-grid loop could probably reach ~1 ms/step.
