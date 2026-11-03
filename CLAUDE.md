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
- Target: 10 fps. Each frame runs `--steps-per-frame` (default 4) simulation
  steps; the original runs ~60 steps/s.
- Domain: a square `--size` px across (default 200) mapped to the display's
  2000 x 2000 LED coordinates. y increases downwards in both, so no flips.
  Particle count defaults to the original lamp's density (`LAMP_DENSITY`,
  ~930 for 200 px). Smaller squares make bigger blobs relative to the display,
  which suits its low resolution (~40 x 40 LEDs; fine shapes don't render).
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
- Mac timing (933 particles, 4 steps/frame): 7.3 ms simulation + 1.5 ms
  rendering per frame. Pi timing not measured yet (guess 10-20x slower).

## History and performance

An earlier version used CasADi to generate per-particle step functions (removed
on 2026-11-03, see git history). It was ~3.4x slower: 18 ms/step for 3,000
particles against ~5 ms for this NumPy version. CasADi's own calculation took
only ~4 ms. Most of the time went on the fixed-size input: a k-nearest-neighbour
query padded to 49 slots (average neighbours ~20) and copying neighbour states
into that array. Consider CasADi again only if derivatives of the step or
generated C code are needed. A Numba cell-grid loop could probably reach ~1 ms/step.
