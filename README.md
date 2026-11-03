# Lava lamp SPH: a Python port

A NumPy/SciPy port of the particle physics in the browser
[Lava Lamp Simulator](https://lava-lamp-simulation.web.app/), and a display
app, `play_lava_lamp.py`, that runs it live on the 1593-LED display
([display1593](https://github.com/billtubbs/display1593)).

## Credit

The original simulator, including its physics model and its WebGL rendering, is
the work of the author of the r/Simulated post
[[OC] Lava lamp simulation using SPH fluid dynamics](https://www.reddit.com/r/Simulated/comments/1t11plc/oc_lava_lamp_simulation_using_sph_fluid_dynamics/).
This repository only translates their JavaScript physics to Python.

The original has no licence notice, so this port is for personal study. Please
ask the original author before reusing or redistributing it.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This installs the `lava_lamp` package in editable mode, with the extras for
the demo (matplotlib, pillow) and the tests (pytest). Use `pip install -e .`
if you only need the simulation (NumPy and SciPy).

### On the Raspberry Pi (display app)

The display app also needs display1593 and its serial library sercommpy,
which are already installed on the Pi for the other display scripts. Install
this package into the same (system) Python:

```sh
git clone https://github.com/billtubbs/lava-lamp.git
cd lava-lamp
pip install -e .
python play_lava_lamp.py --benchmark   # timing only, no display
python play_lava_lamp.py               # run on the display
```

If pip refuses with an "externally-managed-environment" error, add
`--break-system-packages`. The only dependencies are NumPy and SciPy, which
are already installed, so this doesn't replace any system packages.

## Files

- `src/lava_lamp/sph.py`: the simulation (`LavaParams`, `LavaLampSim`, and
  the step functions). The package `lava_lamp` re-exports these names.
- `tests/test_reference.py`: checks the vectorised step against a line-by-line
  translation of the JavaScript. Run `pytest`.
- `src/lava_lamp/fast.py`: Numba versions of the step and the LED splat
  (compiled, parallel, same results to rounding error). Used by
  `play_lava_lamp.py` when Numba is installed (`--backend numba`), and by
  `LavaLampSim(backend="numba")`.
- `src/lava_lamp/render.py`: `PointRenderer` colours a set of points (the LED
  centres) from the particles, like the original's shaders.
- `play_lava_lamp.py`: runs the simulation in a square on the LED display.
  `--benchmark` and `--preview out.gif` work without the display (they only
  need display1593 for the LED positions). See `--help` for the options.
- `tests/test_render.py`: tests for the renderer and the square domain.
- `demo.py`: runs the simulation in the original lamp shape and renders it
  like the original's shaders.
  `python demo.py --gif lava_lamp.gif`

## Usage

```python
import lava_lamp as L

sim = L.LavaLampSim(n_particles=3000, seed=0)
sim.params.temp_speed = 1.5      # parameters can change between steps
sim.step(100)
x, y = sim.positions.T           # particle positions (px)
wax = sim.is_wax
```
