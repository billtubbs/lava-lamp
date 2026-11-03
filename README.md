# Lava lamp SPH: a Python port

A NumPy/SciPy port of the particle physics in the browser
[Lava Lamp Simulator](https://lava-lamp-simulation.web.app/).

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

Or install it as a package with `pip install -e ".[demo]"`, or with
`pip install -e .` if you only need the simulation (NumPy and SciPy).

## Files

- `lava_lamp_sph.py`: the simulation module (`LavaParams`, `LavaLampSim`, and
  the step functions).
- `test_reference.py`: checks the vectorised step against a line-by-line
  translation of the JavaScript. Run `python test_reference.py`.
- `demo.py`: runs the simulation and renders it like the original's shaders.
  `python demo.py --gif lava_lamp.gif`

## Usage

```python
import lava_lamp_sph as L

sim = L.LavaLampSim(n_particles=3000, seed=0)
sim.params.temp_speed = 1.5      # parameters can change between steps
sim.step(100)
x, y = sim.positions.T           # particle positions (px)
wax = sim.is_wax
```
