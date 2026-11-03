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

- `lava_lamp_sph.py`: the module. `LavaParams`, `initial_state`, `lamp_polygon`,
  the step functions (`heat_and_gravity`, `interact`, `apply_boundary`, `step`)
  and the `LavaLampSim` driver.
- `test_reference.py`: checks the vectorised step against a line-by-line Python
  translation of the JS `updateParticles`/`applyBorderForce` (agree to ~1e-13).
- `demo.py`: runs the simulation and renders frames like the original's two WebGL
  shader passes. `python demo.py [--frames 6000] [--every 20] [--gif out.gif]`

Dependencies are in `pyproject.toml` and `requirements.txt`. The venv goes in
`.venv/` (git-ignored).

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

## History and performance

An earlier version used CasADi to generate per-particle step functions (removed
on 2026-11-03, see git history). It was ~3.4x slower: 18 ms/step for 3,000
particles against ~5 ms for this NumPy version. CasADi's own calculation took
only ~4 ms. Most of the time went on the fixed-size input: a k-nearest-neighbour
query padded to 49 slots (average neighbours ~20) and copying neighbour states
into that array. Consider CasADi again only if derivatives of the step or
generated C code are needed. A Numba cell-grid loop could probably reach ~1 ms/step.
