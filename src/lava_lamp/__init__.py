"""Lava lamp particle simulation (NumPy/SciPy port of a browser SPH simulator).

See :mod:`lava_lamp.sph` for the physics and credit to the original author.
"""

from .sph import (
    LAMP_DENSITY,
    STATE_NAMES,
    LavaLampSim,
    LavaParams,
    apply_boundary,
    heat_and_gravity,
    initial_state,
    interact,
    lamp_polygon,
    square_polygon,
    step,
)

__all__ = [
    "LAMP_DENSITY",
    "STATE_NAMES",
    "LavaLampSim",
    "LavaParams",
    "apply_boundary",
    "heat_and_gravity",
    "initial_state",
    "interact",
    "lamp_polygon",
    "square_polygon",
    "step",
]
