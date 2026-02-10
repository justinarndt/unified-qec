"""
Simulation Layer

Circuit generation, sampling (Stim, Stim+Cirq bridge), and noise models.
"""

from unified_qec.simulation.surface_code import SurfaceCodeCircuit, DriftingNoiseModel
from unified_qec.simulation.noise_models import (
    generate_stress_circuit,
    generate_standard_circuit,
    generate_leakage_circuit,
)
