"""
Simulation Layer

Circuit generation, sampling (Stim, Stim+Cirq bridge), and noise models.
"""

from unified_qec.simulation.surface_code import (
    SurfaceCodeCircuit as SurfaceCodeCircuit,
    DriftingNoiseModel as DriftingNoiseModel,
)
from unified_qec.simulation.noise_models import (
    generate_stress_circuit as generate_stress_circuit,
    generate_standard_circuit as generate_standard_circuit,
    generate_leakage_circuit as generate_leakage_circuit,
)
