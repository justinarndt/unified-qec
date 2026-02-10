"""
HoloG Differentiable Calibration — Configuration

Physical constants and standard operators for the JAX-based density-matrix
simulator targeting the IBM Loon processor's plaquette.

Requires: pip install unified-qec[jax]

Migrated from: HoloG/holog_diffg/config.py
"""

try:
    import jax
    import jax.numpy as jnp

    # Enforce double precision for numerical stability
    jax.config.update("jax_enable_x64", True)
    JAX_AVAILABLE = True
except ImportError:
    jnp = None
    JAX_AVAILABLE = False

import numpy as np

# Gate time in microseconds
GATE_TIME_US = 0.06  # 60 ns

# Pauli matrices (numpy for availability without JAX)
I = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
H_gate = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)

# Projectors
P0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
P1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)
