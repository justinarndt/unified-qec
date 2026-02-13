"""
HoloG Differentiable Calibration — Physics

Differentiable Hamiltonians and noise channels for the JAX-based
density-matrix simulator targeting the IBM Loon processor's plaquette.

Requires: pip install unified-qec[jax]

Migrated from: HoloG/holog_diffg/physics.py
"""

import numpy as np
from unified_qec.calibration.config import JAX_AVAILABLE, GATE_TIME_US, Z, II

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
else:
    jnp = np

from unified_qec.calibration.circuit import kron_full


def rx(theta: float) -> np.ndarray:
    """
    Differentiable Rx gate.

    Parameters
    ----------
    theta : float
        Rotation angle in radians.

    Returns
    -------
    ndarray, shape (2, 2)
        Rotation matrix.
    """
    c = jnp.cos(theta / 2)
    s = jnp.sin(theta / 2)
    return jnp.array([[c, -1j * s], [-1j * s, c]], dtype=jnp.complex128)


def damping_channel(rho: np.ndarray, qubit: int, n_qubits: int, t1_us: float = 30.0) -> np.ndarray:
    """
    Apply T1 amplitude damping channel to a density matrix.

    Parameters
    ----------
    rho : ndarray
        Density matrix.
    qubit : int
        Target qubit index.
    n_qubits : int
        Total number of qubits.
    t1_us : float
        T1 relaxation time in microseconds.

    Returns
    -------
    ndarray
        Density matrix after damping.
    """
    gamma = 1.0 - jnp.exp(-GATE_TIME_US / t1_us)

    # Kraus operators
    K0_single = jnp.array([[1, 0], [0, jnp.sqrt(1 - gamma)]], dtype=jnp.complex128)
    K1_single = jnp.array([[0, jnp.sqrt(gamma)], [0, 0]], dtype=jnp.complex128)

    # Embed into full space
    from unified_qec.calibration.circuit import embed_gate
    K0 = embed_gate(K0_single, qubit, n_qubits)
    K1 = embed_gate(K1_single, qubit, n_qubits)

    return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T


def apply_crosstalk(
    rho: np.ndarray,
    q0: int,
    q1: int,
    n_qubits: int,
    zz_strength: float = 0.02
) -> np.ndarray:
    """
    Apply ZZ crosstalk interaction between two qubits.

    Parameters
    ----------
    rho : ndarray
        Density matrix.
    q0, q1 : int
        Qubit indices.
    n_qubits : int
        Total number of qubits.
    zz_strength : float
        ZZ coupling strength (radians per gate time).

    Returns
    -------
    ndarray
        Density matrix after ZZ interaction.
    """
    ops = [II] * n_qubits
    ops[q0] = Z
    ops[q1] = Z
    H_zz = zz_strength * kron_full(ops)

    U = jnp.array(
        np.diag(np.exp(-1j * np.diag(H_zz.real))),
        dtype=jnp.complex128
    ) if not JAX_AVAILABLE else jax.scipy.linalg.expm(-1j * H_zz)

    return U @ rho @ U.conj().T
