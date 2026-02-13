"""
HoloG Differentiable Calibration — Circuit Utilities

Tensor network operations and gate embedding for the density-matrix
simulator.

Requires: pip install unified-qec[jax]

Migrated from: HoloG/holog_diffg/circuit.py
"""

import numpy as np
from typing import List
from unified_qec.calibration.config import JAX_AVAILABLE

if JAX_AVAILABLE:
    import jax.numpy as jnp
else:
    jnp = np


def kron_full(ops: List[np.ndarray]) -> np.ndarray:
    """
    Compute Kronecker product of a list of operators.

    Parameters
    ----------
    ops : list of ndarray
        Operators to combine via tensor product.

    Returns
    -------
    ndarray
        Full Hilbert space operator.
    """
    result = ops[0]
    for op in ops[1:]:
        result = jnp.kron(result, op) if JAX_AVAILABLE else np.kron(result, op)
    return result


def embed_gate(gate: np.ndarray, target: int, n_qubits: int) -> np.ndarray:
    """
    Embed a 1-qubit gate into the full Hilbert space.

    Parameters
    ----------
    gate : ndarray, shape (2, 2)
        Single-qubit gate.
    target : int
        Target qubit index.
    n_qubits : int
        Total number of qubits.

    Returns
    -------
    ndarray, shape (2^n, 2^n)
        Full Hilbert space gate.
    """
    from unified_qec.calibration.config import II
    ops = [II] * n_qubits
    ops[target] = gate
    return kron_full(ops)


def embed_two_qubit_gate(
    gate: np.ndarray, q0: int, q1: int, n_qubits: int
) -> np.ndarray:
    """
    Embed a 2-qubit gate into the full Hilbert space.

    Parameters
    ----------
    gate : ndarray, shape (4, 4)
        Two-qubit gate.
    q0, q1 : int
        Qubit indices.
    n_qubits : int
        Total number of qubits.

    Returns
    -------
    ndarray, shape (2^n, 2^n)
        Full Hilbert space gate.
    """
    from unified_qec.calibration.config import II
    if abs(q1 - q0) != 1:
        raise ValueError("Two-qubit gate requires adjacent qubits in this implementation.")

    ops = []
    i = 0
    while i < n_qubits:
        if i == min(q0, q1):
            ops.append(gate)
            i += 2
        else:
            ops.append(II)
            i += 1
    return kron_full(ops)
