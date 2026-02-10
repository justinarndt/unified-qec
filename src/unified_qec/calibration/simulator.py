"""
HoloG Differentiable Calibration — Simulator

Core density-matrix simulator for the Gross Code degree-6 stabilizer
measurement circuit. Models coherent control errors, ZZ crosstalk,
and T1 damping, returning the syndrome error probability as the loss
function for calibration.

Requires: pip install unified-qec[jax]

Migrated from: HoloG/holog_diffg/simulator.py
"""

import numpy as np
from unified_qec.calibration.config import JAX_AVAILABLE, I, P0, P1

if JAX_AVAILABLE:
    import jax
    import jax.numpy as jnp
else:
    jnp = np

from unified_qec.calibration.circuit import kron_full, embed_gate
from unified_qec.calibration.physics import rx, damping_channel, apply_crosstalk


def simulate_plaquette(
    theta_params: np.ndarray,
    t1_us: float = 30.0,
    zz_strength: float = 0.02,
    n_data: int = 6,
) -> float:
    """
    Simulate a plaquette stabilizer measurement with coherent errors.

    Models the degree-6 stabilizer from IBM's Gross Code.

    Parameters
    ----------
    theta_params : ndarray, shape (n_data,)
        Control pulse angles for each data qubit's Rx gate.
        Ideal value is π for each.
    t1_us : float
        T1 relaxation time in microseconds.
    zz_strength : float
        ZZ crosstalk strength (radians).
    n_data : int
        Number of data qubits in the plaquette.

    Returns
    -------
    float
        Syndrome error probability (loss for calibration).
    """
    n_total = n_data + 1  # data + ancilla
    dim = 2 ** n_total
    ancilla = n_data  # ancilla is the last qubit

    # Initial state: all |0⟩ = |00...0⟩ ⟨00...0|
    psi0 = jnp.zeros(dim, dtype=jnp.complex128)
    psi0 = psi0.at[0].set(1.0) if JAX_AVAILABLE else np.zeros(dim, dtype=np.complex128)
    if not JAX_AVAILABLE:
        psi0[0] = 1.0
    rho = jnp.outer(psi0, psi0.conj())

    # Step 1: Hadamard on ancilla
    from unified_qec.calibration.config import H_gate
    H_full = embed_gate(H_gate, ancilla, n_total)
    rho = H_full @ rho @ H_full.conj().T

    # Step 2: CNOT-like interactions (ancilla controls X on data)
    for i in range(n_data):
        # Apply Rx with calibration angle
        Rx_gate = rx(theta_params[i])
        Rx_full = embed_gate(Rx_gate, i, n_total)
        rho = Rx_full @ rho @ Rx_full.conj().T

        # Apply T1 damping
        rho = damping_channel(rho, i, n_total, t1_us)

        # Apply ZZ crosstalk with ancilla
        rho = apply_crosstalk(rho, i, ancilla, n_total, zz_strength)

    # Step 3: Hadamard on ancilla
    rho = H_full @ rho @ H_full.conj().T

    # Step 4: Measure ancilla — probability of |1⟩
    P1_full = embed_gate(P1, ancilla, n_total)
    p_error = jnp.trace(P1_full @ rho).real

    return float(p_error)
