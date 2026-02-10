"""
Pulse Synthesis for Fidelity Recovery

Physics-aware optimal control synthesis for recovering gate fidelity
on hardware with diagnosed defects. Uses the richer version from
``realtime-qec`` with regularization and smooth control.

Theory
------
Standard control assumes perfect hardware. When defects are present,
standard pulses fail. This synthesizer uses the digital twin's
knowledge of the defects to compute corrective fields:

    H(t) = H_drift + H_defect + Σ_i u_i(t) Z_i

The control u_i(t) is optimized to maximize:
    F = |⟨ψ_target|U(T)|ψ_init⟩|²

Migrated from: realtime-qec/realtime_qec/remediation/pulse_synthesis.py
"""

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from scipy.optimize import minimize
from typing import Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


class PulseSynthesizer:
    """
    Physics-aware optimal control for fidelity recovery.

    Given a diagnosed Hamiltonian (including defects), synthesizes
    control pulses that recover target gate fidelity by navigating
    around the defects via constructive interference.

    Parameters
    ----------
    system_size : int
        Number of qubits in the chain.
    gate_time : float
        Total gate duration in natural units.
    dt : float
        Time step for pulse discretization.
    """

    def __init__(
        self,
        system_size: int = 6,
        gate_time: float = 8.0,
        dt: float = 0.2
    ):
        self.L = system_size
        self.dim = 2 ** system_size
        self.T_gate = gate_time
        self.dt = dt
        self.num_steps = int(gate_time / dt)

        # Pauli matrices
        self.sx = sparse.csr_matrix(np.array([[0., 1.], [1., 0.]]))
        self.sz = sparse.csr_matrix(np.array([[1., 0.], [0., -1.]]))
        self.id = sparse.eye(2)

        # Cache operators
        self._build_operators()

    def _build_operators(self):
        """Pre-compute many-body operators."""
        self.ops_XX = []
        self.ops_Z = []

        for i in range(self.L - 1):
            term = [self.id] * self.L
            term[i] = self.sx
            term[i + 1] = self.sx
            self.ops_XX.append(self._kron_chain(term))

        for i in range(self.L):
            term = [self.id] * self.L
            term[i] = self.sz
            self.ops_Z.append(self._kron_chain(term))

    def _kron_chain(self, ops: list) -> sparse.csr_matrix:
        """Compute tensor product of a list of operators."""
        result = ops[0]
        for op in ops[1:]:
            result = sparse.kron(result, op, format='csr')
        return result

    def _build_hamiltonian(
        self,
        J_couplings: np.ndarray,
        h_fields: np.ndarray
    ) -> sparse.csr_matrix:
        """Build drift Hamiltonian from parameters."""
        H = sparse.csr_matrix((self.dim, self.dim), dtype=complex)
        for i, J in enumerate(J_couplings):
            H += J * self.ops_XX[i]
        for i, h in enumerate(h_fields):
            H += h * self.ops_Z[i]
        return H

    def evolve_with_control(
        self,
        J_couplings: np.ndarray,
        h_fields: np.ndarray,
        control_pulse: np.ndarray,
        initial_state: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Evolve system under control pulse.

        Parameters
        ----------
        J_couplings : array, shape (L-1,)
            Coupling strengths.
        h_fields : array, shape (L,)
            On-site fields.
        control_pulse : array, shape (num_steps, L)
            Time-dependent control amplitudes.
        initial_state : array, optional
            Initial state. Defaults to Néel state.

        Returns
        -------
        final_state : array
            State after evolution.
        """
        H_drift = self._build_hamiltonian(J_couplings, h_fields)

        if initial_state is None:
            psi = np.zeros(self.dim, dtype=complex)
            neel_idx = int("".join(["01"] * (self.L // 2)), 2)
            psi[neel_idx] = 1.0
        else:
            psi = initial_state.astype(complex)

        for step in range(self.num_steps):
            H_t = H_drift.copy()
            for i, amp in enumerate(control_pulse[step]):
                H_t += amp * self.ops_Z[i]
            psi = splinalg.expm_multiply(-1j * H_t * self.dt, psi)

        return psi

    def synthesize(
        self,
        J_diagnosed: np.ndarray,
        h_fields: np.ndarray,
        target_state_idx: Optional[int] = None,
        max_iterations: int = 1000,
        verbose: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        Synthesize optimal control pulse for fidelity recovery.

        Parameters
        ----------
        J_diagnosed : array
            Diagnosed coupling strengths (including defects).
        h_fields : array
            On-site fields.
        target_state_idx : int, optional
            Index of target state in computational basis.
            Defaults to flipped Néel |101010...⟩.
        max_iterations : int
            Maximum optimizer iterations.
        verbose : bool
            Print progress updates.

        Returns
        -------
        optimal_pulse : array, shape (num_steps, L)
            Optimized control pulse.
        fidelity : float
            Achieved gate fidelity.
        """
        if target_state_idx is None:
            target_state_idx = int("".join(["10"] * (self.L // 2)), 2)

        target_psi = np.zeros(self.dim, dtype=complex)
        target_psi[target_state_idx] = 1.0

        initial_controls = np.random.normal(0, 2.0, size=self.num_steps * self.L)
        iteration_count = [0]

        def callback(xk):
            iteration_count[0] += 1
            if verbose and iteration_count[0] % 100 == 0:
                print(f"  Optimizer step {iteration_count[0]}...")

        def loss(ctrl_flat):
            ctrl = ctrl_flat.reshape((self.num_steps, self.L))
            final_psi = self.evolve_with_control(J_diagnosed, h_fields, ctrl)

            fidelity = np.abs(np.vdot(target_psi, final_psi)) ** 2
            infidelity = 1.0 - fidelity

            # Regularization for smooth pulses
            diffs = np.diff(ctrl, axis=0)
            smoothness = np.sum(diffs ** 2) * 0.0001
            power = np.sum(ctrl ** 2) * 0.00001

            return infidelity * 100 + smoothness + power

        if verbose:
            print(f"Synthesizing control pulse ({max_iterations} max iterations)...")

        result = minimize(
            loss,
            initial_controls,
            method='L-BFGS-B',
            options={'maxiter': max_iterations, 'ftol': 1e-6},
            callback=callback
        )

        optimal_pulse = result.x.reshape((self.num_steps, self.L))
        final_psi = self.evolve_with_control(J_diagnosed, h_fields, optimal_pulse)
        fidelity = np.abs(np.vdot(target_psi, final_psi)) ** 2

        if verbose:
            print(f"  Converged. Final fidelity: {fidelity * 100:.2f}%")

        return optimal_pulse, fidelity
