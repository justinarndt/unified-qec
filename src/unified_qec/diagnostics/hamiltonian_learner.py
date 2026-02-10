"""
Hamiltonian Learning and Hardware Diagnostics

Implements inverse problem solving to reconstruct hardware parameters
from experimental observables. Enables detection of calibration drift
and localization of hardware defects.

Uses the richer version from ``realtime-qec`` with bounds, detect_defects,
and comprehensive NumPy documentation. GST comparison method merged from
``stim-cirq-qec``.

Migrated from: realtime-qec/realtime_qec/diagnostics/hamiltonian_learner.py
"""

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
from scipy.optimize import minimize
from typing import Tuple, Optional, Callable
import warnings

warnings.filterwarnings("ignore")


class HamiltonianLearner:
    """
    Digital twin for quantum hardware diagnostics.

    Given experimental time traces (e.g., MBL imbalance dynamics),
    reconstructs the underlying Hamiltonian parameters via gradient-based
    optimization. Enables detection of:
    - Coupling defects (broken/weak links)
    - Crosstalk (enhanced couplings)
    - Frequency drift

    Theory
    ------
    For a spin chain with Hamiltonian:
        H = Σ_i J_i σ^x_i σ^x_{i+1} + Σ_i h_i σ^z_i

    The observable trace I(t) = ⟨ψ(t)|M|ψ(t)⟩ is differentiable w.r.t.
    the parameters {J_i, h_i}. We minimize:
        L(θ) = Σ_t (I_sim(t; θ) - I_exp(t))^2

    to recover θ = {J_i}.

    Parameters
    ----------
    system_size : int
        Number of qubits in the spin chain.
    """

    def __init__(self, system_size: int = 6):
        self.L = system_size
        self.dim = 2 ** system_size

        # Pauli matrices
        self.sx = sparse.csr_matrix(np.array([[0., 1.], [1., 0.]]))
        self.sz = sparse.csr_matrix(np.array([[1., 0.], [0., -1.]]))
        self.id = sparse.eye(2)

        # Cache operators
        self._build_operators()

    def _build_operators(self):
        """Pre-compute many-body operators for efficiency."""
        self.ops_XX = []
        self.ops_Z = []

        # XX coupling terms
        for i in range(self.L - 1):
            term = [self.id] * self.L
            term[i] = self.sx
            term[i + 1] = self.sx
            full_term = self._kron_chain(term)
            self.ops_XX.append(full_term)

        # Z field terms
        for i in range(self.L):
            term = [self.id] * self.L
            term[i] = self.sz
            full_term = self._kron_chain(term)
            self.ops_Z.append(full_term)

        # Imbalance observable: M = Σ_i (-1)^i Z_i
        self.imbalance_op = sum(
            [((-1) ** i) * self.ops_Z[i] for i in range(self.L)]
        )

    def _kron_chain(self, ops: list) -> sparse.csr_matrix:
        """Compute tensor product of a list of operators."""
        result = ops[0]
        for op in ops[1:]:
            result = sparse.kron(result, op, format='csr')
        return result

    def simulate_dynamics(
        self,
        J_couplings: np.ndarray,
        h_fields: np.ndarray,
        t_points: np.ndarray,
        initial_state: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Simulate time evolution and compute imbalance trace.

        Parameters
        ----------
        J_couplings : array, shape (L-1,)
            Coupling strengths between adjacent qubits.
        h_fields : array, shape (L,)
            On-site Z-field strengths.
        t_points : array
            Time points at which to evaluate the observable.
        initial_state : array, optional
            Initial state vector. Defaults to Néel state |010101...⟩.

        Returns
        -------
        imbalance : array
            Imbalance I(t) at each time point.
        """
        # Build Hamiltonian
        H = sparse.csr_matrix((self.dim, self.dim), dtype=complex)
        for i, J in enumerate(J_couplings):
            H += J * self.ops_XX[i]
        for i, h in enumerate(h_fields):
            H += h * self.ops_Z[i]

        # Initial state: Néel |010101...⟩
        if initial_state is None:
            psi = np.zeros(self.dim, dtype=complex)
            neel_idx = int("".join(["01"] * (self.L // 2)), 2)
            psi[neel_idx] = 1.0
        else:
            psi = initial_state.astype(complex)

        # Compute imbalance at each time point
        imbalances = []
        for t in t_points:
            psi_t = splinalg.expm_multiply(-1j * H * t, psi)
            I_t = np.vdot(psi_t, self.imbalance_op.dot(psi_t)).real / self.L
            imbalances.append(I_t)

        return np.array(imbalances)

    def learn_hamiltonian(
        self,
        experimental_trace: np.ndarray,
        t_points: np.ndarray,
        h_fields: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        bounds: Optional[list] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Reconstruct coupling parameters from experimental data.

        Parameters
        ----------
        experimental_trace : array
            Measured imbalance I(t) at each time point.
        t_points : array
            Time points corresponding to experimental_trace.
        h_fields : array, shape (L,)
            Known on-site fields (held fixed during optimization).
        initial_guess : array, optional
            Initial guess for J couplings. Defaults to uniform J=1.
        bounds : list, optional
            Bounds for each coupling. Defaults to [(0, 2)] for each.

        Returns
        -------
        J_recovered : array
            Reconstructed coupling strengths.
        fit_error : float
            Final mean squared error.
        """
        if initial_guess is None:
            initial_guess = np.ones(self.L - 1)
        if bounds is None:
            bounds = [(0.0, 2.0) for _ in range(self.L - 1)]

        def loss(J_guess):
            sim_trace = self.simulate_dynamics(J_guess, h_fields, t_points)
            mse = np.mean((sim_trace - experimental_trace) ** 2)
            return mse * 1e5  # Scale for optimizer

        result = minimize(
            loss,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-9, 'maxiter': 500}
        )

        return result.x, result.fun / 1e5

    def detect_defects(
        self,
        J_recovered: np.ndarray,
        J_nominal: float = 1.0,
        threshold: float = 0.1
    ) -> dict:
        """
        Identify defective couplings from recovered parameters.

        Parameters
        ----------
        J_recovered : array
            Reconstructed coupling strengths.
        J_nominal : float
            Expected nominal coupling value.
        threshold : float
            Fractional deviation to flag as defect.

        Returns
        -------
        defects : dict
            Dictionary with 'weak' and 'strong' coupling indices
            and per-coupling deviation magnitudes.
        """
        deviations = np.abs(J_recovered - J_nominal) / J_nominal

        weak = np.where(J_recovered < J_nominal * (1 - threshold))[0]
        strong = np.where(J_recovered > J_nominal * (1 + threshold))[0]

        return {
            "weak_couplings": weak.tolist(),
            "strong_couplings": strong.tolist(),
            "deviations": deviations.tolist()
        }

    def compare_with_gst(
        self,
        J_recovered: np.ndarray,
        gst_fidelities: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Compare Hamiltonian learning results with GST fidelity data.

        Merged from stim-cirq-qec's GST comparison logic.

        Parameters
        ----------
        J_recovered : array
            Coupling strengths from Hamiltonian learning.
        gst_fidelities : array, optional
            Per-gate fidelities from GST benchmark.

        Returns
        -------
        dict
            Comparison metrics including correlation and per-qubit
            agreement between the two diagnostic methods.
        """
        if gst_fidelities is None:
            return {"status": "no_gst_data", "correlation": None}

        # Convert coupling deviations to effective infidelity proxy
        J_nominal = np.mean(J_recovered)
        infidelity_proxy = np.abs(J_recovered - J_nominal) / J_nominal

        # Truncate to matching length
        min_len = min(len(infidelity_proxy), len(gst_fidelities))
        proxy = infidelity_proxy[:min_len]
        gst = 1 - gst_fidelities[:min_len]  # Convert fidelity to infidelity

        # Compute correlation
        if np.std(proxy) > 0 and np.std(gst) > 0:
            correlation = np.corrcoef(proxy, gst)[0, 1]
        else:
            correlation = 0.0

        return {
            "status": "compared",
            "correlation": float(correlation),
            "hamiltonian_infidelity": proxy.tolist(),
            "gst_infidelity": gst.tolist(),
        }
