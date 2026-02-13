"""
Surface Code Circuit Generation with Drifting Noise

Provides Stim-based circuit generation with time-varying noise models
for testing adaptive QEC algorithms. Includes Ornstein-Uhlenbeck drift
for realistic hardware simulation.

Migrated from: realtime-qec/realtime_qec/simulation/circuits.py
"""

import stim
import pymatching
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class NoiseParameters:
    """Hardware noise specification for stochastic (Pauli) noise channels.

    Attributes
    ----------
    gate_error : float
        Depolarizing error rate per Clifford gate.
    measurement_error : float
        Bit-flip probability before measurement.
    reset_error : float
        Bit-flip probability after reset.
    """
    gate_error: float = 0.001
    measurement_error: float = 0.01
    reset_error: float = 0.005

    def scale(self, factor: float) -> 'NoiseParameters':
        """Return scaled copy of noise parameters."""
        return NoiseParameters(
            gate_error=self.gate_error * factor,
            measurement_error=self.measurement_error * factor,
            reset_error=self.reset_error * factor
        )


class DriftingNoiseModel:
    """
    Non-stationary noise model for QEC simulation.

    Models realistic hardware drift including:
    - Global T1/T2 fluctuations
    - Per-qubit local drift
    - Ornstein-Uhlenbeck colored noise

    Theory
    ------
    Error rate evolves as:
        p(t) = p_base + δ_global(t) + δ_local(q, t)

    where δ follows an OU process:
        dδ = θ(μ - δ)dt + σ dW

    Parameters
    ----------
    num_qubits : int
        Number of data qubits in the system.
    base_params : NoiseParameters, optional
        Baseline noise parameters.
    drift_rate : float
        OU process mean-reversion rate θ.
    drift_target : float
        OU process mean μ (drift converges here).
    noise_std : float
        OU process noise amplitude σ.
    """

    def __init__(
        self,
        num_qubits: int,
        base_params: Optional[NoiseParameters] = None,
        drift_rate: float = 0.005,
        drift_target: float = 0.03,
        noise_std: float = 1e-6
    ):
        self.num_qubits = num_qubits
        self.base = base_params or NoiseParameters()

        # OU process parameters
        self.theta = drift_rate
        self.mu = drift_target
        self.sigma = noise_std

        # State
        self.global_drift = 0.0
        self.local_drift = np.zeros(num_qubits)
        self.time_step = 0

    def step(self) -> float:
        """
        Advance drift by one time step.

        Returns
        -------
        float
            Average effective error rate after drift.
        """
        self.time_step += 1

        # Global OU update
        d_global = self.theta * (self.mu * 0.2 - self.global_drift)
        self.global_drift += d_global + np.random.normal(0, self.sigma)

        # Local OU update (per-qubit)
        d_local = self.theta * (self.mu - self.local_drift)
        self.local_drift += d_local + np.random.normal(0, self.sigma * 2, size=self.num_qubits)

        # Clamp to physical range
        self.local_drift = np.clip(self.local_drift, -0.02, 0.05)

        return self.get_effective_error()

    def get_effective_error(self) -> float:
        """Get current effective gate error rate."""
        return self.base.gate_error + self.global_drift + np.mean(self.local_drift)

    def get_effective_params(self) -> NoiseParameters:
        """Get current noise parameters including drift."""
        return NoiseParameters(
            gate_error=np.clip(self.get_effective_error(), 1e-6, 0.4),
            measurement_error=self.base.measurement_error,
            reset_error=self.base.reset_error
        )

    def reset(self):
        """Reset drift state to baseline."""
        self.global_drift = 0.0
        self.local_drift = np.zeros(self.num_qubits)
        self.time_step = 0


class SurfaceCodeCircuit:
    """
    Rotated surface code circuit generator with configurable noise.

    Wraps Stim's circuit generation with additional features:
    - Dynamic noise parameters per round
    - Syndrome density measurement
    - Batch decoding with PyMatching

    Parameters
    ----------
    distance : int
        Code distance d. Uses d² data qubits.
    rounds : int
        Number of QEC syndrome measurement rounds.
    basis : str
        Memory basis, 'X' or 'Z'.
    """

    def __init__(
        self,
        distance: int = 5,
        rounds: int = 5,
        basis: str = 'Z'
    ):
        self.distance = distance
        self.rounds = rounds
        self.basis = basis.lower()
        self.num_data_qubits = distance ** 2

    def generate(self, params: NoiseParameters) -> stim.Circuit:
        """
        Generate a noisy surface code circuit.

        Parameters
        ----------
        params : NoiseParameters
            Noise parameters for this circuit.

        Returns
        -------
        stim.Circuit
            The generated circuit.
        """
        code_type = f"surface_code:rotated_memory_{self.basis}"

        circuit = stim.Circuit.generated(
            code_type,
            distance=self.distance,
            rounds=self.rounds,
            after_clifford_depolarization=params.gate_error,
            before_measure_flip_probability=params.measurement_error,
            after_reset_flip_probability=params.reset_error
        )

        return circuit

    def run_batch(
        self,
        params: NoiseParameters,
        batch_size: int = 1024,
        correction: float = 0.0
    ) -> Tuple[int, float]:
        """
        Run a batch of shots and decode.

        Parameters
        ----------
        params : NoiseParameters
            Noise parameters for this batch.
        batch_size : int
            Number of shots to sample.
        correction : float
            Correction offset to apply to effective error rate.

        Returns
        -------
        failures : int
            Number of logical errors.
        density : float
            Average syndrome trigger density.
        """
        # Apply correction to effective noise (modeling feedback)
        effective_gate = np.clip(params.gate_error - correction, 1e-6, 0.4)
        effective_params = NoiseParameters(
            gate_error=effective_gate,
            measurement_error=params.measurement_error,
            reset_error=params.reset_error
        )

        # Generate and decode
        circuit = self.generate(effective_params)
        decoder = pymatching.Matching.from_stim_circuit(circuit)

        sampler = circuit.compile_detector_sampler()
        detection_events, observable_flips = sampler.sample(
            shots=batch_size,
            separate_observables=True
        )

        predictions = decoder.decode_batch(detection_events)

        # Handle shape differences
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        if observable_flips.ndim > 1:
            observable_flips = observable_flips.flatten()

        failures = np.sum(observable_flips != predictions)
        density = np.mean(detection_events)

        return failures, density

    def compute_logical_error_rate(
        self,
        params: NoiseParameters,
        num_shots: int = 10000
    ) -> Tuple[float, float]:
        """
        Compute logical error rate with confidence interval.

        Returns
        -------
        error_rate : float
            Estimated logical error rate.
        std_error : float
            Standard error of the estimate.
        """
        failures, _ = self.run_batch(params, batch_size=num_shots)
        error_rate = failures / num_shots
        std_error = np.sqrt(error_rate * (1 - error_rate) / num_shots)

        return error_rate, std_error
