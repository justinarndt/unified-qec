"""
Merged Syndrome-Informed Feedback Controller

Combines features from both stim-cirq-qec and realtime-qec controllers:
- Integral control with configurable gain (both)
- Feedback latency modeling via correction queue (both)
- T1/T2 decay penalty during feedback window (stim-cirq-qec)
- Correction bounds with clamping (realtime-qec)
- History tracking for analysis (realtime-qec)

Migrated from:
  - stim-cirq-qec/src/adaptive_qec/feedback/controller.py
  - realtime-qec/realtime_qec/feedback/controller.py
"""

import numpy as np
from typing import Optional, Tuple, Dict
from collections import deque


class SyndromeFeedbackController:
    """
    Closed-loop controller for adaptive QEC.

    Tracks drift via syndrome density and provides correction
    signals for decoder weight updates. Includes realistic feedback
    latency modeling with T1/T2 decay during the processing window.

    Theory
    ------
    Under stationary noise p, syndrome density converges to:
        ρ_ss = f(p, d)

    When noise drifts to p(t) = p_0 + δ(t), the density deviates:
        ρ(t) = ρ_ss + g(δ(t))

    The integral controller estimates δ(t) from ρ(t) - ρ_ss and
    feeds this correction to the decoder's edge weight calculation.

    Parameters
    ----------
    Ki : float
        Integral gain. Higher values track faster but may oscillate.
    feedback_latency : int
        Number of QEC rounds of delay in the feedback loop.
    correction_bounds : tuple
        (min, max) bounds on the correction signal.
    latency_ns : float
        Physical feedback latency in nanoseconds.
    t1_us : float
        T1 relaxation time in microseconds.
    t2_us : float
        T2 dephasing time in microseconds.

    Hardware Reference
    ------------------
    The ``latency_ns`` parameter models the physical feedback delay
    measured in the FPGA Pauli frame tracking benchmark (``rtl/``).
    The Apex LUTRAM tracker achieves single-cycle (<10 ns) updates
    with zero stalls, while the BRAM baseline incurs 50% throughput
    loss from synchronous read latency. See ``rtl/sim/benchmark_harness.sv``
    for the full BRAM-vs-LUTRAM comparison.
    """

    def __init__(
        self,
        Ki: float = 0.05,
        feedback_latency: int = 10,
        correction_bounds: Tuple[float, float] = (-0.02, 0.15),
        latency_ns: float = 0.0,
        t1_us: float = 100.0,
        t2_us: float = 80.0
    ):
        self.Ki = Ki
        self.latency = feedback_latency
        self.bounds = correction_bounds
        self.latency_ns = latency_ns
        self.t1_us = t1_us
        self.t2_us = t2_us

        self.integrator_state = 0.0
        self.setpoint: Optional[float] = None
        self.correction_queue = deque([0.0] * feedback_latency, maxlen=feedback_latency)

        self.history: Dict[str, list] = {
            "density": [],
            "correction": [],
            "error": [],
            "decay_penalty": [],
        }

    def compute_latency_decay_penalty(self) -> float:
        """
        Compute additional error probability from T1/T2 decay during latency.

        During feedback processing, qubits idle and experience decoherence.
        Returns the probability of an error occurring during this window.

        Returns
        -------
        float
            Decay-induced error probability (0 to ~1).
        """
        if self.latency_ns <= 0:
            return 0.0

        idle_time_us = self.latency_ns / 1000.0
        t1_decay = 1 - np.exp(-idle_time_us / self.t1_us)
        t2_decay = 1 - np.exp(-idle_time_us / self.t2_us)

        # Combined decay penalty (approximately additive for small rates)
        decay_penalty = 0.5 * (t1_decay + t2_decay)
        return min(decay_penalty, 1.0)

    def calibrate(self, plant, num_samples: int = 10) -> float:
        """
        Calibrate setpoint from baseline measurements.

        Parameters
        ----------
        plant : SurfaceCodeCircuit or compatible
            The quantum plant to measure.
        num_samples : int
            Number of measurement batches for averaging.

        Returns
        -------
        float
            The calibrated setpoint (baseline density).
        """
        densities = []
        for _ in range(num_samples):
            _, density = plant.run_batch(correction=0.0)
            densities.append(density)

        self.setpoint = np.mean(densities)
        self.integrator_state = 0.0
        self.correction_queue = deque([0.0] * self.latency, maxlen=self.latency)

        return self.setpoint

    def update(self, measured_density: float) -> float:
        """
        Compute the next correction signal based on observed density.

        Parameters
        ----------
        measured_density : float
            Syndrome density from the most recent QEC round.

        Returns
        -------
        float
            Correction signal to apply to decoder weights.
        """
        if self.setpoint is None:
            raise ValueError("Controller not calibrated. Call calibrate() first.")

        error = measured_density - self.setpoint
        self.integrator_state += error * self.Ki
        self.integrator_state = np.clip(self.integrator_state, *self.bounds)

        self.correction_queue.append(self.integrator_state)
        correction = self.correction_queue[0]

        self.history["density"].append(measured_density)
        self.history["correction"].append(correction)
        self.history["error"].append(error)
        self.history["decay_penalty"].append(self.compute_latency_decay_penalty())

        return correction

    def get_active_correction(self) -> float:
        """Get the currently active correction (accounting for latency)."""
        return self.correction_queue[0]

    def reset(self):
        """Reset controller state while preserving setpoint."""
        self.integrator_state = 0.0
        self.correction_queue = deque([0.0] * self.latency, maxlen=self.latency)
        self.history = {
            "density": [], "correction": [],
            "error": [], "decay_penalty": [],
        }
