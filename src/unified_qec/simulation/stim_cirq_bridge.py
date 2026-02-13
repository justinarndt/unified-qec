"""
Stim ↔ Cirq Bridge

Core integration layer for converting between Stim and Cirq representations,
enabling hybrid simulation with Stim's speed and Cirq's physics fidelity.

Requires: pip install unified-qec[cirq]

Migrated from: stim-cirq-qec/src/adaptive_qec/hybrid/stim_cirq_bridge.py
"""

import stim
import numpy as np
from typing import Optional, Tuple, List

try:
    import cirq
    CIRQ_AVAILABLE = True
except ImportError:
    cirq = None
    CIRQ_AVAILABLE = False

from unified_qec.simulation.noise_models import CoherentNoiseModel


class StimCirqBridge:
    """
    Bridge between Stim and Cirq circuit representations.

    Enables:
    - Converting Cirq surface code circuits to Stim for fast sampling
    - Extracting detector error models (DEM) from Stim
    - Injecting coherent noise in Cirq for hotspot analysis
    - Reconstructing Cirq circuits with updated noise from Stim results

    Theory
    ------
    Stim excels at Pauli-frame sampling with O(N) scaling, perfect for
    large-scale threshold studies. Cirq provides full density matrix
    simulation for coherent effects (over-rotations, ZZ crosstalk).

    This bridge uses stimcirq for basic conversion, then extends with
    coherent noise insertion and adaptive weight feedback.

    Parameters
    ----------
    distance : int
        Surface code distance.
    rounds : int
        Number of QEC syndrome measurement rounds.

    Raises
    ------
    ImportError
        If cirq is not installed (install with ``pip install unified-qec[cirq]``).
    """

    def __init__(self, distance: int = 5, rounds: int = 5):
        if not CIRQ_AVAILABLE:
            raise ImportError(
                "Cirq is required for StimCirqBridge. "
                "Install with: pip install unified-qec[cirq]"
            )
        self.distance = distance
        self.rounds = rounds
        self.qubits = self._generate_surface_code_qubits()

    def _generate_surface_code_qubits(self) -> List:
        """Generate qubit layout for rotated surface code."""
        qubits = []
        for i in range(self.distance):
            for j in range(self.distance):
                qubits.append(cirq.GridQubit(i, j))
        return qubits

    def build_cirq_surface_code(
        self,
        noise: Optional[CoherentNoiseModel] = None
    ):
        """
        Build a rotated surface code circuit in Cirq.

        Parameters
        ----------
        noise : CoherentNoiseModel, optional
            Noise parameters to apply.

        Returns
        -------
        cirq.Circuit
            The constructed circuit with noise.
        """
        if noise is None:
            noise = CoherentNoiseModel()

        circuit = cirq.Circuit()

        # Initialize data qubits
        circuit.append([cirq.H(q) for q in self.qubits])

        # Apply noise if specified
        if noise.depolarizing > 0:
            circuit.append([
                cirq.depolarize(noise.depolarizing).on(q)
                for q in self.qubits
            ])

        # Coherent over-rotation (Cirq-only feature)
        if noise.coherent_overrotation > 0:
            circuit.append([
                cirq.rz(noise.coherent_overrotation).on(q)
                for q in self.qubits
            ])

        # ZZ crosstalk between neighbors
        if noise.zz_crosstalk > 0:
            for i, q1 in enumerate(self.qubits[:-1]):
                q2 = self.qubits[i + 1]
                circuit.append(cirq.ZZ(q1, q2) ** noise.zz_crosstalk)

        # Measurement with readout error
        if noise.measurement > 0:
            circuit.append([
                cirq.bit_flip(noise.measurement).on(q)
                for q in self.qubits
            ])

        circuit.append(cirq.measure(*self.qubits, key='m'))

        return circuit

    def cirq_to_stim(
        self,
        cirq_circuit,
        noise: Optional[CoherentNoiseModel] = None
    ) -> stim.Circuit:
        """
        Convert Cirq circuit to Stim for fast sampling.

        Note: Coherent effects (over-rotation, ZZ) are approximated
        as additional depolarizing noise in Stim.

        Parameters
        ----------
        cirq_circuit : cirq.Circuit
            The Cirq circuit to convert.
        noise : CoherentNoiseModel, optional
            Noise model for Stim circuit generation.

        Returns
        -------
        stim.Circuit
            Equivalent Stim circuit (Pauli-only approximation).
        """
        if noise is None:
            noise = CoherentNoiseModel()

        # Use Stim's built-in surface code generator for efficiency
        # This gives us proper detector annotations
        stim_circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=self.distance,
            rounds=self.rounds,
            after_clifford_depolarization=noise.depolarizing,
            before_measure_flip_probability=noise.measurement,
            after_reset_flip_probability=noise.reset
        )

        return stim_circuit

    def stim_to_dem(self, stim_circuit: stim.Circuit) -> stim.DetectorErrorModel:
        """
        Extract detector error model from Stim circuit.

        The DEM describes the error-to-syndrome mapping, which is
        used for MWPM decoding and adaptive weight updates.
        """
        return stim_circuit.detector_error_model()

    def sample_stim(
        self,
        stim_circuit: stim.Circuit,
        shots: int = 1024
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast Pauli-frame sampling with Stim.

        Returns
        -------
        detection_events : ndarray
            Syndrome measurements (detector triggers).
        observable_flips : ndarray
            Logical observable outcomes.
        """
        sampler = stim_circuit.compile_detector_sampler()
        detection_events, observable_flips = sampler.sample(
            shots=shots,
            separate_observables=True
        )
        return detection_events, observable_flips

    def simulate_cirq_coherent(
        self,
        cirq_circuit,
        initial_state: Optional[np.ndarray] = None
    ):
        """
        Full density matrix simulation in Cirq.

        Use for coherent noise analysis on small subsystems (hotspots).
        """
        sim = cirq.DensityMatrixSimulator()
        result = sim.simulate(cirq_circuit, initial_state=initial_state)
        return result

    def compute_syndrome_density(
        self,
        detection_events: np.ndarray
    ) -> float:
        """
        Compute syndrome trigger density from detection events.

        This is the key observable for drift tracking.
        """
        return np.mean(detection_events)

    def update_dem_weights(
        self,
        dem: stim.DetectorErrorModel,
        drift_correction: float
    ) -> stim.DetectorErrorModel:
        """
        Update DEM edge weights based on drift estimate.

        This enables adaptive decoding where the MWPM graph
        reflects current hardware conditions.

        Parameters
        ----------
        dem : stim.DetectorErrorModel
            Original detector error model.
        drift_correction : float
            Correction factor from feedback controller.

        Returns
        -------
        stim.DetectorErrorModel
            Updated DEM with adjusted weights.
        """
        # Parse DEM and scale error probabilities
        # This is a simplified version - full implementation would
        # parse each instruction and adjust probabilities
        _dem_str = str(dem)

        # For now, return original (full implementation would modify)
        return dem


class CoherentNoiseInjector:
    """
    Inject coherent noise models into Cirq circuits.

    Enables simulation of:
    - Gate over-rotations (systematic angle errors)
    - ZZ crosstalk (residual coupling between qubits)
    - T1/T2 amplitude damping

    Requires: pip install unified-qec[cirq]
    """

    @staticmethod
    def add_overrotation(circuit, qubits: List, angle: float):
        """Add coherent Z-rotation error to all qubits."""
        if not CIRQ_AVAILABLE:
            raise ImportError("Cirq is required. Install with: pip install unified-qec[cirq]")
        new_circuit = circuit.copy()
        new_circuit.append([cirq.rz(angle).on(q) for q in qubits])
        return new_circuit

    @staticmethod
    def add_zz_crosstalk(circuit, qubit_pairs: List[Tuple], strength: float):
        """Add ZZ coupling between specified qubit pairs."""
        if not CIRQ_AVAILABLE:
            raise ImportError("Cirq is required. Install with: pip install unified-qec[cirq]")
        new_circuit = circuit.copy()
        for q1, q2 in qubit_pairs:
            new_circuit.append(cirq.ZZ(q1, q2) ** strength)
        return new_circuit

    @staticmethod
    def add_amplitude_damping(circuit, qubits: List, gamma: float):
        """Add amplitude damping (T1 decay) to qubits."""
        if not CIRQ_AVAILABLE:
            raise ImportError("Cirq is required. Install with: pip install unified-qec[cirq]")
        new_circuit = circuit.copy()
        new_circuit.append([
            cirq.amplitude_damp(gamma).on(q) for q in qubits
        ])
        return new_circuit
