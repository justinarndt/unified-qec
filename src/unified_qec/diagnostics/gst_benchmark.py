"""
Gate Set Tomography (GST) Benchmark

SPAM-robust gate characterization using pyGSTi. Provides a reference
method for comparing against Hamiltonian learning diagnostics.

Requires: pip install unified-qec[gst]

Migrated from: stim-cirq-qec/src/adaptive_qec/diagnostics/gst_benchmark.py
"""

import numpy as np
from typing import Tuple, Optional, Dict

try:
    import pygsti
    PYGSTI_AVAILABLE = True
except ImportError:
    pygsti = None
    PYGSTI_AVAILABLE = False


class GSTBenchmark:
    """
    Gate Set Tomography benchmark for SPAM-robust characterization.

    Uses pyGSTi to generate GST circuits, simulate experiments, and
    run the GST protocol to estimate per-gate fidelities. This provides
    a gold-standard diagnostic method for comparison with Hamiltonian
    learning results.

    Parameters
    ----------
    num_qubits : int
        Number of qubits for GST.
    max_length : int
        Maximum gate sequence length for GST.

    Raises
    ------
    ImportError
        If pyGSTi is not installed.
    """

    def __init__(self, num_qubits: int = 1, max_length: int = 4):
        if not PYGSTI_AVAILABLE:
            raise ImportError(
                "pyGSTi is required for GSTBenchmark. "
                "Install with: pip install unified-qec[gst]"
            )
        self.num_qubits = num_qubits
        self.max_length = max_length
        self._model = None
        self._circuits = None

    def setup(self):
        """Generate GST circuits and target model."""
        # Standard single-qubit gate set
        self._model = pygsti.models.create_explicit_model_from_expressions(
            [('Q0',)],
            ['Gi', 'Gx', 'Gy'],
            ["I(Q0)", "X(pi/2, Q0)", "Y(pi/2, Q0)"],
            effectExpressions=['0', '1']
        )

        # Generate circuit lists for increasing sequence lengths
        fiducials = pygsti.circuits.to_circuits([
            '{}', 'Gx', 'Gy', 'GxGx', 'GxGxGx', 'GyGyGy'
        ])
        germs = pygsti.circuits.to_circuits([
            'Gi', 'Gx', 'Gy', 'GiGi', 'GxGy', 'GxGyGi',
            'GxGiGy', 'GxGiGi', 'GyGiGi', 'GxGxGiGy', 'GxGyGyGi',
        ])

        max_lengths = list(range(1, self.max_length + 1))
        self._circuits = pygsti.circuits.create_lsgst_circuits(
            self._model,
            fiducials, fiducials, germs, max_lengths
        )

    def run_benchmark(
        self,
        depolarizing_rate: float = 0.01,
        num_samples: int = 1000
    ) -> Dict:
        """
        Run GST benchmark with simulated data.

        Parameters
        ----------
        depolarizing_rate : float
            Simulated depolarizing error rate per gate.
        num_samples : int
            Number of samples per circuit.

        Returns
        -------
        dict
            GST results including per-gate fidelities and
            diamond distances.
        """
        if self._circuits is None:
            self.setup()

        # Create noisy model for simulation
        noisy_model = self._model.depolarize(op_noise=depolarizing_rate)

        # Simulate data
        data = pygsti.data.simulate_data(
            noisy_model, self._circuits, num_samples,
            seed=42
        )

        # Run GST
        results = pygsti.run_long_sequence_gst(
            data, self._model, self._circuits,
            verbosity=0
        )

        # Extract fidelities
        estimated_model = results.estimates['default'].models['go0']

        fidelities = {}
        for gate_label in ['Gi', 'Gx', 'Gy']:
            target = self._model.operations[gate_label].to_dense()
            estimated = estimated_model.operations[gate_label].to_dense()
            fidelity = self._process_fidelity(target, estimated)
            fidelities[gate_label] = float(fidelity)

        return {
            "fidelities": fidelities,
            "depolarizing_rate": depolarizing_rate,
            "num_circuits": len(self._circuits),
        }

    @staticmethod
    def _process_fidelity(target: np.ndarray, estimated: np.ndarray) -> float:
        """Compute process fidelity between target and estimated gates."""
        d = target.shape[0]
        overlap = np.abs(np.trace(target.T @ estimated))
        return (overlap + d) / (d * (d + 1))
