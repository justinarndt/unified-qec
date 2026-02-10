"""
Leakage Modeling for Transmon Qubits

Models qubit leakage to |2⟩ (f) state and seepage back to computational
basis. Addresses a critical "Reality Gap" in hardware validation.

Transmons are anharmonic oscillators, not true qubits. During hard pulses
(especially pulse remediation), population can leak to higher levels.
Leakage is "virulent" — it spreads through stabilizers and is invisible
to standard Pauli decoding.

Migrated from: stim-cirq-qec/src/adaptive_qec/physics/leakage.py
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LeakageState:
    """Snapshot of leakage state at a given cycle."""
    cycle: int
    num_leaked: int
    leaked_qubits: List[int]
    total_leakage_events: int
    total_seepage_events: int


class LeakageTracker:
    """
    Track qubit leakage to |2⟩ state during QEC cycles.

    In real transmon hardware, qubits can leak to the |2⟩ (f) state during:
    - Strong drive pulses (especially remediation pulses)
    - Long gate sequences
    - Two-qubit gates with large ZZ coupling

    Leaked qubits:
    - Produce incorrect syndrome measurements (look like biased errors)
    - Are invisible to Pauli-channel simulation (Stim sees them as "lost")
    - Create an error floor that limits suppression at high distances

    Parameters
    ----------
    num_qubits : int
        Number of data qubits to track.
    leakage_rate : float
        Per-gate probability of |1⟩ → |2⟩ transition.
    seepage_rate : float
        Per-gate probability of |2⟩ → |1⟩ return (leakage recovery).
    """

    def __init__(
        self,
        num_qubits: int,
        leakage_rate: float = 0.001,
        seepage_rate: float = 0.01
    ):
        self.num_qubits = num_qubits
        self.leakage_rate = leakage_rate
        self.seepage_rate = seepage_rate

        self.leaked = np.zeros(num_qubits, dtype=bool)
        self.total_leakage_events = 0
        self.total_seepage_events = 0
        self.cycle_count = 0
        self.history: List[LeakageState] = []

    def apply_gate_leakage(
        self,
        qubit_indices: Optional[List[int]] = None,
        multiplier: float = 1.0
    ) -> int:
        """
        Apply probabilistic leakage to qubits during gate operations.

        Parameters
        ----------
        qubit_indices : list of int, optional
            Qubits involved in the gate. If None, all qubits.
        multiplier : float
            Leakage rate multiplier (e.g., 2.0 for two-qubit gates).

        Returns
        -------
        int
            Number of new leakage events this step.
        """
        if qubit_indices is None:
            qubit_indices = list(range(self.num_qubits))

        new_leaks = 0
        effective_rate = self.leakage_rate * multiplier

        for q in qubit_indices:
            if not self.leaked[q]:
                if np.random.random() < effective_rate:
                    self.leaked[q] = True
                    self.total_leakage_events += 1
                    new_leaks += 1

        return new_leaks

    def apply_seepage(self) -> int:
        """
        Apply probabilistic seepage (return from |2⟩ to |1⟩).

        Models Leakage Reduction Units (LRU) or natural T1-like
        decay from the |2⟩ state back to the computational basis.
        """
        seepage_events = 0

        for q in range(self.num_qubits):
            if self.leaked[q]:
                if np.random.random() < self.seepage_rate:
                    self.leaked[q] = False
                    self.total_seepage_events += 1
                    seepage_events += 1

        return seepage_events

    def run_cycle(self, num_gates_per_qubit: int = 5) -> Tuple[int, int]:
        """
        Simulate one QEC cycle of leakage/seepage.

        Parameters
        ----------
        num_gates_per_qubit : int
            Approximate gate count per qubit per cycle.

        Returns
        -------
        tuple
            (new_leaks, seepage_events) for this cycle.
        """
        self.cycle_count += 1

        total_new_leaks = 0
        for _ in range(num_gates_per_qubit):
            total_new_leaks += self.apply_gate_leakage()

        seepage = self.apply_seepage()

        self.history.append(LeakageState(
            cycle=self.cycle_count,
            num_leaked=self.get_num_leaked(),
            leaked_qubits=self.get_leaked_qubits(),
            total_leakage_events=self.total_leakage_events,
            total_seepage_events=self.total_seepage_events
        ))

        return total_new_leaks, seepage

    def get_num_leaked(self) -> int:
        """Get current number of leaked qubits."""
        return int(np.sum(self.leaked))

    def get_leaked_qubits(self) -> List[int]:
        """Get indices of currently leaked qubits."""
        return list(np.where(self.leaked)[0])

    def get_leakage_error_contribution(self) -> float:
        """
        Estimate additional logical error rate from leakage.

        Leaked qubits contribute to logical errors because:
        1. They produce incorrect syndrome measurements
        2. The decoder cannot properly weight leaked locations
        3. Error chains through leaked regions are uncorrectable
        """
        num_leaked = self.get_num_leaked()
        if num_leaked == 0:
            return 0.0

        leaked_fraction = num_leaked / self.num_qubits
        return 0.5 * (1 - np.exp(-5 * leaked_fraction))

    def reset(self):
        """Reset all leakage state."""
        self.leaked = np.zeros(self.num_qubits, dtype=bool)
        self.total_leakage_events = 0
        self.total_seepage_events = 0
        self.cycle_count = 0
        self.history = []

    def get_statistics(self) -> dict:
        """Get summary statistics."""
        return {
            "num_qubits": self.num_qubits,
            "leakage_rate": self.leakage_rate,
            "seepage_rate": self.seepage_rate,
            "current_leaked": self.get_num_leaked(),
            "total_leakage_events": self.total_leakage_events,
            "total_seepage_events": self.total_seepage_events,
            "cycles": self.cycle_count,
            "error_contribution": self.get_leakage_error_contribution()
        }
