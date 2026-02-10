"""
Burst Error Detection for High-Energy Events

Detects cosmic ray impacts and other high-energy events that cause
correlated burst errors across multiple qubits.

Detection Strategy:
1. Monitor syndrome density for sudden spikes
2. Look for spatial clustering of triggered detectors
3. If burst detected, expand Cirq simulation region

Migrated from: stim-cirq-qec/src/adaptive_qec/physics/burst_detector.py
"""

import numpy as np
from typing import Optional, List, Tuple, Set
from dataclasses import dataclass


@dataclass
class BurstEvent:
    """Detected burst error event."""
    cycle: int
    center_qubit: int
    affected_qubits: List[int]
    syndrome_density: float
    severity: float  # 0-1 scale


class BurstErrorDetector:
    """
    Detect burst errors from high-energy events (cosmic rays, etc).

    High-energy events like cosmic ray impacts can instantly depolarize
    a patch of qubits, creating a burst of correlated errors that looks
    very different from normal QEC noise.

    Parameters
    ----------
    distance : int
        Surface code distance.
    spike_threshold : float
        Syndrome density deviation threshold for spike detection.
    spatial_threshold : int
        Minimum cluster size to consider a burst.
    cooldown_cycles : int
        Cycles to wait after detection before checking again.
    """

    def __init__(
        self,
        distance: int = 7,
        spike_threshold: float = 0.3,
        spatial_threshold: int = 3,
        cooldown_cycles: int = 5
    ):
        self.distance = distance
        self.spike_threshold = spike_threshold
        self.spatial_threshold = spatial_threshold
        self.cooldown_cycles = cooldown_cycles

        self.baseline_density: Optional[float] = None
        self.last_detection_cycle: int = -999
        self.cycle_count = 0

        self.detected_bursts: List[BurstEvent] = []
        self.density_history: List[float] = []

    def set_baseline(self, baseline_density: float):
        """Set baseline syndrome density for comparison."""
        self.baseline_density = baseline_density

    def detect(
        self,
        syndromes: np.ndarray,
        cycle: Optional[int] = None
    ) -> Optional[BurstEvent]:
        """
        Detect if current syndrome pattern indicates a burst event.

        Parameters
        ----------
        syndromes : np.ndarray
            Current syndrome measurements (2D or flattened).
        cycle : int, optional
            Current cycle number.

        Returns
        -------
        BurstEvent or None
            Detected burst event, or None if no burst.
        """
        if cycle is None:
            cycle = self.cycle_count
        self.cycle_count = cycle + 1

        if cycle - self.last_detection_cycle < self.cooldown_cycles:
            return None

        syndrome_flat = syndromes.flatten() if syndromes.ndim > 1 else syndromes
        current_density = np.mean(syndrome_flat)
        self.density_history.append(current_density)

        if self.baseline_density is None:
            return None

        deviation = current_density - self.baseline_density
        if deviation < self.spike_threshold:
            return None

        triggered = np.where(syndrome_flat > 0.5)[0]
        if len(triggered) < self.spatial_threshold:
            return None

        center = int(np.median(triggered))
        affected = self._find_affected_qubits(triggered)
        severity = min(deviation / 0.5, 1.0)

        burst = BurstEvent(
            cycle=cycle,
            center_qubit=center,
            affected_qubits=affected,
            syndrome_density=current_density,
            severity=severity
        )

        self.detected_bursts.append(burst)
        self.last_detection_cycle = cycle

        return burst

    def _find_affected_qubits(self, triggered: np.ndarray) -> List[int]:
        """Find data qubits affected by burst."""
        return list(triggered[:min(len(triggered), self.distance ** 2)])

    def get_expanded_cirq_region(
        self,
        burst: BurstEvent,
        expansion_radius: int = 2
    ) -> List[int]:
        """
        Get expanded qubit region for Cirq simulation.

        When a burst is detected, switch to Cirq for a larger region
        to properly model the correlated errors.
        """
        affected_set: Set[int] = set(burst.affected_qubits)

        for _ in range(expansion_radius):
            new_qubits = set()
            for q in affected_set:
                for delta in [-1, 1, -self.distance, self.distance]:
                    neighbor = q + delta
                    if 0 <= neighbor < self.distance ** 2:
                        new_qubits.add(neighbor)
            affected_set.update(new_qubits)

        return sorted(list(affected_set))

    def get_recovery_recommendation(self, burst: BurstEvent) -> dict:
        """
        Get recommended recovery actions for a burst.

        Returns
        -------
        dict
            Recovery recommendations.
        """
        return {
            "apply_extra_rounds": burst.severity > 0.5,
            "decoder_reweight": True,
            "affected_region": burst.affected_qubits,
            "severity": burst.severity,
            "cycles_until_recovery": max(5, int(10 * burst.severity))
        }

    def reset(self):
        """Reset detector state."""
        self.last_detection_cycle = -999
        self.cycle_count = 0
        self.detected_bursts = []
        self.density_history = []

    def get_statistics(self) -> dict:
        """Get detector statistics."""
        return {
            "total_bursts_detected": len(self.detected_bursts),
            "baseline_density": self.baseline_density,
            "cycles_processed": self.cycle_count,
            "density_history_length": len(self.density_history)
        }
