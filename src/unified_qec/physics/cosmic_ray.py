"""
Cosmic Ray Event Simulation

Simulates high-energy cosmic ray impacts that create burst errors
across multiple qubits.

Migrated from: stim-cirq-qec/src/adaptive_qec/physics/cosmic_ray.py
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class CosmicRayImpact:
    """A simulated cosmic ray impact event."""
    cycle: int
    center_qubit: int
    radius: int
    affected_qubits: List[int]
    depolarization_strength: float


class CosmicRaySimulator:
    """
    Simulate cosmic ray impacts on a qubit array.

    Cosmic rays deposit energy in a localized region, causing:
    - High depolarization in the impact center
    - Decreasing noise with distance from center
    - Potentially knocking qubits into leaked states

    Parameters
    ----------
    distance : int
        Surface code distance.
    impact_rate : float
        Expected impacts per 1000 cycles.
    typical_radius : int
        Typical impact radius in qubits.
    max_depol : float
        Maximum depolarization at impact center.
    """

    def __init__(
        self,
        distance: int = 7,
        impact_rate: float = 0.5,
        typical_radius: int = 2,
        max_depol: float = 0.5
    ):
        self.distance = distance
        self.impact_rate = impact_rate
        self.typical_radius = typical_radius
        self.max_depol = max_depol

        self.num_qubits = distance ** 2
        self.impact_history: List[CosmicRayImpact] = []
        self.cycle_count = 0

    def check_for_impact(self, cycle: Optional[int] = None) -> Optional[CosmicRayImpact]:
        """
        Check if a cosmic ray impact occurs this cycle.

        Returns
        -------
        CosmicRayImpact or None
            Impact event if one occurs, else None.
        """
        if cycle is None:
            cycle = self.cycle_count
        self.cycle_count = cycle + 1

        prob_per_cycle = self.impact_rate / 1000.0
        if np.random.random() > prob_per_cycle:
            return None

        return self.generate_impact(cycle)

    def generate_impact(
        self,
        cycle: int,
        center: Optional[int] = None,
        radius: Optional[int] = None
    ) -> CosmicRayImpact:
        """
        Generate a cosmic ray impact event.

        Parameters
        ----------
        cycle : int
            Cycle of impact.
        center : int, optional
            Center qubit. Random if not specified.
        radius : int, optional
            Impact radius. Uses typical if not specified.
        """
        if center is None:
            center = np.random.randint(0, self.num_qubits)
        if radius is None:
            radius = max(1, int(np.random.exponential(self.typical_radius)))

        affected = self._get_qubits_in_radius(center, radius)

        impact = CosmicRayImpact(
            cycle=cycle,
            center_qubit=center,
            radius=radius,
            affected_qubits=affected,
            depolarization_strength=self.max_depol
        )

        self.impact_history.append(impact)
        return impact

    def _get_qubits_in_radius(self, center: int, radius: int) -> List[int]:
        """Get all qubits within Manhattan distance radius of center."""
        center_row = center // self.distance
        center_col = center % self.distance

        affected = []
        for q in range(self.num_qubits):
            row = q // self.distance
            col = q % self.distance
            dist = abs(row - center_row) + abs(col - center_col)
            if dist <= radius:
                affected.append(q)

        return affected

    def get_depolarization_map(self, impact: CosmicRayImpact) -> np.ndarray:
        """
        Get per-qubit depolarization rates for an impact.

        Depolarization decreases exponentially with distance from center.
        """
        depol_map = np.zeros(self.num_qubits)
        center_row = impact.center_qubit // self.distance
        center_col = impact.center_qubit % self.distance

        for q in impact.affected_qubits:
            row = q // self.distance
            col = q % self.distance
            dist = abs(row - center_row) + abs(col - center_col)
            depol_map[q] = impact.depolarization_strength * np.exp(-dist / 2)

        return depol_map

    def apply_to_noise_model(
        self,
        impact: CosmicRayImpact,
        base_depol: float = 0.001
    ) -> np.ndarray:
        """Get modified per-qubit noise rates after applying impact."""
        return base_depol + self.get_depolarization_map(impact)

    def reset(self):
        """Reset simulator state."""
        self.impact_history = []
        self.cycle_count = 0

    def get_statistics(self) -> dict:
        """Get simulation statistics."""
        return {
            "total_impacts": len(self.impact_history),
            "cycles_simulated": self.cycle_count,
            "impact_rate": self.impact_rate,
            "avg_radius": np.mean([i.radius for i in self.impact_history]) if self.impact_history else 0
        }


def inject_cosmic_ray(
    syndromes: np.ndarray,
    center: int,
    radius: int = 2,
    flip_probability: float = 0.8
) -> np.ndarray:
    """
    Inject a cosmic ray event into syndrome data.

    Convenience function for testing burst detection.
    """
    modified = syndromes.copy()
    for i in range(len(modified)):
        if abs(i - center) <= radius:
            if np.random.random() < flip_probability:
                modified[i] = 1 - modified[i]
    return modified
