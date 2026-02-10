"""
Aubry-André Disorder Model

Generator for quasi-periodic disorder fields that drive Many-Body
Localization (MBL). The deterministic potential h_i = Δ cos(2π β i)
with irrational β produces reproducible disorder patterns.

Migrated from: realtime-qec/realtime_qec/diagnostics/hamiltonian_learner.py
"""

import numpy as np


class AubryAndreModel:
    """
    Generator for Aubry-André quasi-periodic disorder fields.

    The potential h_i = Δ cos(2πβi) with β = (1+√5)/2 (golden ratio)
    creates a deterministic, reproducible disorder pattern. Above
    the critical disorder Δ_c = 2J, the system transitions to a
    Many-Body Localized (MBL) phase.

    Notes
    -----
    The MBL transition occurs at Δ/J ≈ 3.5 for interacting systems.
    Default Δ = 6.0 places the system deep in the MBL phase.
    """

    GOLDEN_RATIO = (1 + np.sqrt(5)) / 2

    @staticmethod
    def generate_fields(
        L: int,
        disorder_strength: float = 6.0,
        beta: float = None
    ) -> np.ndarray:
        """
        Generate quasi-periodic on-site fields.

        Parameters
        ----------
        L : int
            System size (number of sites).
        disorder_strength : float
            Amplitude Δ of the quasi-periodic potential.
        beta : float, optional
            Quasi-periodicity parameter. Defaults to golden ratio.

        Returns
        -------
        h_fields : array, shape (L,)
            On-site Z-field strengths.
        """
        if beta is None:
            beta = AubryAndreModel.GOLDEN_RATIO

        return disorder_strength * np.cos(2 * np.pi * beta * np.arange(L))
