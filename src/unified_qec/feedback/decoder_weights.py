"""
Adaptive Decoder Weight Calculator

Dynamic edge weight calculator for MWPM decoding. Adjusts decoder
edge weights based on drift estimates from the SyndromeFeedbackController.

Migrated from: realtime-qec/realtime_qec/feedback/controller.py
"""

import numpy as np


class AdaptiveDecoderWeights:
    """
    Dynamic edge weight calculator for MWPM decoding.

    Adjusts decoder edge weights based on drift estimates from
    the SyndromeFeedbackController, enabling the decoder to
    track non-stationary noise.

    Parameters
    ----------
    base_error_rate : float
        Baseline gate error rate for weight calculation.
    """

    def __init__(self, base_error_rate: float = 0.001):
        self.base_error_rate = base_error_rate

    def compute_weights(self, drift_estimate: float) -> float:
        """
        Compute effective error rate for decoder graph construction.

        Parameters
        ----------
        drift_estimate : float
            Current drift estimate from the feedback controller.

        Returns
        -------
        float
            Effective error rate for MWPM edge weight calculation.
        """
        p_effective = self.base_error_rate + drift_estimate
        return np.clip(p_effective, 1e-6, 0.49)

    def log_likelihood_weight(self, p: float) -> float:
        """
        Convert probability to log-likelihood weight for MWPM.

        The MWPM decoder uses weights w = log((1-p)/p).
        """
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return np.log((1 - p) / p)
