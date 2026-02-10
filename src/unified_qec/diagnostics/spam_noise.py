"""
SPAM Noise Models

Functions for injecting State Preparation and Measurement (SPAM) noise
and analyzing its impact on QEC performance.

Migrated from: stim-cirq-qec/src/adaptive_qec/diagnostics/spam_noise.py
"""

import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass


@dataclass
class SPAMNoiseModel:
    """SPAM noise model parameters.

    Attributes
    ----------
    prep_error_rate : float
        Probability of incorrect state preparation.
    meas_error_rate : float
        Probability of measurement misidentification.
    t1_limited : bool
        Whether errors are T1-limited.
    """
    prep_error_rate: float = 0.001
    meas_error_rate: float = 0.005
    t1_limited: bool = False


def inject_readout_noise(
    measurements: np.ndarray,
    error_rate: float = 0.005
) -> np.ndarray:
    """
    Inject readout (measurement) noise into measurement data.

    Parameters
    ----------
    measurements : np.ndarray
        Clean measurement outcomes (0/1).
    error_rate : float
        Probability of flipping each measurement outcome.

    Returns
    -------
    np.ndarray
        Noisy measurement outcomes.
    """
    noisy = measurements.copy()
    mask = np.random.random(measurements.shape) < error_rate
    noisy[mask] = 1 - noisy[mask]
    return noisy


def inject_state_prep_error(
    num_qubits: int,
    target_state: int = 0,
    error_rate: float = 0.001
) -> np.ndarray:
    """
    Simulate noisy state preparation.

    Parameters
    ----------
    num_qubits : int
        Number of qubits to prepare.
    target_state : int
        Target preparation state for each qubit (0 or 1).
    error_rate : float
        Probability of preparing incorrect state.

    Returns
    -------
    np.ndarray
        Prepared qubit states with possible errors.
    """
    states = np.full(num_qubits, target_state, dtype=int)
    mask = np.random.random(num_qubits) < error_rate
    states[mask] = 1 - states[mask]
    return states


def generate_spam_sweep(
    prep_rates: Optional[List[float]] = None,
    meas_rates: Optional[List[float]] = None,
) -> List[SPAMNoiseModel]:
    """
    Generate a sweep of SPAM noise models for robustness testing.

    Parameters
    ----------
    prep_rates : list of float, optional
        State preparation error rates to sweep.
    meas_rates : list of float, optional
        Measurement error rates to sweep.

    Returns
    -------
    list of SPAMNoiseModel
        List of noise models for benchmarking.
    """
    if prep_rates is None:
        prep_rates = [0.0, 0.001, 0.005, 0.01]
    if meas_rates is None:
        meas_rates = [0.0, 0.005, 0.01, 0.02]

    models = []
    for p in prep_rates:
        for m in meas_rates:
            models.append(SPAMNoiseModel(prep_error_rate=p, meas_error_rate=m))

    return models
