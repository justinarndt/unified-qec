"""
DEM Utility Functions

Converts Stim Detector Error Models (DEM) to sparse matrices for use
with algebraic decoders (BP, OSD, etc.) and computes log-likelihood
ratios from prior error probabilities.

Migrated from: qec/src/asr_mp/dem_utils.py
"""

import numpy as np
import stim
from typing import Tuple
from scipy.sparse import csc_matrix


def dem_to_matrices(
    dem: stim.DetectorErrorModel,
) -> Tuple[csc_matrix, csc_matrix, np.ndarray]:
    """
    Convert a Stim detector error model to sparse matrices.

    Extracts the parity check matrix H, logical observable matrix L,
    and prior error probabilities from a DEM.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        Detector error model from ``stim.Circuit.detector_error_model()``.

    Returns
    -------
    H : csc_matrix, shape (num_detectors, num_errors)
        Parity check matrix mapping errors to detector triggers.
    L : csc_matrix, shape (num_observables, num_errors)
        Logical observable matrix mapping errors to logical flips.
    priors : ndarray, shape (num_errors,)
        Prior error probability for each error mechanism.
    """
    # Collect error mechanisms
    det_indices_list = []
    obs_indices_list = []
    priors_list = []

    num_detectors = dem.num_detectors
    num_observables = dem.num_observables

    for instruction in dem.flattened():
        if instruction.type == "error":
            prob = instruction.args_copy()[0]
            dets = []
            obs = []
            for target in instruction.targets_copy():
                if target.is_relative_detector_id():
                    dets.append(target.val)
                elif target.is_logical_observable_id():
                    obs.append(target.val)

            det_indices_list.append(dets)
            obs_indices_list.append(obs)
            priors_list.append(prob)

    num_errors = len(priors_list)
    priors = np.array(priors_list, dtype=np.float64)

    # Build H matrix (detectors × errors)
    h_rows = []
    h_cols = []
    for col, dets in enumerate(det_indices_list):
        for d in dets:
            h_rows.append(d)
            h_cols.append(col)

    H = csc_matrix(
        (np.ones(len(h_rows), dtype=np.uint8), (h_rows, h_cols)),
        shape=(num_detectors, num_errors),
    )

    # Build L matrix (observables × errors)
    l_rows = []
    l_cols = []
    for col, obs in enumerate(obs_indices_list):
        for o in obs:
            l_rows.append(o)
            l_cols.append(col)

    L = csc_matrix(
        (np.ones(len(l_rows), dtype=np.uint8), (l_rows, l_cols)),
        shape=(num_observables, num_errors),
    )

    return H, L, priors


def get_channel_llrs(priors: np.ndarray) -> np.ndarray:
    """
    Compute log-likelihood ratios from prior error probabilities.

    Parameters
    ----------
    priors : ndarray, shape (num_errors,)
        Prior error probabilities.

    Returns
    -------
    llrs : ndarray, shape (num_errors,)
        Log-likelihood ratios: log((1-p)/p).
    """
    p = np.clip(priors, 1e-15, 1 - 1e-15)
    return np.log((1 - p) / p)
