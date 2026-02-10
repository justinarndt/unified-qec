"""
Unified Noise Models

Consolidates noise dataclasses from all source repositories into a
single module with consistent naming and comprehensive documentation.

Sources:
- CoherentNoiseModel: stim-cirq-qec NoiseModel (depolarizing + coherent)
- SPAMNoiseModel: stim-cirq-qec spam_noise.py
- StressCircuit generators: qec noise_models.py (drift, burst, leakage)
"""

import numpy as np
import stim
from typing import Optional, Tuple, List
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Noise dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CoherentNoiseModel:
    """Noise model supporting both Pauli and coherent channels.

    Used by `StimCirqBridge` for hybrid Stim+Cirq simulation.

    Attributes
    ----------
    depolarizing : float
        Pauli noise rate per gate.
    measurement : float
        Readout error probability.
    reset : float
        Reset error probability.
    coherent_overrotation : float
        Coherent Z-rotation angle (Cirq only).
    zz_crosstalk : float
        ZZ coupling strength (Cirq only).
    t1_us : float
        T1 relaxation time in microseconds.
    t2_us : float
        T2 dephasing time in microseconds.
    cycle_time_us : float
        QEC cycle time in microseconds.
    leakage_rate : float
        Per-gate probability of |1⟩ → |2⟩ transition.
    seepage_rate : float
        Per-gate probability of |2⟩ → |1⟩ return (LRU).
    """
    depolarizing: float = 0.001
    measurement: float = 0.01
    reset: float = 0.005
    coherent_overrotation: float = 0.0
    zz_crosstalk: float = 0.0
    t1_us: float = 100.0
    t2_us: float = 80.0
    cycle_time_us: float = 1.0
    leakage_rate: float = 0.001
    seepage_rate: float = 0.01


@dataclass
class SPAMNoiseModel:
    """SPAM noise model for readout and state prep errors.

    Attributes
    ----------
    readout_bias : float
        Systematic bias in readout (0-1).
    readout_variance : float
        Random noise in readout probability.
    prep_error : float
        State preparation error rate.
    asymmetric : bool
        If True, 0→1 and 1→0 errors have different rates.
    bias_01 : float
        P(measure 1 | true 0) if asymmetric.
    bias_10 : float
        P(measure 0 | true 1) if asymmetric.
    """
    readout_bias: float = 0.01
    readout_variance: float = 0.005
    prep_error: float = 0.005
    asymmetric: bool = False
    bias_01: float = 0.01
    bias_10: float = 0.01


# ---------------------------------------------------------------------------
# SPAM noise injection functions
# ---------------------------------------------------------------------------

def inject_readout_noise(
    measurements: np.ndarray,
    spam_model: SPAMNoiseModel
) -> np.ndarray:
    """
    Inject SPAM noise into measurement outcomes.

    Parameters
    ----------
    measurements : np.ndarray
        Clean measurement outcomes (0 or 1).
    spam_model : SPAMNoiseModel
        SPAM noise parameters.

    Returns
    -------
    np.ndarray
        Noisy measurement outcomes.
    """
    noisy = measurements.copy().astype(float)

    if spam_model.asymmetric:
        mask_0 = measurements == 0
        mask_1 = measurements == 1

        flip_01 = np.random.random(np.sum(mask_0)) < spam_model.bias_01
        flip_10 = np.random.random(np.sum(mask_1)) < spam_model.bias_10

        noisy[mask_0] = np.where(flip_01, 1, 0)
        noisy[mask_1] = np.where(flip_10, 0, 1)
    else:
        flip_prob = spam_model.readout_bias + \
                    np.random.normal(0, spam_model.readout_variance, measurements.shape)
        flip_prob = np.clip(flip_prob, 0, 1)

        flip_mask = np.random.random(measurements.shape) < flip_prob
        noisy = np.where(flip_mask, 1 - measurements, measurements)

    return noisy.astype(int)


def inject_state_prep_error(
    initial_state: np.ndarray,
    spam_model: SPAMNoiseModel
) -> np.ndarray:
    """
    Inject state preparation errors.

    Parameters
    ----------
    initial_state : np.ndarray
        Ideal initial state vector (density matrix diagonal).
    spam_model : SPAMNoiseModel
        SPAM noise parameters.

    Returns
    -------
    np.ndarray
        State with preparation errors applied.
    """
    error_mask = np.random.random(len(initial_state)) < spam_model.prep_error
    noisy_state = initial_state.copy()
    noisy_state[error_mask] = 0.5  # Mixed state
    return noisy_state / np.sum(noisy_state)


def compute_imbalance_with_spam(
    imbalance_clean: np.ndarray,
    spam_model: SPAMNoiseModel
) -> np.ndarray:
    """
    Apply SPAM noise to imbalance trace.

    The imbalance I(t) = ⟨N_odd - N_even⟩ is affected by SPAM:
    - Readout bias adds systematic offset
    - Prep error reduces initial imbalance
    """
    prep_factor = 1 - 2 * spam_model.prep_error
    readout_factor = 1 - 2 * spam_model.readout_bias

    imbalance_noisy = imbalance_clean * prep_factor * readout_factor

    noise = np.random.normal(0, spam_model.readout_variance, len(imbalance_clean))
    imbalance_noisy += noise

    return np.clip(imbalance_noisy, -1, 1)


def generate_spam_sweep(
    bias_levels: list = [0.0, 0.005, 0.01, 0.02, 0.05]
) -> list:
    """Generate a sweep of SPAM noise models for robustness testing."""
    models = []
    for bias in bias_levels:
        models.append(SPAMNoiseModel(
            readout_bias=bias,
            readout_variance=bias / 2,
            prep_error=bias / 2
        ))
    return models


# ---------------------------------------------------------------------------
# Stress-test circuit generators (from qec repo)
# ---------------------------------------------------------------------------

def generate_stress_circuit(
    d: int,
    base_p: float,
    drift_strength: float = 0.2,
    burst_prob: float = 0.0,
    rounds: int | None = None,
) -> stim.Circuit:
    """
    Generate a rotated surface code circuit with stress-test noise.

    Creates a circuit with time-varying noise to simulate realistic
    hardware conditions:
    - Sinusoidal drift in error rates (mimics T1 fluctuations)
    - Correlated burst errors (mimics cosmic ray events)

    Parameters
    ----------
    d : int
        Code distance.
    base_p : float
        Base physical error rate.
    drift_strength : float
        Amplitude of sinusoidal drift (0.3 = ±30%).
    burst_prob : float
        Probability of correlated burst injection.
    rounds : int, optional
        Number of syndrome rounds (default: 3*d).

    Returns
    -------
    stim.Circuit
        Stim circuit with injected noise.
    """
    if rounds is None:
        rounds = d * 3

    # Generate base circuit without noise
    circuit_raw = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=rounds,
        after_clifford_depolarization=0,
        before_round_data_depolarization=0,
        before_measure_flip_probability=0,
        after_reset_flip_probability=0,
    )

    circuit = circuit_raw.flattened()

    # Circuit surgery: inject time-varying noise
    new_circuit = stim.Circuit()
    period = rounds / 2.0
    current_round = 0

    for instruction in circuit:
        if instruction.name == "TICK":
            current_round += 1
            new_circuit.append("TICK")
            continue

        # Calculate drift factor: sinusoidal variation
        drift_factor = 1.0 + drift_strength * np.sin(2 * np.pi * current_round / period)
        p_now = base_p * drift_factor
        targets = instruction.targets_copy()

        # Inject noise based on gate type
        if instruction.name in ["R", "M", "MR"]:
            new_circuit.append(instruction.name, targets)
            new_circuit.append("X_ERROR", targets, p_now)
        elif instruction.name in ["CX", "CZ", "H", "S", "X", "Z", "Y"]:
            new_circuit.append(instruction.name, targets)
            if instruction.name in ["CX", "CZ"]:
                new_circuit.append("DEPOLARIZE2", targets, p_now)
            else:
                new_circuit.append("DEPOLARIZE1", targets, p_now)
        else:
            new_circuit.append(instruction)

    # Burst injection: correlated errors on adjacent qubits
    if burst_prob > 0:
        middle_qubits = list(range(d * d // 2, d * d // 2 + d))
        targets = [stim.target_z(q) for q in middle_qubits]
        burst_circuit = stim.Circuit()
        burst_circuit.append("CORRELATED_ERROR", targets, burst_prob)
        new_circuit = burst_circuit + new_circuit

    return new_circuit


def generate_standard_circuit(
    d: int,
    p: float,
    rounds: int | None = None,
) -> stim.Circuit:
    """
    Generate a standard rotated surface code circuit with uniform noise.

    Parameters
    ----------
    d : int
        Code distance.
    p : float
        Physical error rate (uniform depolarizing).
    rounds : int, optional
        Number of syndrome rounds (default: 3*d).
    """
    if rounds is None:
        rounds = d * 3

    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def generate_leakage_circuit(
    d: int,
    base_p: float,
    leakage_rate: float = 0.001,
    seepage_rate: float = 0.01,
    rounds: int | None = None,
) -> stim.Circuit:
    """
    Generate a circuit with heralded leakage noise model.

    Stim doesn't natively support leakage states, so we approximate
    leakage as persistent bit-flip errors that probabilistically clear.

    Parameters
    ----------
    d : int
        Code distance.
    base_p : float
        Base physical error rate.
    leakage_rate : float
        Probability of leaking to |2⟩ state per gate.
    seepage_rate : float
        Probability of returning from |2⟩ to computational basis.
    rounds : int, optional
        Number of syndrome rounds (default: 3*d).
    """
    if rounds is None:
        rounds = d * 3

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=rounds,
        after_clifford_depolarization=base_p,
        before_round_data_depolarization=base_p,
        before_measure_flip_probability=base_p,
        after_reset_flip_probability=base_p,
    )

    circuit = circuit.flattened()

    new_circuit = stim.Circuit()
    data_qubits = list(range(d * d))

    for instruction in circuit:
        new_circuit.append(instruction)

        # After each TICK, inject leakage effects
        if instruction.name == "TICK":
            for q in data_qubits:
                if np.random.random() < leakage_rate:
                    new_circuit.append("X_ERROR", [q], 0.5)

    return new_circuit
