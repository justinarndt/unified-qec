"""
Full Pipeline Experiment: Diagnose → Control → Remediate → Validate

End-to-end integration demonstrating all layers of unified-qec working
together on a simulated hardware drift scenario.

Pipeline Steps:
1. DIAGNOSE: Run Hamiltonian learning to detect coupling defects
2. CONTROL: Deploy syndrome feedback controller to track drift
3. REMEDIATE: Synthesize corrective pulses for detected defects
4. VALIDATE: Verify improvement via error rate comparison
"""

import numpy as np
from typing import Dict

from unified_qec.diagnostics.hamiltonian_learner import HamiltonianLearner
from unified_qec.diagnostics.aubry_andre import AubryAndreModel
from unified_qec.feedback.controller import SyndromeFeedbackController
from unified_qec.feedback.decoder_weights import AdaptiveDecoderWeights
from unified_qec.remediation.pulse_synthesis import PulseSynthesizer
from unified_qec.physics.leakage import LeakageTracker


def run_full_pipeline(
    code_distance: int = 5,
    system_size: int = 6,
    num_rounds: int = 5,
    drift_rate: float = 0.001,
    base_error_rate: float = 0.001,
    defect_coupling: float = 0.3,
    verbose: bool = True,
) -> Dict:
    """
    Run the complete Diagnose → Control → Remediate → Validate pipeline.

    Parameters
    ----------
    code_distance : int
        Surface code distance for simulation.
    system_size : int
        Spin chain size for Hamiltonian learning.
    num_rounds : int
        Number of QEC rounds.
    drift_rate : float
        Noise drift rate (per-round increase).
    base_error_rate : float
        Baseline physical error rate.
    defect_coupling : float
        Strength of injected coupling defect.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Pipeline results including diagnostics, control performance,
        remediation fidelity, and validation metrics.
    """
    results = {}

    # ========================================
    # Step 1: DIAGNOSE — Hamiltonian Learning
    # ========================================
    if verbose:
        print("=" * 60)
        print("STEP 1: DIAGNOSE — Hamiltonian Learning")
        print("=" * 60)

    learner = HamiltonianLearner(system_size=system_size)

    # Generate disorder fields
    h_fields = AubryAndreModel.generate_fields(system_size)

    # True couplings with one defect at position 2
    J_true = np.ones(system_size - 1)
    J_true[2] = defect_coupling  # Defect: weak coupling

    # Simulate experimental data
    t_points = np.linspace(0.1, 5.0, 30)
    experimental_trace = learner.simulate_dynamics(J_true, h_fields, t_points)

    # Add measurement noise
    experimental_trace += np.random.normal(0, 0.01, len(experimental_trace))

    # Learn Hamiltonian
    J_recovered, fit_error = learner.learn_hamiltonian(
        experimental_trace, t_points, h_fields
    )

    # Detect defects
    defects = learner.detect_defects(J_recovered)

    results["diagnostics"] = {
        "J_true": J_true.tolist(),
        "J_recovered": J_recovered.tolist(),
        "fit_error": float(fit_error),
        "defects": defects,
    }

    if verbose:
        print(f"  True couplings:      {J_true}")
        print(f"  Recovered couplings: {np.round(J_recovered, 3)}")
        print(f"  Fit error:           {fit_error:.6f}")
        print(f"  Weak couplings:      {defects['weak_couplings']}")
        print(f"  Strong couplings:    {defects['strong_couplings']}")

    # ========================================
    # Step 2: CONTROL — Feedback Controller
    # ========================================
    if verbose:
        print()
        print("=" * 60)
        print("STEP 2: CONTROL — Syndrome Feedback Controller")
        print("=" * 60)

    controller = SyndromeFeedbackController(
        Ki=0.05,
        feedback_latency=3,
        correction_bounds=(-0.02, 0.15),
        latency_ns=500.0,
    )

    weights = AdaptiveDecoderWeights(base_error_rate=base_error_rate)

    # Simulate drift scenario
    corrections = []
    densities = []

    for round_idx in range(num_rounds):
        # Simulate drifting noise
        current_rate = base_error_rate + drift_rate * round_idx
        density = current_rate + np.random.normal(0, 0.0005)
        densities.append(density)

        if round_idx == 0:
            controller.setpoint = density
        else:
            correction = controller.update(density)
            corrections.append(correction)

            # Compute adaptive decoder weight
            _effective_p = weights.compute_weights(correction)

    decay_penalty = controller.compute_latency_decay_penalty()

    results["control"] = {
        "corrections": corrections,
        "densities": densities,
        "decay_penalty": float(decay_penalty),
        "history": {k: v for k, v in controller.history.items()},
    }

    if verbose:
        print(f"  Rounds simulated:  {num_rounds}")
        print(f"  Final correction:  {corrections[-1] if corrections else 0:.6f}")
        print(f"  Decay penalty:     {decay_penalty:.6f}")

    # ========================================
    # Step 3: REMEDIATE — Pulse Synthesis
    # ========================================
    if verbose:
        print()
        print("=" * 60)
        print("STEP 3: REMEDIATE — Optimal Control Pulse Synthesis")
        print("=" * 60)

    synthesizer = PulseSynthesizer(
        system_size=system_size,
        gate_time=4.0,
        dt=0.4,
    )

    optimal_pulse, fidelity = synthesizer.synthesize(
        J_diagnosed=J_recovered,
        h_fields=h_fields,
        max_iterations=200,
        verbose=verbose,
    )

    results["remediation"] = {
        "fidelity": float(fidelity),
        "pulse_shape": optimal_pulse.shape,
        "pulse_norm": float(np.linalg.norm(optimal_pulse)),
    }

    if verbose:
        print(f"  Achieved fidelity: {fidelity * 100:.2f}%")

    # ========================================
    # Step 4: VALIDATE — Leakage + Error Rate
    # ========================================
    if verbose:
        print()
        print("=" * 60)
        print("STEP 4: VALIDATE — Leakage & Error Assessment")
        print("=" * 60)

    tracker = LeakageTracker(
        num_qubits=code_distance ** 2,
        leakage_rate=0.0005,
        seepage_rate=0.01,
    )

    for _ in range(20):
        tracker.run_cycle()

    leak_stats = tracker.get_statistics()

    results["validation"] = {
        "leakage_stats": leak_stats,
        "pipeline_status": "complete",
    }

    if verbose:
        print(f"  Leaked qubits:       {leak_stats['current_leaked']}/{leak_stats['num_qubits']}")
        print(f"  Total leak events:   {leak_stats['total_leakage_events']}")
        print(f"  Error contribution:  {leak_stats['error_contribution']:.6f}")
        print()
        print("=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)

    return results


if __name__ == "__main__":
    results = run_full_pipeline(verbose=True)
