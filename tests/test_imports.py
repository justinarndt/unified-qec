"""Test all layer imports and basic functionality."""

import numpy as np
import pytest


def test_root_import():
    import unified_qec
    assert unified_qec.__version__ == "0.1.0"


def test_simulation_imports():
    from unified_qec.simulation.surface_code import SurfaceCodeCircuit, DriftingNoiseModel
    from unified_qec.simulation.noise_models import generate_stress_circuit, generate_standard_circuit


def test_decoding_imports():
    from unified_qec.decoding.dem_utils import dem_to_matrices, get_channel_llrs


def test_diagnostics_imports():
    from unified_qec.diagnostics.hamiltonian_learner import HamiltonianLearner
    from unified_qec.diagnostics.aubry_andre import AubryAndreModel
    from unified_qec.diagnostics.spam_noise import SPAMNoiseModel


def test_feedback_imports():
    from unified_qec.feedback.controller import SyndromeFeedbackController
    from unified_qec.feedback.decoder_weights import AdaptiveDecoderWeights
    from unified_qec.feedback.frequency_analysis import FrequencyAnalyzer, StabilityMargins


def test_physics_imports():
    from unified_qec.physics.leakage import LeakageTracker
    from unified_qec.physics.cosmic_ray import CosmicRaySimulator
    from unified_qec.physics.burst_error import BurstErrorDetector


def test_remediation_imports():
    from unified_qec.remediation.pulse_synthesis import PulseSynthesizer


def test_hamiltonian_learner():
    from unified_qec.diagnostics.hamiltonian_learner import HamiltonianLearner
    from unified_qec.diagnostics.aubry_andre import AubryAndreModel

    learner = HamiltonianLearner(system_size=4)
    h = AubryAndreModel.generate_fields(4)
    J = np.ones(3)
    t = np.linspace(0.1, 2.0, 10)
    trace = learner.simulate_dynamics(J, h, t)
    assert len(trace) == 10
    assert trace.min() >= -1.0
    assert trace.max() <= 1.0


def test_controller():
    from unified_qec.feedback.controller import SyndromeFeedbackController

    ctrl = SyndromeFeedbackController(Ki=0.05, feedback_latency=3)
    ctrl.setpoint = 0.1
    correction = ctrl.update(0.12)
    assert isinstance(correction, float)


def test_decoder_weights():
    from unified_qec.feedback.decoder_weights import AdaptiveDecoderWeights

    weights = AdaptiveDecoderWeights(base_error_rate=0.001)
    eff = weights.compute_weights(0.005)
    assert 1e-6 <= eff <= 0.49
    llw = weights.log_likelihood_weight(eff)
    assert llw > 0


def test_leakage_tracker():
    from unified_qec.physics.leakage import LeakageTracker

    tracker = LeakageTracker(num_qubits=25)
    leaks, seepage = tracker.run_cycle()
    stats = tracker.get_statistics()
    assert stats["num_qubits"] == 25
    assert stats["cycles"] == 1


def test_surface_code_circuit():
    from unified_qec.simulation.surface_code import SurfaceCodeCircuit, NoiseParameters

    sc = SurfaceCodeCircuit(distance=3, rounds=2)
    circuit = sc.generate(NoiseParameters(gate_error=0.01))
    assert circuit.num_detectors > 0


def test_frequency_analyzer():
    from unified_qec.feedback.frequency_analysis import FrequencyAnalyzer

    analyzer = FrequencyAnalyzer(Ki=0.05, latency_s=500e-9)
    margins = analyzer.compute_margins()
    assert hasattr(margins, "phase_margin_deg")
    assert hasattr(margins, "gain_margin_db")
    assert hasattr(margins, "stable")


def test_cosmic_ray():
    from unified_qec.physics.cosmic_ray import CosmicRaySimulator

    sim = CosmicRaySimulator(distance=5)
    impact = sim.generate_impact(cycle=0, center=12, radius=2)
    assert impact.center_qubit == 12
    assert len(impact.affected_qubits) > 0

    depol_map = sim.get_depolarization_map(impact)
    assert len(depol_map) == 25


def test_burst_detector():
    from unified_qec.physics.burst_error import BurstErrorDetector

    detector = BurstErrorDetector(distance=5)
    detector.set_baseline(0.1)

    # Normal syndrome — no burst
    normal = np.random.random(25) * 0.1
    result = detector.detect(normal)
    assert result is None


def test_spam_noise():
    from unified_qec.diagnostics.spam_noise import (
        inject_readout_noise, inject_state_prep_error, generate_spam_sweep
    )

    meas = np.zeros(100)
    noisy = inject_readout_noise(meas, error_rate=0.1)
    assert noisy.sum() > 0

    states = inject_state_prep_error(50, target_state=0, error_rate=0.1)
    assert states.sum() > 0

    sweep = generate_spam_sweep()
    assert len(sweep) > 0


def test_dem_utils():
    import stim
    from unified_qec.decoding.dem_utils import dem_to_matrices, get_channel_llrs

    # Create a simple circuit with detectors
    circuit = stim.Circuit("""
        X_ERROR(0.1) 0
        M 0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    dem = circuit.detector_error_model()
    H, L, priors = dem_to_matrices(dem)
    assert H.shape[0] == 1  # 1 detector
    assert L.shape[0] == 1  # 1 observable

    llrs = get_channel_llrs(priors)
    assert len(llrs) == len(priors)


@pytest.mark.slow
def test_pulse_synthesizer():
    from unified_qec.remediation.pulse_synthesis import PulseSynthesizer

    synth = PulseSynthesizer(system_size=4, gate_time=2.0, dt=0.5)
    J = np.ones(3)
    h = np.zeros(4)
    pulse, fidelity = synth.synthesize(J, h, max_iterations=50, verbose=False)
    assert pulse.shape == (4, 4)  # num_steps x system_size
    assert 0 <= fidelity <= 1


def test_defect_detection():
    from unified_qec.diagnostics.hamiltonian_learner import HamiltonianLearner

    learner = HamiltonianLearner(system_size=4)
    J_recovered = np.array([1.0, 0.5, 1.1])
    defects = learner.detect_defects(J_recovered, J_nominal=1.0, threshold=0.1)
    assert 1 in defects["weak_couplings"]
    assert 2 not in defects["weak_couplings"]
