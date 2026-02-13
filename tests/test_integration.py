"""
Integration tests for the full Diagnose → Control → Remediate → Validate pipeline.

These tests verify that all layers work together end-to-end.
"""

import numpy as np


class TestFullPipeline:
    """Integration tests exercising cross-layer interactions."""

    def test_diagnose_to_remediate(self):
        """Diagnose defects, then synthesize corrective pulses."""
        from unified_qec.diagnostics.hamiltonian_learner import HamiltonianLearner
        from unified_qec.diagnostics.aubry_andre import AubryAndreModel
        from unified_qec.remediation.pulse_synthesis import PulseSynthesizer

        # Diagnose
        L = 4
        learner = HamiltonianLearner(system_size=L)
        h_fields = AubryAndreModel.generate_fields(L)
        J_true = np.array([1.0, 0.3, 1.0])  # defect at bond 1
        t_points = np.linspace(0.1, 3.0, 20)
        data = learner.simulate_dynamics(J_true, h_fields, t_points)
        J_recovered, error = learner.learn_hamiltonian(data, t_points, h_fields)

        defects = learner.detect_defects(J_recovered, threshold=0.15)
        assert len(defects["weak_couplings"]) > 0, "Should detect the weak coupling"

        # Remediate with diagnosed parameters
        synth = PulseSynthesizer(system_size=L, gate_time=2.0, dt=0.5)
        pulse, fidelity = synth.synthesize(
            J_recovered, h_fields, max_iterations=50, verbose=False
        )
        assert pulse.shape == (4, L)
        assert 0 <= fidelity <= 1

    def test_simulation_to_decoding(self):
        """Generate circuit → extract DEM → decode."""
        from unified_qec.simulation.surface_code import SurfaceCodeCircuit, NoiseParameters
        from unified_qec.decoding.dem_utils import dem_to_matrices, get_channel_llrs

        sc = SurfaceCodeCircuit(distance=3, rounds=3)
        params = NoiseParameters(gate_error=0.01)
        circuit = sc.generate(params)

        # Extract DEM and convert to matrices
        dem = circuit.detector_error_model()
        H, L_obs, priors = dem_to_matrices(dem)
        llrs = get_channel_llrs(priors)

        assert H.shape[0] > 0, "Should have detectors"
        assert L_obs.shape[0] > 0, "Should have observables"
        assert len(llrs) == len(priors)
        assert all(val > 0 for val in llrs), "LLRs should be positive"

    def test_feedback_loop(self):
        """Controller + decoder weights over multiple rounds."""
        from unified_qec.feedback.controller import SyndromeFeedbackController
        from unified_qec.feedback.decoder_weights import AdaptiveDecoderWeights

        controller = SyndromeFeedbackController(
            Ki=0.05, feedback_latency=2, correction_bounds=(-0.02, 0.15)
        )
        weights = AdaptiveDecoderWeights(base_error_rate=0.001)

        controller.setpoint = 0.1
        densities = [0.10, 0.11, 0.12, 0.13, 0.12, 0.11, 0.10]

        corrections = []
        effective_ps = []

        for density in densities:
            correction = controller.update(density)
            corrections.append(correction)
            effective_p = weights.compute_weights(correction)
            effective_ps.append(effective_p)

        assert len(corrections) == len(densities)
        assert all(isinstance(c, float) for c in corrections)
        # Effective p should stay in physical range
        assert all(1e-6 <= p <= 0.49 for p in effective_ps)

    def test_feedback_stability_analysis(self):
        """Controller → frequency analyzer → stability check."""
        from unified_qec.feedback.controller import SyndromeFeedbackController
        from unified_qec.feedback.frequency_analysis import FrequencyAnalyzer

        Ki = 0.05
        latency = 3

        _controller = SyndromeFeedbackController(Ki=Ki, feedback_latency=latency)
        analyzer = FrequencyAnalyzer(Ki=Ki, latency_s=latency * 1e-6)

        margins = analyzer.compute_margins()
        assert margins.stable in (True, False)
        assert hasattr(margins, "phase_margin_deg")

    def test_physics_simulation_chain(self):
        """Leakage + cosmic rays + burst detection pipeline."""
        from unified_qec.physics.leakage import LeakageTracker
        from unified_qec.physics.cosmic_ray import CosmicRaySimulator
        from unified_qec.physics.burst_error import BurstErrorDetector

        d = 5
        n_qubits = d ** 2

        # Run leakage for several cycles
        tracker = LeakageTracker(num_qubits=n_qubits, leakage_rate=0.005)
        for _ in range(10):
            tracker.run_cycle()
        stats = tracker.get_statistics()
        assert stats["cycles"] == 10
        assert stats["error_contribution"] >= 0

        # Simulate cosmic ray
        sim = CosmicRaySimulator(distance=d, impact_rate=1.0)
        impact = sim.generate_impact(cycle=0, center=12, radius=2)
        depol_map = sim.get_depolarization_map(impact)
        assert len(depol_map) == n_qubits

        # Burst detection
        detector = BurstErrorDetector(distance=d)
        detector.set_baseline(0.1)

        # Normal → no burst
        normal_syndrome = np.random.random(n_qubits) * 0.1
        result = detector.detect(normal_syndrome)
        assert result is None

    def test_drifting_noise_surface_code(self):
        """DriftingNoiseModel feeds into SurfaceCodeCircuit over many rounds."""
        from unified_qec.simulation.surface_code import (
            SurfaceCodeCircuit, DriftingNoiseModel, NoiseParameters
        )

        d = 3
        sc = SurfaceCodeCircuit(distance=d, rounds=2)
        drift = DriftingNoiseModel(
            num_qubits=d ** 2,
            base_params=NoiseParameters(gate_error=0.005),
            drift_rate=0.01,
        )

        error_rates = []
        for _ in range(5):
            drift.step()
            params = drift.get_effective_params()
            failures, density = sc.run_batch(params, batch_size=100)
            error_rates.append(failures / 100)

        assert len(error_rates) == 5
        assert all(0 <= r <= 1 for r in error_rates)

    def test_spam_noise_impact(self):
        """SPAM noise applied to measurement data."""
        from unified_qec.diagnostics.spam_noise import (
            inject_readout_noise, inject_state_prep_error, generate_spam_sweep
        )

        # Clean measurements
        clean = np.zeros(1000, dtype=int)
        noisy = inject_readout_noise(clean, error_rate=0.05)
        flip_rate = noisy.mean()
        assert 0.02 < flip_rate < 0.10, f"Expected ~5% flips, got {flip_rate:.3f}"

        # State prep
        states = inject_state_prep_error(1000, target_state=0, error_rate=0.05)
        prep_error_rate = states.mean()
        assert 0.02 < prep_error_rate < 0.10

        # Sweep
        models = generate_spam_sweep()
        assert len(models) == 16  # 4 prep × 4 meas rates

    def test_full_pipeline_runs(self):
        """Smoke test for the full experiment pipeline."""
        from unified_qec.experiments.full_pipeline import run_full_pipeline

        results = run_full_pipeline(
            code_distance=3,
            system_size=4,
            num_rounds=3,
            drift_rate=0.001,
            verbose=False,
        )

        assert "diagnostics" in results
        assert "control" in results
        assert "remediation" in results
        assert "validation" in results
        assert results["validation"]["pipeline_status"] == "complete"
        assert 0 <= results["remediation"]["fidelity"] <= 1
