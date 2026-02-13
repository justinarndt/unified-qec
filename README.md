# unified-qec

A research-grade quantum error correction toolkit that consolidates simulation, decoding, diagnostics, feedback control, hardware physics modeling, pulse remediation, differentiable calibration, and FPGA control logic into one cohesive Python + SystemVerilog package.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/justinarndt/unified-qec/actions/workflows/ci.yml/badge.svg)](https://github.com/justinarndt/unified-qec/actions)

## Why unified-qec?

Most QEC research treats each stage of the error correction stack — circuit generation, decoding, noise characterization, feedback control — as an isolated concern. **unified-qec** connects them into a single closed-loop framework where:

- Diagnosed hardware defects inform corrective pulse synthesis
- Decoded syndrome density drives real-time feedback control
- Adaptive decoder weights track non-stationary noise drift
- Leakage, cosmic rays, and burst errors feed back into the simulation loop
- FPGA Pauli frame trackers validate sub-microsecond control latency

The core pipeline: **Diagnose → Control → Remediate → Validate**

## Architecture

```mermaid
graph TD
    subgraph "unified-qec"
        SIM["simulation<br/>Circuit generation, noise models,<br/>Stim + Cirq bridge"]
        DEC["decoding<br/>BP+OSD (ASR-MP), Union-Find,<br/>DEM utilities"]
        DIAG["diagnostics<br/>Hamiltonian learning, GST,<br/>Aubry-André disorder"]
        FB["feedback<br/>Syndrome controller,<br/>adaptive decoder weights"]
        PHYS["physics<br/>Leakage, cosmic rays,<br/>burst errors"]
        REM["remediation<br/>Optimal control<br/>pulse synthesis"]
        CAL["calibration<br/>HoloG JAX density-matrix<br/>simulator"]
        EXP["experiments<br/>End-to-end pipelines"]
    end

    subgraph "rtl/"
        RTL["FPGA Pauli Frame Trackers<br/>LUTRAM (zero-stall) vs BRAM (baseline)"]
    end

    DIAG -->|diagnosed Hamiltonian| REM
    DIAG -->|defect map| FB
    SIM -->|circuits| DEC
    FB -->|weight updates| DEC
    CAL -->|calibrated pulses| SIM
    PHYS -->|noise injection| SIM
    EXP -->|orchestrates| SIM & DEC & DIAG & FB & REM

    style SIM fill:#1a73e8,color:#fff
    style DEC fill:#e8710a,color:#fff
    style DIAG fill:#0d904f,color:#fff
    style FB fill:#9334e6,color:#fff
    style PHYS fill:#d93025,color:#fff
    style REM fill:#185abc,color:#fff
    style CAL fill:#b31412,color:#fff
    style EXP fill:#137333,color:#fff
    style RTL fill:#455a64,color:#fff
```

---

## Module Reference

### `simulation/` — Circuit Generation & Noise Models

Stim-based surface code circuit generation with physically realistic noise channels.

| Class / Function | Description |
|---|---|
| `SurfaceCodeCircuit` | Rotated surface code generator with configurable distance, rounds, and noise. Wraps `stim.Circuit.generated()` with batch decoding via PyMatching. |
| `DriftingNoiseModel` | Non-stationary noise via Ornstein-Uhlenbeck process: `dp = θ(μ − p)dt + σ dW`. Models per-qubit local drift and global T1/T2 fluctuations. |
| `NoiseParameters` | Dataclass for gate/measurement/reset error rates with `scale()` support. |
| `CoherentNoiseModel` | Extended noise model with coherent over-rotation, ZZ crosstalk, T1/T2 decay, and leakage/seepage rates. Used by the Stim↔Cirq bridge. |
| `SPAMNoiseModel` | State preparation and measurement noise: readout bias, prep error, asymmetric bit-flip channels. |
| `StimCirqBridge` | Converts between Stim and Cirq representations for hybrid simulation — Stim for speed, Cirq for coherent noise fidelity. Includes DEM extraction, adaptive weight updates, and coherent noise injection. |
| `CoherentNoiseInjector` | Injects gate over-rotations (`Rz(ε)`), ZZ crosstalk, and amplitude damping into Cirq circuits. |
| `generate_stress_circuit()` | Stress-test circuit with sinusoidal drift and burst noise injection. |
| `generate_leakage_circuit()` | Circuit with heralded leakage noise (persistent bit-flip approximation). |
| `generate_standard_circuit()` | Uniform depolarizing noise for baseline benchmarks. |

---

### `decoding/` — Syndrome Decoding

Two decoder backends with full [sinter](https://github.com/quantumlib/Stim/tree/main/glue/sinter) integration for Monte Carlo threshold estimation.

| Class / Function | Description |
|---|---|
| `ASRMPDecoder` | **Primary decoder.** Belief Propagation + Ordered Statistics Decoding (BP+OSD) via the `ldpc` library. Configurable BP method (`product_sum` / `min_sum`), OSD order, and max iterations. Converts DEM→sparse matrices automatically. |
| `TesseractBPOSD` | Sinter-compatible decoder factory implementing the `sinter.Decoder` interface for parallelized Monte Carlo sampling. |
| `TesseractCompiledDecoder` | Compiled decoder implementing `sinter.CompiledDecoder` with bit-packed shot decoding. |
| `UnionFindDecoder` | Baseline Union-Find / local clustering decoder with O(n·α(n)) per-shot complexity via `fusion-blossom`. Sinter-integrated. |
| `dem_to_matrices()` | Converts `stim.DetectorErrorModel` → sparse parity check matrix **H**, logical observable matrix **L**, and prior vector **p**. |
| `get_channel_llrs()` | Computes log-likelihood ratios `log((1−p)/p)` from prior error probabilities for message-passing initialization. |

---

### `diagnostics/` — Hardware Characterization

Inverse-problem solvers that reconstruct hardware parameters from experimental observables.

| Class / Function | Description |
|---|---|
| `HamiltonianLearner` | Reconstructs coupling parameters of a spin-chain Hamiltonian from time-domain imbalance traces via gradient-based optimization (`L-BFGS-B`). Detects defective couplings by threshold analysis of recovered J values. Includes GST cross-validation. |
| `AubryAndreModel` | Generates quasi-periodic disorder fields `h_i = Δ cos(2πβi)` (β = golden ratio) for Many-Body Localization. Default Δ = 6.0 places the system deep in the MBL phase (critical Δ/J ≈ 3.5). |
| `GSTBenchmark` | Gate Set Tomography via pyGSTi — generates circuits, simulates data, runs GST protocol. Provides per-gate process fidelities as a gold-standard reference. |
| `SPAMNoiseModel` | SPAM noise injection and sweep generation for robustness testing. |

---

### `feedback/` — Real-Time Control

Closed-loop control that tracks hardware drift and adjusts decoder weights in real time.

| Class / Function | Description |
|---|---|
| `SyndromeFeedbackController` | Integral controller tracking syndrome density drift. Estimates `δ(t) = ρ(t) − ρ_ss` and feeds corrections to the decoder. Models realistic feedback latency via a bounded correction queue and T1/T2 decay penalty during the processing window. |
| `AdaptiveDecoderWeights` | Converts drift estimates to effective error rates for MWPM edge weight recalculation: `p_eff = p_base + δ`. Includes LLR conversion `w = log((1−p)/p)`. |

> **Hardware tie-in**: The `latency_ns` parameter directly corresponds to the FPGA update latency measured in the RTL benchmark. The LUTRAM tracker achieves <10 ns per update — fast enough that the feedback decay penalty is negligible. See `rtl/sim/benchmark_harness.sv`.

---

### `physics/` — Hardware Noise Modeling

Beyond-Pauli noise sources that create the "reality gap" between simulation and hardware.

| Class / Function | Description |
|---|---|
| `LeakageTracker` | Tracks per-qubit leakage to the |2⟩ (f) state in transmon hardware. Models gate-induced leakage, seepage (LRU recovery), and estimates the additional logical error rate from leaked qubits. Maintains per-cycle history for analysis. |
| `CosmicRaySimulator` | Simulates high-energy cosmic ray impacts as localized depolarization bursts with exponential falloff from the impact center. Configurable impact rate, radius, and maximum depolarization. |
| `BurstErrorDetector` | Real-time detection of burst errors via syndrome density spike monitoring and spatial clustering. When a burst is detected, recommends expanding the Cirq simulation region for proper correlated-error modeling. |

---

### `remediation/` — Corrective Pulse Synthesis

Physics-aware optimal control that navigates around diagnosed defects.

| Class / Function | Description |
|---|---|
| `PulseSynthesizer` | Given a diagnosed Hamiltonian `H_drift + H_defect`, synthesizes time-dependent Z-control fields `u_i(t)` that maximize state transfer fidelity: `F = |⟨ψ_target|U(T)|ψ_init⟩|²`. Uses L-BFGS-B with smoothness and power regularization. Sparse operator caching for efficiency on 2^L-dimensional Hilbert spaces. |

---

### `calibration/` — Differentiable Calibration (HoloG)

JAX-based density-matrix simulator for gradient-descent calibration of plaquette control pulses.

| Class / Function | Description |
|---|---|
| `calibrate_plaquette()` | Optimizes Rx rotation angles via JAX autodiff to minimize syndrome error probability. Models T1 decay, ZZ crosstalk, and measurement noise in the cost function. |
| `simulate_plaquette()` | Density-matrix simulation of a plaquette stabilizer with configurable decoherence and crosstalk. Forward model for the calibration loop. |
| `run_benchmark()` | End-to-end benchmark comparing initial (π-pulse) vs. optimized parameters. |

---

### `experiments/` — End-to-End Pipelines

Orchestration layer that connects all modules into complete workflows.

| Function | Description |
|---|---|
| `run_full_pipeline()` | Executes the complete **Diagnose → Control → Remediate → Validate** workflow: (1) Hamiltonian learning detects coupling defects, (2) syndrome feedback controller tracks drift, (3) pulse synthesizer generates corrective fields, (4) leakage tracker validates residual error. |

---

### `rtl/` — FPGA Pauli Frame Tracking (SystemVerilog)

Hardware implementation of the real-time Pauli frame update logic, benchmarking two memory strategies on Xilinx UltraScale+.

| Module | Memory | Read Latency | Throughput |
|---|---|---|---|
| `apex_lutram_tracker` | Distributed LUTRAM | 0 cycles (async) | 100% — no stalls |
| `baseline_bram_tracker` | Block RAM | 1 cycle (sync) | ~50% — pipeline stalls |
| `benchmark_harness` | — | — | Rapid-fire + RAW latency testbench |

Both modules use `{Z, X}` 2-bit Pauli frame encoding, proper `rst_n` reset, `$clog2` address sizing, and explicit `(* ram_style *)` synthesis attributes. See [`rtl/README.md`](rtl/README.md) for simulation instructions (Vivado + Icarus Verilog).

---

## Installation

```bash
# Core (Stim + PyMatching)
pip install -e .

# With specific extras
pip install -e ".[jax]"       # HoloG differentiable calibration
pip install -e ".[cirq]"      # Stim-Cirq coherent noise bridge
pip install -e ".[bposd]"     # BP+OSD decoder (ldpc + sinter)
pip install -e ".[gst]"       # Gate Set Tomography (pyGSTi)
pip install -e ".[uf]"        # Union-Find decoder (fusion-blossom)

# Everything
pip install -e ".[all]"

# Development (pytest + ruff + mypy)
pip install -e ".[dev]"
```

## Quick Start

### Diagnose hardware defects

```python
from unified_qec.diagnostics import HamiltonianLearner, AubryAndreModel
import numpy as np

learner = HamiltonianLearner(system_size=6)
h_fields = AubryAndreModel.generate_fields(6)

# Simulate experimental data from defective hardware
J_true = np.array([1.0, 1.0, 0.3, 1.0, 1.0])  # defect at bond 2
t_points = np.linspace(0.1, 5.0, 30)
data = learner.simulate_dynamics(J_true, h_fields, t_points)

# Recover Hamiltonian and detect defects
J_recovered, error = learner.learn_hamiltonian(data, t_points, h_fields)
defects = learner.detect_defects(J_recovered)
print(defects["weak_couplings"])  # → [2]
```

### Deploy feedback control

```python
from unified_qec.feedback import SyndromeFeedbackController, AdaptiveDecoderWeights

controller = SyndromeFeedbackController(Ki=0.05, feedback_latency=3)
weights = AdaptiveDecoderWeights(base_error_rate=0.001)

controller.setpoint = 0.1
correction = controller.update(measured_density=0.12)
effective_p = weights.compute_weights(correction)
```

### Synthesize corrective pulses

```python
from unified_qec.remediation import PulseSynthesizer

synth = PulseSynthesizer(system_size=6, gate_time=8.0)
pulse, fidelity = synth.synthesize(J_recovered, h_fields)
print(f"Recovered fidelity: {fidelity*100:.1f}%")
```

### Simulate and decode with drifting noise

```python
from unified_qec.simulation import SurfaceCodeCircuit, DriftingNoiseModel

circuit = SurfaceCodeCircuit(distance=5, rounds=5)
drift = DriftingNoiseModel(num_qubits=25, drift_rate=0.005)

# Step the drift and generate circuits with current noise
drift.step()
params = drift.get_effective_params()
failures, density = circuit.run_batch(params, batch_size=1024)
```

### Track leakage through QEC cycles

```python
from unified_qec.physics import LeakageTracker

tracker = LeakageTracker(num_qubits=49, leakage_rate=0.001, seepage_rate=0.01)
for cycle in range(100):
    new_leaks, seepage = tracker.run_cycle()
stats = tracker.get_statistics()
print(f"Leaked: {stats['current_leaked']}/49, error contribution: {stats['error_contribution']:.4f}")
```

### Differentiable calibration (requires JAX)

```python
from unified_qec.calibration.calibrate import calibrate_plaquette

optimal_params, error_history = calibrate_plaquette(
    n_data=6, num_steps=200, t1_us=30.0, zz_strength=0.02
)
```

### Run the full pipeline

```python
from unified_qec.experiments.full_pipeline import run_full_pipeline

results = run_full_pipeline(code_distance=5, verbose=True)
```

## Package Layout

```
src/unified_qec/
├── simulation/        Surface code circuits, Stim-Cirq bridge, noise models
│   ├── surface_code.py          SurfaceCodeCircuit, DriftingNoiseModel
│   ├── noise_models.py          CoherentNoiseModel, SPAMNoiseModel, stress circuits
│   └── stim_cirq_bridge.py      StimCirqBridge, CoherentNoiseInjector
├── decoding/          BP+OSD (ASR-MP), Union-Find, DEM utilities
│   ├── asr_mp.py                ASRMPDecoder, TesseractBPOSD (sinter)
│   ├── union_find.py            UnionFindDecoder (fusion-blossom)
│   └── dem_utils.py             DEM→matrix conversion, LLR computation
├── diagnostics/       Hamiltonian learning, Aubry-André, GST, SPAM
│   ├── hamiltonian_learner.py   HamiltonianLearner (inverse problem solver)
│   ├── aubry_andre.py           AubryAndreModel (MBL disorder generator)
│   ├── gst_benchmark.py         GSTBenchmark (pyGSTi wrapper)
│   └── spam_noise.py            SPAMNoiseModel injection utilities
├── feedback/          Syndrome controller, adaptive weights
│   ├── controller.py            SyndromeFeedbackController (integral + decay)
│   └── decoder_weights.py       AdaptiveDecoderWeights (MWPM LLR)
├── physics/           Leakage, cosmic rays, burst errors
│   ├── leakage.py               LeakageTracker (|2⟩ state tracking)
│   ├── cosmic_ray.py            CosmicRaySimulator (localized depolarization)
│   └── burst_error.py           BurstErrorDetector (spatial clustering)
├── remediation/       Optimal control pulse synthesis
│   └── pulse_synthesis.py       PulseSynthesizer (L-BFGS-B + regularization)
├── calibration/       HoloG JAX density-matrix simulator
│   ├── calibrate.py             calibrate_plaquette() (JAX autodiff)
│   ├── simulator.py             simulate_plaquette() (density matrix)
│   ├── config.py                Physical constants, Pauli operators
│   ├── physics.py               Crosstalk and decoherence channels
│   └── circuit.py               Gate embedding utilities
└── experiments/       End-to-end pipeline demonstrations
    └── full_pipeline.py         Diagnose → Control → Remediate → Validate

rtl/
├── src/
│   ├── apex_lutram_tracker.sv     Zero-stall LUTRAM Pauli frame tracker
│   └── baseline_bram_tracker.sv   BRAM baseline with pipeline stalls
└── sim/
    └── benchmark_harness.sv       Rapid-fire + RAW latency testbench
```

## Optional Dependencies

| Extra | Enables | Library |
|---|---|---|
| `[cirq]` | Coherent noise simulation | cirq-core, stimcirq |
| `[jax]` | Differentiable calibration | jax, jaxlib |
| `[gst]` | Gate Set Tomography | pyGSTi |
| `[bposd]` | BP+OSD decoder | ldpc, sinter |
| `[uf]` | Union-Find decoder | fusion-blossom, sinter |
| `[dev]` | Testing & linting | pytest, ruff, mypy |

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v

# Lint check
ruff check src/ tests/
```

## License

MIT
