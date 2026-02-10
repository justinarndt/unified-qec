# Scope & Limitations

This document provides an honest assessment of what `unified-qec` does and does not do. It is intended for reviewers evaluating the project's technical depth.

## What This Is

- **A simulation framework.** All components operate on synthetic data generated via Stim. No real quantum hardware was used.
- **A digital twin concept.** The Diagnose → Control → Remediate → Validate pipeline demonstrates how real-time adaptive QEC *could* work, using physically motivated models.
- **A decoder integration layer.** The ASR-MP and Union-Find decoders wrap existing libraries (`ldpc`, `fusion-blossom`). The contribution is the adaptive weight interface, not the decoding algorithms themselves.

## What This Is Not

- **Not a hardware-validated system.** No calibration data from real IBM/Google/IonQ devices was used. The HoloG calibration layer targets IBM Gross Code geometry but operates entirely in simulation.
- **Not competitive with production decoders.** PyMatching and Chromobius would outperform the included decoders in both speed and accuracy for practical deployment. The ASR-MP decoder is a research prototype exploring adaptivity under drift.
- **Not a real-time system.** The feedback controller and pulse synthesizer run in Python. The RTL modules demonstrate the FPGA architecture concept but have not been synthesized or deployed on hardware.

## Known Simplifications

| Component | Simplification |
|---|---|
| Surface code circuits | Uses Stim's `Circuit.generated()` — no custom stabilizer scheduling |
| Noise models | Pauli (stochastic) noise only; coherent effects require the optional `[cirq]` bridge |
| Leakage | Modeled as persistent bit-flip errors; true |2⟩ state simulation requires Cirq |
| Cosmic rays | Gaussian spatial profile; real impact profiles may differ |
| Pulse synthesis | L-BFGS-B on small chains (4–8 qubits); does not scale to full-device optimization |
| Calibration | Single plaquette (7 qubits max); full-chip calibration would require batched simulation |
| FPGA RTL | LUTRAM vs BRAM comparison only; no decoder logic or full control plane |

## Dependency on External Libraries

Core scientific value comes from combining these tools, not reimplementing them:

- **Stim** — stabilizer circuit simulation and sampling
- **PyMatching** — minimum-weight perfect matching decoder
- **ldpc** — belief propagation + ordered statistics decoding
- **JAX** — automatic differentiation for calibration
- **pyGSTi** — gate set tomography (comparison baseline only)

## Scaling Limits

- Hamiltonian learning: tested up to 10 qubits (Hilbert space = 2¹⁰)
- Pulse synthesis: practical up to ~8 qubits due to matrix exponentiation cost
- Surface code simulation: distance 3–15 typical; d > 15 requires `sinter` for parallelism
- Calibration: single plaquette; multi-plaquette would need graph-based batching
