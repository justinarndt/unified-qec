# RTL Companion — FPGA Pauli Frame Tracking

SystemVerilog modules implementing FPGA-based Pauli frame trackers for
real-time QEC control planes, with a benchmark harness to quantify the
BRAM ↔ LUTRAM throughput gap.

> **Origin**: Evolved from [qec-fpga-benchmark](https://github.com/justinarndt/qec-fpga-benchmark),
> refined with production-quality interfaces (separate read/write ports,
> async reset, auto-sized addressing via `$clog2`).

## Modules

| Module | Memory | Read Latency | Stalls? | Use Case |
|---|---|---|---|---|
| `apex_lutram_tracker` | Distributed LUTRAM | 0 cycles (async) | No | High-throughput control planes |
| `baseline_bram_tracker` | Block RAM | 1 cycle (sync) | Yes (`busy_flag`) | Area-efficient baseline |

## Design Notes

- Both modules track per-qubit Pauli frame as `{Z, X}` 2-bit values
- `apex_lutram_tracker` eliminates pipeline stalls via asynchronous reads
  from distributed RAM, achieving single-cycle update throughput
- `baseline_bram_tracker` serves as a comparison point, demonstrating
  the stall penalty from synchronous BRAM reads
- Both modules include proper `rst_n` reset and explicit `(* ram_style *)`
  synthesis attributes

## Benchmark Harness

The `sim/benchmark_harness.sv` testbench runs two tests:

1. **Rapid-Fire Write** — 10 consecutive write operations, 1 per clock cycle.
   LUTRAM accepts all 10; BRAM stalls on alternating cycles (50% throughput loss).

2. **Read-After-Write Latency** — Write then immediately read the same qubit.
   LUTRAM: data available same cycle (async read). BRAM: 1-cycle delay.

### Running the Benchmark

**Vivado (recommended for synthesis targeting)**:
1. Create RTL project targeting any UltraScale+ part (e.g. `xcvu19p`, `xcku5p`, `xczu7ev`)
2. Add `src/*.sv` and `sim/benchmark_harness.sv` (SystemVerilog)
3. Run Behavioral Simulation → check Tcl console for PASS/FAIL

**Icarus Verilog (open-source)**:
```bash
iverilog -g2012 -o bench rtl/src/*.sv rtl/sim/benchmark_harness.sv
vvp bench
```

### Expected Output

```
============================================================
  QEC FPGA Benchmark: LUTRAM (Apex) vs BRAM (Baseline)
  Target: 49 qubits  |  Clock: 100 MHz (10 ns period)
============================================================

--- TEST 1: Rapid-Fire Write (10 consecutive cycles) ---
[PASS] Apex  LUTRAM  accepted write to qubit 0
[PASS] BRAM  Baseline accepted write to qubit 0
[PASS] Apex  LUTRAM  accepted write to qubit 1
[FAIL] BRAM  Baseline STALLED — dropped qubit 1
...

--- TEST 2: Read-After-Write Latency ---
[PASS] Apex  LUTRAM  read-after-write: got 11 (expected 11)
[PASS] BRAM  Baseline read valid after 1-cycle latency

============================================================
  RESULTS
  Apex  (LUTRAM) accepts:  10 / 10  (0 stalls)
  BRAM  (Baseline) accepts: 5 / 10  (5 stalls)
  LUTRAM throughput: 100%  (single-cycle update)
  BRAM   throughput: 50%   (stall penalty)
============================================================
```

## Connection to Python Framework

The feedback latency modeled in `SyndromeFeedbackController` (see
`src/unified_qec/feedback/controller.py`) directly corresponds to the
BRAM stall penalty measured by this benchmark. The LUTRAM tracker's
zero-stall operation validates the sub-microsecond feedback latency
assumed in the controller's T1/T2 decay penalty calculation.

## Synthesis Targets

Designed for Xilinx FPGAs (UltraScale+). The `(* ram_style *)` pragmas
guide synthesis tool inference.

## File Layout

```
rtl/
├── src/
│   ├── apex_lutram_tracker.sv     Zero-stall LUTRAM Pauli frame tracker
│   └── baseline_bram_tracker.sv   BRAM baseline with pipeline stalls
├── sim/
│   └── benchmark_harness.sv       Rapid-fire + RAW latency testbench
└── README.md
```
