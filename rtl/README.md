# RTL Companion — FPGA Pauli Frame Tracking

SystemVerilog modules implementing FPGA-based Pauli frame trackers for
real-time QEC control planes.

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

## Synthesis Targets

Designed for Xilinx FPGAs (Ultrascale+). The `(* ram_style *)` pragmas
guide synthesis tool inference.
