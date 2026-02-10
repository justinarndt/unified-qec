// APEX LUTRAM Pauli Frame Tracker
//
// Achieves asynchronous reads and single-cycle updates without stalls.
// Uses distributed LUTRAM (Xilinx FPGAs) for zero-latency read path.
//
// Migrated from: qec-fpga-benchmark/src/apex_lutram_tracker.sv

module apex_lutram_tracker #(
    parameter int NUM_QUBITS = 49,
    parameter int ADDR_W    = $clog2(NUM_QUBITS)
)(
    input  logic             clk,
    input  logic             rst_n,
    // Write port
    input  logic             wr_en,
    input  logic [ADDR_W-1:0] wr_addr,
    input  logic [1:0]       wr_pauli,   // {Z, X} Pauli frame
    // Read port (asynchronous / combinational)
    input  logic [ADDR_W-1:0] rd_addr,
    output logic [1:0]       rd_pauli
);

    // Distributed RAM (LUTRAM) — inferred by synthesis
    (* ram_style = "distributed" *)
    logic [1:0] frame [0:NUM_QUBITS-1];

    // Async read — zero cycle latency
    assign rd_pauli = frame[rd_addr];

    // Sync write
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < NUM_QUBITS; i++)
                frame[i] <= 2'b00;
        end else if (wr_en) begin
            frame[wr_addr] <= wr_pauli;
        end
    end

endmodule
