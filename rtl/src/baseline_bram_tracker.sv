// Baseline BRAM Pauli Frame Tracker
//
// Uses BlockRAM for the Pauli frame store. Correctly models the
// synchronous nature of BRAM with pipeline registers and a busy_flag
// to indicate the one-cycle read latency (causes pipeline stalls).
//
// Migrated from: qec-fpga-benchmark/src/baseline_bram_tracker.sv

module baseline_bram_tracker #(
    parameter int NUM_QUBITS = 49,
    parameter int ADDR_W    = $clog2(NUM_QUBITS)
)(
    input  logic             clk,
    input  logic             rst_n,
    // Write port
    input  logic             wr_en,
    input  logic [ADDR_W-1:0] wr_addr,
    input  logic [1:0]       wr_pauli,
    // Read port (synchronous — 1-cycle latency)
    input  logic             rd_en,
    input  logic [ADDR_W-1:0] rd_addr,
    output logic [1:0]       rd_pauli,
    output logic             rd_valid,
    output logic             busy_flag
);

    // BlockRAM — inferred by synthesis
    (* ram_style = "block" *)
    logic [1:0] frame [0:NUM_QUBITS-1];

    // Pipeline register for read latency
    logic rd_pending;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < NUM_QUBITS; i++)
                frame[i] <= 2'b00;
            rd_pauli  <= 2'b00;
            rd_valid  <= 1'b0;
            rd_pending <= 1'b0;
        end else begin
            // Write
            if (wr_en)
                frame[wr_addr] <= wr_pauli;

            // Read pipeline: stage 1 → stage 2
            rd_valid <= rd_pending;
            if (rd_pending)
                rd_pauli <= frame[rd_addr];  // captured 1 cycle late

            rd_pending <= rd_en;
        end
    end

    assign busy_flag = rd_pending;

endmodule
