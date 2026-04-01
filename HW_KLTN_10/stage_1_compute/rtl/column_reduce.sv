// ============================================================================
// Module : column_reduce
// Project: YOLOv10n INT8 Accelerator — V4-VC707
// Description:
//   Cross-row partial sum reduction for the PE cluster.
//   Sums 3 PE row outputs (kh=0,1,2) into 1 result per column per lane.
//
//   For each (column c, lane l):
//     col_psum[c][l] = pe_psum[row=0][c][l]
//                    + pe_psum[row=1][c][l]
//                    + pe_psum[row=2][c][l]
//
//   Implementation: 2-level adder tree (registered output for timing)
//     Level 1: sum_01 = row[0] + row[1]          (combinational)
//     Level 2: sum    = sum_01 + row[2]           (combinational)
//     Output:  registered (1 cycle latency)
//
//   Latency: 1 clock cycle
//   Resources: ~320 LUT + ~200 FF per instance (4 cols × 20 lanes)
//
// Instances: 1 per PE cluster × 16 subclusters = 16 total
//            (processes all 4 columns internally)
// ============================================================================
`timescale 1ns / 1ps

module column_reduce
  import accel_pkg::*;
#(
  parameter int LANES   = accel_pkg::LANES,    // 20
  parameter int PE_ROWS = accel_pkg::PE_ROWS,  // 3
  parameter int PE_COLS = accel_pkg::PE_COLS    // 4
)(
  input  logic              clk,
  input  logic              rst_n,

  // — Enable —
  input  logic              valid_in,

  // — Input: partial sums from 3 PE rows × 4 columns × 20 lanes —
  input  logic signed [31:0] row_psum [PE_ROWS][PE_COLS][LANES],

  // — Output: reduced sums per column × 20 lanes —
  output logic signed [31:0] col_psum [PE_COLS][LANES],
  output logic               valid_out
);

  // ════════════════════════════════════════════════════════════════
  //  COMBINATIONAL ADDER TREE
  //  For each (column, lane): sum 3 rows using 2-level tree.
  //  Level 1: partial = row[0] + row[1]
  //  Level 2: total   = partial + row[2]
  // ════════════════════════════════════════════════════════════════
  logic signed [31:0] sum_comb [PE_COLS][LANES];

  always_comb begin
    for (int c = 0; c < PE_COLS; c++) begin
      for (int l = 0; l < LANES; l++) begin
        // 2-level adder tree: (row0 + row1) + row2
        sum_comb[c][l] = row_psum[0][c][l]
                       + row_psum[1][c][l]
                       + row_psum[2][c][l];
      end
    end
  end

  // ════════════════════════════════════════════════════════════════
  //  OUTPUT REGISTER
  //  Registered output for timing closure at 250 MHz.
  //  1 cycle latency.
  // ════════════════════════════════════════════════════════════════
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int c = 0; c < PE_COLS; c++)
        for (int l = 0; l < LANES; l++)
          col_psum[c][l] <= 32'sd0;
      valid_out <= 1'b0;
    end else begin
      valid_out <= valid_in;
      if (valid_in) begin
        for (int c = 0; c < PE_COLS; c++)
          for (int l = 0; l < LANES; l++)
            col_psum[c][l] <= sum_comb[c][l];
      end
    end
  end

endmodule
