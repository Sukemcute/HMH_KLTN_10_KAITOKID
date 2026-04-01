// Cross-row partial sum reduction: sum 3 PE rows into 1 output per PE column.
// For each (col, lane): col_psum = Σ_{row=0..2} pe_psum[row][col][lane]
`timescale 1ns/1ps
module column_reduce #(
  parameter int LANES   = 32,
  parameter int PE_ROWS = 3,
  parameter int PE_COLS = 4
)(
  input  logic               clk,
  input  logic               rst_n,
  input  logic               en,
  input  accel_pkg::pe_mode_e mode,

  input  logic signed [31:0] pe_psum [PE_ROWS][PE_COLS][LANES],

  output logic signed [31:0] col_psum [PE_COLS][LANES],
  output logic               col_valid
);

  // Combinational reduction + output register (1 cycle latency)
  logic signed [31:0] sum_comb [PE_COLS][LANES];

  always_comb begin
    for (int c = 0; c < PE_COLS; c++) begin
      for (int l = 0; l < LANES; l++) begin
        sum_comb[c][l] = pe_psum[0][c][l]
                       + pe_psum[1][c][l]
                       + pe_psum[2][c][l];
      end
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int c = 0; c < PE_COLS; c++)
        for (int l = 0; l < LANES; l++)
          col_psum[c][l] <= '0;
      col_valid <= 1'b0;
    end else begin
      col_valid <= en;
      if (en) begin
        for (int c = 0; c < PE_COLS; c++)
          for (int l = 0; l < LANES; l++)
            col_psum[c][l] <= sum_comb[c][l];
      end
    end
  end

endmodule
