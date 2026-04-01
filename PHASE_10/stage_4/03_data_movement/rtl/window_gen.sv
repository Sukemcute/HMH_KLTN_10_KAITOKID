`timescale 1ns/1ps

// Spatial window tap generator: produces K tap vectors from input stream.
// Supports K = 1, 3, 5, 7 (for conv1×1, conv3×3, conv5×5/SPPF, DW7×7).
module window_gen #(
  parameter int LANES = 32,
  parameter int K_MAX = 7
)(
  input  logic                clk,
  input  logic                rst_n,
  input  logic                flush,
  input  logic [2:0]          cfg_kw,       // 1, 3, 5, or 7
  input  logic                shift_in_valid,
  input  logic signed [7:0]   shift_in [LANES],

  output logic                taps_valid,
  output logic signed [7:0]   taps [K_MAX][LANES]
);

  // Shift register chain: K_MAX depth × LANES width
  logic signed [7:0] sr [K_MAX][LANES];

  // Count how many rows have been shifted in
  logic [3:0] fill_count;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n || flush) begin
      for (int k = 0; k < K_MAX; k++)
        for (int l = 0; l < LANES; l++)
          sr[k][l] <= '0;
      fill_count <= '0;
    end else if (shift_in_valid) begin
      // Shift chain: sr[K-1] ← sr[K-2] ← ... ← sr[1] ← sr[0] ← shift_in
      for (int k = K_MAX - 1; k > 0; k--)
        for (int l = 0; l < LANES; l++)
          sr[k][l] <= sr[k-1][l];

      for (int l = 0; l < LANES; l++)
        sr[0][l] <= shift_in[l];

      if (fill_count < K_MAX)
        fill_count <= fill_count + 1;
    end
  end

  // Output taps based on configured kernel width
  always_comb begin
    for (int k = 0; k < K_MAX; k++)
      for (int l = 0; l < LANES; l++)
        taps[k][l] = sr[k][l];
  end

  // Valid when enough rows accumulated
  assign taps_valid = (fill_count >= cfg_kw) & shift_in_valid;

endmodule
