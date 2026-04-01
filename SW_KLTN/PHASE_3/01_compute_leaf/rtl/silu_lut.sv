// 256-entry INT8 SiLU lookup table with 32 parallel read ports.
// Preloaded via serial load interface, then supports 32 concurrent reads.
// Implementation: replicated ROM (32 copies for parallel access).
`timescale 1ns/1ps
module silu_lut #(
  parameter int LANES = 32
)(
  input  logic                clk,
  input  logic                load_en,
  input  logic [7:0]          load_addr,
  input  logic signed [7:0]   load_data,
  input  logic [7:0]          idx [LANES],
  output logic signed [7:0]   out [LANES]
);

  // ROM storage (256 entries of INT8)
  logic signed [7:0] rom [256];

  // Load interface
  always_ff @(posedge clk) begin
    if (load_en)
      rom[load_addr] <= load_data;
  end

  // 32 parallel reads (registered output for timing)
  always_ff @(posedge clk) begin
    for (int l = 0; l < LANES; l++)
      out[l] <= rom[idx[l]];
  end

endmodule
