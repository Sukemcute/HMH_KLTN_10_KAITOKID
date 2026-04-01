`timescale 1ns/1ps

// Single Processing Element: 32 lanes using 16 dsp_pair_int8 instances.
// Each DSP pair handles lanes [2i] and [2i+1].
module pe_unit #(
  parameter int LANES = 32
)(
  input  logic              clk,
  input  logic              rst_n,
  input  logic              en,
  input  logic              clear_psum,
  input  accel_pkg::pe_mode_e mode,

  input  logic signed [7:0] x_in  [LANES],
  input  logic signed [7:0] w_in  [LANES],

  output logic signed [31:0] psum_out [LANES],
  output logic               psum_valid
);
  import accel_pkg::*;

  localparam int NUM_PAIRS = LANES / 2;

  // Pipeline valid tracking (4-stage DSP pipeline)
  logic [4:0] valid_sr;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      valid_sr <= '0;
    else
      valid_sr <= {valid_sr[3:0], en};
  end

  assign psum_valid = valid_sr[4];

  // Weight selection per mode.
  // This PE is built from 16 dsp_pair_int8 blocks, so each DSP pair consumes
  // one shared weight for lanes [2g] and [2g+1].
  logic signed [7:0] w_sel [LANES];

  always_comb begin
    case (mode)
      PE_OS1: begin
        // 1×1 conv: broadcast same weight to all lanes
        for (int l = 0; l < LANES; l++)
          w_sel[l] = w_in[0];
      end
      default: begin
        // RS3, DW3, DW7, GEMM: shared weight per DSP pair.
        for (int l = 0; l < LANES; l++)
          w_sel[l] = w_in[l];
      end
    endcase
  end

  // Instantiate DSP pairs
  genvar g;
  generate
    for (g = 0; g < NUM_PAIRS; g++) begin : gen_dsp
      dsp_pair_int8 u_dsp (
        .clk    (clk),
        .rst_n  (rst_n),
        .en     (en),
        .clear  (clear_psum),
        .x_a    (x_in[2*g]),
        .x_b    (x_in[2*g + 1]),
        .w      (w_sel[2*g]),
        .psum_a (psum_out[2*g]),
        .psum_b (psum_out[2*g + 1])
      );
    end
  endgenerate

endmodule
