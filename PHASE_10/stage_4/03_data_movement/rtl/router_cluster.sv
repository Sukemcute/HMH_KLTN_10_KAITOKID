`timescale 1ns/1ps
// Data routing hub between GLB banks, PE cluster, PPU, and swizzle engine.
// Three sub-routers: RIN (activation), RWT (weight), RPS (psum/output).
module router_cluster #(
  parameter int LANES = 32
)(
  input  logic                     clk,
  input  logic                     rst_n,
  input  desc_pkg::router_profile_t cfg_profile,
  input  accel_pkg::pe_mode_e       cfg_mode,

  // RIN: Activation Router (3 input banks → 3 PE rows)
  input  logic [LANES*8-1:0]      bank_input_rd [3],
  output logic signed [7:0]       pe_act [3][LANES],

  // RWT: Weight Router (3 weight banks → 3 PE rows)
  input  logic [LANES*8-1:0]      bank_weight_rd [3],
  output logic signed [7:0]       pe_wgt [3][LANES],

  // RPS: Psum/Output Router (4 PE cols → 4 output banks)
  input  logic signed [31:0]      pe_psum [4][LANES],
  input  logic                    psum_valid,
  output logic [LANES*32-1:0]     bank_output_wr [4],
  output logic                    bank_output_wr_en [4],

  // Bypass paths (MOVE, CONCAT, UPSAMPLE)
  input  logic                    bypass_en,
  input  logic [LANES*8-1:0]     bypass_data_in,
  output logic [LANES*8-1:0]     bypass_data_out,
  output logic                    bypass_valid
);
  import accel_pkg::*;

  // ───── RIN: Activation Routing ─────
  // Unpack flat bus → signed array per PE row based on rin_src select
  always_comb begin
    for (int row = 0; row < 3; row++) begin
      logic [2:0] src_sel;
      src_sel = cfg_profile.rin_src[row];
      for (int lane = 0; lane < LANES; lane++) begin
        case (src_sel)
          3'd0: pe_act[row][lane] = $signed(bank_input_rd[0][lane*8 +: 8]);
          3'd1: pe_act[row][lane] = $signed(bank_input_rd[1][lane*8 +: 8]);
          3'd2: pe_act[row][lane] = $signed(bank_input_rd[2][lane*8 +: 8]);
          default: pe_act[row][lane] = 8'sd0;
        endcase
      end
    end
  end

  // ───── RWT: Weight Routing ─────
  always_comb begin
    for (int row = 0; row < 3; row++) begin
      logic [2:0] src_sel;
      src_sel = cfg_profile.rwt_src[row];
      for (int lane = 0; lane < LANES; lane++) begin
        case (src_sel)
          3'd0: pe_wgt[row][lane] = $signed(bank_weight_rd[0][lane*8 +: 8]);
          3'd1: pe_wgt[row][lane] = $signed(bank_weight_rd[1][lane*8 +: 8]);
          3'd2: pe_wgt[row][lane] = $signed(bank_weight_rd[2][lane*8 +: 8]);
          default: pe_wgt[row][lane] = 8'sd0;
        endcase
      end
    end
  end

  // ───── RPS: Psum → Output Bank Packing ─────
  always_comb begin
    for (int col = 0; col < 4; col++) begin
      bank_output_wr_en[col] = psum_valid;
      for (int lane = 0; lane < LANES; lane++) begin
        bank_output_wr[col][lane*32 +: 32] = pe_psum[col][lane];
      end
    end
  end

  // ───── Bypass Path ─────
  assign bypass_data_out = bypass_en ? bypass_data_in : '0;
  assign bypass_valid    = bypass_en;

endmodule
