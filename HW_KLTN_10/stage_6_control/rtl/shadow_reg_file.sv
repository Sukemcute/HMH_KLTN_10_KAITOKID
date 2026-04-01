// ============================================================================
// Module : shadow_reg_file
// Project: YOLOv10n INT8 Accelerator — V4-VC707
//
// Description:
//   Configuration register bank that latches descriptor fields on
//   `latch_en` pulse and holds them STABLE throughout tile execution.
//
//   Prevents descriptor changes mid-compute from corrupting the datapath.
//   All downstream modules (compute_sequencer, addr_gens, PE cluster)
//   read from these stable outputs, NOT from the raw descriptor.
//
//   Latched on: tile_fsm TS_LOAD_DESC → shadow_latch pulse
//   Outputs stable until: next TS_LOAD_DESC for new tile
// ============================================================================
`timescale 1ns / 1ps

module shadow_reg_file
  import accel_pkg::*;
  import desc_pkg::*;
(
  input  logic          clk,
  input  logic          rst_n,

  // ── Latch control (from tile_fsm) ──
  input  logic          latch_en,          // 1-cycle pulse: capture descriptors

  // ── Raw descriptor inputs ──
  input  layer_desc_t   layer_desc_in,
  input  tile_desc_t    tile_desc_in,

  // ── Stable configuration outputs ──
  // PE mode & activation
  output pe_mode_e      o_pe_mode,
  output act_mode_e     o_activation,

  // Dimensions
  output logic [9:0]    o_cin, o_cout,
  output logic [9:0]    o_hin, o_win,
  output logic [9:0]    o_hout, o_wout,

  // Kernel
  output logic [3:0]    o_kh, o_kw,
  output logic [2:0]    o_stride, o_padding,

  // Multi-pass
  output logic [3:0]    o_num_cin_pass, o_num_k_pass,

  // Swizzle
  output swizzle_mode_e o_swizzle,

  // Zero-point for padding (from tile or external source)
  output int8_t         o_zp_x,

  // Tile identification
  output logic [15:0]   o_tile_id,
  output logic [4:0]    o_layer_id,
  output logic          o_first_tile, o_last_tile,
  output logic          o_hold_skip, o_need_swizzle
);

  // ═══════════════════════════════════════════════════════════════
  // Register bank: latch on latch_en, hold otherwise
  // ═══════════════════════════════════════════════════════════════
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      o_pe_mode       <= PE_RS3;
      o_activation    <= ACT_NONE;
      o_cin           <= 10'd0;
      o_cout          <= 10'd0;
      o_hin           <= 10'd0;
      o_win           <= 10'd0;
      o_hout          <= 10'd0;
      o_wout          <= 10'd0;
      o_kh            <= 4'd0;
      o_kw            <= 4'd0;
      o_stride        <= 3'd1;
      o_padding       <= 3'd0;
      o_num_cin_pass  <= 4'd1;
      o_num_k_pass    <= 4'd1;
      o_swizzle       <= SWZ_NORMAL;
      o_zp_x          <= 8'sd0;
      o_tile_id       <= 16'd0;
      o_layer_id      <= 5'd0;
      o_first_tile    <= 1'b0;
      o_last_tile     <= 1'b0;
      o_hold_skip     <= 1'b0;
      o_need_swizzle  <= 1'b0;
    end else if (latch_en) begin
      // Layer-level config
      o_pe_mode       <= layer_desc_in.pe_mode;
      o_activation    <= layer_desc_in.activation;
      o_cin           <= layer_desc_in.cin;
      o_cout          <= layer_desc_in.cout;
      o_hin           <= layer_desc_in.hin;
      o_win           <= layer_desc_in.win;
      o_hout          <= layer_desc_in.hout;
      o_wout          <= layer_desc_in.wout;
      o_kh            <= layer_desc_in.kh;
      o_kw            <= layer_desc_in.kw;
      o_stride        <= layer_desc_in.stride;
      o_padding       <= layer_desc_in.padding;
      o_num_cin_pass  <= layer_desc_in.num_cin_pass;
      o_num_k_pass    <= layer_desc_in.num_k_pass;
      o_swizzle       <= layer_desc_in.swizzle;

      // Tile-level config
      o_tile_id       <= tile_desc_in.tile_id;
      o_layer_id      <= tile_desc_in.layer_id;
      o_first_tile    <= tile_desc_in.first_tile;
      o_last_tile     <= tile_desc_in.last_tile;
      o_hold_skip     <= tile_desc_in.hold_skip;
      o_need_swizzle  <= tile_desc_in.need_swizzle;

      // ZP_x defaults to 0 (updated externally by DMA if needed)
      o_zp_x          <= 8'sd0;
    end
  end

endmodule
