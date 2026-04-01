// ============================================================================
// Module : addr_gen_input
// Project: YOLOv10n INT8 Accelerator — V4-VC707
//
// Description:
//   Computes GLB input bank address from logical coordinates (h_in, w, cin).
//   Banking rule: bank_id = h_in mod 3 (3 input banks)
//
//   ★ RULE 5: Padding positions output cfg_zp_x (NOT literal 0).
//     In quantized INT8, "true zero" = zero_point, not 0.
//
//   Combinational address logic + 1 cycle registered output.
//
// Ports driven by compute_sequencer iteration counters.
// ============================================================================
`timescale 1ns / 1ps

module addr_gen_input
  import accel_pkg::*;
#(
  parameter int LANES = accel_pkg::LANES  // 20
)(
  input  logic          clk,
  input  logic          rst_n,

  // ── Configuration (from shadow_reg_file, stable during tile) ──
  input  logic [9:0]    cfg_hin,       // Input height (unpadded)
  input  logic [9:0]    cfg_win,       // Input width (unpadded)
  input  logic [9:0]    cfg_cin,       // Input channels (tile)
  input  logic [2:0]    cfg_stride,    // Stride (1 or 2)
  input  logic [2:0]    cfg_padding,   // Padding size (0, 1, 2, 3)
  input  int8_t         cfg_zp_x,      // ★ Zero-point for padding fill [RULE 5]

  // ── Iteration inputs (from compute_sequencer) ──
  input  logic [9:0]    iter_h_out,    // Current output row
  input  logic [9:0]    iter_wblk,     // Current width block (×LANES spatial positions)
  input  logic [9:0]    iter_cin,      // Current input channel
  input  logic [3:0]    iter_kh_row,   // Current kernel row (0, 1, 2 for 3×3)

  // ── Output (registered, 1-cycle latency) ──
  output logic [1:0]    bank_id,       // h_in mod 3 → selects 1 of 3 input banks
  output logic [11:0]   sram_addr,     // Address within selected bank
  output logic          is_padding,    // 1 = out-of-bounds → use pad_value
  output int8_t         pad_value      // ★ = cfg_zp_x (NOT zero!)
);

  // ── Combinational: compute h_in from h_out + stride + kernel row ──
  logic signed [10:0] h_in_raw;  // Signed to detect negative (padding top)
  logic [9:0]         w_in_base;

  always_comb begin
    // h_in = h_out × stride + kh_row - padding
    h_in_raw  = 11'(signed'({1'b0, iter_h_out})) * 11'(signed'({1'b0, cfg_stride}))
              + 11'(signed'({1'b0, iter_kh_row}))
              - 11'(signed'({1'b0, cfg_padding}));

    // w_in base = wblk × LANES (no stride applied here, stride applied per kw in sequencer)
    w_in_base = iter_wblk * LANES[9:0];
  end

  // ── Combinational: padding detection ──
  logic pad_h, pad_w;

  always_comb begin
    // Height padding: h_in < 0 or h_in >= cfg_hin
    pad_h = (h_in_raw < 0) || (h_in_raw >= signed'({1'b0, cfg_hin}));

    // Width padding: w_in_base >= cfg_win
    // (fine-grained per-lane w padding handled by PE masking or sequencer)
    pad_w = (w_in_base >= cfg_win);
  end

  // ── Combinational: address computation ──
  // Banking: bank_id = h_in mod 3
  // Address: row_slot × Cin_tile × Wblk_total + cin × Wblk_total + wblk
  logic [1:0]  bank_comb;
  logic [11:0] addr_comb;

  always_comb begin
    automatic logic [9:0] h_pos = h_in_raw[9:0];  // Unsigned view (valid only if !padding)
    automatic logic [9:0] wblk_total = (cfg_win + LANES[9:0] - 10'd1) / LANES[9:0];

    bank_comb = h_pos[1:0] % 2'd3;  // h mod 3 → bank 0, 1, or 2

    // Simple linear address: h_in_div3 * cin_tile * wblk_total + cin * wblk_total + wblk
    // Simplified for synthesis: (h/3) * stride + channel offset + width offset
    addr_comb = 12'((h_pos / 10'd3) * wblk_total * cfg_cin[9:0])
              + 12'(iter_cin * wblk_total)
              + 12'(iter_wblk);
  end

  // ── Registered output (1-cycle latency for timing closure) ──
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      bank_id    <= 2'd0;
      sram_addr  <= 12'd0;
      is_padding <= 1'b0;
      pad_value  <= 8'sd0;
    end else begin
      bank_id    <= bank_comb;
      sram_addr  <= addr_comb;
      is_padding <= pad_h | pad_w;
      pad_value  <= cfg_zp_x;  // ★ RULE 5: Always output zp_x for padding
    end
  end

endmodule
