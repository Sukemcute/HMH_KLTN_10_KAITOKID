// ============================================================================
// Module : pe_cluster_v4
// Project: YOLOv10n INT8 Accelerator — V4-VC707
//
// Description:
//   ★ CORE V4: 3 rows × 4 columns × 20 lanes PE array = 240 MACs.
//   Each column receives DIFFERENT weight → 4 different output channels.
//   (Eyeriss v2 per-column routing concept.)
//
//   Contains:
//     12 × pe_unit  (3 rows × 4 columns)
//      4 × column_reduce (1 per column, sums 3 rows)
//      1 × comparator_tree (for PE_MP5 maxpool mode)
//
//   Dataflow:
//     act_taps[row][col]: per-column for DW3/DW7; RS3 cols see same cin (broadcast in GLB)
//     wgt_data[row][col] = per-column weight
//     col_psum[col] = sum of 3 PE rows for each column → PPU[col]
//
// Resources per instance: 120 DSP + ~6K LUT + ~7K FF
// Instances: 16 (1 per subcluster)
// ============================================================================
`timescale 1ns / 1ps

module pe_cluster_v4
  import accel_pkg::*;
#(
  parameter int LANES   = accel_pkg::LANES,     // 20
  parameter int PE_ROWS = accel_pkg::PE_ROWS,   // 3
  parameter int PE_COLS = accel_pkg::PE_COLS     // 4
)(
  input  logic          clk,
  input  logic          rst_n,

  // ── Control (from compute_sequencer) ──
  input  pe_mode_e      pe_mode,
  input  logic          pe_enable,
  input  logic          pe_clear_acc,

  // ── Activation: PE_RS3 same cin → cols identical; PE_DW3/DW7 → per-col channel
  input  int8_t         act_taps [PE_ROWS][PE_COLS][LANES],

  // ── Weight input: ★ DIFFERENT per column (per-column from router) ──
  // wgt_data[row][col][lane]: 3 rows × 4 cols × 20 lanes
  input  int8_t         wgt_data [PE_ROWS][PE_COLS][LANES],

  // ── PSUM output: 4 columns × 20 lanes (to PPU or PSUM bank) ──
  output int32_t        col_psum [PE_COLS][LANES],
  output logic          psum_valid,

  // ── Multi-pass accumulation input (from GLB output bank PSUM namespace) ──
  input  int32_t        psum_accum_in [PE_COLS][LANES],
  input  logic          psum_accum_en,

  // ── MaxPool bypass (PE_MP5 mode) ──
  input  int8_t         pool_window [25][LANES],  // 5×5 = 25 inputs per lane
  input  logic          pool_enable,
  output int8_t         pool_max    [LANES],
  output logic          pool_valid
);

  // ═══════════════════════════════════════════════════════════════
  // PE Array: 3 rows × 4 columns = 12 pe_unit instances
  // ═══════════════════════════════════════════════════════════════
  int32_t pe_psum [PE_ROWS][PE_COLS][LANES];  // Raw PE outputs
  logic   pe_psum_valid [PE_ROWS][PE_COLS];   // Per-PE valid (unused, all same timing)

  genvar r, c;
  generate
    for (r = 0; r < PE_ROWS; r++) begin : gen_pe_row
      for (c = 0; c < PE_COLS; c++) begin : gen_pe_col
        // ────────────────────────────────────────
        // PE[row=r, col=c]:
        //   Activation: act_taps[r][c]; Weight: wgt_data[r][c]
        //   Mode:       pe_mode (from descriptor, uniform)
        // ────────────────────────────────────────
        pe_unit #(
          .LANES(LANES)
`ifdef RTL_TRACE
          ,
          .TRACE_CLUSTER(((r == 0) && (c == 0)) ? 1'b1 : 1'b0)
`endif
        ) u_pe (
          .clk        (clk),
          .rst_n      (rst_n),
          .pe_mode    (pe_mode),
          .x_in       (act_taps[r][c]),
          .w_in       (wgt_data[r][c]),    // ★ DIFFERENT weight per col
          .pe_enable  (pe_enable),
          .clear_acc  (pe_clear_acc),
          .psum_out   (pe_psum[r][c])
        );
      end
    end
  endgenerate

  // ═══════════════════════════════════════════════════════════════
  // Column Reduce: 1 instance, sums 3 rows across all 4 columns
  // col_psum[c][l] = pe_psum[0][c][l] + pe_psum[1][c][l] + pe_psum[2][c][l]
  // ═══════════════════════════════════════════════════════════════
  int32_t reduced [PE_COLS][LANES];
  logic   reduced_valid_w;

  // Capture pe_unit psum_valid from any PE (all same timing)
  logic pe_valid_any;
  assign pe_valid_any = gen_pe_row[0].gen_pe_col[0].u_pe.psum_valid;

  column_reduce #(
    .LANES  (LANES),
    .N_ROWS (PE_ROWS),
    .PE_COLS(PE_COLS)
  ) u_col_reduce (
    .clk       (clk),
    .rst_n     (rst_n),
    .row_psum  (pe_psum),           // Full [PE_ROWS][PE_COLS][LANES]
    .valid_in  (pe_valid_any),      // Delayed from PE pipeline (5 cycles)
    .col_psum  (reduced),           // [PE_COLS][LANES]
    .valid_out (reduced_valid_w)
  );

  // ═══════════════════════════════════════════════════════════════
  // Multi-pass Accumulation: add previous PSUM (from GLB output bank)
  // Used when Cin > tile_cin and accumulation spans multiple passes.
  // ═══════════════════════════════════════════════════════════════
  always_comb begin
    for (int cc = 0; cc < PE_COLS; cc++) begin
      for (int ll = 0; ll < LANES; ll++) begin
        if (psum_accum_en)
          col_psum[cc][ll] = reduced[cc][ll] + psum_accum_in[cc][ll];
        else
          col_psum[cc][ll] = reduced[cc][ll];
      end
    end
  end

  assign psum_valid = reduced_valid_w;

  // ═══════════════════════════════════════════════════════════════
  // Comparator Tree: MaxPool 5×5 (PE_MP5 mode bypass)
  // 25 signed INT8 inputs → 1 maximum per lane, 5-stage pipeline.
  // Active ONLY in PE_MP5 mode. PE array output ignored in this mode.
  // ═══════════════════════════════════════════════════════════════
  comparator_tree #(
    .LANES     (LANES),
    .NUM_INPUTS(25)
  ) u_comparator (
    .clk      (clk),
    .rst_n    (rst_n),
    .data_in  (pool_window),
    .valid_in (pool_enable),
    .max_out  (pool_max),
    .valid_out(pool_valid)
  );

  // synthesis translate_off
`ifdef S8_DBG
  always @(posedge clk) begin
    if (rst_n) begin
      if (pe_enable)
        $display("  [PE] %0t pe_enable  act[0][0][0]=%0d act[0][0][1]=%0d  wgt[0][0][0]=%0d wgt[0][0][1]=%0d  clr=%b",
                 $time,
                 act_taps[0][0][0], act_taps[0][0][1],
                 wgt_data[0][0][0], wgt_data[0][0][1],
                 pe_clear_acc);
      if (pe_valid_any)
        $display("  [PE] %0t pe_valid  psum_r0c0[0]=%0d psum_r1c0[0]=%0d psum_r2c0[0]=%0d",
                 $time,
                 pe_psum[0][0][0], pe_psum[1][0][0], pe_psum[2][0][0]);
      if (reduced_valid_w)
        $display("  [COLRED] %0t reduced[0][0]=%0d reduced[0][1]=%0d  col_psum[0][0]=%0d",
                 $time,
                 reduced[0][0], reduced[0][1], col_psum[0][0]);
    end
  end
`endif
`ifdef RTL_TRACE
  always @(posedge clk) begin
    if (rst_n) begin
      if (pe_enable)
        rtl_trace_pkg::rtl_trace_line("S7_PEC",
          $sformatf("en act00=%0d act01=%0d w000=%0d w001=%0d clr=%b",
                    act_taps[0][0][0], act_taps[0][0][1],
                    wgt_data[0][0][0], wgt_data[0][0][1], pe_clear_acc));
      if (pe_valid_any)
        rtl_trace_pkg::rtl_trace_line("S7_PECV",
          $sformatf("p00=%0d p10=%0d p20=%0d",
                    pe_psum[0][0][0], pe_psum[1][0][0], pe_psum[2][0][0]));
      if (reduced_valid_w)
        rtl_trace_pkg::rtl_trace_line("S7_RED",
          $sformatf("r00=%0d r01=%0d c00=%0d",
                    reduced[0][0], reduced[0][1], col_psum[0][0]));
      if (pool_valid)
        rtl_trace_pkg::rtl_trace_line("S7_POOL",
          $sformatf("mx0=%0d mx1=%0d", pool_max[0], pool_max[1]));
    end
  end
`endif
  // synthesis translate_on

endmodule
