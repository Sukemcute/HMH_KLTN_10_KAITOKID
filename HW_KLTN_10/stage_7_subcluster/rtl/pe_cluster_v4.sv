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
//     act_taps[row] = SAME activation for all 4 columns (kh parallelism)
//     wgt_data[row][col] = DIFFERENT weight per column (★ cout parallelism)
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

  // ── Activation input: SAME for all columns (multicast from router) ──
  // act_taps[row][lane]: 3 rows × 20 lanes (row = kh dimension)
  input  int8_t         act_taps [PE_ROWS][LANES],

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
        //   Activation: act_taps[r] = SAME for all columns in this row
        //   Weight:     wgt_data[r][c] = ★ DIFFERENT per column
        //   Mode:       pe_mode (from descriptor, uniform)
        // ────────────────────────────────────────
        pe_unit #(.LANES(LANES)) u_pe (
          .clk        (clk),
          .rst_n      (rst_n),
          .pe_mode    (pe_mode),
          .x_in       (act_taps[r]),       // SAME activation (multicast)
          .w_in       (wgt_data[r][c]),    // ★ DIFFERENT weight per col
          .pe_enable  (pe_enable),
          .clear_acc  (pe_clear_acc),
          .psum_out   (pe_psum[r][c])
        );
      end
    end
  endgenerate

  // ═══════════════════════════════════════════════════════════════
  // Column Reduce: 4 instances, each sums 3 rows for its column
  // col_reduce[c].col_sum[l] = pe_psum[0][c][l] + pe_psum[1][c][l] + pe_psum[2][c][l]
  // ═══════════════════════════════════════════════════════════════
  int32_t reduced [PE_COLS][LANES];   // After 3-row reduction
  logic   reduced_valid [PE_COLS];

  generate
    for (c = 0; c < PE_COLS; c++) begin : gen_col_reduce
      // Pack 3 rows for this column into column_reduce input format
      int32_t col_rows [PE_ROWS][LANES];

      // Assign rows for this column
      always_comb begin
        for (int rr = 0; rr < PE_ROWS; rr++)
          for (int ll = 0; ll < LANES; ll++)
            col_rows[rr][ll] = pe_psum[rr][c][ll];
      end

      column_reduce #(.LANES(LANES), .N_ROWS(PE_ROWS)) u_col_reduce (
        .clk       (clk),
        .row_psum  (col_rows),
        .valid_in  (pe_enable),  // Valid follows PE enable (pipeline-matched)
        .col_sum   (reduced[c]),
        .valid_out (reduced_valid[c])
      );
    end
  endgenerate

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

  assign psum_valid = reduced_valid[0];  // All columns same timing

  // ═══════════════════════════════════════════════════════════════
  // Comparator Tree: MaxPool 5×5 (PE_MP5 mode bypass)
  // 25 signed INT8 inputs → 1 maximum per lane, 5-stage pipeline.
  // Active ONLY in PE_MP5 mode. PE array output ignored in this mode.
  // ═══════════════════════════════════════════════════════════════
  // ★ Instance: comparator_tree — port data_in[25][LANES] (not "window")
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

endmodule
