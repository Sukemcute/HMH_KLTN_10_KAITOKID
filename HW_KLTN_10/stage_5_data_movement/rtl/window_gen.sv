// ============================================================================
// Module : window_gen
// Project: YOLOv10n INT8 Accelerator — V4-VC707
//
// Description:
//   Sliding window shift register. Produces K tap vectors from sequential
//   input stream, where K is configurable: 1, 3, 5, or 7.
//
//   K=1: Conv 1×1 (pass-through, no buffering)
//   K=3: Conv 3×3, DW 3×3
//   K=5: MaxPool 5×5 (SPPF)
//   K=7: DW 7×7 (QC2fCIB)
//
//   Shift chain: taps[K-1] ← taps[K-2] ← ... ← taps[0] ← shift_in
//   taps_valid asserts when fill_count >= cfg_k (enough rows shifted in).
//
//   V4: LANES=20 → 7 × 20 = 140 registers (compact, good for 250 MHz).
//
// Structural: K_MAX generate blocks of LANES-wide shift registers.
// ============================================================================
`timescale 1ns / 1ps

module window_gen
  import accel_pkg::*;
#(
  parameter int LANES = accel_pkg::LANES,  // 20
  parameter int K_MAX = 7                  // Maximum supported kernel width
)(
  input  logic          clk,
  input  logic          rst_n,

  // ── Configuration ──
  input  logic [3:0]    cfg_k,             // Active kernel width: 1, 3, 5, or 7

  // ── Input (1 row of LANES values per cycle) ──
  input  logic          shift_in_valid,    // Shift when high
  input  int8_t         shift_in [LANES],  // New row data (LANES = 20 values)

  // ── Output ──
  output int8_t         taps [K_MAX][LANES], // Up to 7 tap vectors
  output logic          taps_valid,          // Asserted when enough rows buffered

  // ── Control ──
  input  logic          flush              // Reset shift register + fill counter
);

  // ════════════════════════════════════════════════════════════════
  // Shift register storage: K_MAX rows × LANES columns
  // Generated structurally as independent per-lane shift chains.
  // ════════════════════════════════════════════════════════════════
  int8_t sr [K_MAX][LANES];

  // Fill counter: tracks how many rows have been shifted in
  logic [3:0] fill_count;

  // ── Shift register: chain of K_MAX stages ──
  genvar k, ln;
  generate
    for (ln = 0; ln < LANES; ln++) begin : gen_lane
      // Each lane is an independent K_MAX-deep shift chain
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n || flush) begin
          for (int i = 0; i < K_MAX; i++)
            sr[i][ln] <= 8'sd0;
        end else if (shift_in_valid) begin
          // Shift chain: sr[K-1] ← sr[K-2] ← ... ← sr[1] ← sr[0] ← input
          for (int i = K_MAX - 1; i > 0; i--)
            sr[i][ln] <= sr[i-1][ln];
          sr[0][ln] <= shift_in[ln];
        end
      end
    end
  endgenerate

  // ── Fill counter ──
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n || flush)
      fill_count <= 4'd0;
    else if (shift_in_valid && fill_count < K_MAX[3:0])
      fill_count <= fill_count + 4'd1;
  end

  // ── Output tap assignment (direct from shift register) ──
  always_comb begin
    for (int i = 0; i < K_MAX; i++)
      for (int l = 0; l < LANES; l++)
        taps[i][l] = sr[i][l];
  end

  // ── Valid: enough rows shifted in for current kernel width ──
  assign taps_valid = (fill_count >= cfg_k) & shift_in_valid;

endmodule
