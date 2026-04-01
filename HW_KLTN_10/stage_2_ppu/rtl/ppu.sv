// ============================================================================
// Module : ppu (Post-Processing Unit)
// Project: YOLOv10n INT8 Accelerator — V4-VC707
//
// Description:
//   Converts INT32 partial sums → INT8 output activations.
//   5-stage registered pipeline (V4: for 250 MHz timing closure).
//
//   Stage 1: BIAS ADD        — psum + bias → biased (INT32)
//   Stage 2: INT64 MULTIPLY  — biased × M_int → product (INT64)  [RULE 3]
//   Stage 3: HALF-UP ROUND   — (product + half) >>> shift (INT32) [RULE 2]
//   Stage 4: ACTIVATION      — ReLU: max(0, x) or identity       [RULE 4]
//   Stage 5: ZP + CLAMP      — (activated + zp_out) clamped [-128,127] [RULE 10]
//
// Golden Rules Enforced:
//   RULE 2: Half-up rounding (NOT floor)
//   RULE 3: INT64 for multiply stage (INT32 overflows for deep layers)
//   RULE 4: Model uses ReLU (NOT SiLU)
//   RULE 10: zp_out added AFTER activation, BEFORE clamp
//
// Parameters:
//   LANES = 20 (V4 spatial parallelism, from accel_pkg)
//
// Instances: 4 per subcluster (1 per PE column) × 16 subs = 64 total
// ============================================================================
`timescale 1ns / 1ps

module ppu
  import accel_pkg::*;
#(
  parameter int LANES = accel_pkg::LANES  // 20
)(
  input  logic          clk,
  input  logic          rst_n,

  // ── Input: from column_reduce (1 PE column's reduced psum) ──
  input  int32_t        psum_in [LANES],   // 20 × INT32 partial sums
  input  logic          psum_valid,        // Asserted 1 cycle when psum ready

  // ── Quantization parameters (per-cout, from shadow_reg_file) ──
  input  int32_t        bias_val,          // B_int32[cout]
  input  uint32_t       m_int,             // M_int[cout] (treated as unsigned for multiply)
  input  logic [7:0]    shift_val,         // shift[cout], range [0..63]
  input  int8_t         zp_out,            // ZP_out (signed)
  input  act_mode_e     activation,        // ACT_RELU or ACT_NONE

  // ── Output: INT8 activations ──
  output int8_t         act_out [LANES],   // 20 × INT8 results
  output logic          act_valid          // Asserted 1 cycle when output ready
);

  // ════════════════════════════════════════════════════════════════
  // Pipeline valid shift register (5 stages)
  // ════════════════════════════════════════════════════════════════
  logic [4:0] valid_pipe;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      valid_pipe <= 5'b0;
    else
      valid_pipe <= {valid_pipe[3:0], psum_valid};
  end

  assign act_valid = valid_pipe[4];

  // ════════════════════════════════════════════════════════════════
  // Pipeline parameter latch (hold params stable through pipeline)
  // ════════════════════════════════════════════════════════════════
  int32_t    bias_s1;
  uint32_t   m_int_s2;
  logic [7:0] shift_s2, shift_s3;
  act_mode_e act_s3, act_s4;
  int8_t     zp_s4, zp_s5;

  always_ff @(posedge clk) begin
    // Latch on psum_valid (stage 0 → 1)
    if (psum_valid) begin
      bias_s1  <= bias_val;
      m_int_s2 <= m_int;
      shift_s2 <= shift_val;
      act_s3   <= activation;
      zp_s4    <= zp_out;
    end
    // Forward through pipeline
    shift_s3 <= shift_s2;
    act_s4   <= act_s3;
    zp_s5    <= zp_s4;
  end

  // ════════════════════════════════════════════════════════════════
  // STAGE 1: BIAS ADD — psum + bias → biased (INT32 + INT32 → INT32)
  // ════════════════════════════════════════════════════════════════
  int32_t biased_s1 [LANES];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int l = 0; l < LANES; l++)
        biased_s1[l] <= 32'sd0;
    end else if (psum_valid) begin
      for (int l = 0; l < LANES; l++)
        biased_s1[l] <= psum_in[l] + bias_val;
    end
  end

  // ════════════════════════════════════════════════════════════════
  // STAGE 2: INT64 MULTIPLY — biased × M_int → product (INT64)
  //
  // ★ RULE 3: MUST use INT64. INT32 overflows for deep layers.
  //   Worst case: biased ≈ 2×10^9, M_int ≈ 2×10^9 → product ≈ 4×10^18
  //   INT32 max = 2.1×10^9 → OVERFLOW if INT32.
  //   INT64 max = 9.2×10^18 → safe.
  // ════════════════════════════════════════════════════════════════
  int64_t product_s2 [LANES];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int l = 0; l < LANES; l++)
        product_s2[l] <= 64'sd0;
    end else if (valid_pipe[0]) begin
      for (int l = 0; l < LANES; l++) begin
        // Signed × Unsigned → treat m_int as positive value
        // Cast biased to INT64, m_int zero-extended to INT64
        product_s2[l] <= int64_t'(biased_s1[l]) * int64_t'({1'b0, m_int_s2});
      end
    end
  end

  // ════════════════════════════════════════════════════════════════
  // STAGE 3: HALF-UP ROUNDING + ARITHMETIC RIGHT SHIFT
  //
  // ★ RULE 2: MUST use half-up rounding.
  //   Correct:  y = (product + (1 << (shift-1))) >>> shift
  //   WRONG:    y = product >>> shift   ← This is FLOOR, causes 30% errors
  //
  //   Half-up adds 0.5 of the divisor (2^shift) before shifting,
  //   converting truncation into round-to-nearest.
  // ════════════════════════════════════════════════════════════════
  int32_t shifted_s3 [LANES];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int l = 0; l < LANES; l++)
        shifted_s3[l] <= 32'sd0;
    end else if (valid_pipe[1]) begin
      for (int l = 0; l < LANES; l++) begin
        automatic int64_t rounded;
        automatic int     sh = int'(shift_s3);

        // Half-up rounding: add (1 << (shift-1)) before shift
        if (sh > 0)
          rounded = product_s2[l] + (int64_t'(1) <<< (sh - 1));
        else
          rounded = product_s2[l];

        // Arithmetic right shift (sign-extending)
        shifted_s3[l] <= int32_t'(rounded >>> sh);
      end
    end
  end

  // ════════════════════════════════════════════════════════════════
  // STAGE 4: ACTIVATION
  //
  // ★ RULE 4: Model uses ReLU: y = max(0, x)
  //   ACT_NONE: identity (used for PSUM intermediate, some concat paths)
  //   ACT_RELU: clamp negative to 0 (main activation for all Conv layers)
  // ════════════════════════════════════════════════════════════════
  int32_t activated_s4 [LANES];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int l = 0; l < LANES; l++)
        activated_s4[l] <= 32'sd0;
    end else if (valid_pipe[2]) begin
      for (int l = 0; l < LANES; l++) begin
        case (act_s4)
          ACT_RELU: activated_s4[l] <= (shifted_s3[l] > 32'sd0) ? shifted_s3[l] : 32'sd0;
          default:  activated_s4[l] <= shifted_s3[l];  // ACT_NONE: identity
        endcase
      end
    end
  end

  // ════════════════════════════════════════════════════════════════
  // STAGE 5: ADD ZP_OUT + FINAL CLAMP to INT8 [-128, 127]
  //
  // ★ RULE 10: zp_out added AFTER activation, BEFORE final clamp.
  //   Order matters: ReLU(x) + zp ≠ ReLU(x + zp)
  // ════════════════════════════════════════════════════════════════
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int l = 0; l < LANES; l++)
        act_out[l] <= 8'sd0;
    end else if (valid_pipe[3]) begin
      for (int l = 0; l < LANES; l++) begin
        automatic int32_t with_zp;
        with_zp = activated_s4[l] + int32_t'(zp_s5);

        // Saturating clamp to INT8 range [-128, 127]
        if (with_zp > 32'sd127)
          act_out[l] <= int8_t'(8'sd127);
        else if (with_zp < -32'sd128)
          act_out[l] <= int8_t'(-8'sd128);
        else
          act_out[l] <= int8_t'(with_zp[7:0]);
      end
    end
  end

endmodule
