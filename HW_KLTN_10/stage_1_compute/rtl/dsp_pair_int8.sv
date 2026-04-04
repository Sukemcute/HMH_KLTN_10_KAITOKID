// ============================================================================
// Module : dsp_pair_int8
// Project: YOLOv10n INT8 Accelerator — V4-VC707
// Description:
//   Dual signed INT8 × INT8 → INT32 multiply-accumulate packed into a
//   single DSP48E1 using the unsigned-offset packing trick.
//
//   Two adjacent lanes (2i, 2i+1) share ONE weight value per cycle and
//   maintain independent accumulators.
//
//   Packing method:
//     A_packed[24:0] = { x_b_u[7:0], 9'b0, x_a_u[7:0] }   (25 bits)
//     B_packed[17:0] = { 10'b0, w_u[7:0] }                  (18 bits)
//     P = A_packed × B_packed
//     prod_a = P[15:0],  prod_b = P[32:17]  (non-overlapping, 1-bit guard)
//
//   Unsigned-offset correction:
//     a_s × b_s = a_u × b_u − 128×(a_u + b_u) + 16384
//     where a_u = a_s + 128,  b_u = b_s + 128
//
//   Pipeline: ★ 5 stages (V4, +1 vs V3 for 250 MHz target)
//     S1: Input register
//     S2: Signed → Unsigned conversion (registered)
//     S3: DSP packed multiply (registered — maps to DSP48E1 MREG)
//     S4: Product extraction + correction (registered)
//     S5: Accumulate + output register
//
//   Latency: 5 clock cycles from input to first valid accumulation
//   Throughput: 2 MACs per clock cycle (sustained)
//   Resources: 1 DSP48E1 + ~25 LUT + ~48 FF per instance
//
// Golden Rules:
//   RULE 1: Signed INT8 [-128, 127] inputs
//   RULE 3: INT32 accumulators
// ============================================================================
`timescale 1ns / 1ps

module dsp_pair_int8 #(
  parameter bit TRACE_EN = 1'b0  // simulation: log first pair in PE only
)(
  input  logic              clk,
  input  logic              rst_n,

  // — Data inputs (signed INT8) —
  input  logic signed [7:0] x_a,      // Activation lane 2i   (signed)
  input  logic signed [7:0] x_b,      // Activation lane 2i+1 (signed)
  input  logic signed [7:0] w,        // Shared weight for both lanes (signed)

  // — Control —
  input  logic              en,       // Enable accumulation
  input  logic              clear,    // Reset accumulators (start new tile)

  // — Accumulated outputs (signed INT32) —
  output logic signed [31:0] psum_a,  // Running sum for lane 2i
  output logic signed [31:0] psum_b   // Running sum for lane 2i+1
);

  // ╔═══════════════════════════════════════════════════════════════╗
  // ║  STAGE 1: Input Register                                      ║
  // ║  Latch inputs to break combinational path from external logic  ║
  // ╚═══════════════════════════════════════════════════════════════╝
  logic signed [7:0] x_a_s1, x_b_s1, w_s1;
  logic              en_s1,  clear_s1;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      x_a_s1   <= 8'sd0;
      x_b_s1   <= 8'sd0;
      w_s1     <= 8'sd0;
      en_s1    <= 1'b0;
      clear_s1 <= 1'b0;
    end else begin
      x_a_s1   <= x_a;
      x_b_s1   <= x_b;
      w_s1     <= w;
      en_s1    <= en;
      clear_s1 <= clear;
    end
  end

  // ╔═══════════════════════════════════════════════════════════════╗
  // ║  STAGE 2: Signed → Unsigned Conversion                        ║
  // ║  Offset by +128: u = s + 128, maps [-128,127] → [0,255]      ║
  // ║  Must be registered before DSP multiply for timing.           ║
  // ╚═══════════════════════════════════════════════════════════════╝
  logic [7:0] x_a_u_s2, x_b_u_s2, w_u_s2;
  logic       en_s2, clear_s2;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      x_a_u_s2 <= 8'd0;
      x_b_u_s2 <= 8'd0;
      w_u_s2   <= 8'd0;
      en_s2    <= 1'b0;
      clear_s2 <= 1'b0;
    end else begin
      // Unsigned offset: add 128 (bit-trick: flip MSB)
      x_a_u_s2 <= x_a_s1 + 8'sd128;  // = x_a_s1 ^ 8'h80 (equivalent)
      x_b_u_s2 <= x_b_s1 + 8'sd128;
      w_u_s2   <= w_s1   + 8'sd128;
      en_s2    <= en_s1;
      clear_s2 <= clear_s1;
    end
  end

  // ╔═══════════════════════════════════════════════════════════════╗
  // ║  STAGE 3: Packed DSP Multiply                                  ║
  // ║  Pack two 8-bit unsigned values into 25-bit A, share 8-bit B.  ║
  // ║  A[24:0] = { x_b_u[7:0], 9'b0, x_a_u[7:0] }                 ║
  // ║  B[17:0] = { 10'b0, w_u[7:0] }                                ║
  // ║  Products are separated by a 1-bit guard (bit 16 of P).       ║
  // ║  prod_a = P[15:0],  prod_b = P[32:17]                         ║
  // ║  This multiply infers a single DSP48E1 in Vivado synthesis.    ║
  // ╚═══════════════════════════════════════════════════════════════╝
  logic [32:0] dsp_product_s3;  // 25 × 8 = max 33 bits
  logic [7:0]  x_a_u_s3, x_b_u_s3, w_u_s3;
  logic        en_s3, clear_s3;

  // Combinational: form packed operands
  wire [24:0] a_packed = {x_b_u_s2, 9'b0, x_a_u_s2};  // 8 + 9 + 8 = 25 bits
  wire  [7:0] b_value  = w_u_s2;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      dsp_product_s3 <= 33'd0;
      x_a_u_s3       <= 8'd0;
      x_b_u_s3       <= 8'd0;
      w_u_s3         <= 8'd0;
      en_s3          <= 1'b0;
      clear_s3       <= 1'b0;
    end else begin
      // Packed multiply: both products computed in one DSP operation
      dsp_product_s3 <= a_packed * {17'd0, b_value};  // 25 × 8 unsigned
      // Forward unsigned values for correction stage
      x_a_u_s3       <= x_a_u_s2;
      x_b_u_s3       <= x_b_u_s2;
      w_u_s3         <= w_u_s2;
      en_s3          <= en_s2;
      clear_s3       <= clear_s2;
    end
  end

  // ╔═══════════════════════════════════════════════════════════════╗
  // ║  STAGE 4: Product Extraction + Unsigned→Signed Correction      ║
  // ║                                                                 ║
  // ║  Extract non-overlapping products from packed result:           ║
  // ║    raw_a = P[15:0]   (= x_a_u × w_u, max 65025, 16 bits)     ║
  // ║    raw_b = P[32:17]  (= x_b_u × w_u, max 65025, 16 bits)     ║
  // ║                                                                 ║
  // ║  Correction formula (unsigned-offset trick):                    ║
  // ║    signed_prod = raw_u − 128 × (x_u + w_u) + 16384            ║
  // ║                                                                 ║
  // ║  Proof: a_s × b_s = (a_u−128)(b_u−128)                        ║
  // ║       = a_u×b_u − 128×a_u − 128×b_u + 128²                   ║
  // ║       = a_u×b_u − 128×(a_u + b_u) + 16384                     ║
  // ╚═══════════════════════════════════════════════════════════════╝
  logic signed [31:0] prod_a_s4, prod_b_s4;
  logic               en_s4, clear_s4;

  // Combinational extraction from packed product
  wire [15:0] raw_a = dsp_product_s3[15:0];   // Lower product
  wire [15:0] raw_b = dsp_product_s3[32:17];  // Upper product

  // Combinational correction for lane A
  wire signed [31:0] corr_a = $signed({1'b0, raw_a})
                             - $signed(32'd128 * {23'd0, x_a_u_s3} + 32'd128 * {23'd0, w_u_s3})
                             + 32'sd16384;

  // Combinational correction for lane B
  wire signed [31:0] corr_b = $signed({1'b0, raw_b})
                             - $signed(32'd128 * {23'd0, x_b_u_s3} + 32'd128 * {23'd0, w_u_s3})
                             + 32'sd16384;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      prod_a_s4 <= 32'sd0;
      prod_b_s4 <= 32'sd0;
      en_s4     <= 1'b0;
      clear_s4  <= 1'b0;
    end else begin
      prod_a_s4 <= corr_a;
      prod_b_s4 <= corr_b;
      en_s4     <= en_s3;
      clear_s4  <= clear_s3;
    end
  end

  // ╔═══════════════════════════════════════════════════════════════╗
  // ║  STAGE 5: Accumulate + Output Register                         ║
  // ║                                                                 ║
  // ║  Three behaviors on each clock edge:                            ║
  // ║    en=1, clear=1 → Start fresh: acc = product (discard old)    ║
  // ║    en=1, clear=0 → Accumulate:  acc = acc + product            ║
  // ║    en=0           → Hold:       acc = acc (unchanged)           ║
  // ║                                                                 ║
  // ║  The output psum_a / psum_b are directly the accumulator FFs.   ║
  // ╚═══════════════════════════════════════════════════════════════╝
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      psum_a <= 32'sd0;
      psum_b <= 32'sd0;
    end else if (en_s4) begin
      if (clear_s4) begin
        // Start new accumulation with first product
        psum_a <= prod_a_s4;
        psum_b <= prod_b_s4;
      end else begin
        // Add to running sum
        psum_a <= psum_a + prod_a_s4;
        psum_b <= psum_b + prod_b_s4;
      end
    end
    // else: en_s4=0 → hold current value (no change)
  end

  // synthesis translate_off
`ifdef RTL_TRACE
  always @(posedge clk) begin
    if (rst_n && TRACE_EN && en)
      rtl_trace_pkg::rtl_trace_line("S1_DSP",
        $sformatf("xa=%0d xb=%0d w=%0d en=1 clr=%b | acc_a=%0d acc_b=%0d",
                  x_a, x_b, w, clear, psum_a, psum_b));
  end
`endif
  // synthesis translate_on

endmodule
