// ============================================================================
// Module : comparator_tree
// Project: YOLOv10n INT8 Accelerator — V4-VC707
// Description:
//   Pipelined max-comparator tree for MAXPOOL 5×5 (P3: MAXPOOL_5x5).
//   Finds the maximum of 25 signed INT8 inputs per lane.
//   Used ONLY in PE_MP5 mode (SPPF Layer 9).
//
//   5-stage binary reduction pipeline:
//     Stage 1: 25 → 13  (12 pair comparisons + 1 passthrough)
//     Stage 2: 13 → 7   (6 pair comparisons + 1 passthrough)
//     Stage 3:  7 → 4   (3 pair comparisons + 1 passthrough)
//     Stage 4:  4 → 2   (2 pair comparisons)
//     Stage 5:  2 → 1   (1 final comparison)
//
//   CRITICAL: Uses SIGNED comparison (a >= b) for INT8 domain.
//             Unsigned comparison would give wrong results!
//
//   Latency: 5 clock cycles (pipeline fill) + sustained 1 result/cycle
//   Resources: ~500 LUT + ~320 FF per instance (20 lanes)
//
// Instances: 1 per PE cluster × 16 subclusters = 16 total
// ============================================================================
`timescale 1ns / 1ps

module comparator_tree
  import accel_pkg::*;
#(
  parameter int LANES      = accel_pkg::LANES,  // 20
  parameter int NUM_INPUTS = 25                   // 5 × 5 kernel
)(
  input  logic              clk,
  input  logic              rst_n,

  // — Control —
  input  logic              valid_in,

  // — Input: 25 values per lane (5×5 pooling window) —
  input  logic signed [7:0] data_in [NUM_INPUTS][LANES],

  // — Output: maximum value per lane —
  output logic signed [7:0] max_out [LANES],
  output logic              valid_out
);

  // ════════════════════════════════════════════════════════════════
  //  VALID PIPELINE — tracks data through 5 stages
  // ════════════════════════════════════════════════════════════════
  logic [5:0] valid_sr;  // 6 bits: [0]=after S1, ..., [4]=after S5

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      valid_sr <= 6'd0;
    else
      valid_sr <= {valid_sr[4:0], valid_in};
  end

  assign valid_out = valid_sr[5];

  // ════════════════════════════════════════════════════════════════
  //  SIGNED MAX-OF-TWO FUNCTION
  //  CRITICAL: Must be SIGNED comparison for INT8 correctness.
  //  e.g., -1 (0xFF) must be > -128 (0x80).
  //  Unsigned comparison would incorrectly say 0x80 > 0xFF.
  // ════════════════════════════════════════════════════════════════
  function automatic logic signed [7:0] max2(
    input logic signed [7:0] a,
    input logic signed [7:0] b
  );
    return (a >= b) ? a : b;
  endfunction

  // ════════════════════════════════════════════════════════════════
  //  STAGE 1: 25 → 13 (12 pairs + 1 passthrough)
  //  Inputs 0..23 form 12 pairs; input 24 passes through.
  // ════════════════════════════════════════════════════════════════
  logic signed [7:0] s1 [13][LANES];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < 13; i++)
        for (int l = 0; l < LANES; l++)
          s1[i][l] <= 8'sd0;
    end else if (valid_in) begin
      for (int l = 0; l < LANES; l++) begin
        // 12 pair comparisons: max(0,1), max(2,3), ..., max(22,23)
        for (int i = 0; i < 12; i++)
          s1[i][l] <= max2(data_in[2*i][l], data_in[2*i + 1][l]);
        // 1 passthrough: input 24 (the odd one out)
        s1[12][l] <= data_in[24][l];
      end
    end
  end

  // ════════════════════════════════════════════════════════════════
  //  STAGE 2: 13 → 7 (6 pairs + 1 passthrough)
  //  s1[0..11] form 6 pairs; s1[12] passes through.
  // ════════════════════════════════════════════════════════════════
  logic signed [7:0] s2 [7][LANES];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < 7; i++)
        for (int l = 0; l < LANES; l++)
          s2[i][l] <= 8'sd0;
    end else if (valid_sr[0]) begin
      for (int l = 0; l < LANES; l++) begin
        // 6 pair comparisons
        for (int i = 0; i < 6; i++)
          s2[i][l] <= max2(s1[2*i][l], s1[2*i + 1][l]);
        // 1 passthrough
        s2[6][l] <= s1[12][l];
      end
    end
  end

  // ════════════════════════════════════════════════════════════════
  //  STAGE 3: 7 → 4 (3 pairs + 1 passthrough)
  //  s2[0..5] form 3 pairs; s2[6] passes through.
  // ════════════════════════════════════════════════════════════════
  logic signed [7:0] s3 [4][LANES];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < 4; i++)
        for (int l = 0; l < LANES; l++)
          s3[i][l] <= 8'sd0;
    end else if (valid_sr[1]) begin
      for (int l = 0; l < LANES; l++) begin
        // 3 pair comparisons
        for (int i = 0; i < 3; i++)
          s3[i][l] <= max2(s2[2*i][l], s2[2*i + 1][l]);
        // 1 passthrough
        s3[3][l] <= s2[6][l];
      end
    end
  end

  // ════════════════════════════════════════════════════════════════
  //  STAGE 4: 4 → 2 (2 pairs, no passthrough)
  // ════════════════════════════════════════════════════════════════
  logic signed [7:0] s4 [2][LANES];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < 2; i++)
        for (int l = 0; l < LANES; l++)
          s4[i][l] <= 8'sd0;
    end else if (valid_sr[2]) begin
      for (int l = 0; l < LANES; l++) begin
        s4[0][l] <= max2(s3[0][l], s3[1][l]);
        s4[1][l] <= max2(s3[2][l], s3[3][l]);
      end
    end
  end

  // ════════════════════════════════════════════════════════════════
  //  STAGE 5: 2 → 1 (final comparison)
  // ════════════════════════════════════════════════════════════════
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int l = 0; l < LANES; l++)
        max_out[l] <= 8'sd0;
    end else if (valid_sr[3]) begin
      for (int l = 0; l < LANES; l++)
        max_out[l] <= max2(s4[0][l], s4[1][l]);
    end
  end

  // synthesis translate_off
`ifdef RTL_TRACE
  always @(posedge clk) begin
    if (rst_n && valid_in)
      rtl_trace_pkg::rtl_trace_line("S1_CMPIN",
        $sformatf("d0_0=%0d d0_1=%0d", data_in[0][0], data_in[0][1]));
    if (rst_n && valid_out)
      rtl_trace_pkg::rtl_trace_line("S1_CMPOUT",
        $sformatf("max0=%0d max1=%0d", max_out[0], max_out[1]));
  end
`endif
  // synthesis translate_on

endmodule
