`timescale 1ns/1ps

// Post-Processing Unit: 4-stage pipeline
// Stage 1: Bias Add (INT32 + INT32 → INT32)
// Stage 2: Fixed-Point Requant (× M_int >>> shift, half_up rounding)
// Stage 3: Activation (SiLU LUT / ReLU / None / Clamp)
// Stage 4: Clamp + Element-wise Add (skip connection)
module ppu #(
  parameter int LANES = 32
)(
  input  logic                  clk,
  input  logic                  rst_n,
  input  logic                  en,

  input  desc_pkg::post_profile_t cfg_post,
  input  accel_pkg::pe_mode_e     cfg_mode,

  // PSUM input from PE cluster
  input  logic signed [31:0]   psum_in [LANES],
  input  logic                  psum_valid,

  // Per-channel quantization parameters
  input  logic signed [31:0]   bias_val [LANES],
  input  logic signed [31:0]   m_int    [LANES],
  input  logic [5:0]           shift    [LANES],
  input  logic signed [7:0]    zp_out,

  // SiLU LUT
  input  logic signed [7:0]    silu_lut_data [256],

  // Element-wise add (skip connection)
  input  logic signed [7:0]    ewise_in [LANES],
  input  logic                  ewise_valid,

  output logic signed [7:0]    act_out [LANES],
  output logic                  act_valid
);
  import accel_pkg::*;
  import desc_pkg::*;

  // Pipeline valid signals
  logic valid_s1, valid_s2, valid_s3;
  logic ewise_valid_s1, ewise_valid_s2, ewise_valid_s3;
  logic signed [7:0] ewise_s1 [LANES];
  logic signed [7:0] ewise_s2 [LANES];
  logic signed [7:0] ewise_s3 [LANES];

  // ═══════════ Stage 1: Bias Add ═══════════
  logic signed [31:0] biased_s1 [LANES];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int l = 0; l < LANES; l++)
        biased_s1[l] <= '0;
      valid_s1 <= 1'b0;
      ewise_valid_s1 <= 1'b0;
      for (int l = 0; l < LANES; l++)
        ewise_s1[l] <= '0;
    end else begin
      valid_s1 <= psum_valid & en;
      ewise_valid_s1 <= ewise_valid & en;
      if (psum_valid & en) begin
        for (int l = 0; l < LANES; l++) begin
          if (cfg_post.bias_en)
            biased_s1[l] <= psum_in[l] + bias_val[l];
          else
            biased_s1[l] <= psum_in[l];
        end
      end
      if (ewise_valid & en) begin
        for (int l = 0; l < LANES; l++)
          ewise_s1[l] <= ewise_in[l];
      end
    end
  end

  // ═══════════ Stage 2: Fixed-Point Requant ═══════════
  // y_raw = (biased × m_int + rounding) >>> shift
  logic signed [15:0] y_raw_s2 [LANES];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int l = 0; l < LANES; l++)
        y_raw_s2[l] <= '0;
      valid_s2 <= 1'b0;
      ewise_valid_s2 <= 1'b0;
      for (int l = 0; l < LANES; l++)
        ewise_s2[l] <= '0;
    end else begin
      valid_s2 <= valid_s1;
      ewise_valid_s2 <= ewise_valid_s1;
      if (valid_s1) begin
        for (int l = 0; l < LANES; l++) begin
          automatic logic signed [63:0] mult;
          automatic logic signed [63:0] rounded;
          automatic logic signed [31:0] shifted;
          automatic int sh;

          mult = 64'(biased_s1[l]) * 64'(m_int[l]);

          sh = int'(shift[l]);
          if (sh > 0)
            rounded = mult + (64'sd1 <<< (sh - 1));  // half_up rounding
          else
            rounded = mult;

          shifted = 32'(rounded >>> sh);

          // Clamp to 16-bit for SiLU indexing range
          if (shifted > 32'sd32767)
            y_raw_s2[l] <= 16'sd32767;
          else if (shifted < -32'sd32768)
            y_raw_s2[l] <= -16'sd32768;
          else
            y_raw_s2[l] <= shifted[15:0];
        end
      end
      if (ewise_valid_s1) begin
        for (int l = 0; l < LANES; l++)
          ewise_s2[l] <= ewise_s1[l];
      end
    end
  end

  // ═══════════ Stage 3: Activation Function ═══════════
  logic signed [7:0] y_act_s3 [LANES];

  // SiLU LUT index calculation
  function automatic logic [7:0] silu_idx(input logic signed [15:0] val);
    automatic int idx_int;
    idx_int = int'(val) + 128;
    if (idx_int < 0) return 8'd0;
    if (idx_int > 255) return 8'd255;
    return idx_int[7:0];
  endfunction

  // Saturating clamp to INT8
  function automatic logic signed [7:0] clamp_int8(input logic signed [15:0] val);
    if (val > 16'sd127) return 8'sd127;
    if (val < -16'sd128) return -8'sd128;
    return val[7:0];
  endfunction

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int l = 0; l < LANES; l++)
        y_act_s3[l] <= '0;
      valid_s3 <= 1'b0;
      ewise_valid_s3 <= 1'b0;
      for (int l = 0; l < LANES; l++)
        ewise_s3[l] <= '0;
    end else begin
      valid_s3 <= valid_s2;
      ewise_valid_s3 <= ewise_valid_s2;
      if (valid_s2) begin
        for (int l = 0; l < LANES; l++) begin
          case (cfg_post.act_mode)
            ACT_SILU: begin
              y_act_s3[l] <= silu_lut_data[silu_idx(y_raw_s2[l])];
            end
            ACT_RELU: begin
              y_act_s3[l] <= (y_raw_s2[l] > 0) ? clamp_int8(y_raw_s2[l]) : 8'sd0;
            end
            ACT_CLAMP: begin
              y_act_s3[l] <= clamp_int8(y_raw_s2[l]);
            end
            default: begin  // ACT_NONE
              y_act_s3[l] <= clamp_int8(y_raw_s2[l]);
            end
          endcase
        end
      end
      if (ewise_valid_s2) begin
        for (int l = 0; l < LANES; l++)
          ewise_s3[l] <= ewise_s2[l];
      end
    end
  end

  // ═══════════ Stage 4: Clamp + Element-wise Add ═══════════
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int l = 0; l < LANES; l++)
        act_out[l] <= '0;
      act_valid <= 1'b0;
    end else begin
      act_valid <= valid_s3;
      if (valid_s3) begin
        for (int l = 0; l < LANES; l++) begin
          automatic logic signed [15:0] y_add;

          if (cfg_post.ewise_en && ewise_valid_s3)
            y_add = 16'(y_act_s3[l]) + 16'(ewise_s3[l]);
          else
            y_add = 16'(y_act_s3[l]);

          // Add output zero-point and final clamp
          y_add = y_add + 16'(zp_out);

          if (y_add > 16'sd127)
            act_out[l] <= 8'sd127;
          else if (y_add < -16'sd128)
            act_out[l] <= -8'sd128;
          else
            act_out[l] <= y_add[7:0];
        end
      end
    end
  end

endmodule
