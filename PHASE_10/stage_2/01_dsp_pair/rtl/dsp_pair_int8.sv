
`timescale 1ns/1ps
module dsp_pair_int8 (
  input  logic              clk,
  input  logic              rst_n,
  input  logic              en,
  input  logic              clear,
  input  logic signed [7:0] x_a,    // activation lane 2i
  input  logic signed [7:0] x_b,    // activation lane 2i+1
  input  logic signed [7:0] w,      // shared weight
  output logic signed [31:0] psum_a,
  output logic signed [31:0] psum_b
);

  // Stage 1: Signed → Unsigned conversion
  logic [7:0] x_a_u_s1, x_b_u_s1, w_u_s1;
  logic       en_s1, clear_s1;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      x_a_u_s1 <= '0;
      x_b_u_s1 <= '0;
      w_u_s1   <= '0;
      en_s1    <= 1'b0;
      clear_s1 <= 1'b0;
    end else begin
      x_a_u_s1 <= x_a + 8'sd128;
      x_b_u_s1 <= x_b + 8'sd128;
      w_u_s1   <= w   + 8'sd128;
      en_s1    <= en;
      clear_s1 <= clear;
    end
  end

  // Stage 2: DSP multiply (behavioral model)
  // Pack: A[24:0] = {x_b_u, 9'b0, x_a_u}, B[17:0] = {10'b0, w_u}
  logic [42:0] dsp_p_s2;
  logic [7:0]  x_a_u_s2, x_b_u_s2, w_u_s2;
  logic        en_s2, clear_s2;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      dsp_p_s2  <= '0;
      x_a_u_s2  <= '0;
      x_b_u_s2  <= '0;
      w_u_s2    <= '0;
      en_s2     <= 1'b0;
      clear_s2  <= 1'b0;
    end else begin
      dsp_p_s2  <= {x_b_u_s1, 9'b0, x_a_u_s1} * {10'b0, w_u_s1};
      x_a_u_s2  <= x_a_u_s1;
      x_b_u_s2  <= x_b_u_s1;
      w_u_s2    <= w_u_s1;
      en_s2     <= en_s1;
      clear_s2  <= clear_s1;
    end
  end

  // Stage 3: Extract products + unsigned→signed correction
  logic signed [31:0] prod_a_s3, prod_b_s3;
  logic               en_s3, clear_s3;

  wire [15:0] raw_a = dsp_p_s2[15:0];
  wire [15:0] raw_b = dsp_p_s2[32:17];

  // Correction: signed_prod = raw_u - 128*(x_u + w_u) + 16384
  wire signed [31:0] corr_a = $signed({1'b0, raw_a})
                             - $signed(32'(128) * 32'({1'b0, x_a_u_s2} + {1'b0, w_u_s2}))
                             + 32'sd16384;
  wire signed [31:0] corr_b = $signed({1'b0, raw_b})
                             - $signed(32'(128) * 32'({1'b0, x_b_u_s2} + {1'b0, w_u_s2}))
                             + 32'sd16384;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      prod_a_s3 <= '0;
      prod_b_s3 <= '0;
      en_s3     <= 1'b0;
      clear_s3  <= 1'b0;
    end else begin
      prod_a_s3 <= corr_a;
      prod_b_s3 <= corr_b;
      en_s3     <= en_s2;
      clear_s3  <= clear_s2;
    end
  end

  // Stage 4: Accumulate
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      psum_a <= '0;
      psum_b <= '0;
    end else if (en_s3) begin
      if (clear_s3) begin
        psum_a <= prod_a_s3;
        psum_b <= prod_b_s3;
      end else begin
        psum_a <= psum_a + prod_a_s3;
        psum_b <= psum_b + prod_b_s3;
      end
    end
  end

endmodule
