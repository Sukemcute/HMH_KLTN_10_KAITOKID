// Pipelined max comparator tree for MAXPOOL_5x5.
// Finds maximum of 25 INT8 inputs per lane.
// 5-stage pipeline: 25→13→7→4→2→1
`timescale 1ns/1ps
module comparator_tree #(
  parameter int LANES      = 32,
  parameter int NUM_INPUTS = 25
)(
  input  logic               clk,
  input  logic               rst_n,
  input  logic               en,
  input  logic signed [7:0]  data_in [NUM_INPUTS][LANES],
  output logic signed [7:0]  max_out [LANES],
  output logic               max_valid
);

  // Valid pipeline
  logic [5:0] valid_sr;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      valid_sr <= '0;
    else
      valid_sr <= {valid_sr[4:0], en};
  end

  assign max_valid = valid_sr[5];

  // Helper function
  function automatic logic signed [7:0] max2(
    input logic signed [7:0] a, b
  );
    return (a >= b) ? a : b;
  endfunction

  // Stage 1: 25 → 13 (12 pairs + 1 passthrough)
  logic signed [7:0] s1 [13][LANES];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < 13; i++)
        for (int l = 0; l < LANES; l++)
          s1[i][l] <= '0;
    end else if (en) begin
      for (int l = 0; l < LANES; l++) begin
        for (int i = 0; i < 12; i++)
          s1[i][l] <= max2(data_in[2*i][l], data_in[2*i+1][l]);
        s1[12][l] <= data_in[24][l];
      end
    end
  end

  // Stage 2: 13 → 7 (6 pairs + 1 passthrough)
  logic signed [7:0] s2 [7][LANES];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < 7; i++)
        for (int l = 0; l < LANES; l++)
          s2[i][l] <= '0;
    end else if (valid_sr[0]) begin
      for (int l = 0; l < LANES; l++) begin
        for (int i = 0; i < 6; i++)
          s2[i][l] <= max2(s1[2*i][l], s1[2*i+1][l]);
        s2[6][l] <= s1[12][l];
      end
    end
  end

  // Stage 3: 7 → 4 (3 pairs + 1 passthrough)
  logic signed [7:0] s3 [4][LANES];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < 4; i++)
        for (int l = 0; l < LANES; l++)
          s3[i][l] <= '0;
    end else if (valid_sr[1]) begin
      for (int l = 0; l < LANES; l++) begin
        for (int i = 0; i < 3; i++)
          s3[i][l] <= max2(s2[2*i][l], s2[2*i+1][l]);
        s3[3][l] <= s2[6][l];
      end
    end
  end

  // Stage 4: 4 → 2
  logic signed [7:0] s4 [2][LANES];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < 2; i++)
        for (int l = 0; l < LANES; l++)
          s4[i][l] <= '0;
    end else if (valid_sr[2]) begin
      for (int l = 0; l < LANES; l++) begin
        s4[0][l] <= max2(s3[0][l], s3[1][l]);
        s4[1][l] <= max2(s3[2][l], s3[3][l]);
      end
    end
  end

  // Stage 5: 2 → 1
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int l = 0; l < LANES; l++)
        max_out[l] <= '0;
    end else if (valid_sr[3]) begin
      for (int l = 0; l < LANES; l++)
        max_out[l] <= max2(s4[0][l], s4[1][l]);
    end
  end

endmodule
