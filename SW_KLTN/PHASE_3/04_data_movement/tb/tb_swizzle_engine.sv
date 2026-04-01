`timescale 1ns/1ps
module tb_swizzle_engine;
  import accel_pkg::*;
  localparam int LANES = 32;

  logic clk, rst_n, start;
  pe_mode_e mode;
  logic [1:0] cfg_upsample_factor;
  logic [8:0] cfg_concat_ch_offset;
  logic [9:0] cfg_src_h, cfg_src_w, cfg_src_c, cfg_dst_h, cfg_dst_w;
  logic [3:0] cfg_dst_q_in;
  logic [8:0] cfg_dst_cin_tile;

  logic src_rd_en;
  logic [15:0] src_rd_addr;
  logic [1:0] src_rd_bank;
  logic [LANES*8-1:0] src_rd_data;

  logic dst_wr_en;
  logic [15:0] dst_wr_addr;
  logic [1:0] dst_wr_bank;
  logic [LANES*8-1:0] dst_wr_data;
  logic [LANES-1:0] dst_wr_mask;
  logic done;

  swizzle_engine #(.LANES(LANES)) dut (.*);

  initial clk = 0;
  always #5 clk = ~clk;
  int err_cnt = 0;

  // Provide read data on request
  always @(posedge clk) begin
    if (src_rd_en) begin
      for (int l = 0; l < LANES; l++)
        src_rd_data[l*8 +: 8] <= 8'(src_rd_addr[7:0] + l[7:0]);
    end
  end

  initial begin
    $display("╔══════════════════════════════════════════════════════╗");
    $display("║ TESTBENCH: swizzle_engine                           ║");
    $display("╚══════════════════════════════════════════════════════╝");
    rst_n = 0; start = 0; mode = PE_RS3;
    cfg_upsample_factor = 0;
    cfg_concat_ch_offset = 0;
    @(negedge clk); @(negedge clk); rst_n = 1; @(negedge clk);

    // TEST 1: Normal swizzle (no upsample)
    $display("=== TEST 1: Normal pass-through swizzle ===");
    cfg_src_h = 10'd2; cfg_src_w = 10'd32; cfg_src_c = 10'd2;
    cfg_dst_h = 10'd2; cfg_dst_w = 10'd32;
    cfg_dst_q_in = 4'd4; cfg_dst_cin_tile = 9'd32;
    cfg_upsample_factor = 2'd0;
    @(negedge clk);
    start = 1; @(negedge clk); start = 0;

    // Wait for done
    while (!done) @(posedge clk);
    $display("  TEST 1 PASSED (done asserted after normal swizzle)");

    @(negedge clk); @(negedge clk);

    // TEST 2: Upsample 2x
    $display("=== TEST 2: Upsample 2x ===");
    cfg_src_h = 10'd1; cfg_src_w = 10'd32; cfg_src_c = 10'd1;
    cfg_dst_h = 10'd2; cfg_dst_w = 10'd64;
    cfg_upsample_factor = 2'd1;
    @(negedge clk);
    start = 1; @(negedge clk); start = 0;

    while (!done) @(posedge clk);
    $display("  TEST 2 PASSED (done asserted after upsample)");

    $display("════════════════════════════════════════════════════");
    if (err_cnt == 0) $display("★ ALL TESTS PASSED ★");
    else              $display("✗ TOTAL FAILURES: %0d", err_cnt);
    $display("════════════════════════════════════════════════════");
    $finish;
  end
endmodule
