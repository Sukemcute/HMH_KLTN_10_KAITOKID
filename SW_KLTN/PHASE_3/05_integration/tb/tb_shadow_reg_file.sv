`timescale 1ns/1ps
module tb_shadow_reg_file;
  import accel_pkg::*;
  import desc_pkg::*;

  logic clk, rst_n, load;
  tile_desc_t tile_desc;
  layer_desc_t layer_desc;
  post_profile_t post_profile;
  router_profile_t router_profile;

  pe_mode_e o_mode;
  logic [8:0] o_cin_tile, o_cout_tile;
  logic [9:0] o_hin, o_win, o_hout, o_wout;
  logic [3:0] o_kh, o_kw;
  logic [2:0] o_sh, o_sw;
  logic [3:0] o_pad_top, o_pad_bot, o_pad_left, o_pad_right;
  logic [3:0] o_q_in, o_q_out, o_num_cin_pass, o_num_k_pass;
  logic [15:0] o_tile_flags;
  post_profile_t o_post;
  router_profile_t o_router;

  shadow_reg_file dut (.*);

  initial clk = 0;
  always #5 clk = ~clk;
  int err_cnt = 0;

  initial begin
    $display("╔══════════════════════════════════════════════════════╗");
    $display("║ TESTBENCH: shadow_reg_file                          ║");
    $display("╚══════════════════════════════════════════════════════╝");
    rst_n = 0; load = 0; tile_desc = '0; layer_desc = '0;
    post_profile = '0; router_profile = '0;
    @(negedge clk); @(negedge clk); rst_n = 1; @(negedge clk);

    // TEST 1: Load and verify capture
    $display("=== TEST 1: Load descriptor fields ===");
    layer_desc.template_id = PE_RS3;
    layer_desc.tile_cin = 8'd64;
    layer_desc.tile_cout = 8'd32;
    layer_desc.hin = 10'd80;
    layer_desc.win = 10'd640;
    layer_desc.kh = 4'd3;
    layer_desc.kw = 4'd3;
    layer_desc.sh = 3'd1;
    layer_desc.sw = 3'd1;
    layer_desc.num_cin_pass = 4'd2;
    layer_desc.num_k_pass = 4'd1;
    tile_desc.tile_flags = 16'h0041; // FIRST_TILE + BARRIER_BEFORE
    @(negedge clk);
    load = 1;
    @(negedge clk);
    load = 0;
    @(posedge clk); #1;

    if (o_mode != PE_RS3) begin $display("  FAIL: mode"); err_cnt++; end
    if (o_cin_tile != 9'd64) begin $display("  FAIL: cin_tile=%0d", o_cin_tile); err_cnt++; end
    if (o_cout_tile != 9'd32) begin $display("  FAIL: cout_tile=%0d", o_cout_tile); err_cnt++; end
    if (o_hin != 10'd80) begin $display("  FAIL: hin"); err_cnt++; end
    if (o_kh != 4'd3) begin $display("  FAIL: kh"); err_cnt++; end
    if (o_num_cin_pass != 4'd2) begin $display("  FAIL: num_cin_pass"); err_cnt++; end
    if (o_tile_flags != 16'h0041) begin $display("  FAIL: tile_flags"); err_cnt++; end
    if (err_cnt == 0) $display("  TEST 1 PASSED");

    // TEST 2: Values hold when load=0
    $display("=== TEST 2: Hold values after load ===");
    layer_desc.template_id = PE_OS1;
    layer_desc.tile_cin = 8'd128;
    @(negedge clk); @(negedge clk); @(posedge clk); #1;
    if (o_mode != PE_RS3 || o_cin_tile != 9'd64) begin
      $display("  FAIL: values changed without load pulse");
      err_cnt++;
    end else $display("  TEST 2 PASSED");

    $display("════════════════════════════════════════════════════");
    if (err_cnt == 0) $display("★ ALL TESTS PASSED ★");
    else              $display("✗ TOTAL FAILURES: %0d", err_cnt);
    $display("════════════════════════════════════════════════════");
    $finish;
  end
endmodule
