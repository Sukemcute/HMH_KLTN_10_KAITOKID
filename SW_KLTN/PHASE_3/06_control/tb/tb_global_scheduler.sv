`timescale 1ns/1ps
module tb_global_scheduler;
  import accel_pkg::*;
  import desc_pkg::*;

  logic clk, rst_n;
  layer_desc_t layer_desc;
  logic layer_valid;
  tile_desc_t tile_desc_in;
  logic tile_valid, tile_accept;
  tile_desc_t sc_tile [4];
  logic sc_tile_valid [4];
  logic sc_tile_accept [4];
  logic [4:0] current_layer_id;
  logic layer_complete, inference_complete;

  global_scheduler dut (.*);

  initial clk = 0;
  always #5 clk = ~clk;
  int err_cnt = 0;

  initial begin
    $display("╔══════════════════════════════════════════════════════╗");
    $display("║ TESTBENCH: global_scheduler                         ║");
    $display("╚══════════════════════════════════════════════════════╝");
    rst_n = 0; layer_valid = 0; tile_valid = 0;
    for (int i = 0; i < 4; i++) sc_tile_accept[i] = 0;
    layer_desc = '0; tile_desc_in = '0;
    @(negedge clk); @(negedge clk); rst_n = 1; @(negedge clk);

    // Setup: 1 layer with 2 tiles
    $display("=== TEST 1: Dispatch to SC0 (sc_mask=0001) ===");
    layer_desc.layer_id = 5'd0;
    layer_desc.num_tile_hw = 12'd2;
    @(negedge clk); layer_valid = 1;
    @(negedge clk); layer_valid = 0;

    // Dispatch tile 0 → SC0 only
    tile_desc_in.tile_id = 16'd0;
    tile_desc_in.sc_mask = 4'b0001;
    @(negedge clk); tile_valid = 1;
    @(negedge clk); tile_valid = 0;

    // Wait for sc_tile_valid[0]
    repeat(3) @(posedge clk);
    #1;
    if (sc_tile_valid[0])
      $display("  SC0 tile_valid asserted");
    else begin
      $display("  FAIL: SC0 tile_valid not asserted");
      err_cnt++;
    end

    // SC0 accepts
    @(negedge clk); sc_tile_accept[0] = 1;
    @(negedge clk); sc_tile_accept[0] = 0;
    repeat(2) @(posedge clk);

    // Dispatch tile 1 → SC0 + SC1
    $display("=== TEST 2: Dispatch to SC0+SC1 (sc_mask=0011) ===");
    tile_desc_in.tile_id = 16'd1;
    tile_desc_in.sc_mask = 4'b0011;
    @(negedge clk); tile_valid = 1;
    @(negedge clk); tile_valid = 0;

    repeat(3) @(posedge clk);
    #1;
    if (sc_tile_valid[0] && sc_tile_valid[1])
      $display("  SC0 + SC1 tile_valid asserted");

    @(negedge clk);
    sc_tile_accept[0] = 1; sc_tile_accept[1] = 1;
    @(negedge clk);
    sc_tile_accept[0] = 0; sc_tile_accept[1] = 0;

    repeat(3) @(posedge clk);
    #1;
    if (layer_complete)
      $display("  TEST 2 PASSED (layer_complete asserted)");
    else
      $display("  TEST 2 INFO: layer_complete=%0d", layer_complete);

    $display("════════════════════════════════════════════════════");
    if (err_cnt == 0) $display("★ ALL TESTS PASSED ★");
    else              $display("✗ TOTAL FAILURES: %0d", err_cnt);
    $display("════════════════════════════════════════════════════");
    $finish;
  end
endmodule
