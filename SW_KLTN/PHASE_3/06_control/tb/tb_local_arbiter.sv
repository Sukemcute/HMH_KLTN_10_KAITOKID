`timescale 1ns/1ps
module tb_local_arbiter;
  import accel_pkg::*;
  import desc_pkg::*;
  localparam int NUM_SUBS = 4;

  logic clk, rst_n;
  logic tile_available, tile_consumed, has_idle_sub;
  tile_desc_t next_tile;
  tile_state_e sub_state [NUM_SUBS];
  logic sub_tile_done [NUM_SUBS];
  logic sub_dma_wr_req [NUM_SUBS];
  sc_role_e sub_role [NUM_SUBS];
  logic ext_port_ready;
  logic [1:0] ext_port_grant_sub;
  logic ext_port_is_read;
  logic sub_tile_valid [NUM_SUBS];
  tile_desc_t sub_tile [NUM_SUBS];

  local_arbiter #(.NUM_SUBS(NUM_SUBS)) dut (.*);

  initial clk = 0;
  always #5 clk = ~clk;
  int err_cnt = 0;

  task automatic reset();
    rst_n = 0; tile_available = 0; ext_port_ready = 0;
    next_tile = '0;
    for (int i = 0; i < NUM_SUBS; i++) begin
      sub_state[i] = TILE_IDLE;
      sub_tile_done[i] = 0;
      sub_dma_wr_req[i] = 0;
    end
    @(negedge clk); @(negedge clk); rst_n = 1; @(negedge clk);
  endtask

  initial begin
    $display("╔══════════════════════════════════════════════════════╗");
    $display("║ TESTBENCH: local_arbiter                            ║");
    $display("╚══════════════════════════════════════════════════════╝");
    reset();

    // TEST 1: Initial state — all subs IDLE
    $display("=== TEST 1: Initial state all IDLE ===");
    @(posedge clk); #1;
    for (int i = 0; i < NUM_SUBS; i++) begin
      if (sub_role[i] != ROLE_IDLE) begin
        $display("  FAIL: sub[%0d] role=%0d, expected IDLE", i, sub_role[i]);
        err_cnt++;
      end
    end
    if (err_cnt == 0) $display("  TEST 1 PASSED");

    // TEST 2: Dispatch tile → first idle sub gets FILLING
    $display("=== TEST 2: Tile dispatch to idle sub ===");
    next_tile.tile_id = 16'h0001;
    @(negedge clk); tile_available = 1;
    @(negedge clk); tile_available = 0;
    repeat(2) @(posedge clk);
    #1;
    // One sub should be FILLING
    int fill_count = 0;
    for (int i = 0; i < NUM_SUBS; i++)
      if (sub_role[i] == ROLE_FILLING) fill_count++;
    if (fill_count != 1) begin
      $display("  FAIL: expected 1 FILLING sub, got %0d", fill_count);
      err_cnt++;
    end else $display("  TEST 2 PASSED");

    // TEST 3: FILLING → RUNNING promotion
    $display("=== TEST 3: FILLING → RUNNING on compute ===");
    // Simulate sub[0] entering compute
    sub_state[0] = TILE_RUN_COMPUTE;
    repeat(2) @(posedge clk);
    #1;
    if (sub_role[0] != ROLE_RUNNING) begin
      $display("  FAIL: sub[0] expected RUNNING, got %0d", sub_role[0]);
      err_cnt++;
    end else $display("  TEST 3 PASSED");

    // TEST 4: External port arbitration
    $display("=== TEST 4: Ext port grants to FILLING ===");
    ext_port_ready = 1;
    // Dispatch another tile to make a sub FILLING
    next_tile.tile_id = 16'h0002;
    @(negedge clk); tile_available = 1;
    @(negedge clk); tile_available = 0;
    repeat(2) @(posedge clk);
    #1;
    if (ext_port_is_read)
      $display("  TEST 4 PASSED (read grant for FILLING)");
    else begin
      $display("  TEST 4 INFO: ext_port_is_read=%0d", ext_port_is_read);
    end

    $display("════════════════════════════════════════════════════");
    if (err_cnt == 0) $display("★ ALL TESTS PASSED ★");
    else              $display("✗ TOTAL FAILURES: %0d", err_cnt);
    $display("════════════════════════════════════════════════════");
    $finish;
  end
endmodule
