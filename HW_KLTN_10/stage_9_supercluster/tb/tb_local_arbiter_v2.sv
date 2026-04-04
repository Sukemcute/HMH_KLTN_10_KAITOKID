// ============================================================================
// Testbench: tb_local_arbiter_v2 — Stage 9.1
// Tests: Normal rotation, DMA grant arbitration, stall handling.
// ============================================================================
`timescale 1ns / 1ps

module tb_local_arbiter_v2;
  import accel_pkg::*;
  import desc_pkg::*;

  localparam int NS = N_SUBS_PER_SC;

  logic          clk, rst_n;
  logic          tile_valid;
  layer_desc_t   layer_desc_in;
  tile_desc_t    tile_desc_in;
  logic          tile_accept;

  logic          sub_tile_valid [NS];
  layer_desc_t   sub_layer_desc [NS];
  tile_desc_t    sub_tile_desc  [NS];
  logic          sub_tile_accept[NS];
  logic          sub_tile_done  [NS];

  logic [1:0]    dma_grant_id;
  logic          dma_fill_grant, dma_drain_grant;
  sub_role_e     sub_roles [NS];
  logic          all_idle;

  local_arbiter_v2 #(.N_SUBS(NS)) u_dut (.*);

  int pass_cnt = 0, fail_cnt = 0;
  task automatic chk(input string t, input logic ok);
    if (ok) begin pass_cnt++; $display("[PASS] %s", t); end
    else begin fail_cnt++; $display("[FAIL] %s", t); end
  endtask

  initial begin clk = 0; forever #2 clk = ~clk; end

  task automatic do_reset();
    rst_n <= 1'b0;
    tile_valid <= 1'b0;
    for (int i = 0; i < NS; i++) begin
      sub_tile_accept[i] <= 1'b0;
      sub_tile_done[i]   <= 1'b0;
    end
    repeat (5) @(posedge clk);
    rst_n <= 1'b1;
    repeat (2) @(posedge clk);
  endtask

  // ── Test 9.1.1: Normal rotation ──
  task automatic test_9_1_1();
    $display("\n=== 9.1.1 Normal rotation ===");
    do_reset();
    chk("9.1.1 all_idle at start", all_idle);

    // Send tile 0
    layer_desc_in <= '0; layer_desc_in.layer_id <= 5'd0;
    tile_desc_in <= '0; tile_desc_in.tile_id <= 16'd1;
    tile_valid <= 1'b1;
    @(posedge clk);
    chk("9.1.1 tile_accept asserted", tile_accept);
    tile_valid <= 1'b0;
    @(posedge clk);

    // Sub3 (fill_ptr) accepts descriptor
    sub_tile_accept[3] <= 1'b1;
    @(posedge clk);
    sub_tile_accept[3] <= 1'b0;
    repeat (2) @(posedge clk);
    chk("9.1.1 sub3 now COMPUTE", sub_roles[3] == ROLE_COMPUTE);

    // Sub3 finishes compute → becomes FILL
    sub_tile_done[3] <= 1'b1;
    @(posedge clk);
    sub_tile_done[3] <= 1'b0;
    repeat (2) @(posedge clk);
    chk("9.1.1 sub3 rotated to FILL", sub_roles[3] == ROLE_FILL);
  endtask

  // ── Test 9.1.2: DMA grant ──
  task automatic test_9_1_2();
    $display("\n=== 9.1.2 DMA grant arbitration ===");
    do_reset();
    chk("9.1.2 initial DMA grant to sub3", dma_grant_id == 2'd3);
  endtask

  // ── Test 9.1.3: Stall handling ──
  task automatic test_9_1_3();
    $display("\n=== 9.1.3 Stall handling ===");
    do_reset();
    // Don't send any tile → no sub moves from IDLE
    repeat (10) @(posedge clk);
    chk("9.1.3 no spurious transitions", all_idle);
  endtask

  initial begin
    $display("=== Stage 9.1 — local_arbiter_v2 Tests ===");
    test_9_1_1();
    test_9_1_2();
    test_9_1_3();
    $display("\n=== 9.1 SUMMARY: PASS=%0d FAIL=%0d ===", pass_cnt, fail_cnt);
    $finish;
  end

  initial begin #500_000; $display("[TIMEOUT]"); $finish; end
endmodule
