// ============================================================================
// Testbench: tb_barrier_manager — Stage 10.3
// Tests: Individual barrier set/grant/clear, all 4 barriers independent,
//        out-of-order signal.
// ============================================================================
`timescale 1ns / 1ps

module tb_barrier_manager;
  import accel_pkg::*;

  logic       clk, rst_n, soft_reset;
  logic [3:0] barrier_signal, barrier_request, barrier_grant, barrier_pending;

  barrier_manager u_dut (.*);

  int pass_cnt = 0, fail_cnt = 0;
  task automatic chk(input string t, input logic ok);
    if (ok) begin pass_cnt++; $display("[PASS] %s", t); end
    else begin fail_cnt++; $display("[FAIL] %s", t); end
  endtask

  initial begin clk = 0; forever #2 clk = ~clk; end

  task automatic do_reset();
    rst_n <= 1'b0; soft_reset <= 1'b0;
    barrier_signal <= 4'd0; barrier_request <= 4'd0;
    repeat (5) @(posedge clk);
    rst_n <= 1'b1;
    repeat (2) @(posedge clk);
  endtask

  // ── Test 10.3.1: barrier_0 ──
  task automatic test_10_3_1();
    $display("\n=== 10.3.1 barrier_0 (L6→L12) ===");
    do_reset();

    chk("10.3.1 no grant before signal", barrier_grant == 4'd0);

    // Producer signals barrier 0
    barrier_signal <= 4'b0001;
    @(posedge clk);
    barrier_signal <= 4'b0000;
    @(posedge clk);
    chk("10.3.1 pending after signal", barrier_pending[0]);

    // Consumer requests barrier 0
    barrier_request <= 4'b0001;
    @(posedge clk);
    chk("10.3.1 grant asserted", barrier_grant[0]);
    barrier_request <= 4'b0000;
    @(posedge clk);
    @(posedge clk);
    chk("10.3.1 cleared after consume", !barrier_pending[0]);
  endtask

  // ── Test 10.3.2: all 4 barriers ──
  task automatic test_10_3_2();
    $display("\n=== 10.3.2 All 4 barriers independent ===");
    do_reset();

    // Signal all 4 barriers
    barrier_signal <= 4'b1111;
    @(posedge clk);
    barrier_signal <= 4'b0000;
    @(posedge clk);
    chk("10.3.2 all pending", barrier_pending == 4'b1111);

    // Request one at a time
    for (int i = 0; i < 4; i++) begin
      barrier_request <= (4'b0001 << i);
      @(posedge clk);
      chk($sformatf("10.3.2 grant[%0d]", i), barrier_grant[i]);
      barrier_request <= 4'd0;
      @(posedge clk);
    end
    @(posedge clk);
    chk("10.3.2 all cleared", barrier_pending == 4'd0);
  endtask

  // ── Test 10.3.3: out-of-order ──
  task automatic test_10_3_3();
    $display("\n=== 10.3.3 Out-of-order signal ===");
    do_reset();

    // Signal barrier_1 before consumer arrives
    barrier_signal <= 4'b0010;
    @(posedge clk);
    barrier_signal <= 4'b0000;
    repeat (10) @(posedge clk);
    chk("10.3.3 pending held", barrier_pending[1]);

    // Consumer arrives later → instant grant
    barrier_request <= 4'b0010;
    @(posedge clk);
    chk("10.3.3 immediate grant", barrier_grant[1]);
    barrier_request <= 4'd0;
    @(posedge clk);
  endtask

  initial begin
    $display("=== Stage 10.3 — barrier_manager Tests ===");
    test_10_3_1();
    test_10_3_2();
    test_10_3_3();
    $display("\n=== 10.3 SUMMARY: PASS=%0d FAIL=%0d ===", pass_cnt, fail_cnt);
    $finish;
  end

  initial begin #500_000; $display("[TIMEOUT]"); $finish; end
endmodule
