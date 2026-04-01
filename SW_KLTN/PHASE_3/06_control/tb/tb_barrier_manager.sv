// Testbench for barrier_manager
// Test: YOLOv10n 4-barrier scenario
// barrier[0]: L6→L12, barrier[1]: L4→L15, barrier[2]: L13→L18, barrier[3]: L8→L21
`timescale 1ns/1ps

module tb_barrier_manager;

  logic        clk, rst_n, clear_all;
  logic        signal_valid, wait_valid;
  logic [4:0]  signal_barrier_id, wait_barrier_id;
  logic        wait_grant;
  logic [31:0] scoreboard;

  barrier_manager #(.NUM_BARRIERS(32)) uut (.*);

  always #2.5 clk = ~clk;
  int fail_count = 0;

  task automatic reset();
    rst_n = 0; clear_all = 0;
    signal_valid = 0; wait_valid = 0;
    signal_barrier_id = 0; wait_barrier_id = 0;
    repeat(3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
  endtask

  task automatic test_yolov10n_barriers();
    int errors = 0;

    $display("=== TEST: YOLOv10n 4-barrier scenario ===");
    reset();

    // Initially all barriers unsignaled
    // L12 waits for barrier[0] (L6) → should NOT grant
    @(posedge clk);
    wait_valid = 1; wait_barrier_id = 0;
    @(posedge clk);
    if (wait_grant) begin
      $display("  FAIL: barrier[0] granted before L6 done");
      errors++;
    end
    wait_valid = 0;

    // L6 completes → signal barrier[0]
    @(posedge clk);
    signal_valid = 1; signal_barrier_id = 0;
    @(posedge clk);
    signal_valid = 0;
    @(posedge clk);

    if (!scoreboard[0]) begin
      $display("  FAIL: scoreboard[0] not set after signal");
      errors++;
    end

    // L12 waits → should grant now
    @(posedge clk);
    wait_valid = 1; wait_barrier_id = 0;
    @(posedge clk);
    if (!wait_grant) begin
      $display("  FAIL: barrier[0] not granted after L6 done");
      errors++;
    end
    wait_valid = 0;

    // Signal all 4 barriers
    for (int b = 1; b < 4; b++) begin
      @(posedge clk);
      signal_valid = 1; signal_barrier_id = b;
      @(posedge clk);
      signal_valid = 0;
    end
    @(posedge clk);

    if (scoreboard[3:0] !== 4'b1111) begin
      $display("  FAIL: not all 4 barriers set. scoreboard=%b", scoreboard[3:0]);
      errors++;
    end

    // Clear all
    @(posedge clk);
    clear_all = 1;
    @(posedge clk);
    clear_all = 0;
    @(posedge clk);

    if (scoreboard !== 32'd0) begin
      $display("  FAIL: scoreboard not cleared. got=%0h", scoreboard);
      errors++;
    end

    // Verify wait fails after clear
    @(posedge clk);
    wait_valid = 1; wait_barrier_id = 0;
    @(posedge clk);
    if (wait_grant) begin
      $display("  FAIL: barrier granted after clear_all");
      errors++;
    end
    wait_valid = 0;

    if (errors == 0) $display("  TEST PASSED: YOLOv10n barriers correct");
    else $display("  TEST FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  initial begin
    clk = 0;
    $display("\n╔══════════════════════════════════════════════════╗");
    $display("║  TESTBENCH: barrier_manager                     ║");
    $display("╚══════════════════════════════════════════════════╝\n");

    test_yolov10n_barriers();

    $display("\n══════════════════════════════════════════════════");
    if (fail_count == 0) $display("  ★ ALL BARRIER_MANAGER TESTS PASSED ★");
    else $display("  ✗ TOTAL FAILURES: %0d", fail_count);
    $display("══════════════════════════════════════════════════\n");
    $finish;
  end

endmodule
