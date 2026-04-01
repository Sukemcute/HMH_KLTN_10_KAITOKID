// Self-checking testbench for window_gen
// Test 1: K=1 (pass-through)
// Test 2: K=3 (3 consecutive taps)
// Test 3: K=7 (7 consecutive taps)
// Test 4: Flush and restart
`timescale 1ns/1ps

module tb_window_gen;

  localparam int LANES = 32;
  localparam int K_MAX = 7;

  logic              clk, rst_n, flush;
  logic [2:0]        cfg_kw;
  logic              shift_in_valid;
  logic signed [7:0] shift_in [LANES];
  logic              taps_valid;
  logic signed [7:0] taps [K_MAX][LANES];

  window_gen #(.LANES(LANES), .K_MAX(K_MAX)) uut (.*);

  always #2.5 clk = ~clk;
  int fail_count = 0;

  task automatic reset();
    rst_n = 0; flush = 0; shift_in_valid = 0;
    cfg_kw = 3;
    for (int l = 0; l < LANES; l++) shift_in[l] = 0;
    repeat(3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
  endtask

  // Feed a row with identifier 'row_id' into all lanes
  task automatic feed_row(input int row_id);
    @(negedge clk);
    shift_in_valid = 1;
    for (int l = 0; l < LANES; l++)
      shift_in[l] = (row_id * 10 + l) & 8'h7F;  // keep positive for readability
    @(negedge clk);
    shift_in_valid = 0;
  endtask

  // ═══════════ TEST 1: K=1 ═══════════
  task automatic test_k1();
    int errors = 0;

    $display("=== TEST 1: K=1 (pass-through) ===");
    reset();
    cfg_kw = 1;

    // Feed 1 row → should output immediately
    @(negedge clk);
    shift_in_valid = 1;
    for (int l = 0; l < LANES; l++) shift_in[l] = l[7:0];
    @(posedge clk);
    #1;

    if (!taps_valid) begin
      $display("  FAIL: taps_valid not asserted after 1 row (K=1)");
      errors++;
    end

    for (int l = 0; l < LANES; l++) begin
      if (taps[0][l] !== l[7:0]) begin
        $display("  FAIL lane[%0d]: tap[0]=%0d expected=%0d", l, taps[0][l], l);
        errors++;
        if (errors > 5) break;
      end
    end

    @(negedge clk);
    shift_in_valid = 0;

    if (errors == 0) $display("  TEST 1 PASSED");
    else $display("  TEST 1 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  // ═══════════ TEST 2: K=3 ═══════════
  task automatic test_k3();
    int errors = 0;
    logic signed [7:0] expected_rows [3][LANES];

    $display("=== TEST 2: K=3 (3 consecutive taps) ===");
    reset();
    cfg_kw = 3;

    // Feed 3 rows (need K=3 rows before valid)
    for (int r = 0; r < 3; r++) begin
      @(negedge clk);
      shift_in_valid = 1;
      for (int l = 0; l < LANES; l++) begin
        shift_in[l] = (r * 10 + l) & 8'h7F;
        expected_rows[r][l] = (r * 10 + l) & 8'h7F;
      end
    end

    // On the 3rd shift, taps_valid should be asserted.
    @(posedge clk);
    #1;

    if (!taps_valid) begin
      $display("  FAIL: taps_valid not asserted after 3 rows (K=3)");
      errors++;
    end else begin
      // taps[0] = row2 (newest), taps[1] = row1, taps[2] = row0
      for (int l = 0; l < LANES; l++) begin
        automatic logic signed [7:0] exp0 = (2 * 10 + l) & 8'h7F;
        automatic logic signed [7:0] exp1 = (1 * 10 + l) & 8'h7F;
        automatic logic signed [7:0] exp2 = (0 * 10 + l) & 8'h7F;

        if (taps[0][l] !== exp0) begin
          $display("  FAIL lane[%0d]: tap[0]=%0d expected=%0d (row2)", l, taps[0][l], exp0);
          errors++;
        end
        if (taps[1][l] !== exp1) begin
          $display("  FAIL lane[%0d]: tap[1]=%0d expected=%0d (row1)", l, taps[1][l], exp1);
          errors++;
        end
        if (taps[2][l] !== exp2) begin
          $display("  FAIL lane[%0d]: tap[2]=%0d expected=%0d (row0)", l, taps[2][l], exp2);
          errors++;
        end
        if (errors > 10) break;
      end
    end

    @(negedge clk);
    shift_in_valid = 0;

    if (errors == 0) $display("  TEST 2 PASSED");
    else $display("  TEST 2 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  // ═══════════ TEST 3: Flush and restart ═══════════
  task automatic test_flush();
    int errors = 0;

    $display("=== TEST 3: Flush and restart ===");
    reset();
    cfg_kw = 3;

    // Feed 5 rows
    for (int r = 0; r < 5; r++) begin
      @(negedge clk);
      shift_in_valid = 1;
      for (int l = 0; l < LANES; l++) shift_in[l] = r[7:0];
    end
    @(posedge clk);
    shift_in_valid = 0;

    // Flush
    @(negedge clk);
    flush = 1;
    @(negedge clk);
    flush = 0;

    // After flush, feed 1 row → should NOT be valid (need K=3 rows)
    @(negedge clk);
    shift_in_valid = 1;
    for (int l = 0; l < LANES; l++) shift_in[l] = 99;
    @(posedge clk);
    #1;
    @(negedge clk);
    shift_in_valid = 0;

    if (taps_valid) begin
      $display("  FAIL: taps_valid asserted after flush + 1 row (need 3)");
      errors++;
    end

    if (errors == 0) $display("  TEST 3 PASSED");
    else $display("  TEST 3 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  initial begin
    clk = 0;
    $display("\n╔══════════════════════════════════════════════════╗");
    $display("║  TESTBENCH: window_gen                          ║");
    $display("╚══════════════════════════════════════════════════╝\n");

    test_k1();
    test_k3();
    test_flush();

    $display("\n══════════════════════════════════════════════════");
    if (fail_count == 0) $display("  ★ ALL WINDOW_GEN TESTS PASSED ★");
    else $display("  ✗ TOTAL FAILURES: %0d", fail_count);
    $display("══════════════════════════════════════════════════\n");
    $finish;
  end

endmodule
