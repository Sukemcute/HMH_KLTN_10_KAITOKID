// Self-checking testbench for comparator_tree (25-input max, 32 lanes)
// Test 1: Max at known positions
// Test 2: All same values
// Test 3: Signed edge cases (-128 vs 127)
// Test 4: Random stress
`timescale 1ns/1ps

module tb_comparator_tree;

  localparam int LANES      = 32;
  localparam int NUM_INPUTS = 25;

  logic               clk, rst_n, en;
  logic signed [7:0]  data_in [NUM_INPUTS][LANES];
  logic signed [7:0]  max_out [LANES];
  logic               max_valid;

  comparator_tree #(.LANES(LANES), .NUM_INPUTS(NUM_INPUTS)) uut (.*);

  always #2.5 clk = ~clk;

  int fail_count = 0;

  task automatic reset();
    rst_n = 0; en = 0;
    for (int i = 0; i < NUM_INPUTS; i++)
      for (int l = 0; l < LANES; l++)
        data_in[i][l] = -128;
    repeat(3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
  endtask

  task automatic wait_result();
    repeat(7) @(posedge clk);
  endtask

  // ═══════════ TEST 1: Max at specific positions ═══════════
  task automatic test_position();
    int errors = 0;

    $display("=== TEST 1: Max at specific positions ===");

    for (int pos = 0; pos < NUM_INPUTS; pos++) begin
      reset();
      for (int i = 0; i < NUM_INPUTS; i++)
        for (int l = 0; l < LANES; l++)
          data_in[i][l] = -50;

      // Place max at position 'pos' for all lanes
      for (int l = 0; l < LANES; l++)
        data_in[pos][l] = 100;

      @(negedge clk); en = 1;
      @(negedge clk); en = 0;
      wait_result();

      for (int l = 0; l < LANES; l++) begin
        if (max_out[l] !== 8'sd100) begin
          $display("  FAIL pos=%0d lane=%0d: got=%0d expected=100", pos, l, max_out[l]);
          errors++;
          if (errors > 5) break;
        end
      end
      if (errors > 5) break;
    end

    if (errors == 0) $display("  TEST 1 PASSED");
    else $display("  TEST 1 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  // ═══════════ TEST 2: All same values ═══════════
  task automatic test_same();
    int errors = 0;

    $display("=== TEST 2: All same values ===");
    reset();

    for (int i = 0; i < NUM_INPUTS; i++)
      for (int l = 0; l < LANES; l++)
        data_in[i][l] = 42;

    @(negedge clk); en = 1;
    @(negedge clk); en = 0;
    wait_result();

    for (int l = 0; l < LANES; l++) begin
      if (max_out[l] !== 8'sd42) begin
        $display("  FAIL lane=%0d: got=%0d expected=42", l, max_out[l]);
        errors++;
      end
    end

    if (errors == 0) $display("  TEST 2 PASSED");
    else $display("  TEST 2 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  // ═══════════ TEST 3: Signed boundaries ═══════════
  task automatic test_signed();
    int errors = 0;

    $display("=== TEST 3: Signed edge cases ===");
    reset();

    for (int i = 0; i < NUM_INPUTS; i++)
      for (int l = 0; l < LANES; l++)
        data_in[i][l] = -128;

    // Place 127 at input[12] for even lanes, -128 stays for odd
    for (int l = 0; l < LANES; l += 2)
      data_in[12][l] = 127;

    @(negedge clk); en = 1;
    @(negedge clk); en = 0;
    wait_result();

    for (int l = 0; l < LANES; l++) begin
      logic signed [7:0] exp_val;
      exp_val = (l % 2 == 0) ? 8'sd127 : -8'sd128;
      if (max_out[l] !== exp_val) begin
        $display("  FAIL lane=%0d: got=%0d expected=%0d", l, max_out[l], exp_val);
        errors++;
      end
    end

    if (errors == 0) $display("  TEST 3 PASSED");
    else $display("  TEST 3 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  // ═══════════ TEST 4: Random stress ═══════════
  task automatic test_random();
    int errors = 0;

    $display("=== TEST 4: Random stress (200 vectors) ===");

    for (int t = 0; t < 200; t++) begin
      reset();
      for (int i = 0; i < NUM_INPUTS; i++)
        for (int l = 0; l < LANES; l++)
          data_in[i][l] = $random;

      @(negedge clk); en = 1;
      @(negedge clk); en = 0;
      wait_result();

      for (int l = 0; l < LANES; l++) begin
        automatic logic signed [7:0] golden = -128;
        for (int i = 0; i < NUM_INPUTS; i++)
          if (data_in[i][l] > golden)
            golden = data_in[i][l];
        if (max_out[l] !== golden) begin
          $display("  FAIL t=%0d lane=%0d: got=%0d expected=%0d", t, l, max_out[l], golden);
          errors++;
          if (errors > 10) return;
        end
      end
    end

    if (errors == 0) $display("  TEST 4 PASSED");
    else $display("  TEST 4 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  initial begin
    clk = 0;
    $display("\n╔══════════════════════════════════════════════════╗");
    $display("║  TESTBENCH: comparator_tree                     ║");
    $display("╚══════════════════════════════════════════════════╝\n");

    test_position();
    test_same();
    test_signed();
    test_random();

    $display("\n══════════════════════════════════════════════════");
    if (fail_count == 0) $display("  ★ ALL COMPARATOR_TREE TESTS PASSED ★");
    else $display("  ✗ TOTAL FAILURES: %0d", fail_count);
    $display("══════════════════════════════════════════════════\n");
    $finish;
  end

endmodule
