// Self-checking testbench for column_reduce
// Test 1: Known values — verify sum of 3 rows
// Test 2: Zero rows — output = 0
// Test 3: Random stress (1000 vectors)
`timescale 1ns/1ps

module tb_column_reduce;
  import accel_pkg::*;

  localparam int LANES   = 32;
  localparam int PE_ROWS = 3;
  localparam int PE_COLS = 4;

  logic               clk, rst_n, en;
  pe_mode_e           mode;
  logic signed [31:0] pe_psum [PE_ROWS][PE_COLS][LANES];
  logic signed [31:0] col_psum [PE_COLS][LANES];
  logic               col_valid;

  column_reduce #(.LANES(LANES), .PE_ROWS(PE_ROWS), .PE_COLS(PE_COLS)) uut (.*);

  always #2.5 clk = ~clk;

  int fail_count = 0;

  task automatic reset();
    rst_n = 0; en = 0; mode = PE_RS3;
    for (int r = 0; r < PE_ROWS; r++)
      for (int c = 0; c < PE_COLS; c++)
        for (int l = 0; l < LANES; l++)
          pe_psum[r][c][l] = 0;
    repeat(3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
  endtask

  // ═══════════ TEST 1: Known values ═══════════
  task automatic test_known();
    int errors = 0;
    logic signed [31:0] expected;

    $display("=== TEST 1: Known values ===");
    reset();

    for (int r = 0; r < PE_ROWS; r++)
      for (int c = 0; c < PE_COLS; c++)
        for (int l = 0; l < LANES; l++)
          pe_psum[r][c][l] = (r + 1) * 1000 + (c + 1) * 100 + l;

    @(posedge clk);
    en = 1;
    @(posedge clk);
    en = 0;
    @(posedge clk);  // output registered

    for (int c = 0; c < PE_COLS; c++) begin
      for (int l = 0; l < LANES; l++) begin
        expected = 0;
        for (int r = 0; r < PE_ROWS; r++)
          expected += (r + 1) * 1000 + (c + 1) * 100 + l;
        if (col_psum[c][l] !== expected) begin
          $display("  FAIL [col=%0d, lane=%0d]: got=%0d expected=%0d",
                   c, l, col_psum[c][l], expected);
          errors++;
        end
      end
    end

    if (errors == 0) $display("  TEST 1 PASSED");
    else $display("  TEST 1 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  // ═══════════ TEST 2: Random stress ═══════════
  task automatic test_random();
    int errors = 0;
    logic signed [31:0] expected;

    $display("=== TEST 2: Random stress (500 vectors) ===");

    for (int t = 0; t < 500; t++) begin
      reset();
      for (int r = 0; r < PE_ROWS; r++)
        for (int c = 0; c < PE_COLS; c++)
          for (int l = 0; l < LANES; l++)
            pe_psum[r][c][l] = $random;

      @(posedge clk);
      en = 1;
      @(posedge clk);
      en = 0;
      @(posedge clk);

      for (int c = 0; c < PE_COLS; c++) begin
        for (int l = 0; l < LANES; l++) begin
          expected = pe_psum[0][c][l] + pe_psum[1][c][l] + pe_psum[2][c][l];
          if (col_psum[c][l] !== expected) begin
            $display("  FAIL t=%0d [c=%0d,l=%0d]: got=%0d exp=%0d",
                     t, c, l, col_psum[c][l], expected);
            errors++;
            if (errors > 10) begin
              $display("  ... stopping after 10 errors");
              return;
            end
          end
        end
      end
    end

    if (errors == 0) $display("  TEST 2 PASSED");
    else $display("  TEST 2 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  initial begin
    clk = 0;
    $display("\n╔══════════════════════════════════════════════════╗");
    $display("║  TESTBENCH: column_reduce                       ║");
    $display("╚══════════════════════════════════════════════════╝\n");

    test_known();
    test_random();

    $display("\n══════════════════════════════════════════════════");
    if (fail_count == 0) $display("  ★ ALL COLUMN_REDUCE TESTS PASSED ★");
    else $display("  ✗ TOTAL FAILURES: %0d", fail_count);
    $display("══════════════════════════════════════════════════\n");
    $finish;
  end

endmodule
