`timescale 1ns/1ps
module tb_column_reduce;

  // ──────────────────── Parameters ────────────────────
  localparam int LANES   = 32;
  localparam int PE_ROWS = 3;
  localparam int PE_COLS = 4;
  localparam int CLK_HP  = 5;  // half-period 5 ns → 10 ns period

  // ──────────────────── DUT signals ────────────────────
  logic               clk;
  logic               rst_n;
  logic               en;
  accel_pkg::pe_mode_e mode;
  logic signed [31:0] pe_psum [PE_ROWS][PE_COLS][LANES];
  logic signed [31:0] col_psum [PE_COLS][LANES];
  logic               col_valid;

  // ──────────────────── DUT ────────────────────
  column_reduce #(
    .LANES   (LANES),
    .PE_ROWS (PE_ROWS),
    .PE_COLS (PE_COLS)
  ) u_dut (
    .clk      (clk),
    .rst_n    (rst_n),
    .en       (en),
    .mode     (mode),
    .pe_psum  (pe_psum),
    .col_psum (col_psum),
    .col_valid(col_valid)
  );

  // ──────────────────── Clock ────────────────────
  initial clk = 0;
  always #CLK_HP clk = ~clk;

  // ──────────────────── Scoreboard ────────────────────
  int total_tests;
  int pass_count;
  int fail_count;

  task automatic report(string name, int ok);
    total_tests++;
    if (ok) begin
      pass_count++;
      $display("[PASS] %s", name);
    end else begin
      fail_count++;
      $display("[FAIL] %s", name);
    end
  endtask

  // ──────────────────── Helper: drive one vector and wait ────────────────────
  task automatic drive_and_wait();
    en <= 1'b1;
    @(posedge clk);
    en <= 1'b0;
    @(posedge clk);  // output available after 1-cycle latency
  endtask

  // ──────────────────── Helper: clear inputs ────────────────────
  task automatic clear_inputs();
    en <= 1'b0;
    for (int r = 0; r < PE_ROWS; r++)
      for (int c = 0; c < PE_COLS; c++)
        for (int l = 0; l < LANES; l++)
          pe_psum[r][c][l] <= 32'sd0;
  endtask

  // ──────────────────── Main test sequence ────────────────────
  initial begin
    total_tests = 0;
    pass_count  = 0;
    fail_count  = 0;
    mode        = accel_pkg::PE_RS3;
    rst_n       = 1'b0;
    en          = 1'b0;
    clear_inputs();

    // Reset
    repeat (4) @(posedge clk);
    rst_n <= 1'b1;
    @(posedge clk);

    // ───── Test 1: All zeros → output all zeros ─────
    begin
      string tname = "T1_all_zeros";
      int ok = 1;
      clear_inputs();
      drive_and_wait();
      for (int c = 0; c < PE_COLS; c++)
        for (int l = 0; l < LANES; l++)
          if (col_psum[c][l] !== 32'sd0) ok = 0;
      report(tname, ok);
    end

    // ───── Test 2: Known values: pe_psum[r][c][l] = r*100 + c*10 + l ─────
    begin
      string tname = "T2_known_values";
      int ok = 1;
      for (int r = 0; r < PE_ROWS; r++)
        for (int c = 0; c < PE_COLS; c++)
          for (int l = 0; l < LANES; l++)
            pe_psum[r][c][l] <= 32'(r * 100 + c * 10 + l);
      drive_and_wait();
      for (int c = 0; c < PE_COLS; c++) begin
        for (int l = 0; l < LANES; l++) begin
          automatic int exp = 0;
          for (int r = 0; r < PE_ROWS; r++)
            exp = exp + (r * 100 + c * 10 + l);
          if (col_psum[c][l] !== 32'(exp)) begin
            $display("  T2 mismatch c=%0d l=%0d: got %0d exp %0d",
                     c, l, col_psum[c][l], exp);
            ok = 0;
          end
        end
      end
      report(tname, ok);
    end

    // ───── Test 3: Signed overflow safety (near INT32 max) ─────
    begin
      string tname = "T3_overflow_safety";
      int ok = 1;
      // Use values that sum to large positive but won't overflow 32-bit signed:
      // max INT32 = 2147483647.  Use ~700M per row so sum ~2.1G < 2.147G
      for (int r = 0; r < PE_ROWS; r++)
        for (int c = 0; c < PE_COLS; c++)
          for (int l = 0; l < LANES; l++)
            pe_psum[r][c][l] <= 32'sd700_000_000;
      drive_and_wait();
      for (int c = 0; c < PE_COLS; c++)
        for (int l = 0; l < LANES; l++) begin
          automatic logic signed [31:0] exp = 32'sd2_100_000_000;
          if (col_psum[c][l] !== exp) begin
            $display("  T3 mismatch c=%0d l=%0d: got %0d exp %0d",
                     c, l, col_psum[c][l], exp);
            ok = 0;
          end
        end
      report(tname, ok);
    end

    // ───── Test 4: Random stress (100 iterations) ─────
    begin
      string tname = "T4_random_stress";
      int ok = 1;
      for (int iter = 0; iter < 100; iter++) begin
        // Build random inputs and compute expected
        automatic int exp_arr [PE_COLS][LANES];
        for (int c = 0; c < PE_COLS; c++)
          for (int l = 0; l < LANES; l++)
            exp_arr[c][l] = 0;
        for (int r = 0; r < PE_ROWS; r++)
          for (int c = 0; c < PE_COLS; c++)
            for (int l = 0; l < LANES; l++) begin
              automatic logic signed [31:0] v = $random;
              pe_psum[r][c][l] <= v;
              exp_arr[c][l] = exp_arr[c][l] + int'(v);
            end
        drive_and_wait();
        for (int c = 0; c < PE_COLS; c++)
          for (int l = 0; l < LANES; l++)
            if (col_psum[c][l] !== 32'(exp_arr[c][l])) begin
              if (ok) // print only first mismatch
                $display("  T4 iter=%0d mismatch c=%0d l=%0d: got %0d exp %0d",
                         iter, c, l, col_psum[c][l], exp_arr[c][l]);
              ok = 0;
            end
      end
      report(tname, ok);
    end

    // ───── Test 5: Enable gating — en=0 → output frozen ─────
    begin
      string tname = "T5_enable_gating";
      int ok = 1;

      // First, drive known values with en=1 to set outputs
      for (int r = 0; r < PE_ROWS; r++)
        for (int c = 0; c < PE_COLS; c++)
          for (int l = 0; l < LANES; l++)
            pe_psum[r][c][l] <= 32'sd42;
      drive_and_wait();

      // Capture current outputs
      automatic logic signed [31:0] saved [PE_COLS][LANES];
      for (int c = 0; c < PE_COLS; c++)
        for (int l = 0; l < LANES; l++)
          saved[c][l] = col_psum[c][l];

      // Now drive different values with en=0 — outputs should NOT change
      for (int r = 0; r < PE_ROWS; r++)
        for (int c = 0; c < PE_COLS; c++)
          for (int l = 0; l < LANES; l++)
            pe_psum[r][c][l] <= 32'sd999;
      en <= 1'b0;
      @(posedge clk);
      @(posedge clk);

      for (int c = 0; c < PE_COLS; c++)
        for (int l = 0; l < LANES; l++)
          if (col_psum[c][l] !== saved[c][l]) begin
            $display("  T5 mismatch c=%0d l=%0d: got %0d exp %0d (frozen)",
                     c, l, col_psum[c][l], saved[c][l]);
            ok = 0;
          end

      // Also verify col_valid goes low when en=0
      if (col_valid !== 1'b0) begin
        $display("  T5: col_valid should be 0 when en=0, got %0b", col_valid);
        ok = 0;
      end

      report(tname, ok);
    end

    // ──────────────────── Final Summary ────────────────────
    $display("==============================================");
    $display("  column_reduce TB: %0d / %0d PASSED", pass_count, total_tests);
    if (fail_count == 0)
      $display("  RESULT: ALL TESTS PASSED");
    else
      $display("  RESULT: %0d TEST(S) FAILED", fail_count);
    $display("==============================================");
    $finish;
  end

endmodule
