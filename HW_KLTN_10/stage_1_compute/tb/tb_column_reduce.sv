// ============================================================================
// Testbench: tb_column_reduce
// Project:   YOLOv10n INT8 Accelerator — V4-VC707
// Description:
//   Verification of column_reduce: 3 PE rows × 4 cols × 20 lanes → 4 × 20.
//   Tests: known values, overflow boundary, random vectors.
//   PASS CRITERIA: 0 errors.
// ============================================================================
`timescale 1ns / 1ps

module tb_column_reduce;
  import accel_pkg::*;

  localparam real CLK_PERIOD = 4.0;
  localparam int  R = PE_ROWS;  // 3
  localparam int  C = PE_COLS;  // 4
  localparam int  L = LANES;    // 20

  logic              clk   = 1'b0;
  logic              rst_n = 1'b0;
  logic              valid_in;
  logic signed [31:0] row_psum [R][C][L];
  logic signed [31:0] col_psum [C][L];
  logic              valid_out;

  always #(CLK_PERIOD / 2.0) clk = ~clk;

  column_reduce #(.LANES(L), .N_ROWS(R), .PE_COLS(C)) u_dut (
    .clk      (clk),
    .rst_n    (rst_n),
    .valid_in (valid_in),
    .row_psum (row_psum),
    .col_psum (col_psum),
    .valid_out(valid_out)
  );

  integer total_tests = 0, total_errors = 0;

  task automatic do_reset();
    rst_n    = 1'b0;
    valid_in = 1'b0;
    for (int r = 0; r < R; r++)
      for (int c = 0; c < C; c++)
        for (int l = 0; l < L; l++)
          row_psum[r][c][l] = 0;
    repeat (5) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
  endtask

  initial begin
    $display("══════════════════════════════════════════════════════════");
    $display(" TB: column_reduce — V4 (3×4×20, 1-cycle latency)");
    $display("══════════════════════════════════════════════════════════");

    // ═════════════════════════════════════════════════════
    //  TEST 1: Known Values
    // ═════════════════════════════════════════════════════
    begin
      integer t_err = 0;
      logic signed [31:0] golden [C][L];
      $display("\n── TEST 1: Known Values ──");
      do_reset();

      // Set known data: row[r][c][l] = (r+1) * (c+1) * (l+1) * 100
      for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
          for (int l = 0; l < L; l++)
            row_psum[r][c][l] = (r + 1) * (c + 1) * (l + 1) * 100;

      // Golden: sum across 3 rows
      for (int c = 0; c < C; c++)
        for (int l = 0; l < L; l++)
          golden[c][l] = row_psum[0][c][l] + row_psum[1][c][l] + row_psum[2][c][l];

      @(posedge clk);
      valid_in <= 1'b1;
      @(posedge clk);
      valid_in <= 1'b0;
      @(posedge clk); // 1-cycle latency
      @(posedge clk); // extra safety

      for (int c = 0; c < C; c++)
        for (int l = 0; l < L; l++)
          if (col_psum[c][l] !== golden[c][l]) begin
            $display("  ERR [c=%0d][l=%0d]: got=%0d exp=%0d", c, l, col_psum[c][l], golden[c][l]);
            t_err++;
          end

      total_tests++; total_errors += t_err;
      $display("  TEST 1 %s (%0d errors)", t_err == 0 ? "PASS" : "FAIL", t_err);
    end

    // ═════════════════════════════════════════════════════
    //  TEST 2: Positive + Negative Cancellation
    // ═════════════════════════════════════════════════════
    begin
      integer t_err = 0;
      $display("\n── TEST 2: Cancellation ──");
      do_reset();

      for (int c = 0; c < C; c++)
        for (int l = 0; l < L; l++) begin
          row_psum[0][c][l] =  100000;
          row_psum[1][c][l] = -50000;
          row_psum[2][c][l] = -50000;
        end

      @(posedge clk); valid_in <= 1'b1;
      @(posedge clk); valid_in <= 1'b0;
      repeat (2) @(posedge clk);

      for (int c = 0; c < C; c++)
        for (int l = 0; l < L; l++)
          if (col_psum[c][l] !== 32'sd0) begin
            $display("  ERR [c=%0d][l=%0d]: got=%0d exp=0", c, l, col_psum[c][l]);
            t_err++;
          end

      total_tests++; total_errors += t_err;
      $display("  TEST 2 %s", t_err == 0 ? "PASS" : "FAIL");
    end

    // ═════════════════════════════════════════════════════
    //  TEST 3: Random 200 Vectors
    // ═════════════════════════════════════════════════════
    begin
      integer t_err = 0;
      logic signed [31:0] golden [C][L];
      $display("\n── TEST 3: Random 200 Vectors ──");

      for (int iter = 0; iter < 200; iter++) begin
        do_reset();

        for (int r = 0; r < R; r++)
          for (int c = 0; c < C; c++)
            for (int l = 0; l < L; l++)
              row_psum[r][c][l] = $random;

        for (int c = 0; c < C; c++)
          for (int l = 0; l < L; l++)
            golden[c][l] = row_psum[0][c][l] + row_psum[1][c][l] + row_psum[2][c][l];

        @(posedge clk); valid_in <= 1'b1;
        @(posedge clk); valid_in <= 1'b0;
        repeat (2) @(posedge clk);

        for (int c = 0; c < C; c++)
          for (int l = 0; l < L; l++)
            if (col_psum[c][l] !== golden[c][l]) begin
              if (t_err < 5) $display("  ERR iter=%0d [c=%0d][l=%0d]: got=%0d exp=%0d",
                                       iter, c, l, col_psum[c][l], golden[c][l]);
              t_err++;
            end
      end

      total_tests += 200; total_errors += t_err;
      $display("  TEST 3 %s (%0d errors in 200 vectors)", t_err == 0 ? "PASS" : "FAIL", t_err);
    end

    // ═════════════════════════════════════════════════════
    //  TEST 4: valid_out Timing
    // ═════════════════════════════════════════════════════
    begin
      integer t_err = 0;
      $display("\n── TEST 4: valid_out Timing ──");
      do_reset();

      // valid_in pulse → valid_out should appear 1 cycle later
      @(posedge clk); valid_in <= 1'b1;
      @(posedge clk); valid_in <= 1'b0;
      if (valid_out !== 1'b0) begin $display("  ERR: valid_out early"); t_err++; end
      @(posedge clk);
      if (valid_out !== 1'b1) begin $display("  ERR: valid_out not asserted after 1 cycle"); t_err++; end
      @(posedge clk);
      if (valid_out !== 1'b0) begin $display("  ERR: valid_out stuck high"); t_err++; end

      total_tests++; total_errors += t_err;
      $display("  TEST 4 %s", t_err == 0 ? "PASS" : "FAIL");
    end

    // Summary
    $display("\n══════════════════════════════════════════════════════════");
    $display(" FINAL: %0d tests, %0d errors", total_tests, total_errors);
    if (total_errors == 0)
      $display(" ★★★ ALL PASS — column_reduce VERIFIED ★★★");
    else
      $display(" ✗✗✗ FAIL — %0d errors ✗✗✗", total_errors);
    $display("══════════════════════════════════════════════════════════\n");
    $finish;
  end

  initial begin #50_000_000; $display("TIMEOUT"); $finish; end

endmodule
