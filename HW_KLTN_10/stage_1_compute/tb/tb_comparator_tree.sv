// ============================================================================
// Testbench: tb_comparator_tree
// Project:   YOLOv10n INT8 Accelerator — V4-VC707
// Description:
//   Verification of comparator_tree: 25 signed INT8 → 1 max per lane.
//   Tests: boundary values, gradient, per-lane independence, random stress.
//   PASS CRITERIA: 0 errors.
// ============================================================================
`timescale 1ns / 1ps

module tb_comparator_tree;
  import accel_pkg::*;

  localparam real CLK_PERIOD = 4.0;
  localparam int  L = LANES;         // 20
  localparam int  N = 25;            // 5×5 inputs
  localparam int  PIPE = 5;          // 5 registered compute stages; valid_out = valid_sr[5] → 6-cycle latency

  logic              clk   = 1'b0;
  logic              rst_n = 1'b0;
  logic              valid_in;
  logic signed [7:0] data_in [N][L];
  logic signed [7:0] max_out [L];
  logic              valid_out;

  always #(CLK_PERIOD / 2.0) clk = ~clk;

  comparator_tree #(.LANES(L), .NUM_INPUTS(N)) u_dut (
    .clk      (clk),
    .rst_n    (rst_n),
    .valid_in (valid_in),
    .data_in  (data_in),
    .max_out  (max_out),
    .valid_out(valid_out)
  );

  integer total_tests = 0, total_errors = 0;

  task automatic do_reset();
    rst_n    = 1'b0;
    valid_in = 1'b0;
    for (int i = 0; i < N; i++)
      for (int l = 0; l < L; l++)
        data_in[i][l] = 8'sd0;
    repeat (5) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
  endtask

  // Pulse valid_in for 1 cycle, then wait for pipeline
  task automatic pulse_and_wait();
    @(posedge clk);
    valid_in <= 1'b1;
    @(posedge clk);
    valid_in <= 1'b0;
    // Wait 5 pipeline stages + 2 safety
    repeat (PIPE + 2) @(posedge clk);
  endtask

  // Behavioral golden: find max of 25 values
  function automatic logic signed [7:0] golden_max(
    input logic signed [7:0] vals [N]
  );
    automatic logic signed [7:0] m = vals[0];
    for (int i = 1; i < N; i++)
      if (vals[i] > m) m = vals[i];
    return m;
  endfunction

  initial begin
    $display("══════════════════════════════════════════════════════════");
    $display(" TB: comparator_tree — V4 (25→1 max, 5-stage, LANES=%0d)", L);
    $display("══════════════════════════════════════════════════════════");

    // ═════════════════════════════════════════════════════
    //  TEST 1: All same value
    // ═════════════════════════════════════════════════════
    begin
      integer t_err = 0;
      $display("\n── TEST 1: All Same Value ──");

      // All = -128
      do_reset();
      for (int i = 0; i < N; i++)
        for (int l = 0; l < L; l++)
          data_in[i][l] = -8'sd128;
      pulse_and_wait();
      for (int l = 0; l < L; l++)
        if (max_out[l] !== -8'sd128) begin
          $display("  ERR lane %0d: got=%0d exp=-128", l, max_out[l]); t_err++;
        end

      // All = 127
      do_reset();
      for (int i = 0; i < N; i++)
        for (int l = 0; l < L; l++)
          data_in[i][l] = 8'sd127;
      pulse_and_wait();
      for (int l = 0; l < L; l++)
        if (max_out[l] !== 8'sd127) begin
          $display("  ERR lane %0d: got=%0d exp=127", l, max_out[l]); t_err++;
        end

      // All = 0
      do_reset();
      for (int i = 0; i < N; i++)
        for (int l = 0; l < L; l++)
          data_in[i][l] = 8'sd0;
      pulse_and_wait();
      for (int l = 0; l < L; l++)
        if (max_out[l] !== 8'sd0) begin
          $display("  ERR lane %0d: got=%0d exp=0", l, max_out[l]); t_err++;
        end

      total_tests += 3; total_errors += t_err;
      $display("  TEST 1 %s (%0d errors)", t_err == 0 ? "PASS" : "FAIL", t_err);
    end

    // ═════════════════════════════════════════════════════
    //  TEST 2: Single Maximum at Each Position
    // ═════════════════════════════════════════════════════
    begin
      integer t_err = 0;
      $display("\n── TEST 2: Single Max at Each Position ──");

      for (int pos = 0; pos < N; pos++) begin
        do_reset();
        // All -128 except position 'pos' = 127
        for (int i = 0; i < N; i++)
          for (int l = 0; l < L; l++)
            data_in[i][l] = (i == pos) ? 8'sd127 : -8'sd128;
        pulse_and_wait();
        for (int l = 0; l < L; l++)
          if (max_out[l] !== 8'sd127) begin
            if (t_err < 5) $display("  ERR pos=%0d lane=%0d: got=%0d exp=127", pos, l, max_out[l]);
            t_err++;
          end
      end

      total_tests += N; total_errors += t_err;
      $display("  TEST 2 %s (tested %0d positions, %0d errors)", t_err == 0 ? "PASS" : "FAIL", N, t_err);
    end

    // ═════════════════════════════════════════════════════
    //  TEST 3: Ascending Gradient
    // ═════════════════════════════════════════════════════
    begin
      integer t_err = 0;
      $display("\n── TEST 3: Ascending Gradient (max=24 at all lanes) ──");
      do_reset();
      for (int i = 0; i < N; i++)
        for (int l = 0; l < L; l++)
          data_in[i][l] = i[7:0];  // 0, 1, 2, ..., 24
      pulse_and_wait();
      for (int l = 0; l < L; l++)
        if (max_out[l] !== 8'sd24) begin
          $display("  ERR lane %0d: got=%0d exp=24", l, max_out[l]); t_err++;
        end

      total_tests++; total_errors += t_err;
      $display("  TEST 3 %s", t_err == 0 ? "PASS" : "FAIL");
    end

    // ═════════════════════════════════════════════════════
    //  TEST 4: Per-Lane Independence
    // ═════════════════════════════════════════════════════
    begin
      integer t_err = 0;
      logic signed [7:0] exp_max [L];
      $display("\n── TEST 4: Per-Lane Independence ──");
      do_reset();

      // Each lane has max at a different input position
      for (int l = 0; l < L; l++) begin
        for (int i = 0; i < N; i++)
          data_in[i][l] = -8'sd50;
        // Lane l: max at position (l % N)
        data_in[l % N][l] = 8'sd100;
        exp_max[l] = 8'sd100;
      end
      pulse_and_wait();
      for (int l = 0; l < L; l++)
        if (max_out[l] !== exp_max[l]) begin
          $display("  ERR lane %0d: got=%0d exp=%0d", l, max_out[l], exp_max[l]); t_err++;
        end

      total_tests++; total_errors += t_err;
      $display("  TEST 4 %s", t_err == 0 ? "PASS" : "FAIL");
    end

    // ═════════════════════════════════════════════════════
    //  TEST 5: Signed Boundary — Negative Numbers
    // ═════════════════════════════════════════════════════
    begin
      integer t_err = 0;
      $display("\n── TEST 5: Signed Boundary (negative range) ──");
      do_reset();
      // All values negative: -128, -127, ..., -104
      for (int i = 0; i < N; i++)
        for (int l = 0; l < L; l++)
          data_in[i][l] = (-128 + i);
      pulse_and_wait();
      // Max should be -104 (= -128 + 24)
      for (int l = 0; l < L; l++)
        if (max_out[l] !== -8'sd104) begin
          $display("  ERR lane %0d: got=%0d exp=-104", l, max_out[l]); t_err++;
        end

      total_tests++; total_errors += t_err;
      $display("  TEST 5 %s", t_err == 0 ? "PASS" : "FAIL");
    end

    // ═════════════════════════════════════════════════════
    //  TEST 6: Random Stress — 500 Windows
    // ═════════════════════════════════════════════════════
    begin
      integer t_err = 0;
      logic signed [7:0] lane_vals [N];
      logic signed [7:0] exp_v;
      $display("\n── TEST 6: Random Stress (500 windows) ──");

      for (int iter = 0; iter < 500; iter++) begin
        do_reset();
        for (int l = 0; l < L; l++) begin
          for (int i = 0; i < N; i++) begin
            data_in[i][l] = $random;
            lane_vals[i] = data_in[i][l];
          end
        end
        pulse_and_wait();

        for (int l = 0; l < L; l++) begin
          // Compute golden max for this lane
          for (int i = 0; i < N; i++) lane_vals[i] = data_in[i][l];
          exp_v = golden_max(lane_vals);
          if (max_out[l] !== exp_v) begin
            if (t_err < 5) $display("  ERR iter=%0d lane=%0d: got=%0d exp=%0d", iter, l, max_out[l], exp_v);
            t_err++;
          end
        end
      end

      total_tests += 500; total_errors += t_err;
      $display("  TEST 6 %s (500 windows, %0d errors)", t_err == 0 ? "PASS" : "FAIL", t_err);
    end

    // ═════════════════════════════════════════════════════
    //  TEST 7: Pipeline Timing — valid_out delay
    // ═════════════════════════════════════════════════════
    begin
      integer t_err = 0;
      integer count = 0;
      $display("\n── TEST 7: Pipeline Timing ──");
      do_reset();

      // Pulse valid_in, count cycles until valid_out
      @(posedge clk); valid_in <= 1'b1;
      for (int i = 0; i < N; i++)
        for (int l = 0; l < L; l++)
          data_in[i][l] <= $random;
      @(posedge clk); valid_in <= 1'b0;

      count = 0;
      while (!valid_out && count < 20) begin
        @(posedge clk);
        count++;
      end

      // valid_sr[5] → 6 rising edges after valid_in deasserts (5 data stages + 1 tap)
      if (count != 6) begin
        $display("  ERR: valid_out after %0d cycles, expected 6", count);
        t_err++;
      end

      total_tests++; total_errors += t_err;
      $display("  TEST 7 %s (latency=%0d cycles)", t_err == 0 ? "PASS" : "FAIL", count);
    end

    // Summary
    $display("\n══════════════════════════════════════════════════════════");
    $display(" FINAL: %0d tests, %0d errors", total_tests, total_errors);
    if (total_errors == 0)
      $display(" ★★★ ALL PASS — comparator_tree VERIFIED ★★★");
    else
      $display(" ✗✗✗ FAIL — %0d errors ✗✗✗", total_errors);
    $display("══════════════════════════════════════════════════════════\n");
    $finish;
  end

  initial begin #100_000_000; $display("TIMEOUT"); $finish; end

endmodule
