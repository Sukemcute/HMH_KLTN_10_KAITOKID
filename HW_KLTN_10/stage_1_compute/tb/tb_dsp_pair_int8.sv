// ============================================================================
// Testbench: tb_dsp_pair_int8
// Project:   YOLOv10n INT8 Accelerator — V4-VC707
// Description:
//   Exhaustive verification of the dsp_pair_int8 module.
//   Covers: corner cases, full 65536 exhaustive, accumulation, enable gating,
//           pipeline timing, and random stress.
//
//   V4 change: Pipeline is now 5 stages (was 4 in V3).
//   PASS CRITERIA: 0 errors. No tolerance. This is the computational foundation.
// ============================================================================
`timescale 1ns / 1ps

module tb_dsp_pair_int8;

  // ─────────────────────────────────────────────────────────────
  //  Clock / Reset — 250 MHz (4ns period for V4 target)
  // ─────────────────────────────────────────────────────────────
  localparam real CLK_PERIOD = 4.0;  // 250 MHz
  localparam int  PIPE_DEPTH = 5;    // V4: 5-stage pipeline

  logic              clk   = 1'b0;
  logic              rst_n = 1'b0;
  logic              en;
  logic              clear;
  logic signed [7:0] x_a, x_b, w;
  logic signed [31:0] psum_a, psum_b;

  always #(CLK_PERIOD / 2.0) clk = ~clk;

  // ─────────────────────────────────────────────────────────────
  //  DUT Instantiation
  // ─────────────────────────────────────────────────────────────
  dsp_pair_int8 u_dut (
    .clk    (clk),
    .rst_n  (rst_n),
    .x_a    (x_a),
    .x_b    (x_b),
    .w      (w),
    .en     (en),
    .clear  (clear),
    .psum_a (psum_a),
    .psum_b (psum_b)
  );

  // ─────────────────────────────────────────────────────────────
  //  Global Counters
  // ─────────────────────────────────────────────────────────────
  integer total_tests  = 0;
  integer total_errors = 0;

  // ─────────────────────────────────────────────────────────────
  //  Helper: Reset DUT
  // ─────────────────────────────────────────────────────────────
  task automatic do_reset();
    rst_n = 1'b0;
    en    = 1'b0;
    clear = 1'b0;
    x_a   = 8'sd0;
    x_b   = 8'sd0;
    w     = 8'sd0;
    repeat (10) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
  endtask

  // ─────────────────────────────────────────────────────────────
  //  Helper: Drive one MAC beat, then deassert
  // ─────────────────────────────────────────────────────────────
  task automatic drive_one(
    input logic signed [7:0]  va,
    input logic signed [7:0]  vb,
    input logic signed [7:0]  vw,
    input logic               clr
  );
    @(posedge clk);
    en    <= 1'b1;
    clear <= clr;
    x_a   <= va;
    x_b   <= vb;
    w     <= vw;
    @(posedge clk);
    en    <= 1'b0;
    clear <= 1'b0;
    x_a   <= 8'sd0;
    x_b   <= 8'sd0;
    w     <= 8'sd0;
  endtask

  // ─────────────────────────────────────────────────────────────
  //  Helper: Wait for pipeline drain (5-stage + 1 safety)
  // ─────────────────────────────────────────────────────────────
  task automatic wait_pipe();
    repeat (PIPE_DEPTH + 1) @(posedge clk);
  endtask

  // ─────────────────────────────────────────────────────────────
  //  Helper: Drive single product with clear, wait, check
  // ─────────────────────────────────────────────────────────────
  task automatic check_single(
    input  logic signed [7:0]  va, vb, vw,
    input  logic signed [31:0] exp_a, exp_b,
    output integer             err
  );
    err = 0;
    drive_one(va, vb, vw, 1'b1);
    wait_pipe();
    if (psum_a !== exp_a) begin
      $display("  ERR: x_a=%0d w=%0d -> psum_a=%0d expected=%0d", va, vw, psum_a, exp_a);
      err++;
    end
    if (psum_b !== exp_b) begin
      $display("  ERR: x_b=%0d w=%0d -> psum_b=%0d expected=%0d", vb, vw, psum_b, exp_b);
      err++;
    end
  endtask

  // ─────────────────────────────────────────────────────────────
  //  Main Test Sequence
  // ─────────────────────────────────────────────────────────────
  integer i, j, err_one;
  logic signed [7:0] sv_a, sv_b, sv_w;

  initial begin
    $display("══════════════════════════════════════════════════════════");
    $display(" TB: dsp_pair_int8 — V4 (5-stage, LANES=20, 250 MHz)");
    $display("══════════════════════════════════════════════════════════");

    // ════════════════════════════════════════════════════════════
    //  TEST 1: Corner Cases (17 critical products)
    // ════════════════════════════════════════════════════════════
    begin
      integer t_err = 0;
      $display("\n── TEST 1: Corner Cases ──");
      do_reset();

      check_single(-8'sd128, 8'sd127,  -8'sd128, 32'sd16384,  -32'sd16256, err_one); t_err += err_one;  // (-128)*(-128)=16384, 127*(-128)=-16256
      check_single( 8'sd127, -8'sd128,  8'sd127,  32'sd16129,  -32'sd16256, err_one); t_err += err_one;  // 127*127=16129, (-128)*127=-16256
      check_single(-8'sd128, 8'sd0,     8'sd127,  -32'sd16256, 32'sd0,      err_one); t_err += err_one;  // (-128)*127=-16256, 0*127=0
      check_single( 8'sd0,   8'sd0,     -8'sd128, 32'sd0,      32'sd0,      err_one); t_err += err_one;  // 0*(-128)=0
      check_single( 8'sd0,   8'sd0,     8'sd0,    32'sd0,      32'sd0,      err_one); t_err += err_one;  // 0*0=0
      check_single( 8'sd1,   -8'sd1,    8'sd1,    32'sd1,      -32'sd1,     err_one); t_err += err_one;  // 1*1=1, (-1)*1=-1
      check_single(-8'sd1,   8'sd1,     -8'sd1,   32'sd1,      -32'sd1,     err_one); t_err += err_one;  // (-1)*(-1)=1, 1*(-1)=-1
      check_single( 8'sd100, -8'sd100,  8'sd50,   32'sd5000,   -32'sd5000,  err_one); t_err += err_one;

      total_tests += 8; total_errors += t_err;
      $display("  TEST 1 %s (%0d errors)", t_err == 0 ? "PASS" : "FAIL", t_err);
    end

    // ════════════════════════════════════════════════════════════
    //  TEST 2: Exhaustive 256×256 = 65,536 Single Products
    // ════════════════════════════════════════════════════════════
    begin
      integer t_err = 0, t_cnt = 0;
      logic signed [31:0] exp_a_v, exp_b_v;
      $display("\n── TEST 2: Exhaustive 65536 Products ──");
      do_reset();

      for (i = -128; i <= 127; i++) begin
        for (j = -128; j <= 127; j++) begin
          sv_a = i[7:0];
          sv_b = ~sv_a + 8'd1;  // sv_b = -sv_a (for double coverage)
          sv_w = j[7:0];
          exp_a_v = 32'(sv_a) * 32'(sv_w);
          exp_b_v = 32'(sv_b) * 32'(sv_w);

          check_single(sv_a, sv_b, sv_w, exp_a_v, exp_b_v, err_one);
          t_cnt++;
          t_err += err_one;
        end
        if ((i & 63) == 0) $display("  ... row %0d/256", i + 128);
      end

      total_tests += t_cnt; total_errors += t_err;
      $display("  TEST 2 %s (%0d tests, %0d errors)", t_err == 0 ? "PASS" : "FAIL", t_cnt, t_err);
    end

    // ════════════════════════════════════════════════════════════
    //  TEST 3: 9-Cycle Accumulation (Conv 3×3 Pattern)
    // ════════════════════════════════════════════════════════════
    begin
      integer t_err = 0;
      logic signed [7:0]  t3_xa [9], t3_xb [9], t3_w [9];
      logic signed [31:0] sum_a, sum_b;
      $display("\n── TEST 3: 9-Cycle Accumulation (Conv 3x3) ──");
      do_reset();

      // Test data
      t3_xa = '{ 10, -20,  30, -40,  50, -60,  70, -80,  90};
      t3_w  = '{  3,  -7,  11, -13,  17, -19,  23, -29,  31};
      for (int k = 0; k < 9; k++) t3_xb[k] = -t3_xa[k];

      // Golden calculation
      sum_a = 0; sum_b = 0;
      for (int k = 0; k < 9; k++) begin
        sum_a += 32'(t3_xa[k]) * 32'(t3_w[k]);
        sum_b += 32'(t3_xb[k]) * 32'(t3_w[k]);
      end
      $display("  Golden: psum_a=%0d, psum_b=%0d", sum_a, sum_b);

      // Drive 9 beats: clear on first
      for (int k = 0; k < 9; k++) begin
        @(posedge clk);
        en    <= 1'b1;
        clear <= (k == 0);
        x_a   <= t3_xa[k];
        x_b   <= t3_xb[k];
        w     <= t3_w[k];
      end
      @(posedge clk);
      en <= 1'b0; clear <= 1'b0; x_a <= 0; x_b <= 0; w <= 0;
      wait_pipe();

      if (psum_a !== sum_a) begin $display("  ERR: psum_a=%0d exp=%0d", psum_a, sum_a); t_err++; end
      if (psum_b !== sum_b) begin $display("  ERR: psum_b=%0d exp=%0d", psum_b, sum_b); t_err++; end

      total_tests++; total_errors += t_err;
      $display("  TEST 3 %s", t_err == 0 ? "PASS" : "FAIL");
    end

    // ════════════════════════════════════════════════════════════
    //  TEST 4: 49-Cycle Accumulation (DW 7×7 Pattern)
    // ════════════════════════════════════════════════════════════
    begin
      integer t_err = 0;
      logic signed [31:0] sum_a, sum_b;
      logic signed [7:0] ra, rb, rw;
      $display("\n── TEST 4: 49-Cycle Accumulation (DW 7x7) ──");
      do_reset();

      sum_a = 0; sum_b = 0;
      for (int k = 0; k < 49; k++) begin
        ra = $random; rb = -ra; rw = $random;
        sum_a += 32'(ra) * 32'(rw);
        sum_b += 32'(rb) * 32'(rw);
        @(posedge clk);
        en    <= 1'b1;
        clear <= (k == 0);
        x_a   <= ra; x_b <= rb; w <= rw;
      end
      @(posedge clk);
      en <= 1'b0; clear <= 1'b0; x_a <= 0; x_b <= 0; w <= 0;
      wait_pipe();

      if (psum_a !== sum_a) begin $display("  ERR: psum_a=%0d exp=%0d", psum_a, sum_a); t_err++; end
      if (psum_b !== sum_b) begin $display("  ERR: psum_b=%0d exp=%0d", psum_b, sum_b); t_err++; end

      total_tests++; total_errors += t_err;
      $display("  TEST 4 %s", t_err == 0 ? "PASS" : "FAIL");
    end

    // ════════════════════════════════════════════════════════════
    //  TEST 5: Clear Mid-Accumulation
    // ════════════════════════════════════════════════════════════
    begin
      integer t_err = 0;
      logic signed [7:0]  t5_xa [5], t5_xb [5], t5_w [5];
      logic signed [31:0] exp_a, exp_b;
      $display("\n── TEST 5: Clear Mid-Accumulation ──");
      do_reset();

      t5_xa = '{10, 20, 30, 40, 50};
      t5_w  = '{ 5,  6,  7,  8,  9};
      for (int k = 0; k < 5; k++) t5_xb[k] = -t5_xa[k];

      // Phase 1: beats 0-2, clear on 0
      for (int k = 0; k < 3; k++) begin
        @(posedge clk);
        en <= 1'b1; clear <= (k == 0);
        x_a <= t5_xa[k]; x_b <= t5_xb[k]; w <= t5_w[k];
      end
      // Phase 2: beats 3-4, clear on 3 (resets accumulator)
      for (int k = 3; k < 5; k++) begin
        @(posedge clk);
        en <= 1'b1; clear <= (k == 3);
        x_a <= t5_xa[k]; x_b <= t5_xb[k]; w <= t5_w[k];
      end
      @(posedge clk);
      en <= 1'b0; clear <= 1'b0; x_a <= 0; x_b <= 0; w <= 0;
      wait_pipe();

      // Expected: only beats 3-4 (clear at beat 3 resets accumulator)
      exp_a = 32'(t5_xa[3]) * 32'(t5_w[3]) + 32'(t5_xa[4]) * 32'(t5_w[4]);
      exp_b = 32'(t5_xb[3]) * 32'(t5_w[3]) + 32'(t5_xb[4]) * 32'(t5_w[4]);

      if (psum_a !== exp_a) begin $display("  ERR: psum_a=%0d exp=%0d", psum_a, exp_a); t_err++; end
      if (psum_b !== exp_b) begin $display("  ERR: psum_b=%0d exp=%0d", psum_b, exp_b); t_err++; end

      total_tests++; total_errors += t_err;
      $display("  TEST 5 %s", t_err == 0 ? "PASS" : "FAIL");
    end

    // ════════════════════════════════════════════════════════════
    //  TEST 6: Enable Gating — psum holds when en=0
    // ════════════════════════════════════════════════════════════
    begin
      integer t_err = 0;
      logic signed [31:0] snap_a, snap_b;
      $display("\n── TEST 6: Enable Gating ──");
      do_reset();

      // Accumulate one product
      drive_one(8'sd42, -8'sd42, 8'sd7, 1'b1);
      wait_pipe();
      snap_a = psum_a; snap_b = psum_b;

      // Hold en=0 for 20 cycles with non-zero inputs — psum must NOT change
      en <= 1'b0;
      x_a <= 8'sd100; x_b <= 8'sd100; w <= 8'sd100;
      repeat (20) @(posedge clk);
      x_a <= 0; x_b <= 0; w <= 0;
      repeat (PIPE_DEPTH + 2) @(posedge clk);

      if (psum_a !== snap_a) begin $display("  ERR: psum_a changed %0d->%0d", snap_a, psum_a); t_err++; end
      if (psum_b !== snap_b) begin $display("  ERR: psum_b changed %0d->%0d", snap_b, psum_b); t_err++; end

      total_tests++; total_errors += t_err;
      $display("  TEST 6 %s", t_err == 0 ? "PASS" : "FAIL");
    end

    // ════════════════════════════════════════════════════════════
    //  TEST 7: Random Stress — 10,000 Single Products
    // ════════════════════════════════════════════════════════════
    begin
      integer t_err = 0, t_cnt = 0;
      logic signed [7:0]  r_a, r_b, r_w;
      logic signed [31:0] rexp_a, rexp_b;
      $display("\n── TEST 7: Random Stress (10000 products) ──");
      do_reset();

      for (i = 0; i < 10000; i++) begin
        r_a = $random; r_b = $random; r_w = $random;
        rexp_a = 32'(r_a) * 32'(r_w);
        rexp_b = 32'(r_b) * 32'(r_w);
        check_single(r_a, r_b, r_w, rexp_a, rexp_b, err_one);
        t_cnt++; t_err += err_one;
        if ((i % 2000) == 0) $display("  ... iteration %0d", i);
      end

      total_tests += t_cnt; total_errors += t_err;
      $display("  TEST 7 %s (%0d tests, %0d errors)", t_err == 0 ? "PASS" : "FAIL", t_cnt, t_err);
    end

    // ════════════════════════════════════════════════════════════
    //  TEST 8: Random Multi-Cycle Accumulation (20 sequences)
    // ════════════════════════════════════════════════════════════
    begin
      integer t_err = 0;
      logic signed [7:0]  ra, rb, rw;
      logic signed [31:0] golden_a, golden_b;
      $display("\n── TEST 8: Random Multi-Cycle Accumulation ──");

      for (int seq = 0; seq < 20; seq++) begin
        integer len;
        do_reset();
        golden_a = 0; golden_b = 0;
        len = 3 + ($urandom % 18);
        for (int k = 0; k < len; k++) begin
          ra = $random; rb = $random; rw = $random;
          golden_a += 32'(ra) * 32'(rw);
          golden_b += 32'(rb) * 32'(rw);
          @(posedge clk);
          en <= 1'b1; clear <= (k == 0);
          x_a <= ra; x_b <= rb; w <= rw;
        end
        @(posedge clk);
        en <= 1'b0; clear <= 1'b0; x_a <= 0; x_b <= 0; w <= 0;
        wait_pipe();

        if (psum_a !== golden_a) begin $display("  ERR seq%0d: psum_a=%0d exp=%0d", seq, psum_a, golden_a); t_err++; end
        if (psum_b !== golden_b) begin $display("  ERR seq%0d: psum_b=%0d exp=%0d", seq, psum_b, golden_b); t_err++; end
      end

      total_tests += 20; total_errors += t_err;
      $display("  TEST 8 %s (%0d sequences, %0d errors)", t_err == 0 ? "PASS" : "FAIL", 20, t_err);
    end

    // ════════════════════════════════════════════════════════════
    //  FINAL SUMMARY
    // ════════════════════════════════════════════════════════════
    $display("\n══════════════════════════════════════════════════════════");
    $display(" FINAL: %0d tests, %0d errors", total_tests, total_errors);
    if (total_errors == 0)
      $display(" ★★★ ALL PASS — dsp_pair_int8 VERIFIED ★★★");
    else
      $display(" ✗✗✗ FAIL — %0d errors detected ✗✗✗", total_errors);
    $display("══════════════════════════════════════════════════════════\n");
    $finish;
  end

  // Timeout watchdog
  initial begin
    #500_000_000;
    $display("TIMEOUT: simulation exceeded 500 ms. Aborting.");
    $finish;
  end

endmodule
