`timescale 1ns/1ps

module tb_dsp_pair_int8;

  // ---------------------------------------------------------------
  // Clock / Reset
  // ---------------------------------------------------------------
  localparam CLK_PERIOD = 5;          // 200 MHz

  logic              clk   = 0;
  logic              rst_n = 0;
  logic              en;
  logic              clear;
  logic signed [7:0] x_a, x_b, w;
  logic signed [31:0] psum_a, psum_b;

  always #(CLK_PERIOD/2.0) clk = ~clk;

  dsp_pair_int8 dut (.*);

  // ---------------------------------------------------------------
  // Global error / test counters
  // ---------------------------------------------------------------
  integer total_tests   = 0;
  integer total_errors  = 0;
  integer test_errors;             // per-test counter

  // ---------------------------------------------------------------
  // Helper tasks
  // ---------------------------------------------------------------

  // Apply reset for 10 cycles and initialise inputs to safe values
  task automatic do_reset();
    rst_n = 0;
    en    = 0;
    clear = 0;
    x_a   = 0;
    x_b   = 0;
    w     = 0;
    repeat (10) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
  endtask

  // Drive one MAC beat then drop inputs to zero-safe.
  // Does NOT wait for pipeline drain.
  task automatic drive_one (
    input logic signed [7:0] va,
    input logic signed [7:0] vb,
    input logic signed [7:0] vw,
    input logic               clr
  );
    @(posedge clk);
    en    <= 1;
    clear <= clr;
    x_a   <= va;
    x_b   <= vb;
    w     <= vw;
    @(posedge clk);           // launch edge
    en    <= 0;
    clear <= 0;
    x_a   <= 0;
    x_b   <= 0;
    w     <= 0;
  endtask

  // Wait exactly 4 pipeline cycles (+ the 1 already consumed inside
  // drive_one after the launch edge). Net: 4 more edges after
  // drive_one returns.  We add 1 extra safety cycle = 5 waits.
  task automatic wait_pipe();
    repeat (5) @(posedge clk);
  endtask

  // Drive one beat with clear=1, wait pipeline, check psum_a/psum_b.
  // Returns 0 on match, 1 on mismatch.
  task automatic check_single (
    input  logic signed [7:0]  va,
    input  logic signed [7:0]  vb,
    input  logic signed [7:0]  vw,
    input  logic signed [31:0] exp_a,
    input  logic signed [31:0] exp_b,
    output integer             err
  );
    err = 0;
    drive_one(va, vb, vw, 1);
    wait_pipe();
    if (psum_a !== exp_a) begin
      $display("  ERROR: x_a=%0d, w=%0d  psum_a=%0d  expected=%0d",
               va, vw, psum_a, exp_a);
      err = err + 1;
    end
    if (psum_b !== exp_b) begin
      $display("  ERROR: x_b=%0d, w=%0d  psum_b=%0d  expected=%0d",
               vb, vw, psum_b, exp_b);
      err = err + 1;
    end
  endtask

  // ---------------------------------------------------------------
  // Main stimulus
  // ---------------------------------------------------------------
  integer i, j, err_one;
  logic signed [7:0] sv_a, sv_b, sv_w;
  logic signed [31:0] exp_a, exp_b;

  // For Test 3
  logic signed [7:0] t3_xa [0:8];
  logic signed [7:0] t3_xb [0:8];
  logic signed [7:0] t3_w  [0:8];

  // For Test 4
  logic signed [7:0] t4_xa [0:4];
  logic signed [7:0] t4_xb [0:4];
  logic signed [7:0] t4_w  [0:4];

  initial begin
    $display("==========================================================");
    $display(" TESTBENCH: tb_dsp_pair_int8  --  Exhaustive verification");
    $display("==========================================================");

    // ==============================================================
    // TEST 1: Exhaustive 256x256 single-product
    // ==============================================================
    begin
      integer t1_total, t1_errors;
      t1_total  = 0;
      t1_errors = 0;

      $display("\n--- TEST 1: Exhaustive single-product (65536 combos) ---");
      do_reset();

      for (i = -128; i <= 127; i = i + 1) begin
        for (j = -128; j <= 127; j = j + 1) begin
          sv_a = i[7:0];
          sv_b = -sv_a;          // double coverage: x_b = -x_a
          sv_w = j[7:0];
          exp_a = 32'(sv_a) * 32'(sv_w);
          exp_b = 32'(sv_b) * 32'(sv_w);

          check_single(sv_a, sv_b, sv_w, exp_a, exp_b, err_one);
          t1_total  = t1_total  + 1;
          t1_errors = t1_errors + err_one;
        end
        // Progress every 32 rows
        if ((i & 31) == 0)
          $display("  ... x_a=%0d  (row %0d/256)", i, (i+128));
      end

      total_tests  = total_tests  + t1_total;
      total_errors = total_errors + t1_errors;
      if (t1_errors == 0)
        $display("  TEST 1 PASS  (tested %0d products, 0 errors)", t1_total);
      else
        $display("  TEST 1 FAIL  (tested %0d products, %0d errors)", t1_total, t1_errors);
    end

    // ==============================================================
    // TEST 2: Corner cases
    // ==============================================================
    begin
      integer t2_errors;
      t2_errors = 0;

      $display("\n--- TEST 2: Corner cases ---");
      do_reset();

      // (-128) * (-128) = 16384
      check_single(-8'sd128, 8'sd0, -8'sd128, 32'sd16384, 32'sd0, err_one);
      t2_errors = t2_errors + err_one;
      // 127 * 127 = 16129
      check_single(8'sd127, 8'sd0, 8'sd127, 32'sd16129, 32'sd0, err_one);
      t2_errors = t2_errors + err_one;
      // (-128) * 127 = -16256
      check_single(-8'sd128, 8'sd0, 8'sd127, -32'sd16256, 32'sd0, err_one);
      t2_errors = t2_errors + err_one;
      // 0 * (-128) = 0
      check_single(8'sd0, 8'sd0, -8'sd128, 32'sd0, 32'sd0, err_one);
      t2_errors = t2_errors + err_one;
      // 0 * 127 = 0
      check_single(8'sd0, 8'sd0, 8'sd127, 32'sd0, 32'sd0, err_one);
      t2_errors = t2_errors + err_one;
      // 1 * 1 = 1
      check_single(8'sd1, 8'sd0, 8'sd1, 32'sd1, 32'sd0, err_one);
      t2_errors = t2_errors + err_one;
      // (-1) * (-1) = 1
      check_single(-8'sd1, 8'sd0, -8'sd1, 32'sd1, 32'sd0, err_one);
      t2_errors = t2_errors + err_one;
      // (-1) * 1 = -1
      check_single(-8'sd1, 8'sd0, 8'sd1, -32'sd1, 32'sd0, err_one);
      t2_errors = t2_errors + err_one;

      total_tests  = total_tests  + 8;
      total_errors = total_errors + t2_errors;
      if (t2_errors == 0)
        $display("  TEST 2 PASS  (8 corner cases, 0 errors)");
      else
        $display("  TEST 2 FAIL  (8 corner cases, %0d errors)", t2_errors);
    end

    // ==============================================================
    // TEST 3: 9-cycle accumulation (conv3x3 pattern)
    // ==============================================================
    begin
      integer t3_errors;
      logic signed [31:0] sum_a, sum_b;
      t3_errors = 0;

      $display("\n--- TEST 3: 9-cycle accumulation (conv3x3) ---");
      do_reset();

      // x_a values
      t3_xa[0] =  10; t3_xa[1] = -20; t3_xa[2] =  30;
      t3_xa[3] = -40; t3_xa[4] =  50; t3_xa[5] = -60;
      t3_xa[6] =  70; t3_xa[7] = -80; t3_xa[8] =  90;
      // x_b = -x_a for each
      for (i = 0; i < 9; i = i + 1) t3_xb[i] = -t3_xa[i];
      // w values
      t3_w[0] =   3; t3_w[1] =  -7; t3_w[2] =  11;
      t3_w[3] = -13; t3_w[4] =  17; t3_w[5] = -19;
      t3_w[6] =  23; t3_w[7] = -29; t3_w[8] =  31;

      // Expected sums (computed manually):
      // 10*3 + (-20)*(-7) + 30*11 + (-40)*(-13) + 50*17
      // + (-60)*(-19) + 70*23 + (-80)*(-29) + 90*31
      // = 30 + 140 + 330 + 520 + 850 + 1140 + 1610 + 2320 + 2790 = 9730
      sum_a = 32'sd0;
      sum_b = 32'sd0;
      for (i = 0; i < 9; i = i + 1) begin
        sum_a = sum_a + 32'(t3_xa[i]) * 32'(t3_w[i]);
        sum_b = sum_b + 32'(t3_xb[i]) * 32'(t3_w[i]);
      end
      $display("  Expected psum_a = %0d", sum_a);
      $display("  Expected psum_b = %0d", sum_b);

      // Drive 9 beats: clear on first only
      for (i = 0; i < 9; i = i + 1) begin
        @(posedge clk);
        en    <= 1;
        clear <= (i == 0) ? 1'b1 : 1'b0;
        x_a   <= t3_xa[i];
        x_b   <= t3_xb[i];
        w     <= t3_w[i];
      end
      // Deassert after last beat
      @(posedge clk);
      en    <= 0;
      clear <= 0;
      x_a   <= 0;
      x_b   <= 0;
      w     <= 0;
      // Wait pipeline drain: 5 cycles safety
      repeat (5) @(posedge clk);

      if (psum_a !== sum_a) begin
        $display("  ERROR: psum_a=%0d  expected=%0d", psum_a, sum_a);
        t3_errors = t3_errors + 1;
      end
      if (psum_b !== sum_b) begin
        $display("  ERROR: psum_b=%0d  expected=%0d", psum_b, sum_b);
        t3_errors = t3_errors + 1;
      end

      total_tests  = total_tests  + 1;
      total_errors = total_errors + t3_errors;
      if (t3_errors == 0)
        $display("  TEST 3 PASS  (9-cycle accumulation, 0 errors)");
      else
        $display("  TEST 3 FAIL  (9-cycle accumulation, %0d errors)", t3_errors);
    end

    // ==============================================================
    // TEST 4: Clear mid-accumulation
    // ==============================================================
    begin
      integer t4_errors;
      logic signed [31:0] exp4_a, exp4_b;
      t4_errors = 0;

      $display("\n--- TEST 4: Clear mid-accumulation ---");
      do_reset();

      // 5 beats:  [0..2] phase-1,  [3] clear,  [3..4] phase-2
      t4_xa[0] =  10; t4_xa[1] =  20; t4_xa[2] =  30;
      t4_xa[3] =  40; t4_xa[4] =  50;
      t4_w[0]  =   5; t4_w[1]  =   6; t4_w[2]  =   7;
      t4_w[3]  =   8; t4_w[4]  =   9;
      for (i = 0; i < 5; i = i + 1) t4_xb[i] = -t4_xa[i];

      // Phase 1: 3 beats, clear on first
      for (i = 0; i < 3; i = i + 1) begin
        @(posedge clk);
        en    <= 1;
        clear <= (i == 0) ? 1'b1 : 1'b0;
        x_a   <= t4_xa[i];
        x_b   <= t4_xb[i];
        w     <= t4_w[i];
      end

      // Phase 2: 2 beats, clear on FIRST of this phase (index 3)
      for (i = 3; i < 5; i = i + 1) begin
        @(posedge clk);
        en    <= 1;
        clear <= (i == 3) ? 1'b1 : 1'b0;
        x_a   <= t4_xa[i];
        x_b   <= t4_xb[i];
        w     <= t4_w[i];
      end

      // Deassert
      @(posedge clk);
      en    <= 0;
      clear <= 0;
      x_a   <= 0;
      x_b   <= 0;
      w     <= 0;
      repeat (5) @(posedge clk);

      // Expected = only last 2 products
      exp4_a = 32'(t4_xa[3]) * 32'(t4_w[3]) + 32'(t4_xa[4]) * 32'(t4_w[4]);
      exp4_b = 32'(t4_xb[3]) * 32'(t4_w[3]) + 32'(t4_xb[4]) * 32'(t4_w[4]);

      if (psum_a !== exp4_a) begin
        $display("  ERROR: psum_a=%0d  expected=%0d", psum_a, exp4_a);
        t4_errors = t4_errors + 1;
      end
      if (psum_b !== exp4_b) begin
        $display("  ERROR: psum_b=%0d  expected=%0d", psum_b, exp4_b);
        t4_errors = t4_errors + 1;
      end

      total_tests  = total_tests  + 1;
      total_errors = total_errors + t4_errors;
      if (t4_errors == 0)
        $display("  TEST 4 PASS  (clear mid-accumulation, 0 errors)");
      else
        $display("  TEST 4 FAIL  (clear mid-accumulation, %0d errors)", t4_errors);
    end

    // ==============================================================
    // TEST 5: Enable gating
    // ==============================================================
    begin
      integer t5_errors;
      logic signed [31:0] snap_a, snap_b;
      t5_errors = 0;

      $display("\n--- TEST 5: Enable gating ---");
      do_reset();

      // Accumulate 1 product with clear
      drive_one(8'sd42, -8'sd42, 8'sd7, 1);
      wait_pipe();
      snap_a = psum_a;
      snap_b = psum_b;
      $display("  After 1 product: psum_a=%0d, psum_b=%0d", snap_a, snap_b);

      // Now hold en=0 for 10 cycles while driving large values
      // Drive x=0, w=0 for zero-safety per instructions
      en    <= 0;
      clear <= 0;
      x_a   <= 0;
      x_b   <= 0;
      w     <= 0;
      repeat (10) @(posedge clk);

      // Also test with non-zero inputs but en=0 -- should NOT affect psum
      // because en=0 means Stage 4 hold. But pipeline propagates en=0 through
      // en_s1..en_s3, so accumulator ignores. Drive non-zero to be sure.
      x_a   <= 8'sd100;
      x_b   <= 8'sd100;
      w     <= 8'sd100;
      repeat (10) @(posedge clk);
      // Zero-safe again
      x_a   <= 0;
      x_b   <= 0;
      w     <= 0;
      repeat (5) @(posedge clk);

      if (psum_a !== snap_a) begin
        $display("  ERROR: psum_a changed from %0d to %0d", snap_a, psum_a);
        t5_errors = t5_errors + 1;
      end
      if (psum_b !== snap_b) begin
        $display("  ERROR: psum_b changed from %0d to %0d", snap_b, psum_b);
        t5_errors = t5_errors + 1;
      end

      total_tests  = total_tests  + 1;
      total_errors = total_errors + t5_errors;
      if (t5_errors == 0)
        $display("  TEST 5 PASS  (enable gating, 0 errors)");
      else
        $display("  TEST 5 FAIL  (enable gating, %0d errors)", t5_errors);
    end

    // ==============================================================
    // TEST 6: Zero-safety when en=0
    // ==============================================================
    begin
      integer t6_errors;
      logic signed [31:0] snap6_a, snap6_b;
      t6_errors = 0;

      $display("\n--- TEST 6: Zero-safety when en=0 ---");
      do_reset();

      // Accumulate a known product
      drive_one(-8'sd55, 8'sd77, 8'sd11, 1);
      wait_pipe();
      snap6_a = psum_a;
      snap6_b = psum_b;
      $display("  After product: psum_a=%0d, psum_b=%0d", snap6_a, snap6_b);

      // CRITICAL: set en=0, drive x=0, w=0.
      // The pipeline still has "valid" en bits propagating from the
      // previous drive_one (actually no -- drive_one already set en=0).
      // But the correction for (0+128)*(0+128)=16384 is:
      //   raw_u - 128*(x_u + w_u) + 16384 = 16384 - 128*256 + 16384 = 0
      // so even if en_s3 were 1, psum += 0.  Verify psum unchanged.
      en    <= 0;
      clear <= 0;
      x_a   <= 0;
      x_b   <= 0;
      w     <= 0;
      repeat (8) @(posedge clk);     // well past pipeline drain

      if (psum_a !== snap6_a) begin
        $display("  ERROR: psum_a changed from %0d to %0d", snap6_a, psum_a);
        t6_errors = t6_errors + 1;
      end
      if (psum_b !== snap6_b) begin
        $display("  ERROR: psum_b changed from %0d to %0d", snap6_b, psum_b);
        t6_errors = t6_errors + 1;
      end

      // Additional: en=1 with x=0,w=0 should accumulate +0
      // (correction of zero inputs gives product = 0)
      drive_one(8'sd0, 8'sd0, 8'sd0, 0);  // en=1, clear=0
      wait_pipe();
      if (psum_a !== snap6_a) begin
        $display("  ERROR: psum_a changed on 0*0 accumulation: %0d -> %0d",
                 snap6_a, psum_a);
        t6_errors = t6_errors + 1;
      end
      if (psum_b !== snap6_b) begin
        $display("  ERROR: psum_b changed on 0*0 accumulation: %0d -> %0d",
                 snap6_b, psum_b);
        t6_errors = t6_errors + 1;
      end

      total_tests  = total_tests  + 1;
      total_errors = total_errors + t6_errors;
      if (t6_errors == 0)
        $display("  TEST 6 PASS  (zero-safety, 0 errors)");
      else
        $display("  TEST 6 FAIL  (zero-safety, %0d errors)", t6_errors);
    end

    // ==============================================================
    // TEST 7: Random stress (10,000 single products)
    // ==============================================================
    begin
      integer t7_total, t7_errors;
      logic signed [7:0] r_a, r_b, r_w;
      logic signed [31:0] rexp_a, rexp_b;
      t7_total  = 0;
      t7_errors = 0;

      $display("\n--- TEST 7: Random stress (10000 products) ---");
      do_reset();

      for (i = 0; i < 10000; i = i + 1) begin
        r_a = $random;
        r_b = $random;
        r_w = $random;
        rexp_a = 32'(r_a) * 32'(r_w);
        rexp_b = 32'(r_b) * 32'(r_w);

        check_single(r_a, r_b, r_w, rexp_a, rexp_b, err_one);
        t7_total  = t7_total  + 1;
        t7_errors = t7_errors + err_one;

        if ((i % 2000) == 0)
          $display("  ... random iteration %0d", i);
      end

      total_tests  = total_tests  + t7_total;
      total_errors = total_errors + t7_errors;
      if (t7_errors == 0)
        $display("  TEST 7 PASS  (tested %0d random products, 0 errors)", t7_total);
      else
        $display("  TEST 7 FAIL  (tested %0d random products, %0d errors)", t7_total, t7_errors);
    end

    // ==============================================================
    // FINAL SUMMARY
    // ==============================================================
    $display("\n==========================================================");
    $display(" FINAL SUMMARY");
    $display("  Total tests  : %0d", total_tests);
    $display("  Total errors : %0d", total_errors);
    if (total_errors == 0)
      $display("  *** OVERALL PASS ***");
    else
      $display("  *** OVERALL FAIL ***");
    $display("==========================================================\n");

    $finish;
  end

  // ---------------------------------------------------------------
  // Timeout watchdog -- abort after 100ms sim-time
  // ---------------------------------------------------------------
  initial begin
    #100_000_000;
    $display("TIMEOUT: simulation exceeded 100 ms. Aborting.");
    $finish;
  end

endmodule
