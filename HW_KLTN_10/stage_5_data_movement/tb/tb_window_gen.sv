// ============================================================================
// Testbench : tb_window_gen
// Project   : YOLOv10n INT8 Accelerator — V4-VC707
// Tests     : T5.2.1 K=3 (Conv 3x3): shift 3 rows, verify taps[0..2]
//             T5.2.2 K=5 (MaxPool): shift 5 rows, verify taps[0..4]
//             T5.2.3 K=7 (DW 7x7): shift 7 rows, verify taps[0..6]
//             T5.2.4 K=1 (Conv 1x1): shift 1, tap[0] = input
//             T5.2.5 Flush + refill: flush mid-stream, refill, verify
// ============================================================================
`timescale 1ns / 1ps

module tb_window_gen;
  import accel_pkg::*;

  // ──────────────────────────────────────────────────────────────
  //  Parameters
  // ──────────────────────────────────────────────────────────────
  localparam int LANES   = accel_pkg::LANES;  // 20
  localparam int K_MAX   = 7;
  localparam int CLK_NS  = 4;                 // 250 MHz

  // ──────────────────────────────────────────────────────────────
  //  DUT signals
  // ──────────────────────────────────────────────────────────────
  logic        clk, rst_n;
  logic [3:0]  cfg_k;
  logic        shift_in_valid;
  int8_t       shift_in [LANES];
  int8_t       taps [K_MAX][LANES];
  logic        taps_valid;
  logic        flush;

  // ──────────────────────────────────────────────────────────────
  //  DUT instantiation
  // ──────────────────────────────────────────────────────────────
  window_gen #(.LANES(LANES), .K_MAX(K_MAX)) u_dut (
    .clk            (clk),
    .rst_n          (rst_n),
    .cfg_k          (cfg_k),
    .shift_in_valid (shift_in_valid),
    .shift_in       (shift_in),
    .taps           (taps),
    .taps_valid     (taps_valid),
    .flush          (flush)
  );

  // ──────────────────────────────────────────────────────────────
  //  Clock generation: 250 MHz (4 ns period)
  // ──────────────────────────────────────────────────────────────
  initial clk = 1'b0;
  always #(CLK_NS/2) clk = ~clk;

  // ──────────────────────────────────────────────────────────────
  //  Scoreboard
  // ──────────────────────────────────────────────────────────────
  int pass_cnt = 0;
  int fail_cnt = 0;
  int test_cnt = 0;

  task automatic check(string tag, logic cond, string msg);
    test_cnt++;
    if (cond) begin
      pass_cnt++;
    end else begin
      fail_cnt++;
      $display("[FAIL] %s : %s", tag, msg);
    end
  endtask

  // Helper: shift in one row with pattern base_val + lane
  task automatic shift_row(input int base_val);
    @(posedge clk);
    shift_in_valid <= 1'b1;
    for (int l = 0; l < LANES; l++)
      shift_in[l] <= 8'(base_val + l);
    @(posedge clk);
    shift_in_valid <= 1'b0;
    #1;
  endtask

  // Helper: do a reset + flush
  task automatic do_flush();
    @(posedge clk);
    flush <= 1'b1;
    @(posedge clk);
    flush <= 1'b0;
    @(posedge clk);
    #1;
  endtask

  // ──────────────────────────────────────────────────────────────
  //  Main test sequence
  // ──────────────────────────────────────────────────────────────
  initial begin
    $display("===========================================================");
    $display(" tb_window_gen — START");
    $display("===========================================================");

    // Reset
    rst_n          = 1'b0;
    cfg_k          = 4'd3;
    shift_in_valid = 1'b0;
    flush          = 1'b0;
    for (int l = 0; l < LANES; l++)
      shift_in[l] = 8'sd0;
    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // ════════════════════════════════════════════════════════════
    //  T5.2.1: K=3 (Conv 3x3)
    //   Shift 3 rows in. Verify taps after filling.
    //   Shift chain: taps[K-1] ← ... ← taps[0] ← shift_in
    //   After shifting rows A, B, C:
    //     sr[0] = C (most recent), sr[1] = B, sr[2] = A (oldest)
    //     taps[0] = C, taps[1] = B, taps[2] = A
    // ════════════════════════════════════════════════════════════
    $display("\n--- T5.2.1: K=3 (Conv 3x3) ---");
    cfg_k = 4'd3;
    do_flush();

    // taps_valid should be 0 before filling
    check("T5.2.1_pre", taps_valid == 1'b0,
          $sformatf("taps_valid=%0d before fill, expected 0", taps_valid));

    // Shift row 0 (base=10), row 1 (base=20), row 2 (base=30)
    shift_row(10);
    check("T5.2.1_fill1", taps_valid == 1'b0,
          "After 1 shift, taps_valid should be 0 (need 3)");

    shift_row(20);
    check("T5.2.1_fill2", taps_valid == 1'b0,
          "After 2 shifts, taps_valid should be 0 (need 3)");

    // On the 3rd shift, taps_valid should assert (during the shift_in_valid cycle)
    @(posedge clk);
    shift_in_valid <= 1'b1;
    for (int l = 0; l < LANES; l++)
      shift_in[l] <= 8'(30 + l);
    @(posedge clk);
    #1;
    check("T5.2.1_valid", taps_valid == 1'b1,
          $sformatf("After 3 shifts taps_valid=%0d expected 1", taps_valid));

    // Keep shift_in_valid high for one more to verify taps content
    // After shifts of 10, 20, 30:
    //   sr[0]=30+l, sr[1]=20+l, sr[2]=10+l
    shift_in_valid <= 1'b0;
    @(posedge clk);
    #1;

    // Now verify taps content (taps = sr)
    // After shifting 10, 20, 30:
    //   sr[0] = last shifted = 30+l, sr[1] = 20+l, sr[2] = 10+l
    begin
      automatic logic ok0 = 1'b1, ok1 = 1'b1, ok2 = 1'b1;
      for (int l = 0; l < LANES; l++) begin
        if (taps[0][l] != 8'(30 + l)) ok0 = 1'b0;
        if (taps[1][l] != 8'(20 + l)) ok1 = 1'b0;
        if (taps[2][l] != 8'(10 + l)) ok2 = 1'b0;
      end
      check("T5.2.1_tap0", ok0,
            $sformatf("taps[0][0]=%0d expected=%0d", taps[0][0], 30));
      check("T5.2.1_tap1", ok1,
            $sformatf("taps[1][0]=%0d expected=%0d", taps[1][0], 20));
      check("T5.2.1_tap2", ok2,
            $sformatf("taps[2][0]=%0d expected=%0d", taps[2][0], 10));
    end

    // ════════════════════════════════════════════════════════════
    //  T5.2.2: K=5 (MaxPool 5x5)
    //   Shift 5 rows, verify taps[0..4]
    // ════════════════════════════════════════════════════════════
    $display("\n--- T5.2.2: K=5 (MaxPool) ---");
    cfg_k = 4'd5;
    do_flush();

    // Shift rows base = 10, 20, 30, 40, 50
    for (int r = 1; r <= 4; r++)
      shift_row(r * 10);

    // 4th shift done, not yet valid (need 5)
    check("T5.2.2_fill4", taps_valid == 1'b0,
          "After 4 shifts with K=5, taps_valid should be 0");

    // 5th shift — valid during this shift
    @(posedge clk);
    shift_in_valid <= 1'b1;
    for (int l = 0; l < LANES; l++)
      shift_in[l] <= 8'(50 + l);
    @(posedge clk);
    #1;
    check("T5.2.2_valid", taps_valid == 1'b1,
          $sformatf("After 5 shifts taps_valid=%0d expected 1", taps_valid));
    shift_in_valid <= 1'b0;
    @(posedge clk);
    #1;

    // Verify: sr[0]=50+l, sr[1]=40+l, ..., sr[4]=10+l
    begin
      automatic logic ok [5];
      for (int t = 0; t < 5; t++) ok[t] = 1'b1;
      for (int t = 0; t < 5; t++)
        for (int l = 0; l < LANES; l++)
          if (taps[t][l] != 8'((5 - t) * 10 + l)) ok[t] = 1'b0;
      for (int t = 0; t < 5; t++)
        check($sformatf("T5.2.2_tap%0d", t), ok[t],
              $sformatf("taps[%0d][0]=%0d expected=%0d",
                        t, taps[t][0], (5 - t) * 10));
    end

    // ════════════════════════════════════════════════════════════
    //  T5.2.3: K=7 (DW 7x7)
    //   Shift 7 rows, verify taps[0..6]
    // ════════════════════════════════════════════════════════════
    $display("\n--- T5.2.3: K=7 (DW 7x7) ---");
    cfg_k = 4'd7;
    do_flush();

    // Shift rows base = 10, 20, ..., 70
    for (int r = 1; r <= 6; r++)
      shift_row(r * 10);

    check("T5.2.3_fill6", taps_valid == 1'b0,
          "After 6 shifts with K=7, taps_valid should be 0");

    // 7th shift
    @(posedge clk);
    shift_in_valid <= 1'b1;
    for (int l = 0; l < LANES; l++)
      shift_in[l] <= 8'(70 + l);
    @(posedge clk);
    #1;
    check("T5.2.3_valid", taps_valid == 1'b1,
          $sformatf("After 7 shifts taps_valid=%0d expected 1", taps_valid));
    shift_in_valid <= 1'b0;
    @(posedge clk);
    #1;

    // Verify: sr[0]=70+l, ..., sr[6]=10+l
    begin
      automatic logic ok [7];
      for (int t = 0; t < 7; t++) ok[t] = 1'b1;
      for (int t = 0; t < 7; t++)
        for (int l = 0; l < LANES; l++)
          if (taps[t][l] != 8'((7 - t) * 10 + l)) ok[t] = 1'b0;
      for (int t = 0; t < 7; t++)
        check($sformatf("T5.2.3_tap%0d", t), ok[t],
              $sformatf("taps[%0d][0]=%0d expected=%0d",
                        t, taps[t][0], (7 - t) * 10));
    end

    // ════════════════════════════════════════════════════════════
    //  T5.2.4: K=1 (Conv 1x1)
    //   shift 1 → tap[0] = input immediately
    // ════════════════════════════════════════════════════════════
    $display("\n--- T5.2.4: K=1 (Conv 1x1) ---");
    cfg_k = 4'd1;
    do_flush();

    // Shift one row (base=99)
    @(posedge clk);
    shift_in_valid <= 1'b1;
    for (int l = 0; l < LANES; l++)
      shift_in[l] <= 8'(99 + l);
    @(posedge clk);
    #1;
    check("T5.2.4_valid", taps_valid == 1'b1,
          $sformatf("K=1: taps_valid=%0d expected 1 after 1 shift", taps_valid));
    shift_in_valid <= 1'b0;
    @(posedge clk);
    #1;

    // taps[0] = input
    begin
      automatic logic ok = 1'b1;
      for (int l = 0; l < LANES; l++)
        if (taps[0][l] != 8'(99 + l)) ok = 1'b0;
      check("T5.2.4_tap0", ok,
            $sformatf("K=1 taps[0][0]=%0d expected=99", taps[0][0]));
    end

    // ════════════════════════════════════════════════════════════
    //  T5.2.5: Flush + refill
    //   Fill K=3, flush mid-stream, refill, verify
    // ════════════════════════════════════════════════════════════
    $display("\n--- T5.2.5: Flush + refill ---");
    cfg_k = 4'd3;
    do_flush();

    // Fill 3 rows
    shift_row(10);
    shift_row(20);

    @(posedge clk);
    shift_in_valid <= 1'b1;
    for (int l = 0; l < LANES; l++)
      shift_in[l] <= 8'(30 + l);
    @(posedge clk);
    #1;
    check("T5.2.5_pre_flush_valid", taps_valid == 1'b1,
          "Before flush: taps_valid should be 1 after 3 shifts");
    shift_in_valid <= 1'b0;

    // Now flush
    do_flush();

    // After flush, taps_valid must be 0 and shift registers cleared
    check("T5.2.5_post_flush_valid", taps_valid == 1'b0,
          $sformatf("After flush taps_valid=%0d expected 0", taps_valid));

    begin
      automatic logic all_zero = 1'b1;
      for (int t = 0; t < K_MAX; t++)
        for (int l = 0; l < LANES; l++)
          if (taps[t][l] != 8'sd0) all_zero = 1'b0;
      check("T5.2.5_post_flush_zero", all_zero,
            "After flush all taps should be zero");
    end

    // Refill with new data (base=50, 60, 70)
    shift_row(50);
    shift_row(60);

    @(posedge clk);
    shift_in_valid <= 1'b1;
    for (int l = 0; l < LANES; l++)
      shift_in[l] <= 8'(70 + l);
    @(posedge clk);
    #1;
    check("T5.2.5_refill_valid", taps_valid == 1'b1,
          $sformatf("After refill taps_valid=%0d expected 1", taps_valid));
    shift_in_valid <= 1'b0;
    @(posedge clk);
    #1;

    // Verify refilled taps: sr[0]=70+l, sr[1]=60+l, sr[2]=50+l
    begin
      automatic logic ok0 = 1'b1, ok1 = 1'b1, ok2 = 1'b1;
      for (int l = 0; l < LANES; l++) begin
        if (taps[0][l] != 8'(70 + l)) ok0 = 1'b0;
        if (taps[1][l] != 8'(60 + l)) ok1 = 1'b0;
        if (taps[2][l] != 8'(50 + l)) ok2 = 1'b0;
      end
      check("T5.2.5_refill_tap0", ok0,
            $sformatf("Refill taps[0][0]=%0d expected=70", taps[0][0]));
      check("T5.2.5_refill_tap1", ok1,
            $sformatf("Refill taps[1][0]=%0d expected=60", taps[1][0]));
      check("T5.2.5_refill_tap2", ok2,
            $sformatf("Refill taps[2][0]=%0d expected=50", taps[2][0]));
    end

    // ════════════════════════════════════════════════════════════
    //  Summary
    // ════════════════════════════════════════════════════════════
    $display("\n===========================================================");
    $display(" tb_window_gen — RESULTS");
    $display("   Total : %0d", test_cnt);
    $display("   PASS  : %0d", pass_cnt);
    $display("   FAIL  : %0d", fail_cnt);
    if (fail_cnt == 0)
      $display("   >>> ALL TESTS PASSED <<<");
    else
      $display("   >>> SOME TESTS FAILED <<<");
    $display("===========================================================");
    $finish;
  end

endmodule
