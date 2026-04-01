// ============================================================================
// Testbench : tb_compute_sequencer
// Project   : YOLOv10n INT8 Accelerator — V4-VC707
// Tests     : T6.3.1  PE_RS3 iteration count (expected 72 pe_enable cycles)
//             T6.3.2  PE_OS1 iteration count (expected 64 pe_enable cycles)
//             T6.3.3  PE_DW3 iteration count (expected 12 pe_enable cycles)
//             T6.3.4  ppu_trigger timing (once per cout_group, cout_base check)
//             T6.3.5  LANES=20 exact division: W=320/160/80/40/20
// ============================================================================
`timescale 1ns / 1ps

module tb_compute_sequencer;
  import accel_pkg::*;

  // ────────────────────────────────────────────────────────────
  // Clock & reset
  // ────────────────────────────────────────────────────────────
  logic clk, rst_n;
  initial clk = 0;
  always #2 clk = ~clk;  // 4 ns => 250 MHz

  // ────────────────────────────────────────────────────────────
  // DUT signals
  // ────────────────────────────────────────────────────────────
  logic          seq_start, seq_done;
  pe_mode_e      cfg_pe_mode;
  logic [9:0]    cfg_cin, cfg_cout, cfg_hout, cfg_wout;
  logic [3:0]    cfg_kh, cfg_kw;
  logic [2:0]    cfg_stride;

  logic [9:0]    iter_h, iter_wblk, iter_cin, iter_cout_group;
  logic [3:0]    iter_kw, iter_kh_row;
  logic          pe_enable, pe_clear_acc, pe_acc_valid;
  logic          ppu_trigger;
  logic [9:0]    ppu_cout_base;
  logic          pool_enable;
  logic          mp5_shift_en, mp5_win_flush;
  logic [9:0]    agi_iter_cin_mux, ago_iter_cout_grp_mux;
  logic [3:0]    agi_iter_kh_mux;
  logic [9:0]    dbg_iter_mp5_ch;

  // ────────────────────────────────────────────────────────────
  // DUT instantiation
  // ────────────────────────────────────────────────────────────
  compute_sequencer u_dut (
    .clk             (clk),
    .rst_n           (rst_n),
    .seq_start       (seq_start),
    .seq_done        (seq_done),
    .cfg_pe_mode     (cfg_pe_mode),
    .cfg_cin         (cfg_cin),
    .cfg_cout        (cfg_cout),
    .cfg_hout        (cfg_hout),
    .cfg_wout        (cfg_wout),
    .cfg_kh          (cfg_kh),
    .cfg_kw          (cfg_kw),
    .cfg_stride      (cfg_stride),
    .iter_h          (iter_h),
    .iter_wblk       (iter_wblk),
    .iter_cin        (iter_cin),
    .iter_cout_group (iter_cout_group),
    .iter_kw         (iter_kw),
    .iter_kh_row     (iter_kh_row),
    .pe_enable       (pe_enable),
    .pe_clear_acc    (pe_clear_acc),
    .pe_acc_valid    (pe_acc_valid),
    .ppu_trigger     (ppu_trigger),
    .ppu_cout_base        (ppu_cout_base),
    .pool_enable          (pool_enable),
    .mp5_shift_en         (mp5_shift_en),
    .mp5_win_flush        (mp5_win_flush),
    .agi_iter_cin_mux     (agi_iter_cin_mux),
    .agi_iter_kh_mux      (agi_iter_kh_mux),
    .ago_iter_cout_grp_mux(ago_iter_cout_grp_mux),
    .dbg_iter_mp5_ch     (dbg_iter_mp5_ch)
  );

  // ────────────────────────────────────────────────────────────
  // Test infrastructure
  // ────────────────────────────────────────────────────────────
  int pass_cnt = 0;
  int fail_cnt = 0;

  task automatic check(input string tag, input logic cond);
    if (cond) begin
      $display("[PASS] %s", tag);
      pass_cnt++;
    end else begin
      $display("[FAIL] %s", tag);
      fail_cnt++;
    end
  endtask

  task automatic do_reset();
    rst_n     <= 1'b0;
    seq_start <= 1'b0;
    cfg_pe_mode <= PE_RS3;
    cfg_cin   <= 10'd0;
    cfg_cout  <= 10'd0;
    cfg_hout  <= 10'd0;
    cfg_wout  <= 10'd0;
    cfg_kh    <= 4'd0;
    cfg_kw    <= 4'd0;
    cfg_stride <= 3'd1;
    repeat (4) @(posedge clk);
    rst_n <= 1'b1;
    @(posedge clk);
  endtask

  // Run sequencer and count pe_enable + ppu_trigger cycles
  task automatic run_and_count(
    output int pe_count,
    output int ppu_count,
    input int max_cycles = 5000
  );
    pe_count  = 0;
    ppu_count = 0;

    // Pulse seq_start
    seq_start <= 1'b1;
    @(posedge clk);
    seq_start <= 1'b0;

    // Count until seq_done
    for (int cyc = 0; cyc < max_cycles; cyc++) begin
      @(posedge clk);
      if (pe_enable)   pe_count++;
      if (ppu_trigger) ppu_count++;
      if (seq_done) return;
    end
    $display("  WARNING: run_and_count timeout after %0d cycles", max_cycles);
  endtask

  // ================================================================
  // T6.3.1 — PE_RS3 iteration count
  //   Cin=3, Cout=16, Hout=2, Wout=20, kw=3
  //   Cout_groups = 16/4 = 4, Wblks = 20/20 = 1
  //   PE cycles = Hout * Wblks * Cout_groups * Cin * kw
  //             = 2 * 1 * 4 * 3 * 3 = 72
  // ================================================================
  task automatic test_T6_3_1();
    $display("\n===== T6.3.1: PE_RS3 iteration count =====");
    do_reset();

    cfg_pe_mode <= PE_RS3;
    cfg_cin     <= 10'd3;
    cfg_cout    <= 10'd16;
    cfg_hout    <= 10'd2;
    cfg_wout    <= 10'd20;
    cfg_kh      <= 4'd3;
    cfg_kw      <= 4'd3;
    cfg_stride  <= 3'd1;
    @(posedge clk);

    int pe_cnt, ppu_cnt;
    run_and_count(pe_cnt, ppu_cnt);

    $display("  PE_RS3: pe_enable cycles = %0d (expected 72)", pe_cnt);
    $display("  PE_RS3: ppu_trigger count = %0d (expected 8)", ppu_cnt);
    check("T6.3.1-a pe_cycles=72", pe_cnt == 72);
    // ppu_trigger = Hout * Wblks * Cout_groups = 2*1*4 = 8
    check("T6.3.1-b ppu_triggers=8", ppu_cnt == 8);
    check("T6.3.1-c seq_done fired", seq_done === 1'b1);
  endtask

  // ================================================================
  // T6.3.2 — PE_OS1 iteration count
  //   Cin=8, Cout=16, Hout=2, Wout=20. No kw loop.
  //   Cout_groups = 16/4 = 4, Wblks = 1
  //   PE cycles = 2 * 1 * 4 * 8 = 64
  // ================================================================
  task automatic test_T6_3_2();
    $display("\n===== T6.3.2: PE_OS1 iteration count =====");
    do_reset();

    cfg_pe_mode <= PE_OS1;
    cfg_cin     <= 10'd8;
    cfg_cout    <= 10'd16;
    cfg_hout    <= 10'd2;
    cfg_wout    <= 10'd20;
    cfg_kh      <= 4'd1;
    cfg_kw      <= 4'd1;
    cfg_stride  <= 3'd1;
    @(posedge clk);

    int pe_cnt, ppu_cnt;
    run_and_count(pe_cnt, ppu_cnt);

    $display("  PE_OS1: pe_enable cycles = %0d (expected 64)", pe_cnt);
    check("T6.3.2-a pe_cycles=64", pe_cnt == 64);
    // ppu = 2*1*4 = 8
    check("T6.3.2-b ppu_triggers=8", ppu_cnt == 8);
  endtask

  // ================================================================
  // T6.3.3 — PE_DW3 iteration count
  //   C=8, Hout=2, Wout=20, kw=3
  //   Ch_groups = 8/4 = 2, Wblks = 1
  //   PE cycles = Hout * Wblks * Ch_groups * kw = 2 * 1 * 2 * 3 = 12
  // ================================================================
  task automatic test_T6_3_3();
    $display("\n===== T6.3.3: PE_DW3 iteration count =====");
    do_reset();

    cfg_pe_mode <= PE_DW3;
    cfg_cin     <= 10'd8;
    cfg_cout    <= 10'd8;   // DW: cin=cout
    cfg_hout    <= 10'd2;
    cfg_wout    <= 10'd20;
    cfg_kh      <= 4'd3;
    cfg_kw      <= 4'd3;
    cfg_stride  <= 3'd1;
    @(posedge clk);

    int pe_cnt, ppu_cnt;
    run_and_count(pe_cnt, ppu_cnt);

    $display("  PE_DW3: pe_enable cycles = %0d (expected 12)", pe_cnt);
    check("T6.3.3-a pe_cycles=12", pe_cnt == 12);
    // ppu = 2*1*2 = 4
    check("T6.3.3-b ppu_triggers=4", ppu_cnt == 4);
  endtask

  // ================================================================
  // T6.3.4 — ppu_trigger timing
  //   Verify fires exactly once per cout_group.
  //   Verify ppu_cout_base = cout_group * 4.
  //   Config: PE_RS3, Cin=1, Cout=8, Hout=1, Wout=20, kw=1
  //   => Cout_groups=2, triggers should be at cout_base=0 and 4
  // ================================================================
  task automatic test_T6_3_4();
    $display("\n===== T6.3.4: ppu_trigger timing =====");
    do_reset();

    cfg_pe_mode <= PE_RS3;
    cfg_cin     <= 10'd1;
    cfg_cout    <= 10'd8;
    cfg_hout    <= 10'd1;
    cfg_wout    <= 10'd20;
    cfg_kh      <= 4'd3;
    cfg_kw      <= 4'd1;
    cfg_stride  <= 3'd1;
    @(posedge clk);

    // Pulse start
    seq_start <= 1'b1;
    @(posedge clk);
    seq_start <= 1'b0;

    // Collect ppu_trigger events
    logic [9:0] ppu_bases [$];
    int ppu_count = 0;

    for (int cyc = 0; cyc < 2000; cyc++) begin
      @(posedge clk);
      if (ppu_trigger) begin
        ppu_bases.push_back(ppu_cout_base);
        ppu_count++;
        $display("  ppu_trigger #%0d: ppu_cout_base = %0d", ppu_count, ppu_cout_base);
      end
      if (seq_done) break;
    end

    // Expected: 1 (Hout) * 1 (Wblks) * 2 (Cout_groups) = 2 triggers
    check("T6.3.4-a ppu_count=2", ppu_count == 2);
    if (ppu_bases.size() >= 2) begin
      check("T6.3.4-b base[0]=0",  ppu_bases[0] == 10'd0);
      check("T6.3.4-c base[1]=4",  ppu_bases[1] == 10'd4);
    end else begin
      check("T6.3.4-b base[0]=0 (insufficient triggers)", 0);
      check("T6.3.4-c base[1]=4 (insufficient triggers)", 0);
    end
  endtask

  // ================================================================
  // T6.3.5 — LANES=20 exact division
  //   W=320 => 16 blks, W=160 => 8 blks, W=80 => 4 blks,
  //   W=40 => 2 blks, W=20 => 1 blk
  //   Use PE_RS3 with Cin=1, Cout=4 (1 cout_group), kw=1, Hout=1
  //   to minimize total cycles. Expected pe_cycles = Wblks * 1 * 1 * 1 = Wblks
  // ================================================================
  task automatic test_T6_3_5();
    $display("\n===== T6.3.5: LANES=20 exact division =====");
    do_reset();

    int wout_vals [5]   = '{320, 160, 80, 40, 20};
    int expected_blks[5] = '{16,  8,   4,  2,  1};

    for (int i = 0; i < 5; i++) begin
      do_reset();

      cfg_pe_mode <= PE_RS3;
      cfg_cin     <= 10'd1;
      cfg_cout    <= 10'd4;   // 1 cout_group
      cfg_hout    <= 10'd1;
      cfg_wout    <= wout_vals[i][9:0];
      cfg_kh      <= 4'd3;
      cfg_kw      <= 4'd1;
      cfg_stride  <= 3'd1;
      @(posedge clk);

      int pe_cnt, ppu_cnt;
      run_and_count(pe_cnt, ppu_cnt, 10000);

      // PE cycles = Hout(1) * Wblks * Cout_groups(1) * Cin(1) * kw(1) = Wblks
      int exp = expected_blks[i];
      $display("  W=%0d: pe_enable=%0d (expected %0d), ppu=%0d (expected %0d)",
               wout_vals[i], pe_cnt, exp, ppu_cnt, exp);
      check($sformatf("T6.3.5-%0d W=%0d pe_cycles=%0d", i, wout_vals[i], exp),
            pe_cnt == exp);
    end
  endtask

  // ────────────────────────────────────────────────────────────
  // Test runner
  // ────────────────────────────────────────────────────────────
  initial begin
    $display("========================================");
    $display(" tb_compute_sequencer — Stage 6 Control");
    $display("========================================");

    test_T6_3_1();
    test_T6_3_2();
    test_T6_3_3();
    test_T6_3_4();
    test_T6_3_5();

    $display("\n========================================");
    $display(" SUMMARY: %0d tests, %0d PASS, %0d FAIL",
             pass_cnt + fail_cnt, pass_cnt, fail_cnt);
    if (fail_cnt == 0)
      $display(" >>> ALL TESTS PASSED <<<");
    else
      $display(" >>> SOME TESTS FAILED <<<");
    $display("========================================");
    $finish;
  end

  // Timeout
  initial begin
    #200000;
    $display("[TIMEOUT] Simulation exceeded 200 us");
    $finish;
  end

endmodule
