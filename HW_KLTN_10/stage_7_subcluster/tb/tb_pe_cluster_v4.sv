// ============================================================================
// Testbench : tb_pe_cluster_v4
// Project   : YOLOv10n INT8 Accelerator — V4-VC707
// Tests     : T7.1.1  4 independent cout: SAME act, 4 DIFFERENT weights
//             T7.1.2  DW mode: 4 independent channels
//             T7.1.3  MaxPool bypass (PE_MP5)
//
// NOTE: pe_cluster_v4 instantiates pe_unit and column_reduce internally.
//       This TB feeds act_taps and wgt_data directly.
//       LANES=20, PE_ROWS=3, PE_COLS=4.
// ============================================================================
`timescale 1ns / 1ps

module tb_pe_cluster_v4;
  import accel_pkg::*;

  // ────────────────────────────────────────────────────────────
  // Parameters
  // ────────────────────────────────────────────────────────────
  localparam int L = LANES;    // 20
  localparam int R = PE_ROWS;  // 3
  localparam int C = PE_COLS;  // 4

  // ────────────────────────────────────────────────────────────
  // Clock & reset
  // ────────────────────────────────────────────────────────────
  logic clk, rst_n;
  initial clk = 0;
  always #2 clk = ~clk;  // 4 ns => 250 MHz

  // ────────────────────────────────────────────────────────────
  // DUT signals
  // ────────────────────────────────────────────────────────────
  pe_mode_e      pe_mode;
  logic          pe_enable;
  logic          pe_clear_acc;

  int8_t         act_taps    [R][C][L];
  int8_t         wgt_data    [R][C][L];

  int32_t        col_psum    [C][L];
  logic          psum_valid;

  int32_t        psum_accum_in [C][L];
  logic          psum_accum_en;

  int8_t         pool_window [25][L];
  logic          pool_enable;
  int8_t         pool_max    [L];
  logic          pool_valid;

  // ────────────────────────────────────────────────────────────
  // DUT instantiation
  // ────────────────────────────────────────────────────────────
  pe_cluster_v4 #(
    .LANES   (L),
    .PE_ROWS (R),
    .PE_COLS (C)
  ) u_dut (
    .clk            (clk),
    .rst_n          (rst_n),
    .pe_mode        (pe_mode),
    .pe_enable      (pe_enable),
    .pe_clear_acc   (pe_clear_acc),
    .act_taps       (act_taps),
    .wgt_data       (wgt_data),
    .col_psum       (col_psum),
    .psum_valid     (psum_valid),
    .psum_accum_in  (psum_accum_in),
    .psum_accum_en  (psum_accum_en),
    .pool_window    (pool_window),
    .pool_enable    (pool_enable),
    .pool_max       (pool_max),
    .pool_valid     (pool_valid)
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
    rst_n         <= 1'b0;
    pe_mode       <= PE_RS3;
    pe_enable     <= 1'b0;
    pe_clear_acc  <= 1'b0;
    psum_accum_en <= 1'b0;
    pool_enable   <= 1'b0;
    for (int r = 0; r < R; r++)
      for (int c = 0; c < C; c++)
        for (int l = 0; l < L; l++)
          act_taps[r][c][l] <= 8'sd0;
    for (int r = 0; r < R; r++)
      for (int c = 0; c < C; c++)
        for (int l = 0; l < L; l++)
          wgt_data[r][c][l] <= 8'sd0;
    for (int c = 0; c < C; c++)
      for (int l = 0; l < L; l++)
        psum_accum_in[c][l] <= 32'sd0;
    for (int i = 0; i < 25; i++)
      for (int l = 0; l < L; l++)
        pool_window[i][l] <= 8'sd0;
    repeat (4) @(posedge clk);
    rst_n <= 1'b1;
    @(posedge clk);
  endtask

  // ================================================================
  // T7.1.1 — 4 independent cout: SAME activation, 4 DIFFERENT weights
  //   Feed 9 cycles (3x3 kernel) in RS3 mode.
  //   Each column gets a different weight scalar (broadcast to all lanes).
  //   All activations = 1. Weights: col0=1, col1=2, col2=3, col3=4.
  //   After 9 cycles of accumulation (3 kw * 3 kh rows processed):
  //     Each PE accumulates act*wgt for all 9 feeding cycles.
  //     PE[row][col] accumulates 3 kw * act * wgt per cycle feed.
  //     But the sequencer feeds kw one at a time:
  //       9 feed cycles total (3 cin=1 loops * 3 kw).
  //     Each PE[r][c] accumulates: sum of 9 * (act * wgt) = 9 * 1 * wgt_c
  //     col_psum[c][l] = sum over 3 rows of PE[r][c] = 3 * 9 * wgt_c = 27 * wgt_c
  //
  //   Actually, in the cluster: act_taps[row] is the SAME for all cols.
  //   wgt_data[row][col] is DIFFERENT per col.
  //   PE[r][c] accumulates: sum of (act_taps[r][c][l] * wgt_data[r][c][l])
  //   With act=1, wgt_c for all r and l: PE[r][c][l] = 9 * 1 * wgt_c = 9*wgt_c
  //   column_reduce sums 3 rows: col_psum[c][l] = 3 * 9 * wgt_c = 27 * wgt_c
  //
  //   Expected: col_psum[0]=27, col_psum[1]=54, col_psum[2]=81, col_psum[3]=108
  // ================================================================
  task automatic test_T7_1_1();
    int8_t wgt_per_col [C] = '{8'sd1, 8'sd2, 8'sd3, 8'sd4};
    int exp_psum [C]       = '{27, 54, 81, 108};
    logic all_match;

    $display("\n===== T7.1.1: 4 independent cout (RS3 mode) =====");
    do_reset();

    pe_mode <= PE_RS3;

    // Clear accumulators
    pe_clear_acc <= 1'b1;
    @(posedge clk);
    pe_clear_acc <= 1'b0;
    @(posedge clk);

    // Feed 9 cycles: act=1 everywhere, wgt different per column
    for (int cyc = 0; cyc < 9; cyc++) begin
      for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
          for (int l = 0; l < L; l++)
            act_taps[r][c][l] <= 8'sd1;

      for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
          for (int l = 0; l < L; l++)
            wgt_data[r][c][l] <= wgt_per_col[c];

      pe_enable <= 1'b1;
      @(posedge clk);
    end
    pe_enable <= 1'b0;

    // Wait for pipeline drain: DSP_PIPE_DEPTH(5) + column_reduce(1) + margin
    repeat (DSP_PIPE_DEPTH + 5) @(posedge clk);

    // Check col_psum
    for (int c = 0; c < C; c++) begin
      all_match = 1'b1;
      for (int l = 0; l < L; l++) begin
        if (col_psum[c][l] !== exp_psum[c]) begin
          all_match = 1'b0;
          $display("  MISMATCH: col_psum[%0d][%0d] = %0d, expected %0d",
                   c, l, col_psum[c][l], exp_psum[c]);
          break;
        end
      end
      check($sformatf("T7.1.1-col%0d psum=%0d", c, exp_psum[c]), all_match);
    end

    // Verify columns are DIFFERENT from each other
    check("T7.1.1-e col0 != col1",
          col_psum[0][0] !== col_psum[1][0]);
    check("T7.1.1-f col1 != col2",
          col_psum[1][0] !== col_psum[2][0]);
    check("T7.1.1-g col2 != col3",
          col_psum[2][0] !== col_psum[3][0]);
  endtask

  // ================================================================
  // T7.1.2 — DW mode: 4 independent channels
  //   In depthwise, each column processes a different channel.
  //   Act is still multicast (same act_taps for all cols) but in a real
  //   DW scenario the router would route different channels.
  //   For this unit test: feed different activations per row to show
  //   row independence, and different weights per column.
  //
  //   Feed 3 cycles (kw=3). act_taps[row][lane] = row+1.
  //   wgt_data[row][col][lane] = col+1 for all rows.
  //   PE[r][c] accumulates 3 * (r+1)*(c+1) over 3 cycles.
  //   col_psum[c][l] = sum_r(3*(r+1)*(c+1)) = 3*(c+1)*sum_r(r+1)
  //                   = 3*(c+1)*(1+2+3) = 3*(c+1)*6 = 18*(c+1)
  //
  //   Expected: col_psum[0]=18, [1]=36, [2]=54, [3]=72
  // ================================================================
  task automatic test_T7_1_2();
    int exp_psum [C] = '{18, 36, 54, 72};
    logic all_match;

    $display("\n===== T7.1.2: DW mode (4 independent channels) =====");
    do_reset();

    pe_mode <= PE_DW3;

    // Clear
    pe_clear_acc <= 1'b1;
    @(posedge clk);
    pe_clear_acc <= 1'b0;
    @(posedge clk);

    // Feed 3 cycles (kw=3)
    for (int cyc = 0; cyc < 3; cyc++) begin
      for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
          for (int l = 0; l < L; l++)
            act_taps[r][c][l] <= int8_t'(r + 1);

      for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
          for (int l = 0; l < L; l++)
            wgt_data[r][c][l] <= int8_t'(c + 1);

      pe_enable <= 1'b1;
      @(posedge clk);
    end
    pe_enable <= 1'b0;

    // Wait for pipeline drain
    repeat (DSP_PIPE_DEPTH + 5) @(posedge clk);

    for (int c = 0; c < C; c++) begin
      all_match = 1'b1;
      for (int l = 0; l < L; l++) begin
        if (col_psum[c][l] !== exp_psum[c]) begin
          all_match = 1'b0;
          $display("  MISMATCH: col_psum[%0d][%0d] = %0d, expected %0d",
                   c, l, col_psum[c][l], exp_psum[c]);
          break;
        end
      end
      check($sformatf("T7.1.2-ch%0d psum=%0d", c, exp_psum[c]), all_match);
    end
  endtask

  // ================================================================
  // T7.1.3 — MaxPool bypass (PE_MP5 mode)
  //   Feed 25 values per lane into pool_window.
  //   Lane l: window values are {l, l-1, l-2, ...} with known max.
  //   Verify pool_max output. PE psum is ignored.
  //
  //   For simplicity: pool_window[i][l] = i - 12 (range -12 to +12).
  //   Expected max per lane: +12 (window[24]).
  //   For lane 0: override window[0] to +50 => max = +50.
  // ================================================================
  task automatic test_T7_1_3();
    integer wait_cnt;

    $display("\n===== T7.1.3: MaxPool bypass (PE_MP5) =====");
    do_reset();

    pe_mode <= PE_MP5;

    // Blocking '=' (same as tb_comparator_tree): NB drive can miss the same
    // posedge as valid_in, so comparator samples X/zero window data.
    for (int i = 0; i < 25; i++)
      for (int l = 0; l < L; l++)
        pool_window[i][l] = int8_t'(i - 12);

    pool_window[0][0]   = 8'sd50;
    pool_window[10][5]  = 8'sd100;

    // Pulse pool_enable; watch pool_valid during the pipeline (valid_out is a 1-cycle pulse).
    @(posedge clk);
    pool_enable <= 1'b1;
    @(posedge clk);
    pool_enable <= 1'b0;
    wait_cnt = 0;
    repeat (12) begin
      @(posedge clk);
      if (pool_valid)
        wait_cnt++;
    end
    check("T7.1.3-a saw pool_valid pulse during pipeline", wait_cnt >= 1);

    // Lane 0: max should be 50 (from window[0]) — max_out registers hold after pulse
    check($sformatf("T7.1.3-b lane0 max=%0d (exp 50)", pool_max[0]),
          pool_max[0] === 8'sd50);

    // Lane 5: max should be 100 (from window[10])
    check($sformatf("T7.1.3-c lane5 max=%0d (exp 100)", pool_max[5]),
          pool_max[5] === 8'sd100);

    // Generic lanes: max should be +12 (window[24])
    check($sformatf("T7.1.3-d lane1 max=%0d (exp 12)", pool_max[1]),
          pool_max[1] === 8'sd12);
    check($sformatf("T7.1.3-e lane10 max=%0d (exp 12)", pool_max[10]),
          pool_max[10] === 8'sd12);

    $display("  (PE psum during MP5 is don't-care — ignored as expected)");
  endtask

  // ────────────────────────────────────────────────────────────
  // Test runner
  // ────────────────────────────────────────────────────────────
  initial begin
    $display("========================================");
    $display(" tb_pe_cluster_v4 — Stage 7 Subcluster");
    $display("========================================");

`ifdef RTL_TRACE
    rtl_trace_pkg::rtl_trace_open("rtl_cycle_trace_s7_pe_cluster.log");
`endif

    test_T7_1_1();
    test_T7_1_2();
    test_T7_1_3();

    $display("\n========================================");
    $display(" SUMMARY: %0d tests, %0d PASS, %0d FAIL",
             pass_cnt + fail_cnt, pass_cnt, fail_cnt);
    if (fail_cnt == 0)
      $display(" >>> ALL TESTS PASSED <<<");
    else
      $display(" >>> SOME TESTS FAILED <<<");
    $display("========================================");
`ifdef RTL_TRACE
    rtl_trace_pkg::rtl_trace_close();
`endif
    $finish;
  end

  // Timeout
  initial begin
    #100000;
    $display("[TIMEOUT] Simulation exceeded 100 us");
`ifdef RTL_TRACE
    rtl_trace_pkg::rtl_trace_close();
`endif
    $finish;
  end

endmodule
