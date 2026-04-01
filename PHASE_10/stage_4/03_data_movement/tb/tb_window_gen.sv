`timescale 1ns/1ps
// ============================================================================
// Testbench: window_gen
// Verifies shift-register tap generation for K=1,3,5,7 kernel widths.
// K_MAX=7, LANES=32.
// ============================================================================
module tb_window_gen;
  import accel_pkg::*;

  // ---------- Parameters ----------
  localparam int LANES  = 32;
  localparam int K_MAX  = 7;
  localparam int CLK_HP = 5;

  // ---------- Signals ----------
  logic              clk, rst_n;
  logic              flush;
  logic [2:0]        cfg_kw;
  logic              shift_in_valid;
  logic signed [7:0] shift_in [LANES];

  logic              taps_valid;
  logic signed [7:0] taps [K_MAX][LANES];

  // ---------- DUT ----------
  window_gen #(.LANES(LANES), .K_MAX(K_MAX)) dut (
    .clk            (clk),
    .rst_n          (rst_n),
    .flush          (flush),
    .cfg_kw         (cfg_kw),
    .shift_in_valid (shift_in_valid),
    .shift_in       (shift_in),
    .taps_valid     (taps_valid),
    .taps           (taps)
  );

  // ---------- Clock ----------
  initial clk = 0;
  always #(CLK_HP) clk = ~clk;

  // ---------- Scoreboard ----------
  int pass_cnt = 0;
  int fail_cnt = 0;

  task automatic check(string tag, logic cond);
    if (cond) begin
      pass_cnt++;
    end else begin
      fail_cnt++;
      $display("[FAIL] %s @ %0t", tag, $time);
    end
  endtask

  // ---------- Helper: shift in one row with base value ----------
  // Each lane gets base_val + lane_idx (modular 8-bit)
  task automatic shift_row(input logic signed [7:0] base_val);
    @(posedge clk);
    shift_in_valid <= 1'b1;
    for (int l = 0; l < LANES; l++)
      shift_in[l] <= base_val + signed'(8'(l));
    @(posedge clk);
    shift_in_valid <= 1'b0;
  endtask

  // Helper: shift in with hold (valid stays 1 for 1 cycle, then check on next)
  task automatic shift_row_and_sample(input logic signed [7:0] base_val);
    @(posedge clk);
    shift_in_valid <= 1'b1;
    for (int l = 0; l < LANES; l++)
      shift_in[l] <= base_val + signed'(8'(l));
    @(posedge clk);
    shift_in_valid <= 1'b0;
    // taps are registered with combinational output, sample now
  endtask

  // ---------- Stimulus ----------
  initial begin
    $display("============================================================");
    $display("  TB: window_gen");
    $display("============================================================");

    rst_n          = 0;
    flush          = 0;
    cfg_kw         = 3'd3;
    shift_in_valid = 0;
    for (int l = 0; l < LANES; l++) shift_in[l] = 0;

    repeat (4) @(posedge clk);
    rst_n = 1;
    repeat (2) @(posedge clk);

    // ====== T1: K=3 (conv3x3) ======
    $display("\n--- T1: K=3 (conv3x3) ---");
    begin
      cfg_kw = 3'd3;
      @(posedge clk);

      // Shift in rows with base values 10, 20, 30, 40, 50
      // taps_valid should assert when fill_count >= 3 AND shift_in_valid
      // So it becomes valid on the 3rd shift cycle (fill_count goes to 3)

      // Row 0 (base=10)
      shift_row(8'sd10);
      check("T1 not valid after 1 row", taps_valid == 1'b0);

      // Row 1 (base=20)
      shift_row(8'sd20);
      check("T1 not valid after 2 rows", taps_valid == 1'b0);

      // Row 2 (base=30): fill_count becomes 3, and shift_in_valid=1 during this cycle
      // taps_valid = (fill_count >= cfg_kw) & shift_in_valid
      // fill_count goes from 2->3 at the posedge, so at the NEXT shift it should be valid
      shift_row(8'sd30);
      // fill_count is now 3 (registered). taps_valid = (3>=3) & shift_in_valid
      // But shift_in_valid was deasserted, so taps_valid=0 now.

      // Row 3 (base=40): now fill_count=3, shift_in_valid=1 -> taps_valid=1
      @(posedge clk);
      shift_in_valid <= 1'b1;
      for (int l = 0; l < LANES; l++) shift_in[l] <= 8'sd40 + signed'(8'(l));
      @(posedge clk);
      // At this posedge, fill_count=3 (already), shift_in_valid=1 -> taps_valid=1
      check("T1 valid on 4th shift", taps_valid == 1'b1);

      // After the shift: sr[0]=row3(40), sr[1]=row2(30), sr[2]=row1(20)
      // taps[0]=sr[0], taps[1]=sr[1], taps[2]=sr[2]
      // Check taps[0] = most recent (base=40)
      check("T1 taps[0][0]=40", taps[0][0] == 8'sd40);
      check("T1 taps[1][0]=30", taps[1][0] == 8'sd30);
      check("T1 taps[2][0]=20", taps[2][0] == 8'sd20);
      $display("  T1: taps[0][0]=%0d taps[1][0]=%0d taps[2][0]=%0d (exp 40,30,20)",
               taps[0][0], taps[1][0], taps[2][0]);
      shift_in_valid <= 1'b0;
      @(posedge clk);

      // Row 4 (base=50)
      @(posedge clk);
      shift_in_valid <= 1'b1;
      for (int l = 0; l < LANES; l++) shift_in[l] <= 8'sd50 + signed'(8'(l));
      @(posedge clk);
      check("T1 valid on 5th shift", taps_valid == 1'b1);
      // sr[0]=row4(50), sr[1]=row3(40), sr[2]=row2(30)
      check("T1 taps[0][0]=50", taps[0][0] == 8'sd50);
      check("T1 taps[1][0]=40", taps[1][0] == 8'sd40);
      check("T1 taps[2][0]=30", taps[2][0] == 8'sd30);
      shift_in_valid <= 1'b0;
      @(posedge clk);
    end

    // ====== T2: K=1 (conv1x1) ======
    $display("\n--- T2: K=1 (conv1x1) ---");
    begin
      // Flush and reconfigure
      flush <= 1'b1;
      @(posedge clk);
      flush <= 1'b0;
      cfg_kw <= 3'd1;
      @(posedge clk);
      @(posedge clk);

      // K=1: taps_valid when fill_count>=1 AND shift_in_valid
      // First shift: fill goes 0->1 at posedge. But taps_valid reads registered fill_count.
      // Actually: fill_count increments at posedge when shift_in_valid.
      // taps_valid = (fill_count >= 1) & shift_in_valid is combinational.
      // fill_count is registered. So on first shift cycle:
      //   At posedge: fill_count was 0, goes to 1.
      //   But taps_valid uses the NEW fill_count? No - fill_count is ff output.
      //   taps_valid = (fill_count >= cfg_kw) & shift_in_valid
      //   fill_count is the registered value. During the first shift cycle,
      //   fill_count=0 (old value), so taps_valid=0.
      //   After posedge: fill_count=1. On next shift, taps_valid=(1>=1)&1 = 1.

      // Shift row 0
      shift_row(8'sd77);
      // fill_count now 1. Not valid yet (valid requires shift_in_valid=1 too).

      // Shift row 1: fill_count=1 >= K=1, shift_in_valid=1 -> valid
      @(posedge clk);
      shift_in_valid <= 1'b1;
      for (int l = 0; l < LANES; l++) shift_in[l] <= 8'sd88 + signed'(8'(l));
      @(posedge clk);
      check("T2 K=1 valid", taps_valid == 1'b1);
      check("T2 K=1 taps[0][0]=88", taps[0][0] == 8'sd88);
      $display("  T2: K=1 taps[0][0]=%0d (exp 88) valid=%0b", taps[0][0], taps_valid);
      shift_in_valid <= 1'b0;
      @(posedge clk);
    end

    // ====== T3: K=5 ======
    $display("\n--- T3: K=5 ---");
    begin
      flush <= 1'b1;
      @(posedge clk);
      flush <= 1'b0;
      cfg_kw <= 3'd5;
      @(posedge clk);
      @(posedge clk);

      // Need 5 rows to fill, then valid on 6th shift
      for (int r = 0; r < 5; r++) begin
        shift_row(signed'(8'(r * 10 + 1)));
      end
      // fill_count = 5 now

      // 6th shift: fill_count=5 >= 5, shift_in_valid=1 -> valid
      @(posedge clk);
      shift_in_valid <= 1'b1;
      for (int l = 0; l < LANES; l++) shift_in[l] <= 8'sd51 + signed'(8'(l));
      @(posedge clk);
      check("T3 K=5 valid on 6th", taps_valid == 1'b1);

      // sr[0]=row5(51), sr[1]=row4(41), sr[2]=row3(31), sr[3]=row2(21), sr[4]=row1(11)
      check("T3 taps[0][0]=51", taps[0][0] == 8'sd51);
      check("T3 taps[1][0]=41", taps[1][0] == 8'sd41);
      check("T3 taps[2][0]=31", taps[2][0] == 8'sd31);
      check("T3 taps[3][0]=21", taps[3][0] == 8'sd21);
      check("T3 taps[4][0]=11", taps[4][0] == 8'sd11);
      $display("  T3: K=5 taps[0..4][0] = %0d %0d %0d %0d %0d (exp 51,41,31,21,11)",
               taps[0][0], taps[1][0], taps[2][0], taps[3][0], taps[4][0]);
      shift_in_valid <= 1'b0;
      @(posedge clk);
    end

    // ====== T4: K=7 (DW7x7) ======
    $display("\n--- T4: K=7 (DW7x7) ---");
    begin
      flush <= 1'b1;
      @(posedge clk);
      flush <= 1'b0;
      cfg_kw <= 3'd7;
      @(posedge clk);
      @(posedge clk);

      // Fill 7 rows: base values 5,15,25,35,45,55,65
      for (int r = 0; r < 7; r++)
        shift_row(signed'(8'(r * 10 + 5)));

      // fill_count=7. 8th shift triggers valid.
      @(posedge clk);
      shift_in_valid <= 1'b1;
      for (int l = 0; l < LANES; l++) shift_in[l] <= 8'sd75 + signed'(8'(l));
      @(posedge clk);
      check("T4 K=7 valid", taps_valid == 1'b1);

      // sr: [0]=75, [1]=65, [2]=55, [3]=45, [4]=35, [5]=25, [6]=15
      check("T4 taps[0][0]=75", taps[0][0] == 8'sd75);
      check("T4 taps[1][0]=65", taps[1][0] == 8'sd65);
      check("T4 taps[2][0]=55", taps[2][0] == 8'sd55);
      check("T4 taps[3][0]=45", taps[3][0] == 8'sd45);
      check("T4 taps[4][0]=35", taps[4][0] == 8'sd35);
      check("T4 taps[5][0]=25", taps[5][0] == 8'sd25);
      check("T4 taps[6][0]=15", taps[6][0] == 8'sd15);
      $display("  T4: K=7 taps[0..6][0] = %0d %0d %0d %0d %0d %0d %0d",
               taps[0][0], taps[1][0], taps[2][0], taps[3][0],
               taps[4][0], taps[5][0], taps[6][0]);

      // Shift 2 more rows (9th and 10th)
      shift_in_valid <= 1'b0;
      @(posedge clk);
      shift_row(8'sd85);
      @(posedge clk);
      shift_in_valid <= 1'b1;
      for (int l = 0; l < LANES; l++) shift_in[l] <= 8'sd95 + signed'(8'(l));
      @(posedge clk);
      check("T4 after 10 rows valid", taps_valid == 1'b1);
      // sr: [0]=95,[1]=85,[2]=75,[3]=65,[4]=55,[5]=45,[6]=35
      check("T4 10th taps[0][0]=95", taps[0][0] == 8'sd95);
      check("T4 10th taps[6][0]=35", taps[6][0] == 8'sd35);
      $display("  T4 after 10 rows: taps[0][0]=%0d taps[6][0]=%0d (exp 95,35)",
               taps[0][0], taps[6][0]);
      shift_in_valid <= 1'b0;
      @(posedge clk);
    end

    // ====== T5: Flush test ======
    $display("\n--- T5: Flush resets fill_count ---");
    begin
      // Currently K=7, fill_count=7+. Flush.
      flush <= 1'b1;
      @(posedge clk);
      flush <= 1'b0;
      cfg_kw <= 3'd3;
      @(posedge clk);
      @(posedge clk);

      // fill_count should be 0 after flush. Need 3 more rows.
      shift_row(8'sd11);
      check("T5 not valid after flush+1", taps_valid == 1'b0);

      shift_row(8'sd22);
      check("T5 not valid after flush+2", taps_valid == 1'b0);

      shift_row(8'sd33);
      // fill_count = 3 now

      // 4th shift -> valid
      @(posedge clk);
      shift_in_valid <= 1'b1;
      for (int l = 0; l < LANES; l++) shift_in[l] <= 8'sd44 + signed'(8'(l));
      @(posedge clk);
      check("T5 valid after flush + 3 fills + 1 shift", taps_valid == 1'b1);
      check("T5 taps[0][0]=44", taps[0][0] == 8'sd44);
      check("T5 taps[1][0]=33", taps[1][0] == 8'sd33);
      check("T5 taps[2][0]=22", taps[2][0] == 8'sd22);
      $display("  T5: After flush + refill: taps[0..2][0] = %0d %0d %0d (exp 44,33,22)",
               taps[0][0], taps[1][0], taps[2][0]);
      shift_in_valid <= 1'b0;
      @(posedge clk);
    end

    // ====== T6: Data integrity with known per-lane values ======
    $display("\n--- T6: Data integrity (per-lane pattern) ---");
    begin
      int err_cnt;
      err_cnt = 0;

      flush <= 1'b1;
      @(posedge clk);
      flush <= 1'b0;
      cfg_kw <= 3'd3;
      @(posedge clk);
      @(posedge clk);

      // Shift 3 rows: row_r lane_l = (r+1)*10 + l (mod 128 to stay positive signed)
      for (int r = 0; r < 3; r++) begin
        @(posedge clk);
        shift_in_valid <= 1'b1;
        for (int l = 0; l < LANES; l++)
          shift_in[l] <= signed'(8'(((r + 1) * 10 + l) % 128));
        @(posedge clk);
        shift_in_valid <= 1'b0;
      end

      // 4th shift with row3 pattern
      @(posedge clk);
      shift_in_valid <= 1'b1;
      for (int l = 0; l < LANES; l++)
        shift_in[l] <= signed'(8'((4 * 10 + l) % 128));
      @(posedge clk);

      check("T6 valid", taps_valid == 1'b1);

      // Verify every lane of every tap
      // taps[0] = most recent = row3 (base 40)
      // taps[1] = row2 (base 30)
      // taps[2] = row1 (base 20)
      for (int k = 0; k < 3; k++) begin
        logic signed [7:0] exp_base;
        exp_base = signed'(8'((4 - k) * 10));
        for (int l = 0; l < LANES; l++) begin
          logic signed [7:0] exp_val;
          exp_val = signed'(8'(((4 - k) * 10 + l) % 128));
          if (taps[k][l] != exp_val) begin
            err_cnt++;
            if (err_cnt <= 5)
              $display("  [FAIL] T6: taps[%0d][%0d]=%0d exp=%0d", k, l, taps[k][l], exp_val);
          end
          check($sformatf("T6 taps[%0d][%0d]", k, l), taps[k][l] == exp_val);
        end
      end

      shift_in_valid <= 1'b0;
      @(posedge clk);

      if (err_cnt == 0)
        $display("  T6: All %0d lane values verified correctly", 3 * LANES);
      else
        $display("  T6: %0d mismatches in lane data", err_cnt);
    end

    // ====== Summary ======
    repeat (4) @(posedge clk);
    $display("\n============================================================");
    $display("  window_gen: %0d PASSED, %0d FAILED", pass_cnt, fail_cnt);
    if (fail_cnt == 0)
      $display("  >>> ALL TESTS PASSED <<<");
    else
      $display("  >>> SOME TESTS FAILED <<<");
    $display("============================================================");
    $finish;
  end

endmodule
