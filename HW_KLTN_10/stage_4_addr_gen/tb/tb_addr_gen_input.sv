// ============================================================================
// Testbench : tb_addr_gen_input
// Project   : YOLOv10n INT8 Accelerator — V4-VC707
// Tests     : T4.1.1 Banking pattern
//             T4.1.2 Padding detection (pad_value MUST = cfg_zp_x)
//             T4.1.3 L0 config sweep (hin=640, win=640, stride=2, pad=1)
// ============================================================================
`timescale 1ns / 1ps

module tb_addr_gen_input;
  import accel_pkg::*;

  // ──────────────────────────────────────────────────────────────
  //  Parameters
  // ──────────────────────────────────────────────────────────────
  localparam int LANES    = accel_pkg::LANES;  // 20
  localparam int CLK_NS   = 4;                 // 250 MHz

  // ──────────────────────────────────────────────────────────────
  //  DUT signals
  // ──────────────────────────────────────────────────────────────
  logic        clk, rst_n;
  logic [9:0]  cfg_hin, cfg_win, cfg_cin;
  logic [2:0]  cfg_stride, cfg_padding;
  int8_t       cfg_zp_x;
  logic [9:0]  iter_h_out, iter_wblk, iter_cin;
  logic [3:0]  iter_kh_row;

  logic [1:0]  bank_id;
  logic [11:0] sram_addr;
  logic        is_padding;
  int8_t       pad_value;

  // ──────────────────────────────────────────────────────────────
  //  DUT instantiation
  // ──────────────────────────────────────────────────────────────
  addr_gen_input #(.LANES(LANES)) u_dut (
    .clk          (clk),
    .rst_n        (rst_n),
    .cfg_hin      (cfg_hin),
    .cfg_win      (cfg_win),
    .cfg_cin      (cfg_cin),
    .cfg_stride   (cfg_stride),
    .cfg_padding  (cfg_padding),
    .cfg_zp_x     (cfg_zp_x),
    .iter_h_out   (iter_h_out),
    .iter_wblk    (iter_wblk),
    .iter_cin     (iter_cin),
    .iter_kh_row  (iter_kh_row),
    .bank_id      (bank_id),
    .sram_addr    (sram_addr),
    .is_padding   (is_padding),
    .pad_value    (pad_value)
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

  // Helper: apply inputs and wait 1 cycle for registered output
  task automatic apply(
    input logic [9:0] h_out_v, wblk_v, cin_v,
    input logic [3:0] kh_row_v
  );
    @(posedge clk);
    iter_h_out  <= h_out_v;
    iter_wblk   <= wblk_v;
    iter_cin    <= cin_v;
    iter_kh_row <= kh_row_v;
    @(posedge clk);  // wait for registered output
    #1;
  endtask

  // ──────────────────────────────────────────────────────────────
  //  Main test sequence
  // ──────────────────────────────────────────────────────────────
  initial begin
    $display("===========================================================");
    $display(" tb_addr_gen_input — START");
    $display("===========================================================");

    // Reset
    rst_n       = 1'b0;
    cfg_hin     = '0; cfg_win = '0; cfg_cin = '0;
    cfg_stride  = '0; cfg_padding = '0; cfg_zp_x = 8'sd0;
    iter_h_out  = '0; iter_wblk = '0; iter_cin = '0; iter_kh_row = '0;
    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // ════════════════════════════════════════════════════════════
    //  T4.1.1: Banking pattern
    //   hin=8, stride=1, pad=0, kh_row=0 → h_in = h_out
    //   h_in=0→bank0, h_in=1→bank1, h_in=2→bank2, h_in=3→bank0
    // ════════════════════════════════════════════════════════════
    $display("\n--- T4.1.1: Banking pattern ---");
    cfg_hin    = 10'd8;
    cfg_win    = 10'd20;
    cfg_cin    = 10'd1;
    cfg_stride = 3'd1;
    cfg_padding= 3'd0;
    cfg_zp_x   = 8'sd0;

    begin
      logic [1:0] expected_bank [4] = '{2'd0, 2'd1, 2'd2, 2'd0};
      for (int h = 0; h < 4; h++) begin
        apply(h[9:0], 10'd0, 10'd0, 4'd0);
        check("T4.1.1",
              bank_id == expected_bank[h],
              $sformatf("h_in=%0d: bank_id=%0d, expected=%0d",
                        h, bank_id, expected_bank[h]));
      end
    end

    // ════════════════════════════════════════════════════════════
    //  T4.1.2: Padding detection
    //   hin=8, pad=1, stride=1, kh_row=0
    //   h_in = h_out * 1 + 0 - 1  → h_out=0 → h_in=-1 (padding)
    //   h_out=1 → h_in=0 (valid), h_out=9 → h_in=8 (padding)
    //   ★ pad_value MUST = cfg_zp_x (NOT 0)
    // ════════════════════════════════════════════════════════════
    $display("\n--- T4.1.2: Padding detection ---");
    cfg_hin    = 10'd8;
    cfg_win    = 10'd20;
    cfg_cin    = 10'd1;
    cfg_stride = 3'd1;
    cfg_padding= 3'd1;
    cfg_zp_x   = 8'sd42;  // Non-zero zero-point

    // h_out=0, kh=0 → h_in = 0*1 + 0 - 1 = -1 → PADDING
    apply(10'd0, 10'd0, 10'd0, 4'd0);
    check("T4.1.2a", is_padding == 1'b1,
          $sformatf("h_in=-1 should be padding, is_padding=%0d", is_padding));
    check("T4.1.2a_zp", pad_value == 8'sd42,
          $sformatf("pad_value=%0d, expected cfg_zp_x=42", pad_value));

    // h_out=9, kh=0 → h_in = 9*1 + 0 - 1 = 8 → PADDING (>=hin)
    apply(10'd9, 10'd0, 10'd0, 4'd0);
    check("T4.1.2b", is_padding == 1'b1,
          $sformatf("h_in=8 should be padding (hin=8), is_padding=%0d", is_padding));
    check("T4.1.2b_zp", pad_value == 8'sd42,
          $sformatf("pad_value=%0d, expected cfg_zp_x=42", pad_value));

    // h_out=1..8, kh=0 → h_in = 0..7 → NOT padding
    for (int h = 1; h <= 8; h++) begin
      apply(h[9:0], 10'd0, 10'd0, 4'd0);
      check("T4.1.2c", is_padding == 1'b0,
            $sformatf("h_out=%0d h_in=%0d should NOT be padding, is_padding=%0d",
                      h, h-1, is_padding));
    end

    // ★ Check pad_value with a negative zero-point
    cfg_zp_x = -8'sd5;
    apply(10'd0, 10'd0, 10'd0, 4'd0);
    check("T4.1.2d_zp_neg", pad_value == -8'sd5,
          $sformatf("pad_value=%0d, expected cfg_zp_x=-5", pad_value));

    // ════════════════════════════════════════════════════════════
    //  T4.1.3: L0 config sweep
    //   hin=640, win=640, stride=2, pad=1
    //   wblk_total = ceil(640/20) = 32
    //   h_in = h_out*2 + kh - 1
    //   Sample multiple (h_out, wblk, cin, kh) combinations
    // ════════════════════════════════════════════════════════════
    $display("\n--- T4.1.3: L0 config sweep ---");
    cfg_hin    = 10'd640;
    cfg_win    = 10'd640;
    cfg_cin    = 10'd3;
    cfg_stride = 3'd2;
    cfg_padding= 3'd1;
    cfg_zp_x   = 8'sd128;  // Typical unsigned-zero in signed view = -128

    begin
      // wblk_total = ceil(640/20) = 32
      automatic int wblk_total = 32;

      // Test point 1: h_out=0, kh=0 → h_in = 0*2+0-1 = -1 → padding
      apply(10'd0, 10'd0, 10'd0, 4'd0);
      check("T4.1.3_pad_top", is_padding == 1'b1,
            $sformatf("L0 h_out=0,kh=0: h_in=-1 should be padding"));

      // Test point 2: h_out=0, kh=1 → h_in = 0*2+1-1 = 0 → valid
      apply(10'd0, 10'd0, 10'd0, 4'd1);
      check("T4.1.3_valid0", is_padding == 1'b0,
            $sformatf("L0 h_out=0,kh=1: h_in=0 should be valid"));
      check("T4.1.3_bank0", bank_id == 2'd0,
            $sformatf("L0 h_in=0: bank=%0d expected=0", bank_id));

      // Test point 3: h_out=1, kh=0 → h_in = 1*2+0-1 = 1 → valid, bank=1
      apply(10'd1, 10'd5, 10'd2, 4'd0);
      check("T4.1.3_h1", is_padding == 1'b0,
            $sformatf("L0 h_out=1,kh=0: h_in=1 should be valid"));
      check("T4.1.3_h1_bank", bank_id == 2'd1,
            $sformatf("L0 h_in=1: bank=%0d expected=1", bank_id));

      // Test point 4: h_out=319, kh=2 → h_in = 319*2+2-1 = 639 → valid (last row)
      apply(10'd319, 10'd0, 10'd0, 4'd2);
      check("T4.1.3_last_valid", is_padding == 1'b0,
            $sformatf("L0 h_out=319,kh=2: h_in=639 should be valid"));

      // Test point 5: h_out=320, kh=0 → h_in = 320*2+0-1 = 639 → valid
      apply(10'd320, 10'd0, 10'd0, 4'd0);
      check("T4.1.3_h320", is_padding == 1'b0,
            $sformatf("L0 h_out=320,kh=0: h_in=639 should be valid"));

      // Test point 6: h_out=320, kh=2 → h_in = 320*2+2-1 = 641 → padding
      apply(10'd320, 10'd0, 10'd0, 4'd2);
      check("T4.1.3_pad_bot", is_padding == 1'b1,
            $sformatf("L0 h_out=320,kh=2: h_in=641 should be padding"));

      // Test point 7: wide position wblk=31 (last block), should be valid
      apply(10'd5, 10'd31, 10'd1, 4'd1);
      check("T4.1.3_wblk31", is_padding == 1'b0,
            $sformatf("L0 wblk=31 within 32 blocks should be valid"));

      // Test point 8: wblk=32 → w_in_base=32*20=640 >= win=640 → width padding
      apply(10'd5, 10'd32, 10'd1, 4'd1);
      check("T4.1.3_wblk32_pad", is_padding == 1'b1,
            $sformatf("L0 wblk=32: w_in_base=640 >= win=640 should be padding"));
    end

    // ════════════════════════════════════════════════════════════
    //  Summary
    // ════════════════════════════════════════════════════════════
    $display("\n===========================================================");
    $display(" tb_addr_gen_input — RESULTS");
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
