`timescale 1ns/1ps
// ============================================================================
// Testbench: addr_gen_input
// Verifies (h,w,c) -> (bank_id, addr, is_padding, pad_value) mapping.
// Banking rule: bank_id = h mod 3.
// ============================================================================
module tb_addr_gen_input;
  import accel_pkg::*;

  // ---------- Parameters ----------
  localparam int LANES     = 32;
  localparam int MAX_WIDTH = 640;
  localparam int MAX_CIN   = 256;
  localparam int CLK_HP    = 5;

  // ---------- Signals ----------
  logic              clk, rst_n;
  logic [9:0]        cfg_win;
  logic [8:0]        cfg_cin_tile;
  logic [3:0]        cfg_q_in;
  logic [3:0]        cfg_stride;
  logic [3:0]        cfg_pad_top, cfg_pad_bot, cfg_pad_left, cfg_pad_right;
  logic [9:0]        cfg_hin;
  logic signed [7:0] cfg_zp_x;

  logic              req_valid;
  logic [9:0]        req_h, req_w;
  logic [8:0]        req_c;

  logic              out_valid;
  logic [1:0]        out_bank_id;
  logic [15:0]       out_addr;
  logic              out_is_padding;
  logic signed [7:0] out_pad_value;

  // ---------- DUT ----------
  addr_gen_input #(
    .LANES     (LANES),
    .MAX_WIDTH (MAX_WIDTH),
    .MAX_CIN   (MAX_CIN)
  ) dut (
    .clk            (clk),
    .rst_n          (rst_n),
    .cfg_win        (cfg_win),
    .cfg_cin_tile   (cfg_cin_tile),
    .cfg_q_in       (cfg_q_in),
    .cfg_stride     (cfg_stride),
    .cfg_pad_top    (cfg_pad_top),
    .cfg_pad_bot    (cfg_pad_bot),
    .cfg_pad_left   (cfg_pad_left),
    .cfg_pad_right  (cfg_pad_right),
    .cfg_hin        (cfg_hin),
    .cfg_zp_x       (cfg_zp_x),
    .req_valid      (req_valid),
    .req_h          (req_h),
    .req_w          (req_w),
    .req_c          (req_c),
    .out_valid      (out_valid),
    .out_bank_id    (out_bank_id),
    .out_addr       (out_addr),
    .out_is_padding (out_is_padding),
    .out_pad_value  (out_pad_value)
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

  // ---------- Helper: drive one request and wait for output ----------
  task automatic drive_req(input logic [9:0] h, input logic [9:0] w, input logic [8:0] c);
    @(posedge clk);
    req_valid <= 1'b1;
    req_h     <= h;
    req_w     <= w;
    req_c     <= c;
    @(posedge clk);
    req_valid <= 1'b0;
    @(posedge clk);  // 1-cycle latency: output available here
  endtask

  // ---------- Compute expected address (mirror RTL formula) ----------
  function automatic logic [15:0] expected_addr(
    input logic [9:0] h, input logic [9:0] w, input logic [8:0] c,
    input logic [5:0] wblk_total, input logic [3:0] q_in, input logic [8:0] cin_tile
  );
    logic [9:0]  h_div3;
    logic [3:0]  row_slot;
    logic [5:0]  wblk;
    h_div3   = h / 3;
    row_slot = h_div3 % q_in;
    wblk     = w / LANES;
    return (16'(row_slot) * 16'(cin_tile) + 16'(c)) * 16'(wblk_total) + 16'(wblk);
  endfunction

  // ---------- Stimulus ----------
  initial begin
    $display("============================================================");
    $display("  TB: addr_gen_input");
    $display("============================================================");

    rst_n     = 0;
    req_valid = 0;
    req_h     = 0;
    req_w     = 0;
    req_c     = 0;

    // Default config (small layer for T1-T4)
    cfg_win       = 10'd80;
    cfg_hin       = 10'd40;
    cfg_cin_tile  = 9'd8;
    cfg_q_in      = 4'd4;
    cfg_stride    = 4'd1;
    cfg_pad_top   = 4'd1;
    cfg_pad_bot   = 4'd1;
    cfg_pad_left  = 4'd1;
    cfg_pad_right = 4'd1;
    cfg_zp_x      = -8'sd5;

    repeat (4) @(posedge clk);
    rst_n = 1;
    repeat (2) @(posedge clk);

    // ====== T1: Normal access (no padding) ======
    $display("\n--- T1: Normal access (no padding) ---");
    begin
      logic [5:0] wblk_total;
      logic [15:0] exp_addr;
      wblk_total = (cfg_win + LANES - 1) / LANES;

      drive_req(10'd3, 10'd10, 9'd5);
      check("T1 out_valid",      out_valid == 1'b1);
      check("T1 bank_id=h%3=0",  out_bank_id == 2'd0);  // 3%3=0
      check("T1 not padding",    out_is_padding == 1'b0);

      exp_addr = expected_addr(10'd3, 10'd10, 9'd5, wblk_total, cfg_q_in, cfg_cin_tile);
      check("T1 addr correct",   out_addr == exp_addr);
      $display("  T1: h=3 w=10 c=5 -> bank=%0d addr=0x%04h pad=%0b (exp addr=0x%04h)",
               out_bank_id, out_addr, out_is_padding, exp_addr);
    end

    // ====== T2: Padding detection ======
    $display("\n--- T2: Padding detection (h=0 top pad) ---");
    begin
      drive_req(10'd0, 10'd10, 9'd0);  // h=0 < pad_top=1
      check("T2 is_padding",     out_is_padding == 1'b1);
      check("T2 pad_value=zp_x", out_pad_value == cfg_zp_x);
      $display("  T2: h=0 -> is_padding=%0b pad_value=%0d (cfg_zp_x=%0d)",
               out_is_padding, out_pad_value, cfg_zp_x);

      // Also check w=0 left padding
      drive_req(10'd5, 10'd0, 9'd0);  // w=0 < pad_left=1
      check("T2 left pad",       out_is_padding == 1'b1);
      check("T2 left pad value", out_pad_value == cfg_zp_x);
      $display("  T2: w=0 -> is_padding=%0b pad_value=%0d",
               out_is_padding, out_pad_value);

      // Bottom padding: h >= hin - pad_bot = 40-1 = 39
      drive_req(10'd39, 10'd10, 9'd0);
      check("T2 bot pad",        out_is_padding == 1'b1);
      $display("  T2: h=39 (>=hin-pad_bot=39) -> is_padding=%0b", out_is_padding);

      // Right padding: w >= win - pad_right = 80-1 = 79
      drive_req(10'd5, 10'd79, 9'd0);
      check("T2 right pad",      out_is_padding == 1'b1);
      $display("  T2: w=79 (>=win-pad_right=79) -> is_padding=%0b", out_is_padding);
    end

    // ====== T3: Banking pattern: h=0..5 -> bank=0,1,2,0,1,2 ======
    $display("\n--- T3: Banking pattern h mod 3 ---");
    begin
      logic [1:0] expected_bank;
      for (int h = 0; h < 6; h++) begin
        expected_bank = h % 3;
        drive_req(10'(h), 10'd32, 9'd0);  // w=32 (inside valid region, not padding)
        check($sformatf("T3 h=%0d bank=%0d", h, expected_bank),
              out_bank_id == expected_bank);
        $display("  T3: h=%0d -> bank_id=%0d (expected %0d) %s",
                 h, out_bank_id, expected_bank,
                 (out_bank_id == expected_bank) ? "OK" : "MISMATCH");
      end
    end

    // ====== T4: Row slot calculation ======
    $display("\n--- T4: Row slot = (h/3) mod cfg_q_in ---");
    begin
      logic [3:0] exp_slot;
      logic [9:0] h_test;
      for (int i = 0; i < 12; i++) begin
        h_test   = 10'(i);
        exp_slot = (h_test / 3) % cfg_q_in;
        drive_req(h_test, 10'd32, 9'd0);

        // Reconstruct: addr formula uses row_slot
        // Check by verifying the full address matches expected
        begin
          logic [5:0] wblk_total_l;
          logic [15:0] exp_addr_l;
          wblk_total_l = (cfg_win + LANES - 1) / LANES;
          exp_addr_l = expected_addr(h_test, 10'd32, 9'd0, wblk_total_l, cfg_q_in, cfg_cin_tile);
          check($sformatf("T4 h=%0d slot=%0d addr", i, exp_slot),
                out_addr == exp_addr_l);
          $display("  T4: h=%0d -> row_slot=%0d addr=0x%04h (exp=0x%04h) %s",
                   i, exp_slot, out_addr, exp_addr_l,
                   (out_addr == exp_addr_l) ? "OK" : "MISMATCH");
        end
      end
    end

    // ====== T5: L0 realistic sweep ======
    $display("\n--- T5: L0 realistic sweep (cfg_win=640, cfg_hin=642, pad=1) ---");
    begin
      logic [5:0]  wblk_total_l0;
      logic [1:0]  exp_bank;
      logic [15:0] exp_addr_l0;
      logic        exp_pad;
      int          pad_cnt, valid_cnt;

      // Reconfigure for L0
      cfg_win       = 10'd640;
      cfg_hin       = 10'd642;
      cfg_cin_tile  = 9'd3;
      cfg_q_in      = 4'd8;
      cfg_stride    = 4'd1;
      cfg_pad_top   = 4'd1;
      cfg_pad_bot   = 4'd1;
      cfg_pad_left  = 4'd0;
      cfg_pad_right = 4'd0;
      cfg_zp_x      = -8'sd3;
      @(posedge clk);

      wblk_total_l0 = (10'd640 + LANES - 1) / LANES;
      pad_cnt   = 0;
      valid_cnt = 0;

      for (int h = 0; h < 642; h += 32) begin  // sample every 32nd row
        for (int w_idx = 0; w_idx < 2; w_idx++) begin  // w=0 and w=32
          for (int c = 0; c < 3; c++) begin
            logic [9:0] wval;
            wval = 10'(w_idx * 32);

            drive_req(10'(h), wval, 9'(c));

            exp_bank = h % 3;
            exp_pad  = (h < 1) || (h >= 641);  // pad_top=1, hin-pad_bot=642-1=641

            check($sformatf("T5 h=%0d w=%0d c=%0d bank", h, wval, c),
                  out_bank_id == exp_bank);
            check($sformatf("T5 h=%0d w=%0d c=%0d pad", h, wval, c),
                  out_is_padding == exp_pad);

            if (exp_pad) begin
              check($sformatf("T5 h=%0d pad_value", h),
                    out_pad_value == cfg_zp_x);
              pad_cnt++;
            end else begin
              exp_addr_l0 = expected_addr(10'(h), wval, 9'(c),
                                          wblk_total_l0, 4'd8, 9'd3);
              check($sformatf("T5 h=%0d w=%0d c=%0d addr", h, wval, c),
                    out_addr == exp_addr_l0);
              valid_cnt++;
            end
          end
        end
      end
      $display("  T5: Swept %0d padding + %0d valid positions", pad_cnt, valid_cnt);
    end

    // ====== Summary ======
    repeat (4) @(posedge clk);
    $display("\n============================================================");
    $display("  addr_gen_input: %0d PASSED, %0d FAILED", pass_cnt, fail_cnt);
    if (fail_cnt == 0)
      $display("  >>> ALL TESTS PASSED <<<");
    else
      $display("  >>> SOME TESTS FAILED <<<");
    $display("============================================================");
    $finish;
  end

endmodule
