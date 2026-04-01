// ============================================================================
// Testbench : tb_addr_gen_output
// Project   : YOLOv10n INT8 Accelerator — V4-VC707
// Tests     : T4.3.1 Bank mapping: bank_id[col] = col for all 4 columns
//             T4.3.2 Address calculation for various (h_out, wblk, cout_group)
// ============================================================================
`timescale 1ns / 1ps

module tb_addr_gen_output;
  import accel_pkg::*;

  // ──────────────────────────────────────────────────────────────
  //  Parameters
  // ──────────────────────────────────────────────────────────────
  localparam int LANES    = accel_pkg::LANES;    // 20
  localparam int PE_COLS_ = accel_pkg::PE_COLS;  // 4
  localparam int CLK_NS   = 4;                   // 250 MHz

  // ──────────────────────────────────────────────────────────────
  //  DUT signals
  // ──────────────────────────────────────────────────────────────
  logic        clk, rst_n;
  logic [9:0]  cfg_wout, cfg_cout;
  logic [9:0]  iter_h_out, iter_wblk, iter_cout_group;

  logic [1:0]  out_bank_id [PE_COLS_];
  logic [11:0] out_addr    [PE_COLS_];

  // ──────────────────────────────────────────────────────────────
  //  DUT instantiation
  // ──────────────────────────────────────────────────────────────
  addr_gen_output #(.LANES(LANES), .PE_COLS(PE_COLS_)) u_dut (
    .clk             (clk),
    .rst_n           (rst_n),
    .cfg_wout        (cfg_wout),
    .cfg_cout        (cfg_cout),
    .iter_h_out      (iter_h_out),
    .iter_wblk       (iter_wblk),
    .iter_cout_group (iter_cout_group),
    .out_bank_id     (out_bank_id),
    .out_addr        (out_addr)
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

  // Helper: apply inputs and allow combinational settle
  task automatic apply(
    input logic [9:0] h_out_v, wblk_v, cout_grp_v
  );
    @(posedge clk);
    iter_h_out      <= h_out_v;
    iter_wblk       <= wblk_v;
    iter_cout_group <= cout_grp_v;
    @(posedge clk);
    #1;
  endtask

  // ──────────────────────────────────────────────────────────────
  //  Main test sequence
  // ──────────────────────────────────────────────────────────────
  initial begin
    $display("===========================================================");
    $display(" tb_addr_gen_output — START");
    $display("===========================================================");

    // Reset
    rst_n = 1'b0;
    cfg_wout = '0; cfg_cout = '0;
    iter_h_out = '0; iter_wblk = '0; iter_cout_group = '0;
    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // ════════════════════════════════════════════════════════════
    //  T4.3.1: Bank mapping — bank_id[col] = col for all 4 columns
    //   Verify across multiple config and iteration combos
    // ════════════════════════════════════════════════════════════
    $display("\n--- T4.3.1: Bank mapping ---");

    // Config 1: wout=320, cout=16
    cfg_wout = 10'd320;
    cfg_cout = 10'd16;

    apply(10'd0, 10'd0, 10'd0);
    for (int col = 0; col < 4; col++) begin
      check("T4.3.1a",
            out_bank_id[col] == col[1:0],
            $sformatf("col=%0d: bank_id=%0d expected=%0d",
                      col, out_bank_id[col], col));
    end

    // Config 2: wout=640, cout=64 — different config same bank rule
    cfg_wout = 10'd640;
    cfg_cout = 10'd64;

    apply(10'd5, 10'd10, 10'd3);
    for (int col = 0; col < 4; col++) begin
      check("T4.3.1b",
            out_bank_id[col] == col[1:0],
            $sformatf("cfg2 col=%0d: bank_id=%0d expected=%0d",
                      col, out_bank_id[col], col));
    end

    // Config 3: wout=20, cout=4 — minimal
    cfg_wout = 10'd20;
    cfg_cout = 10'd4;

    apply(10'd0, 10'd0, 10'd0);
    for (int col = 0; col < 4; col++) begin
      check("T4.3.1c",
            out_bank_id[col] == col[1:0],
            $sformatf("cfg3 col=%0d: bank_id=%0d expected=%0d",
                      col, out_bank_id[col], col));
    end

    // ════════════════════════════════════════════════════════════
    //  T4.3.2: Address calculation
    //   addr = h_out * cout_groups_total * wblk_total
    //        + cout_group * wblk_total + wblk
    //
    //   wblk_total = ceil(wout / LANES)
    //   cout_groups_total = ceil(cout / PE_COLS)
    //
    //   All columns get the SAME address (they write to different banks).
    // ════════════════════════════════════════════════════════════
    $display("\n--- T4.3.2: Address calculation ---");

    // Case A: wout=320, cout=16
    //   wblk_total = ceil(320/20) = 16
    //   cout_groups_total = ceil(16/4) = 4
    cfg_wout = 10'd320;
    cfg_cout = 10'd16;

    begin
      automatic int wblk_total       = 16;
      automatic int cout_groups_total = 4;

      // (h_out=0, cout_group=0, wblk=0) → addr = 0
      apply(10'd0, 10'd0, 10'd0);
      begin
        automatic int exp = 0;
        for (int col = 0; col < 4; col++) begin
          check("T4.3.2_A0",
                out_addr[col] == exp[11:0],
                $sformatf("A h=0,cg=0,w=0 col%0d: addr=%0d expected=%0d",
                          col, out_addr[col], exp));
        end
      end

      // (h_out=0, cout_group=1, wblk=0) → addr = 1*16 = 16
      apply(10'd0, 10'd0, 10'd1);
      begin
        automatic int exp = 1 * wblk_total;  // 16
        for (int col = 0; col < 4; col++) begin
          check("T4.3.2_A1",
                out_addr[col] == exp[11:0],
                $sformatf("A h=0,cg=1,w=0 col%0d: addr=%0d expected=%0d",
                          col, out_addr[col], exp));
        end
      end

      // (h_out=0, cout_group=0, wblk=5) → addr = 5
      apply(10'd0, 10'd5, 10'd0);
      begin
        automatic int exp = 5;
        for (int col = 0; col < 4; col++) begin
          check("T4.3.2_A2",
                out_addr[col] == exp[11:0],
                $sformatf("A h=0,cg=0,w=5 col%0d: addr=%0d expected=%0d",
                          col, out_addr[col], exp));
        end
      end

      // (h_out=1, cout_group=2, wblk=3) → addr = 1*4*16 + 2*16 + 3 = 64+32+3 = 99
      apply(10'd1, 10'd3, 10'd2);
      begin
        automatic int exp = 1 * cout_groups_total * wblk_total
                          + 2 * wblk_total + 3;  // 64 + 32 + 3 = 99
        for (int col = 0; col < 4; col++) begin
          check("T4.3.2_A3",
                out_addr[col] == exp[11:0],
                $sformatf("A h=1,cg=2,w=3 col%0d: addr=%0d expected=%0d",
                          col, out_addr[col], exp));
        end
      end

      // (h_out=5, cout_group=3, wblk=15) → addr = 5*64 + 3*16 + 15 = 320+48+15 = 383
      apply(10'd5, 10'd15, 10'd3);
      begin
        automatic int exp = 5 * cout_groups_total * wblk_total
                          + 3 * wblk_total + 15;  // 320 + 48 + 15 = 383
        for (int col = 0; col < 4; col++) begin
          check("T4.3.2_A4",
                out_addr[col] == exp[11:0],
                $sformatf("A h=5,cg=3,w=15 col%0d: addr=%0d expected=%0d",
                          col, out_addr[col], exp));
        end
      end
    end

    // Case B: wout=640, cout=64
    //   wblk_total = ceil(640/20) = 32
    //   cout_groups_total = ceil(64/4) = 16
    cfg_wout = 10'd640;
    cfg_cout = 10'd64;

    begin
      automatic int wblk_total       = 32;
      automatic int cout_groups_total = 16;

      // (h_out=2, cout_group=5, wblk=10) → addr = 2*16*32 + 5*32 + 10
      //   = 1024 + 160 + 10 = 1194
      apply(10'd2, 10'd10, 10'd5);
      begin
        automatic int exp = 2 * cout_groups_total * wblk_total
                          + 5 * wblk_total + 10;  // 1194
        for (int col = 0; col < 4; col++) begin
          check("T4.3.2_B0",
                out_addr[col] == exp[11:0],
                $sformatf("B h=2,cg=5,w=10 col%0d: addr=%0d expected=%0d",
                          col, out_addr[col], exp));
        end
      end
    end

    // Case C: wout=20, cout=4 — minimal (wblk_total=1, cout_groups_total=1)
    cfg_wout = 10'd20;
    cfg_cout = 10'd4;

    begin
      automatic int wblk_total       = 1;
      automatic int cout_groups_total = 1;

      // (h_out=3, cout_group=0, wblk=0) → addr = 3*1*1 + 0 + 0 = 3
      apply(10'd3, 10'd0, 10'd0);
      begin
        automatic int exp = 3;
        for (int col = 0; col < 4; col++) begin
          check("T4.3.2_C0",
                out_addr[col] == exp[11:0],
                $sformatf("C h=3,cg=0,w=0 col%0d: addr=%0d expected=%0d",
                          col, out_addr[col], exp));
        end
      end
    end

    // ════════════════════════════════════════════════════════════
    //  Summary
    // ════════════════════════════════════════════════════════════
    $display("\n===========================================================");
    $display(" tb_addr_gen_output — RESULTS");
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
