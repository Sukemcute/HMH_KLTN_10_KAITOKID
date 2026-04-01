// ============================================================================
// Testbench : tb_addr_gen_weight
// Project   : YOLOv10n INT8 Accelerator — V4-VC707
// Tests     : T4.2.1 PE_RS3 — 4 different cout addresses
//             T4.2.2 PE_OS1 — 4 different cout addresses (no kw)
//             T4.2.3 PE_DW3 — 4 different channel addresses
//             T4.2.4 Full L0 weight sweep (Cin=3, Cout=16, kw=3)
// ============================================================================
`timescale 1ns / 1ps

module tb_addr_gen_weight;
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
  pe_mode_e    cfg_pe_mode;
  logic [9:0]  cfg_cin, cfg_cout;
  logic [3:0]  cfg_kw;
  logic [9:0]  iter_cin, iter_cout_group;
  logic [3:0]  iter_kw, iter_kh_row;

  logic [15:0] wgt_addr [PE_COLS_];
  logic [1:0]  wgt_bank_id;

  // ──────────────────────────────────────────────────────────────
  //  DUT instantiation
  // ──────────────────────────────────────────────────────────────
  addr_gen_weight #(.LANES(LANES), .PE_COLS(PE_COLS_)) u_dut (
    .clk             (clk),
    .rst_n           (rst_n),
    .cfg_pe_mode     (cfg_pe_mode),
    .cfg_cin         (cfg_cin),
    .cfg_cout        (cfg_cout),
    .cfg_kw          (cfg_kw),
    .iter_cin        (iter_cin),
    .iter_cout_group (iter_cout_group),
    .iter_kw         (iter_kw),
    .iter_kh_row     (iter_kh_row),
    .wgt_addr        (wgt_addr),
    .wgt_bank_id     (wgt_bank_id)
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

  // Helper: apply inputs and wait 1 clock for combinational settle
  // (addr_gen_weight outputs are purely combinational, no register)
  task automatic apply(
    input logic [9:0] cin_v, cout_grp_v,
    input logic [3:0] kw_v, kh_v
  );
    @(posedge clk);
    iter_cin        <= cin_v;
    iter_cout_group <= cout_grp_v;
    iter_kw         <= kw_v;
    iter_kh_row     <= kh_v;
    @(posedge clk);
    #1;  // small delay for combinational propagation
  endtask

  // ──────────────────────────────────────────────────────────────
  //  Main test sequence
  // ──────────────────────────────────────────────────────────────
  initial begin
    $display("===========================================================");
    $display(" tb_addr_gen_weight — START");
    $display("===========================================================");

    // Reset
    rst_n = 1'b0;
    cfg_pe_mode = PE_RS3;
    cfg_cin = '0; cfg_cout = '0; cfg_kw = '0;
    iter_cin = '0; iter_cout_group = '0; iter_kw = '0; iter_kh_row = '0;
    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // ════════════════════════════════════════════════════════════
    //  T4.2.1: PE_RS3 — 4 different cout addresses
    //   Formula: addr[col] = (cout_group*4 + col) * Cin * Kw + cin * Kw + kw
    //   Cin=3, Cout=16, kw=3, cout_group=0, cin=0, kw=0
    //   addr[0]=(0)*3*3+0=0, addr[1]=(1)*3*3=9, addr[2]=(2)*9=18, addr[3]=(3)*9=27
    // ════════════════════════════════════════════════════════════
    $display("\n--- T4.2.1: PE_RS3 — 4 different cout addresses ---");
    cfg_pe_mode = PE_RS3;
    cfg_cin     = 10'd3;
    cfg_cout    = 10'd16;
    cfg_kw      = 4'd3;

    apply(10'd0, 10'd0, 4'd0, 4'd0);

    begin
      // cout_or_ch[col] = cout_group*4 + col = 0,1,2,3
      // addr[col] = cout_or_ch * cin * kw + cin_iter * kw + kw_iter
      //           = col * 3 * 3 + 0*3 + 0 = col * 9
      automatic int exp [4] = '{0, 9, 18, 27};
      for (int c = 0; c < 4; c++) begin
        check("T4.2.1",
              wgt_addr[c] == exp[c][15:0],
              $sformatf("col%0d: addr=%0d expected=%0d", c, wgt_addr[c], exp[c]));
      end
      // All 4 must differ
      check("T4.2.1_diff",
            (wgt_addr[0] != wgt_addr[1]) &&
            (wgt_addr[1] != wgt_addr[2]) &&
            (wgt_addr[2] != wgt_addr[3]),
            "All 4 addresses must differ");
    end

    // Test with cin=1, kw=2: addr[col] = col*9 + 1*3 + 2 = col*9 + 5
    apply(10'd1, 10'd0, 4'd2, 4'd0);
    begin
      automatic int exp [4] = '{5, 14, 23, 32};
      for (int c = 0; c < 4; c++) begin
        check("T4.2.1b",
              wgt_addr[c] == exp[c][15:0],
              $sformatf("cin=1,kw=2 col%0d: addr=%0d expected=%0d",
                        c, wgt_addr[c], exp[c]));
      end
    end

    // ════════════════════════════════════════════════════════════
    //  T4.2.2: PE_OS1 — 4 different cout addresses (no kw)
    //   Formula: addr[col] = (cout_group*4 + col) * Cin + cin
    //   Cin=16, cout_group=0, cin=0
    //   addr[0]=0, addr[1]=16, addr[2]=32, addr[3]=48
    // ════════════════════════════════════════════════════════════
    $display("\n--- T4.2.2: PE_OS1 — 4 different cout addresses ---");
    cfg_pe_mode = PE_OS1;
    cfg_cin     = 10'd16;
    cfg_cout    = 10'd64;
    cfg_kw      = 4'd1;  // 1x1

    apply(10'd0, 10'd0, 4'd0, 4'd0);

    begin
      automatic int exp [4] = '{0, 16, 32, 48};
      for (int c = 0; c < 4; c++) begin
        check("T4.2.2",
              wgt_addr[c] == exp[c][15:0],
              $sformatf("col%0d: addr=%0d expected=%0d", c, wgt_addr[c], exp[c]));
      end
      check("T4.2.2_diff",
            (wgt_addr[0] != wgt_addr[1]) &&
            (wgt_addr[1] != wgt_addr[2]) &&
            (wgt_addr[2] != wgt_addr[3]),
            "All 4 addresses must differ");
    end

    // Test with cin=5, cout_group=1: addr[col] = (4+col)*16 + 5
    apply(10'd5, 10'd1, 4'd0, 4'd0);
    begin
      automatic int exp [4] = '{69, 85, 101, 117};
      for (int c = 0; c < 4; c++) begin
        check("T4.2.2b",
              wgt_addr[c] == exp[c][15:0],
              $sformatf("cout_grp=1,cin=5 col%0d: addr=%0d expected=%0d",
                        c, wgt_addr[c], exp[c]));
      end
    end

    // ════════════════════════════════════════════════════════════
    //  T4.2.3: PE_DW3 — 4 different CHANNEL addresses (not cout!)
    //   Formula: addr[col] = (ch_group*4 + col) * Kw + kw
    //   Kw=3, ch_group=0, kw=0
    //   addr[0]=0, addr[1]=3, addr[2]=6, addr[3]=9
    // ════════════════════════════════════════════════════════════
    $display("\n--- T4.2.3: PE_DW3 — 4 different CHANNEL addresses ---");
    cfg_pe_mode = PE_DW3;
    cfg_cin     = 10'd16;
    cfg_cout    = 10'd16;  // DW: cout == cin
    cfg_kw      = 4'd3;

    apply(10'd0, 10'd0, 4'd0, 4'd0);

    begin
      automatic int exp [4] = '{0, 3, 6, 9};
      for (int c = 0; c < 4; c++) begin
        check("T4.2.3",
              wgt_addr[c] == exp[c][15:0],
              $sformatf("col%0d: addr=%0d expected=%0d", c, wgt_addr[c], exp[c]));
      end
      check("T4.2.3_diff",
            (wgt_addr[0] != wgt_addr[1]) &&
            (wgt_addr[1] != wgt_addr[2]) &&
            (wgt_addr[2] != wgt_addr[3]),
            "All 4 CHANNEL addresses must differ");
    end

    // Test with kw=2, ch_group=2: addr[col] = (8+col)*3 + 2
    apply(10'd0, 10'd2, 4'd2, 4'd1);
    begin
      automatic int exp [4] = '{26, 29, 32, 35};
      for (int c = 0; c < 4; c++) begin
        check("T4.2.3b",
              wgt_addr[c] == exp[c][15:0],
              $sformatf("ch_grp=2,kw=2 col%0d: addr=%0d expected=%0d",
                        c, wgt_addr[c], exp[c]));
      end
    end

    // Bank check: kh_row=1 → bank = 1 mod 3 = 1
    check("T4.2.3_bank", wgt_bank_id == 2'd1,
          $sformatf("kh_row=1: bank=%0d expected=1", wgt_bank_id));

    // ════════════════════════════════════════════════════════════
    //  T4.2.4: Full L0 weight sweep: Cin=3, Cout=16, kw=3 (PE_RS3)
    //   Sweep all (cout_group, cin, kw) and verify formula.
    // ════════════════════════════════════════════════════════════
    $display("\n--- T4.2.4: Full L0 weight sweep ---");
    cfg_pe_mode = PE_RS3;
    cfg_cin     = 10'd3;
    cfg_cout    = 10'd16;
    cfg_kw      = 4'd3;

    begin
      automatic int cout_groups = 16 / 4;  // 4
      automatic int errors_in_sweep = 0;

      for (int cg = 0; cg < cout_groups; cg++) begin
        for (int ci = 0; ci < 3; ci++) begin
          for (int kw_i = 0; kw_i < 3; kw_i++) begin
            apply(ci[9:0], cg[9:0], kw_i[3:0], 4'd0);

            for (int col = 0; col < 4; col++) begin
              automatic int cout_val = cg * 4 + col;
              automatic int expected = cout_val * 3 * 3 + ci * 3 + kw_i;
              test_cnt++;
              if (wgt_addr[col] == expected[15:0]) begin
                pass_cnt++;
              end else begin
                fail_cnt++;
                errors_in_sweep++;
                if (errors_in_sweep <= 5) begin  // limit output
                  $display("[FAIL] T4.2.4 : cg=%0d,cin=%0d,kw=%0d,col=%0d: addr=%0d expected=%0d",
                           cg, ci, kw_i, col, wgt_addr[col], expected);
                end
              end
            end
          end
        end
      end

      if (errors_in_sweep == 0)
        $display("[INFO] T4.2.4: All %0d address checks passed",
                 cout_groups * 3 * 3 * 4);
      else
        $display("[INFO] T4.2.4: %0d errors in sweep", errors_in_sweep);
    end

    // ════════════════════════════════════════════════════════════
    //  Summary
    // ════════════════════════════════════════════════════════════
    $display("\n===========================================================");
    $display(" tb_addr_gen_weight — RESULTS");
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
