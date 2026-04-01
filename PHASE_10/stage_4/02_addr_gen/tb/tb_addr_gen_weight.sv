`timescale 1ns/1ps
// ============================================================================
// Testbench: addr_gen_weight
// Verifies mode-dependent weight address generation for PE_RS3, PE_OS1,
// PE_DW3, PE_DW7, and PE_GEMM modes.
// ============================================================================
module tb_addr_gen_weight;
  import accel_pkg::*;

  // ---------- Parameters ----------
  localparam int LANES  = 32;
  localparam int CLK_HP = 5;

  // ---------- Signals ----------
  logic              clk, rst_n;
  pe_mode_e          cfg_mode;
  logic [8:0]        cfg_cin_tile;
  logic [8:0]        cfg_cout_tile;
  logic [3:0]        cfg_kw;

  logic              req_valid;
  logic [2:0]        req_kr;
  logic [8:0]        req_cin;
  logic [8:0]        req_cout;
  logic [2:0]        req_kw_idx;

  logic              out_valid;
  logic [1:0]        out_bank_id;
  logic [15:0]       out_addr;

  // ---------- DUT ----------
  addr_gen_weight #(.LANES(LANES)) dut (
    .clk           (clk),
    .rst_n         (rst_n),
    .cfg_mode      (cfg_mode),
    .cfg_cin_tile  (cfg_cin_tile),
    .cfg_cout_tile (cfg_cout_tile),
    .cfg_kw        (cfg_kw),
    .req_valid     (req_valid),
    .req_kr        (req_kr),
    .req_cin       (req_cin),
    .req_cout      (req_cout),
    .req_kw_idx    (req_kw_idx),
    .out_valid     (out_valid),
    .out_bank_id   (out_bank_id),
    .out_addr      (out_addr)
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

  // ---------- Helper ----------
  task automatic drive_req(
    input logic [2:0] kr, input logic [8:0] cin,
    input logic [8:0] cout, input logic [2:0] kw_idx
  );
    @(posedge clk);
    req_valid  <= 1'b1;
    req_kr     <= kr;
    req_cin    <= cin;
    req_cout   <= cout;
    req_kw_idx <= kw_idx;
    @(posedge clk);
    req_valid  <= 1'b0;
    @(posedge clk);  // 1-cycle latency
  endtask

  // ---------- Stimulus ----------
  initial begin
    $display("============================================================");
    $display("  TB: addr_gen_weight");
    $display("============================================================");

    rst_n      = 0;
    req_valid  = 0;
    req_kr     = 0;
    req_cin    = 0;
    req_cout   = 0;
    req_kw_idx = 0;
    cfg_mode      = PE_RS3;
    cfg_cin_tile  = 9'd16;
    cfg_cout_tile = 9'd32;
    cfg_kw        = 4'd3;

    repeat (4) @(posedge clk);
    rst_n = 1;
    repeat (2) @(posedge clk);

    // ====== T1: PE_RS3 basic address computation ======
    $display("\n--- T1: PE_RS3 basic address computation ---");
    begin
      logic [15:0] exp_addr;
      logic [1:0]  exp_bank;

      // addr = cout * cin_tile * kw + cin * kw + kw_idx
      // kr=1, cout=2, cin=5, kw_idx=2
      cfg_mode = PE_RS3;
      cfg_kw   = 4'd3;
      @(posedge clk);

      drive_req(3'd1, 9'd5, 9'd2, 3'd2);
      exp_bank = 2'd1;  // kr[1:0]
      exp_addr = 16'(2) * 16'(16) * 16'(3) + 16'(5) * 16'(3) + 16'(2);
      // = 96 + 15 + 2 = 113

      check("T1 RS3 valid",    out_valid == 1'b1);
      check("T1 RS3 bank",     out_bank_id == exp_bank);
      check("T1 RS3 addr",     out_addr == exp_addr);
      $display("  T1 RS3: kr=1 cout=2 cin=5 kw_idx=2 -> bank=%0d addr=%0d (exp bank=%0d addr=%0d) %s",
               out_bank_id, out_addr, exp_bank, exp_addr,
               (out_addr == exp_addr && out_bank_id == exp_bank) ? "OK" : "MISMATCH");

      // Another RS3: kr=0, cout=0, cin=0, kw_idx=0 -> addr=0, bank=0
      drive_req(3'd0, 9'd0, 9'd0, 3'd0);
      check("T1 RS3 zero",     out_addr == 16'd0);
      check("T1 RS3 zero bank",out_bank_id == 2'd0);
      $display("  T1 RS3: all-zero -> addr=%0d bank=%0d OK", out_addr, out_bank_id);
    end

    // ====== T2: Multiple mode patterns ======
    $display("\n--- T2: PE_OS1 mode ---");
    begin
      logic [15:0] exp_addr;

      // PE_OS1: addr = cout * (cin_tile/3) + cin
      cfg_mode     = PE_OS1;
      cfg_cin_tile = 9'd18;  // divisible by 3
      @(posedge clk);

      drive_req(3'd2, 9'd3, 9'd4, 3'd0);
      exp_addr = 16'(4) * 16'(18/3) + 16'(3);
      // = 4*6 + 3 = 27

      check("T2 OS1 bank",  out_bank_id == 2'd2);
      check("T2 OS1 addr",  out_addr == exp_addr);
      $display("  T2 OS1: kr=2 cin=3 cout=4 -> bank=%0d addr=%0d (exp %0d) %s",
               out_bank_id, out_addr, exp_addr,
               (out_addr == exp_addr) ? "OK" : "MISMATCH");
    end

    $display("\n--- T2b: PE_DW3 mode ---");
    begin
      logic [15:0] exp_addr;

      // PE_DW3: addr = cin * kw + kw_idx
      cfg_mode = PE_DW3;
      cfg_kw   = 4'd3;
      @(posedge clk);

      drive_req(3'd0, 9'd7, 9'd0, 3'd1);
      exp_addr = 16'(7) * 16'(3) + 16'(1);
      // = 22

      check("T2 DW3 bank",  out_bank_id == 2'd0);
      check("T2 DW3 addr",  out_addr == exp_addr);
      $display("  T2 DW3: kr=0 cin=7 kw_idx=1 -> bank=%0d addr=%0d (exp %0d) %s",
               out_bank_id, out_addr, exp_addr,
               (out_addr == exp_addr) ? "OK" : "MISMATCH");
    end

    $display("\n--- T2c: PE_DW7 mode ---");
    begin
      logic [15:0] exp_addr;
      logic [1:0]  exp_bank;

      // PE_DW7: bank = kr[1:0] % 3, addr = cin * kw + kw_idx
      cfg_mode = PE_DW7;
      cfg_kw   = 4'd7;
      @(posedge clk);

      drive_req(3'd5, 9'd3, 9'd0, 3'd6);
      exp_bank = 5[1:0] % 2'd3;  // 5[1:0] = 1, 1%3 = 1
      exp_addr = 16'(3) * 16'(7) + 16'(6);
      // = 27

      check("T2 DW7 bank",  out_bank_id == exp_bank);
      check("T2 DW7 addr",  out_addr == exp_addr);
      $display("  T2 DW7: kr=5 cin=3 kw_idx=6 -> bank=%0d addr=%0d (exp bank=%0d addr=%0d) %s",
               out_bank_id, out_addr, exp_bank, exp_addr,
               (out_addr == exp_addr && out_bank_id == exp_bank) ? "OK" : "MISMATCH");
    end

    $display("\n--- T2d: PE_GEMM mode ---");
    begin
      logic [15:0] exp_addr;

      // PE_GEMM: addr = cout * cin_tile + cin
      cfg_mode     = PE_GEMM;
      cfg_cin_tile = 9'd64;
      @(posedge clk);

      drive_req(3'd1, 9'd10, 9'd3, 3'd0);
      exp_addr = 16'(3) * 16'(64) + 16'(10);
      // = 202

      check("T2 GEMM bank",  out_bank_id == 2'd1);
      check("T2 GEMM addr",  out_addr == exp_addr);
      $display("  T2 GEMM: kr=1 cin=10 cout=3 -> bank=%0d addr=%0d (exp %0d) %s",
               out_bank_id, out_addr, exp_addr,
               (out_addr == exp_addr) ? "OK" : "MISMATCH");
    end

    // ====== T3: Boundary values ======
    $display("\n--- T3: Boundary values ---");
    begin
      logic [15:0] exp_addr;

      // Max kr for RS3 (kr=2 -> bank=2)
      cfg_mode     = PE_RS3;
      cfg_cin_tile = 9'd256;
      cfg_cout_tile = 9'd256;
      cfg_kw       = 4'd3;
      @(posedge clk);

      drive_req(3'd2, 9'd255, 9'd255, 3'd2);
      exp_addr = 16'(255) * 16'(256) * 16'(3) + 16'(255) * 16'(3) + 16'(2);
      check("T3 RS3 max bank", out_bank_id == 2'd2);
      check("T3 RS3 max addr", out_addr == exp_addr[15:0]);
      $display("  T3 RS3 max: kr=2 cout=255 cin=255 kw=2 -> bank=%0d addr=0x%04h (exp 0x%04h)",
               out_bank_id, out_addr, exp_addr[15:0]);

      // kr=0, all zeros -> addr=0 bank=0
      drive_req(3'd0, 9'd0, 9'd0, 3'd0);
      check("T3 zero bank", out_bank_id == 2'd0);
      check("T3 zero addr", out_addr == 16'd0);

      // Single element DW3: cin=0 kw_idx=0
      cfg_mode = PE_DW3;
      cfg_kw   = 4'd1;
      @(posedge clk);

      drive_req(3'd0, 9'd0, 9'd0, 3'd0);
      check("T3 DW3 single", out_addr == 16'd0);
      $display("  T3 boundaries verified");
    end

    // ====== Summary ======
    repeat (4) @(posedge clk);
    $display("\n============================================================");
    $display("  addr_gen_weight: %0d PASSED, %0d FAILED", pass_cnt, fail_cnt);
    if (fail_cnt == 0)
      $display("  >>> ALL TESTS PASSED <<<");
    else
      $display("  >>> SOME TESTS FAILED <<<");
    $display("============================================================");
    $finish;
  end

endmodule
