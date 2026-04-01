`timescale 1ns/1ps
// ============================================================================
// Testbench: addr_gen_output
// Verifies (h_out, w_out, cout) -> (bank_id, addr) mapping.
// bank_id = req_pe_col (PE column index).
// ============================================================================
module tb_addr_gen_output;
  import accel_pkg::*;

  // ---------- Parameters ----------
  localparam int LANES  = 32;
  localparam int CLK_HP = 5;

  // ---------- Signals ----------
  logic              clk, rst_n;
  logic [3:0]        cfg_stride_h;
  logic [3:0]        cfg_q_out;
  logic [8:0]        cfg_cout_tile;
  logic [3:0]        cfg_pe_cols;

  logic              req_valid;
  logic [9:0]        req_h_out;
  logic [9:0]        req_w_out;
  logic [8:0]        req_cout;
  logic [1:0]        req_pe_col;

  logic              out_valid;
  logic [1:0]        out_bank_id;
  logic [15:0]       out_addr;

  // ---------- DUT ----------
  addr_gen_output #(.LANES(LANES)) dut (
    .clk          (clk),
    .rst_n        (rst_n),
    .cfg_stride_h (cfg_stride_h),
    .cfg_q_out    (cfg_q_out),
    .cfg_cout_tile(cfg_cout_tile),
    .cfg_pe_cols  (cfg_pe_cols),
    .req_valid    (req_valid),
    .req_h_out    (req_h_out),
    .req_w_out    (req_w_out),
    .req_cout     (req_cout),
    .req_pe_col   (req_pe_col),
    .out_valid    (out_valid),
    .out_bank_id  (out_bank_id),
    .out_addr     (out_addr)
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
    input logic [9:0] h_out, input logic [9:0] w_out,
    input logic [8:0] cout,  input logic [1:0] pe_col
  );
    @(posedge clk);
    req_valid  <= 1'b1;
    req_h_out  <= h_out;
    req_w_out  <= w_out;
    req_cout   <= cout;
    req_pe_col <= pe_col;
    @(posedge clk);
    req_valid  <= 1'b0;
    @(posedge clk);  // 1-cycle latency
  endtask

  // ---------- Expected address (mirror RTL) ----------
  function automatic logic [15:0] expected_addr(
    input logic [9:0] h_out, input logic [9:0] w_out,
    input logic [8:0] cout,  input logic [3:0] pe_cols,
    input logic [3:0] stride_h, input logic [3:0] q_out,
    input logic [8:0] cout_tile
  );
    logic [5:0] wblk_out, wblk_total;
    logic [9:0] h_group;
    logic [3:0] oslot;

    wblk_out   = w_out / LANES;
    wblk_total = (w_out + LANES - 1) / LANES;  // approximate (matches RTL)
    h_group    = h_out / (pe_cols * stride_h);
    oslot      = h_group % q_out;

    return (16'(oslot) * 16'(cout_tile) + 16'(cout)) * 16'(wblk_total) + 16'(wblk_out);
  endfunction

  // ---------- Stimulus ----------
  initial begin
    $display("============================================================");
    $display("  TB: addr_gen_output");
    $display("============================================================");

    rst_n      = 0;
    req_valid  = 0;
    req_h_out  = 0;
    req_w_out  = 0;
    req_cout   = 0;
    req_pe_col = 0;

    cfg_stride_h = 4'd1;
    cfg_q_out    = 4'd4;
    cfg_cout_tile = 9'd16;
    cfg_pe_cols  = 4'd4;

    repeat (4) @(posedge clk);
    rst_n = 1;
    repeat (2) @(posedge clk);

    // ====== T1: Bank selection = pe_col ======
    $display("\n--- T1: Bank selection = pe_col ---");
    begin
      for (int col = 0; col < 4; col++) begin
        drive_req(10'd0, 10'd64, 9'd0, 2'(col));
        check($sformatf("T1 pe_col=%0d -> bank=%0d", col, col),
              out_bank_id == 2'(col));
        $display("  T1: pe_col=%0d -> out_bank_id=%0d %s",
                 col, out_bank_id,
                 (out_bank_id == 2'(col)) ? "OK" : "MISMATCH");
      end
    end

    // ====== T2: Address calculation with known params ======
    $display("\n--- T2: Address calculation ---");
    begin
      logic [15:0] exp_addr;

      // h_out=8, w_out=64, cout=5, pe_cols=4, stride=1, q_out=4, cout_tile=16
      // h_group = 8 / (4*1) = 2
      // oslot   = 2 % 4 = 2
      // wblk_out = 64/32 = 2
      // wblk_total = (64+31)/32 = 2
      // addr = (2*16+5)*2 + 2 = 37*2+2 = 76
      drive_req(10'd8, 10'd64, 9'd5, 2'd0);
      exp_addr = expected_addr(10'd8, 10'd64, 9'd5, 4'd4, 4'd1, 4'd4, 9'd16);
      check("T2 addr calc", out_addr == exp_addr);
      $display("  T2: h=8 w=64 cout=5 -> addr=%0d (exp=%0d) %s",
               out_addr, exp_addr, (out_addr == exp_addr) ? "OK" : "MISMATCH");

      // h_out=0, w_out=32, cout=0 -> oslot=0, wblk_out=1, wblk_total=1(approx)
      drive_req(10'd0, 10'd32, 9'd0, 2'd1);
      exp_addr = expected_addr(10'd0, 10'd32, 9'd0, 4'd4, 4'd1, 4'd4, 9'd16);
      check("T2 addr zero h", out_addr == exp_addr);
      $display("  T2: h=0 w=32 cout=0 -> addr=%0d (exp=%0d) %s",
               out_addr, exp_addr, (out_addr == exp_addr) ? "OK" : "MISMATCH");

      // With stride=2: h_out=16, pe_cols=4 -> h_group = 16/(4*2) = 2
      cfg_stride_h = 4'd2;
      @(posedge clk);

      drive_req(10'd16, 10'd64, 9'd3, 2'd2);
      exp_addr = expected_addr(10'd16, 10'd64, 9'd3, 4'd4, 4'd2, 4'd4, 9'd16);
      check("T2 stride=2", out_addr == exp_addr);
      $display("  T2 stride=2: h=16 w=64 cout=3 -> addr=%0d (exp=%0d) %s",
               out_addr, exp_addr, (out_addr == exp_addr) ? "OK" : "MISMATCH");

      cfg_stride_h = 4'd1;  // restore
      @(posedge clk);
    end

    // ====== T3: Sweep full output range ======
    $display("\n--- T3: Sweep full output range ---");
    begin
      logic [15:0] exp_addr;
      int err_cnt;
      err_cnt = 0;

      // Sweep h_out=0..31, w_out=0,32,64,96, cout=0..3, pe_col=0
      cfg_stride_h  = 4'd1;
      cfg_q_out     = 4'd8;
      cfg_cout_tile = 9'd8;
      cfg_pe_cols   = 4'd4;
      @(posedge clk);

      for (int h = 0; h < 32; h += 4) begin
        for (int w = 0; w < 128; w += 32) begin
          for (int c = 0; c < 4; c++) begin
            drive_req(10'(h), 10'(w), 9'(c), 2'd0);
            exp_addr = expected_addr(10'(h), 10'(w), 9'(c),
                                     4'd4, 4'd1, 4'd8, 9'd8);
            if (out_addr != exp_addr || out_bank_id != 2'd0) begin
              err_cnt++;
              if (err_cnt <= 5)
                $display("  [FAIL] T3: h=%0d w=%0d c=%0d addr=%0d exp=%0d bank=%0d",
                         h, w, c, out_addr, exp_addr, out_bank_id);
            end
            check($sformatf("T3 h=%0d w=%0d c=%0d", h, w, c),
                  out_addr == exp_addr && out_bank_id == 2'd0);
          end
        end
      end

      if (err_cnt == 0)
        $display("  T3: Full sweep passed (%0d checks)", pass_cnt);
      else
        $display("  T3: %0d mismatches in sweep", err_cnt);
    end

    // ====== Summary ======
    repeat (4) @(posedge clk);
    $display("\n============================================================");
    $display("  addr_gen_output: %0d PASSED, %0d FAILED", pass_cnt, fail_cnt);
    if (fail_cnt == 0)
      $display("  >>> ALL TESTS PASSED <<<");
    else
      $display("  >>> SOME TESTS FAILED <<<");
    $display("============================================================");
    $finish;
  end

endmodule
