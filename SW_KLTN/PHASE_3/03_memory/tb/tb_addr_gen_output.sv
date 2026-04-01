`timescale 1ns/1ps
module tb_addr_gen_output;
  localparam int LANES = 32;
  logic clk, rst_n;
  logic [3:0] cfg_stride_h, cfg_q_out, cfg_pe_cols;
  logic [8:0] cfg_cout_tile;
  logic req_valid;
  logic [9:0] req_h_out, req_w_out;
  logic [8:0] req_cout;
  logic [1:0] req_pe_col;
  logic out_valid;
  logic [1:0] out_bank_id;
  logic [15:0] out_addr;

  addr_gen_output #(.LANES(LANES)) dut (.*);

  initial clk = 0;
  always #5 clk = ~clk;

  int err_cnt = 0;

  task automatic reset();
    rst_n = 0; req_valid = 0;
    @(negedge clk); @(negedge clk);
    rst_n = 1;
    @(negedge clk);
  endtask

  initial begin
    $display("╔══════════════════════════════════════════════════════╗");
    $display("║ TESTBENCH: addr_gen_output                          ║");
    $display("╚══════════════════════════════════════════════════════╝");
    reset();

    cfg_stride_h  = 4'd1;
    cfg_q_out     = 4'd4;
    cfg_cout_tile = 9'd32;
    cfg_pe_cols   = 4'd4;

    // TEST 1: bank_id should equal pe_col
    $display("=== TEST 1: bank_id = pe_col ===");
    for (int col = 0; col < 4; col++) begin
      @(negedge clk);
      req_valid  = 1;
      req_h_out  = 10'd0;
      req_w_out  = 10'd0;
      req_cout   = 9'd0;
      req_pe_col = col[1:0];
      @(negedge clk);
      req_valid = 0;
      @(posedge clk); #1;
      if (out_valid && out_bank_id != col[1:0]) begin
        $display("  FAIL: pe_col=%0d got bank_id=%0d", col, out_bank_id);
        err_cnt++;
      end
    end
    if (err_cnt == 0) $display("  TEST 1 PASSED");

    // TEST 2: Address changes with cout
    $display("=== TEST 2: Address varies with cout ===");
    begin
      logic [15:0] prev_addr;
      for (int c = 0; c < 4; c++) begin
        @(negedge clk);
        req_valid  = 1;
        req_h_out  = 10'd0;
        req_w_out  = 10'd0;
        req_cout   = 9'(c);
        req_pe_col = 2'd0;
        @(negedge clk);
        req_valid = 0;
        @(posedge clk); #1;
        if (c > 0 && out_addr == prev_addr) begin
          $display("  FAIL: cout=%0d same addr as cout=%0d", c, c-1);
          err_cnt++;
        end
        prev_addr = out_addr;
      end
    end
    if (err_cnt == 0) $display("  TEST 2 PASSED");

    $display("════════════════════════════════════════════════════");
    if (err_cnt == 0) $display("★ ALL TESTS PASSED ★");
    else              $display("✗ TOTAL FAILURES: %0d", err_cnt);
    $display("════════════════════════════════════════════════════");
    $finish;
  end
endmodule
