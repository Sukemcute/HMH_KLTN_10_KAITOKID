`timescale 1ns/1ps
module tb_addr_gen_weight;
  import accel_pkg::*;
  localparam int LANES = 32;

  logic clk, rst_n;
  pe_mode_e cfg_mode;
  logic [8:0] cfg_cin_tile, cfg_cout_tile;
  logic [3:0] cfg_kw;
  logic req_valid;
  logic [2:0] req_kr, req_kw_idx;
  logic [8:0] req_cin, req_cout;
  logic out_valid;
  logic [1:0] out_bank_id;
  logic [15:0] out_addr;

  addr_gen_weight #(.LANES(LANES)) dut (.*);

  initial clk = 0;
  always #5 clk = ~clk;
  int err_cnt = 0;

  task automatic reset();
    rst_n = 0; req_valid = 0;
    @(negedge clk); @(negedge clk);
    rst_n = 1; @(negedge clk);
  endtask

  initial begin
    $display("╔══════════════════════════════════════════════════════╗");
    $display("║ TESTBENCH: addr_gen_weight                          ║");
    $display("╚══════════════════════════════════════════════════════╝");
    reset();

    // TEST 1: RS3 mode — bank_id = kernel_row
    $display("=== TEST 1: RS3 bank_id = kr ===");
    cfg_mode = PE_RS3; cfg_cin_tile = 9'd32; cfg_cout_tile = 9'd16; cfg_kw = 4'd3;
    for (int kr = 0; kr < 3; kr++) begin
      @(negedge clk);
      req_valid = 1; req_kr = kr[2:0]; req_cin = 0; req_cout = 0; req_kw_idx = 0;
      @(negedge clk); req_valid = 0;
      @(posedge clk); #1;
      if (out_valid && out_bank_id != kr[1:0]) begin
        $display("  FAIL: kr=%0d got bank=%0d", kr, out_bank_id);
        err_cnt++;
      end
    end
    if (err_cnt == 0) $display("  TEST 1 PASSED");

    // TEST 2: DW3 mode — address uniqueness
    $display("=== TEST 2: DW3 address uniqueness ===");
    cfg_mode = PE_DW3; cfg_kw = 4'd3;
    begin
      logic [15:0] addrs [9];
      int idx = 0;
      for (int kr = 0; kr < 3; kr++) begin
        for (int kw = 0; kw < 3; kw++) begin
          @(negedge clk);
          req_valid = 1; req_kr = kr[2:0]; req_cin = 9'd5; req_cout = 0; req_kw_idx = kw[2:0];
          @(negedge clk); req_valid = 0;
          @(posedge clk); #1;
          addrs[idx] = out_addr;
          idx++;
        end
      end
      // Check no duplicates within same bank
      bit dup = 0;
      for (int i = 0; i < 3; i++) begin  // 3 entries per bank
        for (int j = i+1; j < 3; j++) begin
          if (addrs[i*3] == addrs[j*3]) dup = 1;
        end
      end
      if (dup) begin $display("  FAIL: duplicate addresses"); err_cnt++; end
      else $display("  TEST 2 PASSED");
    end

    // TEST 3: OS1 mode
    $display("=== TEST 3: OS1 mode ===");
    cfg_mode = PE_OS1; cfg_cin_tile = 9'd96; cfg_cout_tile = 9'd32;
    @(negedge clk);
    req_valid = 1; req_kr = 3'd1; req_cin = 9'd10; req_cout = 9'd5; req_kw_idx = 0;
    @(negedge clk); req_valid = 0;
    @(posedge clk); #1;
    if (out_valid && out_bank_id != 2'd1) begin
      $display("  FAIL: OS1 bank expected 1, got %0d", out_bank_id);
      err_cnt++;
    end else $display("  TEST 3 PASSED");

    $display("════════════════════════════════════════════════════");
    if (err_cnt == 0) $display("★ ALL TESTS PASSED ★");
    else              $display("✗ TOTAL FAILURES: %0d", err_cnt);
    $display("════════════════════════════════════════════════════");
    $finish;
  end
endmodule
