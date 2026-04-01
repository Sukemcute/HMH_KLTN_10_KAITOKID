`timescale 1ns/1ps
module tb_router_cluster;
  import accel_pkg::*;
  import desc_pkg::*;
  localparam int LANES = 32;

  logic clk, rst_n;
  router_profile_t cfg_profile;
  pe_mode_e cfg_mode;
  logic [LANES*8-1:0] bank_input_rd [3];
  logic signed [7:0] pe_act [3][LANES];
  logic [LANES*8-1:0] bank_weight_rd [3];
  logic signed [7:0] pe_wgt [3][LANES];
  logic signed [31:0] pe_psum [4][LANES];
  logic psum_valid;
  logic [LANES*32-1:0] bank_output_wr [4];
  logic bank_output_wr_en [4];
  logic bypass_en;
  logic [LANES*8-1:0] bypass_data_in, bypass_data_out;
  logic bypass_valid;

  router_cluster #(.LANES(LANES)) dut (.*);

  initial clk = 0;
  always #5 clk = ~clk;
  int err_cnt = 0;

  initial begin
    $display("╔══════════════════════════════════════════════════════╗");
    $display("║ TESTBENCH: router_cluster                           ║");
    $display("╚══════════════════════════════════════════════════════╝");
    rst_n = 0; psum_valid = 0; bypass_en = 0;
    @(negedge clk); @(negedge clk); rst_n = 1;

    // TEST 1: RIN routing — bank0→row0, bank1→row1, bank2→row2
    $display("=== TEST 1: RIN 1-to-1 routing ===");
    cfg_mode = PE_RS3;
    cfg_profile.rin_src = {3'd2, 3'd1, 3'd0};  // row[0]=bank0, row[1]=bank1, row[2]=bank2
    cfg_profile.rwt_src = {3'd2, 3'd1, 3'd0};
    for (int b = 0; b < 3; b++) begin
      for (int l = 0; l < LANES; l++)
        bank_input_rd[b][l*8 +: 8] = 8'(b * 10 + l);
    end
    @(negedge clk); #1;
    for (int row = 0; row < 3; row++) begin
      for (int l = 0; l < LANES; l++) begin
        if (pe_act[row][l] != $signed(8'(row * 10 + l))) begin
          $display("  FAIL row=%0d lane=%0d: got=%0d exp=%0d", row, l, pe_act[row][l], row*10+l);
          err_cnt++;
        end
      end
    end
    if (err_cnt == 0) $display("  TEST 1 PASSED");

    // TEST 2: Bypass path
    $display("=== TEST 2: Bypass path ===");
    bypass_en = 1;
    bypass_data_in = {LANES{8'hAB}};
    @(negedge clk); #1;
    if (bypass_data_out != bypass_data_in || !bypass_valid) begin
      $display("  FAIL: bypass mismatch");
      err_cnt++;
    end else $display("  TEST 2 PASSED");
    bypass_en = 0;

    // TEST 3: RPS psum packing
    $display("=== TEST 3: RPS psum packing ===");
    psum_valid = 1;
    for (int col = 0; col < 4; col++)
      for (int l = 0; l < LANES; l++)
        pe_psum[col][l] = 32'(col * 100 + l);
    @(negedge clk); #1;
    for (int col = 0; col < 4; col++) begin
      if (!bank_output_wr_en[col]) begin
        $display("  FAIL: col %0d wr_en not set", col);
        err_cnt++;
      end
      for (int l = 0; l < LANES; l++) begin
        logic signed [31:0] got;
        got = $signed(bank_output_wr[col][l*32 +: 32]);
        if (got != 32'(col*100+l)) begin
          $display("  FAIL col=%0d lane=%0d: got=%0d exp=%0d", col, l, got, col*100+l);
          err_cnt++;
          if (err_cnt > 10) break;
        end
      end
      if (err_cnt > 10) break;
    end
    if (err_cnt == 0) $display("  TEST 3 PASSED");
    psum_valid = 0;

    $display("════════════════════════════════════════════════════");
    if (err_cnt == 0) $display("★ ALL TESTS PASSED ★");
    else              $display("✗ TOTAL FAILURES: %0d", err_cnt);
    $display("════════════════════════════════════════════════════");
    $finish;
  end
endmodule
