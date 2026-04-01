`timescale 1ns/1ps
module tb_pe_cluster;
  import accel_pkg::*;
  localparam int LANES = 32, PE_ROWS = 3, PE_COLS = 4;

  logic clk, rst_n, en, clear_psum;
  pe_mode_e mode;
  logic signed [7:0] act_taps [PE_ROWS][LANES];
  logic signed [7:0] wgt_data [PE_ROWS][LANES];
  logic signed [31:0] psum_in [PE_COLS][LANES];
  logic psum_in_valid;
  logic signed [31:0] psum_out [PE_COLS][LANES];
  logic psum_out_valid;
  logic signed [7:0] pool_data_in [25][LANES];
  logic pool_en;
  logic signed [7:0] pool_out [LANES];
  logic pool_out_valid;

  pe_cluster #(.LANES(LANES), .PE_ROWS(PE_ROWS), .PE_COLS(PE_COLS)) dut (.*);

  initial clk = 0;
  always #5 clk = ~clk;
  int err_cnt = 0;

  task automatic reset();
    rst_n = 0; en = 0; clear_psum = 0; psum_in_valid = 0; pool_en = 0;
    for (int r = 0; r < PE_ROWS; r++)
      for (int l = 0; l < LANES; l++) begin
        act_taps[r][l] = 0;
        wgt_data[r][l] = 0;
      end
    for (int c = 0; c < PE_COLS; c++)
      for (int l = 0; l < LANES; l++)
        psum_in[c][l] = 0;
    for (int i = 0; i < 25; i++)
      for (int l = 0; l < LANES; l++)
        pool_data_in[i][l] = 0;
    @(negedge clk); @(negedge clk); rst_n = 1; @(negedge clk);
  endtask

  initial begin
    $display("╔══════════════════════════════════════════════════════╗");
    $display("║ TESTBENCH: pe_cluster                               ║");
    $display("╚══════════════════════════════════════════════════════╝");
    reset();

    // TEST 1: RS3 single MAC cycle — all activations=1, weights=1
    $display("=== TEST 1: RS3 simple MAC (all 1s) ===");
    mode = PE_RS3;
    @(negedge clk);
    clear_psum = 1; en = 1;
    for (int r = 0; r < PE_ROWS; r++)
      for (int l = 0; l < LANES; l++) begin
        act_taps[r][l] = 8'sd1;
        wgt_data[r][l] = 8'sd1;
      end
    @(negedge clk);
    clear_psum = 0; en = 0;

    // Wait for pipeline latency (4 PE cycles + 1 column_reduce)
    repeat(6) @(posedge clk);
    #1;

    // After 1 cycle with act=1, wgt=1: each PE computes 1*1=1 per lane
    // column_reduce sums 3 rows → psum_out[col][lane] = 3
    for (int col = 0; col < PE_COLS; col++) begin
      for (int l = 0; l < LANES; l++) begin
        if (psum_out[col][l] != 32'sd3) begin
          if (err_cnt < 10)
            $display("  FAIL col=%0d lane=%0d: got=%0d exp=3", col, l, psum_out[col][l]);
          err_cnt++;
        end
      end
    end
    if (err_cnt == 0) $display("  TEST 1 PASSED");

    // TEST 2: MAXPOOL
    $display("=== TEST 2: MAXPOOL comparator tree ===");
    pool_en = 1;
    for (int i = 0; i < 25; i++)
      for (int l = 0; l < LANES; l++)
        pool_data_in[i][l] = $signed(8'(i - 12));  // range -12..12
    @(negedge clk);
    repeat(7) @(posedge clk);
    #1;
    for (int l = 0; l < LANES; l++) begin
      if (pool_out[l] != 8'sd12) begin
        $display("  FAIL lane=%0d: max got=%0d exp=12", l, pool_out[l]);
        err_cnt++;
      end
    end
    if (err_cnt == 0) $display("  TEST 2 PASSED");
    pool_en = 0;

    $display("════════════════════════════════════════════════════");
    if (err_cnt == 0) $display("★ ALL TESTS PASSED ★");
    else              $display("✗ TOTAL FAILURES: %0d", err_cnt);
    $display("════════════════════════════════════════════════════");
    $finish;
  end
endmodule
