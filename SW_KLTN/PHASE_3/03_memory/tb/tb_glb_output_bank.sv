`timescale 1ns/1ps

module tb_glb_output_bank;
  import accel_pkg::*;

  localparam int LANES      = 32;
  localparam int BANK_DEPTH = 64;
  localparam int ADDR_W     = $clog2(BANK_DEPTH);

  logic                  clk, rst_n;
  logic                  wr_en;
  logic [ADDR_W-1:0]     wr_addr;
  namespace_e            wr_ns;
  logic [LANES*32-1:0]   wr_data_psum;
  logic [LANES*8-1:0]    wr_data_act;
  logic                  rd_en;
  logic [ADDR_W-1:0]     rd_addr;
  namespace_e            rd_ns;
  logic [LANES*32-1:0]   rd_data_psum;
  logic [LANES*8-1:0]    rd_data_act;

  glb_output_bank #(
    .LANES(LANES),
    .BANK_DEPTH(BANK_DEPTH)
  ) uut (.*);

  always #2.5 clk = ~clk;

  int fail_count = 0;

  task automatic reset();
    rst_n = 0;
    wr_en = 0;
    wr_addr = '0;
    wr_ns = NS_PSUM;
    wr_data_psum = '0;
    wr_data_act = '0;
    rd_en = 0;
    rd_addr = '0;
    rd_ns = NS_PSUM;
    repeat(3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
  endtask

  task automatic test_psum_mode();
    int errors = 0;
    logic [LANES*32-1:0] expected_psum;

    $display("=== TEST 1: PSUM write/read ===");
    reset();

    for (int a = 0; a < 4; a++) begin
      @(negedge clk);
      wr_en = 1;
      wr_addr = a[ADDR_W-1:0];
      wr_ns = NS_PSUM;
      for (int l = 0; l < LANES; l++)
        wr_data_psum[(l+1)*32-1 -: 32] = 32'(a * 1000 + l);
    end
    @(negedge clk);
    wr_en = 0;

    for (int a = 0; a < 4; a++) begin
      @(negedge clk);
      rd_en = 1;
      rd_addr = a[ADDR_W-1:0];
      rd_ns = NS_PSUM;
      @(posedge clk);
      @(negedge clk);
      rd_en = 0;

      for (int l = 0; l < LANES; l++)
        expected_psum[(l+1)*32-1 -: 32] = 32'(a * 1000 + l);

      if (rd_data_psum !== expected_psum) begin
        $display("  FAIL addr=%0d: PSUM mismatch", a);
        errors++;
      end
    end

    if (errors == 0) $display("  TEST 1 PASSED");
    else $display("  TEST 1 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  task automatic test_act_mode();
    int errors = 0;
    logic [LANES*8-1:0] expected_act;

    $display("=== TEST 2: ACT write/read ===");
    reset();

    for (int a = 0; a < 4; a++) begin
      @(negedge clk);
      wr_en = 1;
      wr_addr = a[ADDR_W-1:0];
      wr_ns = NS_ACT;
      for (int l = 0; l < LANES; l++)
        wr_data_act[(l+1)*8-1 -: 8] = (a * LANES + l) & 8'hFF;
    end
    @(negedge clk);
    wr_en = 0;

    for (int a = 0; a < 4; a++) begin
      @(negedge clk);
      rd_en = 1;
      rd_addr = a[ADDR_W-1:0];
      rd_ns = NS_ACT;
      @(posedge clk);
      @(negedge clk);
      rd_en = 0;

      for (int l = 0; l < LANES; l++)
        expected_act[(l+1)*8-1 -: 8] = (a * LANES + l) & 8'hFF;

      if (rd_data_act !== expected_act) begin
        $display("  FAIL addr=%0d: ACT mismatch", a);
        errors++;
      end
    end

    if (errors == 0) $display("  TEST 2 PASSED");
    else $display("  TEST 2 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  task automatic test_namespace_switch();
    int errors = 0;
    logic [LANES*32-1:0] expected_psum;
    logic [LANES*8-1:0]  expected_act;

    $display("=== TEST 3: Namespace switch preserves correct view ===");
    reset();

    // Write PSUM to addr 2
    @(negedge clk);
    wr_en = 1;
    wr_addr = 2;
    wr_ns = NS_PSUM;
    for (int l = 0; l < LANES; l++)
      wr_data_psum[(l+1)*32-1 -: 32] = 32'hCAFE_0000 + l;

    // Write ACT to addr 3
    @(negedge clk);
    wr_addr = 3;
    wr_ns = NS_ACT;
    for (int l = 0; l < LANES; l++)
      wr_data_act[(l+1)*8-1 -: 8] = 8'h80 + l;

    @(negedge clk);
    wr_en = 0;

    // Read addr 2 as PSUM
    @(negedge clk);
    rd_en = 1;
    rd_addr = 2;
    rd_ns = NS_PSUM;
    @(posedge clk);
    @(negedge clk);
    rd_en = 0;

    for (int l = 0; l < LANES; l++)
      expected_psum[(l+1)*32-1 -: 32] = 32'hCAFE_0000 + l;

    if (rd_data_psum !== expected_psum) begin
      $display("  FAIL: PSUM namespace readback mismatch");
      errors++;
    end

    // Read addr 3 as ACT
    @(negedge clk);
    rd_en = 1;
    rd_addr = 3;
    rd_ns = NS_ACT;
    @(posedge clk);
    @(negedge clk);
    rd_en = 0;

    for (int l = 0; l < LANES; l++)
      expected_act[(l+1)*8-1 -: 8] = 8'h80 + l;

    if (rd_data_act !== expected_act) begin
      $display("  FAIL: ACT namespace readback mismatch");
      errors++;
    end

    if (errors == 0) $display("  TEST 3 PASSED");
    else $display("  TEST 3 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  initial begin
    clk = 0;
    $display("\n╔══════════════════════════════════════════════════╗");
    $display("║  TESTBENCH: glb_output_bank                     ║");
    $display("╚══════════════════════════════════════════════════╝\n");

    test_psum_mode();
    test_act_mode();
    test_namespace_switch();

    $display("\n══════════════════════════════════════════════════");
    if (fail_count == 0) $display("  ★ ALL GLB_OUTPUT_BANK TESTS PASSED ★");
    else $display("  ✗ TOTAL FAILURES: %0d", fail_count);
    $display("══════════════════════════════════════════════════\n");
    $finish;
  end

endmodule
