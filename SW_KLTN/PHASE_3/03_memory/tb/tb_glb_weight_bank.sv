`timescale 1ns/1ps

module tb_glb_weight_bank;

  localparam int LANES      = 32;
  localparam int BANK_DEPTH = 64;
  localparam int FIFO_DEPTH = 8;
  localparam int ADDR_W     = $clog2(BANK_DEPTH);

  logic                 clk, rst_n;
  logic                 wr_en;
  logic [ADDR_W-1:0]    wr_addr;
  logic [LANES*8-1:0]   wr_data;
  logic                 rd_en;
  logic [ADDR_W-1:0]    rd_addr;
  logic [LANES*8-1:0]   rd_data;
  logic                 fifo_push;
  logic [LANES*8-1:0]   fifo_din;
  logic                 fifo_pop;
  logic [LANES*8-1:0]   fifo_dout;
  logic                 fifo_empty;
  logic                 fifo_full;

  glb_weight_bank #(
    .LANES(LANES),
    .BANK_DEPTH(BANK_DEPTH),
    .FIFO_DEPTH(FIFO_DEPTH)
  ) uut (.*);

  always #2.5 clk = ~clk;

  int fail_count = 0;

  task automatic reset();
    rst_n = 0;
    wr_en = 0;
    wr_addr = '0;
    wr_data = '0;
    rd_en = 0;
    rd_addr = '0;
    fifo_push = 0;
    fifo_din = '0;
    fifo_pop = 0;
    repeat(3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
  endtask

  task automatic test_sram_rw();
    int errors = 0;
    logic [LANES*8-1:0] expected;

    $display("=== TEST 1: SRAM write/read ===");
    reset();

    for (int a = 0; a < 4; a++) begin
      @(negedge clk);
      wr_en = 1;
      wr_addr = a[ADDR_W-1:0];
      for (int l = 0; l < LANES; l++)
        wr_data[(l+1)*8-1 -: 8] = (a * 17 + l) & 8'hFF;
    end
    @(negedge clk);
    wr_en = 0;

    for (int a = 0; a < 4; a++) begin
      @(negedge clk);
      rd_en = 1;
      rd_addr = a[ADDR_W-1:0];
      @(posedge clk);
      @(negedge clk);
      rd_en = 0;

      for (int l = 0; l < LANES; l++)
        expected[(l+1)*8-1 -: 8] = (a * 17 + l) & 8'hFF;

      if (rd_data !== expected) begin
        $display("  FAIL addr=%0d: SRAM mismatch", a);
        errors++;
      end
    end

    if (errors == 0) $display("  TEST 1 PASSED");
    else $display("  TEST 1 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  task automatic test_fifo_ordering();
    int errors = 0;
    logic [LANES*8-1:0] expected [FIFO_DEPTH];

    $display("=== TEST 2: FIFO FWFT ordering ===");
    reset();

    for (int i = 0; i < 4; i++) begin
      for (int l = 0; l < LANES; l++)
        expected[i][(l+1)*8-1 -: 8] = (i * 33 + l) & 8'hFF;

      @(negedge clk);
      fifo_push = 1;
      fifo_din = expected[i];
    end
    @(negedge clk);
    fifo_push = 0;

    if (fifo_empty) begin
      $display("  FAIL: FIFO should not be empty after pushes");
      errors++;
    end

    for (int i = 0; i < 4; i++) begin
      #1;
      if (fifo_dout !== expected[i]) begin
        $display("  FAIL pop[%0d]: fifo_dout mismatch before pop", i);
        errors++;
      end

      @(negedge clk);
      fifo_pop = 1;
      @(negedge clk);
      fifo_pop = 0;
    end

    if (!fifo_empty) begin
      $display("  FAIL: FIFO should be empty after popping all entries");
      errors++;
    end

    if (errors == 0) $display("  TEST 2 PASSED");
    else $display("  TEST 2 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  task automatic test_fifo_full_flag();
    int errors = 0;

    $display("=== TEST 3: FIFO full flag ===");
    reset();

    for (int i = 0; i < FIFO_DEPTH; i++) begin
      @(negedge clk);
      fifo_push = 1;
      for (int l = 0; l < LANES; l++)
        fifo_din[(l+1)*8-1 -: 8] = (i + l) & 8'hFF;
    end
    @(negedge clk);
    fifo_push = 0;

    if (!fifo_full) begin
      $display("  FAIL: FIFO full flag should assert after %0d pushes", FIFO_DEPTH);
      errors++;
    end
    if (fifo_empty) begin
      $display("  FAIL: FIFO empty flag should be 0 when full");
      errors++;
    end

    if (errors == 0) $display("  TEST 3 PASSED");
    else $display("  TEST 3 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  initial begin
    clk = 0;
    $display("\n╔══════════════════════════════════════════════════╗");
    $display("║  TESTBENCH: glb_weight_bank                     ║");
    $display("╚══════════════════════════════════════════════════╝\n");

    test_sram_rw();
    test_fifo_ordering();
    test_fifo_full_flag();

    $display("\n══════════════════════════════════════════════════");
    if (fail_count == 0) $display("  ★ ALL GLB_WEIGHT_BANK TESTS PASSED ★");
    else $display("  ✗ TOTAL FAILURES: %0d", fail_count);
    $display("══════════════════════════════════════════════════\n");
    $finish;
  end

endmodule
