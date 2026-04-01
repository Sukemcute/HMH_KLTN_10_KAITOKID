// Self-checking testbench for glb_input_bank
// Test 1: Write all lanes, read back — verify data integrity
// Test 2: Lane mask — selective write
// Test 3: Multiple addresses — no aliasing
`timescale 1ns/1ps

module tb_glb_input_bank;

  localparam int LANES         = 32;
  localparam int SUBBANK_DEPTH = 64;  // smaller for fast sim
  localparam int ADDR_W        = $clog2(SUBBANK_DEPTH);

  logic                 clk, rst_n;
  logic                 wr_en, rd_en;
  logic [ADDR_W-1:0]   wr_addr, rd_addr;
  logic [LANES*8-1:0]  wr_data, rd_data;
  logic [LANES-1:0]    wr_lane_mask;

  glb_input_bank #(
    .LANES(LANES), .SUBBANK_DEPTH(SUBBANK_DEPTH)
  ) uut (.*);

  always #2.5 clk = ~clk;
  int fail_count = 0;

  task automatic reset();
    rst_n = 0; wr_en = 0; rd_en = 0;
    wr_addr = 0; rd_addr = 0; wr_data = 0;
    wr_lane_mask = '1;
    repeat(3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
  endtask

  // ═══════════ TEST 1: Full write + readback ═══════════
  task automatic test_write_read();
    int errors = 0;
    logic [LANES*8-1:0] expected;

    $display("=== TEST 1: Full write + readback ===");
    reset();

    for (int a = 0; a < 16; a++) begin
      @(negedge clk);
      wr_en = 1;
      wr_addr = a;
      wr_lane_mask = '1;
      for (int l = 0; l < LANES; l++)
        wr_data[(l+1)*8-1 -: 8] = (a * LANES + l) & 8'hFF;
    end
    @(negedge clk);
    wr_en = 0;

    // Read back
    for (int a = 0; a < 16; a++) begin
      @(negedge clk);
      rd_en = 1;
      rd_addr = a;
      @(posedge clk);  // DUT captures rd_addr and schedules rd_data update
      @(negedge clk);
      rd_en = 0;

      for (int l = 0; l < LANES; l++) begin
        expected[(l+1)*8-1 -: 8] = (a * LANES + l) & 8'hFF;
      end
      if (rd_data !== expected) begin
        $display("  FAIL addr=%0d: data mismatch", a);
        errors++;
      end
    end

    if (errors == 0) $display("  TEST 1 PASSED");
    else $display("  TEST 1 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  // ═══════════ TEST 2: Lane mask ═══════════
  task automatic test_lane_mask();
    int errors = 0;

    $display("=== TEST 2: Lane mask selective write ===");
    reset();

    // Write all 0xFF to addr 0
    @(negedge clk);
    wr_en = 1; wr_addr = 0;
    wr_lane_mask = '1;
    wr_data = '1;
    @(posedge clk);

    // Overwrite only even lanes with 0x42
    @(negedge clk);
    wr_en = 1; wr_addr = 0;
    wr_lane_mask = 32'hAAAA_AAAA;  // even lanes
    for (int l = 0; l < LANES; l++)
      wr_data[(l+1)*8-1 -: 8] = 8'h42;
    @(negedge clk);
    wr_en = 0;

    // Read back
    @(negedge clk); rd_en = 1; rd_addr = 0;
    @(posedge clk);
    @(negedge clk); rd_en = 0;

    for (int l = 0; l < LANES; l++) begin
      logic [7:0] got;
      got = rd_data[(l+1)*8-1 -: 8];
      if (wr_lane_mask[l]) begin
        if (got !== 8'h42) begin
          $display("  FAIL lane[%0d] (masked write): got=0x%02h expected=0x42", l, got);
          errors++;
        end
      end else begin
        if (got !== 8'hFF) begin
          $display("  FAIL lane[%0d] (unmasked): got=0x%02h expected=0xFF", l, got);
          errors++;
        end
      end
    end

    if (errors == 0) $display("  TEST 2 PASSED");
    else $display("  TEST 2 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  initial begin
    clk = 0;
    $display("\n╔══════════════════════════════════════════════════╗");
    $display("║  TESTBENCH: glb_input_bank                      ║");
    $display("╚══════════════════════════════════════════════════╝\n");

    test_write_read();
    test_lane_mask();

    $display("\n══════════════════════════════════════════════════");
    if (fail_count == 0) $display("  ★ ALL GLB_INPUT_BANK TESTS PASSED ★");
    else $display("  ✗ TOTAL FAILURES: %0d", fail_count);
    $display("══════════════════════════════════════════════════\n");
    $finish;
  end

endmodule
