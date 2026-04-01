
`timescale 1ns/1ps

module tb_silu_lut;

  localparam int LANES = 32;

  logic               clk;
  logic               load_en;
  logic [7:0]         load_addr;
  logic signed [7:0]  load_data;
  logic [7:0]         idx [LANES];
  logic signed [7:0]  out [LANES];

  silu_lut #(.LANES(LANES)) uut (.*);

  always #2.5 clk = ~clk;

  int fail_count = 0;

  // Expected SiLU table (simple pattern for testing)
  logic signed [7:0] golden_rom [256];

  task automatic preload_rom();
    // Fill with a known pattern: golden_rom[i] = (i * 3 - 128) clamped to [-128,127]
    for (int i = 0; i < 256; i++) begin
      int val;
      val = (i * 3 + 17) % 256 - 128;
      golden_rom[i] = val[7:0];
    end

    // Load into LUT
    for (int i = 0; i < 256; i++) begin
      @(negedge clk);
      load_en   = 1;
      load_addr = i[7:0];
      load_data = golden_rom[i];
    end
    @(negedge clk);
    load_en = 0;
  endtask

  // ═══════════ TEST 1: Sequential readback ═══════════
  task automatic test_readback();
    int errors = 0;

    $display("=== TEST 1: Preload + sequential readback ===");

    for (int base = 0; base < 256; base += LANES) begin
      @(negedge clk);
      for (int l = 0; l < LANES; l++)
        idx[l] = (base + l) % 256;
      @(posedge clk);  // registered output

      for (int l = 0; l < LANES; l++) begin
        int addr;
        addr = (base + l) % 256;
        if (out[l] !== golden_rom[addr]) begin
          $display("  FAIL addr=%0d lane=%0d: got=%0d expected=%0d",
                   addr, l, out[l], golden_rom[addr]);
          errors++;
          if (errors > 10) return;
        end
      end
    end

    if (errors == 0) $display("  TEST 1 PASSED: All 256 entries verified");
    else $display("  TEST 1 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  // ═══════════ TEST 2: 32 random simultaneous reads ═══════════
  task automatic test_simultaneous();
    int errors = 0;

    $display("=== TEST 2: 32 simultaneous random reads (100 cycles) ===");

    for (int t = 0; t < 100; t++) begin
      @(negedge clk);
      for (int l = 0; l < LANES; l++)
        idx[l] = $random;
      @(posedge clk);

      for (int l = 0; l < LANES; l++) begin
        if (out[l] !== golden_rom[idx[l]]) begin
          $display("  FAIL t=%0d lane=%0d idx=%0d: got=%0d expected=%0d",
                   t, l, idx[l], out[l], golden_rom[idx[l]]);
          errors++;
          if (errors > 10) return;
        end
      end
    end

    if (errors == 0) $display("  TEST 2 PASSED");
    else $display("  TEST 2 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  // ═══════════ TEST 3: Boundary ═══════════
  task automatic test_boundary();
    int errors = 0;

    $display("=== TEST 3: Boundary indices (0 and 255) ===");

    @(negedge clk);
    for (int l = 0; l < LANES; l++)
      idx[l] = (l % 2 == 0) ? 8'd0 : 8'd255;
    @(posedge clk);

    for (int l = 0; l < LANES; l++) begin
      int addr;
      addr = (l % 2 == 0) ? 0 : 255;
      if (out[l] !== golden_rom[addr]) begin
        $display("  FAIL lane=%0d: got=%0d expected=%0d (idx=%0d)",
                 l, out[l], golden_rom[addr], addr);
        errors++;
      end
    end

    if (errors == 0) $display("  TEST 3 PASSED");
    else $display("  TEST 3 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  initial begin
    clk = 0;
    load_en = 0;
    for (int l = 0; l < LANES; l++) idx[l] = 0;

    $display("\n╔══════════════════════════════════════════════════╗");
    $display("║  TESTBENCH: silu_lut                            ║");
    $display("╚══════════════════════════════════════════════════╝\n");

    preload_rom();
    test_readback();
    test_simultaneous();
    test_boundary();

    $display("\n══════════════════════════════════════════════════");
    if (fail_count == 0) $display("  ★ ALL SILU_LUT TESTS PASSED ★");
    else $display("  ✗ TOTAL FAILURES: %0d", fail_count);
    $display("══════════════════════════════════════════════════\n");
    $finish;
  end

endmodule
