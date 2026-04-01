`timescale 1ns/1ps

module tb_metadata_ram;

  localparam int NUM_SLOTS = 16;
  localparam int META_BITS = 32;
  localparam int SLOT_W    = $clog2(NUM_SLOTS);

  logic                  clk, rst_n, clear_all;
  logic                  set_valid;
  logic [SLOT_W-1:0]     set_slot_id;
  logic [META_BITS-1:0]  set_meta;
  logic [SLOT_W-1:0]     query_slot_id;
  logic                  query_valid;
  logic [META_BITS-1:0]  query_meta;
  logic                  advance_ring;
  logic [SLOT_W-1:0]     ring_head, ring_tail;
  logic                  ring_full, ring_empty;

  metadata_ram #(
    .NUM_SLOTS(NUM_SLOTS),
    .META_BITS(META_BITS)
  ) uut (.*);

  always #2.5 clk = ~clk;

  int fail_count = 0;

  task automatic reset();
    rst_n = 0;
    clear_all = 0;
    set_valid = 0;
    set_slot_id = '0;
    set_meta = '0;
    query_slot_id = '0;
    advance_ring = 0;
    repeat(3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
  endtask

  task automatic test_reset_state();
    int errors;
    errors = 0;

    $display("=== TEST 1: Reset state ===");
    reset();

    if (!ring_empty) begin
      $display("  FAIL: ring_empty should be 1 after reset");
      errors++;
    end
    if (ring_full) begin
      $display("  FAIL: ring_full should be 0 after reset");
      errors++;
    end
    if (ring_head !== 0 || ring_tail !== 0) begin
      $display("  FAIL: head=%0d tail=%0d expected 0,0", ring_head, ring_tail);
      errors++;
    end

    for (int s = 0; s < NUM_SLOTS; s++) begin
      query_slot_id = s[SLOT_W-1:0];
      #1;
      if (query_valid !== 1'b0 || query_meta !== '0) begin
        $display("  FAIL: slot %0d not cleared (valid=%0b meta=0x%08h)",
                 s, query_valid, query_meta);
        errors++;
      end
    end

    if (errors == 0) $display("  TEST 1 PASSED");
    else $display("  TEST 1 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  task automatic test_set_and_query();
    int errors;
    errors = 0;

    $display("=== TEST 2: Set valid + query metadata ===");
    reset();

    @(negedge clk);
    set_valid   = 1;
    set_slot_id = 4;
    set_meta    = 32'h1234_ABCD;
    @(negedge clk);
    set_valid   = 0;

    query_slot_id = 4;
    #1;
    if (!query_valid) begin
      $display("  FAIL: slot 4 should be valid");
      errors++;
    end
    if (query_meta !== 32'h1234_ABCD) begin
      $display("  FAIL: slot 4 meta=0x%08h expected 0x1234ABCD", query_meta);
      errors++;
    end
    if (ring_head !== 0 || ring_tail !== 1) begin
      $display("  FAIL: head=%0d tail=%0d expected 0,1", ring_head, ring_tail);
      errors++;
    end
    if (ring_empty) begin
      $display("  FAIL: ring_empty should be 0 after one insert");
      errors++;
    end

    if (errors == 0) $display("  TEST 2 PASSED");
    else $display("  TEST 2 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  task automatic test_advance_ring();
    int errors;
    errors = 0;

    $display("=== TEST 3: Advance ring clears head slot ===");
    reset();

    for (int s = 0; s < 3; s++) begin
      @(negedge clk);
      set_valid   = 1;
      set_slot_id = s[SLOT_W-1:0];
      set_meta    = 32'h1000 + s;
      @(negedge clk);
      set_valid   = 0;
    end

    query_slot_id = 0;
    #1;
    if (!query_valid) begin
      $display("  FAIL: slot 0 should be valid before advance");
      errors++;
    end

    @(negedge clk);
    advance_ring = 1;
    @(negedge clk);
    advance_ring = 0;

    query_slot_id = 0;
    #1;
    if (query_valid) begin
      $display("  FAIL: slot 0 should be cleared after advance");
      errors++;
    end
    query_slot_id = 1;
    #1;
    if (!query_valid || query_meta !== 32'h1001) begin
      $display("  FAIL: slot 1 should remain valid with meta 0x1001");
      errors++;
    end
    if (ring_head !== 1 || ring_tail !== 3) begin
      $display("  FAIL: head=%0d tail=%0d expected 1,3", ring_head, ring_tail);
      errors++;
    end

    if (errors == 0) $display("  TEST 3 PASSED");
    else $display("  TEST 3 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  task automatic test_full_and_clear();
    int errors;
    errors = 0;

    $display("=== TEST 4: Full flag and clear_all ===");
    reset();

    for (int s = 0; s < NUM_SLOTS; s++) begin
      @(negedge clk);
      set_valid   = 1;
      set_slot_id = s[SLOT_W-1:0];
      set_meta    = 32'hA500_0000 + s;
      @(negedge clk);
      set_valid   = 0;
    end

    if (!ring_full) begin
      $display("  FAIL: ring_full should assert after %0d inserts", NUM_SLOTS);
      errors++;
    end
    if (ring_empty) begin
      $display("  FAIL: ring_empty should be 0 when full");
      errors++;
    end

    @(negedge clk);
    clear_all = 1;
    @(negedge clk);
    clear_all = 0;

    if (!ring_empty || ring_full) begin
      $display("  FAIL: clear_all did not reset empty/full properly");
      errors++;
    end
    if (ring_head !== 0 || ring_tail !== 0) begin
      $display("  FAIL: head=%0d tail=%0d expected 0,0 after clear_all",
               ring_head, ring_tail);
      errors++;
    end

    query_slot_id = 7;
    #1;
    if (query_valid || query_meta !== '0) begin
      $display("  FAIL: slot 7 should be cleared after clear_all");
      errors++;
    end

    if (errors == 0) $display("  TEST 4 PASSED");
    else $display("  TEST 4 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  initial begin
    clk = 0;
    $display("\n╔══════════════════════════════════════════════════╗");
    $display("║  TESTBENCH: metadata_ram                        ║");
    $display("╚══════════════════════════════════════════════════╝\n");

    test_reset_state();
    test_set_and_query();
    test_advance_ring();
    test_full_and_clear();

    $display("\n══════════════════════════════════════════════════");
    if (fail_count == 0) $display("  ★ ALL METADATA_RAM TESTS PASSED ★");
    else $display("  ✗ TOTAL FAILURES: %0d", fail_count);
    $display("══════════════════════════════════════════════════\n");
    $finish;
  end

endmodule
