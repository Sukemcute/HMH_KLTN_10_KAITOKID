// ============================================================================
// Testbench : tb_metadata_ram
// Project   : YOLOv10n INT8 Accelerator — V4-VC707
// Description:
//   Verification of metadata_ram ring buffer and slot management.
//   5 tests:
//     T3.4.1  Set valid -> query -> verify valid and meta match
//     T3.4.2  Ring advance: set 4 slots -> advance 2 -> verify head/tail/count
//     T3.4.3  Full detection: fill all 16 slots -> verify ring_full
//     T3.4.4  Clear all: fill -> clear -> verify empty
//     T3.4.5  Wrap-around: fill 16, advance 16, fill 4 -> correct wrap
// ============================================================================
`timescale 1ns / 1ps

module tb_metadata_ram;

  import accel_pkg::*;

  // --------------------------------------------------------------------------
  //  Parameters
  // --------------------------------------------------------------------------
  localparam int NUM_SLOTS = 16;
  localparam int META_BITS = 32;
  localparam int SW        = $clog2(NUM_SLOTS);  // 4

  // Clock: 4 ns period (250 MHz)
  localparam real CLK_PERIOD = 4.0;

  // --------------------------------------------------------------------------
  //  DUT signals
  // --------------------------------------------------------------------------
  logic              clk;
  logic              rst_n;

  // Producer
  logic              set_valid;
  logic [SW-1:0]     set_slot_id;
  logic [META_BITS-1:0] set_meta;

  // Consumer query
  logic [SW-1:0]     query_slot_id;
  logic              query_valid;
  logic [META_BITS-1:0] query_meta;

  // Ring management
  logic              advance_ring;
  logic              clear_all;

  // Ring status
  logic [SW-1:0]     head_ptr;
  logic [SW-1:0]     tail_ptr;
  logic              ring_full;
  logic              ring_empty;

  // --------------------------------------------------------------------------
  //  DUT instantiation
  // --------------------------------------------------------------------------
  metadata_ram #(
    .NUM_SLOTS (NUM_SLOTS),
    .META_BITS (META_BITS)
  ) u_dut (
    .clk           (clk),
    .rst_n         (rst_n),
    .set_valid     (set_valid),
    .set_slot_id   (set_slot_id),
    .set_meta      (set_meta),
    .query_slot_id (query_slot_id),
    .query_valid   (query_valid),
    .query_meta    (query_meta),
    .advance_ring  (advance_ring),
    .clear_all     (clear_all),
    .head_ptr      (head_ptr),
    .tail_ptr      (tail_ptr),
    .ring_full     (ring_full),
    .ring_empty    (ring_empty)
  );

  // --------------------------------------------------------------------------
  //  Clock generation
  // --------------------------------------------------------------------------
  initial clk = 1'b0;
  always #(CLK_PERIOD / 2.0) clk = ~clk;

  // --------------------------------------------------------------------------
  //  Timeout watchdog (5 ms)
  // --------------------------------------------------------------------------
  initial begin
    #5_000_000;
    $display("[TIMEOUT] Simulation exceeded 5 ms — aborting.");
    $finish;
  end

  // --------------------------------------------------------------------------
  //  Test infrastructure
  // --------------------------------------------------------------------------
  int pass_count;
  int fail_count;
  int total_tests;

  task automatic report(input string name, input bit ok);
    total_tests++;
    if (ok) begin
      pass_count++;
      $display("[PASS] %s", name);
    end else begin
      fail_count++;
      $display("[FAIL] %s", name);
    end
  endtask

  // --------------------------------------------------------------------------
  //  Helper tasks
  // --------------------------------------------------------------------------
  task automatic do_reset();
    rst_n        <= 1'b0;
    set_valid    <= 1'b0;
    set_slot_id  <= '0;
    set_meta     <= '0;
    query_slot_id <= '0;
    advance_ring <= 1'b0;
    clear_all    <= 1'b0;
    repeat (4) @(posedge clk);
    rst_n <= 1'b1;
    @(posedge clk);
  endtask

  // Enqueue one entry: set_valid for 1 cycle with given slot_id and meta.
  // The module enqueues at tail_ptr and ignores set_slot_id for ring logic,
  // but set_slot_id is used for the validity/meta store addressing.
  task automatic enqueue_slot(input logic [SW-1:0] slot_id,
                              input logic [META_BITS-1:0] meta);
    @(posedge clk);
    set_valid   <= 1'b1;
    set_slot_id <= slot_id;
    set_meta    <= meta;
    @(posedge clk);
    set_valid   <= 1'b0;
  endtask

  task automatic advance_one();
    @(posedge clk);
    advance_ring <= 1'b1;
    @(posedge clk);
    advance_ring <= 1'b0;
  endtask

  task automatic do_clear();
    @(posedge clk);
    clear_all <= 1'b1;
    @(posedge clk);
    clear_all <= 1'b0;
  endtask

  task automatic query_slot(input logic [SW-1:0] slot_id,
                            output logic valid_out,
                            output logic [META_BITS-1:0] meta_out);
    // Combinational read — apply query address and sample on next edge.
    query_slot_id <= slot_id;
    @(posedge clk);
    valid_out = query_valid;
    meta_out  = query_meta;
  endtask

  // --------------------------------------------------------------------------
  //  Main test sequence
  // --------------------------------------------------------------------------
  initial begin
    pass_count  = 0;
    fail_count  = 0;
    total_tests = 0;

    do_reset();

    // ========================================================================
    // T3.4.1: Set valid -> query -> verify valid and meta match
    // ========================================================================
    begin
      logic v;
      logic [META_BITS-1:0] m;
      bit ok = 1;

      // Enqueue slot 0 with meta = 0xCAFE_BABE.
      enqueue_slot(4'd0, 32'hCAFE_BABE);

      // Query slot 0.
      query_slot(4'd0, v, m);
      if (!v) begin
        $display("  T3.4.1: slot 0 not valid after set_valid");
        ok = 0;
      end
      if (m !== 32'hCAFE_BABE) begin
        $display("  T3.4.1: meta mismatch: exp 0xCAFEBABE got 0x%08X", m);
        ok = 0;
      end

      // Verify ring status: 1 entry, not full, not empty.
      if (ring_empty) begin
        $display("  T3.4.1: ring_empty should be 0");
        ok = 0;
      end
      if (ring_full) begin
        $display("  T3.4.1: ring_full should be 0");
        ok = 0;
      end
      report("T3.4.1 Set valid -> query -> verify", ok);
    end

    // ========================================================================
    // T3.4.2: Ring advance: set 4 slots -> advance 2 -> verify head/tail/count
    // ========================================================================
    begin
      logic v;
      logic [META_BITS-1:0] m;
      bit ok = 1;

      do_reset();

      // Enqueue 4 slots with deterministic metadata.
      for (int i = 0; i < 4; i++)
        enqueue_slot(SW'(i), 32'(32'hA000_0000 + i));

      // After 4 enqueues: head=0, tail=4, count=4.
      @(posedge clk);
      if (head_ptr !== 4'd0) begin
        $display("  T3.4.2: head_ptr exp 0 got %0d", head_ptr);
        ok = 0;
      end
      if (tail_ptr !== 4'd4) begin
        $display("  T3.4.2: tail_ptr exp 4 got %0d", tail_ptr);
        ok = 0;
      end

      // Advance 2 times (consume head slots 0 and 1).
      advance_one();
      advance_one();

      // After 2 advances: head=2, tail=4, count=2.
      @(posedge clk);
      if (head_ptr !== 4'd2) begin
        $display("  T3.4.2: head_ptr exp 2 got %0d", head_ptr);
        ok = 0;
      end
      if (tail_ptr !== 4'd4) begin
        $display("  T3.4.2: tail_ptr exp 4 got %0d (after advance)", tail_ptr);
        ok = 0;
      end

      // Slots 0,1 should be invalidated; slots 2,3 still valid.
      query_slot(4'd0, v, m);
      if (v) begin
        $display("  T3.4.2: slot 0 should be invalid after advance");
        ok = 0;
      end
      query_slot(4'd1, v, m);
      if (v) begin
        $display("  T3.4.2: slot 1 should be invalid after advance");
        ok = 0;
      end
      query_slot(4'd2, v, m);
      if (!v) begin
        $display("  T3.4.2: slot 2 should still be valid");
        ok = 0;
      end
      if (m !== 32'hA000_0002) begin
        $display("  T3.4.2: slot 2 meta exp 0xA0000002 got 0x%08X", m);
        ok = 0;
      end
      query_slot(4'd3, v, m);
      if (!v) begin
        $display("  T3.4.2: slot 3 should still be valid");
        ok = 0;
      end

      if (!ring_empty && !ring_full) begin
        // Expected: neither full nor empty with 2 entries.
      end else begin
        $display("  T3.4.2: unexpected full/empty flags");
        ok = 0;
      end
      report("T3.4.2 Ring advance: head/tail/count", ok);
    end

    // ========================================================================
    // T3.4.3: Full detection: fill all 16 slots -> verify ring_full
    // ========================================================================
    begin
      bit ok = 1;

      do_reset();

      // Fill all 16 slots.
      for (int i = 0; i < NUM_SLOTS; i++)
        enqueue_slot(SW'(i), 32'(32'hB000_0000 + i));

      @(posedge clk);
      if (!ring_full) begin
        $display("  T3.4.3: ring_full should be 1 after 16 enqueues");
        ok = 0;
      end
      if (ring_empty) begin
        $display("  T3.4.3: ring_empty should be 0 after 16 enqueues");
        ok = 0;
      end

      // Verify all slots are valid with correct metadata.
      for (int i = 0; i < NUM_SLOTS; i++) begin
        logic v;
        logic [META_BITS-1:0] m;
        query_slot(SW'(i), v, m);
        if (!v) begin
          $display("  T3.4.3: slot %0d not valid", i);
          ok = 0;
        end
        if (m !== 32'(32'hB000_0000 + i)) begin
          $display("  T3.4.3: slot %0d meta mismatch: exp 0x%08X got 0x%08X",
                   i, 32'hB000_0000 + i, m);
          ok = 0;
        end
      end

      // Tail should have wrapped: tail = 0 (16 mod 16 = 0).
      if (tail_ptr !== 4'd0) begin
        $display("  T3.4.3: tail_ptr exp 0 (wrapped) got %0d", tail_ptr);
        ok = 0;
      end
      report("T3.4.3 Full detection (16 slots)", ok);
    end

    // ========================================================================
    // T3.4.4: Clear all: fill -> clear -> verify empty
    // ========================================================================
    begin
      bit ok = 1;

      do_reset();

      // Fill 8 slots.
      for (int i = 0; i < 8; i++)
        enqueue_slot(SW'(i), 32'(32'hC000_0000 + i));

      @(posedge clk);
      if (ring_empty) begin
        $display("  T3.4.4: ring should not be empty after 8 enqueues");
        ok = 0;
      end

      // Clear all.
      do_clear();

      @(posedge clk);
      if (!ring_empty) begin
        $display("  T3.4.4: ring_empty should be 1 after clear");
        ok = 0;
      end
      if (ring_full) begin
        $display("  T3.4.4: ring_full should be 0 after clear");
        ok = 0;
      end
      if (head_ptr !== 4'd0) begin
        $display("  T3.4.4: head_ptr should be 0 after clear, got %0d", head_ptr);
        ok = 0;
      end
      if (tail_ptr !== 4'd0) begin
        $display("  T3.4.4: tail_ptr should be 0 after clear, got %0d", tail_ptr);
        ok = 0;
      end

      // All slots should be invalid.
      for (int i = 0; i < NUM_SLOTS; i++) begin
        logic v;
        logic [META_BITS-1:0] m;
        query_slot(SW'(i), v, m);
        if (v) begin
          $display("  T3.4.4: slot %0d still valid after clear", i);
          ok = 0;
        end
      end
      report("T3.4.4 Clear all -> verify empty", ok);
    end

    // ========================================================================
    // T3.4.5: Wrap-around: fill 16, advance 16, fill 4 -> correct wrap
    // ========================================================================
    begin
      logic v;
      logic [META_BITS-1:0] m;
      bit ok = 1;

      do_reset();

      // Phase 1: fill all 16 slots.
      for (int i = 0; i < NUM_SLOTS; i++)
        enqueue_slot(SW'(i), 32'(32'hD000_0000 + i));

      // Phase 2: advance (consume) all 16 slots.
      for (int i = 0; i < NUM_SLOTS; i++)
        advance_one();

      // After 16 advances: head=0 (wrapped), tail=0 (wrapped), count=0.
      @(posedge clk);
      if (!ring_empty) begin
        $display("  T3.4.5: ring should be empty after 16 advances");
        ok = 0;
      end
      if (head_ptr !== 4'd0) begin
        $display("  T3.4.5: head_ptr exp 0 (wrapped) got %0d", head_ptr);
        ok = 0;
      end
      if (tail_ptr !== 4'd0) begin
        $display("  T3.4.5: tail_ptr exp 0 (wrapped) got %0d", tail_ptr);
        ok = 0;
      end

      // Phase 3: fill 4 new slots — these wrap into slots 0..3.
      for (int i = 0; i < 4; i++)
        enqueue_slot(SW'(i), 32'(32'hE000_0000 + i));

      // Verify: head=0, tail=4, count=4.
      @(posedge clk);
      if (head_ptr !== 4'd0) begin
        $display("  T3.4.5: head_ptr exp 0 got %0d (after refill)", head_ptr);
        ok = 0;
      end
      if (tail_ptr !== 4'd4) begin
        $display("  T3.4.5: tail_ptr exp 4 got %0d (after refill)", tail_ptr);
        ok = 0;
      end
      if (ring_empty) begin
        $display("  T3.4.5: ring should not be empty after 4 refill");
        ok = 0;
      end
      if (ring_full) begin
        $display("  T3.4.5: ring should not be full with only 4 entries");
        ok = 0;
      end

      // Verify new metadata in wrapped slots 0..3.
      for (int i = 0; i < 4; i++) begin
        query_slot(SW'(i), v, m);
        if (!v) begin
          $display("  T3.4.5: slot %0d should be valid after refill", i);
          ok = 0;
        end
        if (m !== 32'(32'hE000_0000 + i)) begin
          $display("  T3.4.5: slot %0d meta exp 0x%08X got 0x%08X",
                   i, 32'hE000_0000 + i, m);
          ok = 0;
        end
      end

      // Slots 4..15 should be invalid (they were advanced and not refilled).
      for (int i = 4; i < NUM_SLOTS; i++) begin
        query_slot(SW'(i), v, m);
        if (v) begin
          $display("  T3.4.5: slot %0d should be invalid", i);
          ok = 0;
        end
      end
      report("T3.4.5 Wrap-around fill/advance/refill", ok);
    end

    // ========================================================================
    //  Final summary
    // ========================================================================
    $display("");
    $display("==============================================================");
    $display("  tb_metadata_ram — %0d/%0d tests PASSED",
             pass_count, total_tests);
    if (fail_count == 0)
      $display("  >>> ALL TESTS PASSED <<<");
    else
      $display("  >>> %0d TEST(S) FAILED <<<", fail_count);
    $display("==============================================================");
    $finish;
  end

endmodule
