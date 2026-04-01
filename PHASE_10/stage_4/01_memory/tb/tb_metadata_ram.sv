`timescale 1ns/1ps

module tb_metadata_ram;

  import accel_pkg::*;

  // ---------- Parameters ----------
  localparam int NUM_SLOTS  = 16;
  localparam int META_BITS  = 32;
  localparam int SLOT_W     = $clog2(NUM_SLOTS);
  localparam int CLK_PERIOD = 10;

  // ---------- DUT signals ----------
  logic                    clk;
  logic                    rst_n;
  logic                    clear_all;
  logic                    set_valid;
  logic [SLOT_W-1:0]      set_slot_id;
  logic [META_BITS-1:0]   set_meta;
  logic [SLOT_W-1:0]      query_slot_id;
  logic                    query_valid;
  logic [META_BITS-1:0]   query_meta;
  logic                    advance_ring;
  logic [SLOT_W-1:0]      ring_head;
  logic [SLOT_W-1:0]      ring_tail;
  logic                    ring_full;
  logic                    ring_empty;

  // ---------- DUT ----------
  metadata_ram #(
    .NUM_SLOTS (NUM_SLOTS),
    .META_BITS (META_BITS)
  ) u_dut (
    .clk           (clk),
    .rst_n         (rst_n),
    .clear_all     (clear_all),
    .set_valid     (set_valid),
    .set_slot_id   (set_slot_id),
    .set_meta      (set_meta),
    .query_slot_id (query_slot_id),
    .query_valid   (query_valid),
    .query_meta    (query_meta),
    .advance_ring  (advance_ring),
    .ring_head     (ring_head),
    .ring_tail     (ring_tail),
    .ring_full     (ring_full),
    .ring_empty    (ring_empty)
  );

  // ---------- Clock ----------
  initial clk = 0;
  always #(CLK_PERIOD/2) clk = ~clk;

  // ---------- Scoreboard ----------
  int test_count;
  int pass_count;
  int fail_count;

  task automatic check_flag(string test_name, logic expected, logic actual);
    test_count++;
    if (actual === expected) begin
      pass_count++;
    end else begin
      fail_count++;
      $display("[FAIL] %s : expected %0b, got %0b", test_name, expected, actual);
    end
  endtask

  task automatic check_val(string test_name, logic [META_BITS-1:0] expected, logic [META_BITS-1:0] actual);
    test_count++;
    if (actual === expected) begin
      pass_count++;
    end else begin
      fail_count++;
      $display("[FAIL] %s : expected 0x%08h, got 0x%08h", test_name, expected, actual);
    end
  endtask

  task automatic check_ptr(string test_name, logic [SLOT_W-1:0] expected, logic [SLOT_W-1:0] actual);
    test_count++;
    if (actual === expected) begin
      pass_count++;
    end else begin
      fail_count++;
      $display("[FAIL] %s : expected %0d, got %0d", test_name, expected, actual);
    end
  endtask

  // ---------- Helper tasks ----------
  task automatic do_set_valid(input logic [SLOT_W-1:0] slot, input logic [META_BITS-1:0] meta);
    @(posedge clk);
    set_valid   <= 1'b1;
    set_slot_id <= slot;
    set_meta    <= meta;
    @(posedge clk);
    set_valid   <= 1'b0;
  endtask

  task automatic do_advance();
    @(posedge clk);
    advance_ring <= 1'b1;
    @(posedge clk);
    advance_ring <= 1'b0;
  endtask

  task automatic do_clear_all();
    @(posedge clk);
    clear_all <= 1'b1;
    @(posedge clk);
    clear_all <= 1'b0;
    @(posedge clk);  // Wait for clear to take effect
  endtask

  task automatic query(input logic [SLOT_W-1:0] slot);
    // Combinational query — just set the address and let it propagate
    query_slot_id = slot;
    #1;  // Small delay for combinational output to settle
  endtask

  // ---------- Stimulus ----------
  initial begin
    // Init
    clk           = 0;
    rst_n         = 0;
    clear_all     = 0;
    set_valid     = 0;
    set_slot_id   = '0;
    set_meta      = '0;
    query_slot_id = '0;
    advance_ring  = 0;
    test_count    = 0;
    pass_count    = 0;
    fail_count    = 0;

    // Reset
    repeat (4) @(posedge clk);
    rst_n = 1;
    repeat (2) @(posedge clk);

    // ========== T1: Set valid on slot 0 with metadata, query ==========
    $display("\n===== T1: Set valid on slot 0, query =====");
    begin
      int t1_pass;
      t1_pass = 1;

      do_set_valid(4'd0, 32'hDEADBEEF);

      // Query slot 0
      query(4'd0);
      check_flag("T1_slot0_valid", 1'b1, query_valid);
      check_val ("T1_slot0_meta",  32'hDEADBEEF, query_meta);
      if (query_valid !== 1'b1 || query_meta !== 32'hDEADBEEF) t1_pass = 0;

      if (t1_pass) $display("[PASS] T1: Slot 0 valid with correct metadata");
      else         $display("[FAIL] T1: Slot 0 validity/metadata mismatch");
    end

    // ========== T2: Ring advance ==========
    $display("\n===== T2: Ring advance — set 3 slots, advance 2 =====");
    begin
      int t2_pass;
      t2_pass = 1;

      // Clear for clean slate
      do_clear_all();

      // Set 3 slots (tail advances: 0, 1, 2)
      do_set_valid(4'd0, 32'h0000_AA00);
      do_set_valid(4'd1, 32'h0000_BB00);
      do_set_valid(4'd2, 32'h0000_CC00);

      // head should be 0, tail should be 3
      @(posedge clk);
      check_ptr("T2_head_before", 4'd0, ring_head);
      check_ptr("T2_tail_before", 4'd3, ring_tail);

      // Advance 2 times
      do_advance();
      do_advance();

      @(posedge clk);
      check_ptr("T2_head_after", 4'd2, ring_head);

      // Slot 0 and 1 should be invalid now
      query(4'd0);
      check_flag("T2_slot0_invalid", 1'b0, query_valid);

      query(4'd1);
      check_flag("T2_slot1_invalid", 1'b0, query_valid);

      // Slot 2 should still be valid
      query(4'd2);
      check_flag("T2_slot2_valid", 1'b1, query_valid);
      check_val ("T2_slot2_meta", 32'h0000_CC00, query_meta);

      if (fail_count == 0) $display("[PASS] T2: Ring advance — head moved, slots 0,1 invalidated");
      else                 $display("[INFO] T2: Ring advance — see above checks");
    end

    // ========== T3: Ring full ==========
    $display("\n===== T3: Ring full — fill all NUM_SLOTS =====");
    begin
      int t3_pass;
      t3_pass = 1;

      // Clear
      do_clear_all();
      check_flag("T3_initial_empty", 1'b1, ring_empty);

      // Fill all 16 slots
      for (int i = 0; i < NUM_SLOTS; i++) begin
        do_set_valid(i[SLOT_W-1:0], 32'hF000_0000 + i[31:0]);
      end

      @(posedge clk);
      check_flag("T3_ring_full", 1'b1, ring_full);

      // Verify all slots valid
      for (int i = 0; i < NUM_SLOTS; i++) begin
        query(i[SLOT_W-1:0]);
        check_flag($sformatf("T3_slot%0d_valid", i), 1'b1, query_valid);
        check_val ($sformatf("T3_slot%0d_meta", i), 32'hF000_0000 + i[31:0], query_meta);
      end

      if (fail_count == 0) $display("[PASS] T3: Ring full — all 16 slots filled, ring_full=1");
      else                 $display("[INFO] T3: Ring full — see above checks");
    end

    // ========== T4: Ring empty ==========
    $display("\n===== T4: Ring empty — advance until empty =====");
    begin
      int t4_pass;
      t4_pass = 1;

      // Ring is full from T3. Advance all 16 slots.
      for (int i = 0; i < NUM_SLOTS; i++) begin
        do_advance();
      end

      @(posedge clk);
      check_flag("T4_ring_empty", 1'b1, ring_empty);
      check_flag("T4_ring_not_full", 1'b0, ring_full);

      // All slots should be invalid
      for (int i = 0; i < NUM_SLOTS; i++) begin
        query(i[SLOT_W-1:0]);
        check_flag($sformatf("T4_slot%0d_invalid", i), 1'b0, query_valid);
      end

      if (fail_count == 0) $display("[PASS] T4: Ring empty — all slots invalidated, ring_empty=1");
      else                 $display("[INFO] T4: Ring empty — see above checks");
    end

    // ========== T5: Clear all ==========
    $display("\n===== T5: Clear all =====");
    begin
      int t5_pass;
      t5_pass = 1;

      // Fill some slots
      do_set_valid(4'd0, 32'h1111_1111);
      do_set_valid(4'd1, 32'h2222_2222);
      do_set_valid(4'd2, 32'h3333_3333);
      do_set_valid(4'd3, 32'h4444_4444);

      // Verify not empty
      @(posedge clk);
      check_flag("T5_not_empty_before_clear", 1'b0, ring_empty);

      // Clear all
      do_clear_all();

      // All invalid
      for (int i = 0; i < NUM_SLOTS; i++) begin
        query(i[SLOT_W-1:0]);
        check_flag($sformatf("T5_slot%0d_invalid", i), 1'b0, query_valid);
      end

      // Pointers reset
      check_ptr("T5_head_reset", 4'd0, ring_head);
      check_ptr("T5_tail_reset", 4'd0, ring_tail);
      check_flag("T5_empty_after_clear", 1'b1, ring_empty);
      check_flag("T5_not_full_after_clear", 1'b0, ring_full);

      if (fail_count == 0) $display("[PASS] T5: Clear all — all invalid, pointers reset");
      else                 $display("[INFO] T5: Clear all — see above checks");
    end

    // ========== T6: Wrap-around ==========
    $display("\n===== T6: Wrap-around — fill/drain repeatedly =====");
    begin
      int t6_pass;
      t6_pass = 1;

      // Clear first
      do_clear_all();

      // === Round 1: fill 10, drain 10 ===
      for (int i = 0; i < 10; i++)
        do_set_valid(i[SLOT_W-1:0], 32'hA000_0000 + i[31:0]);

      for (int i = 0; i < 10; i++)
        do_advance();

      @(posedge clk);
      check_flag("T6_empty_after_round1", 1'b1, ring_empty);
      // head and tail should both be 10 (mod 16 in pointer space)
      check_ptr("T6_head_round1", 4'd10, ring_head);
      check_ptr("T6_tail_round1", 4'd10, ring_tail);

      // === Round 2: fill 10 more (wraps around past slot 15 to slot 0) ===
      for (int i = 0; i < 10; i++) begin
        logic [SLOT_W-1:0] slot_id;
        slot_id = (10 + i) % NUM_SLOTS;
        do_set_valid(slot_id, 32'hB000_0000 + i[31:0]);
      end

      @(posedge clk);
      // tail_ptr should have advanced 10 more: 10+10 = 20 => ring_tail = 20 mod 16 = 4
      check_ptr("T6_tail_round2", 4'd4, ring_tail);
      check_flag("T6_not_empty_round2", 1'b0, ring_empty);

      // Verify wrapped slots are valid
      for (int i = 0; i < 10; i++) begin
        logic [SLOT_W-1:0] slot_id;
        slot_id = (10 + i) % NUM_SLOTS;
        query(slot_id);
        check_flag($sformatf("T6_round2_slot%0d_valid", slot_id), 1'b1, query_valid);
        check_val ($sformatf("T6_round2_slot%0d_meta",  slot_id), 32'hB000_0000 + i[31:0], query_meta);
      end

      // Drain all 10
      for (int i = 0; i < 10; i++)
        do_advance();

      @(posedge clk);
      check_flag("T6_empty_after_round2", 1'b1, ring_empty);
      // Both pointers wrap: head = 20 mod 16 = 4, tail = 20 mod 16 = 4
      check_ptr("T6_head_round2_final", 4'd4, ring_head);
      check_ptr("T6_tail_round2_final", 4'd4, ring_tail);

      // === Round 3: fill full 16, drain full 16 (another full wrap) ===
      for (int i = 0; i < NUM_SLOTS; i++) begin
        logic [SLOT_W-1:0] slot_id;
        slot_id = (4 + i) % NUM_SLOTS;
        do_set_valid(slot_id, 32'hC000_0000 + i[31:0]);
      end

      @(posedge clk);
      check_flag("T6_full_round3", 1'b1, ring_full);

      for (int i = 0; i < NUM_SLOTS; i++)
        do_advance();

      @(posedge clk);
      check_flag("T6_empty_after_round3", 1'b1, ring_empty);

      if (fail_count == 0) $display("[PASS] T6: Wrap-around — pointers wrap correctly across 3 rounds");
      else                 $display("[INFO] T6: Wrap-around — see above checks");
    end

    // ========== Summary ==========
    $display("\n========================================");
    $display("  tb_metadata_ram SUMMARY");
    $display("  Total checks : %0d", test_count);
    $display("  PASS         : %0d", pass_count);
    $display("  FAIL         : %0d", fail_count);
    if (fail_count == 0)
      $display("  *** ALL TESTS PASSED ***");
    else
      $display("  *** SOME TESTS FAILED ***");
    $display("========================================\n");
    $finish;
  end

endmodule
