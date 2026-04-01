// ============================================================================
// Testbench : tb_tile_fsm
// Project   : YOLOv10n INT8 Accelerator — V4-VC707
// Tests     : T6.1.1  Normal Conv flow (PE_RS3)
//             T6.1.2  DW7x7 multipass (num_k_pass=3)
//             T6.1.3  PE_PASS mode (bypass path)
//             T6.1.4  Barrier wait
// ============================================================================
`timescale 1ns / 1ps

module tb_tile_fsm;
  import accel_pkg::*;
  import desc_pkg::*;

  // ────────────────────────────────────────────────────────────
  // Clock & reset
  // ────────────────────────────────────────────────────────────
  logic clk, rst_n;
  initial clk = 0;
  always #2 clk = ~clk;  // 4 ns period => 250 MHz

  // ────────────────────────────────────────────────────────────
  // DUT signals
  // ────────────────────────────────────────────────────────────
  logic          tile_valid;
  layer_desc_t   layer_desc;
  tile_desc_t    tile_desc;
  logic          tile_accept;

  logic          shadow_latch;

  logic          dma_start;
  logic          dma_is_write;
  logic          dma_done;

  logic          seq_start;
  logic          seq_done;

  logic          ppu_start;
  logic          ppu_done;

  logic          swizzle_start;
  logic          swizzle_done;

  logic          barrier_wait_req;
  logic          barrier_grant;
  logic          barrier_signal;

  logic          page_swap;
  tile_state_e   state;
  logic          tile_done;
  logic [3:0]    cur_k_pass_idx;

  // ────────────────────────────────────────────────────────────
  // DUT instantiation
  // ────────────────────────────────────────────────────────────
  tile_fsm u_dut (
    .clk             (clk),
    .rst_n           (rst_n),
    .tile_valid      (tile_valid),
    .layer_desc      (layer_desc),
    .tile_desc       (tile_desc),
    .tile_accept     (tile_accept),
    .shadow_latch    (shadow_latch),
    .dma_start       (dma_start),
    .dma_is_write    (dma_is_write),
    .dma_done        (dma_done),
    .seq_start       (seq_start),
    .seq_done        (seq_done),
    .ppu_start       (ppu_start),
    .ppu_done        (ppu_done),
    .swizzle_start   (swizzle_start),
    .swizzle_done    (swizzle_done),
    .barrier_wait_req(barrier_wait_req),
    .barrier_grant   (barrier_grant),
    .barrier_signal  (barrier_signal),
    .page_swap       (page_swap),
    .state           (state),
    .tile_done       (tile_done),
    .cur_k_pass_idx  (cur_k_pass_idx)
  );

  // ────────────────────────────────────────────────────────────
  // Instant DMA stub: dma_done = dma_start (combinational loopback)
  // ────────────────────────────────────────────────────────────
  assign dma_done = dma_start;

  // ────────────────────────────────────────────────────────────
  // Test infrastructure
  // ────────────────────────────────────────────────────────────
  int pass_cnt = 0;
  int fail_cnt = 0;
  int test_num = 0;

  task automatic check(input string tag, input logic cond);
    if (cond) begin
      $display("[PASS] %s", tag);
      pass_cnt++;
    end else begin
      $display("[FAIL] %s", tag);
      fail_cnt++;
    end
  endtask

  // Wait for specific state with timeout
  task automatic wait_state(input tile_state_e target, input int max_cyc = 200);
    int cnt = 0;
    while (state !== target && cnt < max_cyc) begin
      @(posedge clk);
      cnt++;
    end
    if (cnt >= max_cyc)
      $display("  WARNING: timeout waiting for state %s (stuck at %s)", target.name(), state.name());
  endtask

  // Helper: build default layer_desc
  function automatic layer_desc_t make_layer_desc(
    pe_mode_e mode, logic [9:0] cin, logic [9:0] cout,
    logic [3:0] kw, logic [3:0] num_k_pass_val
  );
    layer_desc_t ld = '0;
    ld.pe_mode     = mode;
    ld.activation  = ACT_RELU;
    ld.cin         = cin;
    ld.cout        = cout;
    ld.hout        = 10'd2;
    ld.wout        = 10'd20;
    ld.hin         = 10'd4;
    ld.win         = 10'd22;
    ld.kh          = 4'd3;
    ld.kw          = kw;
    ld.stride      = 3'd1;
    ld.padding     = 3'd1;
    ld.num_tiles   = 8'd1;
    ld.num_cin_pass = 4'd1;
    ld.num_k_pass  = num_k_pass_val;
    ld.swizzle     = SWZ_NORMAL;
    return ld;
  endfunction

  // Helper: build default tile_desc (no barrier, no swizzle)
  function automatic tile_desc_t make_tile_desc(
    logic barrier_w, logic need_swz, logic last_t
  );
    tile_desc_t td = '0;
    td.tile_id      = 16'd1;
    td.layer_id     = 5'd0;
    td.sc_mask      = 4'hF;
    td.valid_h      = 6'd2;
    td.valid_w      = 6'd20;
    td.num_cin_pass = 4'd1;
    td.num_k_pass   = 4'd1;
    td.first_tile   = 1'b1;
    td.last_tile    = last_t;
    td.barrier_wait = barrier_w;
    td.barrier_id   = 4'd0;
    td.need_swizzle = need_swz;
    return td;
  endfunction

  // Reset + quiesce
  task automatic do_reset();
    rst_n        <= 1'b0;
    tile_valid   <= 1'b0;
    seq_done     <= 1'b0;
    ppu_done     <= 1'b0;
    swizzle_done <= 1'b0;
    barrier_grant <= 1'b0;
    repeat (4) @(posedge clk);
    rst_n <= 1'b1;
    @(posedge clk);
  endtask

  // ────────────────────────────────────────────────────────────
  // State-history logger
  // ────────────────────────────────────────────────────────────
  tile_state_e state_log [0:31];
  int          state_idx;

  task automatic reset_log();
    state_idx = 0;
    for (int i = 0; i < 32; i++) state_log[i] = TS_IDLE;
  endtask

  always_ff @(posedge clk) begin
    if (rst_n && state_idx < 32) begin
      if (state_idx == 0 || state_log[state_idx-1] !== state) begin
        state_log[state_idx] = state;
        state_idx++;
      end
    end
  end

  // ================================================================
  // T6.1.1 — Normal Conv flow (PE_RS3)
  //   Expected: IDLE -> LOAD_DESC -> PREFILL_WT -> PREFILL_IN ->
  //             COMPUTE -> PE_DRAIN -> PPU_RUN -> WRITEBACK -> DONE -> IDLE
  // ================================================================
  task automatic test_T6_1_1();
    test_num++;
    $display("\n===== T6.1.1: Normal Conv flow (PE_RS3) =====");
    do_reset();
    reset_log();

    layer_desc   = make_layer_desc(PE_RS3, 10'd3, 10'd16, 4'd3, 4'd1);
    tile_desc    = make_tile_desc(.barrier_w(1'b0), .need_swz(1'b0), .last_t(1'b1));

    // Present tile
    @(posedge clk);
    tile_valid <= 1'b1;
    @(posedge clk);
    // tile_accept should fire; after 1 cycle FSM is in LOAD_DESC
    check("T6.1.1-a tile_accept", tile_accept === 1'b1);
    tile_valid <= 1'b0;

    // LOAD_DESC: shadow_latch + page_swap should be asserted
    wait_state(TS_LOAD_DESC);
    check("T6.1.1-b shadow_latch", shadow_latch === 1'b1);
    check("T6.1.1-c page_swap",    page_swap === 1'b1);

    // DMA stub is instant -> PREFILL_WT passes immediately
    wait_state(TS_PREFILL_WT);
    check("T6.1.1-d dma_start (wt)", dma_start === 1'b1);
    check("T6.1.1-e dma_is_write=0", dma_is_write === 1'b0);

    // PREFILL_IN
    wait_state(TS_PREFILL_IN);
    check("T6.1.1-f dma_start (in)", dma_start === 1'b1);

    // COMPUTE: pulse seq_done after 1 cycle to simulate sequencer completing
    wait_state(TS_COMPUTE);
    check("T6.1.1-g seq_start", seq_start === 1'b1);
    @(posedge clk);
    seq_done <= 1'b1;
    @(posedge clk);
    seq_done <= 1'b0;

    // PE_DRAIN: wait DSP_PIPE_DEPTH+1 cycles for drain_cnt to reach threshold
    wait_state(TS_PE_DRAIN);

    // PPU_RUN: pulse ppu_done
    wait_state(TS_PPU_RUN);
    check("T6.1.1-h ppu_start", ppu_start === 1'b1);
    @(posedge clk);
    ppu_done <= 1'b1;
    @(posedge clk);
    ppu_done <= 1'b0;

    // WRITEBACK (last_tile=1 => need_writeback_reg=1)
    wait_state(TS_WRITEBACK);
    check("T6.1.1-i dma_is_write=1", dma_is_write === 1'b1);

    // DONE
    wait_state(TS_DONE);
    check("T6.1.1-j tile_done", tile_done === 1'b1);
    check("T6.1.1-k barrier_signal", barrier_signal === 1'b1);

    // Back to IDLE
    wait_state(TS_IDLE);
    check("T6.1.1-l back_to_IDLE", state === TS_IDLE);

    // Verify state sequence
    $display("  State trace:");
    for (int i = 0; i < state_idx && i < 32; i++)
      $display("    [%0d] %s", i, state_log[i].name());
  endtask

  // ================================================================
  // T6.1.2 — DW7x7 multipass (num_k_pass=3)
  //   COMPUTE -> PE_DRAIN loops 3 times before PPU_RUN
  // ================================================================
  task automatic test_T6_1_2();
    test_num++;
    $display("\n===== T6.1.2: DW7x7 multipass (num_k_pass=3) =====");
    do_reset();
    reset_log();

    layer_desc = make_layer_desc(PE_DW7, 10'd8, 10'd8, 4'd3, 4'd3);
    tile_desc  = make_tile_desc(.barrier_w(1'b0), .need_swz(1'b0), .last_t(1'b0));

    @(posedge clk);
    tile_valid <= 1'b1;
    @(posedge clk);
    tile_valid <= 1'b0;

    // Skip through LOAD_DESC, PREFILL_WT, PREFILL_IN (instant DMA)
    wait_state(TS_COMPUTE);

    int compute_count = 0;
    int drain_count   = 0;

    // Loop: expect 3 rounds of COMPUTE -> PE_DRAIN
    for (int pass = 0; pass < 3; pass++) begin
      wait_state(TS_COMPUTE, 100);
      compute_count++;
      $display("  Pass %0d: entered COMPUTE", pass);
      @(posedge clk);
      seq_done <= 1'b1;
      @(posedge clk);
      seq_done <= 1'b0;

      wait_state(TS_PE_DRAIN, 100);
      drain_count++;
      $display("  Pass %0d: entered PE_DRAIN", pass);
      // Wait for drain to complete (DSP_PIPE_DEPTH+1 cycles)
      repeat (DSP_PIPE_DEPTH + 2) @(posedge clk);
    end

    // After 3 drains, should advance to PPU_RUN
    wait_state(TS_PPU_RUN, 50);
    check("T6.1.2-a compute_loops=3", compute_count == 3);
    check("T6.1.2-b drain_loops=3",   drain_count == 3);
    check("T6.1.2-c reached PPU_RUN",  state === TS_PPU_RUN);

    // Finish off
    ppu_done <= 1'b1;
    @(posedge clk);
    ppu_done <= 1'b0;

    // last_tile=0 => need_writeback_reg=0 => skip WRITEBACK
    wait_state(TS_DONE);
    check("T6.1.2-d tile_done", tile_done === 1'b1);

    wait_state(TS_IDLE);
    check("T6.1.2-e back_to_IDLE", state === TS_IDLE);

    $display("  State trace:");
    for (int i = 0; i < state_idx && i < 32; i++)
      $display("    [%0d] %s", i, state_log[i].name());
  endtask

  // ================================================================
  // T6.1.3 — PE_PASS mode
  //   Expected: IDLE -> LOAD_DESC -> PREFILL_IN -> SWIZZLE -> DONE
  //   (skip WT, COMPUTE, PPU)
  // ================================================================
  task automatic test_T6_1_3();
    test_num++;
    $display("\n===== T6.1.3: PE_PASS mode (bypass path) =====");
    do_reset();
    reset_log();

    layer_desc = make_layer_desc(PE_PASS, 10'd8, 10'd8, 4'd1, 4'd1);
    tile_desc  = make_tile_desc(.barrier_w(1'b0), .need_swz(1'b1), .last_t(1'b0));

    @(posedge clk);
    tile_valid <= 1'b1;
    @(posedge clk);
    tile_valid <= 1'b0;

    // LOAD_DESC
    wait_state(TS_LOAD_DESC);

    // Should skip PREFILL_WT entirely, go to PREFILL_IN
    wait_state(TS_PREFILL_IN);
    check("T6.1.3-a skip_WT (in PREFILL_IN)", state === TS_PREFILL_IN);

    // After DMA done, should go to SWIZZLE (skip COMPUTE + PPU)
    wait_state(TS_SWIZZLE);
    check("T6.1.3-b skip_COMPUTE (in SWIZZLE)", state === TS_SWIZZLE);

    // Pulse swizzle_done
    @(posedge clk);
    swizzle_done <= 1'b1;
    @(posedge clk);
    swizzle_done <= 1'b0;

    wait_state(TS_DONE);
    check("T6.1.3-c tile_done", tile_done === 1'b1);

    wait_state(TS_IDLE);
    check("T6.1.3-d back_to_IDLE", state === TS_IDLE);

    $display("  State trace:");
    for (int i = 0; i < state_idx && i < 32; i++)
      $display("    [%0d] %s", i, state_log[i].name());
  endtask

  // ================================================================
  // T6.1.4 — Barrier wait
  //   tile_desc.barrier_wait=1 => FSM stays in IDLE until barrier_grant=1
  // ================================================================
  task automatic test_T6_1_4();
    test_num++;
    $display("\n===== T6.1.4: Barrier wait =====");
    do_reset();
    reset_log();

    layer_desc    = make_layer_desc(PE_RS3, 10'd3, 10'd16, 4'd3, 4'd1);
    tile_desc     = make_tile_desc(.barrier_w(1'b1), .need_swz(1'b0), .last_t(1'b0));
    barrier_grant <= 1'b0;

    // Present tile with barrier_wait=1
    @(posedge clk);
    tile_valid <= 1'b1;

    // FSM should stay IDLE for several cycles (no grant)
    repeat (5) @(posedge clk);
    check("T6.1.4-a stays_IDLE (no grant)", state === TS_IDLE);
    check("T6.1.4-b barrier_wait_req",      barrier_wait_req === 1'b1);
    check("T6.1.4-c tile_accept=0",         tile_accept === 1'b0);

    // Now grant barrier
    barrier_grant <= 1'b1;
    @(posedge clk);
    check("T6.1.4-d tile_accept=1 after grant", tile_accept === 1'b1);

    @(posedge clk);
    tile_valid    <= 1'b0;
    barrier_grant <= 1'b0;

    // Should have transitioned to LOAD_DESC
    wait_state(TS_LOAD_DESC, 5);
    check("T6.1.4-e in LOAD_DESC", state === TS_LOAD_DESC);

    // Let the rest of the flow complete quickly
    wait_state(TS_COMPUTE, 20);
    seq_done <= 1'b1;
    @(posedge clk);
    seq_done <= 1'b0;

    wait_state(TS_PPU_RUN, 20);
    ppu_done <= 1'b1;
    @(posedge clk);
    ppu_done <= 1'b0;

    wait_state(TS_DONE, 20);
    check("T6.1.4-f tile_done", tile_done === 1'b1);

    wait_state(TS_IDLE, 10);
  endtask

  // ────────────────────────────────────────────────────────────
  // Test runner
  // ────────────────────────────────────────────────────────────
  initial begin
    $display("========================================");
    $display(" tb_tile_fsm — Stage 6 Control");
    $display("========================================");

    test_T6_1_1();
    test_T6_1_2();
    test_T6_1_3();
    test_T6_1_4();

    $display("\n========================================");
    $display(" SUMMARY: %0d tests, %0d PASS, %0d FAIL",
             pass_cnt + fail_cnt, pass_cnt, fail_cnt);
    if (fail_cnt == 0)
      $display(" >>> ALL TESTS PASSED <<<");
    else
      $display(" >>> SOME TESTS FAILED <<<");
    $display("========================================");
    $finish;
  end

  // Timeout safety net
  initial begin
    #50000;
    $display("[TIMEOUT] Simulation exceeded 50 us");
    $finish;
  end

endmodule
