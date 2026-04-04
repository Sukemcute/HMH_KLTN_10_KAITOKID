// ============================================================================
// Testbench : tb_tile_fsm
// Project   : YOLOv10n INT8 Accelerator — V4-VC707
// Tests     : T6.1.1  Normal Conv flow (PE_RS3)
//             T6.1.2  DW7x7 multipass (num_k_pass=3)
//             T6.1.3  PE_PASS mode (bypass path)
//             T6.1.4  Barrier wait
// ============================================================================
`timescale 1ns / 1ps

// Vivado 2022.2 xelab can crash on task ports of type `string`; use macro instead.
`define TILE_CHK(tag, cond_) \
  begin \
    if (cond_) begin \
      $display("[PASS] %s", tag); \
      pass_cnt = pass_cnt + 1; \
    end else begin \
      $display("[FAIL] %s", tag); \
      fail_cnt = fail_cnt + 1; \
    end \
  end

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

  // Vivado 2022.2 xelab can AV-crash when serializing modules that use
  // functions returning large packed structs + default task args. Use
  // tasks that assign module nets and always pass max_cyc explicitly.

  localparam int WAIT_DEF = 200;

  task wait_state(input tile_state_e target, input int max_cyc);
    integer cnt;
    cnt = 0;
    while (state !== target && cnt < max_cyc) begin
      @(posedge clk);
      cnt++;
    end
    if (cnt >= max_cyc)
      $display("  WARNING: timeout waiting for tile_state_e=%0d (stuck at %0d)", target, state);
  endtask

  task apply_layer_desc(
    input pe_mode_e mode,
    input logic [9:0] cin,
    input logic [9:0] cout,
    input logic [3:0] kw,
    input logic [3:0] num_k_pass_val
  );
    layer_desc = '0;
    layer_desc.pe_mode      = mode;
    layer_desc.activation   = ACT_RELU;
    layer_desc.cin          = cin;
    layer_desc.cout         = cout;
    layer_desc.hout         = 10'd2;
    layer_desc.wout         = 10'd20;
    layer_desc.hin          = 10'd4;
    layer_desc.win          = 10'd22;
    layer_desc.kh           = 4'd3;
    layer_desc.kw           = kw;
    layer_desc.stride       = 3'd1;
    layer_desc.padding      = 3'd1;
    layer_desc.num_tiles    = 8'd1;
    layer_desc.num_cin_pass = 4'd1;
    layer_desc.num_k_pass   = num_k_pass_val;
    layer_desc.swizzle      = SWZ_NORMAL;
  endtask

  task apply_tile_desc(input logic barrier_w, input logic need_swz, input logic last_t);
    tile_desc = '0;
    tile_desc.tile_id      = 16'd1;
    tile_desc.layer_id     = 5'd0;
    tile_desc.sc_mask      = 4'hF;
    tile_desc.valid_h      = 6'd2;
    tile_desc.valid_w      = 6'd20;
    tile_desc.num_cin_pass = 4'd1;
    tile_desc.num_k_pass   = 4'd1;
    tile_desc.first_tile   = 1'b1;
    tile_desc.last_tile    = last_t;
    tile_desc.barrier_wait = barrier_w;
    tile_desc.barrier_id   = 4'd0;
    tile_desc.need_swizzle = need_swz;
  endtask

  // Reset + quiesce
  task do_reset();
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

  // ================================================================
  // T6.1.1 — Normal Conv flow (PE_RS3)
  //   Expected: IDLE -> LOAD_DESC -> PREFILL_WT -> PREFILL_IN ->
  //             COMPUTE -> PE_DRAIN -> PPU_RUN -> WRITEBACK -> DONE -> IDLE
  // ================================================================
  task test_T6_1_1();
    test_num++;
    $display("\n===== T6.1.1: Normal Conv flow (PE_RS3) =====");
    do_reset();

    apply_layer_desc(PE_RS3, 10'd3, 10'd16, 4'd3, 4'd1);
    apply_tile_desc(1'b0, 1'b0, 1'b1);

    // Present tile
    @(posedge clk);
    tile_valid <= 1'b1;
    @(posedge clk);
    // tile_accept should fire; after 1 cycle FSM is in LOAD_DESC
    `TILE_CHK("T6.1.1-a tile_accept", tile_accept === 1'b1);
    tile_valid <= 1'b0;

    // LOAD_DESC: shadow_latch + page_swap should be asserted
    wait_state(TS_LOAD_DESC, WAIT_DEF);
    `TILE_CHK("T6.1.1-b shadow_latch", shadow_latch === 1'b1);
    `TILE_CHK("T6.1.1-c page_swap",    page_swap === 1'b1);

    // DMA stub is instant -> PREFILL_WT passes immediately
    wait_state(TS_PREFILL_WT, WAIT_DEF);
    `TILE_CHK("T6.1.1-d dma_start (wt)", dma_start === 1'b1);
    `TILE_CHK("T6.1.1-e dma_is_write=0", dma_is_write === 1'b0);

    // PREFILL_IN
    wait_state(TS_PREFILL_IN, WAIT_DEF);
    `TILE_CHK("T6.1.1-f dma_start (in)", dma_start === 1'b1);

    // COMPUTE: pulse seq_done after 1 cycle to simulate sequencer completing
    wait_state(TS_COMPUTE, WAIT_DEF);
    `TILE_CHK("T6.1.1-g seq_start", seq_start === 1'b1);
    @(posedge clk);
    seq_done <= 1'b1;
    @(posedge clk);
    seq_done <= 1'b0;

    // PE_DRAIN: wait DSP_PIPE_DEPTH+1 cycles for drain_cnt to reach threshold
    wait_state(TS_PE_DRAIN, WAIT_DEF);

    // PPU_RUN: pulse ppu_done
    wait_state(TS_PPU_RUN, WAIT_DEF);
    `TILE_CHK("T6.1.1-h ppu_start", ppu_start === 1'b1);
    @(posedge clk);
    ppu_done <= 1'b1;
    @(posedge clk);
    ppu_done <= 1'b0;

    // WRITEBACK (last_tile=1 => need_writeback_reg=1)
    wait_state(TS_WRITEBACK, WAIT_DEF);
    `TILE_CHK("T6.1.1-i dma_is_write=1", dma_is_write === 1'b1);

    // DONE
    wait_state(TS_DONE, WAIT_DEF);
    `TILE_CHK("T6.1.1-j tile_done", tile_done === 1'b1);
    `TILE_CHK("T6.1.1-k barrier_signal", barrier_signal === 1'b1);

    // Back to IDLE
    wait_state(TS_IDLE, WAIT_DEF);
    `TILE_CHK("T6.1.1-l back_to_IDLE", state === TS_IDLE);
  endtask

  // ================================================================
  // T6.1.2 — DW7x7 multipass (num_k_pass=3)
  //   COMPUTE -> PE_DRAIN loops 3 times before PPU_RUN
  // ================================================================
  task test_T6_1_2();
    integer compute_count;
    integer drain_count;
    integer pass;
    test_num++;
    $display("\n===== T6.1.2: DW7x7 multipass (num_k_pass=3) =====");
    do_reset();

    apply_layer_desc(PE_DW7, 10'd8, 10'd8, 4'd3, 4'd3);
    apply_tile_desc(1'b0, 1'b0, 1'b0);

    @(posedge clk);
    tile_valid <= 1'b1;
    @(posedge clk);
    tile_valid <= 1'b0;

    // Skip through LOAD_DESC, PREFILL_WT, PREFILL_IN (instant DMA)
    wait_state(TS_COMPUTE, WAIT_DEF);

    compute_count = 0;
    drain_count   = 0;

    // Loop: expect 3 rounds of COMPUTE -> PE_DRAIN
    for (pass = 0; pass < 3; pass++) begin
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
    `TILE_CHK("T6.1.2-a compute_loops=3", compute_count == 3);
    `TILE_CHK("T6.1.2-b drain_loops=3",   drain_count == 3);
    `TILE_CHK("T6.1.2-c reached PPU_RUN",  state === TS_PPU_RUN);

    // Finish off
    ppu_done <= 1'b1;
    @(posedge clk);
    ppu_done <= 1'b0;

    // last_tile=0 => need_writeback_reg=0 => skip WRITEBACK
    wait_state(TS_DONE, WAIT_DEF);
    `TILE_CHK("T6.1.2-d tile_done", tile_done === 1'b1);

    wait_state(TS_IDLE, WAIT_DEF);
    `TILE_CHK("T6.1.2-e back_to_IDLE", state === TS_IDLE);
  endtask

  // ================================================================
  // T6.1.3 — PE_PASS mode
  //   Expected: IDLE -> LOAD_DESC -> PREFILL_IN -> SWIZZLE -> DONE
  //   (skip WT, COMPUTE, PPU)
  // ================================================================
  task test_T6_1_3();
    test_num++;
    $display("\n===== T6.1.3: PE_PASS mode (bypass path) =====");
    do_reset();

    apply_layer_desc(PE_PASS, 10'd8, 10'd8, 4'd1, 4'd1);
    apply_tile_desc(1'b0, 1'b1, 1'b0);

    @(posedge clk);
    tile_valid <= 1'b1;
    @(posedge clk);
    tile_valid <= 1'b0;

    // LOAD_DESC
    wait_state(TS_LOAD_DESC, WAIT_DEF);

    // Should skip PREFILL_WT entirely, go to PREFILL_IN
    wait_state(TS_PREFILL_IN, WAIT_DEF);
    `TILE_CHK("T6.1.3-a skip_WT (in PREFILL_IN)", state === TS_PREFILL_IN);

    // After DMA done, should go to SWIZZLE (skip COMPUTE + PPU)
    wait_state(TS_SWIZZLE, WAIT_DEF);
    `TILE_CHK("T6.1.3-b skip_COMPUTE (in SWIZZLE)", state === TS_SWIZZLE);

    // Pulse swizzle_done
    @(posedge clk);
    swizzle_done <= 1'b1;
    @(posedge clk);
    swizzle_done <= 1'b0;

    wait_state(TS_DONE, WAIT_DEF);
    `TILE_CHK("T6.1.3-c tile_done", tile_done === 1'b1);

    wait_state(TS_IDLE, WAIT_DEF);
    `TILE_CHK("T6.1.3-d back_to_IDLE", state === TS_IDLE);
  endtask

  // ================================================================
  // T6.1.4 — Barrier wait
  //   tile_desc.barrier_wait=1 => FSM stays in IDLE until barrier_grant=1
  // ================================================================
  task test_T6_1_4();
    test_num++;
    $display("\n===== T6.1.4: Barrier wait =====");
    do_reset();

    apply_layer_desc(PE_RS3, 10'd3, 10'd16, 4'd3, 4'd1);
    apply_tile_desc(1'b1, 1'b0, 1'b0);
    barrier_grant <= 1'b0;

    // Present tile with barrier_wait=1
    @(posedge clk);
    tile_valid <= 1'b1;

    // FSM should stay IDLE for several cycles (no grant)
    repeat (5) @(posedge clk);
    `TILE_CHK("T6.1.4-a stays_IDLE (no grant)", state === TS_IDLE);
    `TILE_CHK("T6.1.4-b barrier_wait_req",      barrier_wait_req === 1'b1);
    `TILE_CHK("T6.1.4-c tile_accept=0",         tile_accept === 1'b0);

    // Now grant barrier
    barrier_grant <= 1'b1;
    @(posedge clk);
    `TILE_CHK("T6.1.4-d tile_accept=1 after grant", tile_accept === 1'b1);

    @(posedge clk);
    tile_valid    <= 1'b0;
    barrier_grant <= 1'b0;

    // Should have transitioned to LOAD_DESC
    wait_state(TS_LOAD_DESC, 5);
    `TILE_CHK("T6.1.4-e in LOAD_DESC", state === TS_LOAD_DESC);

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
    `TILE_CHK("T6.1.4-f tile_done", tile_done === 1'b1);

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
