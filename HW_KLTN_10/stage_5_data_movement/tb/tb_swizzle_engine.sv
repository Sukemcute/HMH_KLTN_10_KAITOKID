// ============================================================================
// Testbench : tb_swizzle_engine
// Project   : YOLOv10n INT8 Accelerator — V4-VC707
// Tests     : T5.3.1 SWZ_UPSAMPLE2X — 5x20 → 10x40, pixel duplication
//             T5.3.2 SWZ_CONCAT — data passthrough with correct addressing
//             T5.3.3 SWZ_NORMAL — done immediately (identity)
// ============================================================================
`timescale 1ns / 1ps

module tb_swizzle_engine;
  import accel_pkg::*;

  // ──────────────────────────────────────────────────────────────
  //  Parameters
  // ──────────────────────────────────────────────────────────────
  localparam int LANES   = accel_pkg::LANES;  // 20
  localparam int CLK_NS  = 4;                 // 250 MHz

  // ──────────────────────────────────────────────────────────────
  //  DUT signals
  // ──────────────────────────────────────────────────────────────
  logic          clk, rst_n;
  logic          start;
  swizzle_mode_e cfg_mode;
  logic [9:0]    cfg_src_h, cfg_src_w, cfg_src_c;
  logic [9:0]    cfg_dst_h, cfg_dst_w;

  logic          src_rd_en;
  logic [11:0]   src_rd_addr;
  int8_t         src_rd_data [LANES];

  logic          dst_wr_en;
  logic [11:0]   dst_wr_addr;
  int8_t         dst_wr_data [LANES];
  logic [LANES-1:0] dst_wr_mask;

  logic          done;

  // ──────────────────────────────────────────────────────────────
  //  Source memory model (read-only)
  //  Simple memory: address → LANES int8 values
  // ──────────────────────────────────────────────────────────────
  localparam int SRC_MEM_DEPTH = 4096;
  int8_t src_mem [SRC_MEM_DEPTH][LANES];

  // Combinational read — models SRAM array output before the DUT's
  // internal rd_data_lat register (which provides the 1-cycle pipeline).
  // Using always_comb avoids the NBA-to-NBA simulation artifact where
  // two back-to-back always_ff blocks create 2 cycles of delay instead of 1.
  always_comb begin
    for (int l = 0; l < LANES; l++) begin
      if (src_rd_addr < SRC_MEM_DEPTH)
        src_rd_data[l] = src_mem[src_rd_addr][l];
      else
        src_rd_data[l] = 8'sd0;
    end
  end

  // ──────────────────────────────────────────────────────────────
  //  Destination capture memory (write-only from DUT)
  // ──────────────────────────────────────────────────────────────
  localparam int DST_MEM_DEPTH = 4096;
  int8_t dst_mem [DST_MEM_DEPTH][LANES];
  int    dst_wr_count;

  always_ff @(posedge clk) begin
    if (dst_wr_en && dst_wr_addr < DST_MEM_DEPTH) begin
      for (int l = 0; l < LANES; l++)
        if (dst_wr_mask[l])
          dst_mem[dst_wr_addr][l] <= dst_wr_data[l];
      dst_wr_count <= dst_wr_count + 1;
    end
  end

  // ──────────────────────────────────────────────────────────────
  //  DUT instantiation
  // ──────────────────────────────────────────────────────────────
  // Signals for source B (ewise_add) — tie off for basic tests
  logic          src_b_rd_en;
  logic [11:0]   src_b_rd_addr;
  int8_t         src_b_rd_data [LANES];

  // Source B memory model (for ewise_add tests)
  int8_t src_b_mem [SRC_MEM_DEPTH][LANES];

  always_comb begin
    for (int l = 0; l < LANES; l++) begin
      if (src_b_rd_addr < SRC_MEM_DEPTH)
        src_b_rd_data[l] = src_b_mem[src_b_rd_addr][l];
      else
        src_b_rd_data[l] = 8'sd0;
    end
  end

  swizzle_engine #(.LANES(LANES)) u_dut (
    .clk              (clk),
    .rst_n            (rst_n),
    .start            (start),
    .cfg_mode         (cfg_mode),
    .cfg_src_h        (cfg_src_h),
    .cfg_src_w        (cfg_src_w),
    .cfg_src_c        (cfg_src_c),
    .cfg_dst_h        (cfg_dst_h),
    .cfg_dst_w        (cfg_dst_w),
    // Domain alignment — defaults for non-ewise tests
    .cfg_align_m_a    (32'd1),
    .cfg_align_sh_a   (8'd0),
    .cfg_align_zp_a   (8'sd0),
    .cfg_align_m_b    (32'd1),
    .cfg_align_sh_b   (8'd0),
    .cfg_align_zp_b   (8'sd0),
    .cfg_align_zp_out (8'sd0),
    .cfg_align_bypass (1'b1),
    // Source A
    .src_rd_en        (src_rd_en),
    .src_rd_addr      (src_rd_addr),
    .src_rd_data      (src_rd_data),
    // Source B
    .src_b_rd_en      (src_b_rd_en),
    .src_b_rd_addr    (src_b_rd_addr),
    .src_b_rd_data    (src_b_rd_data),
    // Destination
    .dst_wr_en        (dst_wr_en),
    .dst_wr_addr      (dst_wr_addr),
    .dst_wr_data      (dst_wr_data),
    .dst_wr_mask      (dst_wr_mask),
    .done             (done)
  );

  // ──────────────────────────────────────────────────────────────
  //  Clock generation: 250 MHz (4 ns period)
  // ──────────────────────────────────────────────────────────────
  initial clk = 1'b0;
  always #(CLK_NS/2) clk = ~clk;

  // ──────────────────────────────────────────────────────────────
  //  Scoreboard
  // ──────────────────────────────────────────────────────────────
  int pass_cnt = 0;
  int fail_cnt = 0;
  int test_cnt = 0;

  task automatic check(string tag, logic cond, string msg);
    test_cnt++;
    if (cond) begin
      pass_cnt++;
    end else begin
      fail_cnt++;
      $display("[FAIL] %s : %s", tag, msg);
    end
  endtask

  // Helper: start operation and wait for done
  task automatic run_and_wait(input int timeout_cycles = 10000);
    @(posedge clk);
    start <= 1'b1;
    @(posedge clk);
    start <= 1'b0;

    begin
      automatic int cnt = 0;
      while (!done && cnt < timeout_cycles) begin
        @(posedge clk);
        cnt++;
      end
      if (cnt >= timeout_cycles)
        $display("[WARN] Timeout waiting for done after %0d cycles", timeout_cycles);
    end
  endtask

  // Helper: clear memories
  task automatic clear_memories();
    for (int a = 0; a < SRC_MEM_DEPTH; a++)
      for (int l = 0; l < LANES; l++)
        src_mem[a][l] = 8'sd0;
    for (int a = 0; a < DST_MEM_DEPTH; a++)
      for (int l = 0; l < LANES; l++)
        dst_mem[a][l] = 8'sd0;
    dst_wr_count = 0;
  endtask

  // ──────────────────────────────────────────────────────────────
  //  Main test sequence
  // ──────────────────────────────────────────────────────────────
  initial begin
    $display("===========================================================");
    $display(" tb_swizzle_engine — START");
    $display("===========================================================");

    // Reset
    rst_n   = 1'b0;
    start   = 1'b0;
    cfg_mode = SWZ_NORMAL;
    cfg_src_h = '0; cfg_src_w = '0; cfg_src_c = '0;
    cfg_dst_h = '0; cfg_dst_w = '0;
    clear_memories();
    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // ════════════════════════════════════════════════════════════
    //  T5.3.1: SWZ_UPSAMPLE2X — 5h x 20w → 10h x 20w (height dup)
    //   src: 5 rows, 1 channel, 1 wblk (20 pixels wide)
    //   dst: 10 rows, 1 wblk (height doubled by block duplication)
    //
    //   src_wblk_total = ceil(20/20) = 1
    //   dst_wblk_total = ceil(20/20) = 1
    //   For each src (h, c, w):
    //     UP_WRITE_0: row 2h,   same wblk → addr = (2h)*C*1 + c*1 + w
    //     UP_WRITE_1: row 2h+1, same wblk → addr = (2h+1)*C*1 + c*1 + w
    //   Total writes = src_h * src_c * src_wblk_total * 2 = 5*1*1*2 = 10
    // ════════════════════════════════════════════════════════════
    $display("\n--- T5.3.1: SWZ_UPSAMPLE2X ---");
    clear_memories();
    cfg_mode  = SWZ_UPSAMPLE2X;
    cfg_src_h = 10'd5;
    cfg_src_w = 10'd20;
    cfg_src_c = 10'd1;
    cfg_dst_h = 10'd10;
    cfg_dst_w = 10'd20;

    // Fill source: addr = src_h_off + c*1 + w.
    // src_h_off = h * C * src_wblk = h * 1 * 1 = h.  So addr(h)=h.
    for (int a = 0; a < 10; a++)
      for (int l = 0; l < LANES; l++)
        src_mem[a][l] = 8'(a * 10 + l + 1);

    run_and_wait(5000);

    check("T5.3.1_done", done == 1'b1,
          "Upsample should complete");

    check("T5.3.1_wr_count", dst_wr_count == 10,
          $sformatf("Expected 10 writes, got %0d", dst_wr_count));

    // For h=0: UP_WRITE_0 → dst addr 0, UP_WRITE_1 → dst addr 1
    // Both should contain src_mem[0] data (src addr for h=0 is 0).
    begin
      automatic logic data_ok = 1'b1;
      for (int l = 0; l < LANES; l++) begin
        if (dst_mem[0][l] != src_mem[0][l]) data_ok = 1'b0;
        if (dst_mem[1][l] != src_mem[0][l]) data_ok = 1'b0;
      end
      check("T5.3.1_dup", data_ok,
            $sformatf("UP_WRITE_0/1 should duplicate src data, dst[0][0]=%0d dst[1][0]=%0d src[0][0]=%0d",
                      dst_mem[0][0], dst_mem[1][0], src_mem[0][0]));
    end

    // ════════════════════════════════════════════════════════════
    //  T5.3.2: SWZ_CONCAT — data passthrough with correct addressing
    //   src: 2h x 20w, 2 channels → dst: 2h x 20w
    //   src_wblk_total = 1, dst_wblk_total = 1
    //   FSM: SW_READ → SW_READ_WAIT → SW_WRITE
    //   dst_addr = c * dst_wblk_total + w = c * 1 + w
    //   Total writes = src_h * src_c * src_wblk_total = 2 * 2 * 1 = 4
    // ════════════════════════════════════════════════════════════
    $display("\n--- T5.3.2: SWZ_CONCAT ---");
    clear_memories();
    cfg_mode  = SWZ_CONCAT;
    cfg_src_h = 10'd2;
    cfg_src_w = 10'd20;
    cfg_src_c = 10'd2;
    cfg_dst_h = 10'd2;
    cfg_dst_w = 10'd20;

    // Fill source: addr = c*1 + w, so addr 0 = (c=0,w=0), addr 1 = (c=1,w=0)
    for (int a = 0; a < 4; a++)
      for (int l = 0; l < LANES; l++)
        src_mem[a][l] = 8'(a * 20 + l + 5);

    run_and_wait(5000);

    check("T5.3.2_done", done == 1'b1,
          "Concat should complete");

    // Total writes: h=2, c=2, wblk=1 → 4 writes
    check("T5.3.2_wr_count", dst_wr_count == 4,
          $sformatf("Expected 4 writes, got %0d", dst_wr_count));

    // Verify data passthrough: dst_mem should contain src_mem data
    // dst_addr = c * dst_wblk_total + w = same as src_addr
    begin
      automatic logic data_ok = 1'b1;
      for (int a = 0; a < 2; a++)  // check first 2 addresses (h=0)
        for (int l = 0; l < LANES; l++)
          if (dst_mem[a][l] != src_mem[a][l]) data_ok = 1'b0;
      check("T5.3.2_data", data_ok,
            "Concat: dst data should match src data");
    end

    // Verify dst_wr_mask is all ones
    // (checked implicitly since we captured data with mask)

    // ════════════════════════════════════════════════════════════
    //  T5.3.3: SWZ_NORMAL — done immediately (identity)
    //   FSM transitions: SW_IDLE → start → SW_DONE (skips READ/WRITE)
    //   No reads, no writes.
    // ════════════════════════════════════════════════════════════
    $display("\n--- T5.3.3: SWZ_NORMAL ---");
    clear_memories();
    cfg_mode  = SWZ_NORMAL;
    cfg_src_h = 10'd10;
    cfg_src_w = 10'd20;
    cfg_src_c = 10'd3;
    cfg_dst_h = 10'd10;
    cfg_dst_w = 10'd20;

    // Put data in source to verify it's NOT read
    for (int a = 0; a < 10; a++)
      for (int l = 0; l < LANES; l++)
        src_mem[a][l] = 8'(a + l + 1);

    run_and_wait(100);

    check("T5.3.3_done", done == 1'b1,
          "SWZ_NORMAL should complete immediately");

    check("T5.3.3_no_writes", dst_wr_count == 0,
          $sformatf("SWZ_NORMAL: expected 0 writes, got %0d", dst_wr_count));

    // Verify: no src reads happened (DUT src_rd_en should never have pulsed)
    // We check that dst memory is still all zeros
    begin
      automatic logic all_zero = 1'b1;
      for (int a = 0; a < 10; a++)
        for (int l = 0; l < LANES; l++)
          if (dst_mem[a][l] != 8'sd0) all_zero = 1'b0;
      check("T5.3.3_no_data", all_zero,
            "SWZ_NORMAL: dst memory should remain untouched");
    end

    // ════════════════════════════════════════════════════════════
    //  Summary
    // ════════════════════════════════════════════════════════════
    $display("\n===========================================================");
    $display(" tb_swizzle_engine — RESULTS");
    $display("   Total : %0d", test_cnt);
    $display("   PASS  : %0d", pass_cnt);
    $display("   FAIL  : %0d", fail_cnt);
    if (fail_cnt == 0)
      $display("   >>> ALL TESTS PASSED <<<");
    else
      $display("   >>> SOME TESTS FAILED <<<");
    $display("===========================================================");
    $finish;
  end

endmodule
