// ============================================================================
// Testbench : tb_glb_input_bank_db
// Project   : YOLOv10n INT8 Accelerator — V4-VC707
// Description:
//   Verification of glb_input_bank_db double-buffered input SRAM.
//   5 tests:
//     T3.1.1  Basic write -> page_swap -> read -> verify all LANES match
//     T3.1.2  Double-buffer independence (AAA / BBB isolation)
//     T3.1.3  Concurrent read + write (no interference)
//     T3.1.4  Lane mask (partial mask -> only masked lanes updated)
//     T3.1.5  Address sweep (all 2048 addresses -> 0 errors)
// ============================================================================
`timescale 1ns / 1ps

module tb_glb_input_bank_db;

  import accel_pkg::*;

  // --------------------------------------------------------------------------
  //  Parameters
  // --------------------------------------------------------------------------
  localparam int LANES = accel_pkg::LANES;       // 20
  localparam int DEPTH = accel_pkg::GLB_INPUT_DEPTH; // 2048
  localparam int AW    = $clog2(DEPTH);           // 11

  // Clock: 4 ns period (250 MHz)
  localparam real CLK_PERIOD = 4.0;

  // --------------------------------------------------------------------------
  //  DUT signals
  // --------------------------------------------------------------------------
  logic              clk;
  logic              rst_n;
  logic              page_swap;

  logic [AW-1:0]     rd_addr [1];
  int8_t             rd_data [1][LANES];

  logic [AW-1:0]     wr_addr;
  int8_t             wr_data [LANES];
  logic              wr_en;
  logic [LANES-1:0]  wr_lane_mask;

  // --------------------------------------------------------------------------
  //  DUT instantiation
  // --------------------------------------------------------------------------
  glb_input_bank_db #(
    .LANES        (LANES),
    .DEPTH        (DEPTH),
    .N_READ_PORTS (1)
  ) u_dut (
    .clk          (clk),
    .rst_n        (rst_n),
    .page_swap    (page_swap),
    .rd_addr      (rd_addr),
    .rd_data      (rd_data),
    .wr_addr      (wr_addr),
    .wr_data      (wr_data),
    .wr_en        (wr_en),
    .wr_lane_mask (wr_lane_mask)
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
    rst_n      <= 1'b0;
    page_swap  <= 1'b0;
    wr_en      <= 1'b0;
    wr_addr    <= '0;
    rd_addr[0] <= '0;
    wr_lane_mask <= '0;
    for (int i = 0; i < LANES; i++) wr_data[i] <= 8'sd0;
    repeat (4) @(posedge clk);
    rst_n <= 1'b1;
    @(posedge clk);
  endtask

  task automatic write_word(input logic [AW-1:0] addr, input int8_t data [LANES],
                            input logic [LANES-1:0] mask);
    @(posedge clk);
    wr_addr      <= addr;
    wr_en        <= 1'b1;
    wr_lane_mask <= mask;
    for (int i = 0; i < LANES; i++) wr_data[i] <= data[i];
    @(posedge clk);
    wr_en <= 1'b0;
  endtask

  task automatic read_word(input logic [AW-1:0] addr, output int8_t data [LANES]);
    @(posedge clk);
    rd_addr[0] <= addr;
    @(posedge clk); // addr sampled by SRAM
    @(posedge clk); // registered output available
    for (int i = 0; i < LANES; i++) data[i] = rd_data[0][i];
  endtask

  task automatic pulse_page_swap();
    @(posedge clk);
    page_swap <= 1'b1;
    @(posedge clk);
    page_swap <= 1'b0;
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
    // T3.1.1: Basic write -> page_swap -> read -> verify all LANES match
    // ========================================================================
    begin
      int8_t wdata [LANES];
      int8_t rdata [LANES];
      bit ok = 1;

      // After reset, active_page = 0 (page A is compute/read).
      // Shadow page = B (page 1).  Write to shadow (B).
      for (int i = 0; i < LANES; i++) wdata[i] = int8_t'(i + 10);
      write_word(11'd42, wdata, {LANES{1'b1}});

      // Swap so page B becomes active (readable).
      pulse_page_swap();

      // Read from now-active page B
      read_word(11'd42, rdata);
      for (int i = 0; i < LANES; i++) begin
        if (rdata[i] !== wdata[i]) begin
          $display("  T3.1.1 lane %0d: exp %0d got %0d", i, wdata[i], rdata[i]);
          ok = 0;
        end
      end
      report("T3.1.1 Basic write->swap->read", ok);
    end

    // ========================================================================
    // T3.1.2: Double-buffer independence — AAA page A, BBB page B
    // ========================================================================
    begin
      int8_t wdataA [LANES];
      int8_t wdataB [LANES];
      int8_t rdata  [LANES];
      bit ok = 1;

      do_reset();

      // After reset: active=A(0), shadow=B(1).
      // Write 0xAA to shadow (page B), addr 100.
      for (int i = 0; i < LANES; i++) wdataB[i] = int8_t'(8'hBB);
      write_word(11'd100, wdataB, {LANES{1'b1}});

      // Swap: active=B(1), shadow=A(0).
      pulse_page_swap();

      // Write 0xAA to shadow (page A), same addr 100.
      for (int i = 0; i < LANES; i++) wdataA[i] = int8_t'(8'hAA);
      write_word(11'd100, wdataA, {LANES{1'b1}});

      // Read from active page B — should be 0xBB.
      read_word(11'd100, rdata);
      for (int i = 0; i < LANES; i++) begin
        if (rdata[i] !== int8_t'(8'hBB)) begin
          $display("  T3.1.2 page-B lane %0d: exp 0xBB got 0x%02X", i, rdata[i]);
          ok = 0;
        end
      end

      // Swap: active=A(0), shadow=B(1).
      pulse_page_swap();

      // Read from active page A — should be 0xAA.
      read_word(11'd100, rdata);
      for (int i = 0; i < LANES; i++) begin
        if (rdata[i] !== int8_t'(8'hAA)) begin
          $display("  T3.1.2 page-A lane %0d: exp 0xAA got 0x%02X", i, rdata[i]);
          ok = 0;
        end
      end
      report("T3.1.2 Double-buffer independence", ok);
    end

    // ========================================================================
    // T3.1.3: Concurrent read + write — read page A while writing page B
    // ========================================================================
    begin
      int8_t wdataA [LANES];
      int8_t wdataB [LANES];
      int8_t rdata  [LANES];
      bit ok = 1;

      do_reset();

      // Write known data to shadow (page B), addr 50.
      for (int i = 0; i < LANES; i++) wdataB[i] = int8_t'(i + 50);
      write_word(11'd50, wdataB, {LANES{1'b1}});

      // Swap: active=B, shadow=A.
      pulse_page_swap();

      // Write known data to shadow (page A), addr 60.
      for (int i = 0; i < LANES; i++) wdataA[i] = int8_t'(i + 60);
      write_word(11'd60, wdataA, {LANES{1'b1}});

      // Swap: active=A, shadow=B.
      pulse_page_swap();

      // Now perform concurrent: read from active page A addr 60,
      // write to shadow page B addr 70 at the same time.
      for (int i = 0; i < LANES; i++) wr_data[i] <= int8_t'(i + 70);
      @(posedge clk);
      rd_addr[0]   <= 11'd60;
      wr_addr      <= 11'd70;
      wr_en        <= 1'b1;
      wr_lane_mask <= {LANES{1'b1}};
      @(posedge clk);
      wr_en <= 1'b0;

      // Wait 1 cycle for registered read output.
      @(posedge clk);
      for (int i = 0; i < LANES; i++) begin
        if (rd_data[0][i] !== wdataA[i]) begin
          $display("  T3.1.3 lane %0d: exp %0d got %0d", i, wdataA[i], rd_data[0][i]);
          ok = 0;
        end
      end
      report("T3.1.3 Concurrent read+write no interference", ok);
    end

    // ========================================================================
    // T3.1.4: Lane mask — partial mask -> only masked lanes updated
    // ========================================================================
    begin
      int8_t wdata_init [LANES];
      int8_t wdata_partial [LANES];
      int8_t rdata [LANES];
      logic [LANES-1:0] mask;
      bit ok = 1;

      do_reset();

      // Write initial value (0x11) to all lanes on shadow (page B), addr 200.
      for (int i = 0; i < LANES; i++) wdata_init[i] = int8_t'(8'h11);
      write_word(11'd200, wdata_init, {LANES{1'b1}});

      // Now write 0x77 with even-lane mask only (lanes 0,2,4,...).
      for (int i = 0; i < LANES; i++) wdata_partial[i] = int8_t'(8'h77);
      mask = '0;
      for (int i = 0; i < LANES; i += 2) mask[i] = 1'b1;
      write_word(11'd200, wdata_partial, mask);

      // Swap and read.
      pulse_page_swap();
      read_word(11'd200, rdata);

      for (int i = 0; i < LANES; i++) begin
        if (i % 2 == 0) begin
          // Even lanes: should be updated to 0x77.
          if (rdata[i] !== int8_t'(8'h77)) begin
            $display("  T3.1.4 lane %0d (masked): exp 0x77 got 0x%02X", i, rdata[i]);
            ok = 0;
          end
        end else begin
          // Odd lanes: should remain 0x11.
          if (rdata[i] !== int8_t'(8'h11)) begin
            $display("  T3.1.4 lane %0d (unmasked): exp 0x11 got 0x%02X", i, rdata[i]);
            ok = 0;
          end
        end
      end
      report("T3.1.4 Lane mask partial update", ok);
    end

    // ========================================================================
    // T3.1.5: Address sweep — write all 2048 addresses, read back, 0 errors
    // ========================================================================
    begin
      int8_t wdata [LANES];
      int8_t rdata [LANES];
      int err_cnt = 0;
      bit ok;

      do_reset();

      // Write all addresses to shadow page (page B).
      for (int a = 0; a < DEPTH; a++) begin
        for (int i = 0; i < LANES; i++)
          wdata[i] = int8_t'((a + i) & 8'hFF);
        write_word(a[AW-1:0], wdata, {LANES{1'b1}});
      end

      // Swap: active=B.
      pulse_page_swap();

      // Read back all addresses and verify.
      for (int a = 0; a < DEPTH; a++) begin
        read_word(a[AW-1:0], rdata);
        for (int i = 0; i < LANES; i++) begin
          int8_t expected = int8_t'((a + i) & 8'hFF);
          if (rdata[i] !== expected) begin
            if (err_cnt < 20)
              $display("  T3.1.5 addr %0d lane %0d: exp %0d got %0d",
                       a, i, expected, rdata[i]);
            err_cnt++;
          end
        end
      end
      ok = (err_cnt == 0);
      if (!ok)
        $display("  T3.1.5 total errors: %0d / %0d",
                 err_cnt, DEPTH * LANES);
      report("T3.1.5 Address sweep (2048 addrs)", ok);
    end

    // ========================================================================
    //  Final summary
    // ========================================================================
    $display("");
    $display("==============================================================");
    $display("  tb_glb_input_bank_db — %0d/%0d tests PASSED",
             pass_count, total_tests);
    if (fail_count == 0)
      $display("  >>> ALL TESTS PASSED <<<");
    else
      $display("  >>> %0d TEST(S) FAILED <<<", fail_count);
    $display("==============================================================");
    $finish;
  end

endmodule
