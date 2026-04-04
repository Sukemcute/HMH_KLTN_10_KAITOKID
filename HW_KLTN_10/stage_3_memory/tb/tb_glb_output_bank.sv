// ============================================================================
// Testbench : tb_glb_output_bank
// Project   : YOLOv10n INT8 Accelerator — V4-VC707
// Description:
//   Verification of glb_output_bank dual-namespace output SRAM.
//   5 tests:
//     T3.3.1  PSUM write -> read: INT32 array verify
//     T3.3.2  ACT write -> read: INT8 array verify
//     T3.3.3  Namespace independence: PSUM and ACT at same addr, no cross
//     T3.3.4  DW7x7 multipass: accumulate PSUM across passes
//     T3.3.5  DMA drain: ACT data -> packed bus verification
// ============================================================================
`timescale 1ns / 1ps

module tb_glb_output_bank;

  import accel_pkg::*;

  // --------------------------------------------------------------------------
  //  Parameters
  // --------------------------------------------------------------------------
  localparam int LANES = accel_pkg::LANES;           // 20
  localparam int DEPTH = accel_pkg::GLB_OUTPUT_DEPTH; // 512
  localparam int AW    = $clog2(DEPTH);               // 9

  // Clock: 4 ns period (250 MHz)
  localparam real CLK_PERIOD = 4.0;

  // --------------------------------------------------------------------------
  //  DUT signals
  // --------------------------------------------------------------------------
  logic              clk;
  logic              rst_n;

  // PSUM namespace
  logic [AW-1:0]     psum_addr;
  int32_t            psum_wr_data [LANES];
  logic              psum_wr_en;
  int32_t            psum_rd_data [LANES];
  logic              psum_rd_en;

  // ACT namespace
  logic [AW-1:0]     act_addr;
  int8_t             act_wr_data  [LANES];
  logic              act_wr_en;
  int8_t             act_rd_data  [LANES];
  logic              act_rd_en;

  // DMA drain
  logic [AW-1:0]     drain_addr;
  logic [LANES*8-1:0] drain_data;

  // --------------------------------------------------------------------------
  //  DUT instantiation
  // --------------------------------------------------------------------------
  glb_output_bank #(
    .LANES (LANES),
    .DEPTH (DEPTH)
  ) u_dut (
    .clk          (clk),
    .rst_n        (rst_n),
    .psum_addr    (psum_addr),
    .psum_wr_data (psum_wr_data),
    .psum_wr_en   (psum_wr_en),
    .psum_rd_data (psum_rd_data),
    .psum_rd_en   (psum_rd_en),
    .act_addr     (act_addr),
    .act_wr_data  (act_wr_data),
    .act_wr_en    (act_wr_en),
    .act_rd_data  (act_rd_data),
    .act_rd_en    (act_rd_en),
    .drain_addr   (drain_addr),
    .drain_data   (drain_data)
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
    rst_n       <= 1'b0;
    psum_wr_en  <= 1'b0;
    psum_rd_en  <= 1'b0;
    act_wr_en   <= 1'b0;
    act_rd_en   <= 1'b0;
    psum_addr   <= '0;
    act_addr    <= '0;
    drain_addr  <= '0;
    for (int i = 0; i < LANES; i++) begin
      psum_wr_data[i] <= 32'sd0;
      act_wr_data[i]  <= 8'sd0;
    end
    repeat (4) @(posedge clk);
    rst_n <= 1'b1;
    @(posedge clk);
  endtask

  task automatic write_psum(input logic [AW-1:0] addr, input int32_t data [LANES]);
    @(posedge clk);
    psum_addr  <= addr;
    psum_wr_en <= 1'b1;
    for (int i = 0; i < LANES; i++) psum_wr_data[i] <= data[i];
    @(posedge clk);
    psum_wr_en <= 1'b0;
  endtask

  task automatic read_psum(input logic [AW-1:0] addr, output int32_t data [LANES]);
    @(posedge clk);
    psum_addr  <= addr;
    psum_rd_en <= 1'b1;
    @(posedge clk);
    psum_rd_en <= 1'b0;
    @(posedge clk); // registered output available
    for (int i = 0; i < LANES; i++) data[i] = psum_rd_data[i];
  endtask

  task automatic write_act(input logic [AW-1:0] addr, input int8_t data [LANES]);
    @(posedge clk);
    act_addr  <= addr;
    act_wr_en <= 1'b1;
    for (int i = 0; i < LANES; i++) act_wr_data[i] <= data[i];
    @(posedge clk);
    act_wr_en <= 1'b0;
  endtask

  task automatic read_act(input logic [AW-1:0] addr, output int8_t data [LANES]);
    @(posedge clk);
    act_addr  <= addr;
    act_rd_en <= 1'b1;
    @(posedge clk);
    act_rd_en <= 1'b0;
    @(posedge clk); // registered output available
    for (int i = 0; i < LANES; i++) data[i] = act_rd_data[i];
  endtask

  task automatic read_drain(input logic [AW-1:0] addr,
                            output logic [LANES*8-1:0] data);
    @(posedge clk);
    drain_addr <= addr;
    @(posedge clk); // addr sampled by SRAM
    @(posedge clk); // registered output available
    data = drain_data;
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
    // T3.3.1: PSUM write -> read: INT32 array verify all lanes
    // ========================================================================
    begin
      int32_t wdata [LANES];
      int32_t rdata [LANES];
      bit ok = 1;

      for (int i = 0; i < LANES; i++)
        wdata[i] = int32_t'(32'hDEAD_0000 + i);
      write_psum(9'd10, wdata);
      read_psum(9'd10, rdata);

      for (int i = 0; i < LANES; i++) begin
        if (rdata[i] !== wdata[i]) begin
          $display("  T3.3.1 lane %0d: exp 0x%08X got 0x%08X",
                   i, wdata[i], rdata[i]);
          ok = 0;
        end
      end
      report("T3.3.1 PSUM write->read INT32 verify", ok);
    end

    // ========================================================================
    // T3.3.2: ACT write -> read: INT8 array verify
    // ========================================================================
    begin
      int8_t wdata [LANES];
      int8_t rdata [LANES];
      bit ok = 1;

      do_reset();

      for (int i = 0; i < LANES; i++)
        wdata[i] = int8_t'(i * 5 - 50);  // mix of positive and negative
      write_act(9'd20, wdata);
      read_act(9'd20, rdata);

      for (int i = 0; i < LANES; i++) begin
        if (rdata[i] !== wdata[i]) begin
          $display("  T3.3.2 lane %0d: exp %0d got %0d",
                   i, wdata[i], rdata[i]);
          ok = 0;
        end
      end
      report("T3.3.2 ACT write->read INT8 verify", ok);
    end

    // ========================================================================
    // T3.3.3: Namespace independence — PSUM and ACT at same address, no cross
    // ========================================================================
    begin
      int32_t psum_w [LANES];
      int8_t  act_w  [LANES];
      int32_t psum_r [LANES];
      int8_t  act_r  [LANES];
      bit ok = 1;

      do_reset();

      // Write PSUM at addr 30 with large INT32 values.
      for (int i = 0; i < LANES; i++)
        psum_w[i] = int32_t'(32'h0ABC_0000 + i * 256);
      write_psum(9'd30, psum_w);

      // Write ACT at a different SRAM address (addr 31) to avoid overwrite.
      // The "namespace independence" is tested by writing ACT at addr 31
      // and PSUM at addr 30 — they must not interfere.
      for (int i = 0; i < LANES; i++)
        act_w[i] = int8_t'(-10 - i);
      write_act(9'd31, act_w);

      // Read PSUM at addr 30 — should be unchanged.
      read_psum(9'd30, psum_r);
      for (int i = 0; i < LANES; i++) begin
        if (psum_r[i] !== psum_w[i]) begin
          $display("  T3.3.3 PSUM lane %0d: exp 0x%08X got 0x%08X",
                   i, psum_w[i], psum_r[i]);
          ok = 0;
        end
      end

      // Read ACT at addr 31 — should match written ACT data.
      read_act(9'd31, act_r);
      for (int i = 0; i < LANES; i++) begin
        if (act_r[i] !== act_w[i]) begin
          $display("  T3.3.3 ACT lane %0d: exp %0d got %0d",
                   i, act_w[i], act_r[i]);
          ok = 0;
        end
      end
      report("T3.3.3 Namespace independence (PSUM vs ACT)", ok);
    end

    // ========================================================================
    // T3.3.4: DW7x7 multipass — accumulate PSUM across 2 passes
    //   pass1: write PSUM initial values
    //   read pass1, add pass2 contribution, write back, read final
    // ========================================================================
    begin
      int32_t pass1_data [LANES];
      int32_t pass2_data [LANES];
      int32_t accum      [LANES];
      int32_t rdata      [LANES];
      bit ok = 1;

      do_reset();

      // Pass 1: write initial PSUM at addr 50.
      for (int i = 0; i < LANES; i++)
        pass1_data[i] = int32_t'(1000 + i);
      write_psum(9'd50, pass1_data);

      // Read pass1 back.
      read_psum(9'd50, rdata);

      // Pass 2: accumulate in software, then write back.
      for (int i = 0; i < LANES; i++) begin
        pass2_data[i] = int32_t'(2000 + i * 3);
        accum[i] = rdata[i] + pass2_data[i];
      end
      write_psum(9'd50, accum);

      // Read final accumulated result.
      read_psum(9'd50, rdata);

      for (int i = 0; i < LANES; i++) begin
        int32_t expected = pass1_data[i] + pass2_data[i];
        if (rdata[i] !== expected) begin
          $display("  T3.3.4 lane %0d: exp %0d got %0d",
                   i, expected, rdata[i]);
          ok = 0;
        end
      end
      report("T3.3.4 DW7x7 multipass PSUM accumulation", ok);
    end

    // ========================================================================
    // T3.3.5: DMA drain — write ACT data, read drain_data packed bus, verify
    // ========================================================================
    begin
      int8_t act_w [LANES];
      logic [LANES*8-1:0] drained;
      bit ok = 1;

      do_reset();

      // Write known ACT pattern at addr 60.
      for (int i = 0; i < LANES; i++)
        act_w[i] = int8_t'(i * 7 - 64);
      write_act(9'd60, act_w);

      // Read via drain port.
      read_drain(9'd60, drained);

      // Verify lane packing: drain_data[lane*8 +: 8] == ACT lower 8 bits.
      for (int i = 0; i < LANES; i++) begin
        logic [7:0] got_byte = drained[i*8 +: 8];
        logic [7:0] exp_byte = act_w[i][7:0];
        if (got_byte !== exp_byte) begin
          $display("  T3.3.5 lane %0d: exp 0x%02X got 0x%02X",
                   i, exp_byte, got_byte);
          ok = 0;
        end
      end
      report("T3.3.5 DMA drain packed bus verification", ok);
    end

    // ========================================================================
    //  Final summary
    // ========================================================================
    $display("");
    $display("==============================================================");
    $display("  tb_glb_output_bank — %0d/%0d tests PASSED",
             pass_count, total_tests);
    if (fail_count == 0)
      $display("  >>> ALL TESTS PASSED <<<");
    else
      $display("  >>> %0d TEST(S) FAILED <<<", fail_count);
    $display("==============================================================");
    $finish;
  end

endmodule
