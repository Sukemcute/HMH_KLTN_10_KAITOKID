// ============================================================================
// Testbench : tb_glb_weight_bank
// Project   : YOLOv10n INT8 Accelerator — V4-VC707
// Description:
//   Verification of glb_weight_bank 4-read-port weight SRAM.
//   3 tests:
//     T3.2.1  Write broadcast -> read from all 4 ports -> same data
//     T3.2.2  Different addresses per port: verify independent addressing
//     T3.2.3  Simultaneous read + write -> no interference
// ============================================================================
`timescale 1ns / 1ps

module tb_glb_weight_bank;

  import accel_pkg::*;

  // --------------------------------------------------------------------------
  //  Parameters
  // --------------------------------------------------------------------------
  localparam int LANES        = accel_pkg::LANES;           // 20
  localparam int DEPTH        = accel_pkg::GLB_WEIGHT_DEPTH; // 1024
  localparam int N_READ_PORTS = accel_pkg::WEIGHT_READ_PORTS; // 4
  localparam int AW           = $clog2(DEPTH);               // 10

  // Clock: 4 ns period (250 MHz)
  localparam real CLK_PERIOD = 4.0;

  // --------------------------------------------------------------------------
  //  DUT signals
  // --------------------------------------------------------------------------
  logic              clk;
  logic              rst_n;

  logic [AW-1:0]     rd_addr  [N_READ_PORTS];
  int8_t             rd_data  [N_READ_PORTS][LANES];

  logic [AW-1:0]     wr_addr;
  int8_t             wr_data  [LANES];
  logic              wr_en;

  // --------------------------------------------------------------------------
  //  DUT instantiation
  // --------------------------------------------------------------------------
  glb_weight_bank #(
    .LANES        (LANES),
    .DEPTH        (DEPTH),
    .N_READ_PORTS (N_READ_PORTS)
  ) u_dut (
    .clk     (clk),
    .rst_n   (rst_n),
    .rd_addr (rd_addr),
    .rd_data (rd_data),
    .wr_addr (wr_addr),
    .wr_data (wr_data),
    .wr_en   (wr_en)
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
    rst_n  <= 1'b0;
    wr_en  <= 1'b0;
    wr_addr <= '0;
    for (int i = 0; i < LANES; i++) wr_data[i] <= 8'sd0;
    for (int p = 0; p < N_READ_PORTS; p++) rd_addr[p] <= '0;
    repeat (4) @(posedge clk);
    rst_n <= 1'b1;
    @(posedge clk);
  endtask

  task automatic write_word(input logic [AW-1:0] addr, input int8_t data [LANES]);
    @(posedge clk);
    wr_addr <= addr;
    wr_en   <= 1'b1;
    for (int i = 0; i < LANES; i++) wr_data[i] <= data[i];
    @(posedge clk);
    wr_en <= 1'b0;
  endtask

  task automatic read_all_ports(input logic [AW-1:0] addrs [N_READ_PORTS],
                                output int8_t data [N_READ_PORTS][LANES]);
    @(posedge clk);
    for (int p = 0; p < N_READ_PORTS; p++) rd_addr[p] <= addrs[p];
    @(posedge clk); // 1-cycle registered latency
    for (int p = 0; p < N_READ_PORTS; p++)
      for (int i = 0; i < LANES; i++)
        data[p][i] = rd_data[p][i];
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
    // T3.2.1: Write broadcast -> read from all 4 ports -> all return same data
    // ========================================================================
    begin
      int8_t wdata [LANES];
      logic [AW-1:0] addrs [N_READ_PORTS];
      int8_t rdata [N_READ_PORTS][LANES];
      bit ok = 1;

      // Write a known pattern at address 55.
      for (int i = 0; i < LANES; i++) wdata[i] = int8_t'(i + 20);
      write_word(10'd55, wdata);

      // All 4 ports read from the same address 55.
      for (int p = 0; p < N_READ_PORTS; p++) addrs[p] = 10'd55;
      read_all_ports(addrs, rdata);

      for (int p = 0; p < N_READ_PORTS; p++) begin
        for (int i = 0; i < LANES; i++) begin
          if (rdata[p][i] !== wdata[i]) begin
            $display("  T3.2.1 port %0d lane %0d: exp %0d got %0d",
                     p, i, wdata[i], rdata[p][i]);
            ok = 0;
          end
        end
      end
      report("T3.2.1 Write broadcast -> all ports read same data", ok);
    end

    // ========================================================================
    // T3.2.2: Different addresses per port
    // ========================================================================
    begin
      int8_t wdata [4][LANES]; // 4 patterns for 4 addresses
      logic [AW-1:0] addrs [N_READ_PORTS];
      int8_t rdata [N_READ_PORTS][LANES];
      bit ok = 1;

      do_reset();

      // Write distinct patterns at addresses 0, 10, 20, 30.
      for (int a = 0; a < 4; a++) begin
        int8_t d [LANES];
        for (int i = 0; i < LANES; i++) begin
          d[i] = int8_t'((a * 40) + i);
          wdata[a][i] = d[i];
        end
        write_word(AW'(a * 10), d);
      end

      // Read: port0=addr0, port1=addr10, port2=addr20, port3=addr30.
      addrs[0] = 10'd0;
      addrs[1] = 10'd10;
      addrs[2] = 10'd20;
      addrs[3] = 10'd30;
      read_all_ports(addrs, rdata);

      for (int p = 0; p < N_READ_PORTS; p++) begin
        for (int i = 0; i < LANES; i++) begin
          if (rdata[p][i] !== wdata[p][i]) begin
            $display("  T3.2.2 port %0d lane %0d: exp %0d got %0d",
                     p, i, wdata[p][i], rdata[p][i]);
            ok = 0;
          end
        end
      end
      report("T3.2.2 Different addresses per port", ok);
    end

    // ========================================================================
    // T3.2.3: Simultaneous read + write -> no interference
    // ========================================================================
    begin
      int8_t wdata_pre [LANES];
      int8_t wdata_new [LANES];
      logic [AW-1:0] addrs [N_READ_PORTS];
      int8_t rdata [N_READ_PORTS][LANES];
      bit ok = 1;

      do_reset();

      // Pre-load address 100 with a known pattern.
      for (int i = 0; i < LANES; i++) wdata_pre[i] = int8_t'(i + 80);
      write_word(10'd100, wdata_pre);

      // Prepare a new pattern for address 200.
      for (int i = 0; i < LANES; i++) wdata_new[i] = int8_t'(i + 120);

      // Simultaneously: write addr 200 + read addr 100 from all ports.
      @(posedge clk);
      wr_addr <= 10'd200;
      wr_en   <= 1'b1;
      for (int i = 0; i < LANES; i++) wr_data[i] <= wdata_new[i];
      for (int p = 0; p < N_READ_PORTS; p++) rd_addr[p] <= 10'd100;
      @(posedge clk);
      wr_en <= 1'b0;

      // Wait for registered read output.
      @(posedge clk);

      // Verify read data from addr 100 is undisturbed.
      for (int p = 0; p < N_READ_PORTS; p++) begin
        for (int i = 0; i < LANES; i++) begin
          if (rd_data[p][i] !== wdata_pre[i]) begin
            $display("  T3.2.3 read port %0d lane %0d: exp %0d got %0d",
                     p, i, wdata_pre[i], rd_data[p][i]);
            ok = 0;
          end
        end
      end

      // Verify the written data at addr 200 is correct.
      for (int p = 0; p < N_READ_PORTS; p++) addrs[p] = 10'd200;
      read_all_ports(addrs, rdata);
      for (int p = 0; p < N_READ_PORTS; p++) begin
        for (int i = 0; i < LANES; i++) begin
          if (rdata[p][i] !== wdata_new[i]) begin
            $display("  T3.2.3 verify-wr port %0d lane %0d: exp %0d got %0d",
                     p, i, wdata_new[i], rdata[p][i]);
            ok = 0;
          end
        end
      end
      report("T3.2.3 Simultaneous read+write no interference", ok);
    end

    // ========================================================================
    //  Final summary
    // ========================================================================
    $display("");
    $display("==============================================================");
    $display("  tb_glb_weight_bank — %0d/%0d tests PASSED",
             pass_count, total_tests);
    if (fail_count == 0)
      $display("  >>> ALL TESTS PASSED <<<");
    else
      $display("  >>> %0d TEST(S) FAILED <<<", fail_count);
    $display("==============================================================");
    $finish;
  end

endmodule
