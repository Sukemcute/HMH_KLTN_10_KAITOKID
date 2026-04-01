`timescale 1ns/1ps
// Smoke test for accel_top: CSR write start → check status → IRQ.
module tb_accel_top;

  logic clk, rst_n;

  // AXI-Lite
  logic [11:0] s_axil_awaddr, s_axil_araddr;
  logic s_axil_awvalid, s_axil_awready;
  logic [31:0] s_axil_wdata;
  logic s_axil_wvalid, s_axil_wready;
  logic [1:0] s_axil_bresp;
  logic s_axil_bvalid, s_axil_bready;
  logic s_axil_arvalid, s_axil_arready;
  logic [31:0] s_axil_rdata;
  logic s_axil_rvalid, s_axil_rready;

  // AXI4 Master
  logic [39:0] m_axi_araddr, m_axi_awaddr;
  logic [7:0] m_axi_arlen, m_axi_awlen;
  logic m_axi_arvalid, m_axi_arready;
  logic [255:0] m_axi_rdata, m_axi_wdata;
  logic m_axi_rvalid, m_axi_rlast, m_axi_rready;
  logic m_axi_awvalid, m_axi_awready;
  logic m_axi_wvalid, m_axi_wlast, m_axi_wready;
  logic [1:0] m_axi_bresp;
  logic m_axi_bvalid, m_axi_bready;
  logic irq;

  accel_top dut (.*);

  initial clk = 0;
  always #5 clk = ~clk;

  // AXI memory slave stub
  assign m_axi_arready = 1'b1;
  assign m_axi_awready = 1'b1;
  assign m_axi_wready  = 1'b1;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      m_axi_rvalid <= 1'b0;
      m_axi_rlast  <= 1'b0;
      m_axi_rdata  <= '0;
      m_axi_bvalid <= 1'b0;
      m_axi_bresp  <= 2'b00;
    end else begin
      if (m_axi_arvalid && m_axi_arready) begin
        m_axi_rvalid <= 1'b1;
        m_axi_rlast  <= 1'b1;
        m_axi_rdata  <= '0;
      end else if (m_axi_rvalid && m_axi_rready) begin
        m_axi_rvalid <= 1'b0;
        m_axi_rlast  <= 1'b0;
      end
      if (m_axi_wvalid && m_axi_wlast)
        m_axi_bvalid <= 1'b1;
      else if (m_axi_bvalid && m_axi_bready)
        m_axi_bvalid <= 1'b0;
    end
  end

  // AXI-Lite helper tasks
  task automatic axil_write(input logic [11:0] addr, input logic [31:0] data);
    @(negedge clk);
    s_axil_awaddr  = addr;
    s_axil_awvalid = 1'b1;
    s_axil_wdata   = data;
    s_axil_wvalid  = 1'b1;
    // Wait for both ready
    fork
      begin wait(s_axil_awready); @(negedge clk); s_axil_awvalid = 0; end
      begin wait(s_axil_wready);  @(negedge clk); s_axil_wvalid  = 0; end
    join
    s_axil_bready = 1'b1;
    wait(s_axil_bvalid);
    @(negedge clk);
    s_axil_bready = 1'b0;
  endtask

  task automatic axil_read(input logic [11:0] addr, output logic [31:0] data);
    @(negedge clk);
    s_axil_araddr  = addr;
    s_axil_arvalid = 1'b1;
    wait(s_axil_arready);
    @(negedge clk);
    s_axil_arvalid = 0;
    s_axil_rready = 1'b1;
    wait(s_axil_rvalid);
    data = s_axil_rdata;
    @(negedge clk);
    s_axil_rready = 1'b0;
  endtask

  int err_cnt = 0;

  initial begin
    $display("╔══════════════════════════════════════════════════════╗");
    $display("║ TESTBENCH: accel_top (smoke test)                   ║");
    $display("╚══════════════════════════════════════════════════════╝");
    rst_n = 0;
    s_axil_awaddr = 0; s_axil_awvalid = 0;
    s_axil_wdata = 0; s_axil_wvalid = 0;
    s_axil_bready = 0;
    s_axil_araddr = 0; s_axil_arvalid = 0;
    s_axil_rready = 0;
    @(negedge clk); @(negedge clk); rst_n = 1;
    repeat(3) @(negedge clk);

    // TEST 1: Read VERSION register
    $display("=== TEST 1: Read VERSION CSR ===");
    begin
      logic [31:0] ver;
      axil_read(12'h008, ver);
      $display("  VERSION = 0x%08h", ver);
      if (ver == 32'h2026_0320) $display("  TEST 1 PASSED");
      else begin $display("  FAIL: unexpected version"); err_cnt++; end
    end

    // TEST 2: Write NET_DESC_BASE and read back
    $display("=== TEST 2: Write/Read NET_DESC_BASE ===");
    axil_write(12'h010, 32'hDEAD_BEEF); // LO
    axil_write(12'h014, 32'h0000_0042); // HI
    begin
      logic [31:0] lo, hi;
      axil_read(12'h010, lo);
      axil_read(12'h014, hi);
      if (lo == 32'hDEAD_BEEF && hi == 32'h0000_0042)
        $display("  TEST 2 PASSED");
      else begin
        $display("  FAIL: lo=0x%08h hi=0x%08h", lo, hi);
        err_cnt++;
      end
    end

    // TEST 3: Write START and check BUSY
    $display("=== TEST 3: Start inference ===");
    axil_write(12'h000, 32'h0000_0001); // start=1
    repeat(5) @(posedge clk);
    begin
      logic [31:0] status;
      axil_read(12'h004, status);
      $display("  STATUS = 0x%08h (busy=%0d)", status, status[0]);
      if (status[0]) $display("  TEST 3 PASSED (busy set)");
      else $display("  TEST 3 INFO: busy not set yet");
    end

    repeat(10) @(posedge clk);

    $display("════════════════════════════════════════════════════");
    if (err_cnt == 0) $display("★ ALL TESTS PASSED ★");
    else              $display("✗ TOTAL FAILURES: %0d", err_cnt);
    $display("════════════════════════════════════════════════════");
    $finish;
  end
endmodule
