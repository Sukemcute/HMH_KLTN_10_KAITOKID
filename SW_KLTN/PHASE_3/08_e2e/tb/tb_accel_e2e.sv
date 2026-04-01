`timescale 1ns/1ps
// End-to-end testbench stub: full L0→L22 inference test.
// To be populated with golden Python data for bit-exact verification.
module tb_accel_e2e;

  logic clk, rst_n;

  // AXI-Lite
  logic [11:0] s_axil_awaddr, s_axil_araddr;
  logic s_axil_awvalid, s_axil_awready;
  logic [31:0] s_axil_wdata, s_axil_rdata;
  logic s_axil_wvalid, s_axil_wready;
  logic [1:0] s_axil_bresp;
  logic s_axil_bvalid, s_axil_bready;
  logic s_axil_arvalid, s_axil_arready;
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

  // DDR3 memory model (simplified: 1MB address space)
  localparam int MEM_DEPTH = 1024*1024/32; // 32B per entry
  logic [255:0] ddr_mem [MEM_DEPTH];

  // AXI slave
  assign m_axi_arready = 1'b1;
  assign m_axi_awready = 1'b1;
  assign m_axi_wready  = 1'b1;

  logic [7:0] rd_burst_rem;
  logic [39:0] rd_burst_addr;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      m_axi_rvalid  <= 1'b0;
      m_axi_rlast   <= 1'b0;
      m_axi_rdata   <= '0;
      m_axi_bvalid  <= 1'b0;
      m_axi_bresp   <= 2'b00;
      rd_burst_rem  <= '0;
      rd_burst_addr <= '0;
    end else begin
      // Read channel
      if (m_axi_arvalid && m_axi_arready) begin
        rd_burst_addr <= m_axi_araddr;
        rd_burst_rem  <= m_axi_arlen;
        m_axi_rvalid  <= 1'b1;
        m_axi_rdata   <= ddr_mem[m_axi_araddr[24:5]];
        m_axi_rlast   <= (m_axi_arlen == 0);
      end else if (m_axi_rvalid && m_axi_rready) begin
        if (m_axi_rlast) begin
          m_axi_rvalid <= 1'b0;
          m_axi_rlast  <= 1'b0;
        end else begin
          rd_burst_addr <= rd_burst_addr + 32;
          rd_burst_rem  <= rd_burst_rem - 1;
          m_axi_rdata   <= ddr_mem[(rd_burst_addr + 32) >> 5];
          m_axi_rlast   <= (rd_burst_rem == 1);
        end
      end
      // Write channel
      if (m_axi_wvalid && m_axi_wready)
        ddr_mem[m_axi_awaddr[24:5]] <= m_axi_wdata;
      if (m_axi_wvalid && m_axi_wlast)
        m_axi_bvalid <= 1'b1;
      else if (m_axi_bvalid && m_axi_bready)
        m_axi_bvalid <= 1'b0;
    end
  end

  initial begin
    $display("╔══════════════════════════════════════════════════════════╗");
    $display("║ E2E TESTBENCH: accel_top (L0→L22 stub)                 ║");
    $display("║ Populate ddr_mem with golden data for full test.       ║");
    $display("╚══════════════════════════════════════════════════════════╝");

    rst_n = 0;
    s_axil_awaddr = 0; s_axil_awvalid = 0;
    s_axil_wdata = 0; s_axil_wvalid = 0; s_axil_bready = 0;
    s_axil_araddr = 0; s_axil_arvalid = 0; s_axil_rready = 0;

    // Init DDR memory to 0
    for (int i = 0; i < MEM_DEPTH; i++) ddr_mem[i] = '0;

    @(negedge clk); @(negedge clk);
    rst_n = 1;
    repeat(5) @(negedge clk);

    $display("  [INFO] E2E stub initialized. Load golden data via $readmemh.");
    $display("  [INFO] Then write NET_DESC_BASE to CSR and assert START.");

    repeat(100) @(posedge clk);

    $display("════════════════════════════════════════════════════════");
    $display("★ E2E STUB COMPLETE — populate golden data for full test ★");
    $display("════════════════════════════════════════════════════════");
    $finish;
  end

endmodule
