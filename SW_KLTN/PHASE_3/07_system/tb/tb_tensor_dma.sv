`timescale 1ns/1ps
module tb_tensor_dma;
  localparam int AXI_DATA_W = 256;
  localparam int AXI_ADDR_W = 40;

  logic clk, rst_n;

  // AXI4 Master Read
  logic [AXI_ADDR_W-1:0] m_axi_araddr;
  logic [7:0] m_axi_arlen;
  logic [2:0] m_axi_arsize;
  logic [1:0] m_axi_arburst;
  logic m_axi_arvalid, m_axi_arready;
  logic [AXI_DATA_W-1:0] m_axi_rdata;
  logic [1:0] m_axi_rresp;
  logic m_axi_rlast, m_axi_rvalid, m_axi_rready;

  // AXI4 Master Write
  logic [AXI_ADDR_W-1:0] m_axi_awaddr;
  logic [7:0] m_axi_awlen;
  logic [2:0] m_axi_awsize;
  logic [1:0] m_axi_awburst;
  logic m_axi_awvalid, m_axi_awready;
  logic [AXI_DATA_W-1:0] m_axi_wdata;
  logic [AXI_DATA_W/8-1:0] m_axi_wstrb;
  logic m_axi_wlast, m_axi_wvalid, m_axi_wready;
  logic [1:0] m_axi_bresp;
  logic m_axi_bvalid, m_axi_bready;

  // Internal interface
  logic rd_req, rd_data_valid, rd_done;
  logic [AXI_ADDR_W-1:0] rd_addr;
  logic [15:0] rd_byte_len;
  logic [AXI_DATA_W-1:0] rd_data;
  logic wr_req, wr_data_valid_in, wr_done, wr_beat_accept;
  logic [AXI_ADDR_W-1:0] wr_addr;
  logic [15:0] wr_byte_len;
  logic [AXI_DATA_W-1:0] wr_data_in;

  tensor_dma #(.AXI_DATA_W(AXI_DATA_W), .AXI_ADDR_W(AXI_ADDR_W)) dut (.*);

  initial clk = 0;
  always #5 clk = ~clk;
  int err_cnt = 0;

  // AXI slave model
  logic [7:0] rd_beat_cnt;
  logic [7:0] rd_arlen_captured;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      m_axi_arready <= 1'b1;
      m_axi_rvalid  <= 1'b0;
      m_axi_rlast   <= 1'b0;
      m_axi_rdata   <= '0;
      m_axi_rresp   <= 2'b00;
      rd_beat_cnt   <= '0;
      rd_arlen_captured <= '0;
    end else begin
      m_axi_arready <= 1'b1;
      if (m_axi_arvalid && m_axi_arready) begin
        rd_arlen_captured <= m_axi_arlen;
        rd_beat_cnt <= '0;
        m_axi_rvalid <= 1'b1;
        m_axi_rdata <= {224'd0, 32'hCAFE_0000};
      end else if (m_axi_rvalid && m_axi_rready) begin
        rd_beat_cnt <= rd_beat_cnt + 1;
        m_axi_rdata <= {224'd0, 24'd0, rd_beat_cnt + 1};
        if (rd_beat_cnt >= rd_arlen_captured) begin
          m_axi_rlast <= 1'b1;
        end
        if (m_axi_rlast) begin
          m_axi_rvalid <= 1'b0;
          m_axi_rlast  <= 1'b0;
        end
      end
    end
  end

  // Write channel slave
  assign m_axi_awready = 1'b1;
  assign m_axi_wready  = 1'b1;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      m_axi_bvalid <= 1'b0;
      m_axi_bresp  <= 2'b00;
    end else begin
      if (m_axi_wvalid && m_axi_wlast)
        m_axi_bvalid <= 1'b1;
      else if (m_axi_bready && m_axi_bvalid)
        m_axi_bvalid <= 1'b0;
    end
  end

  initial begin
    $display("╔══════════════════════════════════════════════════════╗");
    $display("║ TESTBENCH: tensor_dma                               ║");
    $display("╚══════════════════════════════════════════════════════╝");
    rst_n = 0; rd_req = 0; wr_req = 0; wr_data_valid_in = 0;
    rd_addr = '0; rd_byte_len = '0; wr_addr = '0; wr_byte_len = '0; wr_data_in = '0;
    @(negedge clk); @(negedge clk); rst_n = 1; @(negedge clk);

    // TEST 1: Read 64 bytes (2 beats)
    $display("=== TEST 1: DMA Read 64 bytes ===");
    @(negedge clk);
    rd_req = 1; rd_addr = 40'h00_0000_1000; rd_byte_len = 16'd64;
    @(negedge clk); rd_req = 0;

    // Wait for rd_done
    repeat(20) begin
      @(posedge clk);
      if (rd_done) break;
    end
    #1;
    if (rd_done) $display("  TEST 1 PASSED (rd_done asserted)");
    else begin $display("  FAIL: rd_done not asserted"); err_cnt++; end

    @(negedge clk); @(negedge clk);

    // TEST 2: Write 32 bytes (1 beat)
    $display("=== TEST 2: DMA Write 32 bytes ===");
    @(negedge clk);
    wr_req = 1; wr_addr = 40'h00_0000_2000; wr_byte_len = 16'd32;
    wr_data_in = {224'd0, 32'h1234_5678};
    wr_data_valid_in = 1;
    @(negedge clk); wr_req = 0;

    repeat(20) begin
      @(posedge clk);
      if (wr_done) break;
    end
    #1;
    if (wr_done) $display("  TEST 2 PASSED (wr_done asserted)");
    else begin $display("  FAIL: wr_done not asserted"); err_cnt++; end
    wr_data_valid_in = 0;

    $display("════════════════════════════════════════════════════");
    if (err_cnt == 0) $display("★ ALL TESTS PASSED ★");
    else              $display("✗ TOTAL FAILURES: %0d", err_cnt);
    $display("════════════════════════════════════════════════════");
    $finish;
  end
endmodule
