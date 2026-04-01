`timescale 1ns/1ps
// AXI4 DMA master: handles bulk read/write between internal GLBs and DDR3.
// Splits large transfers into AXI bursts (max 16 beats x 32B = 512B).
module tensor_dma #(
  parameter int AXI_DATA_W = 256,
  parameter int AXI_ADDR_W = 40
)(
  input  logic                  clk,
  input  logic                  rst_n,

  // AXI4 Master Read Channel
  output logic [AXI_ADDR_W-1:0] m_axi_araddr,
  output logic [7:0]            m_axi_arlen,
  output logic [2:0]            m_axi_arsize,
  output logic [1:0]            m_axi_arburst,
  output logic                  m_axi_arvalid,
  input  logic                  m_axi_arready,
  input  logic [AXI_DATA_W-1:0] m_axi_rdata,
  input  logic [1:0]            m_axi_rresp,
  input  logic                  m_axi_rlast,
  input  logic                  m_axi_rvalid,
  output logic                  m_axi_rready,

  // AXI4 Master Write Channel
  output logic [AXI_ADDR_W-1:0] m_axi_awaddr,
  output logic [7:0]            m_axi_awlen,
  output logic [2:0]            m_axi_awsize,
  output logic [1:0]            m_axi_awburst,
  output logic                  m_axi_awvalid,
  input  logic                  m_axi_awready,
  output logic [AXI_DATA_W-1:0] m_axi_wdata,
  output logic [AXI_DATA_W/8-1:0] m_axi_wstrb,
  output logic                  m_axi_wlast,
  output logic                  m_axi_wvalid,
  input  logic                  m_axi_wready,
  input  logic [1:0]            m_axi_bresp,
  input  logic                  m_axi_bvalid,
  output logic                  m_axi_bready,

  // Internal read request (from subclusters)
  input  logic                  rd_req,
  input  logic [AXI_ADDR_W-1:0] rd_addr,
  input  logic [15:0]           rd_byte_len,
  output logic                  rd_data_valid,
  output logic [AXI_DATA_W-1:0] rd_data,
  output logic                  rd_done,

  // Internal write request
  input  logic                  wr_req,
  input  logic [AXI_ADDR_W-1:0] wr_addr,
  input  logic [15:0]           wr_byte_len,
  input  logic [AXI_DATA_W-1:0] wr_data_in,
  input  logic                  wr_data_valid_in,
  output logic                  wr_done,
  // One-cycle strobes when a 256b W beat completes (wvalid & wready in WR_DATA)
  output logic                  wr_beat_accept
);
  localparam int BYTES_PER_BEAT = AXI_DATA_W / 8;  // 32
  localparam int MAX_BURST_BEATS = 16;

  assign m_axi_arsize  = 3'd5;  // 32 bytes
  assign m_axi_arburst = 2'b01; // INCR
  assign m_axi_awsize  = 3'd5;
  assign m_axi_awburst = 2'b01;
  assign m_axi_wstrb   = {(AXI_DATA_W/8){1'b1}};
  assign m_axi_bready  = 1'b1;

  // ───── Read FSM ─────
  typedef enum logic [1:0] { RD_IDLE, RD_AR, RD_DATA, RD_FINISH } rd_state_e;
  rd_state_e rd_state;

  logic [AXI_ADDR_W-1:0] rd_cur_addr;
  logic [15:0]            rd_remain_bytes;
  logic [7:0]             rd_burst_len;

  // Ceil(remain/32): floor division yields 0 when 0<bytes<32 after a burst → (beats-1) underflows.
  wire [15:0] rd_remain_beats = (rd_remain_bytes == 16'd0) ? 16'd0
      : 16'((rd_remain_bytes + 16'(BYTES_PER_BEAT) - 16'd1) / 16'(BYTES_PER_BEAT));
  wire [7:0]  rd_this_len     = (rd_remain_beats > MAX_BURST_BEATS) ?
                                 8'(MAX_BURST_BEATS - 1) : 8'(rd_remain_beats - 1);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rd_state        <= RD_IDLE;
      rd_cur_addr     <= '0;
      rd_remain_bytes <= '0;
      rd_burst_len    <= '0;
    end else begin
      case (rd_state)
        RD_IDLE: if (rd_req) begin
          rd_cur_addr     <= rd_addr;
          rd_remain_bytes <= rd_byte_len;
          rd_state        <= RD_AR;
        end
        RD_AR: if (m_axi_arready) begin
          rd_burst_len <= rd_this_len;
          rd_state     <= RD_DATA;
        end
        RD_DATA: if (m_axi_rvalid && m_axi_rlast) begin
          automatic logic [15:0] rd_xfer = (16'(rd_burst_len) + 16'd1) * 16'(BYTES_PER_BEAT);
          rd_cur_addr <= rd_cur_addr + rd_xfer;
          if (rd_remain_bytes <= rd_xfer) begin
            rd_remain_bytes <= 16'd0;
            rd_state        <= RD_FINISH;
          end else begin
            rd_remain_bytes <= rd_remain_bytes - rd_xfer;
            rd_state        <= RD_AR;
          end
        end
        RD_FINISH: rd_state <= RD_IDLE;
      endcase
    end
  end

  assign m_axi_araddr  = rd_cur_addr;
  assign m_axi_arlen   = rd_this_len;
  assign m_axi_arvalid = (rd_state == RD_AR);
  assign m_axi_rready  = (rd_state == RD_DATA);
  assign rd_data       = m_axi_rdata;
  assign rd_data_valid = (rd_state == RD_DATA) && m_axi_rvalid;
  assign rd_done       = (rd_state == RD_FINISH);

  // ───── Write FSM ─────
  typedef enum logic [1:0] { WR_IDLE, WR_AW, WR_DATA, WR_FINISH } wr_state_e;
  wr_state_e wr_state;

  // Supercluster gates wr_req with dma_busy; that can go high the cycle after wr_req was 0
  // at the clock edge. Arm on any wr_req while idle so the start is not missed.
  logic wr_arm;

  logic [AXI_ADDR_W-1:0] wr_cur_addr;
  logic [15:0]            wr_remain_bytes;
  logic [7:0]             wr_burst_len;
  logic [7:0]             wr_beat_cnt;

  wire [15:0] wr_remain_beats = (wr_remain_bytes == 16'd0) ? 16'd0
      : 16'((wr_remain_bytes + 16'(BYTES_PER_BEAT) - 16'd1) / 16'(BYTES_PER_BEAT));
  wire [7:0]  wr_this_len     = (wr_remain_beats > MAX_BURST_BEATS) ?
                                 8'(MAX_BURST_BEATS - 1) : 8'(wr_remain_beats - 1);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wr_state        <= WR_IDLE;
      wr_cur_addr     <= '0;
      wr_remain_bytes <= '0;
      wr_burst_len    <= '0;
      wr_beat_cnt     <= '0;
      wr_arm          <= 1'b0;
    end else begin
      if (wr_state == WR_IDLE) begin
        if (wr_req)
          wr_arm <= 1'b1;
      end else
        wr_arm <= 1'b0;

      case (wr_state)
        WR_IDLE: if (wr_req || wr_arm) begin
          wr_cur_addr     <= wr_addr;
          wr_remain_bytes <= wr_byte_len;
          wr_state        <= WR_AW;
        end
        WR_AW: if (m_axi_awready) begin
          wr_burst_len <= wr_this_len;
          wr_beat_cnt  <= '0;
          wr_state     <= WR_DATA;
        end
        WR_DATA: if (m_axi_wready && wr_data_valid_in) begin
          wr_beat_cnt <= wr_beat_cnt + 1;
          if (wr_beat_cnt == wr_burst_len) begin
            automatic logic [15:0] wr_xfer = (16'(wr_burst_len) + 16'd1) * 16'(BYTES_PER_BEAT);
            wr_cur_addr <= wr_cur_addr + wr_xfer;
            if (wr_remain_bytes <= wr_xfer) begin
              wr_remain_bytes <= 16'd0;
              wr_state        <= WR_FINISH;
            end else begin
              wr_remain_bytes <= wr_remain_bytes - wr_xfer;
              wr_state        <= WR_AW;
            end
          end
        end
        WR_FINISH: if (m_axi_bvalid) wr_state <= WR_IDLE;
      endcase
    end
  end

  assign m_axi_awaddr  = wr_cur_addr;
  assign m_axi_awlen   = wr_this_len;
  assign m_axi_awvalid = (wr_state == WR_AW);
  assign m_axi_wdata   = wr_data_in;
  assign m_axi_wvalid  = (wr_state == WR_DATA) && wr_data_valid_in;
  assign m_axi_wlast   = (wr_state == WR_DATA) && (wr_beat_cnt == wr_burst_len);
  assign wr_done       = (wr_state == WR_FINISH) && m_axi_bvalid;
  assign wr_beat_accept = (wr_state == WR_DATA) && m_axi_wready && wr_data_valid_in;

endmodule
