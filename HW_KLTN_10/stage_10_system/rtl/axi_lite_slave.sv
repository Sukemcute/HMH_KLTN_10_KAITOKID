// ============================================================================
// Module : axi_lite_slave
// Project: YOLOv10n INT8 Accelerator — V4-VC707
//
// AXI4-Lite slave protocol handler. Translates AXI-Lite transactions into
// simple register read/write strobes for csr_register_bank.
// ============================================================================
`timescale 1ns / 1ps

module axi_lite_slave #(
  parameter int ADDR_W = 12,
  parameter int DATA_W = 32
)(
  input  logic              clk,
  input  logic              rst_n,

  // ═══════════ AXI4-LITE SLAVE ═══════════
  // Write address
  input  logic              s_awvalid,
  output logic              s_awready,
  input  logic [ADDR_W-1:0] s_awaddr,

  // Write data
  input  logic              s_wvalid,
  output logic              s_wready,
  input  logic [DATA_W-1:0] s_wdata,
  input  logic [DATA_W/8-1:0] s_wstrb,

  // Write response
  output logic              s_bvalid,
  input  logic              s_bready,
  output logic [1:0]        s_bresp,

  // Read address
  input  logic              s_arvalid,
  output logic              s_arready,
  input  logic [ADDR_W-1:0] s_araddr,

  // Read data
  output logic              s_rvalid,
  input  logic              s_rready,
  output logic [DATA_W-1:0] s_rdata,
  output logic [1:0]        s_rresp,

  // ═══════════ REGISTER INTERFACE ═══════════
  output logic              reg_wr_en,
  output logic [ADDR_W-1:0] reg_wr_addr,
  output logic [DATA_W-1:0] reg_wr_data,

  output logic              reg_rd_en,
  output logic [ADDR_W-1:0] reg_rd_addr,
  input  logic [DATA_W-1:0] reg_rd_data
);

  assign s_bresp = 2'b00;  // OKAY
  assign s_rresp = 2'b00;

  // ─── WRITE CHANNEL ───
  typedef enum logic [1:0] { W_IDLE, W_DATA, W_RESP } wr_state_e;
  wr_state_e wr_st;

  logic [ADDR_W-1:0] wr_addr_r;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wr_st     <= W_IDLE;
      s_awready <= 1'b1;
      s_wready  <= 1'b0;
      s_bvalid  <= 1'b0;
      reg_wr_en <= 1'b0;
    end else begin
      reg_wr_en <= 1'b0;

      case (wr_st)
        W_IDLE: begin
          s_awready <= 1'b1;
          s_wready  <= 1'b0;
          s_bvalid  <= 1'b0;
          if (s_awvalid && s_awready) begin
            wr_addr_r <= s_awaddr;
            s_awready <= 1'b0;
            s_wready  <= 1'b1;
            wr_st     <= W_DATA;
          end
        end

        W_DATA: begin
          if (s_wvalid && s_wready) begin
            reg_wr_en   <= 1'b1;
            reg_wr_addr <= wr_addr_r;
            reg_wr_data <= s_wdata;
            s_wready    <= 1'b0;
            s_bvalid    <= 1'b1;
            wr_st       <= W_RESP;
          end
        end

        W_RESP: begin
          if (s_bready) begin
            s_bvalid <= 1'b0;
            wr_st    <= W_IDLE;
          end
        end

        default: wr_st <= W_IDLE;
      endcase
    end
  end

  // ─── READ CHANNEL ───
  typedef enum logic [1:0] { R_IDLE, R_READ, R_RESP } rd_state_e;
  rd_state_e rd_st;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rd_st     <= R_IDLE;
      s_arready <= 1'b1;
      s_rvalid  <= 1'b0;
      reg_rd_en <= 1'b0;
    end else begin
      reg_rd_en <= 1'b0;

      case (rd_st)
        R_IDLE: begin
          s_arready <= 1'b1;
          s_rvalid  <= 1'b0;
          if (s_arvalid && s_arready) begin
            reg_rd_en   <= 1'b1;
            reg_rd_addr <= s_araddr;
            s_arready   <= 1'b0;
            rd_st       <= R_READ;
          end
        end

        R_READ: begin
          // 1-cycle register read latency
          s_rdata  <= reg_rd_data;
          s_rvalid <= 1'b1;
          rd_st    <= R_RESP;
        end

        R_RESP: begin
          if (s_rready) begin
            s_rvalid <= 1'b0;
            rd_st    <= R_IDLE;
          end
        end

        default: rd_st <= R_IDLE;
      endcase
    end
  end

endmodule
