// ============================================================================
// Module : csr_register_bank
// Project: YOLOv10n INT8 Accelerator — V4-VC707
//
// CPU-accessible control/status registers via AXI-Lite.
// Wraps axi_lite_slave and implements the CSR address map from csr_pkg.
// ============================================================================
`timescale 1ns / 1ps

module csr_register_bank
  import accel_pkg::*;
  import csr_pkg::*;
#(
  parameter int ADDR_W = 12,
  parameter int DATA_W = 32
)(
  input  logic              clk,
  input  logic              rst_n,

  // ═══════════ AXI4-LITE SLAVE (from CPU / PCIe bridge) ═══════════
  input  logic              s_awvalid,
  output logic              s_awready,
  input  logic [ADDR_W-1:0] s_awaddr,
  input  logic              s_wvalid,
  output logic              s_wready,
  input  logic [DATA_W-1:0] s_wdata,
  input  logic [DATA_W/8-1:0] s_wstrb,
  output logic              s_bvalid,
  input  logic              s_bready,
  output logic [1:0]        s_bresp,
  input  logic              s_arvalid,
  output logic              s_arready,
  input  logic [ADDR_W-1:0] s_araddr,
  output logic              s_rvalid,
  input  logic              s_rready,
  output logic [DATA_W-1:0] s_rdata,
  output logic [1:0]        s_rresp,

  // ═══════════ CONTROL OUTPUTS ═══════════
  output logic              ctrl_start,
  output logic              ctrl_soft_reset,
  output logic              ctrl_irq_clear,
  output logic [63:0]       net_desc_addr,
  output logic [7:0]        layer_start,
  output logic [7:0]        layer_end,
  output logic              irq_mask_en,

  // ═══════════ STATUS INPUTS ═══════════
  input  logic              stat_busy,
  input  logic              stat_done,
  input  logic              stat_error,
  input  logic [31:0]       perf_cycles,
  input  logic [31:0]       perf_stalls,
  input  logic [31:0]       perf_tiles,

  // ═══════════ IRQ OUTPUT ═══════════
  output logic              irq
);

  // ─── AXI-Lite slave instance ───
  logic              reg_wr_en, reg_rd_en;
  logic [ADDR_W-1:0] reg_wr_addr, reg_rd_addr;
  logic [DATA_W-1:0] reg_wr_data, reg_rd_data;

  axi_lite_slave #(.ADDR_W(ADDR_W), .DATA_W(DATA_W)) u_axil (
    .clk(clk), .rst_n(rst_n),
    .s_awvalid(s_awvalid), .s_awready(s_awready), .s_awaddr(s_awaddr),
    .s_wvalid(s_wvalid), .s_wready(s_wready), .s_wdata(s_wdata), .s_wstrb(s_wstrb),
    .s_bvalid(s_bvalid), .s_bready(s_bready), .s_bresp(s_bresp),
    .s_arvalid(s_arvalid), .s_arready(s_arready), .s_araddr(s_araddr),
    .s_rvalid(s_rvalid), .s_rready(s_rready), .s_rdata(s_rdata), .s_rresp(s_rresp),
    .reg_wr_en(reg_wr_en), .reg_wr_addr(reg_wr_addr), .reg_wr_data(reg_wr_data),
    .reg_rd_en(reg_rd_en), .reg_rd_addr(reg_rd_addr), .reg_rd_data(reg_rd_data)
  );

  // ─── Register storage ───
  logic [DATA_W-1:0] ctrl_reg;
  logic [DATA_W-1:0] net_desc_lo, net_desc_hi;
  logic [DATA_W-1:0] layer_start_reg, layer_end_reg;
  logic [DATA_W-1:0] irq_mask_reg;

  assign net_desc_addr = {net_desc_hi, net_desc_lo};
  assign layer_start   = layer_start_reg[7:0];
  assign layer_end     = layer_end_reg[7:0];
  assign irq_mask_en   = irq_mask_reg[0];

  // ─── Control pulse generation ───
  logic ctrl_start_prev;
  assign ctrl_start      = ctrl_reg[CTRL_BIT_START] && !ctrl_start_prev;
  assign ctrl_soft_reset = ctrl_reg[CTRL_BIT_SOFT_RST];
  assign ctrl_irq_clear  = ctrl_reg[CTRL_BIT_IRQ_CLEAR];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      ctrl_start_prev <= 1'b0;
    else
      ctrl_start_prev <= ctrl_reg[CTRL_BIT_START];
  end

  // ─── Write decode ───
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ctrl_reg       <= 32'd0;
      net_desc_lo    <= 32'd0;
      net_desc_hi    <= 32'd0;
      layer_start_reg <= 32'd0;
      layer_end_reg  <= 32'd22;
      irq_mask_reg   <= 32'd0;
    end else begin
      // Auto-clear single-pulse bits
      ctrl_reg[CTRL_BIT_START]     <= 1'b0;
      ctrl_reg[CTRL_BIT_IRQ_CLEAR] <= 1'b0;

      if (reg_wr_en) begin
        case (reg_wr_addr)
          CSR_CTRL:        ctrl_reg       <= reg_wr_data;
          CSR_NET_DESC_LO: net_desc_lo    <= reg_wr_data;
          CSR_NET_DESC_HI: net_desc_hi    <= reg_wr_data;
          CSR_LAYER_START: layer_start_reg <= reg_wr_data;
          CSR_LAYER_END:   layer_end_reg  <= reg_wr_data;
          CSR_IRQ_MASK:    irq_mask_reg   <= reg_wr_data;
          default: ;
        endcase
      end
    end
  end

  // ─── Read decode ───
  always_comb begin
    case (reg_rd_addr)
      CSR_CTRL:        reg_rd_data = ctrl_reg;
      CSR_STATUS:      reg_rd_data = {29'd0, stat_error, stat_done, stat_busy};
      CSR_NET_DESC_LO: reg_rd_data = net_desc_lo;
      CSR_NET_DESC_HI: reg_rd_data = net_desc_hi;
      CSR_LAYER_START: reg_rd_data = layer_start_reg;
      CSR_LAYER_END:   reg_rd_data = layer_end_reg;
      CSR_PERF_CYCLES: reg_rd_data = perf_cycles;
      CSR_PERF_STALLS: reg_rd_data = perf_stalls;
      CSR_PERF_TILES:  reg_rd_data = perf_tiles;
      CSR_IRQ_MASK:    reg_rd_data = irq_mask_reg;
      CSR_VERSION:     reg_rd_data = IP_VERSION;
      CSR_CONFIG:      reg_rd_data = IP_CONFIG;
      default:         reg_rd_data = 32'hDEAD_BEEF;
    endcase
  end

  // ─── IRQ generation ───
  logic irq_pending;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      irq_pending <= 1'b0;
    else if (ctrl_irq_clear)
      irq_pending <= 1'b0;
    else if (stat_done)
      irq_pending <= 1'b1;
  end
  assign irq = irq_pending && irq_mask_en;

endmodule
