// ============================================================================
// Module : accel_top
// Project: YOLOv10n INT8 Accelerator — V4-VC707
//
// System top-level integrating:
//   - Clock/Reset infrastructure (clk_wiz_250, reset_sync)
//   - CSR register bank (AXI-Lite slave)
//   - Descriptor fetch engine
//   - Global scheduler
//   - 4 × SuperCluster wrappers
//   - Barrier manager
//   - AXI4 master mux → DDR3
//   - Performance counters
//
// Flow: CPU writes CSR_START → desc_fetch → scheduler → 4 SCs → IRQ
// ============================================================================
`timescale 1ns / 1ps

module accel_top
  import accel_pkg::*;
  import desc_pkg::*;
  import csr_pkg::*;
#(
  parameter int N_SC    = N_SUPER_CLUSTERS,
  parameter int ADDR_W  = AXI_ADDR_WIDTH,
  parameter int DATA_W  = AXI_DATA_WIDTH,
  parameter int AXIL_AW = 12,
  parameter int AXIL_DW = 32
)(
  input  logic              clk_in_200,
  input  logic              rst_n_async,

  // ═══════════ AXI4-LITE SLAVE (CPU control) ═══════════
  input  logic              s_awvalid,
  output logic              s_awready,
  input  logic [AXIL_AW-1:0] s_awaddr,
  input  logic              s_wvalid,
  output logic              s_wready,
  input  logic [AXIL_DW-1:0] s_wdata,
  input  logic [AXIL_DW/8-1:0] s_wstrb,
  output logic              s_bvalid,
  input  logic              s_bready,
  output logic [1:0]        s_bresp,
  input  logic              s_arvalid,
  output logic              s_arready,
  input  logic [AXIL_AW-1:0] s_araddr,
  output logic              s_rvalid,
  input  logic              s_rready,
  output logic [AXIL_DW-1:0] s_rdata,
  output logic [1:0]        s_rresp,

  // ═══════════ AXI4 MASTER (DDR3 controller) ═══════════
  output logic              ddr_ar_valid,
  input  logic              ddr_ar_ready,
  output logic [ADDR_W-1:0] ddr_ar_addr,
  output logic [7:0]        ddr_ar_len,
  output logic [2:0]        ddr_ar_size,
  output logic [1:0]        ddr_ar_burst,

  input  logic              ddr_r_valid,
  output logic              ddr_r_ready,
  input  logic [DATA_W-1:0] ddr_r_data,
  input  logic              ddr_r_last,

  output logic              ddr_aw_valid,
  input  logic              ddr_aw_ready,
  output logic [ADDR_W-1:0] ddr_aw_addr,
  output logic [7:0]        ddr_aw_len,
  output logic [2:0]        ddr_aw_size,
  output logic [1:0]        ddr_aw_burst,

  output logic              ddr_w_valid,
  input  logic              ddr_w_ready,
  output logic [DATA_W-1:0] ddr_w_data,
  output logic [DATA_W/8-1:0] ddr_w_strb,
  output logic              ddr_w_last,

  input  logic              ddr_b_valid,
  output logic              ddr_b_ready,

  // ═══════════ INTERRUPT ═══════════
  output logic              irq
);

  // ─── Clock & Reset ───
  logic clk, rst_n, pll_locked;

  clk_wiz_250 u_clk (
    .clk_in_200 (clk_in_200),
    .rst_n      (rst_n_async),
    .clk_out_250(clk),
    .locked     (pll_locked)
  );

  reset_sync u_rst (
    .clk        (clk),
    .rst_async_n(rst_n_async & pll_locked),
    .rst_sync_n (rst_n)
  );

  // ─── CSR ───
  logic        ctrl_start, ctrl_soft_reset, ctrl_irq_clear;
  logic [63:0] net_desc_addr;
  logic [7:0]  layer_start_csr, layer_end_csr;
  logic        irq_mask_en;
  logic        stat_busy, stat_done, stat_error;
  logic [31:0] perf_cycles, perf_stalls, perf_tiles;

  csr_register_bank #(.ADDR_W(AXIL_AW), .DATA_W(AXIL_DW)) u_csr (
    .clk(clk), .rst_n(rst_n),
    .s_awvalid(s_awvalid), .s_awready(s_awready), .s_awaddr(s_awaddr),
    .s_wvalid(s_wvalid), .s_wready(s_wready), .s_wdata(s_wdata), .s_wstrb(s_wstrb),
    .s_bvalid(s_bvalid), .s_bready(s_bready), .s_bresp(s_bresp),
    .s_arvalid(s_arvalid), .s_arready(s_arready), .s_araddr(s_araddr),
    .s_rvalid(s_rvalid), .s_rready(s_rready), .s_rdata(s_rdata), .s_rresp(s_rresp),
    .ctrl_start(ctrl_start), .ctrl_soft_reset(ctrl_soft_reset),
    .ctrl_irq_clear(ctrl_irq_clear),
    .net_desc_addr(net_desc_addr),
    .layer_start(layer_start_csr), .layer_end(layer_end_csr),
    .irq_mask_en(irq_mask_en),
    .stat_busy(stat_busy), .stat_done(stat_done), .stat_error(stat_error),
    .perf_cycles(perf_cycles), .perf_stalls(perf_stalls), .perf_tiles(perf_tiles),
    .irq(irq)
  );

  // ─── Performance Counters ───
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n || ctrl_soft_reset) begin
      perf_cycles <= 32'd0;
      perf_stalls <= 32'd0;
    end else if (stat_busy) begin
      perf_cycles <= perf_cycles + 32'd1;
    end
  end

  // ─── Descriptor Fetch Engine ───
  logic        df_busy, df_done;
  logic        df_desc_valid;
  layer_desc_t df_desc_layer;
  tile_desc_t  df_desc_tile;
  logic        df_desc_ready;

  // Desc fetch AXI read
  logic              df_ar_valid, df_ar_ready;
  logic [ADDR_W-1:0] df_ar_addr;
  logic [7:0]        df_ar_len;
  logic [2:0]        df_ar_size;
  logic [1:0]        df_ar_burst;
  logic              df_r_valid, df_r_ready;
  logic [DATA_W-1:0] df_r_data;
  logic              df_r_last;

  desc_fetch_engine u_desc_fetch (
    .clk(clk), .rst_n(rst_n),
    .start(ctrl_start),
    .net_desc_addr(net_desc_addr[ADDR_W-1:0]),
    .layer_start(layer_start_csr),
    .layer_end(layer_end_csr),
    .busy(df_busy), .done(df_done),
    .desc_valid(df_desc_valid),
    .desc_layer(df_desc_layer),
    .desc_tile(df_desc_tile),
    .desc_ready(df_desc_ready),
    .axi_ar_valid(df_ar_valid), .axi_ar_ready(df_ar_ready),
    .axi_ar_addr(df_ar_addr), .axi_ar_len(df_ar_len),
    .axi_ar_size(df_ar_size), .axi_ar_burst(df_ar_burst),
    .axi_r_valid(df_r_valid), .axi_r_ready(df_r_ready),
    .axi_r_data(df_r_data), .axi_r_last(df_r_last)
  );

  // ─── Global Scheduler ───
  logic        sched_enable;
  logic [15:0] total_dispatched;
  logic        all_dispatched;

  logic          sc_tile_valid_w [N_SC];
  layer_desc_t   sc_layer_desc_w [N_SC];
  tile_desc_t    sc_tile_desc_w  [N_SC];
  logic          sc_tile_ready_w [N_SC];
  logic          sc_idle_w       [N_SC];

  // Keep scheduler enabled for whole inference run (not only while fetch FSM busy).
  // If tied to df_busy, fetch drops busy before scheduler can see desc_valid=0 and
  // assert all_dispatched; zero-tile layers never assert desc_valid at all.
  assign sched_enable = stat_busy;

  global_scheduler #(.N_SC(N_SC)) u_scheduler (
    .clk(clk), .rst_n(rst_n), .enable(sched_enable),
    .desc_valid(df_desc_valid),
    .desc_layer(df_desc_layer),
    .desc_tile(df_desc_tile),
    .desc_ready(df_desc_ready),
    .sc_tile_valid(sc_tile_valid_w),
    .sc_layer_desc(sc_layer_desc_w),
    .sc_tile_desc(sc_tile_desc_w),
    .sc_tile_ready(sc_tile_ready_w),
    .total_tiles_dispatched(total_dispatched),
    .all_dispatched(all_dispatched),
    .sc_idle(sc_idle_w)
  );

  assign perf_tiles = {16'd0, total_dispatched};

  // ─── Barrier Manager ───
  logic [3:0] barrier_signal_all, barrier_request_all, barrier_grant_all;
  logic [3:0] barrier_pending;

  barrier_manager u_barrier (
    .clk(clk), .rst_n(rst_n),
    .soft_reset(ctrl_soft_reset),
    .barrier_signal(barrier_signal_all),
    .barrier_request(barrier_request_all),
    .barrier_grant(barrier_grant_all),
    .barrier_pending(barrier_pending)
  );

  // Aggregate barrier signals from all SCs
  always_comb begin
    barrier_signal_all  = 4'd0;
    barrier_request_all = 4'd0;
  end

  // ─── AXI4 Master Mux ───
  localparam int N_AXI_M = N_SC + 1;  // 4 SC + desc_fetch

  logic              mux_ar_valid [N_AXI_M];
  logic              mux_ar_ready [N_AXI_M];
  logic [ADDR_W-1:0] mux_ar_addr  [N_AXI_M];
  logic [7:0]        mux_ar_len   [N_AXI_M];
  logic [2:0]        mux_ar_size  [N_AXI_M];
  logic [1:0]        mux_ar_burst [N_AXI_M];
  logic              mux_r_valid  [N_AXI_M];
  logic              mux_r_ready  [N_AXI_M];
  logic [DATA_W-1:0] mux_r_data   [N_AXI_M];
  logic              mux_r_last   [N_AXI_M];

  logic              mux_aw_valid [N_AXI_M];
  logic              mux_aw_ready [N_AXI_M];
  logic [ADDR_W-1:0] mux_aw_addr  [N_AXI_M];
  logic [7:0]        mux_aw_len   [N_AXI_M];
  logic [2:0]        mux_aw_size  [N_AXI_M];
  logic [1:0]        mux_aw_burst [N_AXI_M];
  logic              mux_w_valid  [N_AXI_M];
  logic              mux_w_ready  [N_AXI_M];
  logic [DATA_W-1:0] mux_w_data   [N_AXI_M];
  logic [DATA_W/8-1:0] mux_w_strb[N_AXI_M];
  logic              mux_w_last   [N_AXI_M];
  logic              mux_b_valid  [N_AXI_M];
  logic              mux_b_ready  [N_AXI_M];

  // desc_fetch is master index N_SC (=4)
  assign mux_ar_valid[N_SC] = df_ar_valid;
  assign df_ar_ready         = mux_ar_ready[N_SC];
  assign mux_ar_addr[N_SC]  = df_ar_addr;
  assign mux_ar_len[N_SC]   = df_ar_len;
  assign mux_ar_size[N_SC]  = df_ar_size;
  assign mux_ar_burst[N_SC] = df_ar_burst;
  assign df_r_valid          = mux_r_valid[N_SC];
  assign mux_r_ready[N_SC]  = df_r_ready;
  assign df_r_data           = mux_r_data[N_SC];
  assign df_r_last           = mux_r_last[N_SC];
  // desc_fetch does not write
  assign mux_aw_valid[N_SC] = 1'b0;
  assign mux_aw_addr[N_SC]  = '0;
  assign mux_aw_len[N_SC]   = 8'd0;
  assign mux_aw_size[N_SC]  = 3'd0;
  assign mux_aw_burst[N_SC] = 2'd0;
  assign mux_w_valid[N_SC]  = 1'b0;
  assign mux_w_data[N_SC]   = '0;
  assign mux_w_strb[N_SC]   = '0;
  assign mux_w_last[N_SC]   = 1'b0;
  assign mux_b_ready[N_SC]  = 1'b1;

  axi4_master_mux #(.N_MASTERS(N_AXI_M)) u_axi_mux (
    .clk(clk), .rst_n(rst_n),
    .m_ar_valid(mux_ar_valid), .m_ar_ready(mux_ar_ready),
    .m_ar_addr(mux_ar_addr), .m_ar_len(mux_ar_len),
    .m_ar_size(mux_ar_size), .m_ar_burst(mux_ar_burst),
    .m_r_valid(mux_r_valid), .m_r_ready(mux_r_ready),
    .m_r_data(mux_r_data), .m_r_last(mux_r_last),
    .m_aw_valid(mux_aw_valid), .m_aw_ready(mux_aw_ready),
    .m_aw_addr(mux_aw_addr), .m_aw_len(mux_aw_len),
    .m_aw_size(mux_aw_size), .m_aw_burst(mux_aw_burst),
    .m_w_valid(mux_w_valid), .m_w_ready(mux_w_ready),
    .m_w_data(mux_w_data), .m_w_strb(mux_w_strb), .m_w_last(mux_w_last),
    .m_b_valid(mux_b_valid), .m_b_ready(mux_b_ready),
    .ddr_ar_valid(ddr_ar_valid), .ddr_ar_ready(ddr_ar_ready),
    .ddr_ar_addr(ddr_ar_addr), .ddr_ar_len(ddr_ar_len),
    .ddr_ar_size(ddr_ar_size), .ddr_ar_burst(ddr_ar_burst),
    .ddr_r_valid(ddr_r_valid), .ddr_r_ready(ddr_r_ready),
    .ddr_r_data(ddr_r_data), .ddr_r_last(ddr_r_last),
    .ddr_aw_valid(ddr_aw_valid), .ddr_aw_ready(ddr_aw_ready),
    .ddr_aw_addr(ddr_aw_addr), .ddr_aw_len(ddr_aw_len),
    .ddr_aw_size(ddr_aw_size), .ddr_aw_burst(ddr_aw_burst),
    .ddr_w_valid(ddr_w_valid), .ddr_w_ready(ddr_w_ready),
    .ddr_w_data(ddr_w_data), .ddr_w_strb(ddr_w_strb), .ddr_w_last(ddr_w_last),
    .ddr_b_valid(ddr_b_valid), .ddr_b_ready(ddr_b_ready)
  );

  // ─── SuperCluster Array ───
  generate
    for (genvar sc = 0; sc < N_SC; sc++) begin : gen_sc

      logic [3:0] sc_barrier_sig;

      supercluster_wrapper u_sc (
        .clk(clk), .rst_n(rst_n),
        .tile_push_valid (sc_tile_valid_w[sc]),
        .tile_push_layer (sc_layer_desc_w[sc]),
        .tile_push_tile  (sc_tile_desc_w[sc]),
        .tile_push_ready (sc_tile_ready_w[sc]),
        .barrier_grant   (barrier_grant_all),
        .barrier_signal  (sc_barrier_sig),
        // AXI master to mux
        .axi_ar_valid (mux_ar_valid[sc]),
        .axi_ar_ready (mux_ar_ready[sc]),
        .axi_ar_addr  (mux_ar_addr[sc]),
        .axi_ar_len   (mux_ar_len[sc]),
        .axi_ar_size  (mux_ar_size[sc]),
        .axi_ar_burst (mux_ar_burst[sc]),
        .axi_r_valid  (mux_r_valid[sc]),
        .axi_r_ready  (mux_r_ready[sc]),
        .axi_r_data   (mux_r_data[sc]),
        .axi_r_last   (mux_r_last[sc]),
        .axi_aw_valid (mux_aw_valid[sc]),
        .axi_aw_ready (mux_aw_ready[sc]),
        .axi_aw_addr  (mux_aw_addr[sc]),
        .axi_aw_len   (mux_aw_len[sc]),
        .axi_aw_size  (mux_aw_size[sc]),
        .axi_aw_burst (mux_aw_burst[sc]),
        .axi_w_valid  (mux_w_valid[sc]),
        .axi_w_ready  (mux_w_ready[sc]),
        .axi_w_data   (mux_w_data[sc]),
        .axi_w_strb   (mux_w_strb[sc]),
        .axi_w_last   (mux_w_last[sc]),
        .axi_b_valid  (mux_b_valid[sc]),
        .axi_b_ready  (mux_b_ready[sc]),
        .sc_idle      (sc_idle_w[sc]),
        .tiles_completed()
      );
    end
  endgenerate

  // ─── Status aggregation ───
  logic all_sc_idle;
  always_comb begin
    all_sc_idle = 1'b1;
    for (int i = 0; i < N_SC; i++)
      if (!sc_idle_w[i]) all_sc_idle = 1'b0;
  end

  // busy/done state machine
  typedef enum logic [1:0] { SYS_IDLE, SYS_RUN, SYS_DONE } sys_state_e;
  sys_state_e sys_st;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n || ctrl_soft_reset) begin
      sys_st    <= SYS_IDLE;
      stat_busy <= 1'b0;
      stat_done <= 1'b0;
      stat_error <= 1'b0;
    end else begin
      case (sys_st)
        SYS_IDLE: begin
          stat_done <= 1'b0;
          if (ctrl_start) begin
            stat_busy <= 1'b1;
            sys_st    <= SYS_RUN;
          end
        end
        SYS_RUN: begin
          // Done when: fetch finished, all SCs idle, and either every tile was
          // handed off (all_dispatched) or no tile existed (total_dispatched==0).
          if (df_done && all_sc_idle
              && (all_dispatched || (total_dispatched == 16'd0))) begin
            stat_busy <= 1'b0;
            stat_done <= 1'b1;
            sys_st    <= SYS_DONE;
          end
        end
        SYS_DONE: begin
          if (ctrl_irq_clear) begin
            stat_done <= 1'b0;
            sys_st    <= SYS_IDLE;
          end
        end
        default: sys_st <= SYS_IDLE;
      endcase
    end
  end

endmodule
