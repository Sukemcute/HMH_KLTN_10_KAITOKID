`timescale 1ns/1ps
// Top control wrapper: CSR MMIO + desc_fetch_engine + barrier_manager + global_scheduler.
module controller_system (
  input  logic               clk,
  input  logic               rst_n,

  // AXI-Lite MMIO (CPU control)
  input  logic [11:0]        mmio_addr,
  input  logic [31:0]        mmio_wdata,
  input  logic               mmio_we,
  input  logic               mmio_re,
  output logic [31:0]        mmio_rdata,
  output logic               irq,

  // AXI4 read master (for descriptor fetch)
  output logic [39:0]        axi_araddr,
  output logic [7:0]         axi_arlen,
  output logic               axi_arvalid,
  input  logic               axi_arready,
  input  logic [255:0]       axi_rdata,
  input  logic               axi_rvalid,
  input  logic               axi_rlast,
  output logic               axi_rready,

  // To 4 SuperClusters
  output desc_pkg::tile_desc_t  sc_tile [4],
  output desc_pkg::layer_desc_t sc_layer_desc,
  output logic                   sc_tile_valid [4],
  input  logic                   sc_tile_accept [4],

  // Barrier net
  input  logic                   barrier_signal_in [4],
  input  logic [4:0]            barrier_signal_id_in [4],
  output logic [31:0]           barrier_scoreboard
);
  import csr_pkg::*;
  import desc_pkg::*;
  import accel_pkg::*;

  // ───── CSR Registers ─────
  logic        csr_start, csr_soft_reset, csr_irq_clr;
  logic        status_busy, status_done;
  logic [63:0] net_desc_base;
  logic [7:0]  layer_start_reg, layer_end_reg;
  logic [31:0] irq_mask;
  logic [63:0] perf_cycle_cnt;
  logic [31:0] perf_tile_done_cnt;

  // CSR Write
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      csr_start     <= 1'b0;
      csr_soft_reset<= 1'b0;
      csr_irq_clr   <= 1'b0;
      net_desc_base <= '0;
      layer_start_reg <= '0;
      layer_end_reg   <= 8'd22;
      irq_mask      <= '0;
    end else begin
      csr_start    <= 1'b0;
      csr_irq_clr  <= 1'b0;
      if (mmio_we) begin
        case (mmio_addr)
          CSR_CTRL[11:0]: begin
            csr_start      <= mmio_wdata[0];
            csr_soft_reset <= mmio_wdata[1];
            csr_irq_clr    <= mmio_wdata[2];
          end
          CSR_NET_DESC_LO[11:0]: net_desc_base[31:0]  <= mmio_wdata;
          CSR_NET_DESC_HI[11:0]: net_desc_base[63:32] <= mmio_wdata;
          CSR_LAYER_START[11:0]: layer_start_reg       <= mmio_wdata[7:0];
          CSR_LAYER_END[11:0]:   layer_end_reg         <= mmio_wdata[7:0];
          CSR_IRQ_MASK[11:0]:    irq_mask              <= mmio_wdata;
          default: ;
        endcase
      end
    end
  end

  // CSR Read
  always_comb begin
    mmio_rdata = 32'd0;
    if (mmio_re) begin
      case (mmio_addr)
        CSR_CTRL[11:0]:          mmio_rdata = {29'd0, csr_irq_clr, csr_soft_reset, csr_start};
        CSR_STATUS[11:0]:        mmio_rdata = {28'd0, 1'b0, irq, status_done, status_busy};
        CSR_VERSION[11:0]:       mmio_rdata = 32'h2026_0320;
        CSR_CAP0[11:0]:          mmio_rdata = {16'd0, 4'(SUPER_CLUSTERS), 4'(SUBS_PER_SC),
                                               4'(PE_ROWS), 4'(PE_COLS)};
        CSR_NET_DESC_LO[11:0]:   mmio_rdata = net_desc_base[31:0];
        CSR_NET_DESC_HI[11:0]:   mmio_rdata = net_desc_base[63:32];
        CSR_LAYER_START[11:0]:   mmio_rdata = {24'd0, layer_start_reg};
        CSR_LAYER_END[11:0]:     mmio_rdata = {24'd0, layer_end_reg};
        CSR_PERF_CYCLE_LO[11:0]: mmio_rdata = perf_cycle_cnt[31:0];
        CSR_PERF_CYCLE_HI[11:0]: mmio_rdata = perf_cycle_cnt[63:32];
        CSR_PERF_TILE_DONE[11:0]:mmio_rdata = perf_tile_done_cnt;
        CSR_BARRIER_STATUS[11:0]:mmio_rdata = barrier_scoreboard;
        default: ;
      endcase
    end
  end

  // ───── Descriptor Fetch Engine ─────
  net_desc_t   fetched_net;
  logic        fetched_net_valid;
  layer_desc_t fetched_layer;
  logic        fetched_layer_valid;
  tile_desc_t  fetched_tile;
  logic        fetched_tile_valid, fetched_tile_ready;
  logic [7:0]  fetch_current_layer;
  logic        fetch_all_done;

  desc_fetch_engine u_fetch (
    .clk            (clk),
    .rst_n          (rst_n),
    .start          (csr_start),
    .axi_araddr     (axi_araddr),
    .axi_arlen      (axi_arlen),
    .axi_arvalid    (axi_arvalid),
    .axi_arready    (axi_arready),
    .axi_rdata      (axi_rdata),
    .axi_rvalid     (axi_rvalid),
    .axi_rlast      (axi_rlast),
    .axi_rready     (axi_rready),
    .net_desc_base  (net_desc_base),
    .layer_start    (layer_start_reg),
    .layer_end      (layer_end_reg),
    .net_desc       (fetched_net),
    .net_desc_valid (fetched_net_valid),
    .layer_desc     (fetched_layer),
    .layer_desc_valid(fetched_layer_valid),
    .tile_desc      (fetched_tile),
    .tile_desc_valid(fetched_tile_valid),
    .tile_desc_ready(fetched_tile_ready),
    .current_layer  (fetch_current_layer),
    .all_layers_done(fetch_all_done)
  );

  // ───── Performance Counters (after fetch: needs tile valid/ready) ─────
  // PERF_TILE_DONE: count DF→GS tile handshakes (one logical descriptor dispatch).
  wire perf_tile_handshake = fetched_tile_valid && fetched_tile_ready;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n || csr_soft_reset) begin
      perf_cycle_cnt     <= '0;
      perf_tile_done_cnt <= '0;
    end else if (status_busy) begin
      perf_cycle_cnt <= perf_cycle_cnt + 1;
      if (perf_tile_handshake)
        perf_tile_done_cnt <= perf_tile_done_cnt + 1;
    end
  end

  // ───── Global Scheduler ─────
  logic sched_layer_complete;
  logic sched_inference_complete_gs;
  // GS should eventually assert when last tile + DMA complete; until then OR in
  // desc_fetch "all layers walked" so CSR STATUS/IRQ is not stuck forever.
  wire sched_inference_complete = sched_inference_complete_gs | fetch_all_done;

  global_scheduler u_sched (
    .clk                (clk),
    .rst_n              (rst_n),
    .layer_desc         (fetched_layer),
    .layer_valid        (fetched_layer_valid),
    .tile_desc_in       (fetched_tile),
    .tile_valid         (fetched_tile_valid),
    .tile_accept        (fetched_tile_ready),
    .sc_tile            (sc_tile),
    .sc_tile_valid      (sc_tile_valid),
    .sc_tile_accept     (sc_tile_accept),
    .current_layer_id   (),
    .layer_complete     (sched_layer_complete),
    .inference_complete (sched_inference_complete_gs)
  );

  assign sc_layer_desc = fetched_layer;

  // ───── Barrier Manager ─────
  // Aggregate barrier signals from all 4 SCs
  logic        barrier_sig;
  logic [4:0]  barrier_sig_id;

  always_comb begin
    barrier_sig    = 1'b0;
    barrier_sig_id = '0;
    for (int i = 0; i < 4; i++) begin
      if (barrier_signal_in[i]) begin
        barrier_sig    = 1'b1;
        barrier_sig_id = barrier_signal_id_in[i];
      end
    end
  end

  barrier_manager #(.NUM_BARRIERS(32)) u_barrier (
    .clk               (clk),
    .rst_n             (rst_n),
    .clear_all         (csr_soft_reset),
    .signal_valid      (barrier_sig),
    .signal_barrier_id (barrier_sig_id),
    .wait_valid        (1'b0),
    .wait_barrier_id   (5'd0),
    .wait_grant        (),
    .scoreboard        (barrier_scoreboard)
  );

  // ───── Status & IRQ ─────
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      status_busy <= 1'b0;
      status_done <= 1'b0;
    end else begin
      if (csr_start) begin
        status_busy <= 1'b1;
        status_done <= 1'b0;
      end
      if (sched_inference_complete) begin
        status_busy <= 1'b0;
        status_done <= 1'b1;
      end
      if (csr_irq_clr)
        status_done <= 1'b0;
    end
  end

  assign irq = status_done && irq_mask[0];

  // ───── Debug checkpoints (e.g. xvlog -d ACCEL_DEBUG) ─────
`ifdef ACCEL_DEBUG
  logic dbg_prev_sched_done;
  logic dbg_prev_fetch_done;
  logic dbg_prev_status_done;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      dbg_prev_sched_done  <= 1'b0;
      dbg_prev_fetch_done  <= 1'b0;
      dbg_prev_status_done <= 1'b0;
    end else begin
      if (csr_start)
        $display("[%t] %m [CHK-CTRL] CSR START (inference kick)", $time);
      if (sched_inference_complete && !dbg_prev_sched_done)
        $display("[%t] %m [CHK-CTRL] sched_inference_complete (fetch_done|gs_done)", $time);
      dbg_prev_sched_done <= sched_inference_complete;
      if (fetch_all_done && !dbg_prev_fetch_done)
        $display("[%t] %m [CHK-CTRL] desc_fetch all_layers_done (DF_DONE)", $time);
      dbg_prev_fetch_done <= fetch_all_done;
      if (status_done && !dbg_prev_status_done && irq_mask[0])
        $display("[%t] %m [CHK-CTRL] status_done -> IRQ", $time);
      dbg_prev_status_done <= status_done;
    end
  end
`endif

endmodule
