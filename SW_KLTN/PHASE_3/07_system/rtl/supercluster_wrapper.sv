`timescale 1ns/1ps
// SuperCluster wrapper: 4 × subcluster_wrapper + local_arbiter + tensor_dma.
//
// Replaces the Phase-A stub that had bare tile_fsm instances and
// debug-pattern AXI writes.  Now instantiates the full compute path
// and uses tensor_dma for multi-beat AXI4 burst transfers.
module supercluster_wrapper #(
  parameter int NUM_SUBS = 4,
  parameter int LANES    = 32,
  // Decouple global_scheduler from sub completion: accept tiles while FIFO not full.
  // (Previously tile_accept required has_idle_sub → 5th tile to same SC deadlocked DF/GS.)
  // Behavioral conv in cosim is slow; shallow FIFO falsely mimics "no compute" deadlock.
  parameter int TILE_INGRESS_DEPTH = 64
)(
  input  logic               clk,
  input  logic               rst_n,

  // From global_scheduler
  input  desc_pkg::tile_desc_t  tile_in,
  input  desc_pkg::layer_desc_t layer_desc,
  input  logic                   tile_valid,
  output logic                   tile_accept,

  // AXI4 Master (shared DDR port)
  output logic [39:0]        m_axi_araddr,
  output logic [7:0]         m_axi_arlen,
  output logic               m_axi_arvalid,
  input  logic               m_axi_arready,
  input  logic [255:0]       m_axi_rdata,
  input  logic               m_axi_rvalid,
  input  logic               m_axi_rlast,
  output logic               m_axi_rready,
  output logic [39:0]        m_axi_awaddr,
  output logic [7:0]         m_axi_awlen,
  output logic               m_axi_awvalid,
  input  logic               m_axi_awready,
  output logic [255:0]       m_axi_wdata,
  output logic               m_axi_wvalid,
  output logic               m_axi_wlast,
  input  logic               m_axi_wready,
  input  logic [1:0]         m_axi_bresp,
  input  logic               m_axi_bvalid,
  output logic               m_axi_bready,

  // PPU parameters (shared across all subclusters, changes per layer)
  input  logic signed [31:0] ppu_bias    [LANES],
  input  logic signed [31:0] ppu_m_int   [LANES],
  input  logic        [5:0]  ppu_shift   [LANES],
  input  logic signed [7:0]  ppu_zp_out,
  input  logic signed [7:0]  ppu_silu_lut [256],

  // Barrier
  output logic               barrier_signal,
  output logic [4:0]         barrier_signal_id,
  input  logic               barrier_grant,

  // Status
  output logic               layer_done,
  output logic [15:0]        tiles_completed
);
  import accel_pkg::*;
  import desc_pkg::*;

  // ═══════════════════════════════════════════════════════════
  //  Per-subcluster wires
  // ═══════════════════════════════════════════════════════════
  tile_state_e sub_state       [NUM_SUBS];
  logic        sub_tile_done   [NUM_SUBS];
  sc_role_e    sub_role        [NUM_SUBS];
  logic        sub_tile_valid_arb [NUM_SUBS];
  tile_desc_t  sub_tile_arb    [NUM_SUBS];

  logic        sub_ext_rd_req  [NUM_SUBS];
  logic [39:0] sub_ext_rd_addr [NUM_SUBS];
  logic [15:0] sub_ext_rd_len  [NUM_SUBS];
  logic        sub_ext_rd_grant[NUM_SUBS];
  logic [255:0]sub_ext_rd_data [NUM_SUBS];
  logic        sub_ext_rd_valid[NUM_SUBS];

  logic        sub_ext_wr_req  [NUM_SUBS];
  logic [39:0] sub_ext_wr_addr [NUM_SUBS];
  logic [15:0] sub_ext_wr_len  [NUM_SUBS];
  logic        sub_ext_wr_grant[NUM_SUBS];
  logic [255:0]sub_ext_wr_data [NUM_SUBS];
  logic        sub_ext_wr_valid[NUM_SUBS];
  logic        sub_ext_wr_done [NUM_SUBS];
  logic        sub_ext_wr_beat [NUM_SUBS];
  logic        sub_ext_rd_done [NUM_SUBS];

  logic        sub_barrier_signal [NUM_SUBS];
  logic [4:0]  sub_barrier_id     [NUM_SUBS];
  logic        sub_layer_done     [NUM_SUBS];

  // ═══════════════════════════════════════════════════════════
  //  GS ingress FIFO (decouples tile_accept from sub idle)
  // ═══════════════════════════════════════════════════════════
  
  localparam int FIFO_LVL_W = $clog2(TILE_INGRESS_DEPTH + 1);
  localparam int FIFO_PTR_W = $clog2(TILE_INGRESS_DEPTH);

  tile_desc_t fifo_mem [TILE_INGRESS_DEPTH];
  logic [FIFO_PTR_W-1:0] fifo_wptr, fifo_rptr;
  logic [FIFO_LVL_W-1:0] fifo_level;

  wire fifo_empty = (fifo_level == 0);
  wire fifo_full  = (fifo_level == TILE_INGRESS_DEPTH);
  wire fifo_push  = tile_valid && tile_accept;
  logic arb_tile_consumed;
  wire fifo_pop   = arb_tile_consumed && !fifo_empty;

  assign tile_accept = rst_n && !fifo_full;

  tile_desc_t fifo_head;
  assign fifo_head = fifo_mem[fifo_rptr];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      fifo_wptr  <= '0;
      fifo_rptr  <= '0;
      fifo_level <= '0;
    end else begin
      unique case ({fifo_push, fifo_pop})
        2'b10: fifo_level <= fifo_level + 1'b1;
        2'b01: fifo_level <= fifo_level - 1'b1;
        default: ;
      endcase
      if (fifo_push) begin
        fifo_mem[fifo_wptr] <= tile_in;
        if (fifo_wptr == FIFO_PTR_W'(TILE_INGRESS_DEPTH - 1))
          fifo_wptr <= '0;
        else
          fifo_wptr <= fifo_wptr + 1'b1;
      end
      if (fifo_pop) begin
        if (fifo_rptr == FIFO_PTR_W'(TILE_INGRESS_DEPTH - 1))
          fifo_rptr <= '0;
        else
          fifo_rptr <= fifo_rptr + 1'b1;
      end
    end
  end

  // ═══════════════════════════════════════════════════════════
  //  Local Arbiter
  // ═══════════════════════════════════════════════════════════
  logic        dma_busy;
  logic [1:0]  arb_ext_grant_sub;
  logic        arb_ext_is_read;

  logic        arb_has_idle_sub;

  local_arbiter #(.NUM_SUBS(NUM_SUBS)) u_arbiter (
    .clk              (clk),
    .rst_n            (rst_n),
    .tile_available   (!fifo_empty),
    .next_tile        (fifo_head),
    .tile_consumed    (arb_tile_consumed),
    .sub_state        (sub_state),
    .sub_tile_done    (sub_tile_done),
    .sub_dma_wr_req   (sub_ext_wr_req),
    .sub_role         (sub_role),
    .ext_port_ready   (!dma_busy),
    .ext_port_grant_sub(arb_ext_grant_sub),
    .ext_port_is_read (arb_ext_is_read),
    .sub_tile_valid   (sub_tile_valid_arb),
    .sub_tile         (sub_tile_arb),
    .has_idle_sub     (arb_has_idle_sub)
  );

  // ═══════════════════════════════════════════════════════════
  //  Subcluster Instances (full compute path)
  // ═══════════════════════════════════════════════════════════
  genvar si;
  generate
    for (si = 0; si < NUM_SUBS; si++) begin : gen_sub
      subcluster_wrapper #(.LANES(LANES)) u_sub (
        .clk            (clk),
        .rst_n          (rst_n),
        .tile_valid     (sub_tile_valid_arb[si]),
        .tile_desc      (sub_tile_arb[si]),
        .layer_desc     (layer_desc),
        .tile_accept    (),
        .ext_rd_req     (sub_ext_rd_req[si]),
        .ext_rd_addr    (sub_ext_rd_addr[si]),
        .ext_rd_len     (sub_ext_rd_len[si]),
        .ext_rd_grant   (sub_ext_rd_grant[si]),
        .ext_rd_data    (sub_ext_rd_data[si]),
        .ext_rd_valid   (sub_ext_rd_valid[si]),
        .ext_wr_req     (sub_ext_wr_req[si]),
        .ext_wr_addr    (sub_ext_wr_addr[si]),
        .ext_wr_len     (sub_ext_wr_len[si]),
        .ext_wr_grant   (sub_ext_wr_grant[si]),
        .ext_wr_data    (sub_ext_wr_data[si]),
        .ext_wr_valid   (sub_ext_wr_valid[si]),
        .ext_wr_done    (sub_ext_wr_done[si]),
        .ext_wr_beat    (sub_ext_wr_beat[si]),
        .ppu_bias       (ppu_bias),
        .ppu_m_int      (ppu_m_int),
        .ppu_shift      (ppu_shift),
        .ppu_zp_out     (ppu_zp_out),
        .ppu_silu_lut   (ppu_silu_lut),
        .barrier_signal (sub_barrier_signal[si]),
        .barrier_signal_id(sub_barrier_id[si]),
        .barrier_grant  (barrier_grant),
        .state          (sub_state[si]),
        .tile_done      (sub_tile_done[si]),
        .layer_done     (sub_layer_done[si])
      );
    end
  endgenerate

  // ═══════════════════════════════════════════════════════════
  //  DMA Arbiter / Grant Latch  (request-based round-robin)
  //  Scans sub_ext_{rd,wr}_req directly so a ROLE_RUNNING sub
  //  that still needs DMA (PREFILL_WT/IN/SKIP or SWIZZLE_STORE)
  //  is always reachable.  Write requests have priority.
  // ═══════════════════════════════════════════════════════════
  logic [1:0]  dma_cur_sub;
  logic        dma_is_read;
  logic [1:0]  dma_rr_ptr;

  logic        dma_rd_req_int, dma_rd_done_int;
  logic [39:0] dma_rd_addr_int;
  logic [15:0] dma_rd_len_int;
  logic [255:0]dma_rd_data_int;
  logic        dma_rd_valid_int;

  logic        dma_wr_req_int, dma_wr_done_int;
  logic        dma_wr_beat_int;
  logic [39:0] dma_wr_addr_int;
  logic [15:0] dma_wr_len_int;
  logic [255:0]dma_wr_data_int;
  logic        dma_wr_valid_int;

  logic [1:0]  next_dma_sub;
  logic        next_dma_is_read;
  logic        next_dma_found;

  logic [1:0] rr_idx [4];
  always_comb begin
    for (int i = 0; i < NUM_SUBS; i++)
      rr_idx[i] = dma_rr_ptr + i[1:0];
  end

  always_comb begin
    next_dma_found   = 1'b0;
    next_dma_sub     = dma_rr_ptr;
    next_dma_is_read = 1'b1;

    // Priority 1: write requests (drain output to DDR)
    for (int i = 0; i < NUM_SUBS; i++) begin
      if (!next_dma_found && sub_ext_wr_req[rr_idx[i]]) begin
        next_dma_found   = 1'b1;
        next_dma_sub     = rr_idx[i];
        next_dma_is_read = 1'b0;
      end
    end

    // Priority 2: read requests (prefill weights / activations)
    if (!next_dma_found) begin
      for (int i = 0; i < NUM_SUBS; i++) begin
        if (!next_dma_found && sub_ext_rd_req[rr_idx[i]]) begin
          next_dma_found   = 1'b1;
          next_dma_sub     = rr_idx[i];
          next_dma_is_read = 1'b1;
        end
      end
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      dma_busy    <= 1'b0;
      dma_cur_sub <= '0;
      dma_is_read <= 1'b0;
      dma_rr_ptr  <= '0;
    end else if (!dma_busy) begin
      if (next_dma_found) begin
        dma_busy    <= 1'b1;
        dma_cur_sub <= next_dma_sub;
        dma_is_read <= next_dma_is_read;
        dma_rr_ptr  <= next_dma_sub + 2'd1;
      end
    end else begin
      if (dma_rd_done_int || dma_wr_done_int)
        dma_busy <= 1'b0;
    end
  end

  // Mux: selected subcluster → tensor_dma
  always_comb begin
    if (dma_busy && dma_is_read) begin
      dma_rd_req_int  = sub_ext_rd_req[dma_cur_sub];
      dma_rd_addr_int = sub_ext_rd_addr[dma_cur_sub];
      dma_rd_len_int  = sub_ext_rd_len[dma_cur_sub];
    end else begin
      dma_rd_req_int  = 1'b0;
      dma_rd_addr_int = '0;
      dma_rd_len_int  = '0;
    end

    if (dma_busy && !dma_is_read) begin
      dma_wr_req_int   = sub_ext_wr_req[dma_cur_sub];
      dma_wr_addr_int  = sub_ext_wr_addr[dma_cur_sub];
      dma_wr_len_int   = sub_ext_wr_len[dma_cur_sub];
      dma_wr_data_int  = sub_ext_wr_data[dma_cur_sub];
      dma_wr_valid_int = sub_ext_wr_valid[dma_cur_sub];
    end else begin
      dma_wr_req_int   = 1'b0;
      dma_wr_addr_int  = '0;
      dma_wr_len_int   = '0;
      dma_wr_data_int  = '0;
      dma_wr_valid_int = 1'b0;
    end
  end

  // Demux: tensor_dma → selected subcluster
  always_comb begin
    for (int i = 0; i < NUM_SUBS; i++) begin
      sub_ext_rd_grant[i] = (dma_busy && dma_is_read && dma_cur_sub == i[1:0]);
      sub_ext_rd_data[i]  = (dma_cur_sub == i[1:0]) ? dma_rd_data_int : '0;
      sub_ext_rd_valid[i] = (dma_cur_sub == i[1:0]) ? dma_rd_valid_int : 1'b0;
      sub_ext_wr_grant[i] = (dma_busy && !dma_is_read && dma_cur_sub == i[1:0]);
      sub_ext_rd_done[i]  = (dma_cur_sub == i[1:0]) ? dma_rd_done_int : 1'b0;
      sub_ext_wr_done[i]  = (dma_cur_sub == i[1:0]) ? dma_wr_done_int : 1'b0;
      sub_ext_wr_beat[i]  = (dma_cur_sub == i[1:0]) ? dma_wr_beat_int : 1'b0;
    end
  end

  // ═══════════════════════════════════════════════════════════
  //  Tensor DMA (multi-beat AXI4 burst engine)
  // ═══════════════════════════════════════════════════════════
  // tensor_dma produces arsize/arburst/awsize/awburst/wstrb internally;
  // they are left unconnected at the supercluster boundary since accel_top
  // AXI arbiter does not pass them through (DDR model uses defaults).
  logic        dma_arvalid, dma_arready;
  logic [39:0] dma_araddr;
  logic [7:0]  dma_arlen;
  logic        dma_rready;

  logic        dma_awvalid, dma_awready;
  logic [39:0] dma_awaddr;
  logic [7:0]  dma_awlen;
  logic [255:0]dma_wdata;
  logic        dma_wvalid, dma_wlast;

  tensor_dma #(
    .AXI_DATA_W(256),
    .AXI_ADDR_W(40)
  ) u_dma (
    .clk             (clk),
    .rst_n           (rst_n),
    // AXI read channel
    .m_axi_araddr    (dma_araddr),
    .m_axi_arlen     (dma_arlen),
    .m_axi_arsize    (),
    .m_axi_arburst   (),
    .m_axi_arvalid   (dma_arvalid),
    .m_axi_arready   (dma_arready),
    .m_axi_rdata     (m_axi_rdata),
    .m_axi_rresp     (2'b00),
    .m_axi_rlast     (m_axi_rlast),
    .m_axi_rvalid    (m_axi_rvalid),
    .m_axi_rready    (dma_rready),
    // AXI write channel
    .m_axi_awaddr    (dma_awaddr),
    .m_axi_awlen     (dma_awlen),
    .m_axi_awsize    (),
    .m_axi_awburst   (),
    .m_axi_awvalid   (dma_awvalid),
    .m_axi_awready   (dma_awready),
    .m_axi_wdata     (dma_wdata),
    .m_axi_wstrb     (),
    .m_axi_wlast     (dma_wlast),
    .m_axi_wvalid    (dma_wvalid),
    .m_axi_wready    (m_axi_wready),
    .m_axi_bresp     (m_axi_bresp),
    .m_axi_bvalid    (m_axi_bvalid),
    .m_axi_bready    (),
    // Internal read interface
    .rd_req          (dma_rd_req_int),
    .rd_addr         (dma_rd_addr_int),
    .rd_byte_len     (dma_rd_len_int),
    .rd_data_valid   (dma_rd_valid_int),
    .rd_data         (dma_rd_data_int),
    .rd_done         (dma_rd_done_int),
    // Internal write interface
    .wr_req          (dma_wr_req_int),
    .wr_addr         (dma_wr_addr_int),
    .wr_byte_len     (dma_wr_len_int),
    .wr_data_in      (dma_wr_data_int),
        .wr_data_valid_in(dma_wr_valid_int),
    .wr_done         (dma_wr_done_int),
    .wr_beat_accept  (dma_wr_beat_int)
  );

  // Map tensor_dma AXI outputs to supercluster m_axi ports
  assign m_axi_araddr  = dma_araddr;
  assign m_axi_arlen   = dma_arlen;
  assign m_axi_arvalid = dma_arvalid;
  assign dma_arready   = m_axi_arready;
  assign m_axi_rready  = dma_rready;

  assign m_axi_awaddr  = dma_awaddr;
  assign m_axi_awlen   = dma_awlen;
  assign m_axi_awvalid = dma_awvalid;
  assign dma_awready   = m_axi_awready;
  assign m_axi_wdata   = dma_wdata;
  assign m_axi_wvalid  = dma_wvalid;
  assign m_axi_wlast   = dma_wlast;
  assign m_axi_bready  = 1'b1;

  // ═══════════════════════════════════════════════════════════
  //  Barrier Aggregation
  // ═══════════════════════════════════════════════════════════
  always_comb begin
    barrier_signal    = 1'b0;
    barrier_signal_id = '0;
    for (int i = 0; i < NUM_SUBS; i++) begin
      if (sub_barrier_signal[i]) begin
        barrier_signal    = 1'b1;
        barrier_signal_id = sub_barrier_id[i];
      end
    end
  end

  // ═══════════════════════════════════════════════════════════
  //  Layer Done & Tile Counting
  // ═══════════════════════════════════════════════════════════
  always_comb begin
    layer_done = 1'b0;
    for (int i = 0; i < NUM_SUBS; i++)
      layer_done = layer_done | sub_layer_done[i];
  end

  logic [15:0] tile_done_cnt;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      tile_done_cnt <= '0;
    else begin
      automatic int unsigned td_inc;
      td_inc = 0;
      for (int i = 0; i < NUM_SUBS; i++)
        if (sub_tile_done[i]) td_inc++;
      tile_done_cnt <= tile_done_cnt + 16'(td_inc);
    end
  end
  assign tiles_completed = tile_done_cnt;

endmodule
