// ============================================================================
// Module : supercluster_wrapper
// Project: YOLOv10n INT8 Accelerator — V4-VC707
//
// One SuperCluster = 4 subclusters + local_arbiter_v2 + tensor_dma_v2 +
//                    tile_ingress_fifo.
// Triple-RUNNING: 3 subs compute, 1 sub fill/drain, roles rotate.
// ============================================================================
`timescale 1ns / 1ps

module supercluster_wrapper
  import accel_pkg::*;
  import desc_pkg::*;
#(
  parameter int LANES   = accel_pkg::LANES,
  parameter int PE_ROWS = accel_pkg::PE_ROWS,
  parameter int PE_COLS = accel_pkg::PE_COLS,
  parameter int N_SUBS  = accel_pkg::N_SUBS_PER_SC,
  parameter int ADDR_W  = accel_pkg::AXI_ADDR_WIDTH,
  parameter int DATA_W  = accel_pkg::AXI_DATA_WIDTH
)(
  input  logic              clk,
  input  logic              rst_n,

  // ═══════════ TILE INPUT (from global_scheduler) ═══════════
  input  logic              tile_push_valid,
  input  layer_desc_t       tile_push_layer,
  input  tile_desc_t        tile_push_tile,
  output logic              tile_push_ready,

  // ═══════════ BARRIER INTERFACE (from/to barrier_manager) ═══════════
  input  logic [3:0]        barrier_grant,
  output logic [3:0]        barrier_signal,

  // ═══════════ AXI4 MASTER (to DDR3 interconnect) ═══════════
  output logic              axi_ar_valid,
  input  logic              axi_ar_ready,
  output logic [ADDR_W-1:0] axi_ar_addr,
  output logic [7:0]        axi_ar_len,
  output logic [2:0]        axi_ar_size,
  output logic [1:0]        axi_ar_burst,

  input  logic              axi_r_valid,
  output logic              axi_r_ready,
  input  logic [DATA_W-1:0] axi_r_data,
  input  logic              axi_r_last,

  output logic              axi_aw_valid,
  input  logic              axi_aw_ready,
  output logic [ADDR_W-1:0] axi_aw_addr,
  output logic [7:0]        axi_aw_len,
  output logic [2:0]        axi_aw_size,
  output logic [1:0]        axi_aw_burst,

  output logic              axi_w_valid,
  input  logic              axi_w_ready,
  output logic [DATA_W-1:0] axi_w_data,
  output logic [DATA_W/8-1:0] axi_w_strb,
  output logic              axi_w_last,

  input  logic              axi_b_valid,
  output logic              axi_b_ready,

  // ═══════════ STATUS ═══════════
  output logic              sc_idle,
  output logic [15:0]       tiles_completed
);

  // ─── Ingress FIFO ───
  logic         fifo_pop_valid;
  layer_desc_t  fifo_pop_layer;
  tile_desc_t   fifo_pop_tile;
  logic         fifo_pop_ready;
  logic         fifo_empty;

  tile_ingress_fifo #(.DEPTH(8)) u_ingress (
    .clk        (clk),
    .rst_n      (rst_n),
    .push_valid (tile_push_valid),
    .push_layer (tile_push_layer),
    .push_tile  (tile_push_tile),
    .push_ready (tile_push_ready),
    .pop_valid  (fifo_pop_valid),
    .pop_layer  (fifo_pop_layer),
    .pop_tile   (fifo_pop_tile),
    .pop_ready  (fifo_pop_ready),
    .count      (),
    .full       (),
    .empty      (fifo_empty)
  );

  // ─── Local Arbiter ───
  logic         arb_tile_accept;
  logic         sub_tile_valid_w [N_SUBS];
  layer_desc_t  sub_layer_desc_w [N_SUBS];
  tile_desc_t   sub_tile_desc_w  [N_SUBS];
  logic         sub_tile_accept_w[N_SUBS];
  logic         sub_tile_done_w  [N_SUBS];
  logic [1:0]   dma_grant_id_w;
  logic         dma_fill_grant_w, dma_drain_grant_w;
  sub_role_e    sub_roles_w [N_SUBS];
  logic         arb_all_idle;

  local_arbiter_v2 #(.N_SUBS(N_SUBS)) u_arbiter (
    .clk            (clk),
    .rst_n          (rst_n),
    .tile_valid     (fifo_pop_valid),
    .layer_desc_in  (fifo_pop_layer),
    .tile_desc_in   (fifo_pop_tile),
    .tile_accept    (arb_tile_accept),
    .sub_tile_valid (sub_tile_valid_w),
    .sub_layer_desc (sub_layer_desc_w),
    .sub_tile_desc  (sub_tile_desc_w),
    .sub_tile_accept(sub_tile_accept_w),
    .sub_tile_done  (sub_tile_done_w),
    .dma_grant_id   (dma_grant_id_w),
    .dma_fill_grant (dma_fill_grant_w),
    .dma_drain_grant(dma_drain_grant_w),
    .sub_roles      (sub_roles_w),
    .all_idle       (arb_all_idle)
  );
  assign fifo_pop_ready = arb_tile_accept;

  // ─── DMA Engine ───
  logic              dma_glb_wr_en;
  logic [1:0]        dma_glb_wr_target, dma_glb_wr_bank_id;
  logic [11:0]       dma_glb_wr_addr;
  logic signed [7:0] dma_glb_wr_data [LANES];
  logic [LANES-1:0]  dma_glb_wr_mask;
  logic              dma_glb_rd_en;
  logic [1:0]        dma_glb_rd_bank_id;
  logic [11:0]       dma_glb_rd_addr;
  logic signed [7:0] dma_glb_rd_data [LANES];

  tensor_dma_v2 #(.LANES(LANES)) u_dma (
    .clk            (clk),
    .rst_n          (rst_n),
    .fill_start     (1'b0),  // Driven by tile_fsm indirectly
    .fill_ddr_addr  ('0),
    .fill_length    (24'd0),
    .fill_target    (2'd0),
    .fill_bank_id   (2'd0),
    .fill_done      (),
    .drain_start    (1'b0),
    .drain_ddr_addr ('0),
    .drain_length   (24'd0),
    .drain_bank_id  (2'd0),
    .drain_done     (),
    .glb_wr_en      (dma_glb_wr_en),
    .glb_wr_target  (dma_glb_wr_target),
    .glb_wr_bank_id (dma_glb_wr_bank_id),
    .glb_wr_addr    (dma_glb_wr_addr),
    .glb_wr_data    (dma_glb_wr_data),
    .glb_wr_mask    (dma_glb_wr_mask),
    .glb_rd_en      (dma_glb_rd_en),
    .glb_rd_bank_id (dma_glb_rd_bank_id),
    .glb_rd_addr    (dma_glb_rd_addr),
    .glb_rd_data    (dma_glb_rd_data),
    .axi_ar_valid   (axi_ar_valid),
    .axi_ar_ready   (axi_ar_ready),
    .axi_ar_addr    (axi_ar_addr),
    .axi_ar_len     (axi_ar_len),
    .axi_ar_size    (axi_ar_size),
    .axi_ar_burst   (axi_ar_burst),
    .axi_r_valid    (axi_r_valid),
    .axi_r_ready    (axi_r_ready),
    .axi_r_data     (axi_r_data),
    .axi_r_last     (axi_r_last),
    .axi_aw_valid   (axi_aw_valid),
    .axi_aw_ready   (axi_aw_ready),
    .axi_aw_addr    (axi_aw_addr),
    .axi_aw_len     (axi_aw_len),
    .axi_aw_size    (axi_aw_size),
    .axi_aw_burst   (axi_aw_burst),
    .axi_w_valid    (axi_w_valid),
    .axi_w_ready    (axi_w_ready),
    .axi_w_data     (axi_w_data),
    .axi_w_strb     (axi_w_strb),
    .axi_w_last     (axi_w_last),
    .axi_b_valid    (axi_b_valid),
    .axi_b_ready    (axi_b_ready)
  );

  // ─── Subcluster Array (4 instances) ───
  logic [3:0] barrier_signal_w;
  assign barrier_signal = barrier_signal_w;

  // Per-sub quant tables (simplified: shared across subs, loaded once per layer)
  int32_t  bias_tbl  [256];
  uint32_t m_int_tbl [256];
  logic [7:0] shift_tbl [256];
  int8_t   zp_out_tbl[256];

  // Tile completion counter
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      tiles_completed <= 16'd0;
    else
      for (int s = 0; s < N_SUBS; s++)
        if (sub_tile_done_w[s])
          tiles_completed <= tiles_completed + 16'd1;
  end

  generate
    for (genvar s = 0; s < N_SUBS; s++) begin : gen_sub
      tile_state_e sub_fsm_st;
      logic sub_barrier_sig;

      subcluster_datapath #(
        .LANES   (LANES),
        .PE_ROWS (PE_ROWS),
        .PE_COLS (PE_COLS)
      ) u_sub (
        .clk             (clk),
        .rst_n           (rst_n),
        .tile_valid      (sub_tile_valid_w[s]),
        .layer_desc_in   (sub_layer_desc_w[s]),
        .tile_desc_in    (sub_tile_desc_w[s]),
        .tile_accept     (sub_tile_accept_w[s]),
        // DMA write port: connect to DMA when this sub has DMA grant
        .ext_wr_en       (dma_glb_wr_en && (dma_grant_id_w == s[1:0])),
        .ext_wr_target   (dma_glb_wr_target),
        .ext_wr_bank_id  (dma_glb_wr_bank_id),
        .ext_wr_addr     (dma_glb_wr_addr),
        .ext_wr_data     (dma_glb_wr_data),
        .ext_wr_mask     (dma_glb_wr_mask),
        // DMA read port
        .ext_rd_en       (dma_glb_rd_en && (dma_grant_id_w == s[1:0])),
        .ext_rd_bank_id  (dma_glb_rd_bank_id),
        .ext_rd_addr     (dma_glb_rd_addr),
        .ext_rd_act_data (),
        .ext_rd_psum_data(),
        // Quant tables
        .bias_table      (bias_tbl),
        .m_int_table     (m_int_tbl),
        .shift_table     (shift_tbl),
        .zp_out_table    (zp_out_tbl),
        // Barrier
        .barrier_grant   (barrier_grant[sub_tile_desc_w[s].barrier_id]),
        .barrier_signal  (sub_barrier_sig),
        // Status
        .fsm_state       (sub_fsm_st),
        .tile_done       (sub_tile_done_w[s]),
        .dbg_k_pass      (),
        .dbg_iter_mp5_ch ()
      );

      assign barrier_signal_w[s] = sub_barrier_sig;
    end
  endgenerate

  assign sc_idle = arb_all_idle && fifo_empty;

endmodule
