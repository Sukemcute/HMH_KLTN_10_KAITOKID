// ============================================================================
// Module : subcluster_datapath
// Project: YOLOv10n INT8 Accelerator — V4-VC707
//
// Description:
//   ★ COMPLETE INTEGRATION: Wire-up of ALL subcluster modules.
//   1 subcluster = 1 FIXED hardware that runs ANY primitive via descriptor.
//
//   Module instances (24+):
//     CONTROL (3):   tile_fsm, shadow_reg_file, compute_sequencer
//     MEMORY (11):   glb_input_bank_db ×3, glb_weight_bank ×3,
//                    glb_output_bank ×4, metadata_ram
//     ADDR_GEN (3):  addr_gen_input, addr_gen_weight, addr_gen_output
//     DATA_MOVE (3): router_cluster_v2, window_gen, swizzle_engine
//     COMPUTE (1):   pe_cluster_v4 (contains 12 PEs + 4 col_reduce + 1 comp_tree)
//     POST_PROC (4): ppu ×4 (1 per PE column, parallel)
//
//   This module has external ports for:
//     - Descriptor injection (from scheduler/testbench)
//     - DMA data fill/drain (GLB ↔ DDR3)
//     - Bias/quant parameter loading
//     - Status signaling
//
// Instances: 16 total (4 SuperClusters × 4 Subclusters each)
// ============================================================================
`timescale 1ns / 1ps

module subcluster_datapath
  import accel_pkg::*;
  import desc_pkg::*;
#(
  parameter int LANES   = accel_pkg::LANES,      // 20
  parameter int PE_ROWS = accel_pkg::PE_ROWS,    // 3
  parameter int PE_COLS = accel_pkg::PE_COLS      // 4
)(
  input  logic          clk,
  input  logic          rst_n,

  // ═══════════ DESCRIPTOR INPUT ═══════════
  input  logic          tile_valid,
  input  layer_desc_t   layer_desc_in,
  input  tile_desc_t    tile_desc_in,
  output logic          tile_accept,

  // ═══════════ DMA DATA PORT (fill GLB from DDR3) ═══════════
  // Write port: testbench or DMA fills input/weight/output banks
  input  logic          ext_wr_en,
  input  logic [1:0]    ext_wr_target,     // 0=input, 1=weight, 2=output
  input  logic [1:0]    ext_wr_bank_id,
  input  logic [11:0]   ext_wr_addr,
  input  int8_t         ext_wr_data [LANES],
  input  logic [LANES-1:0] ext_wr_mask,

  // Read port: testbench or DMA reads output
  input  logic          ext_rd_en,
  input  logic [1:0]    ext_rd_bank_id,
  input  logic [11:0]   ext_rd_addr,
  output int8_t         ext_rd_act_data [LANES],
  output int32_t        ext_rd_psum_data [LANES],

  // ═══════════ BIAS & QUANT PARAMS (per cout, preloaded) ═══════════
  input  int32_t        bias_table [256],
  input  uint32_t       m_int_table [256],
  input  logic [7:0]    shift_table [256],
  input  int8_t         zp_out_table [256],

  // ═══════════ BARRIER INTERFACE ═══════════
  input  logic          barrier_grant,
  output logic          barrier_signal,

  // ═══════════ STATUS ═══════════
  output tile_state_e   fsm_state,
  output logic          tile_done,

  // ★ Stage 8 checkpoints: observe FSM + MP5 channel index
  output logic [3:0]    dbg_k_pass,
  output logic [9:0]    dbg_iter_mp5_ch
);

  // ══════════════════════════════════════════════════════════════
  // INTERNAL WIRES — Organized by connection group
  // ══════════════════════════════════════════════════════════════

  // ── tile_fsm ↔ shadow_reg_file ──
  logic shadow_latch;

  // ── tile_fsm ↔ compute_sequencer ──
  logic seq_start_w, seq_done_w;
  logic [3:0] k_pass_w;

  // ── tile_fsm ↔ DMA (stub: instant for TB) ──
  logic dma_start_w, dma_is_write_w, dma_done_stub;
  assign dma_done_stub = dma_start_w;  // Instant DMA for standalone testing

  // ── tile_fsm ↔ PPU ──
  logic ppu_start_w, ppu_done_w;

  // ── tile_fsm ↔ swizzle ──
  logic swizzle_start_w, swizzle_done_w;

  // ── tile_fsm ↔ page control ──
  logic page_swap_w;

  // ── shadow_reg_file outputs (stable config) ──
  pe_mode_e      cfg_pe_mode;
  act_mode_e     cfg_activation;
  logic [9:0]    cfg_cin, cfg_cout, cfg_hin, cfg_win, cfg_hout, cfg_wout;
  logic [3:0]    cfg_kh, cfg_kw;
  logic [2:0]    cfg_stride, cfg_padding;
  logic [3:0]    cfg_num_cin_pass, cfg_num_k_pass;
  swizzle_mode_e cfg_swizzle;
  int8_t         cfg_zp_x;

  // ── compute_sequencer outputs (iteration counters) ──
  logic [9:0]    iter_h, iter_wblk, iter_cin, iter_cout_group;
  logic [3:0]    iter_kw, iter_kh_row;
  logic          seq_pe_enable, seq_pe_clear, seq_pe_acc_valid;
  logic          seq_ppu_trigger;
  logic [9:0]    seq_ppu_cout_base;
  logic          seq_pool_enable;
  logic          mp5_shift_en_w, mp5_win_flush_w;
  logic [9:0]    agi_iter_cin_mux_w, ago_iter_cout_grp_mux_w;
  logic [3:0]    agi_iter_kh_mux_w;
  logic [9:0]    iter_mp5_ch_w;

  // ── addr_gen_input outputs ──
  logic [1:0]    agi_bank_id;
  logic [11:0]   agi_sram_addr;
  logic          agi_is_padding;
  int8_t         agi_pad_value;

  // ── addr_gen_weight outputs ──
  logic [15:0]   agw_addr [PE_COLS];
  logic [1:0]    agw_bank_id;

  // ── addr_gen_output outputs ──
  logic [1:0]    ago_bank_id [PE_COLS];
  logic [11:0]   ago_addr [PE_COLS];

  // ── GLB data buses ──
  int8_t         in_bank_data  [3][LANES];     // 3 input banks read data
  int8_t         wt_bank_data  [3][PE_COLS][LANES]; // 3 weight banks × 4 read ports
  int32_t        out_psum_rd   [PE_COLS][LANES]; // 4 output banks PSUM read
  int8_t         out_act_rd    [PE_COLS][LANES]; // 4 output banks ACT read

  // ── Router outputs ──
  int8_t         routed_act    [PE_ROWS][LANES];
  int8_t         routed_wgt    [PE_ROWS][PE_COLS][LANES];
  int32_t        routed_psum_out [PE_COLS][LANES];
  logic          routed_psum_wr_en [PE_COLS];
  int8_t         bypass_out_w  [LANES];

  // ── Window gen outputs ──
  int8_t         win_taps [7][LANES];
  logic          win_taps_valid;
  // ★ PE_MP5: pack 5×5 INT8 window per lane (same-wblk horizontal neighbors + zp pad)
  int8_t         pool_packed [25][LANES];
  logic [3:0]    win_k_mux;

  // ── PE cluster outputs ──
  int32_t        pe_col_psum [PE_COLS][LANES];
  logic          pe_psum_valid;
  int8_t         pe_pool_max [LANES];
  logic          pe_pool_valid;

  // ── PPU outputs ──
  int8_t         ppu_act_out [PE_COLS][LANES];
  logic          ppu_act_valid [PE_COLS];

  assign dbg_k_pass      = k_pass_w;
  assign dbg_iter_mp5_ch = iter_mp5_ch_w;

  assign win_k_mux = (cfg_pe_mode == PE_MP5) ? 4'd5 : cfg_kw;

  // ★ Structural pack: 25 INT8 / lane from 5 row taps × 5 horizontal indices (zp pad at wblk edge)
  always_comb begin
    for (int kh = 0; kh < 5; kh++)
      for (int kw = 0; kw < 5; kw++)
        for (int ln = 0; ln < LANES; ln++) begin
          automatic int idx = ln + kw - 2;
          if (idx >= 0 && idx < LANES)
            pool_packed[5 * kh + kw][ln] = win_taps[kh][idx];
          else
            pool_packed[5 * kh + kw][ln] = cfg_zp_x;
        end
  end

  // ══════════════════════════════════════════════════════════════
  // 1. TILE FSM — Phase-level controller
  // ══════════════════════════════════════════════════════════════
  tile_fsm u_tile_fsm (
    .clk             (clk),
    .rst_n           (rst_n),
    .tile_valid      (tile_valid),
    .layer_desc      (layer_desc_in),
    .tile_desc       (tile_desc_in),
    .tile_accept     (tile_accept),
    .shadow_latch    (shadow_latch),
    .dma_start       (dma_start_w),
    .dma_is_write    (dma_is_write_w),
    .dma_done        (dma_done_stub),
    .seq_start       (seq_start_w),
    .seq_done        (seq_done_w),
    .ppu_start       (ppu_start_w),
    .ppu_done        (ppu_done_w),
    .swizzle_start   (swizzle_start_w),
    .swizzle_done    (swizzle_done_w),
    .barrier_wait_req(),
    .barrier_grant   (barrier_grant),
    .barrier_signal  (barrier_signal),
    .page_swap       (page_swap_w),
    .state           (fsm_state),
    .tile_done       (tile_done),
    .cur_k_pass_idx  (k_pass_w)
  );

  // ══════════════════════════════════════════════════════════════
  // 2. SHADOW REG FILE — Stable config latch
  // ══════════════════════════════════════════════════════════════
  shadow_reg_file u_shadow (
    .clk             (clk),
    .rst_n           (rst_n),
    .latch_en        (shadow_latch),
    .layer_desc_in   (layer_desc_in),
    .tile_desc_in    (tile_desc_in),
    .o_pe_mode       (cfg_pe_mode),
    .o_activation    (cfg_activation),
    .o_cin           (cfg_cin),
    .o_cout          (cfg_cout),
    .o_hin           (cfg_hin),
    .o_win           (cfg_win),
    .o_hout          (cfg_hout),
    .o_wout          (cfg_wout),
    .o_kh            (cfg_kh),
    .o_kw            (cfg_kw),
    .o_stride        (cfg_stride),
    .o_padding       (cfg_padding),
    .o_num_cin_pass  (cfg_num_cin_pass),
    .o_num_k_pass    (cfg_num_k_pass),
    .o_swizzle       (cfg_swizzle),
    .o_zp_x          (cfg_zp_x),
    .o_tile_id       (),
    .o_layer_id      (),
    .o_first_tile    (),
    .o_last_tile     (),
    .o_hold_skip     (),
    .o_need_swizzle  ()
  );

  // ══════════════════════════════════════════════════════════════
  // 3. COMPUTE SEQUENCER — Cycle-level iteration
  // ══════════════════════════════════════════════════════════════
  compute_sequencer u_sequencer (
    .clk             (clk),
    .rst_n           (rst_n),
    .seq_start       (seq_start_w),
    .seq_done        (seq_done_w),
    .cfg_pe_mode     (cfg_pe_mode),
    .cfg_cin         (cfg_cin),
    .cfg_cout        (cfg_cout),
    .cfg_hout        (cfg_hout),
    .cfg_wout        (cfg_wout),
    .cfg_kh          (cfg_kh),
    .cfg_kw          (cfg_kw),
    .cfg_stride      (cfg_stride),
    .iter_h          (iter_h),
    .iter_wblk       (iter_wblk),
    .iter_cin        (iter_cin),
    .iter_cout_group (iter_cout_group),
    .iter_kw         (iter_kw),
    .iter_kh_row     (iter_kh_row),
    .pe_enable       (seq_pe_enable),
    .pe_clear_acc    (seq_pe_clear),
    .pe_acc_valid    (seq_pe_acc_valid),
    .ppu_trigger     (seq_ppu_trigger),
    .ppu_cout_base       (seq_ppu_cout_base),
    .pool_enable         (seq_pool_enable),
    .mp5_shift_en        (mp5_shift_en_w),
    .mp5_win_flush       (mp5_win_flush_w),
    .agi_iter_cin_mux    (agi_iter_cin_mux_w),
    .agi_iter_kh_mux     (agi_iter_kh_mux_w),
    .ago_iter_cout_grp_mux(ago_iter_cout_grp_mux_w),
    .dbg_iter_mp5_ch     (iter_mp5_ch_w)
  );

  // ══════════════════════════════════════════════════════════════
  // 4. ADDRESS GENERATORS (driven by compute_sequencer)
  // ══════════════════════════════════════════════════════════════

  // ── 4a. Input address generator ──
  addr_gen_input #(.LANES(LANES)) u_agi (
    .clk          (clk),
    .rst_n        (rst_n),
    .cfg_hin      (cfg_hin),
    .cfg_win      (cfg_win),
    .cfg_cin      (cfg_cin),
    .cfg_stride   (cfg_stride),
    .cfg_padding  (cfg_padding),
    .cfg_zp_x     (cfg_zp_x),       // ★ RULE 5: padding = zp_x
    .iter_h_out   (iter_h),
    .iter_wblk    (iter_wblk),
    .iter_cin     (agi_iter_cin_mux_w),
    .iter_kh_row  (agi_iter_kh_mux_w),
    .bank_id      (agi_bank_id),
    .sram_addr    (agi_sram_addr),
    .is_padding   (agi_is_padding),
    .pad_value    (agi_pad_value)
  );

  // ── 4b. Weight address generator (★ 4 per-column addresses) ──
  addr_gen_weight #(.LANES(LANES), .PE_COLS(PE_COLS)) u_agw (
    .clk             (clk),
    .rst_n           (rst_n),
    .cfg_pe_mode     (cfg_pe_mode),
    .cfg_cin         (cfg_cin),
    .cfg_cout        (cfg_cout),
    .cfg_kw          (cfg_kw),
    .iter_cin        (iter_cin),
    .iter_cout_group (iter_cout_group),
    .iter_kw         (iter_kw),
    .iter_kh_row     (iter_kh_row),
    .wgt_addr        (agw_addr),        // ★ 4 DIFFERENT addresses
    .wgt_bank_id     (agw_bank_id)
  );

  // ── 4c. Output address generator ──
  addr_gen_output #(.LANES(LANES), .PE_COLS(PE_COLS)) u_ago (
    .clk             (clk),
    .rst_n           (rst_n),
    .cfg_wout        (cfg_wout),
    .cfg_cout        (cfg_cout),
    .iter_h_out      (iter_h),
    .iter_wblk       (iter_wblk),
    .iter_cout_group (ago_iter_cout_grp_mux_w),
    .out_bank_id     (ago_bank_id),
    .out_addr        (ago_addr)
  );

  // ══════════════════════════════════════════════════════════════
  // 5. GLB INPUT BANKS ×3 (double-buffered)
  // ══════════════════════════════════════════════════════════════
  genvar bi;
  generate
    for (bi = 0; bi < 3; bi++) begin : gen_in_bank
      glb_input_bank_db #(.LANES(LANES)) u_in_bank (
        .clk         (clk),
        .rst_n       (rst_n),
        .page_swap   (page_swap_w),
        // Compute read: selected by addr_gen_input bank_id
        .rd_addr     ((agi_bank_id == bi[1:0]) ? agi_sram_addr : 12'd0),
        .rd_data     (in_bank_data[bi]),
        // DMA write: from external port
        .wr_addr     (ext_wr_addr),
        .wr_data     (ext_wr_data),
        .wr_en       (ext_wr_en && (ext_wr_target == 2'd0) && (ext_wr_bank_id == bi[1:0])),
        .wr_lane_mask(ext_wr_mask)
      );
    end
  endgenerate

  // ══════════════════════════════════════════════════════════════
  // 6. GLB WEIGHT BANKS ×3 (4 read ports each)
  // ══════════════════════════════════════════════════════════════
  genvar bw;
  generate
    for (bw = 0; bw < 3; bw++) begin : gen_wt_bank
      // 4 read addresses: 1 per PE column (★ per-column weight)
      logic [$clog2(GLB_WEIGHT_DEPTH)-1:0] wt_rd_addr_col [PE_COLS];

      always_comb begin
        for (int c = 0; c < PE_COLS; c++)
          wt_rd_addr_col[c] = (agw_bank_id == bw[1:0]) ? agw_addr[c][$clog2(GLB_WEIGHT_DEPTH)-1:0] : '0;
      end

      glb_weight_bank #(.LANES(LANES)) u_wt_bank (
        .clk         (clk),
        .rst_n       (rst_n),
        .rd_addr     (wt_rd_addr_col),       // ★ 4 DIFFERENT addresses
        .rd_data     (wt_bank_data[bw]),     // [PE_COLS][LANES] output
        // DMA write
        .wr_addr     (ext_wr_addr[$clog2(GLB_WEIGHT_DEPTH)-1:0]),
        .wr_data     (ext_wr_data),
        .wr_en       (ext_wr_en && (ext_wr_target == 2'd1) && (ext_wr_bank_id == bw[1:0]))
      );
    end
  endgenerate

  // ══════════════════════════════════════════════════════════════
  // 7. GLB OUTPUT BANKS ×4 (dual namespace PSUM/ACT)
  // ══════════════════════════════════════════════════════════════
  int8_t glb_act_wr_data [PE_COLS][LANES];
  logic  glb_act_wr_en   [PE_COLS];

  always_comb begin
    for (int b = 0; b < PE_COLS; b++) begin
      glb_act_wr_en[b] = ppu_act_valid[b]
        || ((cfg_pe_mode == PE_MP5) && pe_pool_valid && (iter_mp5_ch_w[1:0] == b[1:0]));
      for (int l = 0; l < LANES; l++)
        glb_act_wr_data[b][l] = ((cfg_pe_mode == PE_MP5) && pe_pool_valid
                                && (iter_mp5_ch_w[1:0] == b[1:0]))
                                ? pe_pool_max[l] : ppu_act_out[b][l];
    end
  end

  genvar bo;
  generate
    for (bo = 0; bo < PE_COLS; bo++) begin : gen_out_bank
      glb_output_bank #(.LANES(LANES)) u_out_bank (
        .clk         (clk),
        .rst_n       (rst_n),
        // PSUM read/write (multipass accumulation)
        .psum_addr   (ago_addr[bo][$clog2(GLB_OUTPUT_DEPTH)-1:0]),
        .psum_wr_data(pe_col_psum[bo]),
        .psum_wr_en  (pe_psum_valid && (cfg_pe_mode != PE_MP5)
                      && (ago_bank_id[bo] == bo[1:0])),
        .psum_rd_data(out_psum_rd[bo]),
        .psum_rd_en  (1'b1),
        // ACT: PPU or PE_MP5 comparator path (no PPU in MP5)
        .act_addr    (ago_addr[bo][$clog2(GLB_OUTPUT_DEPTH)-1:0]),
        .act_wr_data (glb_act_wr_data[bo]),
        .act_wr_en   (glb_act_wr_en[bo]),
        .act_rd_data (out_act_rd[bo]),
        .act_rd_en   (ext_rd_en && (ext_rd_bank_id == bo[1:0])),
        // DMA drain
        .drain_addr  (ext_rd_addr[$clog2(GLB_OUTPUT_DEPTH)-1:0]),
        .drain_data  ()
      );
    end
  endgenerate

  // External read output mux
  assign ext_rd_act_data  = out_act_rd[ext_rd_bank_id];
  assign ext_rd_psum_data = out_psum_rd[ext_rd_bank_id];

  // ══════════════════════════════════════════════════════════════
  // 8. METADATA RAM
  // ══════════════════════════════════════════════════════════════
  metadata_ram u_meta (
    .clk         (clk),
    .rst_n       (rst_n),
    .clear_all   (!rst_n),
    .set_valid   (1'b0),
    .set_slot_id (4'd0),
    .set_meta    (32'd0),
    .query_slot_id(4'd0),
    .query_valid (),
    .query_meta  (),
    .advance_ring(1'b0),
    .ring_head   (),
    .ring_tail   (),
    .ring_full   (),
    .ring_empty  ()
  );

  // ══════════════════════════════════════════════════════════════
  // 9. ROUTER CLUSTER V2 (★ Eyeriss per-column weight routing)
  // ══════════════════════════════════════════════════════════════
  router_cluster_v2 #(.LANES(LANES), .PE_ROWS(PE_ROWS), .PE_COLS(PE_COLS)) u_router (
    .clk           (clk),
    .rst_n         (rst_n),
    .cfg_pe_mode   (cfg_pe_mode),
    .rin_bank_sel  (agi_bank_id),
    // RIN: input banks → PE activation (multicast)
    .glb_in_data   (in_bank_data),
    .pe_act        (routed_act),
    // RWT: weight banks → PE weight (★ per-column)
    .glb_wgt_data  (wt_bank_data),
    .pe_wgt        (routed_wgt),
    // RPS: PE psum → output banks
    .pe_psum_in    (pe_col_psum),
    .psum_valid    (pe_psum_valid),
    .glb_out_psum  (routed_psum_out),
    .glb_out_wr_en (routed_psum_wr_en),
    // Bypass
    .bypass_in     (in_bank_data[0]),
    .bypass_out    (bypass_out_w),
    .bypass_en     (cfg_pe_mode == PE_PASS)
  );

  // ══════════════════════════════════════════════════════════════
  // 10. WINDOW GENERATOR
  // ══════════════════════════════════════════════════════════════
  // ★ Instance: window_gen — K=5 for MP5 (SPPF), else cfg_kw; shift on MAC or MP5 row scan
  window_gen #(.LANES(LANES)) u_window (
    .clk             (clk),
    .rst_n           (rst_n),
    .cfg_k           (win_k_mux),
    .shift_in_valid  (seq_pe_enable | mp5_shift_en_w),
    .shift_in        (routed_act[0]),   // Row 0 activation feeds window
    .taps            (win_taps),
    .taps_valid      (win_taps_valid),
    .flush           (seq_pe_clear | mp5_win_flush_w)
  );

  // ══════════════════════════════════════════════════════════════
  // 11. PE CLUSTER V4 (★ 3×4×20 with per-column weight)
  // ══════════════════════════════════════════════════════════════
  pe_cluster_v4 #(.LANES(LANES), .PE_ROWS(PE_ROWS), .PE_COLS(PE_COLS)) u_pe_cluster (
    .clk           (clk),
    .rst_n         (rst_n),
    .pe_mode       (cfg_pe_mode),
    .pe_enable     (seq_pe_enable),
    .pe_clear_acc  (seq_pe_clear),
    // Activation: from router (SAME for all columns)
    .act_taps      (routed_act),
    // Weight: from router (★ DIFFERENT per column)
    .wgt_data      (routed_wgt),
    // PSUM output
    .col_psum      (pe_col_psum),
    .psum_valid    (pe_psum_valid),
    // Multi-pass accumulation (★ RULE 8: DW7 pass 2+ adds prior PSUM)
    .psum_accum_in (out_psum_rd),
    .psum_accum_en ((cfg_pe_mode == PE_DW7) && (k_pass_w > 4'd0)),
    // MaxPool 5×5 packed window → comparator_tree
    .pool_window   (pool_packed),
    .pool_enable   (seq_pool_enable),
    .pool_max      (pe_pool_max),
    .pool_valid    (pe_pool_valid)
  );

  // ══════════════════════════════════════════════════════════════
  // 12. PPU ×4 (★ 4 parallel PPUs, 1 per PE column)
  // ══════════════════════════════════════════════════════════════
  genvar pp;
  generate
    for (pp = 0; pp < PE_COLS; pp++) begin : gen_ppu
      // Select bias/quant params for this column's cout
      // cout_index = ppu_cout_base + pp (column offset)
      logic [9:0] this_cout;
      assign this_cout = seq_ppu_cout_base + pp[9:0];

      ppu #(.LANES(LANES)) u_ppu (
        .clk         (clk),
        .rst_n       (rst_n),
        .psum_in     (pe_col_psum[pp]),           // This column's PSUM
        .psum_valid  (seq_ppu_trigger),
        .bias_val    (bias_table[this_cout]),      // ★ Per-cout bias [RULE 6]
        .m_int       (m_int_table[this_cout]),     // ★ Per-cout M_int
        .shift_val   (shift_table[this_cout]),     // ★ Per-cout shift
        .zp_out      (zp_out_table[this_cout]),    // ★ Per-cout ZP_out
        .activation  (cfg_activation),             // ReLU or NONE [RULE 4]
        .act_out     (ppu_act_out[pp]),
        .act_valid   (ppu_act_valid[pp])
      );
    end
  endgenerate

  // PPU done: any column signals valid (all same timing)
  assign ppu_done_w = ppu_act_valid[0];

  // ══════════════════════════════════════════════════════════════
  // 13. SWIZZLE ENGINE
  // ══════════════════════════════════════════════════════════════
  swizzle_engine #(.LANES(LANES)) u_swizzle (
    .clk           (clk),
    .rst_n         (rst_n),
    .start         (swizzle_start_w),
    .cfg_mode      (cfg_swizzle),
    .cfg_src_h     (cfg_hout),
    .cfg_src_w     (cfg_wout),
    .cfg_src_c     (cfg_cout),
    .cfg_dst_h     (cfg_hin),
    .cfg_dst_w     (cfg_win),
    .src_rd_en     (),
    .src_rd_addr   (),
    .src_rd_data   (out_act_rd[0]),
    .dst_wr_en     (),
    .dst_wr_addr   (),
    .dst_wr_data   (),
    .dst_wr_mask   (),
    .done          (swizzle_done_w)
  );

endmodule
