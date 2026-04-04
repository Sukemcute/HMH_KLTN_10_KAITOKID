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

  // ★ Stage 8 checkpoints
  output logic [3:0]    dbg_k_pass,
  output logic [9:0]    dbg_iter_mp5_ch,
  output logic          dbg_seq_ppu_trigger,
  output logic          dbg_ppu_trigger_delayed,
  output logic          dbg_ppu_act_valid_0,
  output logic          dbg_pe_psum_valid,
  output logic          dbg_glb_act_wr_en_0,
  output logic          dbg_pe_pool_valid,
  output logic          dbg_seq_pe_enable,
  output logic [3:0]    dbg_fsm_mode_reg,
  output logic          dbg_ppu_done_latch,
  output logic [9:0]    dbg_seq_ppu_cout_base,
  output int32_t        dbg_pe_col_psum_0_0,
  output int8_t         dbg_ppu_act_out_0_0
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
  logic          seq_window_flush_w;
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
  logic [11:0]   agi_sram_addr_row [3];
  logic          agi_is_padding;
  int8_t         agi_pad_value;
  logic [1:0]    agi_bank_id_row [3];
  logic          agi_is_padding_row [3];
  logic [11:0]   agi_bank_rd_addr [3];
  logic [11:0]   agi_bank_rd_addr_col [3][PE_COLS];

  // ── addr_gen_weight outputs ──
  logic [15:0]   agw_addr [PE_COLS];
  logic [1:0]    agw_bank_id;

  // ── addr_gen_output outputs ──
  logic [1:0]    ago_bank_id [PE_COLS];
  logic [11:0]   ago_addr [PE_COLS];

  // ── GLB data buses ──
  int8_t         in_bank_data  [3][PE_COLS][LANES]; // ★ 4 read ports / bank
  int8_t         wt_bank_data  [3][PE_COLS][LANES]; // 3 weight banks × 4 read ports
  int32_t        out_psum_rd   [PE_COLS][LANES]; // 4 output banks PSUM read
  int8_t         out_act_rd    [PE_COLS][LANES]; // 4 output banks ACT read

  // ── Router outputs ──
  int8_t         routed_act    [PE_ROWS][PE_COLS][LANES];
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

  assign dbg_k_pass              = k_pass_w;
  assign dbg_iter_mp5_ch         = iter_mp5_ch_w;

  assign win_k_mux = (cfg_pe_mode == PE_MP5) ? 4'd5 : cfg_kw;

  // ★ Structural pack: 25 INT8 / lane from 5 row taps × 5 horizontal indices
  // Golden: kh=0 top of window; window_gen sr[0]=newest row → map taps[4-kh].
  always_comb begin
    for (int kh = 0; kh < 5; kh++)
      for (int kw = 0; kw < 5; kw++)
        for (int ln = 0; ln < LANES; ln++) begin
          automatic int idx = ln + kw - 2;
          if (idx >= 0 && idx < LANES)
            pool_packed[5 * kh + kw][ln] = win_taps[4 - kh][idx];
          else
            pool_packed[5 * kh + kw][ln] = -8'sd128;
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
    .cur_k_pass_idx  (k_pass_w),
    .dbg_mode_reg    (dbg_fsm_mode_reg),
    .dbg_ppu_done_latch(dbg_ppu_done_latch)
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
    .dbg_iter_mp5_ch     (iter_mp5_ch_w),
    .seq_window_flush    (seq_window_flush_w)
  );

  // ══════════════════════════════════════════════════════════════
  // 4. ADDRESS GENERATORS (driven by compute_sequencer)
  // ══════════════════════════════════════════════════════════════

  // ── 4a. Input address generator ──
  addr_gen_input #(.LANES(LANES)) u_agi (
    .clk          (clk),
    .rst_n        (rst_n),
    .cfg_pe_mode  (cfg_pe_mode),
    .cfg_hin      (cfg_hin),
    .cfg_win      (cfg_win),
    .cfg_cin      (cfg_cin),
    .cfg_stride   (cfg_stride),
    .cfg_padding  (cfg_padding),
    .cfg_zp_x     (cfg_zp_x),
    .iter_h_out   (iter_h),
    .iter_wblk    (iter_wblk),
    .iter_cin     (agi_iter_cin_mux_w),
    .iter_cout_group(iter_cout_group),
    .iter_kh_row  (agi_iter_kh_mux_w),
    .bank_id      (agi_bank_id),
    .sram_addr    (agi_sram_addr),
    .sram_addr_row(agi_sram_addr_row),
    .is_padding   (agi_is_padding),
    .pad_value    (agi_pad_value),
    .bank_id_row    (agi_bank_id_row),
    .is_padding_row (agi_is_padding_row),
    .bank_rd_addr   (agi_bank_rd_addr),
    .bank_rd_addr_col(agi_bank_rd_addr_col)
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
  logic use_parallel_fetch;
  assign use_parallel_fetch = (cfg_pe_mode == PE_RS3) || (cfg_pe_mode == PE_DW3)
                              || (cfg_pe_mode == PE_DW7);

  genvar bi;
  generate
    for (bi = 0; bi < 3; bi++) begin : gen_in_bank
      logic [$clog2(GLB_INPUT_DEPTH)-1:0] in_rd_addr_mux [PE_COLS];

      always_comb begin
        for (int cc = 0; cc < PE_COLS; cc++) begin
          if (use_parallel_fetch)
            in_rd_addr_mux[cc] = agi_bank_rd_addr_col[bi][cc][$clog2(GLB_INPUT_DEPTH)-1:0];
          else if (agi_bank_id == bi[1:0])
            in_rd_addr_mux[cc] = agi_sram_addr[$clog2(GLB_INPUT_DEPTH)-1:0];
          else
            in_rd_addr_mux[cc] = '0;
        end
      end

      glb_input_bank_db #(
        .LANES        (LANES),
        .DEPTH        (GLB_INPUT_DEPTH),
        .N_READ_PORTS (PE_COLS)
      ) u_in_bank (
        .clk         (clk),
        .rst_n       (rst_n),
        .page_swap   (page_swap_w),
        .rd_addr     (in_rd_addr_mux),
        .rd_data     (in_bank_data[bi]),
        .wr_addr     (ext_wr_addr),
        .wr_data     (ext_wr_data),
        .wr_en       (ext_wr_en && (ext_wr_target == 2'd0) && (ext_wr_bank_id == bi[1:0])),
        .wr_lane_mask(ext_wr_mask)
      );
    end
  endgenerate

  // ★ Delay pe_enable/pe_clear 1 cycle for RS3/DW3/DW7 (use_parallel_fetch).
  //   glb_input_bank_db rd_data updates 1 cycle after rd_addr; when read ports
  //   1..3 change address but port0 does not, ports 1..3 still hold previous
  //   data at the same posedge pe_enable samples — MAC sees wrong activations
  //   for columns >0. Enabling MAC on the following cycle aligns all ports.
  logic seq_pe_enable_d1;
  logic seq_pe_clear_d1;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      seq_pe_enable_d1 <= 1'b0;
      seq_pe_clear_d1  <= 1'b0;
    end else begin
      seq_pe_enable_d1 <= seq_pe_enable;
      seq_pe_clear_d1  <= seq_pe_clear;
    end
  end

  logic seq_pe_enable_mac;
  logic seq_pe_clear_mac;
  assign seq_pe_enable_mac = use_parallel_fetch ? seq_pe_enable_d1 : seq_pe_enable;
  assign seq_pe_clear_mac  = use_parallel_fetch ? seq_pe_clear_d1  : seq_pe_clear;

  logic mp5_shift_en_d1;
  logic win_shift_in_valid;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      mp5_shift_en_d1 <= 1'b0;
    else
      mp5_shift_en_d1 <= mp5_shift_en_w;
  end
  assign win_shift_in_valid = (cfg_pe_mode == PE_MP5) ? mp5_shift_en_d1
                             : (seq_pe_enable_mac | mp5_shift_en_w);

  // ★ Crossbar + padding: per (row r, col c) activation for router
  int8_t routed_in_data [3][PE_COLS][LANES];
  always_comb begin
    for (int r = 0; r < PE_ROWS; r++) begin
      for (int c = 0; c < PE_COLS; c++) begin
        for (int ln = 0; ln < LANES; ln++) begin
          if (use_parallel_fetch && agi_is_padding_row[r])
            routed_in_data[r][c][ln] = agi_pad_value;
          else if (use_parallel_fetch)
            routed_in_data[r][c][ln] = in_bank_data[agi_bank_id_row[r]][c][ln];
          else if (cfg_pe_mode == PE_OS1 || cfg_pe_mode == PE_GEMM) begin
            routed_in_data[r][c][ln] = agi_is_padding ? agi_pad_value
                                  : in_bank_data[agi_bank_id][c][ln];
          end else if (cfg_pe_mode == PE_MP5) begin
            // Single-bank stream: use read port 0 only (matches router PE_MP5 path). All
            // rd_addr[*] are equal when active, but duplicate rd_reg per port can diverge
            // in sim if addresses ever differ by a delta — port 0 is the canonical stream.
            routed_in_data[r][c][ln] = agi_is_padding ? -8'sd128
                                  : in_bank_data[agi_bank_id][0][ln];
          end else begin
            routed_in_data[r][c][ln] = in_bank_data[r][c][ln];
          end
        end
      end
    end
  end

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
          wt_rd_addr_col[c] = (use_parallel_fetch || (agw_bank_id == bw[1:0]))
                             ? agw_addr[c][$clog2(GLB_WEIGHT_DEPTH)-1:0] : '0;
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

  // Forward-declared: PPU ACT address delay pipeline (driven in section 12)
  // ★ Must match trigger→act_valid latency exactly:
  //   seq_ppu_trigger → [DSP_PIPE_DEPTH] → ppu_trigger_delayed → [PPU valid_pipe]
  //   → act_valid  = 5 + 4 = 9 cycles (see ppu.sv: valid_pipe[4] is 4 regs after psum_valid).
  // Shift depth from sr[0] to sr[tail] is (ACT_ADDR_DELAY-1); need tail delay = 9 → ACT_ADDR_DELAY=10.
  localparam int ACT_ADDR_DELAY = DSP_PIPE_DEPTH + PPU_PIPE_DEPTH;
  logic [$clog2(GLB_OUTPUT_DEPTH)-1:0] ppu_act_addr_sr [PE_COLS][ACT_ADDR_DELAY];
  logic [$clog2(GLB_OUTPUT_DEPTH)-1:0] ppu_act_addr_delayed;
  assign ppu_act_addr_delayed = ppu_act_addr_sr[0][ACT_ADDR_DELAY-1];

  // ★ Registered PPU→ACT path: capture addr/data when act_valid, assert wr_en next cycle.
  // Avoids combinational mux (ppu_act_valid ? delayed : ago) racing glb_output_bank always_ff.
  logic  ppu_act_wr_en_d [PE_COLS];
  logic [$clog2(GLB_OUTPUT_DEPTH)-1:0] ppu_act_wr_addr_hold [PE_COLS];
  int8_t ppu_act_wr_data_hold [PE_COLS][LANES];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int b = 0; b < PE_COLS; b++) begin
        ppu_act_wr_en_d[b] <= 1'b0;
        ppu_act_wr_addr_hold[b] <= '0;
        for (int l = 0; l < LANES; l++)
          ppu_act_wr_data_hold[b][l] <= 8'sd0;
      end
    end else begin
      for (int b = 0; b < PE_COLS; b++) begin
        ppu_act_wr_en_d[b] <= ppu_act_valid[b];
        if (ppu_act_valid[b]) begin
          ppu_act_wr_addr_hold[b] <= ppu_act_addr_delayed;
          for (int l = 0; l < LANES; l++)
            ppu_act_wr_data_hold[b][l] <= ppu_act_out[b][l];
        end
      end
    end
  end

  int8_t glb_act_wr_data [PE_COLS][LANES];
  logic  glb_act_wr_en   [PE_COLS];

  always_comb begin
    for (int b = 0; b < PE_COLS; b++) begin
      glb_act_wr_en[b] = ppu_act_wr_en_d[b]
        || ((cfg_pe_mode == PE_MP5) && pe_pool_valid && (iter_mp5_ch_w[1:0] == b[1:0]))
        || (ext_wr_en && (ext_wr_target == 2'd2) && (ext_wr_bank_id == b[1:0]));
      for (int l = 0; l < LANES; l++)
        glb_act_wr_data[b][l] =
          (ext_wr_en && (ext_wr_target == 2'd2) && (ext_wr_bank_id == b[1:0]))
            ? ext_wr_data[l]
            : (((cfg_pe_mode == PE_MP5) && pe_pool_valid
                && (iter_mp5_ch_w[1:0] == b[1:0]))
               ? pe_pool_max[l] : ppu_act_wr_data_hold[b][l]);
    end
  end

  genvar bo;
  generate
    for (bo = 0; bo < PE_COLS; bo++) begin : gen_out_bank
      logic [$clog2(GLB_OUTPUT_DEPTH)-1:0] out_act_addr_mux;
      always_comb begin
        if (ext_wr_en && (ext_wr_target == 2'd2) && (ext_wr_bank_id == bo[1:0]))
          out_act_addr_mux = ext_wr_addr[$clog2(GLB_OUTPUT_DEPTH)-1:0];
        else if (ext_rd_en && (ext_rd_bank_id == bo[1:0]))
          out_act_addr_mux = ext_rd_addr[$clog2(GLB_OUTPUT_DEPTH)-1:0];
        else if (ppu_act_wr_en_d[bo])
          out_act_addr_mux = ppu_act_wr_addr_hold[bo];
        else
          out_act_addr_mux = ago_addr[bo][$clog2(GLB_OUTPUT_DEPTH)-1:0];
      end

      glb_output_bank #(.LANES(LANES)) u_out_bank (
        .clk         (clk),
        .rst_n       (rst_n),
        .psum_addr   (ago_addr[bo][$clog2(GLB_OUTPUT_DEPTH)-1:0]),
        .psum_wr_data(pe_col_psum[bo]),
        .psum_wr_en  (pe_psum_valid && (cfg_pe_mode == PE_DW7)
                      && (ago_bank_id[bo] == bo[1:0])),
        .psum_rd_data(out_psum_rd[bo]),
        .psum_rd_en  (1'b1),
        .act_addr    (out_act_addr_mux),
        .act_wr_data (glb_act_wr_data[bo]),
        .act_wr_en   (glb_act_wr_en[bo]),
        .act_rd_data (out_act_rd[bo]),
        .act_rd_en   (ext_rd_en && (ext_rd_bank_id == bo[1:0])),
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
    .clear_all   (1'b0),
    .set_valid   (1'b0),
    .set_slot_id (4'd0),
    .set_meta    (32'd0),
    .query_slot_id(4'd0),
    .query_valid (),
    .query_meta  (),
    .advance_ring(1'b0),
    .head_ptr    (),
    .tail_ptr    (),
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
    // RIN: input banks → PE activation (with crossbar + padding for parallel kh)
    .glb_in_data   (routed_in_data),
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
    .bypass_in     (in_bank_data[0][0]),
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
    .shift_in_valid  (win_shift_in_valid),
    .shift_in        (routed_act[0][0]),
    .taps            (win_taps),
    .taps_valid      (win_taps_valid),
    .flush           (seq_window_flush_w | mp5_win_flush_w)
  );

  // ══════════════════════════════════════════════════════════════
  // 11. PE CLUSTER V4 (★ 3×4×20 with per-column weight)
  // ══════════════════════════════════════════════════════════════
  pe_cluster_v4 #(.LANES(LANES), .PE_ROWS(PE_ROWS), .PE_COLS(PE_COLS)) u_pe_cluster (
    .clk           (clk),
    .rst_n         (rst_n),
    .pe_mode       (cfg_pe_mode),
    .pe_enable     (seq_pe_enable_mac),
    .pe_clear_acc  (seq_pe_clear_mac),
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
  //
  // PPU trigger is DELAYED from seq_ppu_trigger by DSP_PIPE_DEPTH cycles.
  // Reason: seq_ppu_trigger fires at SEQ_ACC_DONE, but the PE pipeline
  // (DSP_PIPE_DEPTH=5 stages) has not fully drained yet. The accumulator
  // needs 5+ more cycles to incorporate all in-flight products.
  // Without this delay, the PPU reads an incomplete PSUM (often zero).
  // ══════════════════════════════════════════════════════════════
  logic [DSP_PIPE_DEPTH:0] ppu_trigger_delay_sr;
  logic [9:0] ppu_cout_base_sr [DSP_PIPE_DEPTH+1];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ppu_trigger_delay_sr <= '0;
      for (int i = 0; i <= DSP_PIPE_DEPTH; i++)
        ppu_cout_base_sr[i] <= 10'd0;
      for (int c = 0; c < PE_COLS; c++)
        for (int i = 0; i < ACT_ADDR_DELAY; i++)
          ppu_act_addr_sr[c][i] <= '0;
    end else begin
      ppu_trigger_delay_sr <= {ppu_trigger_delay_sr[DSP_PIPE_DEPTH-1:0], seq_ppu_trigger};
      ppu_cout_base_sr[0] <= seq_ppu_cout_base;
      for (int i = 1; i <= DSP_PIPE_DEPTH; i++)
        ppu_cout_base_sr[i] <= ppu_cout_base_sr[i-1];
      for (int c = 0; c < PE_COLS; c++) begin
        ppu_act_addr_sr[c][0] <= ago_addr[c][$clog2(GLB_OUTPUT_DEPTH)-1:0];
        for (int i = 1; i < ACT_ADDR_DELAY; i++)
          ppu_act_addr_sr[c][i] <= ppu_act_addr_sr[c][i-1];
      end
    end
  end
  logic ppu_trigger_delayed;
  assign ppu_trigger_delayed = ppu_trigger_delay_sr[DSP_PIPE_DEPTH];
  logic [9:0] ppu_cout_base_delayed;
  assign ppu_cout_base_delayed = ppu_cout_base_sr[DSP_PIPE_DEPTH];

  genvar pp;
  generate
    for (pp = 0; pp < PE_COLS; pp++) begin : gen_ppu
      logic [9:0] this_cout;
      assign this_cout = ppu_cout_base_delayed + pp[9:0];

      ppu #(.LANES(LANES)) u_ppu (
        .clk         (clk),
        .rst_n       (rst_n),
        .psum_in     (pe_col_psum[pp]),
        .psum_valid  (ppu_trigger_delayed),
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
  // Align FSM "PPU done" with ACT SRAM write pulse (1 cyc after act_valid)
  assign ppu_done_w = ppu_act_wr_en_d[0];

  // ══════════════════════════════════════════════════════════════
  // 13. SWIZZLE ENGINE
  // ══════════════════════════════════════════════════════════════
  // ── Domain alignment parameters (loaded by DMA alongside quant params) ──
  uint32_t      align_m_a_reg,  align_m_b_reg;
  logic [7:0]   align_sh_a_reg, align_sh_b_reg;
  int8_t        align_zp_a_reg, align_zp_b_reg, align_zp_out_reg;
  logic         align_bypass_reg;

  // Source B read signals (from skip buffer via output bank 1 or dedicated port)
  logic         swz_b_rd_en;
  logic [11:0]  swz_b_rd_addr;

  swizzle_engine #(.LANES(LANES)) u_swizzle (
    .clk              (clk),
    .rst_n            (rst_n),
    .start            (swizzle_start_w),
    .cfg_mode         (cfg_swizzle),
    .cfg_src_h        (cfg_hout),
    .cfg_src_w        (cfg_wout),
    .cfg_src_c        (cfg_cout),
    .cfg_dst_h        (cfg_hin),
    .cfg_dst_w        (cfg_win),
    // Domain alignment params
    .cfg_align_m_a    (align_m_a_reg),
    .cfg_align_sh_a   (align_sh_a_reg),
    .cfg_align_zp_a   (align_zp_a_reg),
    .cfg_align_m_b    (align_m_b_reg),
    .cfg_align_sh_b   (align_sh_b_reg),
    .cfg_align_zp_b   (align_zp_b_reg),
    .cfg_align_zp_out (align_zp_out_reg),
    .cfg_align_bypass (align_bypass_reg),
    // Source A (from output bank 0 ACT namespace)
    .src_rd_en        (),
    .src_rd_addr      (),
    .src_rd_data      (out_act_rd[0]),
    // Source B (from output bank 1 ACT namespace — skip buffer)
    .src_b_rd_en      (swz_b_rd_en),
    .src_b_rd_addr    (swz_b_rd_addr),
    .src_b_rd_data    (out_act_rd[1]),
    // Destination
    .dst_wr_en        (),
    .dst_wr_addr      (),
    .dst_wr_data      (),
    .dst_wr_mask      (),
    .done             (swizzle_done_w)
  );

  // ══════════════════════════════════════════════════════════════
  // DEBUG ASSIGNS (placed at end so all signals are in scope)
  // ══════════════════════════════════════════════════════════════
  assign dbg_seq_ppu_trigger     = seq_ppu_trigger;
  assign dbg_ppu_trigger_delayed = ppu_trigger_delayed;
  assign dbg_ppu_act_valid_0     = ppu_act_valid[0];
  assign dbg_pe_psum_valid       = pe_psum_valid;
  assign dbg_glb_act_wr_en_0     = glb_act_wr_en[0];
  assign dbg_pe_pool_valid       = pe_pool_valid;
  assign dbg_seq_pe_enable       = seq_pe_enable;
  assign dbg_seq_ppu_cout_base   = seq_ppu_cout_base;
  assign dbg_pe_col_psum_0_0     = pe_col_psum[0][0];
  assign dbg_ppu_act_out_0_0     = ppu_act_out[0][0];

  // synthesis translate_off
`ifdef S8_DBG
  always @(posedge clk) begin
    if (rst_n) begin
      // ── CP-GLB: Input/Weight bank data at the router input ──
      if (seq_pe_enable) begin
        $display("  [GLB-IN] %0t  bank0d[0]=%0d bank1d[0]=%0d bank2d[0]=%0d  padded=%b padval=%0d",
                 $time,
                 in_bank_data[0][0][0], in_bank_data[1][0][0], in_bank_data[2][0][0],
                 agi_is_padding, agi_pad_value);
        $display("  [GLB-WT] %0t  bk0c0[0]=%0d bk0c1[0]=%0d bk1c0[0]=%0d bk2c0[0]=%0d",
                 $time,
                 wt_bank_data[0][0][0], wt_bank_data[0][1][0],
                 wt_bank_data[1][0][0], wt_bank_data[2][0][0]);
        $display("  [ROUTE] %0t  act[0][0]=%0d act[1][0]=%0d act[2][0]=%0d  wgt[0][0][0]=%0d wgt[1][0][0]=%0d wgt[2][0][0]=%0d",
                 $time,
                 routed_act[0][0][0], routed_act[1][0][0], routed_act[2][0][0],
                 routed_wgt[0][0][0], routed_wgt[1][0][0], routed_wgt[2][0][0]);
        $display("  [ADDR] %0t  agi_row0=%0d agi_row1=%0d agi_row2=%0d  agw_bk=%0d agw_c0=%0d agw_c1=%0d",
                 $time,
                 agi_sram_addr_row[0], agi_sram_addr_row[1], agi_sram_addr_row[2],
                 agw_bank_id, agw_addr[0], agw_addr[1]);
      end

      // ── CP-PSUM: PSUM output just before PPU reads ──
      if (ppu_trigger_delayed)
        $display("  [PSUM] %0t  col0[0]=%0d col0[1]=%0d  col1[0]=%0d  col2[0]=%0d  col3[0]=%0d",
                 $time,
                 pe_col_psum[0][0], pe_col_psum[0][1],
                 pe_col_psum[1][0], pe_col_psum[2][0], pe_col_psum[3][0]);

      // ── CP-ACT: registered ACT write (same cycle as glb wr_en to SRAM) ──
      if (ppu_act_wr_en_d[0])
        $display("  [ACT-WR] %0t  act0[0]=%0d act0[1]=%0d  wr_en=%b%b%b%b  addr_hold=%0d",
                 $time,
                 ppu_act_wr_data_hold[0][0], ppu_act_wr_data_hold[0][1],
                 glb_act_wr_en[0], glb_act_wr_en[1], glb_act_wr_en[2], glb_act_wr_en[3],
                 ppu_act_wr_addr_hold[0]);

      // ── CP-MP5: MaxPool path ──
      if (pe_pool_valid)
        $display("  [MP5] %0t pool_max[0]=%0d pool_max[1]=%0d  ch=%0d bank=%0d",
                 $time,
                 pe_pool_max[0], pe_pool_max[1],
                 iter_mp5_ch_w, iter_mp5_ch_w[1:0]);

      // ── CP-CFG: Config readback on shadow latch ──
      if (shadow_latch)
        $display("  [CFG] %0t  mode=%0d cin=%0d cout=%0d kh=%0d kw=%0d hin=%0d win=%0d hout=%0d wout=%0d act=%0d zp_x=%0d",
                 $time,
                 cfg_pe_mode, cfg_cin, cfg_cout, cfg_kh, cfg_kw,
                 cfg_hin, cfg_win, cfg_hout, cfg_wout, cfg_activation, cfg_zp_x);
    end
  end
`endif
`ifdef RTL_TRACE
  always @(posedge clk) begin
    if (rst_n) begin
      if (seq_pe_enable) begin
        rtl_trace_pkg::rtl_trace_line("S7_GLBI",
          $sformatf("r0=%0d r1=%0d r2=%0d pad0=%b pad1=%b pad2=%b pz=%0d",
                    routed_in_data[0][0][0], routed_in_data[1][0][0], routed_in_data[2][0][0],
                    agi_is_padding_row[0], agi_is_padding_row[1], agi_is_padding_row[2],
                    agi_pad_value));
        rtl_trace_pkg::rtl_trace_line("S7_GLBW",
          $sformatf("w00=%0d w01=%0d w10=%0d w20=%0d",
                    wt_bank_data[0][0][0], wt_bank_data[0][1][0],
                    wt_bank_data[1][0][0], wt_bank_data[2][0][0]));
        rtl_trace_pkg::rtl_trace_line("S7_RT",
          $sformatf("a00=%0d a10=%0d a20=%0d g00=%0d g10=%0d g20=%0d",
                    routed_act[0][0][0], routed_act[1][0][0], routed_act[2][0][0],
                    routed_wgt[0][0][0], routed_wgt[1][0][0], routed_wgt[2][0][0]));
        rtl_trace_pkg::rtl_trace_line("S7_ADR",
          $sformatf("r0=%0d r1=%0d r2=%0d wbk=%0d wa0=%0d wa1=%0d",
                    agi_sram_addr_row[0], agi_sram_addr_row[1], agi_sram_addr_row[2],
                    agw_bank_id, agw_addr[0], agw_addr[1]));
      end
      if (ppu_trigger_delayed)
        rtl_trace_pkg::rtl_trace_line("S7_PSUM",
          $sformatf("c00=%0d c01=%0d c10=%0d c20=%0d c30=%0d",
                    pe_col_psum[0][0], pe_col_psum[0][1],
                    pe_col_psum[1][0], pe_col_psum[2][0], pe_col_psum[3][0]));
      if (ppu_act_wr_en_d[0])
        rtl_trace_pkg::rtl_trace_line("S7_ACT",
          $sformatf("a0=%0d a1=%0d we=%b%b%b%b adr=%0d",
                    ppu_act_wr_data_hold[0][0], ppu_act_wr_data_hold[0][1],
                    glb_act_wr_en[0], glb_act_wr_en[1], glb_act_wr_en[2], glb_act_wr_en[3],
                    ppu_act_wr_addr_hold[0]));
      if (pe_pool_valid)
        rtl_trace_pkg::rtl_trace_line("S7_MP5",
          $sformatf("mx0=%0d mx1=%0d ch=%0d", pe_pool_max[0], pe_pool_max[1], iter_mp5_ch_w));
      if (shadow_latch)
        rtl_trace_pkg::rtl_trace_line("S7_CFG",
          $sformatf("mode=%0d ci=%0d co=%0d kh=%0d kw=%0d hi=%0d wi=%0d ho=%0d wo=%0d",
                    cfg_pe_mode, cfg_cin, cfg_cout, cfg_kh, cfg_kw,
                    cfg_hin, cfg_win, cfg_hout, cfg_wout));
    end
  end
`endif
  // synthesis translate_on

endmodule
