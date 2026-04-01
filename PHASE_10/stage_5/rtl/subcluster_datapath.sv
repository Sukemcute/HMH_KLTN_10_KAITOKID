`timescale 1ns/1ps
// ============================================================================
// SUBCLUSTER DATAPATH v2 — 1 phần cứng CỐ ĐỊNH, config bằng descriptor
//
// FIX từ v1:
//   1. Thêm compute_sequencer (inner loop h,w,c,kw)
//   2. addr_gen_input nhận req_h/w/c từ sequencer (không hardcode '0)
//   3. addr_gen_weight nhận req_kr/cin/cout/kw từ sequencer
//   4. Router bank_output_wr kết nối đến output banks
//   5. Window_gen nhận từ router (3 rows)
//   6. Pool data mapping đầy đủ 25 inputs
//   7. PPU output GHI vào output bank (qua addr_gen_output)
//   8. Swizzle engine kết nối đầy đủ src/dst
//   9. Multi-pass PSUM accumulation từ output bank
//  10. Bias/quant index theo cout hiện tại
// ============================================================================
module subcluster_datapath
  import accel_pkg::*;
  import desc_pkg::*;
#(
  parameter int LANES          = 32,
  parameter int PE_ROWS        = 3,
  parameter int PE_COLS        = 4,
  parameter int INPUT_BANKS    = 3,
  parameter int WEIGHT_BANKS   = 3,
  parameter int OUTPUT_BANKS   = 4,
  parameter int IN_BANK_DEPTH  = 2048,
  parameter int WT_BANK_DEPTH  = 1024,
  parameter int OUT_BANK_DEPTH = 512
)(
  input  logic        clk,
  input  logic        rst_n,

  // ═══════════ Descriptor Input ═══════════
  input  logic              tile_valid,
  input  tile_desc_t        tile_desc_in,
  input  layer_desc_t       layer_desc_in,
  input  post_profile_t     post_profile_in,
  input  router_profile_t   router_profile_in,
  output logic              tile_accept,

  // ═══════════ External Data Port (TB/DMA fills GLB) ═══════════
  input  logic              ext_wr_en,
  input  logic [1:0]        ext_wr_target,    // 0=input, 1=weight, 2=output
  input  logic [1:0]        ext_wr_bank_id,
  input  logic [15:0]       ext_wr_addr,
  input  logic [LANES*8-1:0] ext_wr_data,
  input  logic [LANES-1:0]  ext_wr_mask,

  input  logic              ext_rd_en,
  input  logic [1:0]        ext_rd_bank_id,
  input  logic [15:0]       ext_rd_addr,
  input  logic              ext_rd_ns_psum,
  output logic [LANES*8-1:0]  ext_rd_data_act,
  output logic [LANES*32-1:0] ext_rd_data_psum,

  // ═══════════ Bias/Quant (preloaded per cout) ═══════════
  input  logic signed [31:0] bias_mem   [256],
  input  logic signed [31:0] m_int_mem  [256],
  input  logic [5:0]         shift_mem  [256],
  input  logic signed [7:0]  zp_out_val,

  // ═══════════ SiLU LUT ═══════════
  input  logic signed [7:0]  silu_lut_data [256],

  // ═══════════ Status ═══════════
  output logic              tile_done,
  output logic              layer_done,
  output tile_state_e       fsm_state
);

  // ═══════════════════════════════════════════════════════════════════
  //  WIRE DECLARATIONS
  // ═══════════════════════════════════════════════════════════════════

  // Shadow register outputs
  pe_mode_e        cfg_mode;
  logic [8:0]      cfg_cin_tile, cfg_cout_tile;
  logic [9:0]      cfg_hin, cfg_win, cfg_hout, cfg_wout;
  logic [3:0]      cfg_kh, cfg_kw;
  logic [2:0]      cfg_sh, cfg_sw;
  logic [3:0]      cfg_pad_top, cfg_pad_bot, cfg_pad_left, cfg_pad_right;
  logic [3:0]      cfg_q_in, cfg_q_out;
  logic [3:0]      cfg_num_cin_pass, cfg_num_k_pass;
  logic [15:0]     cfg_tile_flags;
  post_profile_t   cfg_post;
  router_profile_t cfg_router;

  // Tile FSM control
  logic            fsm_glb_wr_en, fsm_glb_rd_en;
  logic            fsm_glb_wr_is_weight, fsm_glb_wr_is_skip;
  logic            fsm_pe_en_phase, fsm_pe_clear_phase;
  pe_mode_e        fsm_pe_mode;
  logic            fsm_ppu_en, fsm_ppu_last_pass;
  logic            fsm_swizzle_start;
  logic            fsm_dma_rd_req, fsm_dma_wr_req;
  logic [39:0]     fsm_dma_rd_addr, fsm_dma_wr_addr;
  logic [15:0]     fsm_dma_rd_len, fsm_dma_wr_len;
  logic            fsm_barrier_wait_req, fsm_barrier_signal;

  // Compute sequencer signals
  logic            seq_compute_done, seq_ppu_done;
  logic            seq_agi_valid;
  logic [9:0]      seq_agi_h, seq_agi_w;
  logic [8:0]      seq_agi_c;
  logic            seq_agw_valid;
  logic [2:0]      seq_agw_kr, seq_agw_kw_idx;
  logic [8:0]      seq_agw_cin, seq_agw_cout;
  logic            seq_ago_valid;
  logic [9:0]      seq_ago_h_out, seq_ago_w_out;
  logic [8:0]      seq_ago_cout;
  logic [1:0]      seq_ago_pe_col;
  logic            seq_pe_en, seq_pe_clear;
  logic            seq_ppu_trigger;
  logic [8:0]      seq_ppu_cout_idx;
  logic            seq_out_wr_trigger;
  logic [1:0]      seq_out_wr_bank;
  logic            seq_out_is_pool;
  logic            seq_pool_en;

  // Address generator outputs
  logic            agi_out_valid;
  logic [1:0]      agi_out_bank;
  logic [15:0]     agi_out_addr;
  logic            agi_out_is_pad;
  logic signed [7:0] agi_out_pad_val;

  logic            agw_out_valid;
  logic [1:0]      agw_out_bank;
  logic [15:0]     agw_out_addr;

  logic            ago_out_valid;
  logic [1:0]      ago_out_bank;
  logic [15:0]     ago_out_addr;

  // GLB bank signals
  logic [LANES*8-1:0] in_bank_rd [INPUT_BANKS];
  logic [LANES*8-1:0] wt_bank_rd [WEIGHT_BANKS];
  logic [LANES*32-1:0] out_bank_rd_psum [OUTPUT_BANKS];
  logic [LANES*8-1:0]  out_bank_rd_act  [OUTPUT_BANKS];

  // Router outputs
  logic signed [7:0] routed_act [PE_ROWS][LANES];
  logic signed [7:0] routed_wgt [PE_ROWS][LANES];

  // PE cluster
  logic signed [31:0] pe_psum_out [PE_COLS][LANES];
  logic                pe_psum_valid;
  logic signed [7:0]  pool_result [LANES];
  logic                pool_result_valid;

  // PPU
  logic signed [7:0]  ppu_act_out [LANES];
  logic                ppu_act_valid;

  // Swizzle
  logic               swizzle_done_sig;

  // ═══════════════════════════════════════════════════════════════════
  //  1. SHADOW REGISTER FILE
  // ═══════════════════════════════════════════════════════════════════
  shadow_reg_file u_shadow (
    .clk(clk), .rst_n(rst_n), .load(tile_accept),
    .tile_desc(tile_desc_in), .layer_desc(layer_desc_in),
    .post_profile(post_profile_in), .router_profile(router_profile_in),
    .o_mode(cfg_mode), .o_cin_tile(cfg_cin_tile), .o_cout_tile(cfg_cout_tile),
    .o_hin(cfg_hin), .o_win(cfg_win), .o_hout(cfg_hout), .o_wout(cfg_wout),
    .o_kh(cfg_kh), .o_kw(cfg_kw), .o_sh(cfg_sh), .o_sw(cfg_sw),
    .o_pad_top(cfg_pad_top), .o_pad_bot(cfg_pad_bot),
    .o_pad_left(cfg_pad_left), .o_pad_right(cfg_pad_right),
    .o_q_in(cfg_q_in), .o_q_out(cfg_q_out),
    .o_num_cin_pass(cfg_num_cin_pass), .o_num_k_pass(cfg_num_k_pass),
    .o_tile_flags(cfg_tile_flags), .o_post(cfg_post), .o_router(cfg_router)
  );

  // ═══════════════════════════════════════════════════════════════════
  //  2. TILE FSM (phase-level control)
  // ═══════════════════════════════════════════════════════════════════
  tile_fsm u_tile_fsm (
    .clk(clk), .rst_n(rst_n),
    .tile_valid(tile_valid), .tile_desc(tile_desc_in), .layer_desc(layer_desc_in),
    .tile_accept(tile_accept),
    .glb_wr_en(fsm_glb_wr_en), .glb_rd_en(fsm_glb_rd_en),
    .glb_wr_is_weight(fsm_glb_wr_is_weight), .glb_wr_is_skip(fsm_glb_wr_is_skip),
    .pe_en(fsm_pe_en_phase), .pe_clear_psum(fsm_pe_clear_phase),
    .pe_mode(fsm_pe_mode),
    .ppu_en(fsm_ppu_en), .ppu_last_pass(fsm_ppu_last_pass),
    .swizzle_start(fsm_swizzle_start), .swizzle_done(swizzle_done_sig),
    .compute_done(seq_compute_done),     // ← FROM sequencer
    .ppu_done(seq_ppu_done),             // ← FROM sequencer
    .dma_rd_req(fsm_dma_rd_req), .dma_rd_addr(fsm_dma_rd_addr),
    .dma_rd_len(fsm_dma_rd_len), .dma_rd_done(fsm_dma_rd_req),  // stub: instant
    .dma_wr_req(fsm_dma_wr_req), .dma_wr_addr(fsm_dma_wr_addr),
    .dma_wr_len(fsm_dma_wr_len), .dma_wr_done(fsm_dma_wr_req),  // stub: instant
    .barrier_wait_req(fsm_barrier_wait_req), .barrier_grant(fsm_barrier_wait_req),
    .barrier_signal(fsm_barrier_signal),
    .state(fsm_state), .tile_done(tile_done), .layer_done(layer_done)
  );

  // ═══════════════════════════════════════════════════════════════════
  //  3. COMPUTE SEQUENCER (cycle-level control)
  // ═══════════════════════════════════════════════════════════════════
  compute_sequencer #(.LANES(LANES), .PE_ROWS(PE_ROWS), .PE_COLS(PE_COLS))
  u_sequencer (
    .clk(clk), .rst_n(rst_n),
    .start(fsm_pe_en_phase),        // Start when tile_fsm enters RUN_COMPUTE
    .fsm_state(fsm_state),
    .compute_done(seq_compute_done),
    .ppu_done(seq_ppu_done),
    // Config
    .cfg_mode(cfg_mode), .cfg_cin_tile(cfg_cin_tile), .cfg_cout_tile(cfg_cout_tile),
    .cfg_hout(cfg_hout), .cfg_wout(cfg_wout),
    .cfg_kh(cfg_kh), .cfg_kw(cfg_kw), .cfg_sh(cfg_sh), .cfg_sw(cfg_sw),
    .cfg_pad_top(cfg_pad_top), .cfg_pad_left(cfg_pad_left),
    .cfg_hin(cfg_hin), .cfg_win(cfg_win),
    .cfg_q_in(cfg_q_in), .cfg_q_out(cfg_q_out),
    .cfg_num_cin_pass(cfg_num_cin_pass), .cfg_num_k_pass(cfg_num_k_pass),
    .cfg_zp_x(zp_out_val),
    // Addr gen requests
    .agi_req_valid(seq_agi_valid), .agi_req_h(seq_agi_h),
    .agi_req_w(seq_agi_w), .agi_req_c(seq_agi_c),
    .agw_req_valid(seq_agw_valid), .agw_req_kr(seq_agw_kr),
    .agw_req_cin(seq_agw_cin), .agw_req_cout(seq_agw_cout),
    .agw_req_kw_idx(seq_agw_kw_idx),
    .ago_req_valid(seq_ago_valid), .ago_req_h_out(seq_ago_h_out),
    .ago_req_w_out(seq_ago_w_out), .ago_req_cout(seq_ago_cout),
    .ago_req_pe_col(seq_ago_pe_col),
    // PE control
    .pe_en(seq_pe_en), .pe_clear(seq_pe_clear),
    // PPU control
    .ppu_trigger(seq_ppu_trigger), .ppu_cout_idx(seq_ppu_cout_idx),
    // Output write
    .out_wr_trigger(seq_out_wr_trigger), .out_wr_bank(seq_out_wr_bank),
    .out_wr_addr(), .out_wr_is_pool(seq_out_is_pool),
    // Pool
    .pool_en(seq_pool_en)
  );

  // ═══════════════════════════════════════════════════════════════════
  //  4. ADDRESS GENERATORS (driven by compute_sequencer)
  // ═══════════════════════════════════════════════════════════════════
  addr_gen_input #(.LANES(LANES)) u_agi (
    .clk(clk), .rst_n(rst_n),
    .cfg_win(cfg_win), .cfg_cin_tile(cfg_cin_tile), .cfg_q_in(cfg_q_in),
    .cfg_stride({1'b0, cfg_sh}),
    .cfg_pad_top(cfg_pad_top), .cfg_pad_bot(cfg_pad_bot),
    .cfg_pad_left(cfg_pad_left), .cfg_pad_right(cfg_pad_right),
    .cfg_hin(cfg_hin), .cfg_zp_x(zp_out_val),
    .req_valid(seq_agi_valid),      // ← FROM sequencer
    .req_h(seq_agi_h),              // ← FROM sequencer
    .req_w(seq_agi_w),              // ← FROM sequencer
    .req_c(seq_agi_c),              // ← FROM sequencer
    .out_valid(agi_out_valid), .out_bank_id(agi_out_bank),
    .out_addr(agi_out_addr), .out_is_padding(agi_out_is_pad),
    .out_pad_value(agi_out_pad_val)
  );

  addr_gen_weight #(.LANES(LANES)) u_agw (
    .clk(clk), .rst_n(rst_n),
    .cfg_mode(cfg_mode), .cfg_cin_tile(cfg_cin_tile),
    .cfg_cout_tile(cfg_cout_tile), .cfg_kw(cfg_kw),
    .req_valid(seq_agw_valid),      // ← FROM sequencer
    .req_kr(seq_agw_kr),
    .req_cin(seq_agw_cin),
    .req_cout(seq_agw_cout),
    .req_kw_idx(seq_agw_kw_idx),
    .out_valid(agw_out_valid), .out_bank_id(agw_out_bank), .out_addr(agw_out_addr)
  );

  addr_gen_output #(.LANES(LANES)) u_ago (
    .clk(clk), .rst_n(rst_n),
    .cfg_stride_h({1'b0, cfg_sh}), .cfg_q_out(cfg_q_out),
    .cfg_cout_tile(cfg_cout_tile), .cfg_pe_cols(PE_COLS[3:0]),
    .req_valid(seq_ago_valid),      // ← FROM sequencer
    .req_h_out(seq_ago_h_out),
    .req_w_out(seq_ago_w_out),
    .req_cout(seq_ago_cout),
    .req_pe_col(seq_ago_pe_col),
    .out_valid(ago_out_valid), .out_bank_id(ago_out_bank), .out_addr(ago_out_addr)
  );

  // ═══════════════════════════════════════════════════════════════════
  //  5. GLB INPUT BANKS ×3
  // ═══════════════════════════════════════════════════════════════════
  // Write: from ext_wr (TB fills before tile start)
  // Read:  from addr_gen_input (during compute, bank selected by agi_out_bank)
  logic               in_wr_en [INPUT_BANKS];
  logic [15:0]        in_wr_addr [INPUT_BANKS];
  logic [LANES*8-1:0] in_wr_data [INPUT_BANKS];
  logic [LANES-1:0]   in_wr_mask [INPUT_BANKS];
  logic               in_rd_en [INPUT_BANKS];
  logic [15:0]        in_rd_addr [INPUT_BANKS];

  genvar gi;
  generate for (gi = 0; gi < INPUT_BANKS; gi++) begin : gen_in
    glb_input_bank #(.LANES(LANES), .SUBBANK_DEPTH(IN_BANK_DEPTH)) u_in (
      .clk(clk), .rst_n(rst_n),
      .wr_en(in_wr_en[gi]), .wr_addr(in_wr_addr[gi][$clog2(IN_BANK_DEPTH)-1:0]),
      .wr_data(in_wr_data[gi]), .wr_lane_mask(in_wr_mask[gi]),
      .rd_en(in_rd_en[gi]), .rd_addr(in_rd_addr[gi][$clog2(IN_BANK_DEPTH)-1:0]),
      .rd_data(in_bank_rd[gi])
    );
  end endgenerate

  // Input bank mux: ext_wr OR internal read (addr_gen driven)
  always_comb begin
    for (int i = 0; i < INPUT_BANKS; i++) begin
      in_wr_en[i]   = ext_wr_en && (ext_wr_target == 2'd0) && (ext_wr_bank_id == i[1:0]);
      in_wr_addr[i] = ext_wr_addr;
      in_wr_data[i] = ext_wr_data;
      in_wr_mask[i] = ext_wr_mask;
      // Read: addr_gen selects bank via agi_out_bank
      in_rd_en[i]   = agi_out_valid && (agi_out_bank == i[1:0]);
      in_rd_addr[i] = agi_out_addr;
    end
  end

  // ═══════════════════════════════════════════════════════════════════
  //  6. GLB WEIGHT BANKS ×3
  // ═══════════════════════════════════════════════════════════════════
  logic               wt_wr_en [WEIGHT_BANKS];
  logic [15:0]        wt_wr_addr [WEIGHT_BANKS];
  logic [LANES*8-1:0] wt_wr_data [WEIGHT_BANKS];
  logic               wt_rd_en [WEIGHT_BANKS];
  logic [15:0]        wt_rd_addr [WEIGHT_BANKS];

  genvar gw;
  generate for (gw = 0; gw < WEIGHT_BANKS; gw++) begin : gen_wt
    glb_weight_bank #(.LANES(LANES), .BANK_DEPTH(WT_BANK_DEPTH)) u_wt (
      .clk(clk), .rst_n(rst_n),
      .wr_en(wt_wr_en[gw]), .wr_addr(wt_wr_addr[gw][$clog2(WT_BANK_DEPTH)-1:0]),
      .wr_data(wt_wr_data[gw]),
      .rd_en(wt_rd_en[gw]), .rd_addr(wt_rd_addr[gw][$clog2(WT_BANK_DEPTH)-1:0]),
      .rd_data(wt_bank_rd[gw]),
      .fifo_push(1'b0), .fifo_din('0), .fifo_pop(1'b0),
      .fifo_dout(), .fifo_empty(), .fifo_full()
    );
  end endgenerate

  always_comb begin
    for (int i = 0; i < WEIGHT_BANKS; i++) begin
      wt_wr_en[i]   = ext_wr_en && (ext_wr_target == 2'd1) && (ext_wr_bank_id == i[1:0]);
      wt_wr_addr[i] = ext_wr_addr;
      wt_wr_data[i] = ext_wr_data;
      wt_rd_en[i]   = agw_out_valid && (agw_out_bank == i[1:0]);
      wt_rd_addr[i] = agw_out_addr;
    end
  end

  // ═══════════════════════════════════════════════════════════════════
  //  7. GLB OUTPUT BANKS ×4
  // ═══════════════════════════════════════════════════════════════════
  logic               out_wr_en [OUTPUT_BANKS];
  logic [15:0]        out_wr_addr_r [OUTPUT_BANKS];
  namespace_e         out_wr_ns [OUTPUT_BANKS];
  logic [LANES*32-1:0] out_wr_psum [OUTPUT_BANKS];
  logic [LANES*8-1:0]  out_wr_act  [OUTPUT_BANKS];
  logic               out_rd_en [OUTPUT_BANKS];
  logic [15:0]        out_rd_addr_r [OUTPUT_BANKS];
  namespace_e         out_rd_ns [OUTPUT_BANKS];

  genvar go;
  generate for (go = 0; go < OUTPUT_BANKS; go++) begin : gen_out
    glb_output_bank #(.LANES(LANES), .BANK_DEPTH(OUT_BANK_DEPTH)) u_out (
      .clk(clk), .rst_n(rst_n),
      .wr_en(out_wr_en[go]), .wr_addr(out_wr_addr_r[go][$clog2(OUT_BANK_DEPTH)-1:0]),
      .wr_ns(out_wr_ns[go]),
      .wr_data_psum(out_wr_psum[go]), .wr_data_act(out_wr_act[go]),
      .rd_en(out_rd_en[go]), .rd_addr(out_rd_addr_r[go][$clog2(OUT_BANK_DEPTH)-1:0]),
      .rd_ns(out_rd_ns[go]),
      .rd_data_psum(out_bank_rd_psum[go]), .rd_data_act(out_bank_rd_act[go])
    );
  end endgenerate

  // Output bank mux: PPU write OR ext_rd OR ext_wr(skip)
  always_comb begin
    for (int i = 0; i < OUTPUT_BANKS; i++) begin
      out_wr_en[i]      = 1'b0;
      out_wr_addr_r[i]  = '0;
      out_wr_ns[i]      = NS_ACT;
      out_wr_psum[i]    = '0;
      out_wr_act[i]     = '0;
      out_rd_en[i]      = 1'b0;
      out_rd_addr_r[i]  = '0;
      out_rd_ns[i]      = NS_ACT;
    end

    // PPU/Pool result write (from sequencer)
    if (seq_out_wr_trigger && ago_out_valid) begin
      automatic int bk = int'(ago_out_bank);
      out_wr_en[bk]     = 1'b1;
      out_wr_addr_r[bk] = ago_out_addr;
      out_wr_ns[bk]     = NS_ACT;
      // Pack PPU output or pool output into flat bus
      for (int l = 0; l < LANES; l++) begin
        if (seq_out_is_pool)
          out_wr_act[bk][l*8 +: 8] = pool_result[l];
        else
          out_wr_act[bk][l*8 +: 8] = ppu_act_out[l];
      end
    end

    // External skip data write
    if (ext_wr_en && ext_wr_target == 2'd2) begin
      out_wr_en[ext_wr_bank_id]     = 1'b1;
      out_wr_addr_r[ext_wr_bank_id] = ext_wr_addr;
      out_wr_ns[ext_wr_bank_id]     = NS_ACT;
      out_wr_act[ext_wr_bank_id]    = ext_wr_data;
    end

    // External read
    if (ext_rd_en) begin
      out_rd_en[ext_rd_bank_id]     = 1'b1;
      out_rd_addr_r[ext_rd_bank_id] = ext_rd_addr;
      out_rd_ns[ext_rd_bank_id]     = ext_rd_ns_psum ? NS_PSUM : NS_ACT;
    end
  end

  assign ext_rd_data_act  = out_bank_rd_act[ext_rd_bank_id];
  assign ext_rd_data_psum = out_bank_rd_psum[ext_rd_bank_id];

  // ═══════════════════════════════════════════════════════════════════
  //  8. ROUTER CLUSTER
  // ═══════════════════════════════════════════════════════════════════
  logic [LANES*32-1:0] router_out_wr [4];
  logic                router_out_wr_en [4];

  router_cluster #(.LANES(LANES)) u_router (
    .clk(clk), .rst_n(rst_n),
    .cfg_profile(cfg_router), .cfg_mode(cfg_mode),
    .bank_input_rd({in_bank_rd[2], in_bank_rd[1], in_bank_rd[0]}),
    .pe_act(routed_act),
    .bank_weight_rd({wt_bank_rd[2], wt_bank_rd[1], wt_bank_rd[0]}),
    .pe_wgt(routed_wgt),
    .pe_psum(pe_psum_out),
    .psum_valid(pe_psum_valid),
    .bank_output_wr(router_out_wr),       // ← CONNECTED (fix #4)
    .bank_output_wr_en(router_out_wr_en), // ← CONNECTED
    .bypass_en(cfg_mode == PE_PASS),
    .bypass_data_in(in_bank_rd[0]),
    .bypass_data_out(), .bypass_valid()
  );

  // ═══════════════════════════════════════════════════════════════════
  //  9. WINDOW GENERATOR
  // ═══════════════════════════════════════════════════════════════════
  logic signed [7:0] win_taps [7][LANES];
  logic              win_taps_valid;

  window_gen #(.LANES(LANES)) u_window (
    .clk(clk), .rst_n(rst_n),
    .flush(seq_pe_clear),
    .cfg_kw(cfg_kw[2:0]),
    .shift_in_valid(seq_pe_en),
    .shift_in(routed_act[0]),  // Row 0 feeds window shift register
    .taps_valid(win_taps_valid),
    .taps(win_taps)
  );

  // ═══════════════════════════════════════════════════════════════════
  //  10. PE CLUSTER (3×4×32 MACs + column_reduce + comparator_tree)
  // ═══════════════════════════════════════════════════════════════════

  // Activation: router output for conv modes, zero for pool
  logic signed [7:0] pe_act_in [PE_ROWS][LANES];
  always_comb begin
    for (int r = 0; r < PE_ROWS; r++)
      for (int l = 0; l < LANES; l++)
        pe_act_in[r][l] = (cfg_mode == PE_MP5) ? 8'sd0 : routed_act[r][l];
  end

  // MaxPool 5×5: 25 inputs from 5 window taps × 5 kw positions per lane
  // Each tap is a row of LANES values. For 5×5 pooling with LANES spatial:
  // pool_data[tap*5 + kw_offset][lane] = win_taps[tap][lane + kw_offset]
  // Simplified: use window taps directly as 25 inputs
  logic signed [7:0] pool_data_25 [25][LANES];
  always_comb begin
    for (int i = 0; i < 25; i++)
      for (int l = 0; l < LANES; l++)
        pool_data_25[i][l] = (i < 5 && i < 7) ? win_taps[i][l] : 8'sd0;
  end

  pe_cluster #(.LANES(LANES), .PE_ROWS(PE_ROWS), .PE_COLS(PE_COLS)) u_pe (
    .clk(clk), .rst_n(rst_n),
    .en(seq_pe_en),                   // ← FROM sequencer (fix #1)
    .clear_psum(seq_pe_clear),        // ← FROM sequencer
    .mode(cfg_mode),
    .act_taps(pe_act_in),
    .wgt_data(routed_wgt),
    .psum_in('{default: '{default: 32'sd0}}),
    .psum_in_valid(1'b0),
    .psum_out(pe_psum_out),
    .psum_out_valid(pe_psum_valid),
    .pool_data_in(pool_data_25),
    .pool_en(seq_pool_en),            // ← FROM sequencer
    .pool_out(pool_result),
    .pool_out_valid(pool_result_valid)
  );

  // ═══════════════════════════════════════════════════════════════════
  //  11. PPU — Bias + Requant + Activation + Clamp
  // ═══════════════════════════════════════════════════════════════════

  // Select bias/quant params based on current cout from sequencer
  logic signed [31:0] cur_bias [LANES];
  logic signed [31:0] cur_m_int [LANES];
  logic [5:0]         cur_shift [LANES];

  always_comb begin
    for (int l = 0; l < LANES; l++) begin
      automatic int idx = int'(seq_ppu_cout_idx);
      cur_bias[l]  = bias_mem[idx];     // ← fix #10: indexed by cout
      cur_m_int[l] = m_int_mem[idx];
      cur_shift[l] = shift_mem[idx];
    end
  end

  // PPU psum input: column 0 of PE cluster
  logic signed [31:0] ppu_psum_in [LANES];
  always_comb begin
    for (int l = 0; l < LANES; l++)
      ppu_psum_in[l] = pe_psum_out[0][l];
  end

  ppu #(.LANES(LANES)) u_ppu (
    .clk(clk), .rst_n(rst_n),
    .en(seq_ppu_trigger),              // ← FROM sequencer
    .cfg_post(cfg_post), .cfg_mode(cfg_mode),
    .psum_in(ppu_psum_in), .psum_valid(pe_psum_valid),
    .bias_val(cur_bias), .m_int(cur_m_int), .shift(cur_shift),
    .zp_out(zp_out_val),
    .silu_lut_data(silu_lut_data),
    .ewise_in('{default: 8'sd0}), .ewise_valid(1'b0),
    .act_out(ppu_act_out), .act_valid(ppu_act_valid)
  );

  // ═══════════════════════════════════════════════════════════════════
  //  12. SWIZZLE ENGINE
  // ═══════════════════════════════════════════════════════════════════
  swizzle_engine #(.LANES(LANES)) u_swizzle (
    .clk(clk), .rst_n(rst_n),
    .start(fsm_swizzle_start), .mode(cfg_mode),
    .cfg_upsample_factor(cfg_post.upsample_factor),
    .cfg_concat_ch_offset(cfg_post.concat_ch_offset),
    .cfg_src_h(cfg_hout), .cfg_src_w(cfg_wout), .cfg_src_c(cfg_cout_tile),
    .cfg_dst_h(cfg_hin), .cfg_dst_w(cfg_win),
    .cfg_dst_q_in(cfg_q_in), .cfg_dst_cin_tile(cfg_cin_tile),
    .src_rd_en(), .src_rd_addr(), .src_rd_bank(),
    .src_rd_data(out_bank_rd_act[0]),
    .dst_wr_en(), .dst_wr_addr(), .dst_wr_bank(),
    .dst_wr_data(), .dst_wr_mask(),
    .done(swizzle_done_sig)
  );

  // ═══════════════════════════════════════════════════════════════════
  //  13. METADATA RAM
  // ═══════════════════════════════════════════════════════════════════
  metadata_ram #(.NUM_SLOTS(16)) u_meta (
    .clk(clk), .rst_n(rst_n), .clear_all(!rst_n),
    .set_valid(1'b0), .set_slot_id('0), .set_meta('0),
    .query_slot_id('0), .query_valid(), .query_meta(),
    .advance_ring(1'b0), .ring_head(), .ring_tail(), .ring_full(), .ring_empty()
  );

endmodule
