`timescale 1ns/1ps
// =============================================================================
// SUBCLUSTER = một “slice” của IP duy nhất (RTL_MODULE_SPEC.md / PHASE 0–2 V2).
//
// Kiến trúc mục tiêu (một datapath RTL thống nhất, không nhân đôi conv):
//   tile_fsm → shadow_reg_file
//   → glb_{input,weight,output} + addr_gen_* + metadata_ram
//   → router_cluster → window_gen → pe_cluster (12× pe_unit + column_reduce + pool)
//   → ppu → swizzle_engine → ext_* handshake → tensor_dma (trong supercluster).
//
// Hiện trạng file này: vẫn dùng bộ đệm behavioral + conv/PPU tách khỏi pe_cluster/
// router/ swizzle để cosim nhanh — LỆCH SPEC. Khi IP “đúng một thiết kế”, thay thế
// các khối always_ff behavioral bằng instance các module trong 01_* … 05_* và nối
// đúng tín hiệu theo spec (xem PHASE_3/RTL_IP_DIRECTORY_TREE.txt).
// =============================================================================
//
// Data flow hiện tại (cosim):
//   1. PREFILL_WT   : DMA read weights → bh_weight buffer
//   2. PREFILL_IN   : DMA read activations → bh_input buffer
//   3. PREFILL_SKIP : DMA read skip data → bh_skip buffer (optional)
//   4. RUN_COMPUTE  : behavioral conv → bh_psum (ALL output channels)
//   5. POST_PROCESS : RTL ppu.sv + sequencer → bh_act_out
//   6. SWIZZLE_STORE: bh_act_out → DMA write to DDR
//
// Input in DDR  : HWC layout
// Weight in DDR : OIHW layout (cout × cin × kh × kw contiguous bytes)
// Output to DDR : HWC layout
module subcluster_wrapper #(
  parameter int LANES = 32
)(
  input  logic               clk,
  input  logic               rst_n,

  input  logic               tile_valid,
  input  desc_pkg::tile_desc_t  tile_desc,
  input  desc_pkg::layer_desc_t layer_desc,
  output logic               tile_accept,

  output logic               ext_rd_req,
  output logic [39:0]        ext_rd_addr,
  output logic [15:0]        ext_rd_len,
  input  logic               ext_rd_grant,
  input  logic [255:0]       ext_rd_data,
  input  logic               ext_rd_valid,
  // Pulse when tensor_dma finishes the full rd_len transfer (must match write path ext_wr_done)
  input  logic               ext_rd_done,

  output logic               ext_wr_req,
  output logic [39:0]        ext_wr_addr,
  output logic [15:0]        ext_wr_len,
  input  logic               ext_wr_grant,
  output logic [255:0]       ext_wr_data,
  output logic               ext_wr_valid,
  input  logic               ext_wr_done,
  input  logic               ext_wr_beat,

  input  logic signed [31:0] ppu_bias    [LANES],
  input  logic signed [31:0] ppu_m_int   [LANES],
  input  logic        [5:0]  ppu_shift   [LANES],
  input  logic signed [7:0]  ppu_zp_out,
  input  logic signed [7:0]  ppu_silu_lut [256],

  output logic               barrier_signal,
  output logic [4:0]         barrier_signal_id,
  input  logic               barrier_grant,

  output accel_pkg::tile_state_e state,
  output logic               tile_done,
  output logic               layer_done
);
  import accel_pkg::*;
  import desc_pkg::*;

  // ═══════════════════════════════════════════════════════════
  //  Post-processing profile lookup
  // ═══════════════════════════════════════════════════════════
  post_profile_t derived_post_profile;
  always_comb begin
    derived_post_profile = '0;
    case (layer_desc.post_profile_id)
      8'd0: begin
        derived_post_profile.bias_en    = 1'b1;
        derived_post_profile.quant_mode = QMODE_PER_CHANNEL;
        derived_post_profile.act_mode   = ACT_SILU;
      end
      8'd1: begin
        derived_post_profile.bias_en    = 1'b1;
        derived_post_profile.quant_mode = QMODE_PER_CHANNEL;
        derived_post_profile.act_mode   = ACT_NONE;
      end
      8'd2: begin
        derived_post_profile.bias_en    = 1'b1;
        derived_post_profile.quant_mode = QMODE_PER_CHANNEL;
        derived_post_profile.act_mode   = ACT_SILU;
        derived_post_profile.ewise_en   = 1'b1;
      end
      8'd3: begin
        derived_post_profile.bias_en    = 1'b1;
        derived_post_profile.quant_mode = QMODE_PER_CHANNEL;
        derived_post_profile.act_mode   = ACT_RELU;
      end
      default: begin
        derived_post_profile.bias_en    = 1'b1;
        derived_post_profile.quant_mode = QMODE_PER_CHANNEL;
        derived_post_profile.act_mode   = ACT_SILU;
      end
    endcase
  end

  // ═══════════════════════════════════════════════════════════
  //  Shadow Register File
  // ═══════════════════════════════════════════════════════════
  logic              shadow_load;
  pe_mode_e          cfg_mode;
  logic [8:0]        cfg_cin_tile, cfg_cout_tile;
  logic [9:0]        cfg_hin, cfg_win, cfg_hout, cfg_wout;
  logic [3:0]        cfg_kh, cfg_kw;
  logic [2:0]        cfg_sh, cfg_sw;
  logic [3:0]        cfg_pad_top, cfg_pad_bot, cfg_pad_left, cfg_pad_right;
  logic [3:0]        cfg_q_in, cfg_q_out;
  logic [3:0]        cfg_num_cin_pass, cfg_num_k_pass;
  logic [15:0]       cfg_tile_flags;
  post_profile_t     cfg_post;
  router_profile_t   cfg_router;

  shadow_reg_file u_shadow (
    .clk           (clk),
    .rst_n         (rst_n),
    .load          (shadow_load),
    .tile_desc     (tile_desc),
    .layer_desc    (layer_desc),
    .post_profile  (derived_post_profile),
    .router_profile('0),
    .o_mode        (cfg_mode),
    .o_cin_tile    (cfg_cin_tile),
    .o_cout_tile   (cfg_cout_tile),
    .o_hin         (cfg_hin),
    .o_win         (cfg_win),
    .o_hout        (cfg_hout),
    .o_wout        (cfg_wout),
    .o_kh          (cfg_kh),
    .o_kw          (cfg_kw),
    .o_sh          (cfg_sh),
    .o_sw          (cfg_sw),
    .o_pad_top     (cfg_pad_top),
    .o_pad_bot     (cfg_pad_bot),
    .o_pad_left    (cfg_pad_left),
    .o_pad_right   (cfg_pad_right),
    .o_q_in        (cfg_q_in),
    .o_q_out       (cfg_q_out),
    .o_num_cin_pass(cfg_num_cin_pass),
    .o_num_k_pass  (cfg_num_k_pass),
    .o_tile_flags  (cfg_tile_flags),
    .o_post        (cfg_post),
    .o_router      (cfg_router)
  );

  // ═══════════════════════════════════════════════════════════
  //  Tile FSM
  // ═══════════════════════════════════════════════════════════
  logic fsm_glb_wr_en, fsm_glb_rd_en;
  logic fsm_glb_wr_is_weight, fsm_glb_wr_is_skip;
  logic fsm_pe_en, fsm_pe_clear;
  pe_mode_e fsm_pe_mode;
  logic fsm_ppu_en, fsm_ppu_last;
  logic fsm_swizzle_start, swizzle_done_sig;
  logic fsm_dma_rd_req, fsm_dma_rd_done;
  logic [39:0] fsm_dma_rd_addr;
  logic [15:0] fsm_dma_rd_len;
  logic fsm_dma_wr_req, fsm_dma_wr_done;
  logic [39:0] fsm_dma_wr_addr;
  logic [15:0] fsm_dma_wr_len;
  logic compute_done_sig;
  logic ppu_done_sig;

  tile_fsm u_fsm (
    .clk              (clk),
    .rst_n            (rst_n),
    .tile_valid       (tile_valid),
    .tile_desc        (tile_desc),
    .layer_desc       (layer_desc),
    .tile_accept      (tile_accept),
    .glb_wr_en        (fsm_glb_wr_en),
    .glb_rd_en        (fsm_glb_rd_en),
    .glb_wr_is_weight (fsm_glb_wr_is_weight),
    .glb_wr_is_skip   (fsm_glb_wr_is_skip),
    .pe_en            (fsm_pe_en),
    .pe_clear_psum    (fsm_pe_clear),
    .pe_mode          (fsm_pe_mode),
    .ppu_en           (fsm_ppu_en),
    .ppu_last_pass    (fsm_ppu_last),
    .swizzle_start    (fsm_swizzle_start),
    .swizzle_done     (swizzle_done_sig),
    .compute_done     (compute_done_sig),
    .ppu_done         (ppu_done_sig),
    .dma_rd_req       (fsm_dma_rd_req),
    .dma_rd_addr      (fsm_dma_rd_addr),
    .dma_rd_len       (fsm_dma_rd_len),
    .dma_rd_done      (fsm_dma_rd_done),
    .dma_wr_req       (fsm_dma_wr_req),
    .dma_wr_addr      (fsm_dma_wr_addr),
    .dma_wr_len       (fsm_dma_wr_len),
    .dma_wr_done      (ext_wr_done),
    .barrier_wait_req (),
    .barrier_grant    (barrier_grant),
    .barrier_signal   (barrier_signal),
    .state            (state),
    .tile_done        (tile_done),
    .layer_done       (layer_done)
  );

`ifdef ACCEL_DEBUG
`ifndef ACCEL_TILE_DONE_LOG_MAX
  `define ACCEL_TILE_DONE_LOG_MAX 128
`endif
  integer dbg_tile_done_n;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      dbg_tile_done_n <= 0;
    else if (tile_done) begin
      if (dbg_tile_done_n < `ACCEL_TILE_DONE_LOG_MAX || dbg_tile_done_n[9:0] == 10'd0)
        $display("[%t] %m [CHK-TILE-DONE] tile_id=%0h layer=%0d (conv+PPU+store path finished this sub)",
                 $time, tile_desc.tile_id, tile_desc.layer_id);
      dbg_tile_done_n <= dbg_tile_done_n + 1;
    end
  end
`endif

  assign shadow_load = (state == TILE_LOAD_CFG);

  // ═══════════════════════════════════════════════════════════
  //  DMA Fill Phase Tracking
  // ═══════════════════════════════════════════════════════════
  logic [15:0] fill_addr;
  logic        fill_phase_start;

  assign fill_phase_start = (state == TILE_LOAD_CFG)
                          || (state == TILE_PREFILL_WT   && fsm_dma_rd_done)
                          || (state == TILE_PREFILL_IN   && fsm_dma_rd_done);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      fill_addr <= '0;
    else if (fill_phase_start)
      fill_addr <= '0;
    else if (fsm_glb_wr_en && ext_rd_valid)
      fill_addr <= fill_addr + 1;
  end

  // Read completion: same source as tensor_dma.rd_done (beat counting drifted vs DMA and deadlocked PREFILL)
  assign fsm_dma_rd_done = ext_rd_done;

  // ═══════════════════════════════════════════════════════════
  //  Behavioral DMA Byte Buffers
  // ═══════════════════════════════════════════════════════════
  localparam int BH_BUF_SIZE = 262144;  // 256 KB per buffer
  logic [7:0] bh_input  [BH_BUF_SIZE];
  logic [7:0] bh_weight [BH_BUF_SIZE];
  logic [7:0] bh_skip   [BH_BUF_SIZE];

  logic [17:0] bh_in_byte_idx, bh_wt_byte_idx, bh_skip_byte_idx;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      bh_in_byte_idx   <= '0;
      bh_wt_byte_idx   <= '0;
      bh_skip_byte_idx <= '0;
    end else begin
      if (state == TILE_LOAD_CFG) begin
        bh_in_byte_idx   <= '0;
        bh_wt_byte_idx   <= '0;
        bh_skip_byte_idx <= '0;
      end

      if (ext_rd_valid && fsm_glb_wr_en) begin
        if (fsm_glb_wr_is_weight) begin
          for (int i = 0; i < 32; i++)
            bh_weight[bh_wt_byte_idx + i] <= ext_rd_data[i*8 +: 8];
          bh_wt_byte_idx <= bh_wt_byte_idx + 32;
        end else if (fsm_glb_wr_is_skip) begin
          for (int i = 0; i < 32; i++)
            bh_skip[bh_skip_byte_idx + i] <= ext_rd_data[i*8 +: 8];
          bh_skip_byte_idx <= bh_skip_byte_idx + 32;
        end else begin
          for (int i = 0; i < 32; i++)
            bh_input[bh_in_byte_idx + i] <= ext_rd_data[i*8 +: 8];
          bh_in_byte_idx <= bh_in_byte_idx + 32;
        end
      end
    end
  end

  // ═══════════════════════════════════════════════════════════
  //  Behavioral PSUM and ACT Output Buffers
  // ═══════════════════════════════════════════════════════════
  localparam int BH_MAX_SPATIAL = 2048;
  localparam int BH_MAX_COUT    = 512;
  logic signed [31:0] bh_psum    [BH_MAX_SPATIAL * BH_MAX_COUT];
  logic        [7:0]  bh_act_out [BH_MAX_SPATIAL * BH_MAX_COUT];

  // ═══════════════════════════════════════════════════════════
  //  Internal PPU Params (all channels, loaded from hex files)
  // ═══════════════════════════════════════════════════════════
  logic signed [31:0] bh_bias_all  [BH_MAX_COUT];
  logic signed [31:0] bh_m_int_all [BH_MAX_COUT];
  logic        [5:0]  bh_shift_all [BH_MAX_COUT];
  logic signed [7:0]  bh_zp_out_val;
  logic signed [7:0]  bh_silu_all  [256];
  act_mode_e          bh_act_mode;
  logic               bh_bias_en;
  logic               bh_ewise_en;

  // Track which layer's PPU params are loaded
  logic [4:0] bh_ppu_loaded_layer;

  // Load PPU params from hex files at tile start
  localparam string BH_PPU_DIR = "E:/KLTN_HMH_FINAL/SW_KLTN/PHASE_4/02_golden_data/";

  task automatic bh_load_ppu_params(input int layer_idx, input int cout);
    string fname;
    integer fd;
    string line;
    logic [31:0] v32;
    logic [7:0] v8;

    // bias (INT32, 8 per line)
    $sformat(fname, "%sppu_bias_L%02d.hex", BH_PPU_DIR, layer_idx);
    fd = $fopen(fname, "r");
    if (fd != 0) begin
      for (int ch = 0; ch < cout; ch += 8) begin
        if (!$feof(fd) && $fgets(line, fd)) begin
          for (int j = 0; j < 8 && (ch+j) < cout; j++) begin
            v32 = '0;
            for (int k = 0; k < 8; k++) begin
              automatic logic [3:0] nib;
              automatic byte c = line.getc(j*8 + k);
              if      (c >= "0" && c <= "9") nib = c - "0";
              else if (c >= "A" && c <= "F") nib = c - "A" + 10;
              else if (c >= "a" && c <= "f") nib = c - "a" + 10;
              else nib = 0;
              v32 = {v32[27:0], nib};
            end
            bh_bias_all[ch+j] = $signed(v32);
          end
        end
      end
      $fclose(fd);
    end else begin
      for (int ch = 0; ch < cout; ch++) bh_bias_all[ch] = 32'sd0;
    end

    // m_int (same format)
    $sformat(fname, "%sppu_m_int_L%02d.hex", BH_PPU_DIR, layer_idx);
    fd = $fopen(fname, "r");
    if (fd != 0) begin
      for (int ch = 0; ch < cout; ch += 8) begin
        if (!$feof(fd) && $fgets(line, fd)) begin
          for (int j = 0; j < 8 && (ch+j) < cout; j++) begin
            v32 = '0;
            for (int k = 0; k < 8; k++) begin
              automatic logic [3:0] nib;
              automatic byte c = line.getc(j*8 + k);
              if      (c >= "0" && c <= "9") nib = c - "0";
              else if (c >= "A" && c <= "F") nib = c - "A" + 10;
              else if (c >= "a" && c <= "f") nib = c - "a" + 10;
              else nib = 0;
              v32 = {v32[27:0], nib};
            end
            bh_m_int_all[ch+j] = $signed(v32);
          end
        end
      end
      $fclose(fd);
    end else begin
      for (int ch = 0; ch < cout; ch++) bh_m_int_all[ch] = 32'sd1;
    end

    // shift (UINT8, 32 per line)
    $sformat(fname, "%sppu_shift_L%02d.hex", BH_PPU_DIR, layer_idx);
    fd = $fopen(fname, "r");
    if (fd != 0) begin
      for (int ch = 0; ch < cout; ch += 32) begin
        if (!$feof(fd) && $fgets(line, fd)) begin
          for (int j = 0; j < 32 && (ch+j) < cout; j++) begin
            v8 = '0;
            for (int k = 0; k < 2; k++) begin
              automatic logic [3:0] nib;
              automatic byte c = line.getc(j*2 + k);
              if      (c >= "0" && c <= "9") nib = c - "0";
              else if (c >= "A" && c <= "F") nib = c - "A" + 10;
              else if (c >= "a" && c <= "f") nib = c - "a" + 10;
              else nib = 0;
              v8 = {v8[3:0], nib};
            end
            bh_shift_all[ch+j] = v8[5:0];
          end
        end
      end
      $fclose(fd);
    end else begin
      for (int ch = 0; ch < cout; ch++) bh_shift_all[ch] = 6'd0;
    end

    // zp_out (single INT8)
    $sformat(fname, "%sppu_zp_out_L%02d.hex", BH_PPU_DIR, layer_idx);
    fd = $fopen(fname, "r");
    if (fd != 0) begin
      if ($fgets(line, fd)) begin
        v8 = '0;
        for (int k = 0; k < 2; k++) begin
          automatic logic [3:0] nib;
          automatic byte c = line.getc(k);
          if      (c >= "0" && c <= "9") nib = c - "0";
          else if (c >= "A" && c <= "F") nib = c - "A" + 10;
          else if (c >= "a" && c <= "f") nib = c - "a" + 10;
          else nib = 0;
          v8 = {v8[3:0], nib};
        end
        bh_zp_out_val = $signed(v8);
      end
      $fclose(fd);
    end else begin
      bh_zp_out_val = 8'sd0;
    end

    // SiLU LUT (256 UINT8, 32 per line)
    $sformat(fname, "%ssilu_lut_L%02d.hex", BH_PPU_DIR, layer_idx);
    fd = $fopen(fname, "r");
    if (fd != 0) begin
      for (int i = 0; i < 256; i += 32) begin
        if (!$feof(fd) && $fgets(line, fd)) begin
          for (int j = 0; j < 32 && (i+j) < 256; j++) begin
            v8 = '0;
            for (int k = 0; k < 2; k++) begin
              automatic logic [3:0] nib;
              automatic byte c = line.getc(j*2 + k);
              if      (c >= "0" && c <= "9") nib = c - "0";
              else if (c >= "A" && c <= "F") nib = c - "A" + 10;
              else if (c >= "a" && c <= "f") nib = c - "a" + 10;
              else nib = 0;
              v8 = {v8[3:0], nib};
            end
            bh_silu_all[i+j] = $signed(v8);
          end
        end
      end
      $fclose(fd);
    end else begin
      for (int i = 0; i < 256; i++) bh_silu_all[i] = $signed(8'(i));
    end
  endtask

  // ═══════════════════════════════════════════════════════════
  //  Tile-local dimensions (registered on LOAD_CFG)
  // ═══════════════════════════════════════════════════════════
  logic [5:0]  t_valid_h, t_valid_w;
  logic [3:0]  t_halo_top, t_halo_left;
  logic signed [4:0] t_pad_top_eff, t_pad_left_eff;
  logic [8:0]  t_cin, t_cout;
  logic [3:0]  t_kh, t_kw;
  logic [2:0]  t_sh, t_sw;
  logic [9:0]  t_in_w;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      t_valid_h  <= '0; t_valid_w  <= '0;
      t_halo_top <= '0; t_halo_left<= '0;
      t_pad_top_eff  <= '0; t_pad_left_eff <= '0;
      t_cin      <= '0; t_cout     <= '0;
      t_kh       <= '0; t_kw       <= '0;
      t_sh       <= '0; t_sw       <= '0;
      t_in_w     <= '0;
      bh_ppu_loaded_layer <= 5'h1F;
    end else if (state == TILE_LOAD_CFG) begin
      t_valid_h  <= tile_desc.valid_h;
      t_valid_w  <= tile_desc.valid_w;
      t_halo_top <= tile_desc.halo_top;
      t_halo_left<= tile_desc.halo_left;
      // Effective padding: kernel rows/cols that fall outside the DMA'd region
      t_pad_top_eff  <= 5'(((layer_desc.kh - 4'd1) >> 1)) - 5'(tile_desc.halo_top);
      t_pad_left_eff <= 5'(((layer_desc.kw - 4'd1) >> 1)) - 5'(tile_desc.halo_left);
      t_cin      <= layer_desc.tile_cin;
      t_cout     <= layer_desc.tile_cout;
      t_kh       <= layer_desc.kh;
      t_kw       <= layer_desc.kw;
      t_sh       <= layer_desc.sh;
      t_sw       <= layer_desc.sw;
      t_in_w     <= layer_desc.win;
      bh_act_mode <= derived_post_profile.act_mode;
      bh_bias_en  <= derived_post_profile.bias_en;
      bh_ewise_en <= derived_post_profile.ewise_en;
      if (layer_desc.layer_id != bh_ppu_loaded_layer) begin
        bh_load_ppu_params(int'(layer_desc.layer_id), int'(layer_desc.cout_total));
        bh_ppu_loaded_layer <= layer_desc.layer_id;
      end
    end
  end

  // ═══════════════════════════════════════════════════════════
  //  RUN_COMPUTE Phase — Behavioral Convolution (ALL channels)
  // ═══════════════════════════════════════════════════════════
  wire [15:0] compute_total = 16'(t_valid_h) * 16'(t_valid_w);
  logic [15:0] compute_cnt;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      compute_cnt <= '0;
    else if (state == TILE_LOAD_CFG || state == TILE_WAIT_READY)
      compute_cnt <= '0;
    else if (state == TILE_RUN_COMPUTE && compute_cnt < compute_total)
      compute_cnt <= compute_cnt + 1;
  end

  // If valid_h*valid_w==0, still leave RUN_COMPUTE (old: && (compute_total!='0) deadlocked here).
  assign compute_done_sig = (compute_cnt >= compute_total);

  // Compute one spatial position per cycle, ALL output channels
  // Input indexing: ih = oh*stride + kr - pad_top_eff (accounts for same-padding)
  // If ih < 0 or >= in_h_actual: zero-padded
  always_ff @(posedge clk) begin
    if (state == TILE_RUN_COMPUTE && compute_cnt < compute_total) begin
      automatic int oh, ow, och, ic, kr, kc;
      automatic int ih, iw;
      automatic int cin_local, kh_local, kw_local, sh_local, sw_local, cout_local;
      automatic int in_h_actual, in_w_actual;
      automatic int pad_t, pad_l;
      automatic int in_idx, wt_idx;
      automatic int sp;
      automatic logic signed [31:0] acc;

      oh = int'(compute_cnt) / int'(t_valid_w);
      ow = int'(compute_cnt) % int'(t_valid_w);
      sp = int'(compute_cnt);
      cin_local = int'(t_cin);
      cout_local = int'(t_cout);
      kh_local  = int'(t_kh);
      kw_local  = int'(t_kw);
      sh_local  = int'(t_sh);
      sw_local  = int'(t_sw);
      pad_t = int'(t_pad_top_eff);
      pad_l = int'(t_pad_left_eff);
      in_h_actual = (int'(t_valid_h) - 1) * sh_local + kh_local;
      in_w_actual = (int'(t_valid_w) - 1) * sw_local + kw_local;

      for (och = 0; och < cout_local && och < BH_MAX_COUT; och++) begin
        acc = 32'sd0;
        for (ic = 0; ic < cin_local && ic < 512; ic++) begin
          for (kr = 0; kr < kh_local && kr < 8; kr++) begin
            for (kc = 0; kc < kw_local && kc < 8; kc++) begin
              ih = oh * sh_local + kr - pad_t;
              iw = ow * sw_local + kc - pad_l;
              if (ih >= 0 && ih < in_h_actual && iw >= 0 && iw < in_w_actual) begin
                in_idx = ih * in_w_actual * cin_local + iw * cin_local + ic;
                wt_idx = och * cin_local * kh_local * kw_local
                       + ic * kh_local * kw_local + kr * kw_local + kc;
                if (in_idx >= 0 && in_idx < BH_BUF_SIZE &&
                    wt_idx >= 0 && wt_idx < BH_BUF_SIZE) begin
                  acc = acc + 32'($signed(bh_input[in_idx]) * $signed(bh_weight[wt_idx]));
                end
              end
            end
          end
        end
        bh_psum[sp * BH_MAX_COUT + och] <= acc;
      end
    end
  end

  // ═══════════════════════════════════════════════════════════
  //  POST_PROCESS — RTL ppu.sv (4-stage pipeline) + sequencer
  //  Feeds 32 channels per beat from bh_psum (from conv); cout may exceed 32.
  //  Inter-issue gap = 4 cycles to match pipeline depth (no structural hazard).
  // ═══════════════════════════════════════════════════════════
  wire [15:0] ppu_n_blk   = (16'(t_cout) + 16'd31) >> 5;  // ceil(cout/32), min 1 if cout>0
  wire [15:0] ppu_n_blk_x = (t_cout == '0) ? 16'd1 : ppu_n_blk;
  wire [15:0] ppu_ops_total = compute_total * ppu_n_blk_x;

  logic [15:0] ppu_sent;
  logic [15:0] ppu_recv;
  logic [2:0]  ppu_stall_cnt;
  logic        act_valid_rtl;

  // Combinational valid so PPU samples psum_in + psum_valid in the same cycle (registered valid was +1 cyc late).
  wire ppu_psum_valid = (state == TILE_POST_PROCESS) && fsm_ppu_en
      && (ppu_stall_cnt == 3'd0) && (ppu_sent < ppu_ops_total);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ppu_sent       <= '0;
      ppu_recv       <= '0;
      ppu_stall_cnt  <= '0;
    end else begin
      if (state != TILE_POST_PROCESS) begin
        ppu_sent       <= '0;
        ppu_recv       <= '0;
        ppu_stall_cnt  <= '0;
      end else if (fsm_ppu_en) begin
        if (act_valid_rtl) begin
          automatic int unsigned sp_cap, cb_cap, och0;
          automatic int unsigned k;
          k   = int'(ppu_recv);
          sp_cap = k / int'(ppu_n_blk_x);
          cb_cap = k % int'(ppu_n_blk_x);
          och0   = cb_cap * LANES;
          for (int l = 0; l < LANES; l++) begin
            automatic int unsigned och, dst;
            och = och0 + l;
            if (och < int'(t_cout)) begin
              dst = int'(sp_cap) * int'(t_cout) + och;
              if (dst < BH_MAX_SPATIAL * BH_MAX_COUT)
                bh_act_out[dst] <= ppu_act_out[l];
            end
          end
          ppu_recv <= ppu_recv + 16'd1;
        end

        if (ppu_stall_cnt != 3'd0)
          ppu_stall_cnt <= ppu_stall_cnt - 3'd1;
        else if (ppu_sent < ppu_ops_total) begin
          ppu_sent      <= ppu_sent + 16'd1;
          ppu_stall_cnt <= 3'd4;
        end
      end
    end
  end

  assign ppu_done_sig = (ppu_recv >= ppu_ops_total) && (state == TILE_POST_PROCESS);

  // Linear op index → (spatial, channel-block) for the beat being issued (ppu_sent before +1)
  wire [15:0] ppu_issue_lin = ppu_sent;
  wire [15:0] ppu_iss_sp    = ppu_issue_lin / ppu_n_blk_x;
  wire [15:0] ppu_iss_cb    = ppu_issue_lin - ppu_iss_sp * ppu_n_blk_x;

  logic signed [31:0] ppu_psum_in [LANES];
  logic signed [31:0] ppu_bias_lane [LANES];
  logic signed [31:0] ppu_m_int_lane [LANES];
  logic [5:0]         ppu_shift_lane [LANES];
  logic signed [7:0]  ppu_ewise_in [LANES];

  always_comb begin
    for (int l = 0; l < LANES; l++) begin
      automatic int unsigned och;
      automatic int unsigned base_psum;
      automatic int unsigned skip_i;
      och       = int'(ppu_iss_cb) * LANES + l;
      base_psum = int'(ppu_iss_sp) * BH_MAX_COUT + och;
      if (och < int'(t_cout))
        ppu_psum_in[l] = bh_psum[base_psum];
      else
        ppu_psum_in[l] = 32'sd0;

      if (och < BH_MAX_COUT) begin
        ppu_bias_lane[l]  = bh_bias_all[och];
        ppu_m_int_lane[l] = bh_m_int_all[och];
        ppu_shift_lane[l] = bh_shift_all[och];
      end else begin
        ppu_bias_lane[l]  = 32'sd0;
        ppu_m_int_lane[l] = 32'sd1;
        ppu_shift_lane[l] = 6'd0;
      end

      skip_i = int'(ppu_iss_sp) * int'(t_cout) + och;
      if (bh_ewise_en && och < int'(t_cout) && skip_i < BH_BUF_SIZE)
        ppu_ewise_in[l] = $signed(bh_skip[skip_i]);
      else
        ppu_ewise_in[l] = 8'sd0;
    end
  end

  logic signed [7:0] ppu_act_out [LANES];

  ppu #(.LANES(LANES)) u_ppu (
    .clk           (clk),
    .rst_n         (rst_n),
    .en            (fsm_ppu_en),
    .cfg_post      (cfg_post),
    .cfg_mode      (cfg_mode),
    .psum_in       (ppu_psum_in),
    .psum_valid    (ppu_psum_valid),
    .bias_val      (ppu_bias_lane),
    .m_int         (ppu_m_int_lane),
    .shift         (ppu_shift_lane),
    .zp_out        (ppu_zp_out),
    .silu_lut_data (ppu_silu_lut),
    .ewise_in      (ppu_ewise_in),
    .ewise_valid   (ppu_psum_valid & cfg_post.ewise_en),
    .act_out       (ppu_act_out),
    .act_valid     (act_valid_rtl)
  );

  // ═══════════════════════════════════════════════════════════
  //  Output GLB Bank (kept for interface compatibility, used minimally)
  // ═══════════════════════════════════════════════════════════
  logic                glb_out_wr_en;
  logic [8:0]          glb_out_wr_addr;
  namespace_e          glb_out_wr_ns;
  logic [LANES*32-1:0] glb_out_wr_psum;
  logic [LANES*8-1:0]  glb_out_wr_act;
  logic                glb_out_rd_en;
  logic [8:0]          glb_out_rd_addr;
  namespace_e          glb_out_rd_ns;
  logic [LANES*32-1:0] glb_out_rd_psum;
  logic [LANES*8-1:0]  glb_out_rd_act;

  glb_output_bank #(.LANES(LANES)) u_glb_out (
    .clk          (clk),
    .rst_n        (rst_n),
    .wr_en        (glb_out_wr_en),
    .wr_addr      (glb_out_wr_addr),
    .wr_ns        (glb_out_wr_ns),
    .wr_data_psum (glb_out_wr_psum),
    .wr_data_act  (glb_out_wr_act),
    .rd_en        (glb_out_rd_en),
    .rd_addr      (glb_out_rd_addr),
    .rd_ns        (glb_out_rd_ns),
    .rd_data_psum (glb_out_rd_psum),
    .rd_data_act  (glb_out_rd_act)
  );

  // GLB is not used for the behavioral path but kept to avoid synthesis warnings
  assign glb_out_wr_en   = 1'b0;
  assign glb_out_wr_addr = '0;
  assign glb_out_wr_ns   = NS_PSUM;
  assign glb_out_wr_psum = '0;
  assign glb_out_wr_act  = '0;
  assign glb_out_rd_en   = 1'b0;
  assign glb_out_rd_addr = '0;
  assign glb_out_rd_ns   = NS_PSUM;

  // ═══════════════════════════════════════════════════════════
  //  SWIZZLE_STORE Phase — DMA Write from bh_act_out
  // ═══════════════════════════════════════════════════════════
  logic [15:0] store_cnt;

  // store_active was a 1-cycle-late FF: ext_wr_grant could align with tensor_dma WR_DATA
  // while store_active was still 0 → no W beats, no AW traffic past TB audit, tiles stuck.
  // Drive write strobes combinationally for the whole SWIZZLE_STORE+spill window.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      store_cnt <= '0;
    else if (state != TILE_SWIZZLE_STORE)
      store_cnt <= '0;
    else if (ext_wr_beat)
      store_cnt <= store_cnt + 1;
  end

  assign swizzle_done_sig = 1'b1;

  // Pack 32 bytes from bh_act_out per DMA beat (HWC order)
  logic [255:0] store_data;
  always_comb begin
    store_data = '0;
    for (int i = 0; i < 32; i++) begin
      automatic int byte_idx;
      byte_idx = int'(store_cnt) * 32 + i;
      if (byte_idx < int'(t_valid_h) * int'(t_valid_w) * int'(t_cout))
        store_data[i*8 +: 8] = bh_act_out[byte_idx];
    end
  end

  // ═══════════════════════════════════════════════════════════
  //  External Port Mapping
  // ═══════════════════════════════════════════════════════════
  assign ext_rd_req  = fsm_dma_rd_req;
  assign ext_rd_addr = fsm_dma_rd_addr;
  assign ext_rd_len  = fsm_dma_rd_len;

  assign ext_wr_req  = fsm_dma_wr_req;
  assign ext_wr_addr = fsm_dma_wr_addr;
  assign ext_wr_len  = fsm_dma_wr_len;

  assign ext_wr_data  = store_data;
  assign ext_wr_valid = (state == TILE_SWIZZLE_STORE) && fsm_dma_wr_req && ext_wr_grant;

  assign barrier_signal_id = tile_desc.layer_id;

endmodule
