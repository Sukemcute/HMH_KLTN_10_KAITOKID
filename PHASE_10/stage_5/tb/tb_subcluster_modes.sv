`timescale 1ns/1ps
// ============================================================================
// TB: tb_subcluster_modes
// Unified testbench for subcluster_datapath — verifies that THE SAME HARDWARE
// produces correct results for each pe_mode by descriptor configuration alone.
//
// Test matrix:
//   TEST 1: PE_RS3  — Conv 3x3 (L0 style)
//   TEST 2: PE_OS1  — Conv 1x1 (QC2f cv1 style)
//   TEST 3: PE_DW3  — Depthwise 3x3 (SCDown cv2 style)
//   TEST 4: PE_MP5  — MaxPool 5x5 (SPPF style)
//   TEST 5: PE_PASS — Upsample 2x
//
// Flow per test:
//   1. Fill GLB banks via ext_wr ports (simulating DMA preload)
//   2. Configure descriptor for target mode
//   3. Assert tile_valid -> subcluster computes -> wait tile_done
//   4. Read output via ext_rd ports
//   5. Compare vs golden (behavioral model in testbench)
// ============================================================================
module tb_subcluster_modes;
  import accel_pkg::*;
  import desc_pkg::*;

  // ══════════════════════════════════════════════════════════════
  // Parameters
  // ══════════════════════════════════════════════════════════════
  localparam int LANES          = 32;
  localparam int PE_ROWS        = 3;
  localparam int PE_COLS        = 4;
  localparam int INPUT_BANKS    = 3;
  localparam int WEIGHT_BANKS   = 3;
  localparam int OUTPUT_BANKS   = 4;
  localparam int IN_BANK_DEPTH  = 2048;
  localparam int WT_BANK_DEPTH  = 1024;
  localparam int OUT_BANK_DEPTH = 512;
  localparam int CLK_HP         = 5;  // 200 MHz => 5ns half-period

  // ══════════════════════════════════════════════════════════════
  // DUT signals
  // ══════════════════════════════════════════════════════════════
  logic              clk, rst_n;

  // Descriptor input
  logic              tile_valid;
  tile_desc_t        tile_desc_in;
  layer_desc_t       layer_desc_in;
  post_profile_t     post_profile_in;
  router_profile_t   router_profile_in;
  logic              tile_accept;

  // External data ports
  logic              ext_wr_en;
  logic [1:0]        ext_wr_target;
  logic [1:0]        ext_wr_bank_id;
  logic [15:0]       ext_wr_addr;
  logic [LANES*8-1:0] ext_wr_data;
  logic [LANES-1:0]  ext_wr_mask;

  logic              ext_rd_en;
  logic [1:0]        ext_rd_bank_id;
  logic [15:0]       ext_rd_addr;
  logic              ext_rd_ns_psum;
  logic [LANES*8-1:0]  ext_rd_data_act;
  logic [LANES*32-1:0] ext_rd_data_psum;

  // Bias/Quant params
  logic signed [31:0] bias_mem  [256];
  logic signed [31:0] m_int_mem [256];
  logic [5:0]         shift_mem [256];
  logic signed [7:0]  zp_out_val;

  // SiLU LUT
  logic signed [7:0]  silu_lut_data [256];

  // Status
  logic              tile_done;
  logic              layer_done;
  tile_state_e       fsm_state;

  // ══════════════════════════════════════════════════════════════
  // DUT Instantiation
  // ══════════════════════════════════════════════════════════════
  subcluster_datapath #(
    .LANES         (LANES),
    .PE_ROWS       (PE_ROWS),
    .PE_COLS       (PE_COLS),
    .INPUT_BANKS   (INPUT_BANKS),
    .WEIGHT_BANKS  (WEIGHT_BANKS),
    .OUTPUT_BANKS  (OUTPUT_BANKS),
    .IN_BANK_DEPTH (IN_BANK_DEPTH),
    .WT_BANK_DEPTH (WT_BANK_DEPTH),
    .OUT_BANK_DEPTH(OUT_BANK_DEPTH)
  ) u_dut (
    .clk              (clk),
    .rst_n            (rst_n),
    .tile_valid       (tile_valid),
    .tile_desc_in     (tile_desc_in),
    .layer_desc_in    (layer_desc_in),
    .post_profile_in  (post_profile_in),
    .router_profile_in(router_profile_in),
    .tile_accept      (tile_accept),
    .ext_wr_en        (ext_wr_en),
    .ext_wr_target    (ext_wr_target),
    .ext_wr_bank_id   (ext_wr_bank_id),
    .ext_wr_addr      (ext_wr_addr),
    .ext_wr_data      (ext_wr_data),
    .ext_wr_mask      (ext_wr_mask),
    .ext_rd_en        (ext_rd_en),
    .ext_rd_bank_id   (ext_rd_bank_id),
    .ext_rd_addr      (ext_rd_addr),
    .ext_rd_ns_psum   (ext_rd_ns_psum),
    .ext_rd_data_act  (ext_rd_data_act),
    .ext_rd_data_psum (ext_rd_data_psum),
    .bias_mem         (bias_mem),
    .m_int_mem        (m_int_mem),
    .shift_mem        (shift_mem),
    .zp_out_val       (zp_out_val),
    .silu_lut_data    (silu_lut_data),
    .tile_done        (tile_done),
    .layer_done       (layer_done),
    .fsm_state        (fsm_state)
  );

  // ══════════════════════════════════════════════════════════════
  // Clock generation
  // ══════════════════════════════════════════════════════════════
  initial clk = 0;
  always #(CLK_HP) clk = ~clk;

  // ══════════════════════════════════════════════════════════════
  // Global test tracking
  // ══════════════════════════════════════════════════════════════
  int total_tests, total_pass, total_fail;
  string test_names [5];
  string test_results [5];
  real   test_match_pct [5];
  int    test_max_diff [5];

  initial begin
    total_tests = 0;
    total_pass  = 0;
    total_fail  = 0;
  end

  // ══════════════════════════════════════════════════════════════
  // Helper tasks
  // ══════════════════════════════════════════════════════════════

  // ────────── Reset ──────────
  task automatic do_reset();
    rst_n      <= 1'b0;
    tile_valid <= 1'b0;
    ext_wr_en  <= 1'b0;
    ext_rd_en  <= 1'b0;
    zp_out_val <= 8'sd0;
    for (int i = 0; i < 256; i++) begin
      bias_mem[i]  = 32'sd0;
      m_int_mem[i] = 32'sd1;
      shift_mem[i] = 6'd0;
      silu_lut_data[i] = 8'(i) - 8'sd128;
    end
    tile_desc_in    = '0;
    layer_desc_in   = '0;
    post_profile_in = '0;
    router_profile_in = '0;
    repeat (6) @(posedge clk);
    rst_n <= 1'b1;
    repeat (4) @(posedge clk);
  endtask

  // ────────── Fill input bank ──────────
  // Writes one 256-bit (32-byte) word at given (bank_id, addr)
  task automatic fill_input_bank(
    input logic [1:0] bank_id,
    input logic [15:0] addr,
    input logic [LANES*8-1:0] data
  );
    @(posedge clk);
    ext_wr_en      <= 1'b1;
    ext_wr_target  <= 2'd0;  // input bank
    ext_wr_bank_id <= bank_id;
    ext_wr_addr    <= addr;
    ext_wr_data    <= data;
    ext_wr_mask    <= {LANES{1'b1}};
    @(posedge clk);
    ext_wr_en      <= 1'b0;
  endtask

  // ────────── Fill weight bank ──────────
  task automatic fill_weight_bank(
    input logic [1:0] bank_id,
    input logic [15:0] addr,
    input logic [LANES*8-1:0] data
  );
    @(posedge clk);
    ext_wr_en      <= 1'b1;
    ext_wr_target  <= 2'd1;  // weight bank
    ext_wr_bank_id <= bank_id;
    ext_wr_addr    <= addr;
    ext_wr_data    <= data;
    ext_wr_mask    <= {LANES{1'b1}};
    @(posedge clk);
    ext_wr_en      <= 1'b0;
  endtask

  // ────────── Fill output bank (for skip data) ──────────
  task automatic fill_output_bank(
    input logic [1:0] bank_id,
    input logic [15:0] addr,
    input logic [LANES*8-1:0] data
  );
    @(posedge clk);
    ext_wr_en      <= 1'b1;
    ext_wr_target  <= 2'd2;  // output bank (skip)
    ext_wr_bank_id <= bank_id;
    ext_wr_addr    <= addr;
    ext_wr_data    <= data;
    ext_wr_mask    <= {LANES{1'b1}};
    @(posedge clk);
    ext_wr_en      <= 1'b0;
  endtask

  // ────────── Read output bank (ACT namespace) ──────────
  task automatic read_output_bank_act(
    input  logic [1:0]  bank_id,
    input  logic [15:0] addr,
    output logic [LANES*8-1:0] data
  );
    @(posedge clk);
    ext_rd_en      <= 1'b1;
    ext_rd_bank_id <= bank_id;
    ext_rd_addr    <= addr;
    ext_rd_ns_psum <= 1'b0;  // ACT namespace
    @(posedge clk);
    ext_rd_en      <= 1'b0;
    @(posedge clk);  // 1-cycle read latency (registered output)
    data = ext_rd_data_act;
  endtask

  // ────────── Read output bank (PSUM namespace) ──────────
  task automatic read_output_bank_psum(
    input  logic [1:0]  bank_id,
    input  logic [15:0] addr,
    output logic [LANES*32-1:0] data
  );
    @(posedge clk);
    ext_rd_en      <= 1'b1;
    ext_rd_bank_id <= bank_id;
    ext_rd_addr    <= addr;
    ext_rd_ns_psum <= 1'b1;  // PSUM namespace
    @(posedge clk);
    ext_rd_en      <= 1'b0;
    @(posedge clk);  // 1-cycle read latency
    data = ext_rd_data_psum;
  endtask

  // ────────── Launch tile and wait for done ──────────
  task automatic launch_tile_and_wait(input int timeout_cycles = 50000);
    int cyc_cnt;
    // Assert tile_valid
    @(posedge clk);
    tile_valid <= 1'b1;
    @(posedge clk);
    // Wait for tile_accept (IDLE -> LOAD_CFG transition)
    while (!tile_accept) begin
      @(posedge clk);
    end
    tile_valid <= 1'b0;
    // Wait for tile_done
    cyc_cnt = 0;
    while (!tile_done) begin
      @(posedge clk);
      cyc_cnt++;
      if (cyc_cnt >= timeout_cycles) begin
        $display("  [ERROR] Tile execution TIMEOUT after %0d cycles! FSM state=%0d", cyc_cnt, fsm_state);
        return;
      end
    end
    $display("  Tile completed in %0d cycles", cyc_cnt);
    @(posedge clk);  // Let DONE settle
  endtask

  // ────────── Build default descriptor ──────────
  task automatic build_default_descriptors();
    tile_desc_in    = '0;
    layer_desc_in   = '0;
    post_profile_in = '0;
    router_profile_in = '0;
    // tile flags: first_tile=1, last_tile=1, need_swizzle=0, need_spill=0
    tile_desc_in.tile_flags = 16'h0003;
    tile_desc_in.valid_h    = 6'd4;
    tile_desc_in.valid_w    = 6'd1;  // wblk count
    // Router: default direct mapping (bank0->row0, bank1->row1, bank2->row2)
    router_profile_in.rin_src = '{3'd0, 3'd1, 3'd2};
    router_profile_in.rwt_src = '{3'd0, 3'd1, 3'd2};
    router_profile_in.rin_dst_mask = '{4'hF, 4'hF, 4'hF};
    router_profile_in.rwt_h_multicast = 1'b0;
    router_profile_in.rps_accum_mode = 2'd0;
    router_profile_in.concat_offset_mode = 1'b0;
    router_profile_in.upsample_dup_mode = 1'b0;
  endtask

  // ────────── Clamp INT32 to INT8 ──────────
  function automatic logic signed [7:0] clamp_i8(input logic signed [31:0] val);
    if (val > 32'sd127) return 8'sd127;
    if (val < -32'sd128) return -8'sd128;
    return val[7:0];
  endfunction

  // ────────── PPU golden model (matches RTL 4-stage pipeline) ──────────
  // Performs: bias add -> requant -> activation -> zp_out clamp
  function automatic logic signed [7:0] ppu_golden(
    input logic signed [31:0] psum,
    input logic signed [31:0] bias,
    input logic signed [31:0] m,
    input int                 sh,
    input act_mode_e          act,
    input logic               bias_en
  );
    automatic logic signed [31:0] biased;
    automatic logic signed [63:0] mult, rounded;
    automatic logic signed [31:0] shifted;
    automatic logic signed [15:0] y_raw;
    automatic logic signed [7:0]  y_act;
    automatic logic signed [15:0] y_final;

    // Stage 1: Bias
    if (bias_en)
      biased = psum + bias;
    else
      biased = psum;

    // Stage 2: Requant
    mult = 64'(biased) * 64'(m);
    if (sh > 0)
      rounded = mult + (64'sd1 <<< (sh - 1));
    else
      rounded = mult;
    shifted = 32'(rounded >>> sh);
    if (shifted > 32'sd32767) y_raw = 16'sd32767;
    else if (shifted < -32'sd32768) y_raw = -16'sd32768;
    else y_raw = shifted[15:0];

    // Stage 3: Activation
    case (act)
      ACT_RELU: begin
        if (y_raw > 0) begin
          if (y_raw > 16'sd127) y_act = 8'sd127;
          else y_act = y_raw[7:0];
        end else
          y_act = 8'sd0;
      end
      default: begin // ACT_NONE/CLAMP
        if (y_raw > 16'sd127) y_act = 8'sd127;
        else if (y_raw < -16'sd128) y_act = -8'sd128;
        else y_act = y_raw[7:0];
      end
    endcase

    // Stage 4: zp_out + final clamp
    y_final = 16'(y_act) + 16'(zp_out_val);
    if (y_final > 16'sd127) return 8'sd127;
    else if (y_final < -16'sd128) return -8'sd128;
    else return y_final[7:0];
  endfunction

  // ══════════════════════════════════════════════════════════════
  // TEST 1: PE_RS3 — Conv 3x3
  // Config: Cin=3, Cout=4, Hin=6, Win=34(padded), Hout=2, Wout=16,
  //         stride=2, pad=1, kh=3, kw=3, act=ReLU
  // ══════════════════════════════════════════════════════════════
  task automatic test_pe_rs3();
    // Local config
    localparam int CIN  = 3;
    localparam int COUT = 4;
    localparam int HIN  = 6;
    localparam int WIN  = 34;  // padded width (fits in 2 wblks: 64)
    localparam int HOUT = 2;
    localparam int WOUT = 16;
    localparam int KH   = 3;
    localparam int KW   = 3;
    localparam int SH   = 2;
    localparam int SW   = 2;
    localparam int WBLK_TOTAL = (WIN + LANES - 1) / LANES;  // 2

    // Storage for golden input and weights
    logic signed [7:0] input_data [HIN][WBLK_TOTAL][CIN][LANES];
    logic signed [7:0] weight_data [COUT][CIN][KH][KW];
    logic signed [31:0] bias_data [COUT];
    logic signed [31:0] m_int_data [COUT];
    int shift_data [COUT];
    logic signed [7:0] golden_out [HOUT][WOUT][COUT];
    logic signed [7:0] dut_out [HOUT][WOUT][COUT];

    int match_cnt, total_cnt, max_diff;

    $display("\n================================================================");
    $display("  TEST 1: PE_RS3 (Conv 3x3) -- L0 style");
    $display("================================================================");

    // Initialize input data: deterministic pattern
    for (int h = 0; h < HIN; h++)
      for (int wb = 0; wb < WBLK_TOTAL; wb++)
        for (int c = 0; c < CIN; c++)
          for (int l = 0; l < LANES; l++) begin
            int w_abs;
            w_abs = wb * LANES + l;
            if (w_abs < WIN)
              input_data[h][wb][c][l] = 8'((l * CIN + c + h + 1) % 127);
            else
              input_data[h][wb][c][l] = 8'sd0;
          end

    // Initialize weights: small known values
    for (int co = 0; co < COUT; co++)
      for (int ci = 0; ci < CIN; ci++)
        for (int kh_i = 0; kh_i < KH; kh_i++)
          for (int kw_i = 0; kw_i < KW; kw_i++)
            weight_data[co][ci][kh_i][kw_i] = 8'((co * 7 + ci * 3 + kh_i * 2 + kw_i + 1) % 5) - 8'sd2;

    // Bias and quant params
    for (int co = 0; co < COUT; co++) begin
      bias_data[co]  = 32'(co * 10);
      m_int_data[co] = 32'sd1;    // identity scale
      shift_data[co] = 0;         // no shift
    end

    // Fill PPU params
    for (int i = 0; i < 256; i++) begin
      bias_mem[i]  = (i < COUT) ? bias_data[i]  : 32'sd0;
      m_int_mem[i] = (i < COUT) ? m_int_data[i] : 32'sd1;
      shift_mem[i] = (i < COUT) ? shift_data[i] : 6'd0;
    end
    zp_out_val = 8'sd0;

    // Fill input banks: bank_id = h % 3
    // Address layout: (row_slot * cin_tile + c) * wblk_total + wblk
    // row_slot = (h/3) % q_in,  q_in = ceil(hin/3) = 2
    begin
      int q_in_val;
      q_in_val = (HIN + 2) / 3;  // ceil(6/3) = 2
      for (int h = 0; h < HIN; h++) begin
        int bank_id, row_slot;
        bank_id  = h % 3;
        row_slot = (h / 3) % q_in_val;
        for (int c = 0; c < CIN; c++) begin
          for (int wb = 0; wb < WBLK_TOTAL; wb++) begin
            logic [15:0] addr;
            logic [LANES*8-1:0] packed_data;
            addr = 16'((row_slot * CIN + c) * WBLK_TOTAL + wb);
            packed_data = '0;
            for (int l = 0; l < LANES; l++)
              packed_data[l*8 +: 8] = input_data[h][wb][c][l];
            fill_input_bank(bank_id[1:0], addr, packed_data);
          end
        end
      end
    end

    // Fill weight banks: for RS3, weight[co][ci][kh] goes to weight_bank[kh%3]
    // Each PE row r processes kernel row r (weight[co][ci][r][kw] packed in 32 lanes)
    // Simplified: pack kw*cin weights linearly per cout into weight bank
    for (int row = 0; row < KH; row++) begin
      for (int co = 0; co < COUT; co++) begin
        logic [LANES*8-1:0] packed_wt;
        packed_wt = '0;
        for (int ci = 0; ci < CIN; ci++) begin
          for (int kw_i = 0; kw_i < KW; kw_i++) begin
            int lane_idx;
            lane_idx = ci * KW + kw_i;
            if (lane_idx < LANES)
              packed_wt[lane_idx*8 +: 8] = weight_data[co][ci][row][kw_i];
          end
        end
        fill_weight_bank(row[1:0], 16'(co), packed_wt);
      end
    end

    // Compute golden output
    for (int ho = 0; ho < HOUT; ho++) begin
      for (int wo = 0; wo < WOUT; wo++) begin
        for (int co = 0; co < COUT; co++) begin
          automatic logic signed [31:0] acc;
          acc = 32'sd0;
          for (int ci = 0; ci < CIN; ci++)
            for (int kh_i = 0; kh_i < KH; kh_i++)
              for (int kw_i = 0; kw_i < KW; kw_i++) begin
                int ih, iw, wb_idx;
                logic signed [7:0] x_val, w_val;
                ih = ho * SH + kh_i;
                iw = wo * SW + kw_i;
                if (ih >= 0 && ih < HIN && iw >= 0 && iw < WIN) begin
                  wb_idx = iw / LANES;
                  x_val = input_data[ih][wb_idx][ci][iw % LANES];
                end else
                  x_val = 8'sd0;
                w_val = weight_data[co][ci][kh_i][kw_i];
                acc = acc + 32'(x_val) * 32'(w_val);
              end
          golden_out[ho][wo][co] = ppu_golden(acc, bias_data[co], m_int_data[co],
                                              shift_data[co], ACT_RELU, 1'b1);
        end
      end
    end

    // Configure descriptors
    build_default_descriptors();
    layer_desc_in.template_id  = 4'(PE_RS3);
    layer_desc_in.tile_cin     = 8'(CIN);
    layer_desc_in.tile_cout    = 8'(COUT);
    layer_desc_in.hin          = 10'(HIN);
    layer_desc_in.win          = 10'(WIN);
    layer_desc_in.hout         = 10'(HOUT);
    layer_desc_in.wout         = 10'(WOUT);
    layer_desc_in.kh           = 4'(KH);
    layer_desc_in.kw           = 4'(KW);
    layer_desc_in.sh           = 3'(SH);
    layer_desc_in.sw           = 3'(SW);
    layer_desc_in.pad_top      = 4'd0;
    layer_desc_in.pad_bot      = 4'd0;
    layer_desc_in.pad_left     = 4'd0;
    layer_desc_in.pad_right    = 4'd0;
    layer_desc_in.num_cin_pass = 4'd1;
    layer_desc_in.num_k_pass   = 4'd1;
    layer_desc_in.q_in         = 4'd2;
    layer_desc_in.q_out        = 4'd1;
    layer_desc_in.cin_total    = 9'(CIN);
    layer_desc_in.cout_total   = 9'(COUT);

    tile_desc_in.valid_h = 6'(HOUT);
    tile_desc_in.valid_w = 6'(WBLK_TOTAL);
    tile_desc_in.tile_flags = 16'h0003; // first_tile=1, last_tile=1

    post_profile_in.bias_en    = 1'b1;
    post_profile_in.quant_mode = QMODE_PER_CHANNEL;
    post_profile_in.act_mode   = ACT_RELU;
    post_profile_in.ewise_en   = 1'b0;
    post_profile_in.upsample_factor   = 2'd0;
    post_profile_in.concat_ch_offset  = 8'd0;

    // Launch tile
    launch_tile_and_wait(100000);

    // Read output from output banks and compare
    // Output stored in output bank[0], PSUM namespace for raw, or ACT after PPU
    // Since PPU writes to output bank, we read from ACT namespace
    match_cnt = 0;
    total_cnt = 0;
    max_diff  = 0;

    // Read output from output banks - output is stored sequentially
    // For a simple check, read back PSUM from output bank col 0
    for (int ho = 0; ho < HOUT; ho++) begin
      for (int wo_blk = 0; wo_blk < (WOUT + LANES - 1) / LANES; wo_blk++) begin
        for (int co = 0; co < COUT; co++) begin
          logic [LANES*32-1:0] psum_word;
          read_output_bank_psum(2'(co % OUTPUT_BANKS), 16'(ho * WBLK_TOTAL + wo_blk), psum_word);
          for (int l = 0; l < LANES; l++) begin
            int wo_abs;
            logic signed [31:0] got_psum;
            logic signed [7:0]  got_act, exp_act;
            int diff;
            wo_abs = wo_blk * LANES + l;
            if (wo_abs < WOUT) begin
              got_psum = $signed(psum_word[l*32 +: 32]);
              exp_act  = golden_out[ho][wo_abs][co];
              got_act  = ppu_golden(got_psum, bias_data[co], m_int_data[co],
                                    shift_data[co], ACT_RELU, 1'b1);
              diff = int'(got_act) - int'(exp_act);
              if (diff < 0) diff = -diff;
              if (diff > max_diff) max_diff = diff;
              if (diff <= 1) match_cnt++;
              total_cnt++;
            end
          end
        end
      end
    end

    // Report
    begin
      real pct;
      pct = (total_cnt > 0) ? (real'(match_cnt) / real'(total_cnt) * 100.0) : 0.0;
      test_names[0] = "PE_RS3 (Conv 3x3)";
      test_match_pct[0] = pct;
      test_max_diff[0] = max_diff;
      if (pct >= 99.9) begin
        $display("  [PASS] PE_RS3: %0d/%0d match (%.1f%%), max LSB diff = %0d",
                 match_cnt, total_cnt, pct, max_diff);
        test_results[0] = "PASS";
        total_pass++;
      end else begin
        $display("  [FAIL] PE_RS3: %0d/%0d match (%.1f%%), max LSB diff = %0d",
                 match_cnt, total_cnt, pct, max_diff);
        test_results[0] = "FAIL";
        total_fail++;
      end
      total_tests++;
    end
  endtask

  // ══════════════════════════════════════════════════════════════
  // TEST 2: PE_OS1 — Conv 1x1
  // Config: Cin=4, Cout=8, H=4, W=32, stride=1, pad=0, act=ReLU
  // ══════════════════════════════════════════════════════════════
  task automatic test_pe_os1();
    localparam int CIN  = 4;
    localparam int COUT = 8;
    localparam int H    = 4;
    localparam int W    = 32;
    localparam int WBLK = (W + LANES - 1) / LANES;  // 1

    logic signed [7:0] input_data [H][WBLK][CIN][LANES];
    logic signed [7:0] weight_data [COUT][CIN];
    logic signed [31:0] bias_data [COUT];
    logic signed [31:0] m_int_data [COUT];
    int shift_data [COUT];
    logic signed [7:0] golden_out [H][W][COUT];

    int match_cnt, total_cnt, max_diff;

    $display("\n================================================================");
    $display("  TEST 2: PE_OS1 (Conv 1x1) -- QC2f cv1 style");
    $display("================================================================");

    // Init input
    for (int h = 0; h < H; h++)
      for (int wb = 0; wb < WBLK; wb++)
        for (int c = 0; c < CIN; c++)
          for (int l = 0; l < LANES; l++)
            input_data[h][wb][c][l] = 8'((h * 13 + l * 3 + c + 5) % 100);

    // Init weights (broadcast to all lanes)
    for (int co = 0; co < COUT; co++)
      for (int ci = 0; ci < CIN; ci++)
        weight_data[co][ci] = 8'((co * 5 + ci * 3 + 1) % 7) - 8'sd3;

    // Bias and quant
    for (int co = 0; co < COUT; co++) begin
      bias_data[co]  = 32'(co * 5);
      m_int_data[co] = 32'sd1;
      shift_data[co] = 0;
    end
    for (int i = 0; i < 256; i++) begin
      bias_mem[i]  = (i < COUT) ? bias_data[i]  : 32'sd0;
      m_int_mem[i] = (i < COUT) ? m_int_data[i] : 32'sd1;
      shift_mem[i] = (i < COUT) ? shift_data[i] : 6'd0;
    end
    zp_out_val = 8'sd0;

    // Fill input banks
    begin
      int q_in_val;
      q_in_val = (H + 2) / 3;
      for (int h = 0; h < H; h++) begin
        int bank_id, row_slot;
        bank_id  = h % 3;
        row_slot = (h / 3) % q_in_val;
        for (int c = 0; c < CIN; c++) begin
          for (int wb = 0; wb < WBLK; wb++) begin
            logic [15:0] addr;
            logic [LANES*8-1:0] packed_data;
            addr = 16'((row_slot * CIN + c) * WBLK + wb);
            packed_data = '0;
            for (int l = 0; l < LANES; l++)
              packed_data[l*8 +: 8] = input_data[h][wb][c][l];
            fill_input_bank(bank_id[1:0], addr, packed_data);
          end
        end
      end
    end

    // Fill weight banks: OS1 broadcasts weight[0] to all lanes
    // weight_bank[row] addr[co] => all lanes get weight_data[co][ci_for_row]
    for (int row = 0; row < PE_ROWS; row++) begin
      for (int co = 0; co < COUT; co++) begin
        logic [LANES*8-1:0] packed_wt;
        packed_wt = '0;
        // For OS1: put weight per cin in lane positions
        for (int ci = 0; ci < CIN; ci++) begin
          if (ci < LANES)
            packed_wt[ci*8 +: 8] = weight_data[co][ci];
        end
        fill_weight_bank(row[1:0], 16'(co), packed_wt);
      end
    end

    // Golden
    for (int h = 0; h < H; h++)
      for (int w = 0; w < W; w++)
        for (int co = 0; co < COUT; co++) begin
          automatic logic signed [31:0] acc;
          acc = 32'sd0;
          for (int ci = 0; ci < CIN; ci++) begin
            int wb_idx;
            wb_idx = w / LANES;
            acc = acc + 32'(input_data[h][wb_idx][ci][w % LANES]) * 32'(weight_data[co][ci]);
          end
          golden_out[h][w][co] = ppu_golden(acc, bias_data[co], m_int_data[co],
                                            shift_data[co], ACT_RELU, 1'b1);
        end

    // Descriptors
    build_default_descriptors();
    layer_desc_in.template_id  = 4'(PE_OS1);
    layer_desc_in.tile_cin     = 8'(CIN);
    layer_desc_in.tile_cout    = 8'(COUT);
    layer_desc_in.hin          = 10'(H);
    layer_desc_in.win          = 10'(W);
    layer_desc_in.hout         = 10'(H);
    layer_desc_in.wout         = 10'(W);
    layer_desc_in.kh           = 4'd1;
    layer_desc_in.kw           = 4'd1;
    layer_desc_in.sh           = 3'd1;
    layer_desc_in.sw           = 3'd1;
    layer_desc_in.pad_top      = 4'd0;
    layer_desc_in.pad_bot      = 4'd0;
    layer_desc_in.pad_left     = 4'd0;
    layer_desc_in.pad_right    = 4'd0;
    layer_desc_in.num_cin_pass = 4'd1;
    layer_desc_in.num_k_pass   = 4'd1;
    layer_desc_in.q_in         = 4'd2;
    layer_desc_in.q_out        = 4'd1;
    layer_desc_in.cin_total    = 9'(CIN);
    layer_desc_in.cout_total   = 9'(COUT);

    tile_desc_in.valid_h    = 6'(H);
    tile_desc_in.valid_w    = 6'(WBLK);
    tile_desc_in.tile_flags = 16'h0003;

    post_profile_in.bias_en    = 1'b1;
    post_profile_in.quant_mode = QMODE_PER_CHANNEL;
    post_profile_in.act_mode   = ACT_RELU;
    post_profile_in.ewise_en   = 1'b0;
    post_profile_in.upsample_factor  = 2'd0;
    post_profile_in.concat_ch_offset = 8'd0;

    launch_tile_and_wait(100000);

    // Check: read PSUM from output banks
    match_cnt = 0;
    total_cnt = 0;
    max_diff  = 0;

    for (int h = 0; h < H; h++) begin
      for (int wb = 0; wb < WBLK; wb++) begin
        for (int co = 0; co < COUT; co++) begin
          logic [LANES*32-1:0] psum_word;
          read_output_bank_psum(2'(co % OUTPUT_BANKS), 16'(h * WBLK + wb), psum_word);
          for (int l = 0; l < LANES; l++) begin
            int w_abs;
            logic signed [31:0] got_psum;
            logic signed [7:0] got_act, exp_act;
            int diff;
            w_abs = wb * LANES + l;
            if (w_abs < W) begin
              got_psum = $signed(psum_word[l*32 +: 32]);
              exp_act  = golden_out[h][w_abs][co];
              got_act  = ppu_golden(got_psum, bias_data[co], m_int_data[co],
                                    shift_data[co], ACT_RELU, 1'b1);
              diff = int'(got_act) - int'(exp_act);
              if (diff < 0) diff = -diff;
              if (diff > max_diff) max_diff = diff;
              if (diff <= 1) match_cnt++;
              total_cnt++;
            end
          end
        end
      end
    end

    begin
      real pct;
      pct = (total_cnt > 0) ? (real'(match_cnt) / real'(total_cnt) * 100.0) : 0.0;
      test_names[1] = "PE_OS1 (Conv 1x1)";
      test_match_pct[1] = pct;
      test_max_diff[1] = max_diff;
      if (pct >= 99.0) begin
        $display("  [PASS] PE_OS1: %0d/%0d match (%.1f%%), max LSB diff = %0d",
                 match_cnt, total_cnt, pct, max_diff);
        test_results[1] = "PASS";
        total_pass++;
      end else begin
        $display("  [FAIL] PE_OS1: %0d/%0d match (%.1f%%), max LSB diff = %0d",
                 match_cnt, total_cnt, pct, max_diff);
        test_results[1] = "FAIL";
        total_fail++;
      end
      total_tests++;
    end
  endtask

  // ══════════════════════════════════════════════════════════════
  // TEST 3: PE_DW3 — Depthwise Conv 3x3
  // Config: C=4, Hin=6, Win=34, Hout=2, Wout=16, stride=2, pad=1
  // ══════════════════════════════════════════════════════════════
  task automatic test_pe_dw3();
    localparam int CH   = 4;
    localparam int HIN  = 6;
    localparam int WIN  = 34;
    localparam int HOUT = 2;
    localparam int WOUT = 16;
    localparam int KH   = 3;
    localparam int KW   = 3;
    localparam int SH   = 2;
    localparam int SW   = 2;
    localparam int PAD  = 1;
    localparam int WBLK_TOTAL = (WIN + LANES - 1) / LANES;

    logic signed [7:0] input_data [HIN][WBLK_TOTAL][CH][LANES];
    logic signed [7:0] weight_data [CH][KH][KW];
    logic signed [31:0] bias_data [CH];
    logic signed [31:0] m_int_data [CH];
    int shift_data [CH];
    logic signed [7:0] golden_out [HOUT][WOUT][CH];

    int match_cnt, total_cnt, max_diff;

    $display("\n================================================================");
    $display("  TEST 3: PE_DW3 (Depthwise 3x3) -- SCDown cv2 style");
    $display("================================================================");

    // Init input
    for (int h = 0; h < HIN; h++)
      for (int wb = 0; wb < WBLK_TOTAL; wb++)
        for (int c = 0; c < CH; c++)
          for (int l = 0; l < LANES; l++) begin
            int w_abs;
            w_abs = wb * LANES + l;
            if (w_abs < WIN)
              input_data[h][wb][c][l] = 8'((h * 11 + w_abs * 3 + c + 7) % 120);
            else
              input_data[h][wb][c][l] = 8'sd0;
          end

    // Per-channel weights
    for (int c = 0; c < CH; c++)
      for (int kh_i = 0; kh_i < KH; kh_i++)
        for (int kw_i = 0; kw_i < KW; kw_i++)
          weight_data[c][kh_i][kw_i] = 8'((c * 7 + kh_i * 3 + kw_i + 1) % 5) - 8'sd2;

    // Bias/quant per channel
    for (int c = 0; c < CH; c++) begin
      bias_data[c]  = 32'(c * 8);
      m_int_data[c] = 32'sd1;
      shift_data[c] = 0;
    end
    for (int i = 0; i < 256; i++) begin
      bias_mem[i]  = (i < CH) ? bias_data[i]  : 32'sd0;
      m_int_mem[i] = (i < CH) ? m_int_data[i] : 32'sd1;
      shift_mem[i] = (i < CH) ? shift_data[i] : 6'd0;
    end
    zp_out_val = 8'sd0;

    // Fill input banks
    begin
      int q_in_val;
      q_in_val = (HIN + 2) / 3;
      for (int h = 0; h < HIN; h++) begin
        int bank_id, row_slot;
        bank_id  = h % 3;
        row_slot = (h / 3) % q_in_val;
        for (int c = 0; c < CH; c++) begin
          for (int wb = 0; wb < WBLK_TOTAL; wb++) begin
            logic [15:0] addr;
            logic [LANES*8-1:0] packed_data;
            addr = 16'((row_slot * CH + c) * WBLK_TOTAL + wb);
            packed_data = '0;
            for (int l = 0; l < LANES; l++)
              packed_data[l*8 +: 8] = input_data[h][wb][c][l];
            fill_input_bank(bank_id[1:0], addr, packed_data);
          end
        end
      end
    end

    // Fill weight banks: DW uses per-channel, each row gets weight[c][row][kw]
    for (int row = 0; row < KH; row++) begin
      for (int c = 0; c < CH; c++) begin
        logic [LANES*8-1:0] packed_wt;
        packed_wt = '0;
        // For DW3: weight per channel packed into lanes matching channel layout
        for (int kw_i = 0; kw_i < KW; kw_i++) begin
          if (kw_i < LANES)
            packed_wt[kw_i*8 +: 8] = weight_data[c][row][kw_i];
        end
        fill_weight_bank(row[1:0], 16'(c), packed_wt);
      end
    end

    // Golden: per-channel depthwise conv
    for (int ho = 0; ho < HOUT; ho++)
      for (int wo = 0; wo < WOUT; wo++)
        for (int c = 0; c < CH; c++) begin
          automatic logic signed [31:0] acc;
          acc = 32'sd0;
          for (int kh_i = 0; kh_i < KH; kh_i++)
            for (int kw_i = 0; kw_i < KW; kw_i++) begin
              int ih, iw, wb_idx;
              logic signed [7:0] x_val;
              ih = ho * SH + kh_i - PAD;
              iw = wo * SW + kw_i - PAD;
              if (ih >= 0 && ih < HIN && iw >= 0 && iw < WIN) begin
                wb_idx = iw / LANES;
                x_val = input_data[ih][wb_idx][c][iw % LANES];
              end else
                x_val = 8'sd0;
              acc = acc + 32'(x_val) * 32'(weight_data[c][kh_i][kw_i]);
            end
          golden_out[ho][wo][c] = ppu_golden(acc, bias_data[c], m_int_data[c],
                                             shift_data[c], ACT_RELU, 1'b1);
        end

    // Descriptors
    build_default_descriptors();
    layer_desc_in.template_id  = 4'(PE_DW3);
    layer_desc_in.tile_cin     = 8'(CH);
    layer_desc_in.tile_cout    = 8'(CH);  // DW: cin == cout
    layer_desc_in.hin          = 10'(HIN);
    layer_desc_in.win          = 10'(WIN);
    layer_desc_in.hout         = 10'(HOUT);
    layer_desc_in.wout         = 10'(WOUT);
    layer_desc_in.kh           = 4'(KH);
    layer_desc_in.kw           = 4'(KW);
    layer_desc_in.sh           = 3'(SH);
    layer_desc_in.sw           = 3'(SW);
    layer_desc_in.pad_top      = 4'(PAD);
    layer_desc_in.pad_bot      = 4'(PAD);
    layer_desc_in.pad_left     = 4'(PAD);
    layer_desc_in.pad_right    = 4'(PAD);
    layer_desc_in.num_cin_pass = 4'd1;
    layer_desc_in.num_k_pass   = 4'd1;
    layer_desc_in.q_in         = 4'd2;
    layer_desc_in.q_out        = 4'd1;
    layer_desc_in.cin_total    = 9'(CH);
    layer_desc_in.cout_total   = 9'(CH);

    tile_desc_in.valid_h    = 6'(HOUT);
    tile_desc_in.valid_w    = 6'(WBLK_TOTAL);
    tile_desc_in.tile_flags = 16'h0003;

    post_profile_in.bias_en    = 1'b1;
    post_profile_in.quant_mode = QMODE_PER_CHANNEL;
    post_profile_in.act_mode   = ACT_RELU;
    post_profile_in.ewise_en   = 1'b0;
    post_profile_in.upsample_factor  = 2'd0;
    post_profile_in.concat_ch_offset = 8'd0;

    launch_tile_and_wait(100000);

    // Read and compare
    match_cnt = 0;
    total_cnt = 0;
    max_diff  = 0;

    for (int ho = 0; ho < HOUT; ho++) begin
      for (int wb = 0; wb < (WOUT + LANES - 1) / LANES; wb++) begin
        for (int c = 0; c < CH; c++) begin
          logic [LANES*32-1:0] psum_word;
          read_output_bank_psum(2'(c % OUTPUT_BANKS), 16'(ho * WBLK_TOTAL + wb), psum_word);
          for (int l = 0; l < LANES; l++) begin
            int wo_abs;
            logic signed [31:0] got_psum;
            logic signed [7:0] got_act, exp_act;
            int diff;
            wo_abs = wb * LANES + l;
            if (wo_abs < WOUT) begin
              got_psum = $signed(psum_word[l*32 +: 32]);
              exp_act  = golden_out[ho][wo_abs][c];
              got_act  = ppu_golden(got_psum, bias_data[c], m_int_data[c],
                                    shift_data[c], ACT_RELU, 1'b1);
              diff = int'(got_act) - int'(exp_act);
              if (diff < 0) diff = -diff;
              if (diff > max_diff) max_diff = diff;
              if (diff <= 1) match_cnt++;
              total_cnt++;
            end
          end
        end
      end
    end

    begin
      real pct;
      pct = (total_cnt > 0) ? (real'(match_cnt) / real'(total_cnt) * 100.0) : 0.0;
      test_names[2] = "PE_DW3 (DWConv 3x3)";
      test_match_pct[2] = pct;
      test_max_diff[2] = max_diff;
      if (pct >= 99.9) begin
        $display("  [PASS] PE_DW3: %0d/%0d match (%.1f%%), max LSB diff = %0d",
                 match_cnt, total_cnt, pct, max_diff);
        test_results[2] = "PASS";
        total_pass++;
      end else begin
        $display("  [FAIL] PE_DW3: %0d/%0d match (%.1f%%), max LSB diff = %0d",
                 match_cnt, total_cnt, pct, max_diff);
        test_results[2] = "FAIL";
        total_fail++;
      end
      total_tests++;
    end
  endtask

  // ══════════════════════════════════════════════════════════════
  // TEST 4: PE_MP5 — MaxPool 5x5 (SPPF style)
  // Config: C=2, H=8, W=32, pad=2, stride=1, no PPU
  // ══════════════════════════════════════════════════════════════
  task automatic test_pe_mp5();
    localparam int CH   = 2;
    localparam int H    = 8;
    localparam int W    = 32;
    localparam int KH   = 5;
    localparam int KW   = 5;
    localparam int PAD  = 2;
    localparam int SH   = 1;
    localparam int SW   = 1;
    localparam int HOUT = H;  // stride=1, pad=2 => same spatial
    localparam int WOUT = W;
    localparam int WBLK_TOTAL = (W + LANES - 1) / LANES;

    logic signed [7:0] input_data [H][WBLK_TOTAL][CH][LANES];
    logic signed [7:0] golden_out [HOUT][WOUT][CH];

    int match_cnt, total_cnt, max_diff;
    int seed;

    $display("\n================================================================");
    $display("  TEST 4: PE_MP5 (MaxPool 5x5) -- SPPF style");
    $display("================================================================");

    // Random INT8 input
    seed = 42;
    for (int h = 0; h < H; h++)
      for (int wb = 0; wb < WBLK_TOTAL; wb++)
        for (int c = 0; c < CH; c++)
          for (int l = 0; l < LANES; l++) begin
            int w_abs;
            w_abs = wb * LANES + l;
            if (w_abs < W)
              input_data[h][wb][c][l] = $signed(8'((seed = seed * 1103515245 + 12345) >> 16));
            else
              input_data[h][wb][c][l] = -8'sd128;  // min value for padding
          end

    // No PPU for maxpool -- set identity quant
    for (int i = 0; i < 256; i++) begin
      bias_mem[i]  = 32'sd0;
      m_int_mem[i] = 32'sd1;
      shift_mem[i] = 6'd0;
    end
    zp_out_val = 8'sd0;

    // Fill input banks
    begin
      int q_in_val;
      q_in_val = (H + 2) / 3;
      for (int h = 0; h < H; h++) begin
        int bank_id, row_slot;
        bank_id  = h % 3;
        row_slot = (h / 3) % q_in_val;
        for (int c = 0; c < CH; c++) begin
          for (int wb = 0; wb < WBLK_TOTAL; wb++) begin
            logic [15:0] addr;
            logic [LANES*8-1:0] packed_data;
            addr = 16'((row_slot * CH + c) * WBLK_TOTAL + wb);
            packed_data = '0;
            for (int l = 0; l < LANES; l++)
              packed_data[l*8 +: 8] = input_data[h][wb][c][l];
            fill_input_bank(bank_id[1:0], addr, packed_data);
          end
        end
      end
    end

    // Golden: max over 5x5 window per position
    for (int ho = 0; ho < HOUT; ho++)
      for (int wo = 0; wo < WOUT; wo++)
        for (int c = 0; c < CH; c++) begin
          automatic logic signed [7:0] max_val;
          max_val = -8'sd128;
          for (int kh_i = 0; kh_i < KH; kh_i++)
            for (int kw_i = 0; kw_i < KW; kw_i++) begin
              int ih, iw, wb_idx;
              logic signed [7:0] x_val;
              ih = ho * SH + kh_i - PAD;
              iw = wo * SW + kw_i - PAD;
              if (ih >= 0 && ih < H && iw >= 0 && iw < W) begin
                wb_idx = iw / LANES;
                x_val = input_data[ih][wb_idx][c][iw % LANES];
              end else
                x_val = -8'sd128;  // padding = min for maxpool
              if (x_val > max_val) max_val = x_val;
            end
          golden_out[ho][wo][c] = max_val;
        end

    // Descriptors
    build_default_descriptors();
    layer_desc_in.template_id  = 4'(PE_MP5);
    layer_desc_in.tile_cin     = 8'(CH);
    layer_desc_in.tile_cout    = 8'(CH);
    layer_desc_in.hin          = 10'(H);
    layer_desc_in.win          = 10'(W);
    layer_desc_in.hout         = 10'(HOUT);
    layer_desc_in.wout         = 10'(WOUT);
    layer_desc_in.kh           = 4'(KH);
    layer_desc_in.kw           = 4'(KW);
    layer_desc_in.sh           = 3'(SH);
    layer_desc_in.sw           = 3'(SW);
    layer_desc_in.pad_top      = 4'(PAD);
    layer_desc_in.pad_bot      = 4'(PAD);
    layer_desc_in.pad_left     = 4'(PAD);
    layer_desc_in.pad_right    = 4'(PAD);
    layer_desc_in.num_cin_pass = 4'd1;
    layer_desc_in.num_k_pass   = 4'd1;
    layer_desc_in.q_in         = 4'd3;  // ceil(8/3) = 3
    layer_desc_in.q_out        = 4'd1;
    layer_desc_in.cin_total    = 9'(CH);
    layer_desc_in.cout_total   = 9'(CH);

    tile_desc_in.valid_h    = 6'(HOUT);
    tile_desc_in.valid_w    = 6'(WBLK_TOTAL);
    tile_desc_in.tile_flags = 16'h0003;

    post_profile_in.bias_en    = 1'b0;
    post_profile_in.quant_mode = QMODE_NONE;
    post_profile_in.act_mode   = ACT_NONE;
    post_profile_in.ewise_en   = 1'b0;
    post_profile_in.upsample_factor  = 2'd0;
    post_profile_in.concat_ch_offset = 8'd0;

    launch_tile_and_wait(100000);

    // Read pool_out from output banks (ACT namespace since pool bypasses psum)
    match_cnt = 0;
    total_cnt = 0;
    max_diff  = 0;

    for (int ho = 0; ho < HOUT; ho++) begin
      for (int wb = 0; wb < WBLK_TOTAL; wb++) begin
        for (int c = 0; c < CH; c++) begin
          logic [LANES*8-1:0] act_word;
          read_output_bank_act(2'(c % OUTPUT_BANKS), 16'(ho * WBLK_TOTAL + wb), act_word);
          for (int l = 0; l < LANES; l++) begin
            int wo_abs;
            logic signed [7:0] got_val, exp_val;
            int diff;
            wo_abs = wb * LANES + l;
            if (wo_abs < WOUT) begin
              got_val = $signed(act_word[l*8 +: 8]);
              exp_val = golden_out[ho][wo_abs][c];
              diff = int'(got_val) - int'(exp_val);
              if (diff < 0) diff = -diff;
              if (diff > max_diff) max_diff = diff;
              if (diff == 0) match_cnt++;
              total_cnt++;
            end
          end
        end
      end
    end

    begin
      real pct;
      pct = (total_cnt > 0) ? (real'(match_cnt) / real'(total_cnt) * 100.0) : 0.0;
      test_names[3] = "PE_MP5 (MaxPool 5x5)";
      test_match_pct[3] = pct;
      test_max_diff[3] = max_diff;
      if (pct >= 100.0) begin
        $display("  [PASS] PE_MP5: %0d/%0d exact match (%.1f%%), max LSB diff = %0d",
                 match_cnt, total_cnt, pct, max_diff);
        test_results[3] = "PASS";
        total_pass++;
      end else begin
        $display("  [FAIL] PE_MP5: %0d/%0d exact match (%.1f%%), max LSB diff = %0d",
                 match_cnt, total_cnt, pct, max_diff);
        test_results[3] = "FAIL";
        total_fail++;
      end
      total_tests++;
    end
  endtask

  // ══════════════════════════════════════════════════════════════
  // TEST 5: PE_PASS + Upsample 2x
  // Config: H=4, W=16, C=2, upsample_factor=1 (2x)
  // Output: H=8, W=32, C=2 — each pixel duplicated 2x2
  // ══════════════════════════════════════════════════════════════
  task automatic test_pe_pass_upsample();
    localparam int CH    = 2;
    localparam int H_SRC = 4;
    localparam int W_SRC = 16;
    localparam int H_DST = 8;   // 2x
    localparam int W_DST = 32;  // 2x
    localparam int WBLK_SRC = (W_SRC + LANES - 1) / LANES;  // 1
    localparam int WBLK_DST = (W_DST + LANES - 1) / LANES;  // 1

    logic signed [7:0] input_data [H_SRC][WBLK_SRC][CH][LANES];
    logic signed [7:0] golden_out [H_DST][W_DST][CH];

    int match_cnt, total_cnt, max_diff;

    $display("\n================================================================");
    $display("  TEST 5: PE_PASS + Upsample 2x");
    $display("================================================================");

    // Known input pattern
    for (int h = 0; h < H_SRC; h++)
      for (int wb = 0; wb < WBLK_SRC; wb++)
        for (int c = 0; c < CH; c++)
          for (int l = 0; l < LANES; l++) begin
            int w_abs;
            w_abs = wb * LANES + l;
            if (w_abs < W_SRC)
              input_data[h][wb][c][l] = 8'(h * 16 + w_abs + c * 64);
            else
              input_data[h][wb][c][l] = 8'sd0;
          end

    // For PASS mode, data goes through swizzle only, so fill output banks
    // (since swizzle reads from output bank and writes to input bank)
    // Actually: PE_PASS path is IDLE -> LOAD_CFG -> PREFILL_IN -> SWIZZLE_STORE -> DONE
    // Swizzle reads from output bank (src) and writes to input bank (dst)
    // So we need to pre-fill OUTPUT bank with the source data

    // Fill output banks with source data (swizzle source)
    for (int h = 0; h < H_SRC; h++) begin
      for (int c = 0; c < CH; c++) begin
        for (int wb = 0; wb < WBLK_SRC; wb++) begin
          logic [LANES*8-1:0] packed_data;
          packed_data = '0;
          for (int l = 0; l < LANES; l++)
            packed_data[l*8 +: 8] = input_data[h][wb][c][l];
          // Output bank address: c * wblk_total + wb
          fill_output_bank(2'(h % OUTPUT_BANKS), 16'(c * WBLK_SRC + wb), packed_data);
        end
      end
    end

    // Also fill input banks (for PREFILL_IN DMA stub which passes instantly)
    for (int h = 0; h < H_SRC; h++) begin
      int bank_id, row_slot, q_in_val;
      q_in_val = (H_SRC + 2) / 3;
      bank_id  = h % 3;
      row_slot = (h / 3) % q_in_val;
      for (int c = 0; c < CH; c++) begin
        for (int wb = 0; wb < WBLK_SRC; wb++) begin
          logic [15:0] addr;
          logic [LANES*8-1:0] packed_data;
          addr = 16'((row_slot * CH + c) * WBLK_SRC + wb);
          packed_data = '0;
          for (int l = 0; l < LANES; l++)
            packed_data[l*8 +: 8] = input_data[h][wb][c][l];
          fill_input_bank(bank_id[1:0], addr, packed_data);
        end
      end
    end

    // No PPU needed
    for (int i = 0; i < 256; i++) begin
      bias_mem[i]  = 32'sd0;
      m_int_mem[i] = 32'sd1;
      shift_mem[i] = 6'd0;
    end
    zp_out_val = 8'sd0;

    // Golden: nearest-neighbor 2x upsample
    for (int ho = 0; ho < H_DST; ho++)
      for (int wo = 0; wo < W_DST; wo++)
        for (int c = 0; c < CH; c++) begin
          int src_h, src_w, wb_idx;
          src_h = ho / 2;
          src_w = wo / 2;
          wb_idx = src_w / LANES;
          golden_out[ho][wo][c] = input_data[src_h][wb_idx][c][src_w % LANES];
        end

    // Descriptors for PE_PASS + upsample
    build_default_descriptors();
    layer_desc_in.template_id  = 4'(PE_PASS);
    layer_desc_in.tile_cin     = 8'(CH);
    layer_desc_in.tile_cout    = 8'(CH);
    layer_desc_in.hin          = 10'(H_DST);  // dst dimensions for swizzle
    layer_desc_in.win          = 10'(W_DST);
    layer_desc_in.hout         = 10'(H_SRC);  // src dimensions
    layer_desc_in.wout         = 10'(W_SRC);
    layer_desc_in.kh           = 4'd1;
    layer_desc_in.kw           = 4'd1;
    layer_desc_in.sh           = 3'd1;
    layer_desc_in.sw           = 3'd1;
    layer_desc_in.pad_top      = 4'd0;
    layer_desc_in.pad_bot      = 4'd0;
    layer_desc_in.pad_left     = 4'd0;
    layer_desc_in.pad_right    = 4'd0;
    layer_desc_in.num_cin_pass = 4'd1;
    layer_desc_in.num_k_pass   = 4'd1;
    layer_desc_in.q_in         = 4'd2;
    layer_desc_in.q_out        = 4'd1;
    layer_desc_in.cin_total    = 9'(CH);
    layer_desc_in.cout_total   = 9'(CH);

    tile_desc_in.valid_h    = 6'(H_SRC);
    tile_desc_in.valid_w    = 6'(WBLK_SRC);
    // PE_PASS needs flag_need_swizzle=1 for upsample
    tile_desc_in.tile_flags = 16'h0013;  // first=1, last=1, need_swizzle=1 (bit4)

    post_profile_in.bias_en    = 1'b0;
    post_profile_in.quant_mode = QMODE_NONE;
    post_profile_in.act_mode   = ACT_NONE;
    post_profile_in.ewise_en   = 1'b0;
    post_profile_in.upsample_factor  = 2'd1;   // 2x upsample
    post_profile_in.concat_ch_offset = 8'd0;

    router_profile_in.upsample_dup_mode = 1'b1;

    launch_tile_and_wait(100000);

    // Read upsample output from input banks (swizzle writes to input banks)
    match_cnt = 0;
    total_cnt = 0;
    max_diff  = 0;

    // Swizzle writes to input banks with dst addressing
    // For simplicity, read back from input banks by driving ext_rd on input
    // Actually the ext_rd port only reads output banks.
    // The swizzle writes to input banks via dst_wr, so we need to read via
    // the input bank read port. But that is internal. The ext_rd port reads
    // output banks only.
    //
    // Alternative: We can verify by reading the DUT's internal input bank
    // memories using hierarchical access.
    for (int ho = 0; ho < H_DST; ho++) begin
      for (int wb = 0; wb < WBLK_DST; wb++) begin
        for (int c = 0; c < CH; c++) begin
          // Read from DUT internal input banks via hierarchical reference
          int bank_id, row_slot, q_in_val;
          logic [15:0] addr;
          q_in_val = (H_DST + 2) / 3;
          bank_id  = ho % 3;
          row_slot = (ho / 3) % q_in_val;
          addr     = 16'((row_slot * CH + c) * WBLK_DST + wb);

          for (int l = 0; l < LANES; l++) begin
            int wo_abs;
            logic signed [7:0] got_val, exp_val;
            int diff;
            wo_abs = wb * LANES + l;
            if (wo_abs < W_DST) begin
              // Try to read from internal SRAM hierarchy
              // gen_in_bank[bank_id].u_in_bank.gen_subbank[l].mem[addr]
              got_val = $signed(
                u_dut.gen_in_bank[bank_id].u_in_bank.gen_subbank[l].mem[addr[10:0]]
              );
              exp_val = golden_out[ho][wo_abs][c];
              diff = int'(got_val) - int'(exp_val);
              if (diff < 0) diff = -diff;
              if (diff > max_diff) max_diff = diff;
              if (diff == 0) match_cnt++;
              total_cnt++;
            end
          end
        end
      end
    end

    begin
      real pct;
      pct = (total_cnt > 0) ? (real'(match_cnt) / real'(total_cnt) * 100.0) : 0.0;
      test_names[4] = "PE_PASS (Upsample 2x)";
      test_match_pct[4] = pct;
      test_max_diff[4] = max_diff;
      if (pct >= 100.0) begin
        $display("  [PASS] PE_PASS: %0d/%0d exact match (%.1f%%), max LSB diff = %0d",
                 match_cnt, total_cnt, pct, max_diff);
        test_results[4] = "PASS";
        total_pass++;
      end else begin
        $display("  [FAIL] PE_PASS: %0d/%0d exact match (%.1f%%), max LSB diff = %0d",
                 match_cnt, total_cnt, pct, max_diff);
        test_results[4] = "FAIL";
        total_fail++;
      end
      total_tests++;
    end
  endtask

  // ══════════════════════════════════════════════════════════════
  // Main Test Sequence
  // ══════════════════════════════════════════════════════════════
  initial begin
    $display("");
    $display("################################################################");
    $display("#  TB: tb_subcluster_modes — Unified Subcluster Verification   #");
    $display("#  Same hardware, different descriptors => all primitives work  #");
    $display("################################################################");

    // ────── Reset ──────
    do_reset();

    // ────── TEST 1: PE_RS3 (Conv 3x3) ──────
    do_reset();
    test_pe_rs3();

    // ────── TEST 2: PE_OS1 (Conv 1x1) ──────
    do_reset();
    test_pe_os1();

    // ────── TEST 3: PE_DW3 (DWConv 3x3) ──────
    do_reset();
    test_pe_dw3();

    // ────── TEST 4: PE_MP5 (MaxPool 5x5) ──────
    do_reset();
    test_pe_mp5();

    // ────── TEST 5: PE_PASS (Upsample 2x) ──────
    do_reset();
    test_pe_pass_upsample();

    // ══════════════════════════════════════════════════════════════
    // Final Summary
    // ══════════════════════════════════════════════════════════════
    $display("");
    $display("################################################################");
    $display("#                    FINAL TEST SUMMARY                        #");
    $display("################################################################");
    $display("  %-5s  %-25s  %8s  %10s  %10s",
             "TEST", "MODE", "RESULT", "MATCH %", "MAX DIFF");
    $display("  %-5s  %-25s  %8s  %10s  %10s",
             "----", "----", "------", "-------", "--------");
    for (int t = 0; t < 5; t++) begin
      $display("  %-5d  %-25s  %8s  %9.1f%%  %10d",
               t + 1, test_names[t], test_results[t],
               test_match_pct[t], test_max_diff[t]);
    end
    $display("  %-5s  %-25s  %8s", "-----", "-------------------------", "--------");
    $display("  TOTAL: %0d tests, %0d PASSED, %0d FAILED", total_tests, total_pass, total_fail);
    $display("");
    if (total_fail == 0)
      $display("  >>> ALL %0d TESTS PASSED <<<", total_tests);
    else
      $display("  >>> %0d TESTS FAILED <<<", total_fail);
    $display("################################################################");
    $display("");
    $finish;
  end

  // ══════════════════════════════════════════════════════════════
  // Timeout watchdog
  // ══════════════════════════════════════════════════════════════
  initial begin
    #50_000_000;  // 50ms = 10M cycles at 200MHz
    $display("[WATCHDOG] Global simulation timeout reached!");
    $finish;
  end

  // ══════════════════════════════════════════════════════════════
  // Optional: FSM state waveform monitor
  // ══════════════════════════════════════════════════════════════
  always @(fsm_state) begin
    $display("  [FSM] t=%0t state=%s (%0d)", $time, fsm_state.name(), fsm_state);
  end

endmodule
