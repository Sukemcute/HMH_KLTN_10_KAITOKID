`timescale 1ns/1ps
// ============================================================================
// Golden Verification Testbench for dwconv7x7_engine (P8 DW_7x7_MULTIPASS)
// Depthwise 7x7 convolution with 3-pass multi-pass strategy.
//
// Computes: Y[h][w][c] = Sum_{kh=0..6, kw=0..6} X[h+kh][w+kw][c] * W[kh][kw][c]
//           + bias[c] -> requant -> activation -> clamp -> INT8
//
// Multipass: Pass 1 (kh 0-2), Pass 2 (kh 3-5), Pass 3 (kh 6)
//            PSUM accumulated between passes, PPU applied after Pass 3
//
// Test scenarios:
//   TEST 1 - All-ones:     input=1, weight=1, bias=0, expect 49 per position
//   TEST 2 - Single ch:    C=1, H=9, W=32, pad=3, known weights
//   TEST 3 - Multi-ch ReLU: C=4, H=14, W=32, random, act_mode=RELU
// ============================================================================
module tb_dwconv7x7_golden;

  import yolo_accel_pkg::*;

  // ═══════════════════════════════════════════════════════════════════
  // CLOCK / RESET
  // ═══════════════════════════════════════════════════════════════════
  localparam real CLK_PERIOD = 5.0;  // 200 MHz
  logic clk, rst_n;

  initial clk = 1'b0;
  always #(CLK_PERIOD/2) clk = ~clk;

  // ═══════════════════════════════════════════════════════════════════
  // LOCAL PARAMETERS
  // ═══════════════════════════════════════════════════════════════════
  localparam int TB_LANES     = 32;
  localparam int TB_MAX_W_PAD = 672;
  localparam int TB_MAX_COUT  = 256;

  // Maximum SRAM sizes
  localparam int IFM_SRAM_DEPTH = 131072;  // Input SRAM (LANES-wide)
  localparam int WGT_SRAM_DEPTH = 65536;   // Weight SRAM (1 byte per entry)
  localparam int OFM_SRAM_DEPTH = 65536;   // Output SRAM (LANES-wide)

  // ═══════════════════════════════════════════════════════════════════
  // DUT SIGNALS
  // ═══════════════════════════════════════════════════════════════════
  logic        start, done, busy;

  // Configuration
  logic [8:0]  cfg_channels;
  logic [9:0]  cfg_h;
  logic [9:0]  cfg_w;
  logic [9:0]  cfg_hout;
  logic [9:0]  cfg_wout;
  logic [9:0]  cfg_w_pad;
  act_mode_e   cfg_act_mode;
  logic signed [7:0] cfg_zp_out;

  // Input SRAM
  logic [23:0]       ifm_rd_addr;
  logic              ifm_rd_en;
  logic signed [7:0] ifm_rd_data [TB_LANES];

  // Weight SRAM
  logic [23:0]       wgt_rd_addr;
  logic              wgt_rd_en;
  logic signed [7:0] wgt_rd_data;

  // Bias & Quantization params
  logic signed [31:0] bias_arr  [TB_MAX_COUT];
  logic signed [31:0] m_int_arr [TB_MAX_COUT];
  logic [5:0]         shift_arr [TB_MAX_COUT];

  // SiLU LUT
  logic signed [7:0]  silu_lut [256];

  // Output SRAM
  logic [23:0]       ofm_wr_addr;
  logic              ofm_wr_en;
  logic signed [7:0] ofm_wr_data [TB_LANES];

  // ═══════════════════════════════════════════════════════════════════
  // BEHAVIORAL SRAM MODELS
  // ═══════════════════════════════════════════════════════════════════
  logic signed [7:0] ifm_sram [IFM_SRAM_DEPTH][TB_LANES];
  logic signed [7:0] wgt_sram [WGT_SRAM_DEPTH];
  logic signed [7:0] ofm_sram [OFM_SRAM_DEPTH][TB_LANES];

  // Input SRAM: 1-cycle read latency
  always_ff @(posedge clk) begin
    if (ifm_rd_en) begin
      for (int l = 0; l < TB_LANES; l++)
        ifm_rd_data[l] <= ifm_sram[ifm_rd_addr][l];
    end
  end

  // Weight SRAM: 1-cycle read latency
  always_ff @(posedge clk) begin
    if (wgt_rd_en)
      wgt_rd_data <= wgt_sram[wgt_rd_addr];
  end

  // Output SRAM: capture writes
  always_ff @(posedge clk) begin
    if (ofm_wr_en) begin
      for (int l = 0; l < TB_LANES; l++)
        ofm_sram[ofm_wr_addr][l] <= ofm_wr_data[l];
    end
  end

  // ═══════════════════════════════════════════════════════════════════
  // DUT INSTANTIATION
  // ═══════════════════════════════════════════════════════════════════
  dwconv7x7_engine #(
    .LANES     (TB_LANES),
    .MAX_W_PAD (TB_MAX_W_PAD)
  ) u_dut (
    .clk         (clk),
    .rst_n       (rst_n),
    .start       (start),
    .done        (done),
    .busy        (busy),
    .cfg_channels(cfg_channels),
    .cfg_h       (cfg_h),
    .cfg_w       (cfg_w),
    .cfg_hout    (cfg_hout),
    .cfg_wout    (cfg_wout),
    .cfg_w_pad   (cfg_w_pad),
    .cfg_act_mode(cfg_act_mode),
    .cfg_zp_out  (cfg_zp_out),
    .ifm_rd_addr (ifm_rd_addr),
    .ifm_rd_en   (ifm_rd_en),
    .ifm_rd_data (ifm_rd_data),
    .wgt_rd_addr (wgt_rd_addr),
    .wgt_rd_en   (wgt_rd_en),
    .wgt_rd_data (wgt_rd_data),
    .bias_arr    (bias_arr),
    .m_int_arr   (m_int_arr),
    .shift_arr   (shift_arr),
    .silu_lut    (silu_lut),
    .ofm_wr_addr (ofm_wr_addr),
    .ofm_wr_en   (ofm_wr_en),
    .ofm_wr_data (ofm_wr_data)
  );

  // ═══════════════════════════════════════════════════════════════════
  // SiLU LUT INITIALIZATION
  // ═══════════════════════════════════════════════════════════════════
  function automatic logic signed [7:0] compute_silu_entry(int idx);
    automatic real x_val, sig, silu_val;
    automatic int  result;
    x_val = real'(idx - 128) / 16.0;
    if (x_val < -8.0)
      sig = 0.0;
    else if (x_val > 8.0)
      sig = 1.0;
    else
      sig = 1.0 / (1.0 + $exp(-x_val));
    silu_val = real'(idx - 128) * sig;
    if (silu_val >= 0.0)
      result = int'(silu_val + 0.5);
    else
      result = int'(silu_val - 0.5);
    if (result > 127)  result = 127;
    if (result < -128) result = -128;
    return result[7:0];
  endfunction

  // ═══════════════════════════════════════════════════════════════════
  // GOLDEN REFERENCE DATA
  // ═══════════════════════════════════════════════════════════════════
  localparam int GOLD_MAX_H  = 26;   // Max H_pad across tests (20+6=26)
  localparam int GOLD_MAX_W  = 48;   // Max W_pad across tests (32+6=38, round up)
  localparam int GOLD_MAX_C  = 16;   // Max channels across tests

  // Padded input: gold_input[h_pad][w_pad][channel]
  logic signed [7:0] gold_input  [GOLD_MAX_H][GOLD_MAX_W][GOLD_MAX_C];
  // Depthwise weights: gold_weight[channel][kh][kw]
  logic signed [7:0] gold_weight [GOLD_MAX_C][7][7];
  // Golden output: dynamic array [hout * wout * channels]
  logic signed [7:0] gold_output [];

  // Per-test parameters
  int t_channels, t_h_pad, t_w_pad, t_hout, t_wout;
  act_mode_e t_act_mode;
  logic signed [7:0] t_zp_out;

  // ═══════════════════════════════════════════════════════════════════
  // GOLDEN: Behavioral depthwise 7x7 convolution + PPU
  // For depthwise: cin = cout = channels, one channel at a time
  // Stride = 1, pad = 3 (padding already in padded input)
  // ═══════════════════════════════════════════════════════════════════
  task automatic run_golden();
    gold_output = new[t_hout * t_wout * t_channels];

    for (int ho = 0; ho < t_hout; ho++) begin
      for (int wo = 0; wo < t_wout; wo++) begin
        for (int ch = 0; ch < t_channels; ch++) begin
          automatic longint acc = 0;
          automatic longint mult64;
          automatic longint rounded;
          automatic int     shifted;
          automatic int     act_in;
          automatic int     act_val;
          automatic int     final_val;
          automatic int     flat_idx;

          // 7x7 depthwise convolution accumulation
          for (int kh = 0; kh < 7; kh++) begin
            for (int kw = 0; kw < 7; kw++) begin
              // Input position: (ho + kh, wo + kw) in padded coordinates
              automatic int h_in = ho + kh;
              automatic int w_in = wo + kw;
              acc += longint'(gold_input[h_in][w_in][ch])
                   * longint'(gold_weight[ch][kh][kw]);
            end
          end

          // Add bias
          acc += longint'(bias_arr[ch]);

          // Requantization: (acc * m_int + round_bias) >>> shift
          mult64 = acc * longint'(m_int_arr[ch]);
          if (shift_arr[ch] > 0)
            rounded = mult64 + (longint'(1) <<< (shift_arr[ch] - 1));
          else
            rounded = mult64;
          shifted = int'(rounded >>> shift_arr[ch]);

          // Clamp to 16-bit for activation indexing
          if (shifted > 32767)
            act_in = 32767;
          else if (shifted < -32768)
            act_in = -32768;
          else
            act_in = shifted;

          // Activation
          case (t_act_mode)
            ACT_SILU: begin
              automatic int silu_idx;
              silu_idx = act_in + 128;
              if (silu_idx < 0)   silu_idx = 0;
              if (silu_idx > 255) silu_idx = 255;
              act_val = int'(silu_lut[silu_idx]);
            end
            ACT_RELU: begin
              if (act_in > 0) begin
                if (act_in > 127)
                  act_val = 127;
                else if (act_in < -128)
                  act_val = -128;
                else
                  act_val = act_in;
              end else begin
                act_val = 0;
              end
            end
            default: begin  // ACT_NONE, ACT_CLAMP
              if (act_in > 127)
                act_val = 127;
              else if (act_in < -128)
                act_val = -128;
              else
                act_val = act_in;
            end
          endcase

          // Add output zero-point and final clamp
          final_val = act_val + int'(t_zp_out);
          if (final_val > 127)
            final_val = 127;
          else if (final_val < -128)
            final_val = -128;

          flat_idx = ho * t_wout * t_channels + wo * t_channels + ch;
          gold_output[flat_idx] = final_val[7:0];
        end
      end
    end
  endtask

  // ═══════════════════════════════════════════════════════════════════
  // COMPARISON: RTL output SRAM vs golden
  // Output SRAM layout: [Hout][C][num_wblk_out]
  // Address = h_out * channels * num_wblk_out + ch * num_wblk_out + wblk
  // ═══════════════════════════════════════════════════════════════════
  function automatic int compare_outputs(
    input int hout, input int wout, input int channels
  );
    automatic int mismatches = 0;
    automatic int printed    = 0;
    automatic int num_wblk   = (wout + TB_LANES - 1) / TB_LANES;

    for (int ho = 0; ho < hout; ho++) begin
      for (int ch = 0; ch < channels; ch++) begin
        for (int wblk = 0; wblk < num_wblk; wblk++) begin
          automatic int ofm_addr;
          ofm_addr = ho * channels * num_wblk + ch * num_wblk + wblk;
          for (int l = 0; l < TB_LANES; l++) begin
            automatic int wo = wblk * TB_LANES + l;
            if (wo < wout) begin
              automatic int flat_idx;
              automatic logic signed [7:0] rtl_val, gold_val;
              flat_idx = ho * wout * channels + wo * channels + ch;
              rtl_val  = ofm_sram[ofm_addr][l];
              gold_val = gold_output[flat_idx];
              if (rtl_val !== gold_val) begin
                mismatches++;
                if (printed < 10) begin
                  $display("  MISMATCH @ (h=%0d, w=%0d, ch=%0d): expected=%0d, got=%0d",
                           ho, wo, ch, int'(gold_val), int'(rtl_val));
                  printed++;
                end
              end
            end
          end
        end
      end
    end
    return mismatches;
  endfunction

  // ═══════════════════════════════════════════════════════════════════
  // HELPER TASKS
  // ═══════════════════════════════════════════════════════════════════

  task automatic reset_dut();
    rst_n = 1'b0;
    start = 1'b0;
    repeat (10) @(posedge clk);
    rst_n = 1'b1;
    repeat (5) @(posedge clk);
  endtask

  task automatic clear_srams();
    for (int i = 0; i < IFM_SRAM_DEPTH; i++)
      for (int l = 0; l < TB_LANES; l++)
        ifm_sram[i][l] = 8'sd0;
    for (int i = 0; i < WGT_SRAM_DEPTH; i++)
      wgt_sram[i] = 8'sd0;
    for (int i = 0; i < OFM_SRAM_DEPTH; i++)
      for (int l = 0; l < TB_LANES; l++)
        ofm_sram[i][l] = 8'sd0;
    for (int i = 0; i < TB_MAX_COUT; i++) begin
      bias_arr[i]  = 32'sd0;
      m_int_arr[i] = 32'sd1;
      shift_arr[i] = 6'd0;
    end
  endtask

  task automatic clear_gold_arrays();
    for (int h = 0; h < GOLD_MAX_H; h++)
      for (int w = 0; w < GOLD_MAX_W; w++)
        for (int c = 0; c < GOLD_MAX_C; c++)
          gold_input[h][w][c] = 8'sd0;
    for (int c = 0; c < GOLD_MAX_C; c++)
      for (int kh = 0; kh < 7; kh++)
        for (int kw = 0; kw < 7; kw++)
          gold_weight[c][kh][kw] = 8'sd0;
  endtask

  // Fill input SRAM from gold_input (padded)
  // Layout: addr = h_pad * channels * num_wblk_in + ch * num_wblk_in + wblk
  // ifm_sram[addr][l] = gold_input[h_pad][wblk*LANES + l][ch]
  task automatic fill_ifm_sram(int h_pad, int w_pad, int channels);
    automatic int num_wblk_in = (w_pad + TB_LANES - 1) / TB_LANES;
    for (int h = 0; h < h_pad; h++) begin
      for (int ch = 0; ch < channels; ch++) begin
        for (int wblk = 0; wblk < num_wblk_in; wblk++) begin
          automatic int addr = h * channels * num_wblk_in + ch * num_wblk_in + wblk;
          for (int l = 0; l < TB_LANES; l++) begin
            automatic int w_idx = wblk * TB_LANES + l;
            if (w_idx < w_pad)
              ifm_sram[addr][l] = gold_input[h][w_idx][ch];
            else
              ifm_sram[addr][l] = 8'sd0;
          end
        end
      end
    end
  endtask

  // Fill weight SRAM from gold_weight
  // Layout: addr = ch * 49 + kh * 7 + kw
  task automatic fill_wgt_sram(int channels);
    for (int ch = 0; ch < channels; ch++) begin
      for (int kh = 0; kh < 7; kh++) begin
        for (int kw = 0; kw < 7; kw++) begin
          automatic int addr = ch * 49 + kh * 7 + kw;
          wgt_sram[addr] = gold_weight[ch][kh][kw];
        end
      end
    end
  endtask

  // Launch DUT and wait for done
  task automatic run_dut();
    @(posedge clk);
    start <= 1'b1;
    @(posedge clk);
    start <= 1'b0;

    fork
      begin
        wait (done === 1'b1);
      end
      begin
        repeat (5_000_000) @(posedge clk);
        $display("  ERROR: DUT timed out (5M cycles)!");
        $stop;
      end
    join_any
    disable fork;

    repeat (5) @(posedge clk);
  endtask

  // ═══════════════════════════════════════════════════════════════════
  // RANDOM HELPERS
  // ═══════════════════════════════════════════════════════════════════
  function automatic logic signed [7:0] rand_int8();
    automatic int rv;
    rv = $urandom_range(0, 255);
    return rv[7:0] - 8'sd128;
  endfunction

  function automatic logic signed [7:0] rand_range(int lo, int hi);
    automatic int rv;
    rv = $urandom_range(0, hi - lo) + lo;
    if (rv > 127)  rv = 127;
    if (rv < -128) rv = -128;
    return rv[7:0];
  endfunction

  // ═══════════════════════════════════════════════════════════════════
  // TEST RESULTS
  // ═══════════════════════════════════════════════════════════════════
  int total_tests  = 0;
  int total_passed = 0;

  // ═══════════════════════════════════════════════════════════════════
  // TEST 1 - ALL-ONES
  // Input: all 1s, Weight: all 1s (7x7=49 per channel)
  // Bias: 0, M_int: 1, shift: 0
  // Expected: output = clamp(49, -128, 127) = 49 per position
  //
  // C=1, H=7 (unpadded), pad=3 -> H_pad=13, W=32 -> W_pad=38
  // Hout=7, Wout=32 (stride=1)
  // ═══════════════════════════════════════════════════════════════════
  task automatic test1_all_ones();
    automatic int mismatches;
    automatic int channels = 1;
    automatic int hin = 7, win = 32;
    automatic int pad = 3;
    automatic int h_pad = hin + 2 * pad;   // 13
    automatic int w_pad = win + 2 * pad;   // 38
    automatic int hout = hin;              // 7 (stride=1)
    automatic int wout = win;              // 32

    $display("════════════════════════════════════════════════════");
    $display("TEST 1 - All-ones: C=%0d, Hin=%0d, Win=%0d, pad=%0d",
             channels, hin, win, pad);
    $display("  Hout=%0d, Wout=%0d, expect output=49 everywhere", hout, wout);
    $display("════════════════════════════════════════════════════");

    clear_srams();
    clear_gold_arrays();
    reset_dut();

    // Fill padded input: zero everywhere, then 1s in the valid region
    for (int h = 0; h < h_pad; h++)
      for (int w = 0; w < w_pad; w++)
        gold_input[h][w][0] = 8'sd0;

    for (int h = pad; h < pad + hin; h++)
      for (int w = pad; w < pad + win; w++)
        gold_input[h][w][0] = 8'sd1;

    // All weights = 1
    for (int kh = 0; kh < 7; kh++)
      for (int kw = 0; kw < 7; kw++)
        gold_weight[0][kh][kw] = 8'sd1;

    // Identity quantization: m_int=1, shift=0, bias=0
    bias_arr[0]  = 32'sd0;
    m_int_arr[0] = 32'sd1;
    shift_arr[0] = 6'd0;

    fill_ifm_sram(h_pad, w_pad, channels);
    fill_wgt_sram(channels);

    // Configure DUT
    cfg_channels = 9'(channels);
    cfg_h        = 10'(hin);
    cfg_w        = 10'(win);
    cfg_hout     = 10'(hout);
    cfg_wout     = 10'(wout);
    cfg_w_pad    = 10'(w_pad);
    cfg_act_mode = ACT_NONE;
    cfg_zp_out   = 8'sd0;

    // Golden
    t_channels = channels;
    t_h_pad    = h_pad;
    t_w_pad    = w_pad;
    t_hout     = hout;
    t_wout     = wout;
    t_act_mode = ACT_NONE;
    t_zp_out   = 8'sd0;
    run_golden();

    // Spot-check golden: center pixel should be 49 (full 7x7 window all 1s)
    begin
      automatic int center_h = hout / 2;
      automatic int center_w = wout / 2;
      automatic int flat = center_h * wout * channels + center_w * channels + 0;
      $display("  Golden spot-check: center pixel (%0d,%0d,ch0) = %0d (expect 49)",
               center_h, center_w, int'(gold_output[flat]));
    end

    // Also check a border pixel where some kernel taps fall on padding (0)
    begin
      automatic int flat = 0 * wout * channels + 0 * channels + 0;
      $display("  Golden spot-check: corner (0,0,ch0) = %0d (expect <49 due to padding overlap)",
               int'(gold_output[flat]));
    end

    run_dut();

    mismatches = compare_outputs(hout, wout, channels);

    total_tests++;
    if (mismatches == 0) begin
      $display("  TEST 1: PASS");
      total_passed++;
    end else begin
      $display("  TEST 1: FAIL (%0d mismatches)", mismatches);
    end
    $display("");
  endtask

  // ═══════════════════════════════════════════════════════════════════
  // TEST 2 - SINGLE CHANNEL with known weights
  // C=1, H=9, W=32, pad=3 -> H_pad=15, W_pad=38
  // Hout=9, Wout=32
  //
  // Structured weights: w[kh][kw] = kh*7 + kw + 1 (1..49)
  // Input: incrementing pattern in valid region, 0 in padding
  //
  // Verifies the 3-pass PSUM accumulation:
  //   Pass 1: kh=0,1,2 -> psum1
  //   Pass 2: kh=3,4,5 -> psum2 = psum1 + pass2_contrib
  //   Pass 3: kh=6 ->     final = psum2 + pass3_contrib -> PPU
  //
  // Uses realistic quantization: m_int=50, shift=10
  // ═══════════════════════════════════════════════════════════════════
  task automatic test2_single_channel();
    automatic int mismatches;
    automatic int channels = 1;
    automatic int hin = 9, win = 32;
    automatic int pad = 3;
    automatic int h_pad = hin + 2 * pad;   // 15
    automatic int w_pad = win + 2 * pad;   // 38
    automatic int hout = hin;              // 9
    automatic int wout = win;              // 32

    $display("════════════════════════════════════════════════════");
    $display("TEST 2 - Single channel: C=%0d, Hin=%0d, Win=%0d, pad=%0d",
             channels, hin, win, pad);
    $display("  Known weights (1..49), incrementing input");
    $display("  3-pass PSUM accumulation verification");
    $display("════════════════════════════════════════════════════");

    clear_srams();
    clear_gold_arrays();
    reset_dut();

    // Fill padded input: zero padding, valid region = (h-pad)*10 + (w-pad) mod 20
    for (int h = 0; h < h_pad; h++)
      for (int w = 0; w < w_pad; w++)
        gold_input[h][w][0] = 8'sd0;

    for (int h = pad; h < pad + hin; h++)
      for (int w = pad; w < pad + win; w++) begin
        automatic int val = ((h - pad) * 3 + (w - pad)) % 20 + 1;
        if (val > 127) val = 127;
        gold_input[h][w][0] = val[7:0];
      end

    // Structured weights: kh*7 + kw + 1 (range 1..49)
    for (int kh = 0; kh < 7; kh++)
      for (int kw = 0; kw < 7; kw++)
        gold_weight[0][kh][kw] = 8'(kh * 7 + kw + 1);

    // Realistic quantization
    // Max MAC: 49 taps * 127 * 49 ~ 305,000
    // m_int=50, shift=10: (305000 * 50 + 512) >> 10 ~ 14893 -> needs more shift
    // Use m_int=1, shift=10 to keep in range: 305000/1024 ~ 298
    // Better: m_int=1, shift=12: 305000/4096 ~ 74 -> fits INT8
    bias_arr[0]  = 32'sd100;     // small bias offset
    m_int_arr[0] = 32'sd1;
    shift_arr[0] = 6'd12;

    fill_ifm_sram(h_pad, w_pad, channels);
    fill_wgt_sram(channels);

    // Configure DUT
    cfg_channels = 9'(channels);
    cfg_h        = 10'(hin);
    cfg_w        = 10'(win);
    cfg_hout     = 10'(hout);
    cfg_wout     = 10'(wout);
    cfg_w_pad    = 10'(w_pad);
    cfg_act_mode = ACT_NONE;
    cfg_zp_out   = 8'sd0;

    // Golden
    t_channels = channels;
    t_h_pad    = h_pad;
    t_w_pad    = w_pad;
    t_hout     = hout;
    t_wout     = wout;
    t_act_mode = ACT_NONE;
    t_zp_out   = 8'sd0;
    run_golden();

    // Display a few golden values for manual verification
    $display("  Golden values at row 4 (center): [%0d, %0d, %0d, %0d, ...]",
             int'(gold_output[4 * wout * channels + 0 * channels + 0]),
             int'(gold_output[4 * wout * channels + 1 * channels + 0]),
             int'(gold_output[4 * wout * channels + 2 * channels + 0]),
             int'(gold_output[4 * wout * channels + 3 * channels + 0]));

    run_dut();

    mismatches = compare_outputs(hout, wout, channels);

    total_tests++;
    if (mismatches == 0) begin
      $display("  TEST 2: PASS");
      total_passed++;
    end else begin
      $display("  TEST 2: FAIL (%0d mismatches)", mismatches);
    end
    $display("");
  endtask

  // ═══════════════════════════════════════════════════════════════════
  // TEST 3 - MULTI-CHANNEL WITH ReLU
  // C=4, H=14, W=32, pad=3 -> H_pad=20, W_pad=38
  // Hout=14, Wout=32
  //
  // Random weights, random input
  // act_mode=RELU -> negative outputs clamped to 0
  //
  // Quantization: m_int=2, shift=14
  //   Max MAC: 49 * 128 * 128 ~ 802,816
  //   (802816 * 2 + 8192) >> 14 ~ 98 -> fits INT8
  // ═══════════════════════════════════════════════════════════════════
  task automatic test3_multichannel_relu();
    automatic int mismatches;
    automatic int channels = 4;
    automatic int hin = 14, win = 32;
    automatic int pad = 3;
    automatic int h_pad = hin + 2 * pad;   // 20
    automatic int w_pad = win + 2 * pad;   // 38
    automatic int hout = hin;              // 14
    automatic int wout = win;              // 32
    automatic int relu_violations = 0;

    $display("════════════════════════════════════════════════════");
    $display("TEST 3 - Multi-channel ReLU: C=%0d, Hin=%0d, Win=%0d",
             channels, hin, win);
    $display("  Random weights/input, act_mode=RELU");
    $display("════════════════════════════════════════════════════");

    clear_srams();
    clear_gold_arrays();
    reset_dut();

    // Fill padded input: zero padding, random inner values
    for (int h = 0; h < h_pad; h++)
      for (int w = 0; w < w_pad; w++)
        for (int c = 0; c < channels; c++)
          gold_input[h][w][c] = 8'sd0;

    for (int h = pad; h < pad + hin; h++)
      for (int w = pad; w < pad + win; w++)
        for (int c = 0; c < channels; c++)
          gold_input[h][w][c] = rand_int8();

    // Random weights [-5, 5] (small to avoid saturation in most cases)
    for (int c = 0; c < channels; c++)
      for (int kh = 0; kh < 7; kh++)
        for (int kw = 0; kw < 7; kw++)
          gold_weight[c][kh][kw] = rand_range(-5, 5);

    // Quantization params per channel
    for (int c = 0; c < channels; c++) begin
      bias_arr[c]  = $signed($urandom_range(0, 200)) - 100;
      m_int_arr[c] = 32'sd2;
      shift_arr[c] = 6'd14;
    end

    fill_ifm_sram(h_pad, w_pad, channels);
    fill_wgt_sram(channels);

    // Configure DUT
    cfg_channels = 9'(channels);
    cfg_h        = 10'(hin);
    cfg_w        = 10'(win);
    cfg_hout     = 10'(hout);
    cfg_wout     = 10'(wout);
    cfg_w_pad    = 10'(w_pad);
    cfg_act_mode = ACT_RELU;
    cfg_zp_out   = 8'sd0;

    // Golden
    t_channels = channels;
    t_h_pad    = h_pad;
    t_w_pad    = w_pad;
    t_hout     = hout;
    t_wout     = wout;
    t_act_mode = ACT_RELU;
    t_zp_out   = 8'sd0;
    run_golden();

    // Verify ReLU property: no golden output should be negative
    for (int i = 0; i < hout * wout * channels; i++) begin
      if (int'(gold_output[i]) < 0)
        relu_violations++;
    end
    $display("  ReLU property check: %0d violations out of %0d outputs (expect 0)",
             relu_violations, hout * wout * channels);

    run_dut();

    mismatches = compare_outputs(hout, wout, channels);

    total_tests++;
    if (mismatches == 0 && relu_violations == 0) begin
      $display("  TEST 3: PASS");
      total_passed++;
    end else begin
      if (mismatches > 0)
        $display("  TEST 3: FAIL (%0d mismatches)", mismatches);
      if (relu_violations > 0)
        $display("  TEST 3: FAIL (%0d ReLU violations)", relu_violations);
    end
    $display("");
  endtask

  // ═══════════════════════════════════════════════════════════════════
  // MAIN TEST SEQUENCE
  // ═══════════════════════════════════════════════════════════════════
  initial begin
    $display("");
    $display("================================================================");
    $display("  Golden Verification TB for dwconv7x7_engine (P8 DW_7x7_MULTI)");
    $display("================================================================");
    $display("");

    // Initialize SiLU LUT
    for (int i = 0; i < 256; i++)
      silu_lut[i] = compute_silu_entry(i);

    // Initialize signals
    clk     = 1'b0;
    rst_n   = 1'b0;
    start   = 1'b0;
    cfg_channels = '0;
    cfg_h        = '0;
    cfg_w        = '0;
    cfg_hout     = '0;
    cfg_wout     = '0;
    cfg_w_pad    = '0;
    cfg_act_mode = ACT_NONE;
    cfg_zp_out   = 8'sd0;

    // Seed random
    $urandom(42);

    // Run all tests
    test1_all_ones();
    test2_single_channel();
    test3_multichannel_relu();

    // Final summary
    $display("================================================================");
    $display("  FINAL RESULTS: %0d / %0d tests PASSED", total_passed, total_tests);
    if (total_passed == total_tests)
      $display("  >>>  ALL TESTS PASSED  <<<");
    else
      $display("  >>>  SOME TESTS FAILED  <<<");
    $display("================================================================");
    $display("");

    $finish;
  end

  // ═══════════════════════════════════════════════════════════════════
  // TIMEOUT WATCHDOG
  // ═══════════════════════════════════════════════════════════════════
  initial begin
    #100_000_000;  // 100 ms
    $display("FATAL: Global simulation timeout reached!");
    $finish;
  end

endmodule
