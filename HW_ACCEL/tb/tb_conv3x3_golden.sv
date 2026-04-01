`timescale 1ns/1ps
// ============================================================================
// Golden Verification Testbench for conv3x3_engine
// Computes expected output in pure behavioral SystemVerilog, then compares
// bit-exactly against RTL output read from the output SRAM.
//
// Test scenarios:
//   TEST 1 - Minimal:      stride=1, Cin=1,  Cout=1, H=3,  W=32
//   TEST 2 - Layer 0:      stride=2, Cin=3,  Cout=4, Hin=8, Win=64
//   TEST 3 - Large Cin:    stride=1, Cin=16, Cout=4, Hin=4, Win=32
//   TEST 4 - Random Stress: stride=1, Cin=8,  Cout=8, Hin=6, Win=64
// ============================================================================
module tb_conv3x3_golden;

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
  localparam int TB_MAX_CIN   = 384;
  localparam int TB_MAX_COUT  = 256;
  localparam int TB_MAX_HIN   = 642;

  // Maximum SRAM sizes (generous for all tests)
  localparam int IFM_SRAM_DEPTH = 65536;  // input SRAM entries (each LANES wide)
  localparam int WGT_SRAM_DEPTH = 65536;  // weight SRAM entries (each 1 byte)
  localparam int OFM_SRAM_DEPTH = 65536;  // output SRAM entries (each LANES wide)

  // ═══════════════════════════════════════════════════════════════════
  // DUT SIGNALS
  // ═══════════════════════════════════════════════════════════════════
  logic        start, done, busy;

  logic [9:0]  cfg_w_pad;
  logic [9:0]  cfg_hout;
  logic [9:0]  cfg_wout;
  logic [8:0]  cfg_cin;
  logic [8:0]  cfg_cout;
  logic [1:0]  cfg_stride;
  act_mode_e   cfg_act_mode;
  logic signed [7:0] cfg_zp_out;

  logic [23:0]       ifm_rd_addr;
  logic              ifm_rd_en;
  logic signed [7:0] ifm_rd_data [TB_LANES];

  logic [23:0]       wgt_rd_addr;
  logic              wgt_rd_en;
  logic signed [7:0] wgt_rd_data;

  logic signed [31:0] bias_arr  [TB_MAX_COUT];
  logic signed [31:0] m_int_arr [TB_MAX_COUT];
  logic [5:0]         shift_arr [TB_MAX_COUT];

  logic signed [7:0]  silu_lut [256];

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
  conv3x3_engine #(
    .LANES     (TB_LANES),
    .MAX_W_PAD (TB_MAX_W_PAD)
  ) u_dut (
    .clk         (clk),
    .rst_n       (rst_n),
    .start       (start),
    .done        (done),
    .busy        (busy),
    .cfg_w_pad   (cfg_w_pad),
    .cfg_hout    (cfg_hout),
    .cfg_wout    (cfg_wout),
    .cfg_cin     (cfg_cin),
    .cfg_cout    (cfg_cout),
    .cfg_stride  (cfg_stride),
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
  // Approximation: silu(x) = x * sigmoid(x)
  // LUT index i maps to signed value (i - 128), scaled by 16
  // silu_lut[i] = clamp(round((i-128) * sigmoid((i-128)/16.0) * 16), -128, 127)
  // We use a precomputed table for accuracy.
  // ═══════════════════════════════════════════════════════════════════
  function automatic logic signed [7:0] compute_silu_entry(int idx);
    // x_real = idx - 128 (range: -128 to 127)
    // Input to SiLU is x_real / 16.0 (the LUT covers fixed-point range)
    // But per the DUT, the index is computed from silu_index(act_in) where
    // act_in is the requantized INT16 value, and silu_index = val + 128.
    // So the LUT maps signed value v = (idx-128) to silu(v) clamped to INT8.
    //
    // silu(v) = v * sigmoid(v) = v / (1 + exp(-v))
    // For INT8 inputs: v range is [-128, 127]
    //
    // We precompute with the formula:
    //   silu_lut[i] = clamp(round( (i-128) * sigmoid((i-128)/16.0) * 16 ), -128, 127)
    //
    // Using a piecewise approximation for sigmoid:
    automatic real x_val, sig, silu_val;
    automatic int  result;
    x_val = real'(idx - 128) / 16.0;
    // sigmoid approximation
    if (x_val < -8.0)
      sig = 0.0;
    else if (x_val > 8.0)
      sig = 1.0;
    else
      sig = 1.0 / (1.0 + $exp(-x_val));
    silu_val = real'(idx - 128) * sig;
    // Round to nearest
    if (silu_val >= 0.0)
      result = int'(silu_val + 0.5);
    else
      result = int'(silu_val - 0.5);
    // Clamp
    if (result > 127)  result = 127;
    if (result < -128) result = -128;
    return result[7:0];
  endfunction

  // ═══════════════════════════════════════════════════════════════════
  // GOLDEN REFERENCE COMPUTATION
  // ═══════════════════════════════════════════════════════════════════

  // Padded input array for golden model (allocated per test)
  // input_padded[h_pad][w_pad][cin] - stored flattened for flexibility
  localparam int GOLD_MAX_H = 16;   // max H_pad across tests
  localparam int GOLD_MAX_W = 128;  // max W_pad across tests
  localparam int GOLD_MAX_C = 16;   // max Cin across tests

  logic signed [7:0] gold_input [GOLD_MAX_H][GOLD_MAX_W][GOLD_MAX_C];
  logic signed [7:0] gold_weight[TB_MAX_COUT][GOLD_MAX_C][3][3];
  logic signed [7:0] gold_output[];  // dynamic array: [h_out * wout * cout] flat

  // Per-test parameters (set before calling run_golden)
  int t_cin, t_cout, t_hin_pad, t_win_pad, t_hout, t_wout, t_stride, t_pad;
  act_mode_e t_act_mode;
  logic signed [7:0] t_zp_out;

  // ── Golden: Behavioral convolution ──
  task automatic run_golden();
    automatic int num_wblk_out = (t_wout + TB_LANES - 1) / TB_LANES;
    gold_output = new[t_hout * t_wout * t_cout];

    for (int ho = 0; ho < t_hout; ho++) begin
      for (int wo = 0; wo < t_wout; wo++) begin
        for (int co = 0; co < t_cout; co++) begin
          automatic longint acc = 0;
          automatic longint mult64;
          automatic longint rounded;
          automatic int     shifted;
          automatic int     act_in;
          automatic int     act_val;
          automatic int     final_val;
          automatic int     flat_idx;

          // Convolution accumulation
          for (int ci = 0; ci < t_cin; ci++) begin
            for (int kh = 0; kh < 3; kh++) begin
              for (int kw = 0; kw < 3; kw++) begin
                automatic int h_in = ho * t_stride + kh;
                automatic int w_in = wo * t_stride + kw;
                acc += longint'(gold_input[h_in][w_in][ci])
                     * longint'(gold_weight[co][ci][kh][kw]);
              end
            end
          end

          // Add bias
          acc += longint'(bias_arr[co]);

          // Requantization: (acc * m_int + round_bias) >>> shift
          mult64 = acc * longint'(m_int_arr[co]);
          if (shift_arr[co] > 0)
            rounded = mult64 + (longint'(1) <<< (shift_arr[co] - 1));
          else
            rounded = mult64;
          shifted = int'(rounded >>> shift_arr[co]);

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

          flat_idx = ho * t_wout * t_cout + wo * t_cout + co;
          gold_output[flat_idx] = final_val[7:0];
        end
      end
    end
  endtask

  // ═══════════════════════════════════════════════════════════════════
  // COMPARISON: RTL output SRAM vs golden
  // ═══════════════════════════════════════════════════════════════════
  function automatic int compare_outputs(
    input int hout, input int wout, input int cout,
    input int num_wblk_out
  );
    automatic int mismatches = 0;
    automatic int printed    = 0;
    for (int ho = 0; ho < hout; ho++) begin
      for (int co = 0; co < cout; co++) begin
        for (int wblk = 0; wblk < num_wblk_out; wblk++) begin
          automatic int ofm_addr;
          ofm_addr = ho * cout * num_wblk_out + co * num_wblk_out + wblk;
          for (int l = 0; l < TB_LANES; l++) begin
            automatic int wo = wblk * TB_LANES + l;
            if (wo < wout) begin
              automatic int flat_idx;
              automatic logic signed [7:0] rtl_val, gold_val;
              flat_idx = ho * wout * cout + wo * cout + co;
              rtl_val  = ofm_sram[ofm_addr][l];
              gold_val = gold_output[flat_idx];
              if (rtl_val !== gold_val) begin
                mismatches++;
                if (printed < 10) begin
                  $display("  MISMATCH @ (ho=%0d, wo=%0d, co=%0d): expected=%0d, got=%0d",
                           ho, wo, co, int'(gold_val), int'(rtl_val));
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
    for (int co = 0; co < TB_MAX_COUT; co++)
      for (int ci = 0; ci < GOLD_MAX_C; ci++)
        for (int kh = 0; kh < 3; kh++)
          for (int kw = 0; kw < 3; kw++)
            gold_weight[co][ci][kh][kw] = 8'sd0;
  endtask

  // Fill input SRAM from gold_input, matching DUT memory layout
  // Layout: addr = h_pad * cin * num_wblk_in + cin_idx * num_wblk_in + wblk
  // Word [wblk]: ifm_sram[addr][l] = input[h_pad][wblk*LANES + l][cin_idx]
  task automatic fill_ifm_sram(int h_pad, int w_pad, int cin);
    automatic int num_wblk_in = (w_pad + TB_LANES - 1) / TB_LANES;
    for (int h = 0; h < h_pad; h++) begin
      for (int ci = 0; ci < cin; ci++) begin
        for (int wblk = 0; wblk < num_wblk_in; wblk++) begin
          automatic int addr = h * cin * num_wblk_in + ci * num_wblk_in + wblk;
          for (int l = 0; l < TB_LANES; l++) begin
            automatic int w_idx = wblk * TB_LANES + l;
            if (w_idx < w_pad)
              ifm_sram[addr][l] = gold_input[h][w_idx][ci];
            else
              ifm_sram[addr][l] = 8'sd0;
          end
        end
      end
    end
  endtask

  // Fill weight SRAM from gold_weight, matching DUT memory layout
  // Layout: addr = cout * cin * 9 + cin_idx * 9 + kh * 3 + kw
  task automatic fill_wgt_sram(int cout, int cin);
    for (int co = 0; co < cout; co++) begin
      for (int ci = 0; ci < cin; ci++) begin
        for (int kh = 0; kh < 3; kh++) begin
          for (int kw = 0; kw < 3; kw++) begin
            automatic int addr = co * cin * 9 + ci * 9 + kh * 3 + kw;
            wgt_sram[addr] = gold_weight[co][ci][kh][kw];
          end
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

    // Wait for done (with timeout)
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

    // Allow a few cycles for final write to settle
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
  // TEST 1 - MINIMAL
  // Conv3x3 stride=1, Cin=1, Cout=1, Hin=3, Win=32, pad=1
  // All weights=1, input=identity ramp, bias=0
  // Output should be sum of 3x3 window
  // ═══════════════════════════════════════════════════════════════════
  task automatic test1_minimal();
    automatic int mismatches;
    automatic int hin = 3, win = 32, cin = 1, cout = 1;
    automatic int stride_v = 1, pad_v = 1;
    automatic int h_pad = hin + 2 * pad_v;   // 5
    automatic int w_pad = win + 2 * pad_v;   // 34
    automatic int hout = hin;                 // (5-3)/1+1 = 3
    automatic int wout = win;                 // (34-3)/1+1 = 32

    $display("════════════════════════════════════════════════════");
    $display("TEST 1 - Minimal: stride=1, Cin=1, Cout=1, H=3, W=32");
    $display("════════════════════════════════════════════════════");

    clear_srams();
    clear_gold_arrays();
    reset_dut();

    // Fill padded input: row 0 and row 4 are zero (padding).
    // Cols 0 and col 33 are zero (padding). Inner pixels = row index.
    for (int h = 0; h < h_pad; h++)
      for (int w = 0; w < w_pad; w++)
        gold_input[h][w][0] = 8'sd0;  // zero everything

    for (int h = 1; h <= hin; h++)
      for (int w = 1; w <= win; w++)
        gold_input[h][w][0] = 8'(h);  // value = row (1,2,3)

    // Weights: all 1
    for (int kh = 0; kh < 3; kh++)
      for (int kw = 0; kw < 3; kw++)
        gold_weight[0][0][kh][kw] = 8'sd1;

    // Quantization: identity (m_int=1, shift=0, bias=0, zp_out=0)
    bias_arr[0]  = 32'sd0;
    m_int_arr[0] = 32'sd1;
    shift_arr[0] = 6'd0;

    // Fill SRAMs
    fill_ifm_sram(h_pad, w_pad, cin);
    fill_wgt_sram(cout, cin);

    // Configure DUT
    cfg_w_pad    = 10'(w_pad);
    cfg_hout     = 10'(hout);
    cfg_wout     = 10'(wout);
    cfg_cin      = 9'(cin);
    cfg_cout     = 9'(cout);
    cfg_stride   = 2'd1;
    cfg_act_mode = ACT_NONE;
    cfg_zp_out   = 8'sd0;

    // Golden
    t_cin      = cin;
    t_cout     = cout;
    t_hin_pad  = h_pad;
    t_win_pad  = w_pad;
    t_hout     = hout;
    t_wout     = wout;
    t_stride   = stride_v;
    t_pad      = pad_v;
    t_act_mode = ACT_NONE;
    t_zp_out   = 8'sd0;
    run_golden();

    // Run DUT
    run_dut();

    // Compare
    mismatches = compare_outputs(hout, wout, cout, (wout + TB_LANES - 1) / TB_LANES);

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
  // TEST 2 - LAYER 0 STYLE
  // Conv3x3 stride=2, Cin=3, Cout=4, Hin=8, Win=64, pad=1
  // Random weights, random input, realistic quantization, ACT_SILU
  // ═══════════════════════════════════════════════════════════════════
  task automatic test2_layer0();
    automatic int mismatches;
    automatic int hin = 8, win = 64, cin = 3, cout = 4;
    automatic int stride_v = 2, pad_v = 1;
    automatic int h_pad = hin + 2 * pad_v;   // 10
    automatic int w_pad = win + 2 * pad_v;   // 66
    automatic int hout = (h_pad - 3) / stride_v + 1;  // (10-3)/2+1 = 4
    automatic int wout = (w_pad - 3) / stride_v + 1;  // (66-3)/2+1 = 32

    $display("════════════════════════════════════════════════════");
    $display("TEST 2 - Layer 0 style: stride=2, Cin=3, Cout=4, Hin=8, Win=64");
    $display("════════════════════════════════════════════════════");

    clear_srams();
    clear_gold_arrays();
    reset_dut();

    // Fill padded input: zero padding borders, random inner values [-10,10]
    for (int h = 0; h < h_pad; h++)
      for (int w = 0; w < w_pad; w++)
        for (int c = 0; c < cin; c++)
          gold_input[h][w][c] = 8'sd0;

    for (int h = 1; h <= hin; h++)
      for (int w = 1; w <= win; w++)
        for (int c = 0; c < cin; c++)
          gold_input[h][w][c] = rand_range(-10, 10);

    // Random weights [-5, 5]
    for (int co = 0; co < cout; co++)
      for (int ci = 0; ci < cin; ci++)
        for (int kh = 0; kh < 3; kh++)
          for (int kw = 0; kw < 3; kw++)
            gold_weight[co][ci][kh][kw] = rand_range(-5, 5);

    // Realistic quantization parameters
    // m_int chosen so that output stays in INT8 range after shift
    // With Cin=3, max MAC ~= 3*9*10*5 = 1350, plus bias ~ small
    // m_int ~ 100, shift ~ 10 → (1350 * 100 + 512) >> 10 ~ 131 (fits INT8)
    for (int co = 0; co < cout; co++) begin
      bias_arr[co]  = $signed($urandom_range(0, 200)) - 100;
      m_int_arr[co] = 32'sd80 + $signed($urandom_range(0, 40));
      shift_arr[co] = 6'd10;
    end

    fill_ifm_sram(h_pad, w_pad, cin);
    fill_wgt_sram(cout, cin);

    // Configure DUT
    cfg_w_pad    = 10'(w_pad);
    cfg_hout     = 10'(hout);
    cfg_wout     = 10'(wout);
    cfg_cin      = 9'(cin);
    cfg_cout     = 9'(cout);
    cfg_stride   = 2'd2;
    cfg_act_mode = ACT_SILU;
    cfg_zp_out   = 8'sd0;

    // Golden
    t_cin      = cin;
    t_cout     = cout;
    t_hin_pad  = h_pad;
    t_win_pad  = w_pad;
    t_hout     = hout;
    t_wout     = wout;
    t_stride   = stride_v;
    t_pad      = pad_v;
    t_act_mode = ACT_SILU;
    t_zp_out   = 8'sd0;
    run_golden();

    run_dut();

    mismatches = compare_outputs(hout, wout, cout, (wout + TB_LANES - 1) / TB_LANES);

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
  // TEST 3 - LARGE CIN
  // Conv3x3 stride=1, Cin=16, Cout=4, Hin=4, Win=32, pad=1
  // All weights=1, all inputs=1 → pre-PPU output per pos = 9*16 = 144
  // ACT_NONE, identity requant
  // ═══════════════════════════════════════════════════════════════════
  task automatic test3_large_cin();
    automatic int mismatches;
    automatic int hin = 4, win = 32, cin = 16, cout = 4;
    automatic int stride_v = 1, pad_v = 1;
    automatic int h_pad = hin + 2 * pad_v;   // 6
    automatic int w_pad = win + 2 * pad_v;   // 34
    automatic int hout = hin;                 // 4
    automatic int wout = win;                 // 32

    $display("════════════════════════════════════════════════════");
    $display("TEST 3 - Large Cin: stride=1, Cin=16, Cout=4, Hin=4, Win=32");
    $display("════════════════════════════════════════════════════");

    clear_srams();
    clear_gold_arrays();
    reset_dut();

    // Fill padded input: zero borders, inner = 1
    for (int h = 0; h < h_pad; h++)
      for (int w = 0; w < w_pad; w++)
        for (int c = 0; c < cin; c++)
          gold_input[h][w][c] = 8'sd0;

    for (int h = 1; h <= hin; h++)
      for (int w = 1; w <= win; w++)
        for (int c = 0; c < cin; c++)
          gold_input[h][w][c] = 8'sd1;

    // All weights = 1
    for (int co = 0; co < cout; co++)
      for (int ci = 0; ci < cin; ci++)
        for (int kh = 0; kh < 3; kh++)
          for (int kw = 0; kw < 3; kw++)
            gold_weight[co][ci][kh][kw] = 8'sd1;

    // Identity quantization: m_int=1, shift=0, bias=0
    // 9*16=144 exceeds INT8 range, so use shift to bring into range
    // Use m_int=1, shift=1 to get 72 (with rounding) → fits INT8
    // Actually let's use per-spec identity: result will be clamped to 127
    for (int co = 0; co < cout; co++) begin
      bias_arr[co]  = 32'sd0;
      m_int_arr[co] = 32'sd1;
      shift_arr[co] = 6'd0;
    end

    fill_ifm_sram(h_pad, w_pad, cin);
    fill_wgt_sram(cout, cin);

    // Configure DUT
    cfg_w_pad    = 10'(w_pad);
    cfg_hout     = 10'(hout);
    cfg_wout     = 10'(wout);
    cfg_cin      = 9'(cin);
    cfg_cout     = 9'(cout);
    cfg_stride   = 2'd1;
    cfg_act_mode = ACT_NONE;
    cfg_zp_out   = 8'sd0;

    // Golden
    t_cin      = cin;
    t_cout     = cout;
    t_hin_pad  = h_pad;
    t_win_pad  = w_pad;
    t_hout     = hout;
    t_wout     = wout;
    t_stride   = stride_v;
    t_pad      = pad_v;
    t_act_mode = ACT_NONE;
    t_zp_out   = 8'sd0;
    run_golden();

    run_dut();

    mismatches = compare_outputs(hout, wout, cout, (wout + TB_LANES - 1) / TB_LANES);

    total_tests++;
    if (mismatches == 0) begin
      $display("  TEST 3: PASS");
      total_passed++;
    end else begin
      $display("  TEST 3: FAIL (%0d mismatches)", mismatches);
    end
    $display("");
  endtask

  // ═══════════════════════════════════════════════════════════════════
  // TEST 4 - RANDOM STRESS
  // Conv3x3 stride=1, Cin=8, Cout=8, Hin=6, Win=64, pad=1
  // Fully random input/weights, random quantization, ACT_RELU
  // ═══════════════════════════════════════════════════════════════════
  task automatic test4_random_stress();
    automatic int mismatches;
    automatic int hin = 6, win = 64, cin = 8, cout = 8;
    automatic int stride_v = 1, pad_v = 1;
    automatic int h_pad = hin + 2 * pad_v;   // 8
    automatic int w_pad = win + 2 * pad_v;   // 66
    automatic int hout = hin;                 // 6
    automatic int wout = win;                 // 64

    $display("════════════════════════════════════════════════════");
    $display("TEST 4 - Random Stress: stride=1, Cin=8, Cout=8, Hin=6, Win=64");
    $display("════════════════════════════════════════════════════");

    clear_srams();
    clear_gold_arrays();
    reset_dut();

    // Fill padded input: zero borders, random inner [-128,127]
    for (int h = 0; h < h_pad; h++)
      for (int w = 0; w < w_pad; w++)
        for (int c = 0; c < cin; c++)
          gold_input[h][w][c] = 8'sd0;

    for (int h = 1; h <= hin; h++)
      for (int w = 1; w <= win; w++)
        for (int c = 0; c < cin; c++)
          gold_input[h][w][c] = rand_int8();

    // Random weights [-128, 127]
    for (int co = 0; co < cout; co++)
      for (int ci = 0; ci < cin; ci++)
        for (int kh = 0; kh < 3; kh++)
          for (int kw = 0; kw < 3; kw++)
            gold_weight[co][ci][kh][kw] = rand_int8();

    // Random but valid quantization params
    // With Cin=8, max MAC ~= 8*9*128*128 = 1,179,648
    // Use m_int ~ 1-4, shift ~ 14-18 to bring into INT8 range
    // (1,179,648 * 4 + 8192) >> 16 ~ 72 → fits INT8
    for (int co = 0; co < cout; co++) begin
      bias_arr[co]  = $signed($urandom_range(0, 2000)) - 1000;
      m_int_arr[co] = 32'sd1 + $signed($urandom_range(0, 3));
      shift_arr[co] = 6'd16 + $urandom_range(0, 2);
    end

    fill_ifm_sram(h_pad, w_pad, cin);
    fill_wgt_sram(cout, cin);

    // Configure DUT
    cfg_w_pad    = 10'(w_pad);
    cfg_hout     = 10'(hout);
    cfg_wout     = 10'(wout);
    cfg_cin      = 9'(cin);
    cfg_cout     = 9'(cout);
    cfg_stride   = 2'd1;
    cfg_act_mode = ACT_RELU;
    cfg_zp_out   = 8'sd0;

    // Golden
    t_cin      = cin;
    t_cout     = cout;
    t_hin_pad  = h_pad;
    t_win_pad  = w_pad;
    t_hout     = hout;
    t_wout     = wout;
    t_stride   = stride_v;
    t_pad      = pad_v;
    t_act_mode = ACT_RELU;
    t_zp_out   = 8'sd0;
    run_golden();

    run_dut();

    mismatches = compare_outputs(hout, wout, cout, (wout + TB_LANES - 1) / TB_LANES);

    total_tests++;
    if (mismatches == 0) begin
      $display("  TEST 4: PASS");
      total_passed++;
    end else begin
      $display("  TEST 4: FAIL (%0d mismatches)", mismatches);
    end
    $display("");
  endtask

  // ═══════════════════════════════════════════════════════════════════
  // MAIN TEST SEQUENCE
  // ═══════════════════════════════════════════════════════════════════
  initial begin
    $display("");
    $display("╔══════════════════════════════════════════════════════╗");
    $display("║  Golden Verification TB for conv3x3_engine          ║");
    $display("╚══════════════════════════════════════════════════════╝");
    $display("");

    // Initialize SiLU LUT
    for (int i = 0; i < 256; i++)
      silu_lut[i] = compute_silu_entry(i);

    // Initialize signals
    clk     = 1'b0;
    rst_n   = 1'b0;
    start   = 1'b0;
    cfg_w_pad    = '0;
    cfg_hout     = '0;
    cfg_wout     = '0;
    cfg_cin      = '0;
    cfg_cout     = '0;
    cfg_stride   = '0;
    cfg_act_mode = ACT_NONE;
    cfg_zp_out   = 8'sd0;

    // Seed random
    $urandom(42);

    // Run all tests
    test1_minimal();
    test2_layer0();
    test3_large_cin();
    test4_random_stress();

    // Final summary
    $display("╔══════════════════════════════════════════════════════╗");
    $display("║  FINAL RESULTS: %0d / %0d tests PASSED              ║",
             total_passed, total_tests);
    if (total_passed == total_tests)
      $display("║  >>>  ALL TESTS PASSED  <<<                         ║");
    else
      $display("║  >>>  SOME TESTS FAILED  <<<                        ║");
    $display("╚══════════════════════════════════════════════════════╝");
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
