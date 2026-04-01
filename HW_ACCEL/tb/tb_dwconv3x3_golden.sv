`timescale 1ns/1ps
// ============================================================================
// Testbench: dwconv3x3_engine with golden-model comparison
//
// Tests:
//   TEST 1 — Minimal:     C=1,  H=3,  W=32,  stride=1
//   TEST 2 — SCDown style: C=8,  H=8,  W=64,  stride=2
//   TEST 3 — Random:       C=16, H=6,  W=32,  stride=1
//
// Golden model:
//   for (h, w, c):
//     acc = Sum_{kh,kw} input[h*s+kh][w*s+kw][c] * wgt[c][kh][kw]
//     biased  = acc + bias[c]
//     requant = (biased * m_int[c] + round) >>> shift[c]
//     act     = activation(clamp16(requant))
//     out     = clamp8(act + zp_out)
// ============================================================================
module tb_dwconv3x3_golden;
  import yolo_accel_pkg::*;

  // -------------------------------------------------------------------------
  // Parameters
  // -------------------------------------------------------------------------
  localparam int CLK_PERIOD  = 5;     // 200 MHz
  localparam int TB_LANES    = 32;
  localparam int TB_MAX_W    = 672;
  localparam int TIMEOUT     = 500_000;

  // -------------------------------------------------------------------------
  // DUT signals
  // -------------------------------------------------------------------------
  logic        clk, rst_n, start, done, busy;

  logic [9:0]  cfg_w_pad, cfg_hout, cfg_wout;
  logic [8:0]  cfg_channels;
  logic [1:0]  cfg_stride;
  act_mode_e   cfg_act_mode;
  logic signed [7:0] cfg_zp_out;

  logic [23:0]       ifm_rd_addr;
  logic              ifm_rd_en;
  logic signed [7:0] ifm_rd_data [TB_LANES];

  logic [23:0]       wgt_rd_addr;
  logic              wgt_rd_en;
  logic signed [7:0] wgt_rd_data;

  logic signed [31:0] bias_arr  [MAX_CIN];
  logic signed [31:0] m_int_arr [MAX_CIN];
  logic [5:0]         shift_arr [MAX_CIN];

  logic signed [7:0]  silu_lut [256];

  logic [23:0]       ofm_wr_addr;
  logic              ofm_wr_en;
  logic signed [7:0] ofm_wr_data [TB_LANES];

  // -------------------------------------------------------------------------
  // Memory models
  // -------------------------------------------------------------------------
  // IFM SRAM: addressed as flat byte array, returns TB_LANES bytes per read
  // Layout: [H_pad][C][num_wblk_in], word = LANES int8 values
  localparam int IFM_DEPTH = 1 << 20;   // 1M entries -- large enough
  logic signed [7:0] ifm_mem [IFM_DEPTH];

  // Weight SRAM: byte-addressed, layout [C][3][3]
  localparam int WGT_DEPTH = MAX_CIN * 9;
  logic signed [7:0] wgt_mem [WGT_DEPTH];

  // OFM SRAM: LANES-wide words captured on write
  localparam int OFM_DEPTH = 1 << 20;
  logic signed [7:0] ofm_mem [OFM_DEPTH];

  // IFM read model (1-cycle latency: data available the cycle after rd_en)
  always_ff @(posedge clk) begin
    if (ifm_rd_en) begin
      for (int l = 0; l < TB_LANES; l++) begin
        automatic int flat;
        flat = int'(ifm_rd_addr) * TB_LANES + l;
        if (flat < IFM_DEPTH)
          ifm_rd_data[l] <= ifm_mem[flat];
        else
          ifm_rd_data[l] <= 8'sd0;
      end
    end
  end

  // Weight read model (1-cycle latency)
  always_ff @(posedge clk) begin
    if (wgt_rd_en) begin
      if (int'(wgt_rd_addr) < WGT_DEPTH)
        wgt_rd_data <= wgt_mem[wgt_rd_addr];
      else
        wgt_rd_data <= 8'sd0;
    end
  end

  // OFM write capture
  always_ff @(posedge clk) begin
    if (ofm_wr_en) begin
      for (int l = 0; l < TB_LANES; l++) begin
        automatic int flat;
        flat = int'(ofm_wr_addr) * TB_LANES + l;
        if (flat < OFM_DEPTH)
          ofm_mem[flat] <= ofm_wr_data[l];
      end
    end
  end

  // -------------------------------------------------------------------------
  // DUT instantiation
  // -------------------------------------------------------------------------
  dwconv3x3_engine #(
    .LANES     (TB_LANES),
    .MAX_W_PAD (TB_MAX_W)
  ) u_dut (
    .clk          (clk),
    .rst_n        (rst_n),
    .start        (start),
    .done         (done),
    .busy         (busy),
    .cfg_w_pad    (cfg_w_pad),
    .cfg_hout     (cfg_hout),
    .cfg_wout     (cfg_wout),
    .cfg_channels (cfg_channels),
    .cfg_stride   (cfg_stride),
    .cfg_act_mode (cfg_act_mode),
    .cfg_zp_out   (cfg_zp_out),
    .ifm_rd_addr  (ifm_rd_addr),
    .ifm_rd_en    (ifm_rd_en),
    .ifm_rd_data  (ifm_rd_data),
    .wgt_rd_addr  (wgt_rd_addr),
    .wgt_rd_en    (wgt_rd_en),
    .wgt_rd_data  (wgt_rd_data),
    .bias_arr     (bias_arr),
    .m_int_arr    (m_int_arr),
    .shift_arr    (shift_arr),
    .silu_lut     (silu_lut),
    .ofm_wr_addr  (ofm_wr_addr),
    .ofm_wr_en    (ofm_wr_en),
    .ofm_wr_data  (ofm_wr_data)
  );

  // -------------------------------------------------------------------------
  // Clock
  // -------------------------------------------------------------------------
  initial clk = 0;
  always #(CLK_PERIOD/2.0) clk = ~clk;

  // -------------------------------------------------------------------------
  // Golden-model helper functions
  // -------------------------------------------------------------------------

  // SiLU lookup (mirrors yolo_accel_pkg::silu_index)
  function automatic logic signed [7:0] golden_silu(input int val);
    automatic int idx;
    idx = val + 128;
    if (idx < 0)   idx = 0;
    if (idx > 255) idx = 255;
    return silu_lut[idx];
  endfunction

  // Full per-channel PPU (bias + requant + activation + zp + clamp)
  function automatic logic signed [7:0] golden_ppu(
    input int          acc,
    input int          ch,
    input act_mode_e   act,
    input int          zp_out
  );
    automatic longint  biased, mult, rounded;
    automatic int      shifted, act_in_i;
    automatic logic signed [7:0] act_val;
    automatic int      final_val;

    biased = longint'(acc) + longint'(bias_arr[ch]);

    // Requant: (biased * m_int + round) >>> shift
    mult = biased * longint'(m_int_arr[ch]);
    if (shift_arr[ch] > 0)
      rounded = mult + (longint'(1) <<< (shift_arr[ch] - 1));
    else
      rounded = mult;
    shifted = int'(rounded >>> shift_arr[ch]);

    // Clamp to 16-bit for activation
    if (shifted > 32767)       act_in_i = 32767;
    else if (shifted < -32768) act_in_i = -32768;
    else                       act_in_i = shifted;

    // Activation
    case (act)
      ACT_SILU: act_val = golden_silu(act_in_i);
      ACT_RELU: act_val = (act_in_i > 0) ?
                           ((act_in_i > 127) ? 8'sd127 : act_in_i[7:0]) : 8'sd0;
      default:  begin // ACT_NONE / ACT_CLAMP
        if (act_in_i > 127)       act_val = 8'sd127;
        else if (act_in_i < -128) act_val = -8'sd128;
        else                      act_val = act_in_i[7:0];
      end
    endcase

    // Zero-point + final clamp
    final_val = int'(act_val) + zp_out;
    if (final_val > 127)       return 8'sd127;
    else if (final_val < -128) return -8'sd128;
    else                       return final_val[7:0];
  endfunction

  // -------------------------------------------------------------------------
  // Test infrastructure
  // -------------------------------------------------------------------------
  int total_tests, total_pass, total_fail;

  task automatic run_test(
    input string name,
    input int    C, H, W, stride,
    input act_mode_e act,
    input int    zp,
    input bit    random_data
  );
    // Derived dimensions
    automatic int pad        = 1;
    automatic int H_pad      = H + 2 * pad;
    automatic int W_pad      = W + 2 * pad;
    automatic int Hout       = (H_pad - 3) / stride + 1;
    automatic int Wout       = (W_pad - 3) / stride + 1;
    automatic int nwblk_in   = (W_pad + TB_LANES - 1) / TB_LANES;
    automatic int nwblk_out  = (Wout + TB_LANES - 1) / TB_LANES;
    automatic int seed       = 42;
    automatic int mismatches = 0;

    $display("──────────────────────────────────────────────────");
    $display("  TEST: %s", name);
    $display("  C=%0d  H=%0d  W=%0d  stride=%0d  act=%s  zp=%0d",
             C, H, W, stride, act.name(), zp);
    $display("  H_pad=%0d  W_pad=%0d  Hout=%0d  Wout=%0d", H_pad, W_pad, Hout, Wout);
    $display("──────────────────────────────────────────────────");

    // ---- Initialize memories ----
    for (int i = 0; i < IFM_DEPTH; i++) ifm_mem[i] = 8'sd0;
    for (int i = 0; i < WGT_DEPTH; i++) wgt_mem[i] = 8'sd0;
    for (int i = 0; i < OFM_DEPTH; i++) ofm_mem[i] = 8'sd0;

    // Fill IFM (padded layout): ifm_mem[ (h*C*nwblk_in + c*nwblk_in + w_word) * LANES + l ]
    // Pad regions stay 0 (zero padding).
    for (int h = 0; h < H; h++) begin
      for (int c = 0; c < C; c++) begin
        for (int w = 0; w < W; w++) begin
          automatic int hp   = h + pad;
          automatic int wp   = w + pad;
          automatic int ww   = wp / TB_LANES;
          automatic int wl   = wp % TB_LANES;
          automatic int flat = (hp * C * nwblk_in + c * nwblk_in + ww) * TB_LANES + wl;
          if (random_data) begin
            ifm_mem[flat] = $signed(8'($random(seed) % 256));
          end else begin
            // Deterministic: (h + w + c) mod 127
            ifm_mem[flat] = $signed(8'((h + w + c) % 127));
          end
        end
      end
    end

    // Fill weights [C][3][3]
    for (int c = 0; c < C; c++) begin
      for (int kh = 0; kh < 3; kh++) begin
        for (int kw = 0; kw < 3; kw++) begin
          if (random_data)
            wgt_mem[c * 9 + kh * 3 + kw] = $signed(8'($random(seed) % 256));
          else
            wgt_mem[c * 9 + kh * 3 + kw] = $signed(8'((c + kh + kw + 1) % 5));
        end
      end
    end

    // Fill per-channel bias / quant params
    for (int c = 0; c < C; c++) begin
      if (random_data) begin
        bias_arr[c]  = $random(seed) % 1024;
        m_int_arr[c] = ($random(seed) % 4096) + 1;
        shift_arr[c] = ($random(seed) % 16) + 4;
      end else begin
        bias_arr[c]  = 32'(c);
        m_int_arr[c] = 32'd1;
        shift_arr[c] = 6'd0;
      end
    end

    // Init SiLU LUT (simple identity-clamp for non-SILU tests)
    for (int i = 0; i < 256; i++) begin
      automatic int val = i - 128;
      silu_lut[i] = (val > 127) ? 8'sd127 : (val < -128) ? -8'sd128 : val[7:0];
    end

    // ---- Configure DUT ----
    cfg_w_pad    = W_pad[9:0];
    cfg_hout     = Hout[9:0];
    cfg_wout     = Wout[9:0];
    cfg_channels = C[8:0];
    cfg_stride   = stride[1:0];
    cfg_act_mode = act;
    cfg_zp_out   = zp[7:0];

    // ---- Launch ----
    @(posedge clk);
    start <= 1'b1;
    @(posedge clk);
    start <= 1'b0;

    // Wait for done
    begin
      automatic int cyc = 0;
      while (!done && cyc < TIMEOUT) begin
        @(posedge clk);
        cyc++;
      end
      if (cyc >= TIMEOUT) begin
        $display("  ** TIMEOUT after %0d cycles **", cyc);
        total_fail++;
        return;
      end
      $display("  Engine completed in %0d cycles", cyc);
    end

    // ---- Golden check ----
    for (int ho = 0; ho < Hout; ho++) begin
      for (int c = 0; c < C; c++) begin
        for (int wo = 0; wo < Wout; wo++) begin
          automatic int acc = 0;
          automatic logic signed [7:0] expected;
          automatic logic signed [7:0] actual;
          automatic int ofm_ww, ofm_wl, ofm_flat;

          // Convolution
          for (int kh = 0; kh < 3; kh++) begin
            for (int kw = 0; kw < 3; kw++) begin
              automatic int hp = ho * stride + kh;
              automatic int wp = wo * stride + kw;
              automatic int iww  = wp / TB_LANES;
              automatic int iwl  = wp % TB_LANES;
              automatic int iflat = (hp * C * nwblk_in + c * nwblk_in + iww) * TB_LANES + iwl;
              automatic int wflat = c * 9 + kh * 3 + kw;
              acc = acc + int'(ifm_mem[iflat]) * int'(wgt_mem[wflat]);
            end
          end

          expected = golden_ppu(acc, c, act, zp);

          // Read DUT output
          ofm_ww   = wo / TB_LANES;
          ofm_wl   = wo % TB_LANES;
          ofm_flat = (ho * C * nwblk_out + c * nwblk_out + ofm_ww) * TB_LANES + ofm_wl;
          actual   = ofm_mem[ofm_flat];

          if (actual !== expected) begin
            if (mismatches < 20)
              $display("  MISMATCH [h=%0d c=%0d w=%0d]: DUT=%0d  golden=%0d  (acc=%0d)",
                       ho, c, wo, actual, expected, acc);
            mismatches++;
          end
        end
      end
    end

    total_tests++;
    if (mismatches == 0) begin
      $display("  PASS -- all %0d outputs match", Hout * C * Wout);
      total_pass++;
    end else begin
      $display("  FAIL -- %0d / %0d mismatches", mismatches, Hout * C * Wout);
      total_fail++;
    end
  endtask

  // -------------------------------------------------------------------------
  // Main stimulus
  // -------------------------------------------------------------------------
  initial begin
    $display("============================================================");
    $display(" tb_dwconv3x3_golden — DW 3x3 engine verification");
    $display("============================================================");

    total_tests = 0;
    total_pass  = 0;
    total_fail  = 0;

    rst_n = 1'b0;
    start = 1'b0;
    cfg_w_pad    = '0;
    cfg_hout     = '0;
    cfg_wout     = '0;
    cfg_channels = '0;
    cfg_stride   = '0;
    cfg_act_mode = ACT_NONE;
    cfg_zp_out   = '0;

    repeat (10) @(posedge clk);
    rst_n = 1'b1;
    repeat (5) @(posedge clk);

    // TEST 1: Minimal — C=1, H=3, W=32, stride=1, no activation
    run_test("Minimal (C=1 H=3 W=32 s=1)",
             1, 3, 32, 1, ACT_NONE, 0, 1'b0);

    repeat (10) @(posedge clk);

    // TEST 2: SCDown style — C=8, H=8, W=64, stride=2, ReLU activation
    run_test("SCDown (C=8 H=8 W=64 s=2)",
             8, 8, 64, 2, ACT_RELU, 0, 1'b0);

    repeat (10) @(posedge clk);

    // TEST 3: Random — C=16, H=6, W=32, stride=1, no activation, random data
    run_test("Random (C=16 H=6 W=32 s=1)",
             16, 6, 32, 1, ACT_NONE, 0, 1'b1);

    repeat (10) @(posedge clk);

    // ---- Summary ----
    $display("");
    $display("============================================================");
    $display(" SUMMARY:  %0d / %0d tests passed, %0d failed",
             total_pass, total_tests, total_fail);
    $display("============================================================");

    if (total_fail == 0)
      $display(" ALL TESTS PASSED");
    else
      $display(" SOME TESTS FAILED");

    $finish;
  end

  // -------------------------------------------------------------------------
  // Timeout watchdog
  // -------------------------------------------------------------------------
  initial begin
    #(TIMEOUT * CLK_PERIOD * 10);
    $display("GLOBAL TIMEOUT -- aborting");
    $finish;
  end

endmodule
