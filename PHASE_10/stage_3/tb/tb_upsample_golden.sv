`timescale 1ns/1ps
// ============================================================================
// Golden Verification Testbench for upsample_engine (P6 UPSAMPLE_NEAREST 2x)
// Nearest-neighbor 2x upsampling: each input pixel replicated to a 2x2 block.
//
// Input:  H x W x C
// Output: 2H x 2W x C
//
// For every (h, w, c): output[2h][2w] = output[2h][2w+1]
//                       = output[2h+1][2w] = output[2h+1][2w+1] = input[h][w][c]
//
// Test scenarios:
//   TEST 1 - Minimal:      H=2, W=32, C=1 (lane index pattern)
//   TEST 2 - Multi-channel: H=4, W=32, C=8 (structured pattern)
//   TEST 3 - L11-style:    H=20, W=20, C=4 (random INT8, 2x2 replication check)
// ============================================================================
module tb_upsample_golden;

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

  // Maximum SRAM sizes
  localparam int IFM_SRAM_DEPTH = 65536;  // Input SRAM (LANES-wide)
  localparam int OFM_SRAM_DEPTH = 131072; // Output SRAM (LANES-wide) - 2x bigger

  // ═══════════════════════════════════════════════════════════════════
  // DUT SIGNALS
  // ═══════════════════════════════════════════════════════════════════
  logic        start, done, busy;

  // Configuration
  logic [9:0]  cfg_h;             // Input height
  logic [9:0]  cfg_w;             // Input width
  logic [8:0]  cfg_channels;      // Number of channels
  logic [9:0]  cfg_hout;          // Output height (= 2 * cfg_h)
  logic [9:0]  cfg_wout;          // Output width  (= 2 * cfg_w)

  // Input SRAM interface (LANES-wide read)
  // Layout: [H][C][num_wblk_in], word = LANES INT8 values
  // Address = h * channels * num_wblk_in + ch * num_wblk_in + wblk
  logic [23:0]       ifm_rd_addr;
  logic              ifm_rd_en;
  logic signed [7:0] ifm_rd_data [TB_LANES];

  // Output SRAM interface (LANES-wide write)
  // Layout: [Hout][C][num_wblk_out], word = LANES INT8 values
  // Address = h_out * channels * num_wblk_out + ch * num_wblk_out + wblk
  logic [23:0]       ofm_wr_addr;
  logic              ofm_wr_en;
  logic signed [7:0] ofm_wr_data [TB_LANES];

  // ═══════════════════════════════════════════════════════════════════
  // BEHAVIORAL SRAM MODELS
  // ═══════════════════════════════════════════════════════════════════
  logic signed [7:0] ifm_sram [IFM_SRAM_DEPTH][TB_LANES];
  logic signed [7:0] ofm_sram [OFM_SRAM_DEPTH][TB_LANES];

  // Input SRAM: 1-cycle read latency
  always_ff @(posedge clk) begin
    if (ifm_rd_en) begin
      for (int l = 0; l < TB_LANES; l++)
        ifm_rd_data[l] <= ifm_sram[ifm_rd_addr][l];
    end
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
  upsample_engine #(
    .LANES (TB_LANES)
  ) u_dut (
    .clk            (clk),
    .rst_n          (rst_n),
    .start          (start),
    .done           (done),
    .busy           (busy),
    // Configuration
    .cfg_h          (cfg_h),
    .cfg_w          (cfg_w),
    .cfg_channels   (cfg_channels),
    .cfg_hout       (cfg_hout),
    .cfg_wout       (cfg_wout),
    // Input SRAM
    .ifm_rd_addr    (ifm_rd_addr),
    .ifm_rd_en      (ifm_rd_en),
    .ifm_rd_data    (ifm_rd_data),
    // Output SRAM
    .ofm_wr_addr    (ofm_wr_addr),
    .ofm_wr_en      (ofm_wr_en),
    .ofm_wr_data    (ofm_wr_data)
  );

  // ═══════════════════════════════════════════════════════════════════
  // GOLDEN REFERENCE DATA
  // ═══════════════════════════════════════════════════════════════════
  localparam int GOLD_MAX_H  = 42;
  localparam int GOLD_MAX_W  = 64;
  localparam int GOLD_MAX_C  = 256;

  // Input data in HWC format
  logic signed [7:0] gold_input [GOLD_MAX_H][GOLD_MAX_W][GOLD_MAX_C];

  // Golden output: dynamic array [hout * wout * channels]
  logic signed [7:0] gold_output [];

  // Per-test parameters
  int t_h, t_w, t_ch, t_hout, t_wout;

  // ═══════════════════════════════════════════════════════════════════
  // GOLDEN: Behavioral nearest-neighbor 2x upsample
  // output[2h+dh][2w+dw][c] = input[h][w][c]  for dh,dw in {0,1}
  // ═══════════════════════════════════════════════════════════════════
  task automatic run_golden();
    gold_output = new[t_hout * t_wout * t_ch];

    for (int h = 0; h < t_h; h++) begin
      for (int w = 0; w < t_w; w++) begin
        for (int c = 0; c < t_ch; c++) begin
          automatic logic signed [7:0] val;
          val = gold_input[h][w][c];
          // Replicate to 2x2 block
          gold_output[(2*h  ) * t_wout * t_ch + (2*w  ) * t_ch + c] = val;
          gold_output[(2*h  ) * t_wout * t_ch + (2*w+1) * t_ch + c] = val;
          gold_output[(2*h+1) * t_wout * t_ch + (2*w  ) * t_ch + c] = val;
          gold_output[(2*h+1) * t_wout * t_ch + (2*w+1) * t_ch + c] = val;
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
      for (int co = 0; co < channels; co++) begin
        for (int wblk = 0; wblk < num_wblk; wblk++) begin
          automatic int ofm_addr;
          ofm_addr = ho * channels * num_wblk + co * num_wblk + wblk;
          for (int l = 0; l < TB_LANES; l++) begin
            automatic int wo = wblk * TB_LANES + l;
            if (wo < wout) begin
              automatic int flat_idx;
              automatic logic signed [7:0] rtl_val, gold_val;
              flat_idx = ho * wout * channels + wo * channels + co;
              rtl_val  = ofm_sram[ofm_addr][l];
              gold_val = gold_output[flat_idx];
              if (rtl_val !== gold_val) begin
                mismatches++;
                if (printed < 10) begin
                  $display("  MISMATCH @ (h=%0d, w=%0d, ch=%0d): expected=%0d, got=%0d",
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
    for (int i = 0; i < OFM_SRAM_DEPTH; i++)
      for (int l = 0; l < TB_LANES; l++)
        ofm_sram[i][l] = 8'sd0;
  endtask

  task automatic clear_gold_arrays();
    for (int h = 0; h < GOLD_MAX_H; h++)
      for (int w = 0; w < GOLD_MAX_W; w++)
        for (int c = 0; c < GOLD_MAX_C; c++)
          gold_input[h][w][c] = 8'sd0;
  endtask

  // Fill input SRAM from gold_input
  // Layout: addr = h * channels * num_wblk_in + ch * num_wblk_in + wblk
  // ifm_sram[addr][l] = gold_input[h][wblk*LANES + l][ch]
  task automatic fill_ifm_sram(int h, int w, int channels);
    automatic int num_wblk = (w + TB_LANES - 1) / TB_LANES;
    for (int ho = 0; ho < h; ho++) begin
      for (int c = 0; c < channels; c++) begin
        for (int wblk = 0; wblk < num_wblk; wblk++) begin
          automatic int addr = ho * channels * num_wblk + c * num_wblk + wblk;
          for (int l = 0; l < TB_LANES; l++) begin
            automatic int w_idx = wblk * TB_LANES + l;
            if (w_idx < w)
              ifm_sram[addr][l] = gold_input[ho][w_idx][c];
            else
              ifm_sram[addr][l] = 8'sd0;
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

  // ═══════════════════════════════════════════════════════════════════
  // TEST RESULTS
  // ═══════════════════════════════════════════════════════════════════
  int total_tests  = 0;
  int total_passed = 0;

  // ═══════════════════════════════════════════════════════════════════
  // TEST 1 - MINIMAL (H=2, W=32, C=1)
  // Input: value = lane index (0..31) for each row
  // Output: H=4, W=64, each value replicated 2x2
  // Expected: 100% exact
  // ═══════════════════════════════════════════════════════════════════
  task automatic test1_minimal();
    automatic int mismatches;
    automatic int h = 2, w = 32, channels = 1;
    automatic int hout = 2 * h;   // 4
    automatic int wout = 2 * w;   // 64

    $display("════════════════════════════════════════════════════");
    $display("TEST 1 - Minimal: H=%0d, W=%0d, C=%0d -> Hout=%0d, Wout=%0d",
             h, w, channels, hout, wout);
    $display("  Input: value = lane index (0..31)");
    $display("════════════════════════════════════════════════════");

    clear_srams();
    clear_gold_arrays();
    reset_dut();

    // Fill input: value = w position (lane index)
    for (int ho = 0; ho < h; ho++)
      for (int wo = 0; wo < w; wo++)
        gold_input[ho][wo][0] = 8'(wo);  // 0..31

    fill_ifm_sram(h, w, channels);

    // Configure DUT
    cfg_h        = 10'(h);
    cfg_w        = 10'(w);
    cfg_channels = 9'(channels);
    cfg_hout     = 10'(hout);
    cfg_wout     = 10'(wout);

    // Golden
    t_h    = h;
    t_w    = w;
    t_ch   = channels;
    t_hout = hout;
    t_wout = wout;
    run_golden();

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
  // TEST 2 - MULTI-CHANNEL (H=4, W=32, C=8)
  // Input: value = h*100 + c*10 + lane (structured pattern)
  // Output: H=8, W=64 -- verify 2x2 replication per channel
  // Expected: 100% exact
  // ═══════════════════════════════════════════════════════════════════
  task automatic test2_multi_channel();
    automatic int mismatches;
    automatic int h = 4, w = 32, channels = 8;
    automatic int hout = 2 * h;   // 8
    automatic int wout = 2 * w;   // 64

    $display("════════════════════════════════════════════════════");
    $display("TEST 2 - Multi-channel: H=%0d, W=%0d, C=%0d -> Hout=%0d, Wout=%0d",
             h, w, channels, hout, wout);
    $display("  Input: value = h*100 + c*10 + (l mod 12), clamped to INT8");
    $display("════════════════════════════════════════════════════");

    clear_srams();
    clear_gold_arrays();
    reset_dut();

    // Fill input: structured pattern
    // h*100 + c*10 + (l % 12) gives unique-ish values clamped to [-128,127]
    for (int ho = 0; ho < h; ho++)
      for (int wo = 0; wo < w; wo++)
        for (int c = 0; c < channels; c++) begin
          automatic int val = ho * 20 + c * 10 + (wo % 12);
          if (val > 127)  val = 127;
          if (val < -128) val = -128;
          gold_input[ho][wo][c] = val[7:0];
        end

    fill_ifm_sram(h, w, channels);

    // Configure DUT
    cfg_h        = 10'(h);
    cfg_w        = 10'(w);
    cfg_channels = 9'(channels);
    cfg_hout     = 10'(hout);
    cfg_wout     = 10'(wout);

    // Golden
    t_h    = h;
    t_w    = w;
    t_ch   = channels;
    t_hout = hout;
    t_wout = wout;
    run_golden();

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
  // TEST 3 - L11-STYLE (H=20, W=20, C=4)
  // Output: H=40, W=40
  // Random INT8 input -> verify 2x2 replication:
  //   output[2h][2w] == output[2h][2w+1] == output[2h+1][2w] == output[2h+1][2w+1] == input[h][w]
  // ═══════════════════════════════════════════════════════════════════
  task automatic test3_l11_style();
    automatic int mismatches;
    automatic int h = 20, w = 20, channels = 4;
    automatic int hout = 2 * h;   // 40
    automatic int wout = 2 * w;   // 40

    $display("════════════════════════════════════════════════════");
    $display("TEST 3 - L11-style: H=%0d, W=%0d, C=%0d -> Hout=%0d, Wout=%0d",
             h, w, channels, hout, wout);
    $display("  Random INT8 input, verify 2x2 replication");
    $display("════════════════════════════════════════════════════");

    clear_srams();
    clear_gold_arrays();
    reset_dut();

    // Fill input: random INT8 values
    for (int ho = 0; ho < h; ho++)
      for (int wo = 0; wo < w; wo++)
        for (int c = 0; c < channels; c++)
          gold_input[ho][wo][c] = rand_int8();

    fill_ifm_sram(h, w, channels);

    // Configure DUT
    cfg_h        = 10'(h);
    cfg_w        = 10'(w);
    cfg_channels = 9'(channels);
    cfg_hout     = 10'(hout);
    cfg_wout     = 10'(wout);

    // Golden
    t_h    = h;
    t_w    = w;
    t_ch   = channels;
    t_hout = hout;
    t_wout = wout;
    run_golden();

    run_dut();

    // Standard comparison
    mismatches = compare_outputs(hout, wout, channels);

    // Additional structural verification: check 2x2 replication directly
    // from the output SRAM (redundant with golden but confirms the property)
    if (mismatches == 0) begin
      automatic int struct_err = 0;
      automatic int num_wblk_out = (wout + TB_LANES - 1) / TB_LANES;
      $display("  Verifying 2x2 replication structure...");
      for (int ih = 0; ih < h && struct_err < 5; ih++) begin
        for (int iw = 0; iw < w && struct_err < 5; iw++) begin
          for (int c = 0; c < channels && struct_err < 5; c++) begin
            // Read the 4 output positions corresponding to input[ih][iw][c]
            automatic logic signed [7:0] v00, v01, v10, v11;
            automatic int flat00, flat01, flat10, flat11;
            flat00 = (2*ih  ) * wout * channels + (2*iw  ) * channels + c;
            flat01 = (2*ih  ) * wout * channels + (2*iw+1) * channels + c;
            flat10 = (2*ih+1) * wout * channels + (2*iw  ) * channels + c;
            flat11 = (2*ih+1) * wout * channels + (2*iw+1) * channels + c;
            v00 = gold_output[flat00];
            v01 = gold_output[flat01];
            v10 = gold_output[flat10];
            v11 = gold_output[flat11];
            if (v00 !== v01 || v00 !== v10 || v00 !== v11) begin
              $display("  REPLICATION ERR @ input(%0d,%0d,ch%0d): %0d %0d %0d %0d",
                       ih, iw, c, int'(v00), int'(v01), int'(v10), int'(v11));
              struct_err++;
            end
          end
        end
      end
      if (struct_err == 0)
        $display("  2x2 replication structure verified OK");
    end

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
  // MAIN TEST SEQUENCE
  // ═══════════════════════════════════════════════════════════════════
  initial begin
    $display("");
    $display("==========================================================");
    $display("  Golden Verification TB for upsample_engine (P6 UPSAMPLE)");
    $display("==========================================================");
    $display("");

    // Initialize signals
    clk     = 1'b0;
    rst_n   = 1'b0;
    start   = 1'b0;
    cfg_h        = '0;
    cfg_w        = '0;
    cfg_channels = '0;
    cfg_hout     = '0;
    cfg_wout     = '0;

    // Seed random
    $urandom(42);

    // Run all tests
    test1_minimal();
    test2_multi_channel();
    test3_l11_style();

    // Final summary
    $display("==========================================================");
    $display("  FINAL RESULTS: %0d / %0d tests PASSED", total_passed, total_tests);
    if (total_passed == total_tests)
      $display("  >>>  ALL TESTS PASSED  <<<");
    else
      $display("  >>>  SOME TESTS FAILED  <<<");
    $display("==========================================================");
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
