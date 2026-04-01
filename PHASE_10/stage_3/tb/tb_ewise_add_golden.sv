`timescale 1ns/1ps
// ============================================================================
// Golden Verification Testbench for ewise_add_engine (P7 EWISE_ADD)
// Element-wise addition with domain alignment (requantization).
//
// For each (h, w, c):
//   real_a = (q_a - zp_a) * scale_a
//   real_b = (q_b - zp_b) * scale_b
//   real_out = real_a + real_b
//   q_out = clamp(round(real_out / scale_out) + zp_out, -128, 127)
//
// Implemented as:
//   q_out = clamp( requant(q_a - zp_a, m_a, shift_a)
//                + requant(q_b - zp_b, m_b, shift_b) + zp_out, -128, 127)
//
// Test scenarios:
//   TEST 1 - Same scale:       simple add, no requant needed
//   TEST 2 - Different scales: domain alignment with requantization
//   TEST 3 - Saturation test:  values near 127 -> clamp at INT8 max
//   TEST 4 - QC2fCIB residual: realistic L22 bottleneck residual connection
// ============================================================================
module tb_ewise_add_golden;

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
  localparam int IFM_A_SRAM_DEPTH = 65536;
  localparam int IFM_B_SRAM_DEPTH = 65536;
  localparam int OFM_SRAM_DEPTH   = 65536;

  // ═══════════════════════════════════════════════════════════════════
  // DUT SIGNALS
  // ═══════════════════════════════════════════════════════════════════
  logic        start, done, busy;

  // Configuration
  logic [9:0]  cfg_h;             // Spatial height (same for A, B, output)
  logic [9:0]  cfg_w;             // Spatial width
  logic [8:0]  cfg_channels;      // Number of channels (same for A, B, output)

  // Quantization params for input A
  logic signed [31:0] cfg_scale_a;    // m_int for A->output requant
  logic [5:0]         cfg_shift_a;    // shift for A->output requant
  logic signed [7:0]  cfg_zp_a;       // zero-point of A input

  // Quantization params for input B
  logic signed [31:0] cfg_scale_b;    // m_int for B->output requant
  logic [5:0]         cfg_shift_b;    // shift for B->output requant
  logic signed [7:0]  cfg_zp_b;       // zero-point of B input

  // Output quantization
  logic signed [7:0]  cfg_zp_out;     // output zero-point

  // Input A SRAM interface
  logic [23:0]       ifm_a_rd_addr;
  logic              ifm_a_rd_en;
  logic signed [7:0] ifm_a_rd_data [TB_LANES];

  // Input B SRAM interface
  logic [23:0]       ifm_b_rd_addr;
  logic              ifm_b_rd_en;
  logic signed [7:0] ifm_b_rd_data [TB_LANES];

  // Output SRAM interface
  logic [23:0]       ofm_wr_addr;
  logic              ofm_wr_en;
  logic signed [7:0] ofm_wr_data [TB_LANES];

  // ═══════════════════════════════════════════════════════════════════
  // BEHAVIORAL SRAM MODELS
  // ═══════════════════════════════════════════════════════════════════
  logic signed [7:0] ifm_a_sram [IFM_A_SRAM_DEPTH][TB_LANES];
  logic signed [7:0] ifm_b_sram [IFM_B_SRAM_DEPTH][TB_LANES];
  logic signed [7:0] ofm_sram   [OFM_SRAM_DEPTH][TB_LANES];

  // Input A SRAM: 1-cycle read latency
  always_ff @(posedge clk) begin
    if (ifm_a_rd_en) begin
      for (int l = 0; l < TB_LANES; l++)
        ifm_a_rd_data[l] <= ifm_a_sram[ifm_a_rd_addr][l];
    end
  end

  // Input B SRAM: 1-cycle read latency
  always_ff @(posedge clk) begin
    if (ifm_b_rd_en) begin
      for (int l = 0; l < TB_LANES; l++)
        ifm_b_rd_data[l] <= ifm_b_sram[ifm_b_rd_addr][l];
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
  ewise_add_engine #(
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
    .cfg_scale_a    (cfg_scale_a),
    .cfg_shift_a    (cfg_shift_a),
    .cfg_zp_a       (cfg_zp_a),
    .cfg_scale_b    (cfg_scale_b),
    .cfg_shift_b    (cfg_shift_b),
    .cfg_zp_b       (cfg_zp_b),
    .cfg_zp_out     (cfg_zp_out),
    // Input A SRAM
    .ifm_a_rd_addr  (ifm_a_rd_addr),
    .ifm_a_rd_en    (ifm_a_rd_en),
    .ifm_a_rd_data  (ifm_a_rd_data),
    // Input B SRAM
    .ifm_b_rd_addr  (ifm_b_rd_addr),
    .ifm_b_rd_en    (ifm_b_rd_en),
    .ifm_b_rd_data  (ifm_b_rd_data),
    // Output SRAM
    .ofm_wr_addr    (ofm_wr_addr),
    .ofm_wr_en      (ofm_wr_en),
    .ofm_wr_data    (ofm_wr_data)
  );

  // ═══════════════════════════════════════════════════════════════════
  // GOLDEN REFERENCE DATA
  // ═══════════════════════════════════════════════════════════════════
  localparam int GOLD_MAX_H  = 16;
  localparam int GOLD_MAX_W  = 64;
  localparam int GOLD_MAX_C  = 256;

  // Source data arrays
  logic signed [7:0] gold_a [GOLD_MAX_H][GOLD_MAX_W][GOLD_MAX_C];
  logic signed [7:0] gold_b [GOLD_MAX_H][GOLD_MAX_W][GOLD_MAX_C];

  // Golden output: dynamic array [h * w * channels]
  logic signed [7:0] gold_output [];

  // Per-test parameters
  int t_h, t_w, t_ch;
  int t_scale_a, t_shift_a, t_zp_a_int;
  int t_scale_b, t_shift_b, t_zp_b_int;
  int t_zp_out_int;

  // ═══════════════════════════════════════════════════════════════════
  // GOLDEN: Behavioral element-wise add with domain alignment
  // For each (h, w, c):
  //   val_a = requant(q_a - zp_a, scale_a, shift_a)
  //   val_b = requant(q_b - zp_b, scale_b, shift_b)
  //   q_out = clamp(val_a + val_b + zp_out, -128, 127)
  // ═══════════════════════════════════════════════════════════════════
  task automatic run_golden();
    gold_output = new[t_h * t_w * t_ch];

    for (int h = 0; h < t_h; h++) begin
      for (int w = 0; w < t_w; w++) begin
        for (int c = 0; c < t_ch; c++) begin
          automatic longint dq_a, dq_b;
          automatic longint mult_a, mult_b;
          automatic longint round_a, round_b;
          automatic int     shift_a, shift_b;
          automatic int     sum_val;
          automatic int     final_val;
          automatic int     flat_idx;

          // Dequant A: subtract zero-point
          dq_a = longint'(gold_a[h][w][c]) - longint'(t_zp_a_int);

          // Requant A to output domain
          mult_a = dq_a * longint'(t_scale_a);
          if (t_shift_a > 0)
            round_a = mult_a + (longint'(1) <<< (t_shift_a - 1));
          else
            round_a = mult_a;
          shift_a = int'(round_a >>> t_shift_a);

          // Dequant B: subtract zero-point
          dq_b = longint'(gold_b[h][w][c]) - longint'(t_zp_b_int);

          // Requant B to output domain
          mult_b = dq_b * longint'(t_scale_b);
          if (t_shift_b > 0)
            round_b = mult_b + (longint'(1) <<< (t_shift_b - 1));
          else
            round_b = mult_b;
          shift_b = int'(round_b >>> t_shift_b);

          // Add and add output zero-point
          sum_val = shift_a + shift_b + t_zp_out_int;

          // Clamp to INT8
          if (sum_val > 127)
            final_val = 127;
          else if (sum_val < -128)
            final_val = -128;
          else
            final_val = sum_val;

          flat_idx = h * t_w * t_ch + w * t_ch + c;
          gold_output[flat_idx] = final_val[7:0];
        end
      end
    end
  endtask

  // ═══════════════════════════════════════════════════════════════════
  // COMPARISON: RTL output SRAM vs golden
  // Output SRAM layout: [H][C][num_wblk]
  // Address = h * channels * num_wblk + ch * num_wblk + wblk
  // ═══════════════════════════════════════════════════════════════════
  function automatic int compare_outputs(
    input int h, input int w, input int channels
  );
    automatic int mismatches = 0;
    automatic int printed    = 0;
    automatic int num_wblk   = (w + TB_LANES - 1) / TB_LANES;

    for (int ho = 0; ho < h; ho++) begin
      for (int co = 0; co < channels; co++) begin
        for (int wblk = 0; wblk < num_wblk; wblk++) begin
          automatic int ofm_addr;
          ofm_addr = ho * channels * num_wblk + co * num_wblk + wblk;
          for (int l = 0; l < TB_LANES; l++) begin
            automatic int wo = wblk * TB_LANES + l;
            if (wo < w) begin
              automatic int flat_idx;
              automatic logic signed [7:0] rtl_val, gold_val;
              flat_idx = ho * w * channels + wo * channels + co;
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
    for (int i = 0; i < IFM_A_SRAM_DEPTH; i++)
      for (int l = 0; l < TB_LANES; l++)
        ifm_a_sram[i][l] = 8'sd0;
    for (int i = 0; i < IFM_B_SRAM_DEPTH; i++)
      for (int l = 0; l < TB_LANES; l++)
        ifm_b_sram[i][l] = 8'sd0;
    for (int i = 0; i < OFM_SRAM_DEPTH; i++)
      for (int l = 0; l < TB_LANES; l++)
        ofm_sram[i][l] = 8'sd0;
  endtask

  task automatic clear_gold_arrays();
    for (int h = 0; h < GOLD_MAX_H; h++)
      for (int w = 0; w < GOLD_MAX_W; w++)
        for (int c = 0; c < GOLD_MAX_C; c++) begin
          gold_a[h][w][c] = 8'sd0;
          gold_b[h][w][c] = 8'sd0;
        end
  endtask

  // Fill input A SRAM from gold_a
  task automatic fill_ifm_a_sram(int h, int w, int channels);
    automatic int num_wblk = (w + TB_LANES - 1) / TB_LANES;
    for (int ho = 0; ho < h; ho++) begin
      for (int c = 0; c < channels; c++) begin
        for (int wblk = 0; wblk < num_wblk; wblk++) begin
          automatic int addr = ho * channels * num_wblk + c * num_wblk + wblk;
          for (int l = 0; l < TB_LANES; l++) begin
            automatic int w_idx = wblk * TB_LANES + l;
            if (w_idx < w)
              ifm_a_sram[addr][l] = gold_a[ho][w_idx][c];
            else
              ifm_a_sram[addr][l] = 8'sd0;
          end
        end
      end
    end
  endtask

  // Fill input B SRAM from gold_b
  task automatic fill_ifm_b_sram(int h, int w, int channels);
    automatic int num_wblk = (w + TB_LANES - 1) / TB_LANES;
    for (int ho = 0; ho < h; ho++) begin
      for (int c = 0; c < channels; c++) begin
        for (int wblk = 0; wblk < num_wblk; wblk++) begin
          automatic int addr = ho * channels * num_wblk + c * num_wblk + wblk;
          for (int l = 0; l < TB_LANES; l++) begin
            automatic int w_idx = wblk * TB_LANES + l;
            if (w_idx < w)
              ifm_b_sram[addr][l] = gold_b[ho][w_idx][c];
            else
              ifm_b_sram[addr][l] = 8'sd0;
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
  // TEST 1 - SAME SCALE (simple add)
  // A: values=[10,20,30], scale=0.1, zp=0
  // B: values=[5,10,15],  scale=0.1, zp=0
  // Output: scale=0.1, zp=0 -> expect [15,30,45]
  // Requant: identity (m_int=1, shift=0), so output = A + B directly
  // H=1, W=32, C=3
  // ═══════════════════════════════════════════════════════════════════
  task automatic test1_same_scale();
    automatic int mismatches;
    automatic int h = 1, w = 32, channels = 3;

    $display("════════════════════════════════════════════════════");
    $display("TEST 1 - Same scale: simple add, no requant");
    $display("  A=[10,20,30], B=[5,10,15] -> expect [15,30,45]");
    $display("  H=%0d, W=%0d, C=%0d", h, w, channels);
    $display("════════════════════════════════════════════════════");

    clear_srams();
    clear_gold_arrays();
    reset_dut();

    // Fill input A: channel 0=10, ch1=20, ch2=30 for all spatial positions
    for (int wo = 0; wo < w; wo++) begin
      gold_a[0][wo][0] = 8'sd10;
      gold_a[0][wo][1] = 8'sd20;
      gold_a[0][wo][2] = 8'sd30;
    end

    // Fill input B: channel 0=5, ch1=10, ch2=15
    for (int wo = 0; wo < w; wo++) begin
      gold_b[0][wo][0] = 8'sd5;
      gold_b[0][wo][1] = 8'sd10;
      gold_b[0][wo][2] = 8'sd15;
    end

    fill_ifm_a_sram(h, w, channels);
    fill_ifm_b_sram(h, w, channels);

    // Same-scale: identity requant (m_int=1, shift=0)
    cfg_h        = 10'(h);
    cfg_w        = 10'(w);
    cfg_channels = 9'(channels);
    cfg_scale_a  = 32'sd1;
    cfg_shift_a  = 6'd0;
    cfg_zp_a     = 8'sd0;
    cfg_scale_b  = 32'sd1;
    cfg_shift_b  = 6'd0;
    cfg_zp_b     = 8'sd0;
    cfg_zp_out   = 8'sd0;

    // Golden
    t_h          = h;
    t_w          = w;
    t_ch         = channels;
    t_scale_a    = 1;
    t_shift_a    = 0;
    t_zp_a_int   = 0;
    t_scale_b    = 1;
    t_shift_b    = 0;
    t_zp_b_int   = 0;
    t_zp_out_int = 0;
    run_golden();

    // Verify golden output manually for first spatial position
    $display("  Golden check: ch0=%0d, ch1=%0d, ch2=%0d (expect 15, 30, 45)",
             int'(gold_output[0]), int'(gold_output[1]), int'(gold_output[2]));

    run_dut();

    mismatches = compare_outputs(h, w, channels);

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
  // TEST 2 - DIFFERENT SCALES
  // A: scale=0.2, zp=5, random values
  // B: scale=0.1, zp=-3, random values
  // Output: scale=0.15, zp=0
  //
  // Requant A->Out: (q_a - 5) * (0.2/0.15) = (q_a - 5) * 1.333
  //   m_int_a = round(1.333 * 2^10) = 1365, shift=10
  // Requant B->Out: (q_b + 3) * (0.1/0.15) = (q_b + 3) * 0.6667
  //   m_int_b = round(0.6667 * 2^10) = 683, shift=10
  //
  // H=2, W=32, C=4
  // ═══════════════════════════════════════════════════════════════════
  task automatic test2_diff_scales();
    automatic int mismatches;
    automatic int h = 2, w = 32, channels = 4;

    $display("════════════════════════════════════════════════════");
    $display("TEST 2 - Different scales: domain alignment");
    $display("  A: scale=0.2,zp=5  B: scale=0.1,zp=-3  Out: scale=0.15,zp=0");
    $display("  H=%0d, W=%0d, C=%0d", h, w, channels);
    $display("════════════════════════════════════════════════════");

    clear_srams();
    clear_gold_arrays();
    reset_dut();

    // Fill inputs with random moderate values
    for (int ho = 0; ho < h; ho++)
      for (int wo = 0; wo < w; wo++)
        for (int c = 0; c < channels; c++) begin
          gold_a[ho][wo][c] = rand_range(-40, 40);
          gold_b[ho][wo][c] = rand_range(-40, 40);
        end

    fill_ifm_a_sram(h, w, channels);
    fill_ifm_b_sram(h, w, channels);

    cfg_h        = 10'(h);
    cfg_w        = 10'(w);
    cfg_channels = 9'(channels);
    cfg_scale_a  = 32'sd1365;
    cfg_shift_a  = 6'd10;
    cfg_zp_a     = 8'sd5;
    cfg_scale_b  = 32'sd683;
    cfg_shift_b  = 6'd10;
    cfg_zp_b     = -8'sd3;
    cfg_zp_out   = 8'sd0;

    // Golden
    t_h          = h;
    t_w          = w;
    t_ch         = channels;
    t_scale_a    = 1365;
    t_shift_a    = 10;
    t_zp_a_int   = 5;
    t_scale_b    = 683;
    t_shift_b    = 10;
    t_zp_b_int   = -3;
    t_zp_out_int = 0;
    run_golden();

    run_dut();

    mismatches = compare_outputs(h, w, channels);

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
  // TEST 3 - SATURATION TEST
  // A: values near 127, B: values near 127
  // With identity requant, sum = ~254, expected to saturate at 127
  // H=1, W=32, C=2
  // ═══════════════════════════════════════════════════════════════════
  task automatic test3_saturation();
    automatic int mismatches;
    automatic int neg_mismatches;
    automatic int h = 1, w = 32, channels = 2;

    $display("════════════════════════════════════════════════════");
    $display("TEST 3 - Saturation: values near 127 -> clamp at INT8 max");
    $display("  H=%0d, W=%0d, C=%0d", h, w, channels);
    $display("════════════════════════════════════════════════════");

    clear_srams();
    clear_gold_arrays();
    reset_dut();

    // Fill A: all 120 (near max positive)
    for (int wo = 0; wo < w; wo++)
      for (int c = 0; c < channels; c++)
        gold_a[0][wo][c] = 8'sd120;

    // Fill B: all 100 (near max positive)
    for (int wo = 0; wo < w; wo++)
      for (int c = 0; c < channels; c++)
        gold_b[0][wo][c] = 8'sd100;

    fill_ifm_a_sram(h, w, channels);
    fill_ifm_b_sram(h, w, channels);

    // Identity requant: 120 + 100 = 220, clamped to 127
    cfg_h        = 10'(h);
    cfg_w        = 10'(w);
    cfg_channels = 9'(channels);
    cfg_scale_a  = 32'sd1;
    cfg_shift_a  = 6'd0;
    cfg_zp_a     = 8'sd0;
    cfg_scale_b  = 32'sd1;
    cfg_shift_b  = 6'd0;
    cfg_zp_b     = 8'sd0;
    cfg_zp_out   = 8'sd0;

    // Golden
    t_h          = h;
    t_w          = w;
    t_ch         = channels;
    t_scale_a    = 1;
    t_shift_a    = 0;
    t_zp_a_int   = 0;
    t_scale_b    = 1;
    t_shift_b    = 0;
    t_zp_b_int   = 0;
    t_zp_out_int = 0;
    run_golden();

    // Verify golden: all outputs should be 127
    $display("  Golden check: first output=%0d (expect 127)", int'(gold_output[0]));

    run_dut();

    mismatches = compare_outputs(h, w, channels);

    // Also verify that negative saturation works
    // Test with A=-120, B=-120 -> sum=-240, clamped to -128
    clear_srams();
    clear_gold_arrays();

    for (int wo = 0; wo < w; wo++)
      for (int c = 0; c < channels; c++) begin
        gold_a[0][wo][c] = -8'sd120;
        gold_b[0][wo][c] = -8'sd120;
      end

    fill_ifm_a_sram(h, w, channels);
    fill_ifm_b_sram(h, w, channels);

    // Reuse same config
    run_golden();

    $display("  Golden check (neg): first output=%0d (expect -128)", int'(gold_output[0]));

    reset_dut();
    run_dut();

    neg_mismatches = compare_outputs(h, w, channels);
    mismatches += neg_mismatches;

    total_tests++;
    if (mismatches == 0) begin
      $display("  TEST 3: PASS (positive and negative saturation)");
      total_passed++;
    end else begin
      $display("  TEST 3: FAIL (%0d mismatches)", mismatches);
    end
    $display("");
  endtask

  // ═══════════════════════════════════════════════════════════════════
  // TEST 4 - QC2fCIB RESIDUAL STYLE (L22 bottleneck)
  // Realistic scales from L22 bottleneck residual connection
  //   Residual input:  scale=0.0235, zp=3
  //   Bottleneck output: scale=0.0189, zp=-2
  //   Output:          scale=0.0210, zp=0
  //
  // m_int_a = round(0.0235/0.0210 * 2^12) = round(1.119 * 4096) = 4584
  // m_int_b = round(0.0189/0.0210 * 2^12) = round(0.900 * 4096) = 3686
  // shift = 12
  //
  // H=4, W=32, C=64
  // ═══════════════════════════════════════════════════════════════════
  task automatic test4_qc2fcib_residual();
    automatic int mismatches;
    automatic int h = 4, w = 32, channels = 64;

    $display("════════════════════════════════════════════════════");
    $display("TEST 4 - QC2fCIB residual style (L22 bottleneck)");
    $display("  Residual: scale=0.0235,zp=3  Bottleneck: scale=0.0189,zp=-2");
    $display("  Output: scale=0.0210,zp=0");
    $display("  H=%0d, W=%0d, C=%0d", h, w, channels);
    $display("════════════════════════════════════════════════════");

    clear_srams();
    clear_gold_arrays();
    reset_dut();

    // Fill inputs with typical activation ranges
    for (int ho = 0; ho < h; ho++)
      for (int wo = 0; wo < w; wo++)
        for (int c = 0; c < channels; c++) begin
          gold_a[ho][wo][c] = rand_range(-50, 50);
          gold_b[ho][wo][c] = rand_range(-30, 30);
        end

    fill_ifm_a_sram(h, w, channels);
    fill_ifm_b_sram(h, w, channels);

    cfg_h        = 10'(h);
    cfg_w        = 10'(w);
    cfg_channels = 9'(channels);
    cfg_scale_a  = 32'sd4584;    // 0.0235/0.0210 * 2^12
    cfg_shift_a  = 6'd12;
    cfg_zp_a     = 8'sd3;
    cfg_scale_b  = 32'sd3686;    // 0.0189/0.0210 * 2^12
    cfg_shift_b  = 6'd12;
    cfg_zp_b     = -8'sd2;
    cfg_zp_out   = 8'sd0;

    // Golden
    t_h          = h;
    t_w          = w;
    t_ch         = channels;
    t_scale_a    = 4584;
    t_shift_a    = 12;
    t_zp_a_int   = 3;
    t_scale_b    = 3686;
    t_shift_b    = 12;
    t_zp_b_int   = -2;
    t_zp_out_int = 0;
    run_golden();

    run_dut();

    mismatches = compare_outputs(h, w, channels);

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
    $display("==========================================================");
    $display("  Golden Verification TB for ewise_add_engine (P7 EWISE_ADD)");
    $display("==========================================================");
    $display("");

    // Initialize signals
    clk     = 1'b0;
    rst_n   = 1'b0;
    start   = 1'b0;
    cfg_h        = '0;
    cfg_w        = '0;
    cfg_channels = '0;
    cfg_scale_a  = 32'sd1;
    cfg_shift_a  = 6'd0;
    cfg_zp_a     = 8'sd0;
    cfg_scale_b  = 32'sd1;
    cfg_shift_b  = 6'd0;
    cfg_zp_b     = 8'sd0;
    cfg_zp_out   = 8'sd0;

    // Seed random
    $urandom(42);

    // Run all tests
    test1_same_scale();
    test2_diff_scales();
    test3_saturation();
    test4_qc2fcib_residual();

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
