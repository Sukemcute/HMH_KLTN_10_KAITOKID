`timescale 1ns/1ps
// ============================================================================
// Golden Verification Testbench for concat_engine (P5 CONCAT)
// Channel concatenation with domain alignment (requantization).
//
// concat_engine reads two input feature maps A and B from SRAMs,
// requantizes each to a common output domain, and writes the
// concatenated result (C_a + C_b channels) to the output SRAM.
//
// Test scenarios:
//   TEST 1 - Same scale:      trivial concat, no requant needed
//   TEST 2 - Different scales: domain alignment with requantization
//   TEST 3 - L12-style:       real QConcat parameters, 384 output channels
// ============================================================================
module tb_concat_golden;

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
  localparam int TB_MAX_CH    = 384;  // Max total output channels

  // Maximum SRAM sizes (generous for all tests)
  localparam int IFM_A_SRAM_DEPTH = 65536;  // Input A SRAM (LANES-wide)
  localparam int IFM_B_SRAM_DEPTH = 65536;  // Input B SRAM (LANES-wide)
  localparam int OFM_SRAM_DEPTH   = 65536;  // Output SRAM (LANES-wide)

  // ═══════════════════════════════════════════════════════════════════
  // DUT SIGNALS
  // ═══════════════════════════════════════════════════════════════════
  logic        start, done, busy;

  // Configuration
  logic [9:0]  cfg_h;             // Spatial height (same for A, B, output)
  logic [9:0]  cfg_w;             // Spatial width
  logic [8:0]  cfg_ch_a;          // Channels in A
  logic [8:0]  cfg_ch_b;          // Channels in B

  // Quantization params for input A domain
  logic signed [31:0] cfg_scale_a;    // m_int for A->output requant
  logic [5:0]         cfg_shift_a;    // shift for A->output requant
  logic signed [7:0]  cfg_zp_a;       // zero-point of A input

  // Quantization params for input B domain
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
  // The concat_engine is expected to have the following interface.
  // Since the engine does not exist yet, this defines the target API.
  // ═══════════════════════════════════════════════════════════════════
  concat_engine #(
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
    .cfg_ch_a       (cfg_ch_a),
    .cfg_ch_b       (cfg_ch_b),
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
  localparam int GOLD_MAX_H  = 42;   // Max output height across tests
  localparam int GOLD_MAX_W  = 64;   // Max output width across tests
  localparam int GOLD_MAX_CA = 256;  // Max channels A
  localparam int GOLD_MAX_CB = 256;  // Max channels B

  // Source data arrays (unpadded, HWC format)
  logic signed [7:0] gold_a [GOLD_MAX_H][GOLD_MAX_W][GOLD_MAX_CA];
  logic signed [7:0] gold_b [GOLD_MAX_H][GOLD_MAX_W][GOLD_MAX_CB];

  // Golden output: dynamic array [h * w * (ch_a + ch_b)]
  logic signed [7:0] gold_output [];

  // Per-test parameters
  int t_h, t_w, t_ch_a, t_ch_b;
  int t_scale_a, t_shift_a, t_zp_a_int;
  int t_scale_b, t_shift_b, t_zp_b_int;
  int t_zp_out_int;

  // ═══════════════════════════════════════════════════════════════════
  // GOLDEN: Behavioral concat with domain alignment
  // For each spatial position (h, w):
  //   output[h][w][0..ch_a-1] = requant(A[h][w][c] - zp_a, scale_a, shift_a) + zp_out
  //   output[h][w][ch_a..ch_a+ch_b-1] = requant(B[h][w][c] - zp_b, scale_b, shift_b) + zp_out
  // ═══════════════════════════════════════════════════════════════════
  task automatic run_golden();
    automatic int total_ch = t_ch_a + t_ch_b;
    gold_output = new[t_h * t_w * total_ch];

    for (int h = 0; h < t_h; h++) begin
      for (int w = 0; w < t_w; w++) begin
        // Process input A channels
        for (int c = 0; c < t_ch_a; c++) begin
          automatic longint dequant_val;
          automatic longint mult64;
          automatic longint rounded;
          automatic int     shifted;
          automatic int     final_val;
          automatic int     flat_idx;

          // Dequant: subtract input zp
          dequant_val = longint'(gold_a[h][w][c]) - longint'(t_zp_a_int);

          // Requant to output domain: (dequant_val * scale_a + round) >>> shift_a
          mult64 = dequant_val * longint'(t_scale_a);
          if (t_shift_a > 0)
            rounded = mult64 + (longint'(1) <<< (t_shift_a - 1));
          else
            rounded = mult64;
          shifted = int'(rounded >>> t_shift_a);

          // Add output zero-point and clamp to INT8
          final_val = shifted + t_zp_out_int;
          if (final_val > 127)       final_val = 127;
          else if (final_val < -128) final_val = -128;

          flat_idx = h * t_w * total_ch + w * total_ch + c;
          gold_output[flat_idx] = final_val[7:0];
        end

        // Process input B channels
        for (int c = 0; c < t_ch_b; c++) begin
          automatic longint dequant_val;
          automatic longint mult64;
          automatic longint rounded;
          automatic int     shifted;
          automatic int     final_val;
          automatic int     flat_idx;

          dequant_val = longint'(gold_b[h][w][c]) - longint'(t_zp_b_int);

          mult64 = dequant_val * longint'(t_scale_b);
          if (t_shift_b > 0)
            rounded = mult64 + (longint'(1) <<< (t_shift_b - 1));
          else
            rounded = mult64;
          shifted = int'(rounded >>> t_shift_b);

          final_val = shifted + t_zp_out_int;
          if (final_val > 127)       final_val = 127;
          else if (final_val < -128) final_val = -128;

          flat_idx = h * t_w * total_ch + w * total_ch + t_ch_a + c;
          gold_output[flat_idx] = final_val[7:0];
        end
      end
    end
  endtask

  // ═══════════════════════════════════════════════════════════════════
  // COMPARISON: RTL output SRAM vs golden
  // Output SRAM layout: [H][C_total][num_wblk], where C_total = ch_a + ch_b
  // Each word is LANES wide.
  // Address = h * C_total * num_wblk + ch * num_wblk + wblk
  // ═══════════════════════════════════════════════════════════════════
  function automatic int compare_outputs(
    input int h, input int w, input int ch_total
  );
    automatic int mismatches = 0;
    automatic int printed    = 0;
    automatic int num_wblk   = (w + TB_LANES - 1) / TB_LANES;

    for (int ho = 0; ho < h; ho++) begin
      for (int co = 0; co < ch_total; co++) begin
        for (int wblk = 0; wblk < num_wblk; wblk++) begin
          automatic int ofm_addr;
          ofm_addr = ho * ch_total * num_wblk + co * num_wblk + wblk;
          for (int l = 0; l < TB_LANES; l++) begin
            automatic int wo = wblk * TB_LANES + l;
            if (wo < w) begin
              automatic int flat_idx;
              automatic logic signed [7:0] rtl_val, gold_val;
              flat_idx = ho * w * ch_total + wo * ch_total + co;
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
      for (int w = 0; w < GOLD_MAX_W; w++) begin
        for (int c = 0; c < GOLD_MAX_CA; c++)
          gold_a[h][w][c] = 8'sd0;
        for (int c = 0; c < GOLD_MAX_CB; c++)
          gold_b[h][w][c] = 8'sd0;
      end
  endtask

  // Fill input A SRAM from gold_a
  // Layout: addr = h * ch_a * num_wblk + ch * num_wblk + wblk
  // ifm_a_sram[addr][l] = gold_a[h][wblk*LANES + l][ch]
  task automatic fill_ifm_a_sram(int h, int w, int ch_a);
    automatic int num_wblk = (w + TB_LANES - 1) / TB_LANES;
    for (int ho = 0; ho < h; ho++) begin
      for (int c = 0; c < ch_a; c++) begin
        for (int wblk = 0; wblk < num_wblk; wblk++) begin
          automatic int addr = ho * ch_a * num_wblk + c * num_wblk + wblk;
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
  task automatic fill_ifm_b_sram(int h, int w, int ch_b);
    automatic int num_wblk = (w + TB_LANES - 1) / TB_LANES;
    for (int ho = 0; ho < h; ho++) begin
      for (int c = 0; c < ch_b; c++) begin
        for (int wblk = 0; wblk < num_wblk; wblk++) begin
          automatic int addr = ho * ch_b * num_wblk + c * num_wblk + wblk;
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
  // TEST 1 - SAME SCALE (trivial concat, no requant needed)
  // A: 4 channels, scale=0.1, zp=0, values=[10,20,30,40] per spatial pos
  // B: 2 channels, scale=0.1, zp=0, values=[50,60]
  // Output: 6 channels, [10,20,30,40,50,60], scale=0.1, zp=0
  // Since scales match, requant is identity: m_int=1, shift=0
  // Expected: 100% exact (no requant needed)
  // ═══════════════════════════════════════════════════════════════════
  task automatic test1_same_scale();
    automatic int mismatches;
    automatic int h = 2, w = 32, ch_a = 4, ch_b = 2;
    automatic int ch_total = ch_a + ch_b;  // 6

    $display("════════════════════════════════════════════════════");
    $display("TEST 1 - Same scale: trivial concat, no requant");
    $display("  A: 4 ch, B: 2 ch -> Output: 6 ch, H=%0d, W=%0d", h, w);
    $display("════════════════════════════════════════════════════");

    clear_srams();
    clear_gold_arrays();
    reset_dut();

    // Fill input A: channel c gets value (c+1)*10
    for (int ho = 0; ho < h; ho++)
      for (int wo = 0; wo < w; wo++) begin
        gold_a[ho][wo][0] = 8'sd10;
        gold_a[ho][wo][1] = 8'sd20;
        gold_a[ho][wo][2] = 8'sd30;
        gold_a[ho][wo][3] = 8'sd40;
      end

    // Fill input B: channel c gets value (c+5)*10
    for (int ho = 0; ho < h; ho++)
      for (int wo = 0; wo < w; wo++) begin
        gold_b[ho][wo][0] = 8'sd50;
        gold_b[ho][wo][1] = 8'sd60;
      end

    fill_ifm_a_sram(h, w, ch_a);
    fill_ifm_b_sram(h, w, ch_b);

    // Same-scale requant: identity transform (m_int=1, shift=0, zp=0)
    cfg_h       = 10'(h);
    cfg_w       = 10'(w);
    cfg_ch_a    = 9'(ch_a);
    cfg_ch_b    = 9'(ch_b);
    cfg_scale_a = 32'sd1;
    cfg_shift_a = 6'd0;
    cfg_zp_a    = 8'sd0;
    cfg_scale_b = 32'sd1;
    cfg_shift_b = 6'd0;
    cfg_zp_b    = 8'sd0;
    cfg_zp_out  = 8'sd0;

    // Golden
    t_h          = h;
    t_w          = w;
    t_ch_a       = ch_a;
    t_ch_b       = ch_b;
    t_scale_a    = 1;
    t_shift_a    = 0;
    t_zp_a_int   = 0;
    t_scale_b    = 1;
    t_shift_b    = 0;
    t_zp_b_int   = 0;
    t_zp_out_int = 0;
    run_golden();

    run_dut();

    mismatches = compare_outputs(h, w, ch_total);

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
  // TEST 2 - DIFFERENT SCALES (domain alignment)
  // A: 2 channels, scale_A=0.2, zp_A=5
  // B: 2 channels, scale_B=0.1, zp_B=-3
  // Output: scale_out=0.15, zp_out=0
  //
  // Requant A->Out: real_val = (q_a - 5) * 0.2
  //                 q_out = real_val / 0.15 + 0 = (q_a - 5) * (0.2/0.15)
  //                       = (q_a - 5) * 1.333...
  //   Approximate: m_int_a = round(1.333 * 2^10) = 1365, shift=10
  //
  // Requant B->Out: real_val = (q_b + 3) * 0.1
  //                 q_out = real_val / 0.15 + 0 = (q_b + 3) * (0.1/0.15)
  //                       = (q_b + 3) * 0.6667
  //   Approximate: m_int_b = round(0.6667 * 2^10) = 683, shift=10
  //
  // H=2, W=32 (1 wblk)
  // ═══════════════════════════════════════════════════════════════════
  task automatic test2_diff_scales();
    automatic int mismatches;
    automatic int h = 2, w = 32, ch_a = 2, ch_b = 2;
    automatic int ch_total = ch_a + ch_b;

    $display("════════════════════════════════════════════════════");
    $display("TEST 2 - Different scales: domain alignment");
    $display("  A: scale=0.2,zp=5  B: scale=0.1,zp=-3  Out: scale=0.15,zp=0");
    $display("  H=%0d, W=%0d, ch_a=%0d, ch_b=%0d", h, w, ch_a, ch_b);
    $display("════════════════════════════════════════════════════");

    clear_srams();
    clear_gold_arrays();
    reset_dut();

    // Fill input A: random values in moderate range
    for (int ho = 0; ho < h; ho++)
      for (int wo = 0; wo < w; wo++)
        for (int c = 0; c < ch_a; c++)
          gold_a[ho][wo][c] = rand_range(-50, 50);

    // Fill input B: random values in moderate range
    for (int ho = 0; ho < h; ho++)
      for (int wo = 0; wo < w; wo++)
        for (int c = 0; c < ch_b; c++)
          gold_b[ho][wo][c] = rand_range(-50, 50);

    fill_ifm_a_sram(h, w, ch_a);
    fill_ifm_b_sram(h, w, ch_b);

    // Requant params: A->Out
    // m_int_a = round(0.2/0.15 * 2^10) = round(1.3333 * 1024) = 1365
    // shift_a = 10
    cfg_h       = 10'(h);
    cfg_w       = 10'(w);
    cfg_ch_a    = 9'(ch_a);
    cfg_ch_b    = 9'(ch_b);
    cfg_scale_a = 32'sd1365;
    cfg_shift_a = 6'd10;
    cfg_zp_a    = 8'sd5;
    cfg_scale_b = 32'sd683;
    cfg_shift_b = 6'd10;
    cfg_zp_b    = -8'sd3;
    cfg_zp_out  = 8'sd0;

    // Golden
    t_h          = h;
    t_w          = w;
    t_ch_a       = ch_a;
    t_ch_b       = ch_b;
    t_scale_a    = 1365;
    t_shift_a    = 10;
    t_zp_a_int   = 5;
    t_scale_b    = 683;
    t_shift_b    = 10;
    t_zp_b_int   = -3;
    t_zp_out_int = 0;
    run_golden();

    run_dut();

    mismatches = compare_outputs(h, w, ch_total);

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
  // TEST 3 - L12-STYLE (real QConcat parameters)
  // A: 256 channels (from upsample L11), scale_a=0.0078, zp_a=2
  // B: 128 channels (from backbone L6),  scale_b=0.0153, zp_b=-1
  // Output: 384 channels, scale_out=0.0120, zp_out=0
  //
  // Requant A->Out: m_int = round(0.0078/0.0120 * 2^14) = round(0.65 * 16384)
  //               = 10650, shift=14
  // Requant B->Out: m_int = round(0.0153/0.0120 * 2^14) = round(1.275 * 16384)
  //               = 20890, shift=14
  //
  // Small spatial for feasibility: H=2, W=32 (1 wblk)
  // ═══════════════════════════════════════════════════════════════════
  task automatic test3_l12_style();
    automatic int mismatches;
    automatic int h = 2, w = 32, ch_a = 256, ch_b = 128;
    automatic int ch_total = ch_a + ch_b;  // 384

    $display("════════════════════════════════════════════════════");
    $display("TEST 3 - L12-style QConcat: 256+128=384 channels");
    $display("  Real quantization params, H=%0d, W=%0d", h, w);
    $display("════════════════════════════════════════════════════");

    clear_srams();
    clear_gold_arrays();
    reset_dut();

    // Fill input A: 256 channels, small random values (typical INT8 activations)
    for (int ho = 0; ho < h; ho++)
      for (int wo = 0; wo < w; wo++)
        for (int c = 0; c < ch_a; c++)
          gold_a[ho][wo][c] = rand_range(-30, 30);

    // Fill input B: 128 channels
    for (int ho = 0; ho < h; ho++)
      for (int wo = 0; wo < w; wo++)
        for (int c = 0; c < ch_b; c++)
          gold_b[ho][wo][c] = rand_range(-30, 30);

    fill_ifm_a_sram(h, w, ch_a);
    fill_ifm_b_sram(h, w, ch_b);

    // L12-style requant params
    cfg_h       = 10'(h);
    cfg_w       = 10'(w);
    cfg_ch_a    = 9'(ch_a);
    cfg_ch_b    = 9'(ch_b);
    cfg_scale_a = 32'sd10650;   // 0.0078/0.0120 * 2^14
    cfg_shift_a = 6'd14;
    cfg_zp_a    = 8'sd2;
    cfg_scale_b = 32'sd20890;   // 0.0153/0.0120 * 2^14
    cfg_shift_b = 6'd14;
    cfg_zp_b    = -8'sd1;
    cfg_zp_out  = 8'sd0;

    // Golden
    t_h          = h;
    t_w          = w;
    t_ch_a       = ch_a;
    t_ch_b       = ch_b;
    t_scale_a    = 10650;
    t_shift_a    = 14;
    t_zp_a_int   = 2;
    t_scale_b    = 20890;
    t_shift_b    = 14;
    t_zp_b_int   = -1;
    t_zp_out_int = 0;
    run_golden();

    run_dut();

    mismatches = compare_outputs(h, w, ch_total);

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
    $display("======================================================");
    $display("  Golden Verification TB for concat_engine (P5 CONCAT)");
    $display("======================================================");
    $display("");

    // Initialize signals
    clk     = 1'b0;
    rst_n   = 1'b0;
    start   = 1'b0;
    cfg_h       = '0;
    cfg_w       = '0;
    cfg_ch_a    = '0;
    cfg_ch_b    = '0;
    cfg_scale_a = 32'sd1;
    cfg_shift_a = 6'd0;
    cfg_zp_a    = 8'sd0;
    cfg_scale_b = 32'sd1;
    cfg_shift_b = 6'd0;
    cfg_zp_b    = 8'sd0;
    cfg_zp_out  = 8'sd0;

    // Seed random
    $urandom(42);

    // Run all tests
    test1_same_scale();
    test2_diff_scales();
    test3_l12_style();

    // Final summary
    $display("======================================================");
    $display("  FINAL RESULTS: %0d / %0d tests PASSED", total_passed, total_tests);
    if (total_passed == total_tests)
      $display("  >>>  ALL TESTS PASSED  <<<");
    else
      $display("  >>>  SOME TESTS FAILED  <<<");
    $display("======================================================");
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
