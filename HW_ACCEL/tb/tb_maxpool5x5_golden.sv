`timescale 1ns/1ps
// ============================================================================
// Testbench: maxpool5x5_engine with golden-model comparison
//
// Tests:
//   TEST 1 — All-same:  input all 42 -> output all 42
//   TEST 2 — Gradient:  input[h][w] = h+w, verify max over 5x5 window
//   TEST 3 — SPPF triple: P1=MP(X), P2=MP(P1), P3=MP(P2)
//
// Golden model:
//   for (h, w, c):
//     output = max of 5x5 window centered at (h,w), padded with -128
//
// No weights, no PPU -- pure max-pooling.
// ============================================================================
module tb_maxpool5x5_golden;
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

  logic [9:0]  cfg_h, cfg_w;
  logic [8:0]  cfg_channels;

  logic [23:0]       ifm_rd_addr;
  logic              ifm_rd_en;
  logic signed [7:0] ifm_rd_data [TB_LANES];

  logic [23:0]       ofm_wr_addr;
  logic              ofm_wr_en;
  logic signed [7:0] ofm_wr_data [TB_LANES];

  // -------------------------------------------------------------------------
  // Memory models
  // -------------------------------------------------------------------------
  // IFM SRAM: padded layout [H_pad][C][num_wblk_pad], word = LANES bytes
  // Flat indexing: mem[ addr * LANES + lane ]
  localparam int MEM_DEPTH = 1 << 20;
  logic signed [7:0] ifm_mem [MEM_DEPTH];
  logic signed [7:0] ofm_mem [MEM_DEPTH];

  // IFM read model (1-cycle latency)
  always_ff @(posedge clk) begin
    if (ifm_rd_en) begin
      for (int l = 0; l < TB_LANES; l++) begin
        automatic int flat;
        flat = int'(ifm_rd_addr) * TB_LANES + l;
        if (flat < MEM_DEPTH)
          ifm_rd_data[l] <= ifm_mem[flat];
        else
          ifm_rd_data[l] <= -8'sd128;
      end
    end
  end

  // OFM write capture
  always_ff @(posedge clk) begin
    if (ofm_wr_en) begin
      for (int l = 0; l < TB_LANES; l++) begin
        automatic int flat;
        flat = int'(ofm_wr_addr) * TB_LANES + l;
        if (flat < MEM_DEPTH)
          ofm_mem[flat] <= ofm_wr_data[l];
      end
    end
  end

  // -------------------------------------------------------------------------
  // DUT instantiation
  // -------------------------------------------------------------------------
  maxpool5x5_engine #(
    .LANES     (TB_LANES),
    .MAX_W_PAD (TB_MAX_W)
  ) u_dut (
    .clk          (clk),
    .rst_n        (rst_n),
    .start        (start),
    .done         (done),
    .busy         (busy),
    .cfg_h        (cfg_h),
    .cfg_w        (cfg_w),
    .cfg_channels (cfg_channels),
    .ifm_rd_addr  (ifm_rd_addr),
    .ifm_rd_en    (ifm_rd_en),
    .ifm_rd_data  (ifm_rd_data),
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
  // Golden model: max over 5x5 window with pad=-128
  // Operates on a "padded" array where padding=2 on each side holds -128.
  // Input pixel at (h,w,c) in the original H x W x C array maps to
  // padded coordinate (h+2, w+2, c).
  //
  // For output (ho, wo, c):
  //   max = max over kh in [0,5), kw in [0,5):
  //           padded[ho + kh][wo + kw][c]
  //
  // Since stride=1, Hout=H, Wout=W.
  // -------------------------------------------------------------------------

  // Helper: read from the padded IFM mem (layout matches DUT's SRAM)
  function automatic logic signed [7:0] read_padded_ifm(
    input int hp, wp, c, C, nwblk_pad
  );
    automatic int ww   = wp / TB_LANES;
    automatic int wl   = wp % TB_LANES;
    automatic int flat = (hp * C * nwblk_pad + c * nwblk_pad + ww) * TB_LANES + wl;
    if (flat >= 0 && flat < MEM_DEPTH)
      return ifm_mem[flat];
    else
      return -8'sd128;
  endfunction

  function automatic logic signed [7:0] golden_maxpool5x5(
    input int ho, wo, c, H, W, C_tot, nwblk_pad
  );
    automatic logic signed [7:0] mx = -8'sd128;
    for (int kh = 0; kh < 5; kh++) begin
      for (int kw = 0; kw < 5; kw++) begin
        automatic int hp = ho + kh;
        automatic int wp = wo + kw;
        automatic logic signed [7:0] val;
        val = read_padded_ifm(hp, wp, c, C_tot, nwblk_pad);
        if (val > mx) mx = val;
      end
    end
    return mx;
  endfunction

  // -------------------------------------------------------------------------
  // Test infrastructure
  // -------------------------------------------------------------------------
  int total_tests, total_pass, total_fail;

  // Populate padded IFM memory for a given test.
  // Pad regions filled with -128.
  task automatic fill_ifm_padded(
    input int H, W, C,
    input int mode    // 0=constant(42), 1=gradient(h+w), 2=copy from ofm_mem
  );
    automatic int pad        = 2;
    automatic int H_pad      = H + 2 * pad;
    automatic int W_pad      = W + 2 * pad;
    automatic int nwblk_pad  = (W_pad + TB_LANES - 1) / TB_LANES;

    // Clear entire IFM
    for (int i = 0; i < MEM_DEPTH; i++) ifm_mem[i] = -8'sd128;

    // Fill valid region (pad..pad+H-1, pad..pad+W-1) for each channel
    for (int h = 0; h < H; h++) begin
      for (int c = 0; c < C; c++) begin
        for (int w = 0; w < W; w++) begin
          automatic int hp   = h + pad;
          automatic int wp   = w + pad;
          automatic int ww   = wp / TB_LANES;
          automatic int wl   = wp % TB_LANES;
          automatic int flat = (hp * C * nwblk_pad + c * nwblk_pad + ww) * TB_LANES + wl;
          automatic logic signed [7:0] val;

          case (mode)
            0: val = 8'sd42;
            1: begin
              // Gradient: (h+w) clamped to INT8
              automatic int gv = h + w;
              if (gv > 127) val = 8'sd127;
              else          val = gv[7:0];
            end
            2: begin
              // Copy from ofm_mem (for SPPF chaining)
              // OFM was stored in non-padded layout [H][C][nwblk_out]
              automatic int nwblk_out = (W + TB_LANES - 1) / TB_LANES;
              automatic int oww  = w / TB_LANES;
              automatic int owl  = w % TB_LANES;
              automatic int oflat = (h * C * nwblk_out + c * nwblk_out + oww) * TB_LANES + owl;
              if (oflat >= 0 && oflat < MEM_DEPTH)
                val = ofm_mem[oflat];
              else
                val = -8'sd128;
            end
            default: val = 8'sd0;
          endcase

          if (flat < MEM_DEPTH) ifm_mem[flat] = val;
        end
      end
    end
  endtask

  // Run one maxpool5x5 pass and check against golden
  task automatic run_maxpool_and_check(
    input string name,
    input int    H, W, C,
    output int   mismatches
  );
    automatic int nwblk_pad = (W + 4 + TB_LANES - 1) / TB_LANES;
    automatic int nwblk_out = (W + TB_LANES - 1) / TB_LANES;
    automatic int cyc = 0;

    mismatches = 0;

    // Clear OFM
    for (int i = 0; i < MEM_DEPTH; i++) ofm_mem[i] = -8'sd128;

    // Configure
    cfg_h        = H[9:0];
    cfg_w        = W[9:0];
    cfg_channels = C[8:0];

    // Launch
    @(posedge clk);
    start <= 1'b1;
    @(posedge clk);
    start <= 1'b0;

    // Wait for done
    while (!done && cyc < TIMEOUT) begin
      @(posedge clk);
      cyc++;
    end
    if (cyc >= TIMEOUT) begin
      $display("  ** TIMEOUT after %0d cycles for %s **", cyc, name);
      mismatches = -1;
      return;
    end
    $display("  %s completed in %0d cycles", name, cyc);

    // Golden check (Hout=H, Wout=W for stride=1, pad=2)
    for (int ho = 0; ho < H; ho++) begin
      for (int c = 0; c < C; c++) begin
        for (int wo = 0; wo < W; wo++) begin
          automatic logic signed [7:0] expected;
          automatic logic signed [7:0] actual;
          automatic int oww, owl, oflat;

          expected = golden_maxpool5x5(ho, wo, c, H, W, C, nwblk_pad);

          oww   = wo / TB_LANES;
          owl   = wo % TB_LANES;
          oflat = (ho * C * nwblk_out + c * nwblk_out + oww) * TB_LANES + owl;
          actual = ofm_mem[oflat];

          if (actual !== expected) begin
            if (mismatches < 20)
              $display("  MISMATCH %s [h=%0d c=%0d w=%0d]: DUT=%0d  golden=%0d",
                       name, ho, c, wo, actual, expected);
            mismatches++;
          end
        end
      end
    end
  endtask

  // -------------------------------------------------------------------------
  // Main stimulus
  // -------------------------------------------------------------------------
  initial begin
    $display("============================================================");
    $display(" tb_maxpool5x5_golden — MaxPool 5x5 engine verification");
    $display("============================================================");

    total_tests = 0;
    total_pass  = 0;
    total_fail  = 0;

    rst_n = 1'b0;
    start = 1'b0;
    cfg_h = '0;
    cfg_w = '0;
    cfg_channels = '0;

    repeat (10) @(posedge clk);
    rst_n = 1'b1;
    repeat (5) @(posedge clk);

    // ================================================================
    // TEST 1: All-same — input all 42, output should all be 42
    // ================================================================
    begin
      automatic int H = 8, W = 32, C = 2;
      automatic int mm;

      $display("──────────────────────────────────────────────────");
      $display("  TEST 1: All-same (value=42)  H=%0d W=%0d C=%0d", H, W, C);
      $display("──────────────────────────────────────────────────");

      fill_ifm_padded(H, W, C, 0);   // mode=0 -> constant 42
      run_maxpool_and_check("All-same", H, W, C, mm);

      total_tests++;
      if (mm == 0) begin
        $display("  PASS -- all %0d outputs = 42 as expected", H * W * C);
        total_pass++;
      end else begin
        $display("  FAIL -- %0d mismatches", mm);
        total_fail++;
      end
    end

    repeat (10) @(posedge clk);

    // ================================================================
    // TEST 2: Gradient — input[h][w] = h+w, verify max is correct
    // ================================================================
    begin
      automatic int H = 8, W = 32, C = 1;
      automatic int mm;

      $display("──────────────────────────────────────────────────");
      $display("  TEST 2: Gradient (h+w)  H=%0d W=%0d C=%0d", H, W, C);
      $display("──────────────────────────────────────────────────");

      fill_ifm_padded(H, W, C, 1);   // mode=1 -> gradient h+w
      run_maxpool_and_check("Gradient", H, W, C, mm);

      total_tests++;
      if (mm == 0) begin
        $display("  PASS -- all %0d outputs match golden max", H * W * C);
        total_pass++;
      end else begin
        $display("  FAIL -- %0d mismatches", mm);
        total_fail++;
      end
    end

    repeat (10) @(posedge clk);

    // ================================================================
    // TEST 3: SPPF triple — P1=MP(X), P2=MP(P1), P3=MP(P2)
    // Apply maxpool 3 times in series, checking each stage.
    // Uses gradient input.
    // ================================================================
    begin
      automatic int H = 8, W = 32, C = 2;
      automatic int mm1, mm2, mm3;
      automatic int all_pass = 1;

      $display("──────────────────────────────────────────────────");
      $display("  TEST 3: SPPF triple  H=%0d W=%0d C=%0d", H, W, C);
      $display("──────────────────────────────────────────────────");

      // Stage P1: MP(X)
      $display("  --- Stage P1: MP(X) ---");
      fill_ifm_padded(H, W, C, 1);   // gradient input
      run_maxpool_and_check("P1=MP(X)", H, W, C, mm1);
      if (mm1 != 0) all_pass = 0;

      repeat (10) @(posedge clk);

      // Stage P2: MP(P1) — copy P1 output to padded IFM
      $display("  --- Stage P2: MP(P1) ---");
      fill_ifm_padded(H, W, C, 2);   // mode=2 -> copy from ofm_mem
      run_maxpool_and_check("P2=MP(P1)", H, W, C, mm2);
      if (mm2 != 0) all_pass = 0;

      repeat (10) @(posedge clk);

      // Stage P3: MP(P2) — copy P2 output to padded IFM
      $display("  --- Stage P3: MP(P2) ---");
      fill_ifm_padded(H, W, C, 2);   // mode=2 -> copy from ofm_mem
      run_maxpool_and_check("P3=MP(P2)", H, W, C, mm3);
      if (mm3 != 0) all_pass = 0;

      total_tests++;
      if (all_pass) begin
        $display("  PASS -- SPPF triple all 3 stages match golden");
        total_pass++;
      end else begin
        $display("  FAIL -- SPPF triple: P1=%0d P2=%0d P3=%0d mismatches",
                 mm1, mm2, mm3);
        total_fail++;
      end
    end

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
