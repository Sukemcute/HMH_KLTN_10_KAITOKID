`timescale 1ns/1ps
// ============================================================================
// Testbench for Post-Processing Unit (PPU)
// 4-stage pipeline: Bias Add -> Requant -> Activation -> Clamp+Ewise
//
// Tests:
//   1. Half-up rounding verification (most critical)
//   2. Bias addition
//   3. ReLU activation
//   4. SiLU activation via LUT
//   5. ACT_NONE (identity / clamp)
//   6. Output zero-point with saturation
//   7. Element-wise add (skip connection)
//   8. Full pipeline golden comparison (Conv L0 style)
//   9. Stress test (1000 random vectors)
// ============================================================================
module tb_ppu;
  import accel_pkg::*;
  import desc_pkg::*;

  // --------------------------------------------------------------------------
  // Parameters
  // --------------------------------------------------------------------------
  localparam int LANES     = 32;
  localparam int CLK_HALF  = 5;   // 5 ns half-period => 10 ns period
  localparam int PIPE_LAT  = 4;   // 4-stage pipeline latency

  // --------------------------------------------------------------------------
  // DUT signals
  // --------------------------------------------------------------------------
  logic                  clk;
  logic                  rst_n;
  logic                  en;

  post_profile_t         cfg_post;
  pe_mode_e              cfg_mode;

  logic signed [31:0]    psum_in   [LANES];
  logic                  psum_valid;

  logic signed [31:0]    bias_val  [LANES];
  logic signed [31:0]    m_int     [LANES];
  logic        [5:0]     shift     [LANES];
  logic signed [7:0]     zp_out;

  logic signed [7:0]     silu_lut_data [256];

  logic signed [7:0]     ewise_in  [LANES];
  logic                  ewise_valid;

  logic signed [7:0]     act_out   [LANES];
  logic                  act_valid;

  // --------------------------------------------------------------------------
  // DUT instantiation
  // --------------------------------------------------------------------------
  ppu #(.LANES(LANES)) u_dut (
    .clk           (clk),
    .rst_n         (rst_n),
    .en            (en),
    .cfg_post      (cfg_post),
    .cfg_mode      (cfg_mode),
    .psum_in       (psum_in),
    .psum_valid    (psum_valid),
    .bias_val      (bias_val),
    .m_int         (m_int),
    .shift         (shift),
    .zp_out        (zp_out),
    .silu_lut_data (silu_lut_data),
    .ewise_in      (ewise_in),
    .ewise_valid   (ewise_valid),
    .act_out       (act_out),
    .act_valid     (act_valid)
  );

  // --------------------------------------------------------------------------
  // Clock generation: 10 ns period
  // --------------------------------------------------------------------------
  initial clk = 1'b0;
  always #(CLK_HALF) clk = ~clk;

  // --------------------------------------------------------------------------
  // Global counters
  // --------------------------------------------------------------------------
  int total_tests;
  int total_errors;
  int test_errors;  // per-test error count

  // --------------------------------------------------------------------------
  // Golden model helper functions (mirroring RTL exactly)
  // --------------------------------------------------------------------------

  // Clamp a signed value to INT16 range
  function automatic logic signed [15:0] golden_clamp16(input longint val);
    if (val > 32767)       return 16'sd32767;
    else if (val < -32768) return -16'sd32768;
    else                   return val[15:0];
  endfunction

  // Clamp a signed value to INT8 range
  function automatic logic signed [7:0] golden_clamp8(input int val);
    if (val > 127)        return 8'sd127;
    else if (val < -128)  return -8'sd128;
    else                  return val[7:0];
  endfunction

  // Full golden model for one lane (bias, requant, activation, ewise, zp, clamp)
  function automatic logic signed [7:0] golden_ppu_lane(
    input logic signed [31:0] psum,
    input logic signed [31:0] bias,
    input logic signed [31:0] m,
    input logic        [5:0]  sh,
    input logic signed [7:0]  zp,
    input act_mode_e          act,
    input logic               bias_en,
    input logic               ewise_en_flag,
    input logic               ewise_vld,
    input logic signed [7:0]  ewise_val
  );
    longint biased, mult, rounded;
    int shifted32;
    int clamped16;
    int y_act;
    int y_add;
    int sh_int;

    // Stage 1: Bias add
    if (bias_en)
      biased = longint'(psum) + longint'(bias);
    else
      biased = longint'(psum);

    // Stage 2: Requantize with half-up rounding
    mult = biased * longint'(m);
    sh_int = int'(sh);
    if (sh_int > 0)
      rounded = mult + (longint'(1) <<< (sh_int - 1));
    else
      rounded = mult;
    // RTL truncates to 32-bit: shifted = 32'(rounded >>> sh)
    shifted32 = int'(rounded >>> sh_int);

    // Clamp 32-bit to 16-bit (matches RTL exactly)
    if (shifted32 > 32767)
      clamped16 = 32767;
    else if (shifted32 < -32768)
      clamped16 = -32768;
    else
      clamped16 = shifted32;

    // Stage 3: Activation
    case (act)
      ACT_RELU: begin
        if (clamped16 > 0) begin
          if (clamped16 > 127) y_act = 127;
          else if (clamped16 < -128) y_act = -128;
          else y_act = clamped16;
        end else begin
          y_act = 0;
        end
      end
      ACT_SILU: begin
        // LUT lookup: index = clamp(val+128, 0, 255)
        automatic int idx;
        idx = clamped16 + 128;
        if (idx < 0) idx = 0;
        if (idx > 255) idx = 255;
        y_act = int'(silu_lut_data[idx]);
      end
      ACT_CLAMP: begin
        if (clamped16 > 127) y_act = 127;
        else if (clamped16 < -128) y_act = -128;
        else y_act = clamped16;
      end
      default: begin // ACT_NONE
        if (clamped16 > 127) y_act = 127;
        else if (clamped16 < -128) y_act = -128;
        else y_act = clamped16;
      end
    endcase

    // Stage 4: Ewise + zp + clamp
    if (ewise_en_flag && ewise_vld)
      y_add = y_act + int'(ewise_val);
    else
      y_add = y_act;

    y_add = y_add + int'(zp);

    if (y_add > 127) return 8'sd127;
    else if (y_add < -128) return -8'sd128;
    else return y_add[7:0];
  endfunction

  // --------------------------------------------------------------------------
  // Tasks
  // --------------------------------------------------------------------------

  // Reset DUT
  task automatic do_reset();
    rst_n      <= 1'b0;
    en         <= 1'b0;
    psum_valid <= 1'b0;
    ewise_valid <= 1'b0;
    zp_out     <= 8'sd0;
    cfg_mode   <= PE_RS3;
    cfg_post   <= '0;
    for (int l = 0; l < LANES; l++) begin
      psum_in[l]  <= 32'sd0;
      bias_val[l] <= 32'sd0;
      m_int[l]    <= 32'sd1;
      shift[l]    <= 6'd0;
      ewise_in[l] <= 8'sd0;
    end
    repeat (4) @(posedge clk);
    rst_n <= 1'b1;
    en    <= 1'b1;
    repeat (2) @(posedge clk);
  endtask

  // Drive one vector through pipeline and wait for output
  task automatic drive_and_wait();
    @(posedge clk);
    psum_valid  <= 1'b1;
    ewise_valid <= (cfg_post.ewise_en) ? 1'b1 : 1'b0;
    @(posedge clk);
    psum_valid  <= 1'b0;
    ewise_valid <= 1'b0;
    // Wait for pipeline latency (4 stages)
    repeat (PIPE_LAT) @(posedge clk);
    // act_valid should be asserted now -- sample on next posedge
  endtask

  // Compare DUT output vs golden for all lanes
  task automatic check_output(
    input string       test_name,
    input logic signed [7:0] golden [LANES]
  );
    for (int l = 0; l < LANES; l++) begin
      if (act_out[l] !== golden[l]) begin
        $display("  [FAIL] %s lane %0d: DUT=%0d, GOLDEN=%0d",
                 test_name, l, act_out[l], golden[l]);
        test_errors++;
      end
    end
  endtask

  // Set all lanes to same values (scalar broadcast)
  task automatic set_uniform(
    input logic signed [31:0] psum,
    input logic signed [31:0] bias,
    input logic signed [31:0] m,
    input logic        [5:0]  sh,
    input logic signed [7:0]  ew
  );
    for (int l = 0; l < LANES; l++) begin
      psum_in[l]  <= psum;
      bias_val[l] <= bias;
      m_int[l]    <= m;
      shift[l]    <= sh;
      ewise_in[l] <= ew;
    end
  endtask

  // Compute golden for all lanes (uses current signal state)
  function automatic void compute_golden_all(
    ref logic signed [7:0] golden [LANES]
  );
    for (int l = 0; l < LANES; l++) begin
      golden[l] = golden_ppu_lane(
        psum_in[l], bias_val[l], m_int[l], shift[l], zp_out,
        cfg_post.act_mode, cfg_post.bias_en,
        cfg_post.ewise_en, (cfg_post.ewise_en ? 1'b1 : 1'b0),
        ewise_in[l]
      );
    end
  endfunction

  // --------------------------------------------------------------------------
  // Main test sequence
  // --------------------------------------------------------------------------
  logic signed [7:0] golden [LANES];

  initial begin
    total_tests  = 0;
    total_errors = 0;

    $display("================================================================");
    $display(" PPU Testbench - Starting");
    $display("================================================================");

    do_reset();

    // ====================================================================
    // TEST 1: Half-Up Rounding Verification (MOST IMPORTANT)
    // ====================================================================
    begin
      automatic int errors_before;
      test_errors = 0;
      errors_before = total_errors;
      $display("\n--- TEST 1: Half-Up Rounding Verification ---");

      // Configure: bias_en=1, ACT_NONE (identity), no ewise, zp=0
      cfg_post.bias_en   <= 1'b0;   // no bias so psum goes straight to requant
      cfg_post.act_mode  <= ACT_CLAMP;
      cfg_post.ewise_en  <= 1'b0;
      cfg_post.quant_mode <= QMODE_PER_CHANNEL;
      zp_out <= 8'sd0;

      // --- Sub-test 1a: acc=100, M=3, shift=1 ---
      // mult = 300, rounded = 300+1 = 301, shifted = 301>>>1 = 150
      // (floor would give 300>>>1 = 150, same here; but half-up adds 1 before shift)
      // Actually: 301 >> 1 = 150 (in arithmetic shift, 301/2 = 150.5, floor = 150)
      // The key is the rounding bias is added.
      set_uniform(32'sd100, 32'sd0, 32'sd3, 6'd1, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T1a(acc=100,M=3,sh=1)", golden);
      $display("  T1a: acc=100, M=3, sh=1 -> golden[0]=%0d", golden[0]);

      // --- Sub-test 1b: acc=101, M=7, shift=2 ---
      // mult = 707, rounded = 707+2 = 709, shifted = 709>>>2 = 177
      set_uniform(32'sd101, 32'sd0, 32'sd7, 6'd2, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T1b(acc=101,M=7,sh=2)", golden);
      $display("  T1b: acc=101, M=7, sh=2 -> golden[0]=%0d", golden[0]);

      // --- Sub-test 1c: acc=-100, M=3, shift=1 ---
      // mult = -300, rounded = -300+1 = -299, shifted = -299>>>1 = -150
      // (arithmetic right shift of -299 by 1: -150 in two's complement)
      set_uniform(-32'sd100, 32'sd0, 32'sd3, 6'd1, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T1c(acc=-100,M=3,sh=1)", golden);
      $display("  T1c: acc=-100, M=3, sh=1 -> golden[0]=%0d", golden[0]);

      // --- Sub-test 1d: 32 carefully chosen values across lanes ---
      $display("  T1d: 32 carefully chosen rounding-sensitive values...");
      begin
        // Values chosen so that the pre-rounding result has a fractional part
        // near 0.5, making half-up rounding differ from truncation
        automatic int test_accs [32] = '{
           50,  -50,   99,  -99,  127, -127,  200, -200,
          255, -255,   33,  -33,   77,  -77,  150, -150,
            1,   -1,   63,  -63,  111, -111,   17,  -17,
          250, -250,   45,  -45,   88,  -88,  199, -199
        };
        automatic int test_ms [32] = '{
            3,    3,    5,    5,    7,    7,   11,   11,
           13,   13,   17,   17,   19,   19,   23,   23,
           29,   29,   31,   31,   37,   37,   41,   41,
           43,   43,   47,   47,   53,   53,   59,   59
        };
        automatic int test_shs [32] = '{
            1,    1,    2,    2,    3,    3,    4,    4,
            5,    5,    1,    1,    2,    2,    3,    3,
            4,    4,    5,    5,    1,    1,    2,    2,
            3,    3,    4,    4,    5,    5,    1,    1
        };

        for (int l = 0; l < LANES; l++) begin
          psum_in[l]  <= 32'(test_accs[l]);
          bias_val[l] <= 32'sd0;
          m_int[l]    <= 32'(test_ms[l]);
          shift[l]    <= 6'(test_shs[l]);
          ewise_in[l] <= 8'sd0;
        end
        compute_golden_all(golden);
        drive_and_wait();
        check_output("T1d(32 rounding values)", golden);

        // Print a few lanes for visibility
        for (int l = 0; l < 8; l++) begin
          $display("    lane %0d: acc=%0d, M=%0d, sh=%0d -> golden=%0d, dut=%0d %s",
            l, test_accs[l], test_ms[l], test_shs[l],
            golden[l], act_out[l],
            (act_out[l] === golden[l]) ? "OK" : "MISMATCH");
        end
      end

      // --- Sub-test 1e: shift=0 edge case (no rounding) ---
      set_uniform(32'sd10, 32'sd0, 32'sd5, 6'd0, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T1e(shift=0)", golden);
      $display("  T1e: shift=0 -> golden[0]=%0d (no rounding, just clamp)", golden[0]);

      // --- Sub-test 1f: Large shift ---
      set_uniform(32'sd10000, 32'sd0, 32'sd100, 6'd20, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T1f(large shift=20)", golden);
      $display("  T1f: large shift=20 -> golden[0]=%0d", golden[0]);

      total_tests++;
      total_errors += test_errors;
      $display("  TEST 1 %s (%0d errors)",
               (test_errors == 0) ? "PASSED" : "FAILED", test_errors);
    end

    // ====================================================================
    // TEST 2: Bias Addition
    // ====================================================================
    begin
      test_errors = 0;
      $display("\n--- TEST 2: Bias Addition ---");

      // bias_en = 1, ACT_CLAMP (identity), no ewise, zp=0, M=1, shift=0
      cfg_post.bias_en   <= 1'b1;
      cfg_post.act_mode  <= ACT_CLAMP;
      cfg_post.ewise_en  <= 1'b0;
      zp_out <= 8'sd0;

      // psum=1000, bias=500 -> biased=1500, requant(1500*1 +0)>>0=1500, clamp16=1500, clamp8=127
      set_uniform(32'sd1000, 32'sd500, 32'sd1, 6'd0, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T2a(psum=1000,bias=500)", golden);
      $display("  T2a: psum=1000, bias=500 -> golden=%0d (clamp to 127)", golden[0]);

      // psum=-1000, bias=500 -> biased=-500, clamp8=-128
      set_uniform(-32'sd1000, 32'sd500, 32'sd1, 6'd0, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T2b(psum=-1000,bias=500)", golden);
      $display("  T2b: psum=-1000, bias=500 -> golden=%0d", golden[0]);

      // psum=0, bias=0 -> biased=0
      set_uniform(32'sd0, 32'sd0, 32'sd1, 6'd0, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T2c(psum=0,bias=0)", golden);
      $display("  T2c: psum=0, bias=0 -> golden=%0d", golden[0]);

      // bias_en=0 -> biased = psum (ignore bias)
      cfg_post.bias_en <= 1'b0;
      set_uniform(32'sd50, 32'sd9999, 32'sd1, 6'd0, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T2d(bias_en=0)", golden);
      $display("  T2d: bias_en=0, psum=50, bias=9999 -> golden=%0d (bias ignored)", golden[0]);

      // Realistic: psum=50000, bias=1000, M=100, shift=15
      cfg_post.bias_en <= 1'b1;
      set_uniform(32'sd50000, 32'sd1000, 32'sd100, 6'd15, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T2e(realistic)", golden);
      $display("  T2e: psum=50000, bias=1000, M=100, sh=15 -> golden=%0d", golden[0]);

      total_tests++;
      total_errors += test_errors;
      $display("  TEST 2 %s (%0d errors)",
               (test_errors == 0) ? "PASSED" : "FAILED", test_errors);
    end

    // ====================================================================
    // TEST 3: ReLU Activation
    // ====================================================================
    begin
      test_errors = 0;
      $display("\n--- TEST 3: ReLU Activation ---");

      cfg_post.bias_en   <= 1'b0;
      cfg_post.act_mode  <= ACT_RELU;
      cfg_post.ewise_en  <= 1'b0;
      zp_out <= 8'sd0;

      // Positive requanted -> clamped INT8
      set_uniform(32'sd50, 32'sd0, 32'sd1, 6'd0, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T3a(positive)", golden);
      $display("  T3a: psum=50 (positive) -> golden=%0d", golden[0]);

      // Large positive -> clamp to 127
      set_uniform(32'sd200, 32'sd0, 32'sd1, 6'd0, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T3b(large positive)", golden);
      $display("  T3b: psum=200 (large positive) -> golden=%0d (clamped to 127)", golden[0]);

      // Negative -> output = 0
      set_uniform(-32'sd50, 32'sd0, 32'sd1, 6'd0, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T3c(negative)", golden);
      $display("  T3c: psum=-50 (negative) -> golden=%0d (ReLU zeros)", golden[0]);

      // Zero -> output = 0 (RTL checks y_raw > 0, so zero maps to 0)
      set_uniform(32'sd0, 32'sd0, 32'sd1, 6'd0, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T3d(zero)", golden);
      $display("  T3d: psum=0 (zero) -> golden=%0d (ReLU: 0 is NOT > 0)", golden[0]);

      // Large negative -> 0
      set_uniform(-32'sd10000, 32'sd0, 32'sd1, 6'd0, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T3e(large neg)", golden);
      $display("  T3e: psum=-10000 -> golden=%0d", golden[0]);

      // After requant: positive but small
      set_uniform(32'sd30000, 32'sd0, 32'sd50, 6'd15, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T3f(requant positive)", golden);
      $display("  T3f: psum=30000, M=50, sh=15 -> golden=%0d", golden[0]);

      total_tests++;
      total_errors += test_errors;
      $display("  TEST 3 %s (%0d errors)",
               (test_errors == 0) ? "PASSED" : "FAILED", test_errors);
    end

    // ====================================================================
    // TEST 4: SiLU Activation via LUT
    // ====================================================================
    begin
      test_errors = 0;
      $display("\n--- TEST 4: SiLU Activation via LUT ---");

      // Load SiLU LUT with known pattern: lut[i] = clamp(i - 128, -128, 127)
      // This makes it an identity-like mapping for values in [-128, 127]
      for (int i = 0; i < 256; i++) begin
        automatic int val;
        val = i - 128;
        if (val > 127) silu_lut_data[i] = 8'sd127;
        else if (val < -128) silu_lut_data[i] = -8'sd128;
        else silu_lut_data[i] = val[7:0];
      end

      cfg_post.bias_en   <= 1'b0;
      cfg_post.act_mode  <= ACT_SILU;
      cfg_post.ewise_en  <= 1'b0;
      zp_out <= 8'sd0;

      // val=50 -> idx = 50+128 = 178 -> lut[178] = 178-128 = 50
      set_uniform(32'sd50, 32'sd0, 32'sd1, 6'd0, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T4a(SiLU val=50)", golden);
      $display("  T4a: val=50, lut idx=178 -> golden=%0d", golden[0]);

      // val=-50 -> idx = -50+128 = 78 -> lut[78] = 78-128 = -50
      set_uniform(-32'sd50, 32'sd0, 32'sd1, 6'd0, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T4b(SiLU val=-50)", golden);
      $display("  T4b: val=-50, lut idx=78 -> golden=%0d", golden[0]);

      // val = -128 (boundary) -> idx=0 -> lut[0] = -128
      set_uniform(-32'sd128, 32'sd0, 32'sd1, 6'd0, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T4c(SiLU val=-128)", golden);
      $display("  T4c: val=-128, idx=0 -> golden=%0d", golden[0]);

      // val = 127 -> idx=255 -> lut[255] = 127
      set_uniform(32'sd127, 32'sd0, 32'sd1, 6'd0, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T4d(SiLU val=127)", golden);
      $display("  T4d: val=127, idx=255 -> golden=%0d", golden[0]);

      // val = -200 (saturated to 16-bit) -> idx = clamp(-200+128, 0, 255) = 0
      // But y_raw_s2 = -200 (within 16-bit), so idx = -200+128 = -72, clamped to 0
      set_uniform(-32'sd200, 32'sd0, 32'sd1, 6'd0, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T4e(SiLU val=-200)", golden);
      $display("  T4e: val=-200 (idx clamps to 0) -> golden=%0d", golden[0]);

      // Load a non-trivial SiLU approximation pattern for one more check
      // lut[i] = approximate SiLU(x) where x = i-128
      for (int i = 0; i < 256; i++) begin
        automatic int x;
        x = i - 128;
        // Simplified SiLU: negative -> compressed, positive -> identity-ish
        if (x < -64) silu_lut_data[i] = 8'sd0;
        else if (x < 0) silu_lut_data[i] = (x/2)[7:0];
        else silu_lut_data[i] = (x > 127) ? 8'sd127 : x[7:0];
      end

      // val = -32 -> idx = 96 -> x = -32, lut = -32/2 = -16
      set_uniform(-32'sd32, 32'sd0, 32'sd1, 6'd0, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T4f(SiLU custom lut)", golden);
      $display("  T4f: val=-32, custom SiLU lut -> golden=%0d", golden[0]);

      total_tests++;
      total_errors += test_errors;
      $display("  TEST 4 %s (%0d errors)",
               (test_errors == 0) ? "PASSED" : "FAILED", test_errors);
    end

    // ====================================================================
    // TEST 5: ACT_NONE (Identity / Clamp)
    // ====================================================================
    begin
      test_errors = 0;
      $display("\n--- TEST 5: ACT_NONE (Identity) ---");

      cfg_post.bias_en   <= 1'b0;
      cfg_post.act_mode  <= ACT_NONE;
      cfg_post.ewise_en  <= 1'b0;
      zp_out <= 8'sd0;

      // Positive passes through, clamped to INT8
      set_uniform(32'sd42, 32'sd0, 32'sd1, 6'd0, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T5a(pos identity)", golden);
      $display("  T5a: val=42 -> golden=%0d (pass-through)", golden[0]);

      // Negative passes through
      set_uniform(-32'sd42, 32'sd0, 32'sd1, 6'd0, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T5b(neg identity)", golden);
      $display("  T5b: val=-42 -> golden=%0d (pass-through, no ReLU)", golden[0]);

      // Large value clamped to 127
      set_uniform(32'sd500, 32'sd0, 32'sd1, 6'd0, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T5c(clamp high)", golden);
      $display("  T5c: val=500 -> golden=%0d (clamped to 127)", golden[0]);

      // Large negative clamped to -128
      set_uniform(-32'sd500, 32'sd0, 32'sd1, 6'd0, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T5d(clamp low)", golden);
      $display("  T5d: val=-500 -> golden=%0d (clamped to -128)", golden[0]);

      total_tests++;
      total_errors += test_errors;
      $display("  TEST 5 %s (%0d errors)",
               (test_errors == 0) ? "PASSED" : "FAILED", test_errors);
    end

    // ====================================================================
    // TEST 6: Output Zero-Point
    // ====================================================================
    begin
      test_errors = 0;
      $display("\n--- TEST 6: Output Zero-Point ---");

      cfg_post.bias_en   <= 1'b0;
      cfg_post.act_mode  <= ACT_CLAMP;  // identity pass-through
      cfg_post.ewise_en  <= 1'b0;

      // act_val=50, zp_out=10 -> clamp(60) = 60
      zp_out <= 8'sd10;
      set_uniform(32'sd50, 32'sd0, 32'sd1, 6'd0, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T6a(zp=10,val=50)", golden);
      $display("  T6a: act=50, zp=10 -> golden=%0d", golden[0]);

      // act_val=120, zp_out=10 -> clamp(130) = 127 (saturate!)
      zp_out <= 8'sd10;
      set_uniform(32'sd120, 32'sd0, 32'sd1, 6'd0, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T6b(zp=10,val=120,sat)", golden);
      $display("  T6b: act=120, zp=10 -> golden=%0d (saturated to 127)", golden[0]);

      // act_val=-120, zp_out=-10 -> clamp(-130) = -128 (saturate!)
      zp_out <= -8'sd10;
      set_uniform(-32'sd120, 32'sd0, 32'sd1, 6'd0, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T6c(zp=-10,val=-120,sat)", golden);
      $display("  T6c: act=-120, zp=-10 -> golden=%0d (saturated to -128)", golden[0]);

      // zp_out = 0 -> no effect
      zp_out <= 8'sd0;
      set_uniform(32'sd50, 32'sd0, 32'sd1, 6'd0, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T6d(zp=0)", golden);
      $display("  T6d: act=50, zp=0 -> golden=%0d", golden[0]);

      // Negative zp
      zp_out <= -8'sd20;
      set_uniform(32'sd50, 32'sd0, 32'sd1, 6'd0, 8'sd0);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T6e(zp=-20)", golden);
      $display("  T6e: act=50, zp=-20 -> golden=%0d", golden[0]);

      total_tests++;
      total_errors += test_errors;
      $display("  TEST 6 %s (%0d errors)",
               (test_errors == 0) ? "PASSED" : "FAILED", test_errors);
    end

    // ====================================================================
    // TEST 7: Element-wise Add (Skip Connection)
    // ====================================================================
    begin
      test_errors = 0;
      $display("\n--- TEST 7: Element-wise Add (Skip Connection) ---");

      cfg_post.bias_en   <= 1'b0;
      cfg_post.act_mode  <= ACT_CLAMP;
      cfg_post.ewise_en  <= 1'b1;
      zp_out <= 8'sd0;

      // y_act=50, ewise=30 -> y_add=80, +zp(0) -> clamp(80)=80
      set_uniform(32'sd50, 32'sd0, 32'sd1, 6'd0, 8'sd30);
      compute_golden_all(golden);
      // Need to drive ewise_valid along with psum_valid
      drive_and_wait();
      check_output("T7a(ewise=30)", golden);
      $display("  T7a: act=50, ewise=30 -> golden=%0d", golden[0]);

      // y_act=100, ewise=50 -> y_add=150 -> clamp=127
      set_uniform(32'sd100, 32'sd0, 32'sd1, 6'd0, 8'sd50);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T7b(ewise=50,sat)", golden);
      $display("  T7b: act=100, ewise=50 -> golden=%0d (saturated)", golden[0]);

      // Negative ewise
      set_uniform(32'sd30, 32'sd0, 32'sd1, 6'd0, -8'sd40);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T7c(neg ewise)", golden);
      $display("  T7c: act=30, ewise=-40 -> golden=%0d", golden[0]);

      // Both negative, saturate to -128
      set_uniform(-32'sd100, 32'sd0, 32'sd1, 6'd0, -8'sd50);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T7d(both neg,sat)", golden);
      $display("  T7d: act=-100, ewise=-50 -> golden=%0d (saturated to -128)", golden[0]);

      // ewise_en = 0 -> ewise ignored
      cfg_post.ewise_en <= 1'b0;
      set_uniform(32'sd50, 32'sd0, 32'sd1, 6'd0, 8'sd99);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T7e(ewise_en=0)", golden);
      $display("  T7e: ewise_en=0 -> golden=%0d (ewise ignored)", golden[0]);

      // Ewise + zp combined
      cfg_post.ewise_en <= 1'b1;
      zp_out <= 8'sd5;
      set_uniform(32'sd40, 32'sd0, 32'sd1, 6'd0, 8'sd20);
      compute_golden_all(golden);
      drive_and_wait();
      check_output("T7f(ewise+zp)", golden);
      $display("  T7f: act=40, ewise=20, zp=5 -> golden=%0d", golden[0]);

      total_tests++;
      total_errors += test_errors;
      $display("  TEST 7 %s (%0d errors)",
               (test_errors == 0) ? "PASSED" : "FAILED", test_errors);
    end

    // ====================================================================
    // TEST 8: Full Pipeline Golden Comparison (Conv L0 style)
    // ====================================================================
    begin
      automatic int seed;
      test_errors = 0;
      $display("\n--- TEST 8: Full Pipeline Golden Comparison (Conv L0 style) ---");

      seed = 42;

      cfg_post.bias_en    <= 1'b1;
      cfg_post.act_mode   <= ACT_RELU;
      cfg_post.ewise_en   <= 1'b0;
      cfg_post.quant_mode <= QMODE_PER_CHANNEL;

      // Generate random per-lane parameters
      for (int l = 0; l < LANES; l++) begin
        automatic int rand_val;
        // psum: random INT32 in [-50000, 50000]
        rand_val = $urandom(seed + l*5) % 100001;
        psum_in[l] <= 32'(rand_val - 50000);

        // bias: random INT32 in [-1000, 1000]
        rand_val = $urandom(seed + l*5 + 1) % 2001;
        bias_val[l] <= 32'(rand_val - 1000);

        // m_int: random in [1, 200]
        rand_val = $urandom(seed + l*5 + 2) % 200;
        m_int[l] <= 32'(rand_val + 1);

        // shift: random 6-bit in [8, 20]
        rand_val = $urandom(seed + l*5 + 3) % 13;
        shift[l] <= 6'(rand_val + 8);

        ewise_in[l] <= 8'sd0;
      end

      // zp_out: random INT8
      begin
        automatic int rand_val;
        rand_val = $urandom(seed + 999) % 256;
        zp_out <= 8'(rand_val - 128);
      end

      // Wait one cycle for signal assignments to take effect
      @(posedge clk);

      compute_golden_all(golden);
      drive_and_wait();
      check_output("T8(full pipeline)", golden);

      // Print details for first 4 lanes
      for (int l = 0; l < 4; l++) begin
        $display("  lane %0d: psum=%0d, bias=%0d, M=%0d, sh=%0d, zp=%0d -> golden=%0d, dut=%0d %s",
          l, psum_in[l], bias_val[l], m_int[l], shift[l], zp_out,
          golden[l], act_out[l],
          (act_out[l] === golden[l]) ? "OK" : "MISMATCH");
      end

      total_tests++;
      total_errors += test_errors;
      $display("  TEST 8 %s (%0d errors)",
               (test_errors == 0) ? "PASSED" : "FAILED", test_errors);
    end

    // ====================================================================
    // TEST 9: Stress Test (1000 random vectors)
    // ====================================================================
    begin
      automatic int vec_errors;
      automatic int mismatch_lanes;
      test_errors = 0;
      vec_errors  = 0;
      $display("\n--- TEST 9: Stress Test (1000 random vectors) ---");

      cfg_post.bias_en    <= 1'b1;
      cfg_post.act_mode   <= ACT_RELU;
      cfg_post.ewise_en   <= 1'b0;
      cfg_post.quant_mode <= QMODE_PER_CHANNEL;

      for (int v = 0; v < 1000; v++) begin
        mismatch_lanes = 0;

        // Randomize per-lane parameters
        for (int l = 0; l < LANES; l++) begin
          psum_in[l]  <= 32'($urandom % 100001) - 32'sd50000;
          bias_val[l] <= 32'($urandom % 2001) - 32'sd1000;
          m_int[l]    <= 32'($urandom % 200) + 32'sd1;
          shift[l]    <= 6'($urandom % 13) + 6'd8;
          ewise_in[l] <= 8'sd0;
        end
        zp_out <= 8'($urandom % 256) - 8'sd128;

        // Randomly pick activation mode (mostly RELU, some others)
        begin
          automatic int mode_sel;
          mode_sel = $urandom % 10;
          if (mode_sel < 7)
            cfg_post.act_mode <= ACT_RELU;
          else if (mode_sel < 9)
            cfg_post.act_mode <= ACT_CLAMP;
          else
            cfg_post.act_mode <= ACT_NONE;
        end

        // Randomly enable/disable bias
        cfg_post.bias_en <= ($urandom % 10 < 8) ? 1'b1 : 1'b0;

        // Let signal assignments settle
        @(posedge clk);

        compute_golden_all(golden);
        drive_and_wait();

        // Check
        for (int l = 0; l < LANES; l++) begin
          if (act_out[l] !== golden[l]) begin
            mismatch_lanes++;
            if (test_errors < 20) begin // Limit printout
              $display("  [FAIL] vec %0d lane %0d: DUT=%0d, GOLDEN=%0d (psum=%0d bias=%0d M=%0d sh=%0d zp=%0d act=%0d)",
                v, l, act_out[l], golden[l],
                psum_in[l], bias_val[l], m_int[l], shift[l], zp_out,
                cfg_post.act_mode);
            end
            test_errors++;
          end
        end
        if (mismatch_lanes > 0) vec_errors++;

        // Progress report every 200 vectors
        if ((v+1) % 200 == 0)
          $display("  ... %0d/1000 vectors done, %0d errors so far", v+1, test_errors);
      end

      total_tests++;
      total_errors += test_errors;
      $display("  TEST 9 %s (%0d lane errors across %0d/%0d vectors)",
               (test_errors == 0) ? "PASSED" : "FAILED",
               test_errors, vec_errors, 1000);
    end

    // ====================================================================
    // FINAL SUMMARY
    // ====================================================================
    $display("\n================================================================");
    $display(" PPU Testbench - FINAL SUMMARY");
    $display("================================================================");
    $display("  Tests run:    %0d", total_tests);
    $display("  Total errors: %0d", total_errors);
    if (total_errors == 0)
      $display("  *** ALL TESTS PASSED ***");
    else
      $display("  *** %0d ERRORS DETECTED - REVIEW FAILURES ABOVE ***", total_errors);
    $display("================================================================\n");

    $finish;
  end

  // --------------------------------------------------------------------------
  // Timeout watchdog
  // --------------------------------------------------------------------------
  initial begin
    #10_000_000;
    $display("[TIMEOUT] Simulation exceeded 10ms limit!");
    $finish;
  end

endmodule
