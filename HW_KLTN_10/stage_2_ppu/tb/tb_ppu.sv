// ============================================================================
// Testbench : tb_ppu
// Project   : YOLOv10n INT8 Accelerator — V4-VC707
//
// Description:
//   Comprehensive 8-test verification of the PPU (Post-Processing Unit).
//   Covers half-up rounding, INT64 overflow, ReLU, ZP+clamp, golden vector,
//   per-lane independence, ACT_NONE, and random stress.
//
// Tests:
//   2.1.1  Half-up rounding vs Floor (100 cases)
//   2.1.2  INT64 overflow protection
//   2.1.3  ReLU activation
//   2.1.4  ZP_out + final clamp
//   2.1.5  Golden Python vector comparison (20 lanes)
//   2.1.6  Per-lane independence
//   2.1.7  ACT_NONE mode
//   2.1.8  Stress 1,000 random vectors
// ============================================================================
`timescale 1ns / 1ps

module tb_ppu;

  import accel_pkg::*;

  // ════════════════════════════════════════════════════════════════════════
  //  Parameters
  // ════════════════════════════════════════════════════════════════════════
  localparam int LANES     = accel_pkg::LANES;  // 20
  localparam int CLK_PERIOD = 4;                // 4 ns → 250 MHz
  localparam int PIPE_DEPTH = 5;                // 5-stage pipeline

  // ════════════════════════════════════════════════════════════════════════
  //  DUT Signals
  // ════════════════════════════════════════════════════════════════════════
  logic        clk;
  logic        rst_n;

  int32_t      psum_in [LANES];
  logic        psum_valid;

  int32_t      bias_val;
  uint32_t     m_int;
  logic [7:0]  shift_val;
  int8_t       zp_out;
  act_mode_e   activation;

  int8_t       act_out [LANES];
  logic        act_valid;

  // ════════════════════════════════════════════════════════════════════════
  //  DUT Instantiation
  // ════════════════════════════════════════════════════════════════════════
  ppu #(
    .LANES(LANES)
  ) u_dut (
    .clk        (clk),
    .rst_n      (rst_n),
    .psum_in    (psum_in),
    .psum_valid (psum_valid),
    .bias_val   (bias_val),
    .m_int      (m_int),
    .shift_val  (shift_val),
    .zp_out     (zp_out),
    .activation (activation),
    .act_out    (act_out),
    .act_valid  (act_valid)
  );

  // ════════════════════════════════════════════════════════════════════════
  //  Clock Generation — 250 MHz (4 ns period)
  // ════════════════════════════════════════════════════════════════════════
  initial clk = 1'b0;
  always #(CLK_PERIOD / 2) clk = ~clk;

  // ════════════════════════════════════════════════════════════════════════
  //  Timeout Watchdog — 10 ms
  // ════════════════════════════════════════════════════════════════════════
  initial begin
    #10_000_000;
    $display("[TIMEOUT] Simulation exceeded 10 ms. Aborting.");
    $finish;
  end

  // ════════════════════════════════════════════════════════════════════════
  //  Global Error Counter
  // ════════════════════════════════════════════════════════════════════════
  int total_errors;
  int test_errors;

  // ════════════════════════════════════════════════════════════════════════
  //  Golden Behavioral Model
  //  Replicates RTL pipeline logic exactly.
  // ════════════════════════════════════════════════════════════════════════
  function automatic logic signed [7:0] golden_ppu(
    input int32_t   psum,
    input int32_t   bias,
    input uint32_t  m,
    input logic [7:0] sh,
    input int8_t    zp,
    input act_mode_e act
  );
    int64_t  biased64;
    int64_t  product;
    int64_t  rounded;
    int32_t  shifted;
    int32_t  activated;
    int32_t  with_zp;
    int      shift_int;

    // Stage 1: Bias add (INT32)
    biased64 = int64_t'(int32_t'(psum + bias));

    // Stage 2: INT64 multiply — signed biased * unsigned m_int
    // {1'b0, m} creates a 33-bit unsigned value, extended to INT64
    product = biased64 * int64_t'({1'b0, m});

    // Stage 3: Half-up rounding + arithmetic right shift
    shift_int = int'(sh);
    if (shift_int > 0)
      rounded = product + (int64_t'(1) <<< (shift_int - 1));
    else
      rounded = product;
    shifted = int32_t'(rounded >>> shift_int);

    // Stage 4: Activation
    case (act)
      ACT_RELU: activated = (shifted > 32'sd0) ? shifted : 32'sd0;
      default:  activated = shifted;
    endcase

    // Stage 5: ZP add + final clamp to [-128, 127]
    with_zp = activated + int32_t'(zp);
    if (with_zp > 32'sd127)
      return int8_t'(8'sd127);
    else if (with_zp < -32'sd128)
      return int8_t'(-8'sd128);
    else
      return int8_t'(with_zp[7:0]);
  endfunction

  // ════════════════════════════════════════════════════════════════════════
  //  Helper Tasks
  // ════════════════════════════════════════════════════════════════════════

  // Reset the DUT
  task automatic do_reset();
    rst_n      <= 1'b0;
    psum_valid <= 1'b0;
    bias_val   <= 32'sd0;
    m_int      <= 32'd0;
    shift_val  <= 8'd0;
    zp_out     <= 8'sd0;
    activation <= ACT_NONE;
    for (int i = 0; i < LANES; i++)
      psum_in[i] <= 32'sd0;
    repeat (4) @(posedge clk);
    rst_n <= 1'b1;
    repeat (2) @(posedge clk);
  endtask

  // Drive one vector through the PPU and wait for output
  task automatic drive_and_wait(
    input int32_t   psum_arr [LANES],
    input int32_t   bias_v,
    input uint32_t  m_v,
    input logic [7:0] sh_v,
    input int8_t    zp_v,
    input act_mode_e act_v
  );
    @(posedge clk);
    for (int i = 0; i < LANES; i++)
      psum_in[i] <= psum_arr[i];
    bias_val   <= bias_v;
    m_int      <= m_v;
    shift_val  <= sh_v;
    zp_out     <= zp_v;
    activation <= act_v;
    psum_valid <= 1'b1;
    @(posedge clk);
    psum_valid <= 1'b0;
    // Wait for pipeline: 5 clock stages
    repeat (PIPE_DEPTH) @(posedge clk);
    // act_valid should be high now — sample on next posedge
    @(negedge clk);  // sample in middle of act_valid cycle
  endtask

  // Check all lanes against golden
  task automatic check_lanes(
    input int32_t   psum_arr [LANES],
    input int32_t   bias_v,
    input uint32_t  m_v,
    input logic [7:0] sh_v,
    input int8_t    zp_v,
    input act_mode_e act_v,
    input string    test_name,
    inout int       errs
  );
    int8_t expected;
    for (int l = 0; l < LANES; l++) begin
      expected = golden_ppu(psum_arr[l], bias_v, m_v, sh_v, zp_v, act_v);
      if (act_out[l] !== expected) begin
        $display("  [FAIL] %s lane %0d: psum=%0d got=%0d exp=%0d",
                 test_name, l, psum_arr[l], act_out[l], expected);
        errs++;
      end
    end
  endtask

  // ════════════════════════════════════════════════════════════════════════
  //  Simple PRNG (xorshift32) for deterministic random
  // ════════════════════════════════════════════════════════════════════════
  int unsigned prng_state;

  function automatic int unsigned xorshift32();
    int unsigned x;
    x = prng_state;
    x = x ^ (x << 13);
    x = x ^ (x >> 17);
    x = x ^ (x << 5);
    prng_state = x;
    return x;
  endfunction

  // Return random signed 32-bit in range [-limit, +limit]
  function automatic int32_t rand_signed(input int unsigned limit);
    int unsigned r;
    int64_t      val;
    r = xorshift32();
    // Use modulo to stay within range, then random sign
    val = int64_t'(r % (limit + 1));
    if (xorshift32() & 1)
      return int32_t'(-val);
    else
      return int32_t'(val);
  endfunction

  // ════════════════════════════════════════════════════════════════════════
  //  MAIN TEST SEQUENCE
  // ════════════════════════════════════════════════════════════════════════
  initial begin
    int32_t  psum_arr [LANES];
    int      pass_count;
    int      fail_count;

    total_errors = 0;
    pass_count   = 0;
    fail_count   = 0;

    $display("================================================================");
    $display("  tb_ppu — PPU Verification (8 Tests)");
    $display("  LANES = %0d, Clock = %0d MHz, Pipeline = %0d stages",
             LANES, 1000 / CLK_PERIOD, PIPE_DEPTH);
    $display("================================================================");

    do_reset();

    // ──────────────────────────────────────────────────────────────────
    //  TEST 2.1.1: Half-up rounding vs Floor (100 cases)
    // ──────────────────────────────────────────────────────────────────
    begin
      int8_t  expected_golden;
      int     case_count;
      int32_t psum_v;
      int32_t bias_v;
      uint32_t m_v;
      logic [7:0] sh_v;

      test_errors = 0;
      case_count  = 0;
      $display("\n[TEST 2.1.1] Half-up rounding vs Floor — 100 cases");

      // Specific documented case: biased=15, M_int=100, shift=3
      // product = 15 * 100 = 1500
      // floor = 1500 >>> 3 = 187
      // half-up = (1500 + 4) >>> 3 = 1504 >>> 3 = 188
      for (int i = 0; i < LANES; i++)
        psum_arr[i] = 32'sd15;
      drive_and_wait(psum_arr, 32'sd0, 32'd100, 8'd3, 8'sd0, ACT_NONE);
      for (int l = 0; l < LANES; l++) begin
        if (act_out[l] !== 8'sd127) begin
          // After clamp: 188 > 127 → 127. But let's check golden.
        end
      end
      check_lanes(psum_arr, 32'sd0, 32'd100, 8'd3, 8'sd0, ACT_NONE,
                  "T2.1.1-specific", test_errors);

      // Generate 100 cases where half-up != floor
      // Condition: product has bit (sh-1) set (i.e., the rounding bit is 1)
      // We construct: biased * m_int such that bit (sh-1) of product is 1
      prng_state = 32'hDEAD_BEEF;
      case_count = 0;
      while (case_count < 100) begin
        int64_t  product64;
        int      sh_int;
        int64_t  half_bit;

        // Random shift in [1, 20]
        sh_int = int'((xorshift32() % 20) + 1);
        sh_v   = sh_int[7:0];

        // Random small biased and m_int to keep product manageable
        psum_v = rand_signed(32'd1000);
        bias_v = rand_signed(32'd500);
        m_v    = xorshift32() % 32'd10000 + 1;

        // Compute product to check if rounding bit is set
        product64 = int64_t'(int32_t'(psum_v + bias_v)) * int64_t'({1'b0, m_v});
        half_bit  = (int64_t'(1) <<< (sh_int - 1));

        // Only use cases where the rounding bit matters (half_bit is set in product)
        if ((product64 & half_bit) != 0) begin
          for (int i = 0; i < LANES; i++)
            psum_arr[i] = psum_v;
          drive_and_wait(psum_arr, bias_v, m_v, sh_v, 8'sd0, ACT_NONE);
          check_lanes(psum_arr, bias_v, m_v, sh_v, 8'sd0, ACT_NONE,
                      $sformatf("T2.1.1-case%0d", case_count), test_errors);
          case_count++;
        end
      end

      if (test_errors == 0) begin
        $display("  [PASS] Test 2.1.1: All 100 half-up rounding cases correct.");
        pass_count++;
      end else begin
        $display("  [FAIL] Test 2.1.1: %0d errors in half-up rounding.", test_errors);
        fail_count++;
      end
      total_errors += test_errors;
    end

    // ──────────────────────────────────────────────────────────────────
    //  TEST 2.1.2: INT64 overflow protection
    // ──────────────────────────────────────────────────────────────────
    begin
      test_errors = 0;
      $display("\n[TEST 2.1.2] INT64 overflow protection");

      // biased = 2,000,000,000, M_int = 1,500,000,000
      // We set psum = 2_000_000_000, bias = 0 so biased = 2_000_000_000
      // product = 2e9 * 1.5e9 = 3e18 > INT32_MAX (2.1e9) but < INT64_MAX (9.2e18)
      for (int i = 0; i < LANES; i++)
        psum_arr[i] = 32'sd2_000_000_000;

      // Use shift = 31 so the result fits in INT32 after shifting
      // product = 3e18, shifted = (3e18 + 2^30) >>> 31 ≈ 1_395_348_782
      // After clamp → 127
      drive_and_wait(psum_arr, 32'sd0, 32'd1_500_000_000, 8'd31, 8'sd0, ACT_NONE);
      check_lanes(psum_arr, 32'sd0, 32'd1_500_000_000, 8'd31, 8'sd0, ACT_NONE,
                  "T2.1.2-overflow", test_errors);

      // Additional case: negative large
      for (int i = 0; i < LANES; i++)
        psum_arr[i] = -32'sd2_000_000_000;
      drive_and_wait(psum_arr, 32'sd0, 32'd1_500_000_000, 8'd31, 8'sd0, ACT_NONE);
      check_lanes(psum_arr, 32'sd0, 32'd1_500_000_000, 8'd31, 8'sd0, ACT_NONE,
                  "T2.1.2-neg-overflow", test_errors);

      // Case with bias contributing to large biased
      for (int i = 0; i < LANES; i++)
        psum_arr[i] = 32'sd1_000_000_000;
      drive_and_wait(psum_arr, 32'sd1_000_000_000, 32'd2_000_000_000, 8'd31, 8'sd0, ACT_NONE);
      check_lanes(psum_arr, 32'sd1_000_000_000, 32'd2_000_000_000, 8'd31, 8'sd0, ACT_NONE,
                  "T2.1.2-bias-overflow", test_errors);

      if (test_errors == 0) begin
        $display("  [PASS] Test 2.1.2: INT64 overflow protection verified.");
        pass_count++;
      end else begin
        $display("  [FAIL] Test 2.1.2: %0d errors.", test_errors);
        fail_count++;
      end
      total_errors += test_errors;
    end

    // ──────────────────────────────────────────────────────────────────
    //  TEST 2.1.3: ReLU activation
    // ──────────────────────────────────────────────────────────────────
    begin
      int32_t relu_inputs [7];
      int8_t  relu_expected [7];

      test_errors = 0;
      $display("\n[TEST 2.1.3] ReLU activation");

      // shifted = [-100, -1, 0, 1, 50, 127, 200]
      // After ReLU: [0, 0, 0, 1, 50, 127, 200]
      // After clamp (zp=0): [0, 0, 0, 1, 50, 127, 127] (200 > 127 → 127)
      relu_inputs[0] = -32'sd100;
      relu_inputs[1] = -32'sd1;
      relu_inputs[2] = 32'sd0;
      relu_inputs[3] = 32'sd1;
      relu_inputs[4] = 32'sd50;
      relu_inputs[5] = 32'sd127;
      relu_inputs[6] = 32'sd200;

      relu_expected[0] = 8'sd0;
      relu_expected[1] = 8'sd0;
      relu_expected[2] = 8'sd0;
      relu_expected[3] = 8'sd1;
      relu_expected[4] = 8'sd50;
      relu_expected[5] = 8'sd127;
      relu_expected[6] = 8'sd127;  // 200 clamped to 127

      // To get "shifted" = value, we set: psum = value, bias = 0, m_int = 1, shift = 0
      // biased = value, product = value * 1 = value, shifted = value >>> 0 = value
      for (int tc = 0; tc < 7; tc++) begin
        for (int i = 0; i < LANES; i++)
          psum_arr[i] = relu_inputs[tc];
        drive_and_wait(psum_arr, 32'sd0, 32'd1, 8'd0, 8'sd0, ACT_RELU);
        for (int l = 0; l < LANES; l++) begin
          if (act_out[l] !== relu_expected[tc]) begin
            $display("  [FAIL] T2.1.3 case %0d lane %0d: shifted=%0d got=%0d exp=%0d",
                     tc, l, relu_inputs[tc], act_out[l], relu_expected[tc]);
            test_errors++;
          end
        end
      end

      if (test_errors == 0) begin
        $display("  [PASS] Test 2.1.3: ReLU activation correct for all 7 cases.");
        pass_count++;
      end else begin
        $display("  [FAIL] Test 2.1.3: %0d errors.", test_errors);
        fail_count++;
      end
      total_errors += test_errors;
    end

    // ──────────────────────────────────────────────────────────────────
    //  TEST 2.1.4: ZP_out + final clamp
    // ──────────────────────────────────────────────────────────────────
    begin
      test_errors = 0;
      $display("\n[TEST 2.1.4] ZP_out + final clamp");

      // Case A: activated=120, zp_out=10 → 130 → clamp → 127
      // Use ACT_NONE so activated = shifted = psum (with m=1, shift=0, bias=0)
      for (int i = 0; i < LANES; i++)
        psum_arr[i] = 32'sd120;
      drive_and_wait(psum_arr, 32'sd0, 32'd1, 8'd0, 8'sd10, ACT_NONE);
      for (int l = 0; l < LANES; l++) begin
        if (act_out[l] !== 8'sd127) begin
          $display("  [FAIL] T2.1.4-A lane %0d: got=%0d exp=127", l, act_out[l]);
          test_errors++;
        end
      end

      // Case B: activated=-50, zp_out=-80 → -130 → clamp → -128
      for (int i = 0; i < LANES; i++)
        psum_arr[i] = -32'sd50;
      drive_and_wait(psum_arr, 32'sd0, 32'd1, 8'd0, -8'sd80, ACT_NONE);
      for (int l = 0; l < LANES; l++) begin
        if (act_out[l] !== -8'sd128) begin
          $display("  [FAIL] T2.1.4-B lane %0d: got=%0d exp=-128", l, act_out[l]);
          test_errors++;
        end
      end

      // Case C: activated=50, zp_out=20 → 70 → no clamp → 70
      for (int i = 0; i < LANES; i++)
        psum_arr[i] = 32'sd50;
      drive_and_wait(psum_arr, 32'sd0, 32'd1, 8'd0, 8'sd20, ACT_NONE);
      for (int l = 0; l < LANES; l++) begin
        if (act_out[l] !== 8'sd70) begin
          $display("  [FAIL] T2.1.4-C lane %0d: got=%0d exp=70", l, act_out[l]);
          test_errors++;
        end
      end

      if (test_errors == 0) begin
        $display("  [PASS] Test 2.1.4: ZP_out + clamp correct for all 3 cases.");
        pass_count++;
      end else begin
        $display("  [FAIL] Test 2.1.4: %0d errors.", test_errors);
        fail_count++;
      end
      total_errors += test_errors;
    end

    // ──────────────────────────────────────────────────────────────────
    //  TEST 2.1.5: Golden Python vector comparison (20 lanes)
    //  Realistic L0 parameters: bias=1500, M_int=1073741824, shift=15
    //  zp_out = -3, activation = ACT_RELU
    // ──────────────────────────────────────────────────────────────────
    begin
      int32_t  golden_psum [20];
      int8_t   golden_exp  [20];

      test_errors = 0;
      $display("\n[TEST 2.1.5] Golden Python vector comparison — 20 lanes");

      // Realistic L0 Conv psum values (typical INT32 range)
      golden_psum[0]  =  32'sd12345;
      golden_psum[1]  = -32'sd6789;
      golden_psum[2]  =  32'sd30000;
      golden_psum[3]  = -32'sd15000;
      golden_psum[4]  =  32'sd500;
      golden_psum[5]  =  32'sd0;
      golden_psum[6]  = -32'sd1;
      golden_psum[7]  =  32'sd99999;
      golden_psum[8]  = -32'sd50000;
      golden_psum[9]  =  32'sd7777;
      golden_psum[10] =  32'sd2048;
      golden_psum[11] = -32'sd1024;
      golden_psum[12] =  32'sd65536;
      golden_psum[13] = -32'sd32768;
      golden_psum[14] =  32'sd100;
      golden_psum[15] =  32'sd256;
      golden_psum[16] = -32'sd128;
      golden_psum[17] =  32'sd44444;
      golden_psum[18] = -32'sd88888;
      golden_psum[19] =  32'sd1;

      // Compute expected using golden function
      for (int l = 0; l < 20; l++) begin
        golden_exp[l] = golden_ppu(golden_psum[l], 32'sd1500,
                                    32'd1073741824, 8'd15, -8'sd3, ACT_RELU);
      end

      // Print expected values for reference
      $display("  Expected outputs (golden):");
      for (int l = 0; l < 20; l++)
        $display("    lane[%2d]: psum=%10d → exp=%4d", l, golden_psum[l], golden_exp[l]);

      // Drive through DUT
      for (int i = 0; i < LANES; i++)
        psum_arr[i] = golden_psum[i];
      drive_and_wait(psum_arr, 32'sd1500, 32'd1073741824, 8'd15, -8'sd3, ACT_RELU);

      // Check each lane
      for (int l = 0; l < LANES; l++) begin
        if (act_out[l] !== golden_exp[l]) begin
          $display("  [FAIL] T2.1.5 lane %0d: psum=%0d got=%0d exp=%0d",
                   l, golden_psum[l], act_out[l], golden_exp[l]);
          test_errors++;
        end
      end

      if (test_errors == 0) begin
        $display("  [PASS] Test 2.1.5: All 20 lanes match golden vector.");
        pass_count++;
      end else begin
        $display("  [FAIL] Test 2.1.5: %0d errors.", test_errors);
        fail_count++;
      end
      total_errors += test_errors;
    end

    // ──────────────────────────────────────────────────────────────────
    //  TEST 2.1.6: Per-lane independence
    //  Different psum per lane, same quant params → each lane independent
    // ──────────────────────────────────────────────────────────────────
    begin
      test_errors = 0;
      $display("\n[TEST 2.1.6] Per-lane independence");

      // Each lane gets a unique psum value
      for (int i = 0; i < LANES; i++)
        psum_arr[i] = int32_t'((i * 137) - 1000);  // spread: -1000, -863, -726, ...

      drive_and_wait(psum_arr, 32'sd200, 32'd50000, 8'd10, 8'sd5, ACT_RELU);

      // Check each lane independently
      for (int l = 0; l < LANES; l++) begin
        int8_t exp_v;
        exp_v = golden_ppu(psum_arr[l], 32'sd200, 32'd50000, 8'd10, 8'sd5, ACT_RELU);
        if (act_out[l] !== exp_v) begin
          $display("  [FAIL] T2.1.6 lane %0d: psum=%0d got=%0d exp=%0d",
                   l, psum_arr[l], act_out[l], exp_v);
          test_errors++;
        end
      end

      // Second vector: wider spread including negatives
      for (int i = 0; i < LANES; i++)
        psum_arr[i] = int32_t'((i * 5000) - 50000);

      drive_and_wait(psum_arr, -32'sd100, 32'd123456, 8'd14, -8'sd10, ACT_RELU);
      for (int l = 0; l < LANES; l++) begin
        int8_t exp_v;
        exp_v = golden_ppu(psum_arr[l], -32'sd100, 32'd123456, 8'd14, -8'sd10, ACT_RELU);
        if (act_out[l] !== exp_v) begin
          $display("  [FAIL] T2.1.6-vec2 lane %0d: psum=%0d got=%0d exp=%0d",
                   l, psum_arr[l], act_out[l], exp_v);
          test_errors++;
        end
      end

      // Third vector: all distinct with ACT_NONE
      for (int i = 0; i < LANES; i++)
        psum_arr[i] = int32_t'(i * 11 - 110);

      drive_and_wait(psum_arr, 32'sd0, 32'd1, 8'd0, 8'sd0, ACT_NONE);
      for (int l = 0; l < LANES; l++) begin
        int8_t exp_v;
        exp_v = golden_ppu(psum_arr[l], 32'sd0, 32'd1, 8'd0, 8'sd0, ACT_NONE);
        if (act_out[l] !== exp_v) begin
          $display("  [FAIL] T2.1.6-vec3 lane %0d: psum=%0d got=%0d exp=%0d",
                   l, psum_arr[l], act_out[l], exp_v);
          test_errors++;
        end
      end

      if (test_errors == 0) begin
        $display("  [PASS] Test 2.1.6: Per-lane independence verified (3 vectors).");
        pass_count++;
      end else begin
        $display("  [FAIL] Test 2.1.6: %0d errors.", test_errors);
        fail_count++;
      end
      total_errors += test_errors;
    end

    // ──────────────────────────────────────────────────────────────────
    //  TEST 2.1.7: ACT_NONE mode (pass-through, no activation)
    //  shifted → clamp(shifted + zp)
    // ──────────────────────────────────────────────────────────────────
    begin
      test_errors = 0;
      $display("\n[TEST 2.1.7] ACT_NONE mode");

      // Case A: Negative shifted should pass through (no ReLU clamp)
      // shifted = -50, zp = 0 → -50
      for (int i = 0; i < LANES; i++)
        psum_arr[i] = -32'sd50;
      drive_and_wait(psum_arr, 32'sd0, 32'd1, 8'd0, 8'sd0, ACT_NONE);
      for (int l = 0; l < LANES; l++) begin
        if (act_out[l] !== -8'sd50) begin
          $display("  [FAIL] T2.1.7-A lane %0d: got=%0d exp=-50", l, act_out[l]);
          test_errors++;
        end
      end

      // Case B: Negative shifted with zp → negative result
      // shifted = -30, zp = -10 → -40
      for (int i = 0; i < LANES; i++)
        psum_arr[i] = -32'sd30;
      drive_and_wait(psum_arr, 32'sd0, 32'd1, 8'd0, -8'sd10, ACT_NONE);
      for (int l = 0; l < LANES; l++) begin
        if (act_out[l] !== -8'sd40) begin
          $display("  [FAIL] T2.1.7-B lane %0d: got=%0d exp=-40", l, act_out[l]);
          test_errors++;
        end
      end

      // Case C: Positive shifted, no activation needed
      // shifted = 100, zp = 20 → 120
      for (int i = 0; i < LANES; i++)
        psum_arr[i] = 32'sd100;
      drive_and_wait(psum_arr, 32'sd0, 32'd1, 8'd0, 8'sd20, ACT_NONE);
      for (int l = 0; l < LANES; l++) begin
        if (act_out[l] !== 8'sd120) begin
          $display("  [FAIL] T2.1.7-C lane %0d: got=%0d exp=120", l, act_out[l]);
          test_errors++;
        end
      end

      // Case D: Negative shifted would be zeroed by ReLU, but ACT_NONE preserves it
      // shifted = -100, zp = 50 → -50
      for (int i = 0; i < LANES; i++)
        psum_arr[i] = -32'sd100;
      drive_and_wait(psum_arr, 32'sd0, 32'd1, 8'd0, 8'sd50, ACT_NONE);
      for (int l = 0; l < LANES; l++) begin
        if (act_out[l] !== -8'sd50) begin
          $display("  [FAIL] T2.1.7-D lane %0d: got=%0d exp=-50", l, act_out[l]);
          test_errors++;
        end
      end

      // Case E: Large negative with clamp
      // shifted = -200, zp = 0 → -200 → clamp → -128
      for (int i = 0; i < LANES; i++)
        psum_arr[i] = -32'sd200;
      drive_and_wait(psum_arr, 32'sd0, 32'd1, 8'd0, 8'sd0, ACT_NONE);
      for (int l = 0; l < LANES; l++) begin
        if (act_out[l] !== -8'sd128) begin
          $display("  [FAIL] T2.1.7-E lane %0d: got=%0d exp=-128", l, act_out[l]);
          test_errors++;
        end
      end

      if (test_errors == 0) begin
        $display("  [PASS] Test 2.1.7: ACT_NONE mode correct for all 5 cases.");
        pass_count++;
      end else begin
        $display("  [FAIL] Test 2.1.7: %0d errors.", test_errors);
        fail_count++;
      end
      total_errors += test_errors;
    end

    // ──────────────────────────────────────────────────────────────────
    //  TEST 2.1.8: Stress 1,000 random vectors
    // ──────────────────────────────────────────────────────────────────
    begin
      int32_t  stress_bias;
      uint32_t stress_m;
      logic [7:0] stress_sh;
      int8_t   stress_zp;
      act_mode_e stress_act;
      int      vec_errors;

      test_errors = 0;
      $display("\n[TEST 2.1.8] Stress 1,000 random vectors");

      prng_state = 32'hCAFE_BABE;

      for (int v = 0; v < 1000; v++) begin
        vec_errors = 0;

        // Random psum in [-10^8, 10^8]
        for (int i = 0; i < LANES; i++)
          psum_arr[i] = rand_signed(32'd100_000_000);

        // Random bias in [-10^6, 10^6]
        stress_bias = rand_signed(32'd1_000_000);

        // Random M_int in [1, 2^31 - 1]
        stress_m = (xorshift32() & 32'h7FFF_FFFE) + 32'd1;

        // Random shift in [0, 31]
        stress_sh = xorshift32() % 32;

        // Random zp_out in [-128, 127]
        stress_zp = int8_t'(xorshift32() % 256);

        // Random activation: RELU or NONE
        stress_act = (xorshift32() & 1) ? ACT_RELU : ACT_NONE;

        drive_and_wait(psum_arr, stress_bias, stress_m, stress_sh,
                       stress_zp, stress_act);

        for (int l = 0; l < LANES; l++) begin
          int8_t exp_v;
          exp_v = golden_ppu(psum_arr[l], stress_bias, stress_m, stress_sh,
                             stress_zp, stress_act);
          if (act_out[l] !== exp_v) begin
            if (vec_errors < 3) begin  // limit per-vector spam
              $display("  [FAIL] T2.1.8 vec %0d lane %0d: psum=%0d bias=%0d m=%0d sh=%0d zp=%0d act=%s got=%0d exp=%0d",
                       v, l, psum_arr[l], stress_bias, stress_m, stress_sh,
                       stress_zp, stress_act.name(), act_out[l], exp_v);
            end
            vec_errors++;
          end
        end

        test_errors += vec_errors;

        // Progress indicator every 200 vectors
        if ((v + 1) % 200 == 0)
          $display("  ... completed %0d / 1000 vectors (%0d errors so far)",
                   v + 1, test_errors);
      end

      if (test_errors == 0) begin
        $display("  [PASS] Test 2.1.8: All 1,000 random vectors (20,000 lanes) correct.");
        pass_count++;
      end else begin
        $display("  [FAIL] Test 2.1.8: %0d errors across 1,000 vectors.", test_errors);
        fail_count++;
      end
      total_errors += test_errors;
    end

    // ──────────────────────────────────────────────────────────────────
    //  FINAL SUMMARY
    // ──────────────────────────────────────────────────────────────────
    $display("\n================================================================");
    $display("  FINAL SUMMARY");
    $display("================================================================");
    $display("  Tests passed : %0d / 8", pass_count);
    $display("  Tests failed : %0d / 8", fail_count);
    $display("  Total errors : %0d", total_errors);
    $display("================================================================");

    if (total_errors == 0)
      $display("  *** ALL 8 TESTS PASSED ***");
    else
      $display("  *** %0d TEST(S) FAILED ***", fail_count);

    $display("================================================================\n");
    $finish;
  end

endmodule
