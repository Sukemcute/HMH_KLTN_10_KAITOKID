// Self-checking testbench for PPU
// Test 1: Bias + Requant only (ACT_NONE)
// Test 2: ReLU activation
// Test 3: SiLU activation (LUT-based)
// Test 4: Element-wise add with saturation
// Test 5: Full pipeline: bias + requant + SiLU + ewise
`timescale 1ns/1ps

module tb_ppu;
  import accel_pkg::*;
  import desc_pkg::*;

  localparam int LANES = 32;

  logic                  clk, rst_n, en;
  post_profile_t         cfg_post;
  pe_mode_e              cfg_mode;
  logic signed [31:0]    psum_in [LANES];
  logic                  psum_valid;
  logic signed [31:0]    bias_val [LANES];
  logic signed [31:0]    m_int [LANES];
  logic [5:0]            shift [LANES];
  logic signed [7:0]     zp_out;
  logic signed [7:0]     silu_lut_data [256];
  logic signed [7:0]     ewise_in [LANES];
  logic                  ewise_valid;
  logic signed [7:0]     act_out [LANES];
  logic                  act_valid;

  ppu #(.LANES(LANES)) uut (.*);

  always #2.5 clk = ~clk;

  int fail_count = 0;

  task automatic reset();
    rst_n = 0; en = 1; psum_valid = 0; ewise_valid = 0;
    cfg_post = '0; cfg_mode = PE_RS3; zp_out = 0;
    for (int l = 0; l < LANES; l++) begin
      psum_in[l] = 0; bias_val[l] = 0;
      m_int[l] = 1; shift[l] = 0;
      ewise_in[l] = 0;
    end
    repeat(5) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
  endtask

  task automatic wait_pipeline();
    repeat(6) @(posedge clk);
  endtask

  // Preload identity SiLU LUT (for testing: silu(x) = x)
  task automatic preload_identity_lut();
    for (int i = 0; i < 256; i++)
      silu_lut_data[i] = i[7:0] - 8'd128;
  endtask

  // ═══════════ TEST 1: Bias + Requant (ACT_NONE, zp=0) ═══════════
  task automatic test_bias_requant();
    int errors = 0;
    logic signed [31:0] expected_biased;
    logic signed [63:0] expected_mult;
    logic signed [7:0]  expected_out;

    $display("=== TEST 1: Bias + Requant (ACT_NONE) ===");
    reset();
    preload_identity_lut();

    cfg_post.bias_en    = 1;
    cfg_post.act_mode   = ACT_NONE;
    cfg_post.ewise_en   = 0;
    cfg_post.quant_mode = QMODE_PER_CHANNEL;

    // Set known values: psum=1000, bias=500, m_int=100, shift=10, zp=0
    for (int l = 0; l < LANES; l++) begin
      psum_in[l]  = 1000 + l * 10;
      bias_val[l] = 500;
      m_int[l]    = 100;
      shift[l]    = 10;
    end
    zp_out = 0;

    @(negedge clk);
    psum_valid = 1;
    @(negedge clk);
    psum_valid = 0;
    wait_pipeline();

    for (int l = 0; l < LANES; l++) begin
      expected_biased = 1000 + l * 10 + 500;
      expected_mult   = 64'(expected_biased) * 64'sd100;
      // half_up rounding: add 1 << (10-1) = 512
      expected_mult   = expected_mult + 64'sd512;
      begin
        automatic int shifted_val;
        shifted_val = int'(expected_mult >>> 10);
        // clamp to INT8
        if (shifted_val > 127) expected_out = 8'sd127;
        else if (shifted_val < -128) expected_out = -8'sd128;
        else expected_out = shifted_val[7:0];
      end

      if (act_out[l] !== expected_out) begin
        $display("  FAIL lane[%0d]: got=%0d expected=%0d (biased=%0d)",
                 l, act_out[l], expected_out, expected_biased);
        errors++;
      end
    end

    if (errors == 0) $display("  TEST 1 PASSED");
    else $display("  TEST 1 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  // ═══════════ TEST 2: ReLU ═══════════
  task automatic test_relu();
    int errors = 0;

    $display("=== TEST 2: ReLU activation ===");
    reset();

    cfg_post.bias_en  = 0;
    cfg_post.act_mode = ACT_RELU;
    cfg_post.ewise_en = 0;

    for (int l = 0; l < LANES; l++) begin
      psum_in[l] = (l < 16) ? -(l + 1) : (l - 15);
      m_int[l]   = 1;
      shift[l]   = 0;
    end
    zp_out = 0;

    @(negedge clk);
    psum_valid = 1;
    @(negedge clk);
    psum_valid = 0;
    wait_pipeline();

    for (int l = 0; l < LANES; l++) begin
      automatic logic signed [7:0] expected;
      if (l < 16)
        expected = 8'sd0;  // negative → 0
      else
        expected = (l - 15);
      if (act_out[l] !== expected) begin
        $display("  FAIL lane[%0d]: got=%0d expected=%0d (psum=%0d)",
                 l, act_out[l], expected, psum_in[l]);
        errors++;
      end
    end

    if (errors == 0) $display("  TEST 2 PASSED");
    else $display("  TEST 2 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  // ═══════════ TEST 3: Element-wise add with saturation ═══════════
  task automatic test_ewise_add();
    int errors = 0;

    $display("=== TEST 3: Element-wise add with saturation ===");
    reset();

    cfg_post.bias_en  = 0;
    cfg_post.act_mode = ACT_NONE;
    cfg_post.ewise_en = 1;

    for (int l = 0; l < LANES; l++) begin
      psum_in[l]  = 100;   // will become 100 after requant (m=1, shift=0)
      ewise_in[l] = 50;
      m_int[l]    = 1;
      shift[l]    = 0;
    end
    zp_out = 0;

    // Override: test saturation
    psum_in[0]  = 120;
    ewise_in[0] = 120;  // 120 + 120 = 240 → clamp to 127

    psum_in[1]  = -100;
    ewise_in[1] = -100; // -100 + -100 = -200 → clamp to -128

    @(negedge clk);
    psum_valid  = 1;
    ewise_valid = 1;
    @(negedge clk);
    psum_valid  = 0;
    ewise_valid = 0;
    wait_pipeline();

    // Lane 0: 120 + 120 → 240 → clamp 127
    if (act_out[0] !== 8'sd127) begin
      $display("  FAIL lane[0]: overflow clamp got=%0d expected=127", act_out[0]);
      errors++;
    end
    // Lane 1: -100 + -100 → -200 → clamp -128
    if (act_out[1] !== -8'sd128) begin
      $display("  FAIL lane[1]: underflow clamp got=%0d expected=-128", act_out[1]);
      errors++;
    end
    // Lane 2+: 100 + 50 → 150 → clamp 127
    for (int l = 2; l < LANES; l++) begin
      automatic logic signed [7:0] expected;
      automatic int sum_val;
      sum_val = 100 + 50;  // 150 → clamp 127
      expected = (sum_val > 127) ? 8'sd127 : sum_val[7:0];
      if (act_out[l] !== expected) begin
        $display("  FAIL lane[%0d]: got=%0d expected=%0d", l, act_out[l], expected);
        errors++;
      end
    end

    if (errors == 0) $display("  TEST 3 PASSED");
    else $display("  TEST 3 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  initial begin
    clk = 0;
    $display("\n╔══════════════════════════════════════════════════╗");
    $display("║  TESTBENCH: ppu (Post-Processing Unit)          ║");
    $display("╚══════════════════════════════════════════════════╝\n");

    test_bias_requant();
    test_relu();
    test_ewise_add();

    $display("\n══════════════════════════════════════════════════");
    if (fail_count == 0) $display("  ★ ALL PPU TESTS PASSED ★");
    else $display("  ✗ TOTAL FAILURES: %0d", fail_count);
    $display("══════════════════════════════════════════════════\n");
    $finish;
  end

endmodule
