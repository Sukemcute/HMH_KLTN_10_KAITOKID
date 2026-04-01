
`timescale 1ns/1ps

module tb_pe_unit;
  import accel_pkg::*;

  localparam int LANES = 32;

  logic              clk, rst_n, en, clear_psum;
  pe_mode_e          mode;
  logic signed [7:0] x_in  [LANES];
  logic signed [7:0] w_in  [LANES];
  logic signed [31:0] psum_out [LANES];
  logic               psum_valid;

  pe_unit #(.LANES(LANES)) uut (.*);

  localparam CLK_PERIOD = 5;
  always #(CLK_PERIOD/2.0) clk = ~clk;

  int fail_count = 0;

  task automatic reset();
    rst_n = 0; en = 0; clear_psum = 0;
    mode = PE_RS3;
    for (int i = 0; i < LANES; i++) begin
      x_in[i] = 0; w_in[i] = 0;
    end
    repeat(5) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
  endtask

  // ═══════════ TEST 1: Single product, shared weight per DSP pair ═══════════
  task automatic test_single_product();
    int errors = 0;
    logic signed [31:0] expected;
    int pair_idx;

    $display("=== TEST 1: Single product, 32 lanes (RS3 pair-shared weights) ===");
    reset();
    mode = PE_RS3;

    // Drive on negedge so DUT samples stable values at next posedge.
    @(negedge clk);
    en = 1; clear_psum = 1;
    for (int i = 0; i < LANES; i += 2) begin
      x_in[i] = (i - 16);     // -16 to +15
      x_in[i + 1] = (i - 15); // paired odd lane
      w_in[i] = (i + 1);      // one shared weight per pair
      w_in[i + 1] = (i + 2);  // ignored by RTL in RS3 mode
    end
    @(negedge clk);
    en = 0;
    clear_psum = 0;
    repeat(6) @(posedge clk);

    for (int i = 0; i < LANES; i++) begin
      pair_idx = (i / 2) * 2;
      expected = 32'($signed(x_in[i])) * 32'($signed(w_in[pair_idx]));
      if (psum_out[i] !== expected) begin
        $display("  FAIL lane[%0d]: x=%0d w=%0d got=%0d expected=%0d",
                 i, x_in[i], w_in[pair_idx], psum_out[i], expected);
        errors++;
      end
    end

    if (errors == 0)
      $display("  TEST 1 PASSED");
    else
      $display("  TEST 1 FAILED: %0d lane errors", errors);
    fail_count += errors;
  endtask

  // ═══════════ TEST 2: 9-cycle accumulation ═══════════
  task automatic test_accumulation();
    int errors = 0;
    logic signed [31:0] expected [LANES];
    logic signed [7:0] tx [9], tw [9];

    $display("=== TEST 2: 9-cycle accumulation (conv3x3) ===");
    reset();
    mode = PE_RS3;

    tx = '{10, -20, 30, -40, 50, -60, 70, -80, 90};
    tw = '{3,  -7,  11, -13, 17, -19, 23, -29, 31};

    for (int l = 0; l < LANES; l++)
      expected[l] = 0;

    for (int c = 0; c < 9; c++) begin
      @(negedge clk);
      en = 1;
      clear_psum = (c == 0);
      for (int l = 0; l < LANES; l++) begin
        x_in[l] = tx[c] + l[7:0];  // vary per lane
        w_in[l] = tw[c];
        expected[l] += 32'(tx[c] + $signed(l[7:0])) * 32'(tw[c]);
      end
    end
    @(negedge clk);
    en = 0;
    clear_psum = 0;
    repeat(6) @(posedge clk);

    for (int l = 0; l < LANES; l++) begin
      if (psum_out[l] !== expected[l]) begin
        $display("  FAIL lane[%0d]: got=%0d expected=%0d", l, psum_out[l], expected[l]);
        errors++;
        if (errors > 5) break;
      end
    end

    if (errors == 0)
      $display("  TEST 2 PASSED");
    else
      $display("  TEST 2 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  // ═══════════ TEST 3: OS1 mode (broadcast weight) ═══════════
  task automatic test_os1_broadcast();
    int errors = 0;
    logic signed [31:0] expected;
    logic signed [7:0] broadcast_w;

    $display("=== TEST 3: OS1 mode — weight broadcast ===");
    reset();
    mode = PE_OS1;

    broadcast_w = 8'sd7;

    @(negedge clk);
    en = 1; clear_psum = 1;
    for (int l = 0; l < LANES; l++) begin
      x_in[l] = l[7:0] + 1;
      w_in[l] = broadcast_w;  // only w_in[0] used in OS1
    end
    w_in[0] = broadcast_w;
    @(negedge clk);
    en = 0;
    clear_psum = 0;
    repeat(6) @(posedge clk);

    for (int l = 0; l < LANES; l++) begin
      expected = 32'($signed(l[7:0] + 1)) * 32'(broadcast_w);
      if (psum_out[l] !== expected) begin
        $display("  FAIL lane[%0d]: got=%0d expected=%0d", l, psum_out[l], expected);
        errors++;
      end
    end

    if (errors == 0)
      $display("  TEST 3 PASSED");
    else
      $display("  TEST 3 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  // ═══════════ TEST 4: Random stress with pair-shared weights ═══════════
  task automatic test_random();
    int errors = 0;
    logic signed [31:0] expected [LANES];
    logic signed [7:0] pair_w;

    $display("=== TEST 4: Random stress (100 pair-shared RS3 products) ===");

    for (int t = 0; t < 100; t++) begin
      reset();
      mode = PE_RS3;

      @(negedge clk);
      en = 1; clear_psum = 1;
      for (int l = 0; l < LANES; l += 2) begin
        pair_w = $random;
        x_in[l] = $random;
        x_in[l + 1] = $random;
        w_in[l] = pair_w;
        w_in[l + 1] = $random; // ignored by RTL
        expected[l] = 32'(x_in[l]) * 32'(pair_w);
        expected[l + 1] = 32'(x_in[l + 1]) * 32'(pair_w);
      end
      @(negedge clk);
      en = 0;
      clear_psum = 0;
      repeat(6) @(posedge clk);

      for (int l = 0; l < LANES; l++) begin
        if (psum_out[l] !== expected[l]) begin
          $display("  FAIL test[%0d] lane[%0d]: x=%0d pair_w=%0d got=%0d exp=%0d",
                   t, l, x_in[l], w_in[(l / 2) * 2], psum_out[l], expected[l]);
          errors++;
          if (errors > 10) break;
        end
      end
      if (errors > 10) break;
    end

    if (errors == 0)
      $display("  TEST 4 PASSED");
    else
      $display("  TEST 4 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  initial begin
    clk = 0;
    $display("\n╔══════════════════════════════════════════════════╗");
    $display("║  TESTBENCH: pe_unit (32 lanes)                  ║");
    $display("╚══════════════════════════════════════════════════╝\n");

    test_single_product();
    test_accumulation();
    test_os1_broadcast();
    test_random();

    $display("\n══════════════════════════════════════════════════");
    if (fail_count == 0)
      $display("  ★ ALL PE_UNIT TESTS PASSED ★");
    else
      $display("  ✗ TOTAL FAILURES: %0d", fail_count);
    $display("══════════════════════════════════════════════════\n");
    $finish;
  end

endmodule
