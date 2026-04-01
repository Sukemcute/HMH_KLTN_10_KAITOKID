`timescale 1ns/1ps

module tb_pe_unit;

  import accel_pkg::*;

  // ----------------------------------------------------------------
  // Parameters
  // ----------------------------------------------------------------
  localparam int LANES     = 32;
  localparam int CLK_HALF  = 5;          // 5 ns half-period → 10 ns period
  localparam int PIPE_LAT  = 4;          // DSP pair pipeline latency (cycles)
  localparam int VALID_LAT = 5;          // psum_valid latency (valid_sr[4])

  // ----------------------------------------------------------------
  // DUT signals
  // ----------------------------------------------------------------
  logic              clk;
  logic              rst_n;
  logic              en;
  logic              clear_psum;
  pe_mode_e          mode;

  logic signed [7:0]  x_in  [LANES];
  logic signed [7:0]  w_in  [LANES];

  logic signed [31:0] psum_out [LANES];
  logic               psum_valid;

  // ----------------------------------------------------------------
  // DUT instantiation
  // ----------------------------------------------------------------
  pe_unit #(.LANES(LANES)) u_dut (
    .clk        (clk),
    .rst_n      (rst_n),
    .en         (en),
    .clear_psum (clear_psum),
    .mode       (mode),
    .x_in       (x_in),
    .w_in       (w_in),
    .psum_out   (psum_out),
    .psum_valid (psum_valid)
  );

  // ----------------------------------------------------------------
  // Clock generation
  // ----------------------------------------------------------------
  initial clk = 0;
  always #(CLK_HALF) clk = ~clk;

  // ----------------------------------------------------------------
  // Scoreboard counters
  // ----------------------------------------------------------------
  int tests_passed = 0;
  int tests_failed = 0;
  int total_tests  = 0;

  // ----------------------------------------------------------------
  // Helper tasks
  // ----------------------------------------------------------------

  // Wait N rising edges
  task automatic wait_clk(int n);
    repeat (n) @(posedge clk);
  endtask

  // Apply reset
  task automatic apply_reset();
    rst_n      = 0;
    en         = 0;
    clear_psum = 0;
    mode       = PE_RS3;
    for (int l = 0; l < LANES; l++) begin
      x_in[l] = 0;
      w_in[l] = 0;
    end
    wait_clk(5);
    rst_n = 1;
    wait_clk(2);
  endtask

  // Drive all inputs to zero and disable
  task automatic idle_inputs();
    en         = 0;
    clear_psum = 0;
    for (int l = 0; l < LANES; l++) begin
      x_in[l] = 0;
      w_in[l] = 0;
    end
  endtask

  // Effective weight for a given lane as seen by the DSP pair.
  // The DSP pair at group g gets w_sel[2*g]. So both lanes 2g and 2g+1
  // use the weight that was routed to the even lane of their pair.
  function automatic logic signed [7:0] effective_weight(
    int             lane,
    pe_mode_e       m,
    logic signed [7:0] w_array [LANES]
  );
    if (m == PE_OS1)
      return w_array[0];
    else
      return w_array[(lane / 2) * 2];  // w_sel[2*g] where g = lane/2
  endfunction

  // Golden single-cycle product: x * w_effective (signed 8×8 → 32-bit)
  function automatic logic signed [31:0] golden_product(
    logic signed [7:0] x_val,
    logic signed [7:0] w_val
  );
    return 32'(x_val) * 32'(w_val);
  endfunction

  // ================================================================
  // TEST 1: PE_RS3 mode — Per-lane single MAC
  // ================================================================
  task automatic test1_rs3_per_lane();
    string test_name = "TEST 1: PE_RS3 per-lane MAC";
    int    err_cnt   = 0;
    logic signed [31:0] expected;

    $display("\n========== %s ==========", test_name);
    apply_reset();

    mode = PE_RS3;

    // Drive: x_in[l] = l, w_in[l] = 1, clear=1, en=1 for 1 cycle
    @(posedge clk);
    en         = 1;
    clear_psum = 1;
    for (int l = 0; l < LANES; l++) begin
      x_in[l] = 8'(l);
      w_in[l] = 8'sd1;
    end

    @(posedge clk);
    idle_inputs();

    // Wait for pipeline to drain
    wait_clk(PIPE_LAT + 1);

    // Check outputs: x=l, w_eff=1 for all lanes → expected = l * 1 = l
    for (int l = 0; l < LANES; l++) begin
      expected = 32'(l);  // l * 1 = l (l ranges 0..31, all positive)
      if (psum_out[l] !== expected) begin
        $display("  [FAIL] Lane %0d: got %0d, expected %0d", l, psum_out[l], expected);
        err_cnt++;
      end
    end

    total_tests++;
    if (err_cnt == 0) begin
      $display("  [PASS] %s", test_name);
      tests_passed++;
    end else begin
      $display("  [FAIL] %s — %0d lane errors", test_name, err_cnt);
      tests_failed++;
    end
  endtask

  // ================================================================
  // TEST 2: PE_RS3 mode — 9-cycle accumulation (conv3x3 simulation)
  // ================================================================
  task automatic test2_rs3_9cycle_accum();
    string test_name = "TEST 2: PE_RS3 9-cycle accumulation";
    int    err_cnt   = 0;
    logic signed [7:0]  x_vals [9][LANES];
    logic signed [7:0]  w_vals [9][LANES];
    logic signed [31:0] golden [LANES];
    logic signed [7:0]  w_eff;

    $display("\n========== %s ==========", test_name);
    apply_reset();

    mode = PE_RS3;

    // Pre-compute 9 cycles of known activations and weights
    // Simulate kw=0..2 x cin=0..2 (3x3 kernel)
    for (int i = 0; i < 9; i++) begin
      for (int l = 0; l < LANES; l++) begin
        x_vals[i][l] = 8'((l + i * 3 + 1) % 120);       // vary per lane and cycle
        w_vals[i][l] = 8'(((l / 2) * 2 + i + 1) % 100); // weight per DSP-pair group
      end
    end

    // Compute golden accumulated results
    for (int l = 0; l < LANES; l++) begin
      golden[l] = 0;
      for (int i = 0; i < 9; i++) begin
        // Effective weight: w_vals[i][(l/2)*2]
        w_eff = w_vals[i][(l / 2) * 2];
        golden[l] += golden_product(x_vals[i][l], w_eff);
      end
    end

    // Drive 9 cycles: clear=1 only on first cycle
    for (int i = 0; i < 9; i++) begin
      @(posedge clk);
      en         = 1;
      clear_psum = (i == 0) ? 1'b1 : 1'b0;
      for (int l = 0; l < LANES; l++) begin
        x_in[l] = x_vals[i][l];
        w_in[l] = w_vals[i][l];
      end
    end

    @(posedge clk);
    idle_inputs();

    // Wait for the last input to fully propagate
    wait_clk(PIPE_LAT + 1);

    // Verify
    for (int l = 0; l < LANES; l++) begin
      if (psum_out[l] !== golden[l]) begin
        $display("  [FAIL] Lane %0d: got %0d, expected %0d", l, psum_out[l], golden[l]);
        err_cnt++;
      end
    end

    total_tests++;
    if (err_cnt == 0) begin
      $display("  [PASS] %s", test_name);
      tests_passed++;
    end else begin
      $display("  [FAIL] %s — %0d lane errors", test_name, err_cnt);
      tests_failed++;
    end
  endtask

  // ================================================================
  // TEST 3: PE_OS1 mode — Weight broadcast
  // ================================================================
  task automatic test3_os1_broadcast();
    string test_name = "TEST 3: PE_OS1 weight broadcast";
    int    err_cnt   = 0;
    logic signed [31:0] expected;

    $display("\n========== %s ==========", test_name);
    apply_reset();

    mode = PE_OS1;

    @(posedge clk);
    en         = 1;
    clear_psum = 1;
    for (int l = 0; l < LANES; l++) begin
      x_in[l] = 8'(l);
      w_in[l] = (l == 0) ? 8'sd5 : 8'sd99;  // only w_in[0] should matter
    end

    @(posedge clk);
    idle_inputs();

    wait_clk(PIPE_LAT + 1);

    // Expected: psum_out[l] = l * 5 (w_in[0]=5 broadcast to all)
    for (int l = 0; l < LANES; l++) begin
      expected = 32'(8'(l)) * 32'(8'sd5);
      if (psum_out[l] !== expected) begin
        $display("  [FAIL] Lane %0d: got %0d, expected %0d", l, psum_out[l], expected);
        err_cnt++;
      end
    end

    total_tests++;
    if (err_cnt == 0) begin
      $display("  [PASS] %s", test_name);
      tests_passed++;
    end else begin
      $display("  [FAIL] %s — %0d lane errors", test_name, err_cnt);
      tests_failed++;
    end
  endtask

  // ================================================================
  // TEST 4: PE_DW3 mode — Per-channel weight
  // ================================================================
  task automatic test4_dw3_per_channel();
    string test_name = "TEST 4: PE_DW3 per-channel weight";
    int    err_cnt   = 0;
    logic signed [31:0] expected;
    logic signed [7:0]  w_eff;

    $display("\n========== %s ==========", test_name);
    apply_reset();

    mode = PE_DW3;

    @(posedge clk);
    en         = 1;
    clear_psum = 1;
    for (int l = 0; l < LANES; l++) begin
      x_in[l] = 8'sd10;
      w_in[l] = 8'(l);
    end

    @(posedge clk);
    idle_inputs();

    wait_clk(PIPE_LAT + 1);

    // Expected: psum_out[l] = 10 * w_eff, where w_eff = w_in[(l/2)*2]
    // Because DSP pair shares w_sel[2*g] for both lanes 2g, 2g+1
    for (int l = 0; l < LANES; l++) begin
      w_eff    = 8'((l / 2) * 2);  // w_in[(l/2)*2] = (l/2)*2
      expected = 32'(8'sd10) * 32'(w_eff);
      if (psum_out[l] !== expected) begin
        $display("  [FAIL] Lane %0d: got %0d, expected %0d (w_eff=%0d)",
                 l, psum_out[l], expected, w_eff);
        err_cnt++;
      end
    end

    total_tests++;
    if (err_cnt == 0) begin
      $display("  [PASS] %s", test_name);
      tests_passed++;
    end else begin
      $display("  [FAIL] %s — %0d lane errors", test_name, err_cnt);
      tests_failed++;
    end
  endtask

  // ================================================================
  // TEST 5: Multi-tile — clear between tiles
  // ================================================================
  task automatic test5_multi_tile_clear();
    string test_name = "TEST 5: Multi-tile clear between tiles";
    int    err_cnt   = 0;
    logic signed [31:0] golden_b [LANES];
    logic signed [7:0]  x_val;

    $display("\n========== %s ==========", test_name);
    apply_reset();

    mode = PE_RS3;

    // --- Tile A: accumulate 3 cycles ---
    for (int i = 0; i < 3; i++) begin
      @(posedge clk);
      en         = 1;
      clear_psum = (i == 0) ? 1'b1 : 1'b0;
      for (int l = 0; l < LANES; l++) begin
        x_in[l] = 8'(l + i + 1);
        w_in[l] = 8'sd2;
      end
    end

    // Wait for tile A to drain pipeline before starting tile B
    @(posedge clk);
    idle_inputs();
    wait_clk(PIPE_LAT + 2);

    // --- Tile B: accumulate 3 cycles with clear on first ---
    for (int i = 0; i < 3; i++) begin
      @(posedge clk);
      en         = 1;
      clear_psum = (i == 0) ? 1'b1 : 1'b0;
      for (int l = 0; l < LANES; l++) begin
        x_in[l] = 8'(l + 10);
        w_in[l] = 8'sd3;
      end
    end

    @(posedge clk);
    idle_inputs();

    // Compute golden for tile B only
    // All 3 cycles have same x = l+10, w_eff = 3 (uniform weight → same for all pairs)
    // Accumulation: clear on first cycle → psum = prod; then += prod twice more
    // golden = 3 * (x_val * 3)
    for (int l = 0; l < LANES; l++) begin
      x_val = 8'(l + 10);  // signed 8-bit cast
      golden_b[l] = 3 * (32'(x_val) * 32'(8'sd3));
    end

    wait_clk(PIPE_LAT + 1);

    for (int l = 0; l < LANES; l++) begin
      if (psum_out[l] !== golden_b[l]) begin
        $display("  [FAIL] Lane %0d: got %0d, expected %0d", l, psum_out[l], golden_b[l]);
        err_cnt++;
      end
    end

    total_tests++;
    if (err_cnt == 0) begin
      $display("  [PASS] %s", test_name);
      tests_passed++;
    end else begin
      $display("  [FAIL] %s — %0d lane errors", test_name, err_cnt);
      tests_failed++;
    end
  endtask

  // ================================================================
  // TEST 6: Pipeline timing verification
  // ================================================================
  task automatic test6_pipeline_timing();
    string test_name = "TEST 6: Pipeline timing (psum_valid latency)";
    int    err_cnt   = 0;
    int    en_cycle;
    int    valid_cycle;
    bit    found_valid;

    $display("\n========== %s ==========", test_name);
    apply_reset();

    mode = PE_RS3;

    // Drive en=1 for exactly 1 cycle
    @(posedge clk);
    en_cycle = 0;
    en         = 1;
    clear_psum = 1;
    for (int l = 0; l < LANES; l++) begin
      x_in[l] = 8'sd1;
      w_in[l] = 8'sd1;
    end

    @(posedge clk);
    idle_inputs();

    // Monitor psum_valid for up to 10 cycles
    found_valid = 0;
    for (int c = 1; c <= 10; c++) begin
      @(posedge clk);
      if (psum_valid && !found_valid) begin
        valid_cycle = c;
        found_valid = 1;
      end
    end

    total_tests++;
    if (!found_valid) begin
      $display("  [FAIL] %s — psum_valid never asserted", test_name);
      err_cnt++;
      tests_failed++;
    end else if (valid_cycle !== VALID_LAT) begin
      $display("  [FAIL] %s — psum_valid at cycle %0d, expected %0d",
               test_name, valid_cycle, VALID_LAT);
      err_cnt++;
      tests_failed++;
    end else begin
      $display("  [PASS] %s — psum_valid at cycle %0d (correct)", test_name, valid_cycle);
      tests_passed++;
    end

    // Also verify psum_valid is high for exactly 1 cycle
    // (since en was high for exactly 1 cycle, valid_sr should produce 1 pulse)
    begin
      int high_count = 0;
      apply_reset();
      mode = PE_RS3;

      @(posedge clk);
      en         = 1;
      clear_psum = 1;
      for (int l = 0; l < LANES; l++) begin
        x_in[l] = 8'sd1;
        w_in[l] = 8'sd1;
      end

      @(posedge clk);
      idle_inputs();

      for (int c = 0; c < 12; c++) begin
        @(posedge clk);
        if (psum_valid) high_count++;
      end

      total_tests++;
      if (high_count !== 1) begin
        $display("  [FAIL] psum_valid pulse width: %0d cycles, expected 1", high_count);
        tests_failed++;
      end else begin
        $display("  [PASS] psum_valid pulse width: 1 cycle (correct)");
        tests_passed++;
      end
    end
  endtask

  // ================================================================
  // TEST 7: Random 32-lane stress
  // ================================================================
  task automatic test7_random_stress();
    string test_name = "TEST 7: Random 32-lane stress";
    int    err_cnt   = 0;
    int    iter_errs;
    logic signed [7:0]  rand_x [LANES];
    logic signed [7:0]  rand_w [LANES];
    logic signed [7:0]  w_eff;
    logic signed [31:0] expected;
    int seed;

    $display("\n========== %s ==========", test_name);

    seed = 42;

    for (int iter = 0; iter < 100; iter++) begin
      apply_reset();
      mode = PE_RS3;
      iter_errs = 0;

      // Generate random vectors
      for (int l = 0; l < LANES; l++) begin
        rand_x[l] = 8'($random(seed));
        rand_w[l] = 8'($random(seed));
      end

      // Drive single MAC cycle
      @(posedge clk);
      en         = 1;
      clear_psum = 1;
      for (int l = 0; l < LANES; l++) begin
        x_in[l] = rand_x[l];
        w_in[l] = rand_w[l];
      end

      @(posedge clk);
      idle_inputs();

      wait_clk(PIPE_LAT + 1);

      // Verify all 32 lanes
      for (int l = 0; l < LANES; l++) begin
        w_eff    = rand_w[(l / 2) * 2];
        expected = golden_product(rand_x[l], w_eff);
        if (psum_out[l] !== expected) begin
          if (iter_errs < 3)  // limit prints per iteration
            $display("  [FAIL] Iter %0d, Lane %0d: got %0d, expected %0d (x=%0d, w_eff=%0d)",
                     iter, l, psum_out[l], expected, rand_x[l], w_eff);
          iter_errs++;
        end
      end

      if (iter_errs > 0) err_cnt++;
    end

    total_tests++;
    if (err_cnt == 0) begin
      $display("  [PASS] %s — 100 iterations, all 32 lanes correct", test_name);
      tests_passed++;
    end else begin
      $display("  [FAIL] %s — %0d / 100 iterations had errors", test_name, err_cnt);
      tests_failed++;
    end
  endtask

  // ================================================================
  // Main test sequence
  // ================================================================
  initial begin
    $display("==========================================================");
    $display("  tb_pe_unit — Comprehensive Testbench");
    $display("==========================================================");

    test1_rs3_per_lane();
    test2_rs3_9cycle_accum();
    test3_os1_broadcast();
    test4_dw3_per_channel();
    test5_multi_tile_clear();
    test6_pipeline_timing();
    test7_random_stress();

    $display("\n==========================================================");
    $display("  SUMMARY: %0d / %0d tests PASSED", tests_passed, total_tests);
    if (tests_failed > 0)
      $display("           %0d / %0d tests FAILED", tests_failed, total_tests);
    else
      $display("           ALL TESTS PASSED");
    $display("==========================================================\n");

    $finish;
  end

  // Timeout watchdog
  initial begin
    #500000;
    $display("\n[TIMEOUT] Simulation exceeded 500 us — aborting.");
    $finish;
  end

endmodule
