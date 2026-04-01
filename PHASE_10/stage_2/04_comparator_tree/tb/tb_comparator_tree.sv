`timescale 1ns/1ps
module tb_comparator_tree;

  // ──────────────────── Parameters ────────────────────
  localparam int LANES      = 32;
  localparam int NUM_INPUTS = 25;
  localparam int CLK_HP     = 5;  // half-period 5 ns → 10 ns period
  localparam int PIPE_DEPTH = 6;  // 5-stage pipeline → 6 cycles for valid_sr[5]

  // ──────────────────── DUT signals ────────────────────
  logic              clk;
  logic              rst_n;
  logic              en;
  logic signed [7:0] data_in [NUM_INPUTS][LANES];
  logic signed [7:0] max_out [LANES];
  logic              max_valid;

  // ──────────────────── DUT ────────────────────
  comparator_tree #(
    .LANES      (LANES),
    .NUM_INPUTS (NUM_INPUTS)
  ) u_dut (
    .clk      (clk),
    .rst_n    (rst_n),
    .en       (en),
    .data_in  (data_in),
    .max_out  (max_out),
    .max_valid(max_valid)
  );

  // ──────────────────── Clock ────────────────────
  initial clk = 0;
  always #CLK_HP clk = ~clk;

  // ──────────────────── Scoreboard ────────────────────
  int total_tests;
  int pass_count;
  int fail_count;

  task automatic report(string name, int ok);
    total_tests++;
    if (ok) begin
      pass_count++;
      $display("[PASS] %s", name);
    end else begin
      fail_count++;
      $display("[FAIL] %s", name);
    end
  endtask

  // ──────────────────── Helper: drive one vector and wait pipeline ────────────────────
  task automatic drive_and_wait();
    en <= 1'b1;
    @(posedge clk);
    en <= 1'b0;
    // Wait for pipeline flush (5 more cycles for stages 2-5, output at valid_sr[5])
    repeat (PIPE_DEPTH) @(posedge clk);
  endtask

  // ──────────────────── Helper: clear inputs ────────────────────
  task automatic clear_inputs();
    en <= 1'b0;
    for (int i = 0; i < NUM_INPUTS; i++)
      for (int l = 0; l < LANES; l++)
        data_in[i][l] <= 8'sd0;
  endtask

  // ──────────────────── Helper: signed max of 25 values ────────────────────
  function automatic logic signed [7:0] compute_max(
    input logic signed [7:0] vals [NUM_INPUTS]
  );
    automatic logic signed [7:0] m = vals[0];
    for (int i = 1; i < NUM_INPUTS; i++)
      if (vals[i] > m) m = vals[i];
    return m;
  endfunction

  // ──────────────────── Main test sequence ────────────────────
  initial begin
    total_tests = 0;
    pass_count  = 0;
    fail_count  = 0;
    rst_n       = 1'b0;
    en          = 1'b0;
    clear_inputs();

    // Reset
    repeat (4) @(posedge clk);
    rst_n <= 1'b1;
    @(posedge clk);

    // ───── Test 1: All same value (42) → output = 42 ─────
    begin
      string tname = "T1_all_same_42";
      int ok = 1;
      for (int i = 0; i < NUM_INPUTS; i++)
        for (int l = 0; l < LANES; l++)
          data_in[i][l] <= 8'sd42;
      drive_and_wait();
      for (int l = 0; l < LANES; l++)
        if (max_out[l] !== 8'sd42) begin
          $display("  T1 mismatch lane=%0d: got %0d exp 42", l, max_out[l]);
          ok = 0;
        end
      report(tname, ok);
    end

    // ───── Test 2: Only one maximum: input[12]=100, rest=-50 ─────
    begin
      string tname = "T2_single_max_100";
      int ok = 1;
      for (int i = 0; i < NUM_INPUTS; i++)
        for (int l = 0; l < LANES; l++)
          data_in[i][l] <= (i == 12) ? 8'sd100 : -8'sd50;
      drive_and_wait();
      for (int l = 0; l < LANES; l++)
        if (max_out[l] !== 8'sd100) begin
          $display("  T2 mismatch lane=%0d: got %0d exp 100", l, max_out[l]);
          ok = 0;
        end
      report(tname, ok);
    end

    // ───── Test 3: Boundary: contains -128 and 127 → output = 127 ─────
    begin
      string tname = "T3_boundary_min_max";
      int ok = 1;
      for (int i = 0; i < NUM_INPUTS; i++)
        for (int l = 0; l < LANES; l++)
          data_in[i][l] <= -8'sd128;
      // Place 127 at position 0 for all lanes
      for (int l = 0; l < LANES; l++)
        data_in[0][l] <= 8'sd127;
      drive_and_wait();
      for (int l = 0; l < LANES; l++)
        if (max_out[l] !== 8'sd127) begin
          $display("  T3 mismatch lane=%0d: got %0d exp 127", l, max_out[l]);
          ok = 0;
        end
      report(tname, ok);
    end

    // ───── Test 4: All -128 → output = -128 ─────
    begin
      string tname = "T4_all_neg128";
      int ok = 1;
      for (int i = 0; i < NUM_INPUTS; i++)
        for (int l = 0; l < LANES; l++)
          data_in[i][l] <= -8'sd128;
      drive_and_wait();
      for (int l = 0; l < LANES; l++)
        if (max_out[l] !== -8'sd128) begin
          $display("  T4 mismatch lane=%0d: got %0d exp -128", l, max_out[l]);
          ok = 0;
        end
      report(tname, ok);
    end

    // ───── Test 5: Gradient: input[i] = i - 12 (range -12..12) → max = 12 ─────
    begin
      string tname = "T5_gradient";
      int ok = 1;
      for (int i = 0; i < NUM_INPUTS; i++)
        for (int l = 0; l < LANES; l++)
          data_in[i][l] <= 8'(i - 12);
      drive_and_wait();
      for (int l = 0; l < LANES; l++)
        if (max_out[l] !== 8'sd12) begin
          $display("  T5 mismatch lane=%0d: got %0d exp 12", l, max_out[l]);
          ok = 0;
        end
      report(tname, ok);
    end

    // ───── Test 6: Per-lane independence ─────
    // Lane L has max at position (L % 25), rest are -100
    begin
      string tname = "T6_per_lane_independence";
      int ok = 1;
      for (int i = 0; i < NUM_INPUTS; i++)
        for (int l = 0; l < LANES; l++)
          data_in[i][l] <= -8'sd100;
      // Set per-lane max: lane l has max=50+l at position l%25
      for (int l = 0; l < LANES; l++)
        data_in[l % NUM_INPUTS][l] <= 8'(50 + (l % 78));  // keep in INT8 range
      drive_and_wait();
      for (int l = 0; l < LANES; l++) begin
        automatic logic signed [7:0] exp_val = 8'(50 + (l % 78));
        if (max_out[l] !== exp_val) begin
          $display("  T6 mismatch lane=%0d: got %0d exp %0d", l, max_out[l], exp_val);
          ok = 0;
        end
      end
      report(tname, ok);
    end

    // ───── Test 7: Random stress (200 iterations) ─────
    begin
      string tname = "T7_random_stress";
      int ok = 1;
      for (int iter = 0; iter < 200; iter++) begin
        // Generate random inputs and compute expected max per lane
        automatic logic signed [7:0] exp_max [LANES];
        for (int l = 0; l < LANES; l++)
          exp_max[l] = -8'sd128;
        for (int i = 0; i < NUM_INPUTS; i++)
          for (int l = 0; l < LANES; l++) begin
            automatic logic signed [7:0] v = $random;
            data_in[i][l] <= v;
            if (v > exp_max[l]) exp_max[l] = v;
          end
        drive_and_wait();
        for (int l = 0; l < LANES; l++)
          if (max_out[l] !== exp_max[l]) begin
            if (ok)
              $display("  T7 iter=%0d mismatch lane=%0d: got %0d exp %0d",
                       iter, l, max_out[l], exp_max[l]);
            ok = 0;
          end
      end
      report(tname, ok);
    end

    // ──────────────────── Final Summary ────────────────────
    $display("==============================================");
    $display("  comparator_tree TB: %0d / %0d PASSED", pass_count, total_tests);
    if (fail_count == 0)
      $display("  RESULT: ALL TESTS PASSED");
    else
      $display("  RESULT: %0d TEST(S) FAILED", fail_count);
    $display("==============================================");
    $finish;
  end

endmodule
