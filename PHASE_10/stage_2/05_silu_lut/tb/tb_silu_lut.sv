`timescale 1ns/1ps
module tb_silu_lut;

  // ──────────────────── Parameters ────────────────────
  localparam int LANES     = 32;
  localparam int LUT_DEPTH = 256;
  localparam int CLK_HP    = 5;  // half-period 5 ns → 10 ns period

  // ──────────────────── DUT signals ────────────────────
  logic              clk;
  logic              load_en;
  logic [7:0]        load_addr;
  logic signed [7:0] load_data;
  logic [7:0]        idx [LANES];
  logic signed [7:0] out [LANES];

  // ──────────────────── DUT ────────────────────
  silu_lut #(
    .LANES(LANES)
  ) u_dut (
    .clk      (clk),
    .load_en  (load_en),
    .load_addr(load_addr),
    .load_data(load_data),
    .idx      (idx),
    .out      (out)
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

  // ──────────────────── Helper: load entire LUT table ────────────────────
  task automatic load_table(input logic signed [7:0] table [LUT_DEPTH]);
    for (int i = 0; i < LUT_DEPTH; i++) begin
      load_en   <= 1'b1;
      load_addr <= 8'(i);
      load_data <= table[i];
      @(posedge clk);
    end
    load_en <= 1'b0;
    @(posedge clk);
  endtask

  // ──────────────────── Helper: read one set of indices ────────────────────
  // Drive idx, wait 1 cycle for registered output, then sample
  task automatic read_and_wait();
    @(posedge clk);  // idx is sampled on this edge
    @(posedge clk);  // output register has result
  endtask

  // ──────────────────── SiLU golden computation ────────────────────
  // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
  // scale = 0.1: input = (i - 128) * 0.1, output = clamp(round(SiLU(input) / 0.1), -128, 127)
  // Actually: lut[i] = clamp(round(SiLU((i-128)*scale) * scale_inv), -128, 127)
  // with scale = 0.1, scale_inv = 1/0.1 = 10
  function automatic logic signed [7:0] silu_golden(int i);
    real x, sx, y;
    int  yi;
    x  = real'(i - 128) * 0.1;
    sx = x / (1.0 + $exp(-x));
    y  = sx * 10.0;  // scale_inv = 1/scale = 10
    // Round to nearest
    if (y >= 0.0)
      yi = int'(y + 0.5);
    else
      yi = int'(y - 0.5);
    // Clamp to INT8
    if (yi > 127)  yi = 127;
    if (yi < -128) yi = -128;
    return 8'(yi);
  endfunction

  // ──────────────────── Main test sequence ────────────────────
  initial begin
    total_tests = 0;
    pass_count  = 0;
    fail_count  = 0;
    load_en     = 1'b0;
    load_addr   = 8'd0;
    load_data   = 8'sd0;
    for (int l = 0; l < LANES; l++)
      idx[l] = 8'd0;

    repeat (4) @(posedge clk);

    // ───── Test 1: Load and verify — load known table, read back every entry ─────
    begin
      string tname = "T1_load_and_verify";
      int ok = 1;
      automatic logic signed [7:0] test_table [LUT_DEPTH];

      // Create a known table: table[i] = i - 128 (full INT8 range)
      for (int i = 0; i < LUT_DEPTH; i++)
        test_table[i] = 8'(i - 128);

      load_table(test_table);

      // Read back all 256 entries, 32 at a time (8 batches)
      for (int batch = 0; batch < 8; batch++) begin
        for (int l = 0; l < LANES; l++)
          idx[l] <= 8'(batch * 32 + l);
        read_and_wait();
        for (int l = 0; l < LANES; l++) begin
          automatic int addr = batch * 32 + l;
          if (out[l] !== test_table[addr]) begin
            $display("  T1 mismatch addr=%0d: got %0d exp %0d",
                     addr, out[l], test_table[addr]);
            ok = 0;
          end
        end
      end
      report(tname, ok);
    end

    // ───── Test 2: Parallel reads — 32 different indices simultaneously ─────
    begin
      string tname = "T2_parallel_reads";
      int ok = 1;
      // Use the table already loaded (table[i] = i - 128)
      // Pick spread-out indices: lane l reads index l*8 (mod 256)
      for (int l = 0; l < LANES; l++)
        idx[l] <= 8'((l * 8) % 256);
      read_and_wait();
      for (int l = 0; l < LANES; l++) begin
        automatic int addr = (l * 8) % 256;
        automatic logic signed [7:0] exp_val = 8'(addr - 128);
        if (out[l] !== exp_val) begin
          $display("  T2 mismatch lane=%0d idx=%0d: got %0d exp %0d",
                   l, addr, out[l], exp_val);
          ok = 0;
        end
      end
      report(tname, ok);
    end

    // ───── Test 3: SiLU golden values ─────
    begin
      string tname = "T3_silu_golden";
      int ok = 1;
      automatic logic signed [7:0] silu_table [LUT_DEPTH];

      // Compute golden SiLU table
      for (int i = 0; i < LUT_DEPTH; i++)
        silu_table[i] = silu_golden(i);

      // Load the SiLU table
      load_table(silu_table);

      // Read back all 256 entries and verify
      for (int batch = 0; batch < 8; batch++) begin
        for (int l = 0; l < LANES; l++)
          idx[l] <= 8'(batch * 32 + l);
        read_and_wait();
        for (int l = 0; l < LANES; l++) begin
          automatic int addr = batch * 32 + l;
          automatic logic signed [7:0] exp_val = silu_table[addr];
          if (out[l] !== exp_val) begin
            $display("  T3 mismatch addr=%0d: got %0d exp %0d",
                     addr, out[l], exp_val);
            ok = 0;
          end
        end
      end
      report(tname, ok);
    end

    // ───── Test 4: Boundary — idx=0 and idx=255 ─────
    begin
      string tname = "T4_boundary";
      int ok = 1;
      // The SiLU table is still loaded from Test 3
      automatic logic signed [7:0] silu_table [LUT_DEPTH];
      for (int i = 0; i < LUT_DEPTH; i++)
        silu_table[i] = silu_golden(i);

      // All lanes read idx=0
      for (int l = 0; l < LANES; l++)
        idx[l] <= 8'd0;
      read_and_wait();
      for (int l = 0; l < LANES; l++)
        if (out[l] !== silu_table[0]) begin
          $display("  T4 mismatch idx=0 lane=%0d: got %0d exp %0d",
                   l, out[l], silu_table[0]);
          ok = 0;
        end

      // All lanes read idx=255
      for (int l = 0; l < LANES; l++)
        idx[l] <= 8'd255;
      read_and_wait();
      for (int l = 0; l < LANES; l++)
        if (out[l] !== silu_table[255]) begin
          $display("  T4 mismatch idx=255 lane=%0d: got %0d exp %0d",
                   l, out[l], silu_table[255]);
          ok = 0;
        end
      report(tname, ok);
    end

    // ───── Test 5: All-same index — all 32 lanes read same index ─────
    begin
      string tname = "T5_all_same_index";
      int ok = 1;
      automatic logic signed [7:0] silu_table [LUT_DEPTH];
      for (int i = 0; i < LUT_DEPTH; i++)
        silu_table[i] = silu_golden(i);

      // Test with several indices
      for (int test_idx = 0; test_idx < 256; test_idx += 37) begin
        for (int l = 0; l < LANES; l++)
          idx[l] <= 8'(test_idx);
        read_and_wait();
        // All outputs should be identical
        for (int l = 0; l < LANES; l++)
          if (out[l] !== silu_table[test_idx]) begin
            $display("  T5 mismatch idx=%0d lane=%0d: got %0d exp %0d",
                     test_idx, l, out[l], silu_table[test_idx]);
            ok = 0;
          end
        // Also verify all lanes match lane 0
        for (int l = 1; l < LANES; l++)
          if (out[l] !== out[0]) begin
            $display("  T5 lane mismatch idx=%0d: lane0=%0d lane%0d=%0d",
                     test_idx, out[0], l, out[l]);
            ok = 0;
          end
      end
      report(tname, ok);
    end

    // ──────────────────── Final Summary ────────────────────
    $display("==============================================");
    $display("  silu_lut TB: %0d / %0d PASSED", pass_count, total_tests);
    if (fail_count == 0)
      $display("  RESULT: ALL TESTS PASSED");
    else
      $display("  RESULT: %0d TEST(S) FAILED", fail_count);
    $display("==============================================");
    $finish;
  end

endmodule
