`timescale 1ns/1ps
// ============================================================================
// Golden Verification Testbench for conv1x1_engine (P1)
//
// Strategy:
//   1. Populate IFM/WGT SRAMs and bias/quant arrays with known data
//   2. Pulse start → wait for done
//   3. Compare OFM SRAM contents vs golden reference computed in-TB
//
// Test scenarios:
//   TEST 1 — Minimal:          Cin=1,  Cout=1,  H=1, W=32
//   TEST 2 — QC2f cv1 style:   Cin=32, Cout=32, H=4, W=64
//   TEST 3 — Channel reduction: Cin=64, Cout=16, H=2, W=32
//   TEST 4 — Random stress:     Cin=16, Cout=16, H=4, W=64, ACT_RELU
// ============================================================================
module tb_conv1x1_golden;

  import yolo_accel_pkg::*;

  // ══════════════════════════════════════════════════════════════════════════
  // Parameters
  // ══════════════════════════════════════════════════════════════════════════
  localparam int TB_LANES     = 32;
  localparam int CLK_PERIOD   = 10;   // 5ns half-period → 10ns period
  localparam int MAX_IFM_WORDS = 512 * 1024;  // Large enough for all tests
  localparam int MAX_WGT_BYTES = 64 * 1024;
  localparam int MAX_OFM_WORDS = 512 * 1024;

  // ══════════════════════════════════════════════════════════════════════════
  // Clock / Reset
  // ══════════════════════════════════════════════════════════════════════════
  logic clk;
  logic rst_n;

  initial clk = 0;
  always #(CLK_PERIOD/2) clk = ~clk;

  // ══════════════════════════════════════════════════════════════════════════
  // DUT Signals
  // ══════════════════════════════════════════════════════════════════════════
  logic        start, done, busy;
  logic [9:0]  cfg_h, cfg_w;
  logic [8:0]  cfg_cin, cfg_cout;
  act_mode_e   cfg_act_mode;
  logic signed [7:0] cfg_zp_out;

  // IFM SRAM interface
  logic [23:0]       ifm_rd_addr;
  logic              ifm_rd_en;
  logic signed [7:0] ifm_rd_data [TB_LANES];

  // WGT SRAM interface
  logic [23:0]       wgt_rd_addr;
  logic              wgt_rd_en;
  logic signed [7:0] wgt_rd_data;

  // Bias & quant arrays
  logic signed [31:0] bias_arr  [MAX_COUT];
  logic signed [31:0] m_int_arr [MAX_COUT];
  logic [5:0]         shift_arr [MAX_COUT];

  // SiLU LUT
  logic signed [7:0]  silu_lut [256];

  // OFM SRAM interface
  logic [23:0]       ofm_wr_addr;
  logic              ofm_wr_en;
  logic signed [7:0] ofm_wr_data [TB_LANES];

  // ══════════════════════════════════════════════════════════════════════════
  // Behavioral SRAM Models
  // ══════════════════════════════════════════════════════════════════════════

  // IFM: [addr] → LANES bytes (1-cycle read latency)
  logic signed [7:0] ifm_mem [MAX_IFM_WORDS][TB_LANES];

  always_ff @(posedge clk) begin
    if (ifm_rd_en) begin
      for (int l = 0; l < TB_LANES; l++)
        ifm_rd_data[l] <= ifm_mem[ifm_rd_addr][l];
    end
  end

  // WGT: [addr] → 1 byte (1-cycle read latency)
  logic signed [7:0] wgt_mem [MAX_WGT_BYTES];

  always_ff @(posedge clk) begin
    if (wgt_rd_en)
      wgt_rd_data <= wgt_mem[wgt_rd_addr];
  end

  // OFM: capture writes
  logic signed [7:0] ofm_mem [MAX_OFM_WORDS][TB_LANES];

  always_ff @(posedge clk) begin
    if (ofm_wr_en) begin
      for (int l = 0; l < TB_LANES; l++)
        ofm_mem[ofm_wr_addr][l] <= ofm_wr_data[l];
    end
  end

  // ══════════════════════════════════════════════════════════════════════════
  // DUT Instantiation
  // ══════════════════════════════════════════════════════════════════════════
  conv1x1_engine #(.LANES(TB_LANES)) u_dut (
    .clk          (clk),
    .rst_n        (rst_n),
    .start        (start),
    .done         (done),
    .busy         (busy),
    .cfg_h        (cfg_h),
    .cfg_w        (cfg_w),
    .cfg_cin      (cfg_cin),
    .cfg_cout     (cfg_cout),
    .cfg_act_mode (cfg_act_mode),
    .cfg_zp_out   (cfg_zp_out),
    .ifm_rd_addr  (ifm_rd_addr),
    .ifm_rd_en    (ifm_rd_en),
    .ifm_rd_data  (ifm_rd_data),
    .wgt_rd_addr  (wgt_rd_addr),
    .wgt_rd_en    (wgt_rd_en),
    .wgt_rd_data  (wgt_rd_data),
    .bias_arr     (bias_arr),
    .m_int_arr    (m_int_arr),
    .shift_arr    (shift_arr),
    .silu_lut     (silu_lut),
    .ofm_wr_addr  (ofm_wr_addr),
    .ofm_wr_en    (ofm_wr_en),
    .ofm_wr_data  (ofm_wr_data)
  );

  // ══════════════════════════════════════════════════════════════════════════
  // Golden Reference Model (pure software computation)
  // ══════════════════════════════════════════════════════════════════════════
  logic signed [7:0] golden_ofm [MAX_OFM_WORDS][TB_LANES];

  task automatic compute_golden(
    input int H, W, Cin, Cout,
    input act_mode_e act_mode,
    input logic signed [7:0] zp_out
  );
    int num_wblk_g;
    num_wblk_g = (W + TB_LANES - 1) / TB_LANES;

    for (int h = 0; h < H; h++) begin
      for (int co = 0; co < Cout; co++) begin
        for (int wb = 0; wb < num_wblk_g; wb++) begin
          for (int l = 0; l < TB_LANES; l++) begin
            automatic longint acc;
            automatic longint biased;
            automatic longint mult;
            automatic longint rounded;
            automatic int     requanted;
            automatic int     act_in;
            automatic int     act_val;
            automatic int     final_val;

            // ── Accumulate across all input channels ──
            acc = 0;
            for (int ci = 0; ci < Cin; ci++) begin
              automatic int ifm_addr;
              automatic int wgt_addr;
              ifm_addr = h * Cin * num_wblk_g + ci * num_wblk_g + wb;
              wgt_addr = co * Cin + ci;
              acc = acc + longint'(ifm_mem[ifm_addr][l]) * longint'(wgt_mem[wgt_addr]);
            end

            // ── Bias add ──
            biased = acc + longint'(bias_arr[co]);

            // ── Requantization: (biased * m_int + round) >>> shift ──
            mult = biased * longint'(m_int_arr[co]);
            if (shift_arr[co] > 0)
              rounded = mult + (longint'(1) <<< (shift_arr[co] - 1));
            else
              rounded = mult;
            requanted = int'(rounded >>> shift_arr[co]);

            // ── Clamp to 16-bit for activation ──
            if (requanted > 32767)
              act_in = 32767;
            else if (requanted < -32768)
              act_in = -32768;
            else
              act_in = requanted;

            // ── Activation ──
            case (act_mode)
              ACT_SILU: begin
                automatic int idx;
                idx = act_in + 128;
                if (idx < 0)   idx = 0;
                if (idx > 255) idx = 255;
                act_val = int'(silu_lut[idx]);
              end
              ACT_RELU: begin
                if (act_in > 0) begin
                  if (act_in > 127) act_val = 127;
                  else if (act_in < -128) act_val = -128;
                  else act_val = act_in;
                end else begin
                  act_val = 0;
                end
              end
              default: begin  // ACT_NONE, ACT_CLAMP
                if (act_in > 127) act_val = 127;
                else if (act_in < -128) act_val = -128;
                else act_val = act_in;
              end
            endcase

            // ── Output zero-point + final clamp ──
            final_val = act_val + int'(zp_out);
            if (final_val > 127)
              golden_ofm[h * Cout * num_wblk_g + co * num_wblk_g + wb][l] = 8'sd127;
            else if (final_val < -128)
              golden_ofm[h * Cout * num_wblk_g + co * num_wblk_g + wb][l] = -8'sd128;
            else
              golden_ofm[h * Cout * num_wblk_g + co * num_wblk_g + wb][l] = final_val[7:0];
          end
        end
      end
    end
  endtask

  // ══════════════════════════════════════════════════════════════════════════
  // Comparison & Reporting
  // ══════════════════════════════════════════════════════════════════════════
  function automatic int compare_ofm(
    input int H, W, Cout,
    input string test_name
  );
    int num_wblk_c;
    int mismatches;
    int addr;
    num_wblk_c = (W + TB_LANES - 1) / TB_LANES;
    mismatches = 0;

    for (int h = 0; h < H; h++) begin
      for (int co = 0; co < Cout; co++) begin
        for (int wb = 0; wb < num_wblk_c; wb++) begin
          addr = h * Cout * num_wblk_c + co * num_wblk_c + wb;
          for (int l = 0; l < TB_LANES; l++) begin
            if (ofm_mem[addr][l] !== golden_ofm[addr][l]) begin
              mismatches++;
              if (mismatches <= 20) begin
                $display("  MISMATCH [%s] h=%0d co=%0d wb=%0d lane=%0d : DUT=%0d GOLD=%0d (addr=%0d)",
                  test_name, h, co, wb, l,
                  ofm_mem[addr][l], golden_ofm[addr][l], addr);
              end
            end
          end
        end
      end
    end

    return mismatches;
  endfunction

  // ══════════════════════════════════════════════════════════════════════════
  // Helper Tasks
  // ══════════════════════════════════════════════════════════════════════════
  task automatic reset_dut();
    rst_n <= 1'b0;
    start <= 1'b0;
    repeat (5) @(posedge clk);
    rst_n <= 1'b1;
    repeat (2) @(posedge clk);
  endtask

  task automatic clear_memories();
    for (int i = 0; i < MAX_IFM_WORDS; i++)
      for (int l = 0; l < TB_LANES; l++)
        ifm_mem[i][l] = 8'sd0;
    for (int i = 0; i < MAX_WGT_BYTES; i++)
      wgt_mem[i] = 8'sd0;
    for (int i = 0; i < MAX_OFM_WORDS; i++)
      for (int l = 0; l < TB_LANES; l++) begin
        ofm_mem[i][l]    = 8'sd0;
        golden_ofm[i][l] = 8'sd0;
      end
    for (int i = 0; i < MAX_COUT; i++) begin
      bias_arr[i]  = 32'sd0;
      m_int_arr[i] = 32'sd1;
      shift_arr[i] = 6'd0;
    end
  endtask

  task automatic run_engine();
    @(posedge clk);
    start <= 1'b1;
    @(posedge clk);
    start <= 1'b0;
    // Wait for done with timeout
    fork
      begin
        wait (done === 1'b1);
      end
      begin
        repeat (500_000) @(posedge clk);
        $display("ERROR: Timeout waiting for done signal!");
        $finish;
      end
    join_any
    disable fork;
    repeat (2) @(posedge clk);
  endtask

  // ══════════════════════════════════════════════════════════════════════════
  // Main Test Sequence
  // ══════════════════════════════════════════════════════════════════════════
  int total_pass, total_fail;

  initial begin
    total_pass = 0;
    total_fail = 0;

    $display("================================================================");
    $display("  conv1x1_engine Golden Verification Testbench");
    $display("================================================================");

    // Initialize SiLU LUT (identity-ish for basic tests; not used until SiLU test)
    for (int i = 0; i < 256; i++) begin
      automatic int signed sv;
      sv = i - 128;
      // Simple approximation: clamp(sv, -128, 127)
      silu_lut[i] = (sv > 127) ? 8'sd127 : ((sv < -128) ? -8'sd128 : sv[7:0]);
    end

    // ════════════════════════════════════════════════════════════════════
    // TEST 1: Minimal — Cin=1, Cout=1, H=1, W=32
    // ════════════════════════════════════════════════════════════════════
    begin
      automatic int H = 1, W = 32, Cin = 1, Cout = 1;
      automatic int num_wblk = (W + TB_LANES - 1) / TB_LANES;
      automatic int mismatches;

      $display("\n--- TEST 1: Minimal (Cin=%0d, Cout=%0d, H=%0d, W=%0d) ---", Cin, Cout, H, W);
      clear_memories();
      reset_dut();

      // Fill IFM: pattern — lane index
      for (int h = 0; h < H; h++)
        for (int ci = 0; ci < Cin; ci++)
          for (int wb = 0; wb < num_wblk; wb++)
            for (int l = 0; l < TB_LANES; l++)
              ifm_mem[h * Cin * num_wblk + ci * num_wblk + wb][l] = l[7:0];

      // Fill WGT: weight = 2
      for (int co = 0; co < Cout; co++)
        for (int ci = 0; ci < Cin; ci++)
          wgt_mem[co * Cin + ci] = 8'sd2;

      // Bias = 10, m_int = 1, shift = 0, zp = 0
      for (int co = 0; co < Cout; co++) begin
        bias_arr[co]  = 32'sd10;
        m_int_arr[co] = 32'sd1;
        shift_arr[co] = 6'd0;
      end

      cfg_h        = H[9:0];
      cfg_w        = W[9:0];
      cfg_cin      = Cin[8:0];
      cfg_cout     = Cout[8:0];
      cfg_act_mode = ACT_NONE;
      cfg_zp_out   = 8'sd0;

      compute_golden(H, W, Cin, Cout, ACT_NONE, 8'sd0);
      run_engine();

      mismatches = compare_ofm(H, W, Cout, "TEST1");
      if (mismatches == 0) begin
        $display("  TEST 1: PASS");
        total_pass++;
      end else begin
        $display("  TEST 1: FAIL (%0d mismatches)", mismatches);
        total_fail++;
      end
    end

    // ════════════════════════════════════════════════════════════════════
    // TEST 2: QC2f cv1 style — Cin=32, Cout=32, H=4, W=64
    // ════════════════════════════════════════════════════════════════════
    begin
      automatic int H = 4, W = 64, Cin = 32, Cout = 32;
      automatic int num_wblk = (W + TB_LANES - 1) / TB_LANES;
      automatic int mismatches;

      $display("\n--- TEST 2: QC2f cv1 (Cin=%0d, Cout=%0d, H=%0d, W=%0d) ---", Cin, Cout, H, W);
      clear_memories();
      reset_dut();

      // Fill IFM: deterministic pattern
      for (int h = 0; h < H; h++)
        for (int ci = 0; ci < Cin; ci++)
          for (int wb = 0; wb < num_wblk; wb++)
            for (int l = 0; l < TB_LANES; l++) begin
              // Small signed values to avoid overflow before requant
              automatic int val;
              val = ((h * 7 + ci * 3 + wb * 5 + l) % 11) - 5;
              ifm_mem[h * Cin * num_wblk + ci * num_wblk + wb][l] = val[7:0];
            end

      // Fill WGT: small deterministic weights
      for (int co = 0; co < Cout; co++)
        for (int ci = 0; ci < Cin; ci++) begin
          automatic int val;
          val = ((co * 5 + ci * 3) % 7) - 3;
          wgt_mem[co * Cin + ci] = val[7:0];
        end

      // Bias = small per-channel, requant: m_int=1, shift=4
      for (int co = 0; co < Cout; co++) begin
        bias_arr[co]  = 32'(co) - 32'sd16;
        m_int_arr[co] = 32'sd1;
        shift_arr[co] = 6'd4;
      end

      cfg_h        = H[9:0];
      cfg_w        = W[9:0];
      cfg_cin      = Cin[8:0];
      cfg_cout     = Cout[8:0];
      cfg_act_mode = ACT_NONE;
      cfg_zp_out   = 8'sd0;

      compute_golden(H, W, Cin, Cout, ACT_NONE, 8'sd0);
      run_engine();

      mismatches = compare_ofm(H, W, Cout, "TEST2");
      if (mismatches == 0) begin
        $display("  TEST 2: PASS");
        total_pass++;
      end else begin
        $display("  TEST 2: FAIL (%0d mismatches)", mismatches);
        total_fail++;
      end
    end

    // ════════════════════════════════════════════════════════════════════
    // TEST 3: Channel reduction — Cin=64, Cout=16, H=2, W=32
    // ════════════════════════════════════════════════════════════════════
    begin
      automatic int H = 2, W = 32, Cin = 64, Cout = 16;
      automatic int num_wblk = (W + TB_LANES - 1) / TB_LANES;
      automatic int mismatches;

      $display("\n--- TEST 3: Channel reduction (Cin=%0d, Cout=%0d, H=%0d, W=%0d) ---", Cin, Cout, H, W);
      clear_memories();
      reset_dut();

      // Fill IFM: alternating small positive/negative
      for (int h = 0; h < H; h++)
        for (int ci = 0; ci < Cin; ci++)
          for (int wb = 0; wb < num_wblk; wb++)
            for (int l = 0; l < TB_LANES; l++) begin
              automatic int val;
              val = ((h + ci + l) % 2 == 0) ? 1 : -1;
              ifm_mem[h * Cin * num_wblk + ci * num_wblk + wb][l] = val[7:0];
            end

      // Fill WGT: identity-like pattern with some variation
      for (int co = 0; co < Cout; co++)
        for (int ci = 0; ci < Cin; ci++) begin
          automatic int val;
          val = ((co + ci) % 5 == 0) ? 2 : ((co + ci) % 3 == 0 ? -1 : 0);
          wgt_mem[co * Cin + ci] = val[7:0];
        end

      // Bias = 0, requant: m_int=1, shift=2
      for (int co = 0; co < Cout; co++) begin
        bias_arr[co]  = 32'sd0;
        m_int_arr[co] = 32'sd1;
        shift_arr[co] = 6'd2;
      end

      cfg_h        = H[9:0];
      cfg_w        = W[9:0];
      cfg_cin      = Cin[8:0];
      cfg_cout     = Cout[8:0];
      cfg_act_mode = ACT_NONE;
      cfg_zp_out   = 8'sd5;

      compute_golden(H, W, Cin, Cout, ACT_NONE, 8'sd5);
      run_engine();

      mismatches = compare_ofm(H, W, Cout, "TEST3");
      if (mismatches == 0) begin
        $display("  TEST 3: PASS");
        total_pass++;
      end else begin
        $display("  TEST 3: FAIL (%0d mismatches)", mismatches);
        total_fail++;
      end
    end

    // ════════════════════════════════════════════════════════════════════
    // TEST 4: Random stress — Cin=16, Cout=16, H=4, W=64, ACT_RELU
    // ════════════════════════════════════════════════════════════════════
    begin
      automatic int H = 4, W = 64, Cin = 16, Cout = 16;
      automatic int num_wblk = (W + TB_LANES - 1) / TB_LANES;
      automatic int mismatches;
      automatic int seed;

      $display("\n--- TEST 4: Random stress (Cin=%0d, Cout=%0d, H=%0d, W=%0d, ACT_RELU) ---", Cin, Cout, H, W);
      clear_memories();
      reset_dut();

      // Seed for reproducibility
      seed = 42;

      // Fill IFM: random signed 8-bit
      for (int h = 0; h < H; h++)
        for (int ci = 0; ci < Cin; ci++)
          for (int wb = 0; wb < num_wblk; wb++)
            for (int l = 0; l < TB_LANES; l++) begin
              automatic int rval;
              rval = $urandom(seed) % 256;
              seed = seed + 1;
              ifm_mem[h * Cin * num_wblk + ci * num_wblk + wb][l] = rval[7:0] - 8'd128;
            end

      // Fill WGT: random signed 8-bit (small range to avoid huge accumulation)
      for (int co = 0; co < Cout; co++)
        for (int ci = 0; ci < Cin; ci++) begin
          automatic int rval;
          rval = $urandom(seed) % 16;
          seed = seed + 1;
          wgt_mem[co * Cin + ci] = rval[7:0] - 8'd8;
        end

      // Per-channel bias, m_int, shift
      for (int co = 0; co < Cout; co++) begin
        automatic int rval;
        rval = $urandom(seed) % 200;
        seed = seed + 1;
        bias_arr[co]  = 32'(rval) - 32'sd100;
        m_int_arr[co] = 32'sd1;
        shift_arr[co] = 6'd6;
      end

      cfg_h        = H[9:0];
      cfg_w        = W[9:0];
      cfg_cin      = Cin[8:0];
      cfg_cout     = Cout[8:0];
      cfg_act_mode = ACT_RELU;
      cfg_zp_out   = 8'sd0;

      compute_golden(H, W, Cin, Cout, ACT_RELU, 8'sd0);
      run_engine();

      mismatches = compare_ofm(H, W, Cout, "TEST4");
      if (mismatches == 0) begin
        $display("  TEST 4: PASS");
        total_pass++;
      end else begin
        $display("  TEST 4: FAIL (%0d mismatches)", mismatches);
        total_fail++;
      end
    end

    // ════════════════════════════════════════════════════════════════════
    // Summary
    // ════════════════════════════════════════════════════════════════════
    $display("\n================================================================");
    $display("  SUMMARY:  %0d PASSED,  %0d FAILED  (of %0d tests)",
             total_pass, total_fail, total_pass + total_fail);
    if (total_fail == 0)
      $display("  >>> ALL TESTS PASSED <<<");
    else
      $display("  >>> SOME TESTS FAILED <<<");
    $display("================================================================\n");

    $finish;
  end

endmodule
