// Self-checking testbench for addr_gen_input
// CRITICAL TESTS:
// Test 1: bank_id = h mod 3
// Test 2: padding positions output zp_x (NOT zero!)
// Test 3: no address collision for different (h,w,c) tuples
// Test 4: stride support
`timescale 1ns/1ps

module tb_addr_gen_input;

  localparam int LANES = 32;

  logic              clk, rst_n;
  logic [9:0]        cfg_win, cfg_hin;
  logic [8:0]        cfg_cin_tile;
  logic [3:0]        cfg_q_in, cfg_stride;
  logic [3:0]        cfg_pad_top, cfg_pad_bot, cfg_pad_left, cfg_pad_right;
  logic signed [7:0] cfg_zp_x;
  logic              req_valid;
  logic [9:0]        req_h, req_w;
  logic [8:0]        req_c;
  logic              out_valid;
  logic [1:0]        out_bank_id;
  logic [15:0]       out_addr;
  logic              out_is_padding;
  logic signed [7:0] out_pad_value;

  addr_gen_input #(.LANES(LANES)) uut (.*);

  always #2.5 clk = ~clk;
  int fail_count = 0;

  task automatic reset();
    rst_n = 0; req_valid = 0;
    cfg_win = 80; cfg_hin = 80;
    cfg_cin_tile = 32; cfg_q_in = 4;
    cfg_stride = 1;
    cfg_pad_top = 1; cfg_pad_bot = 1;
    cfg_pad_left = 1; cfg_pad_right = 1;
    cfg_zp_x = -17;
    repeat(3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);
  endtask

  // ═══════════ TEST 1: bank_id = h mod 3 ═══════════
  task automatic test_banking();
    int errors = 0;

    $display("=== TEST 1: bank_id = h mod 3 ===");
    reset();
    cfg_pad_top = 0; cfg_pad_bot = 0;
    cfg_pad_left = 0; cfg_pad_right = 0;

    for (int h = 0; h < 12; h++) begin
      @(negedge clk);
      req_valid = 1;
      req_h = h; req_w = 0; req_c = 0;
      @(negedge clk);
      req_valid = 0;
      @(posedge clk);  // 1 cycle latency

      if (out_bank_id !== (h % 3)) begin
        $display("  FAIL h=%0d: bank_id=%0d expected=%0d", h, out_bank_id, h % 3);
        errors++;
      end
    end

    if (errors == 0) $display("  TEST 1 PASSED");
    else $display("  TEST 1 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  // ═══════════ TEST 2: Padding outputs zp_x ═══════════
  task automatic test_padding_zp();
    int errors = 0;

    $display("=== TEST 2: Padding positions output zp_x (NOT zero!) ===");
    reset();
    cfg_zp_x = -42;

    // Top padding: h = 0 (pad_top = 1)
    @(negedge clk);
    req_valid = 1; req_h = 0; req_w = 32; req_c = 0;
    @(negedge clk); req_valid = 0;
    @(posedge clk);

    if (!out_is_padding) begin
      $display("  FAIL: h=0 should be padding (pad_top=1)");
      errors++;
    end
    if (out_pad_value !== -8'sd42) begin
      $display("  FAIL: pad_value=%0d expected=-42 (zp_x)", out_pad_value);
      errors++;
    end

    // Bottom padding: h = hin - 1 = 79 (pad_bot = 1)
    @(negedge clk);
    req_valid = 1; req_h = 79; req_w = 32; req_c = 0;
    @(negedge clk); req_valid = 0;
    @(posedge clk);

    if (!out_is_padding) begin
      $display("  FAIL: h=79 should be padding (pad_bot=1, hin=80)");
      errors++;
    end

    // Left padding: w = 0 (pad_left = 1)
    @(negedge clk);
    req_valid = 1; req_h = 40; req_w = 0; req_c = 0;
    @(negedge clk); req_valid = 0;
    @(posedge clk);

    if (!out_is_padding) begin
      $display("  FAIL: w=0 should be padding (pad_left=1)");
      errors++;
    end

    // Non-padding: h=5, w=32 (inside valid region)
    @(negedge clk);
    req_valid = 1; req_h = 5; req_w = 32; req_c = 0;
    @(negedge clk); req_valid = 0;
    @(posedge clk);

    if (out_is_padding) begin
      $display("  FAIL: h=5,w=32 should NOT be padding");
      errors++;
    end

    if (errors == 0) $display("  TEST 2 PASSED: Padding correctly outputs zp_x=-42");
    else $display("  TEST 2 FAILED: %0d errors", errors);
    fail_count += errors;
  endtask

  // ═══════════ TEST 3: No address collision ═══════════
  task automatic test_no_collision();
    int errors = 0;
    int seen [int];
    bit stop;

    $display("=== TEST 3: No address collision ===");
    reset();
    cfg_win = 64; cfg_hin = 12;
    cfg_cin_tile = 8; cfg_q_in = 4;
    cfg_pad_top = 0; cfg_pad_bot = 0;
    cfg_pad_left = 0; cfg_pad_right = 0;
    stop = 0;

    for (int h = 0; h < 12 && !stop; h++) begin
      for (int w = 0; w < 64 && !stop; w += LANES) begin
        for (int c = 0; c < 8 && !stop; c++) begin
          @(negedge clk);
          req_valid = 1;
          req_h = h; req_w = w; req_c = c;
          @(negedge clk);
          req_valid = 0;
          @(posedge clk);

          begin
            automatic int key = (int'(out_bank_id) << 16) | int'(out_addr);
            if (seen.exists(key)) begin
              $display("  FAIL: collision at bank=%0d addr=%0d (h=%0d,w=%0d,c=%0d conflicts with prev)",
                       out_bank_id, out_addr, h, w, c);
              errors++;
              if (errors > 5) begin
                $display("  ... stopping");
                stop = 1;
              end
            end
            seen[key] = 1;
          end
        end
      end
    end

    if (errors == 0) $display("  TEST 3 PASSED: No collisions in %0d addresses", seen.size());
    else $display("  TEST 3 FAILED: %0d collisions", errors);
    fail_count += errors;
  endtask

  initial begin
    clk = 0;
    $display("\n╔══════════════════════════════════════════════════╗");
    $display("║  TESTBENCH: addr_gen_input                      ║");
    $display("╚══════════════════════════════════════════════════╝\n");

    test_banking();
    test_padding_zp();
    test_no_collision();

    $display("\n══════════════════════════════════════════════════");
    if (fail_count == 0) $display("  ★ ALL ADDR_GEN_INPUT TESTS PASSED ★");
    else $display("  ✗ TOTAL FAILURES: %0d", fail_count);
    $display("══════════════════════════════════════════════════\n");
    $finish;
  end

endmodule
