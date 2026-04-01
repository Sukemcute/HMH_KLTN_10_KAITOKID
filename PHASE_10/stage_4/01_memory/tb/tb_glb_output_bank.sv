`timescale 1ns/1ps

module tb_glb_output_bank;

  import accel_pkg::*;

  // ---------- Parameters ----------
  localparam int LANES      = 32;
  localparam int BANK_DEPTH = 512;
  localparam int ADDR_W     = $clog2(BANK_DEPTH);
  localparam int CLK_PERIOD = 10;

  // ---------- DUT signals ----------
  logic                  clk;
  logic                  rst_n;
  logic                  wr_en;
  logic [ADDR_W-1:0]    wr_addr;
  accel_pkg::namespace_e wr_ns;
  logic [LANES*32-1:0]  wr_data_psum;
  logic [LANES*8-1:0]   wr_data_act;
  logic                  rd_en;
  logic [ADDR_W-1:0]    rd_addr;
  accel_pkg::namespace_e rd_ns;
  logic [LANES*32-1:0]  rd_data_psum;
  logic [LANES*8-1:0]   rd_data_act;

  // ---------- DUT ----------
  glb_output_bank #(
    .LANES      (LANES),
    .BANK_DEPTH (BANK_DEPTH)
  ) u_dut (
    .clk          (clk),
    .rst_n        (rst_n),
    .wr_en        (wr_en),
    .wr_addr      (wr_addr),
    .wr_ns        (wr_ns),
    .wr_data_psum (wr_data_psum),
    .wr_data_act  (wr_data_act),
    .rd_en        (rd_en),
    .rd_addr      (rd_addr),
    .rd_ns        (rd_ns),
    .rd_data_psum (rd_data_psum),
    .rd_data_act  (rd_data_act)
  );

  // ---------- Clock ----------
  initial clk = 0;
  always #(CLK_PERIOD/2) clk = ~clk;

  // ---------- Scoreboard ----------
  int test_count;
  int pass_count;
  int fail_count;

  task automatic check_psum(string test_name, logic [LANES*32-1:0] expected, logic [LANES*32-1:0] actual);
    test_count++;
    if (actual === expected) begin
      pass_count++;
    end else begin
      fail_count++;
      $display("[FAIL] %s (PSUM) : mismatch", test_name);
      $display("  expected[255:0] = 0x%064h", expected[255:0]);
      $display("  actual  [255:0] = 0x%064h", actual[255:0]);
    end
  endtask

  task automatic check_act(string test_name, logic [LANES*8-1:0] expected, logic [LANES*8-1:0] actual);
    test_count++;
    if (actual === expected) begin
      pass_count++;
    end else begin
      fail_count++;
      $display("[FAIL] %s (ACT) : expected 0x%064h, got 0x%064h", test_name, expected, actual);
    end
  endtask

  // ---------- Helper tasks ----------
  task automatic do_write_psum(input logic [ADDR_W-1:0] addr, input logic [LANES*32-1:0] data);
    @(posedge clk);
    wr_en        <= 1'b1;
    wr_addr      <= addr;
    wr_ns        <= NS_PSUM;
    wr_data_psum <= data;
    wr_data_act  <= '0;
    @(posedge clk);
    wr_en        <= 1'b0;
  endtask

  task automatic do_write_act(input logic [ADDR_W-1:0] addr, input logic [LANES*8-1:0] data);
    @(posedge clk);
    wr_en        <= 1'b1;
    wr_addr      <= addr;
    wr_ns        <= NS_ACT;
    wr_data_psum <= '0;
    wr_data_act  <= data;
    @(posedge clk);
    wr_en        <= 1'b0;
  endtask

  task automatic do_read(input logic [ADDR_W-1:0] addr, input accel_pkg::namespace_e ns);
    @(posedge clk);
    rd_en   <= 1'b1;
    rd_addr <= addr;
    rd_ns   <= ns;
    @(posedge clk);
    rd_en   <= 1'b0;
    @(posedge clk);  // registered output available
  endtask

  // ---------- Pattern generators ----------
  function automatic logic [LANES*32-1:0] gen_psum_pattern(input int seed);
    logic [LANES*32-1:0] p;
    for (int i = 0; i < LANES; i++)
      p[(i+1)*32-1 -: 32] = (seed + i * 7) & 32'hFFFFFFFF;
    return p;
  endfunction

  function automatic logic [LANES*8-1:0] gen_act_pattern(input int seed);
    logic [LANES*8-1:0] p;
    for (int i = 0; i < LANES; i++)
      p[(i+1)*8-1 -: 8] = (seed + i * 3) & 8'hFF;
    return p;
  endfunction

  // ---------- Stimulus ----------
  initial begin
    // Init
    clk          = 0;
    rst_n        = 0;
    wr_en        = 0;
    rd_en        = 0;
    wr_addr      = '0;
    wr_ns        = NS_PSUM;
    wr_data_psum = '0;
    wr_data_act  = '0;
    rd_addr      = '0;
    rd_ns        = NS_PSUM;
    test_count   = 0;
    pass_count   = 0;
    fail_count   = 0;

    // Reset
    repeat (4) @(posedge clk);
    rst_n = 1;
    repeat (2) @(posedge clk);

    // ========== T1: Write PSUM, read back ==========
    $display("\n===== T1: Write PSUM namespace, read back =====");
    begin
      logic [LANES*32-1:0] t1_data;
      t1_data = gen_psum_pattern(42);

      do_write_psum(9'd0, t1_data);
      do_read(9'd0, NS_PSUM);
      check_psum("T1_psum_rw", t1_data, rd_data_psum);

      if (rd_data_psum === t1_data) $display("[PASS] T1: PSUM write/read — exact 32-bit per lane");
      else                          $display("[FAIL] T1: PSUM write/read — mismatch");
    end

    // ========== T2: Write ACT, read back ==========
    $display("\n===== T2: Write ACT namespace, read back =====");
    begin
      logic [LANES*8-1:0] t2_data;
      t2_data = gen_act_pattern(77);

      do_write_act(9'd1, t2_data);
      do_read(9'd1, NS_ACT);
      check_act("T2_act_rw", t2_data, rd_data_act);

      if (rd_data_act === t2_data) $display("[PASS] T2: ACT write/read — exact 8-bit per lane");
      else                         $display("[FAIL] T2: ACT write/read — mismatch");
    end

    // ========== T3: Namespace switch ==========
    $display("\n===== T3: Namespace switch =====");
    begin
      int t3_pass;
      logic [LANES*32-1:0] t3_psum;
      logic [LANES*8-1:0]  t3_act;
      t3_pass = 1;

      // Write PSUM at addr 10
      t3_psum = gen_psum_pattern(111);
      do_write_psum(9'd10, t3_psum);

      // Write ACT at addr 11
      t3_act = gen_act_pattern(222);
      do_write_act(9'd11, t3_act);

      // Read PSUM from addr 10
      do_read(9'd10, NS_PSUM);
      check_psum("T3_psum_switch", t3_psum, rd_data_psum);
      if (rd_data_psum !== t3_psum) t3_pass = 0;

      // Read ACT from addr 11
      do_read(9'd11, NS_ACT);
      check_act("T3_act_switch", t3_act, rd_data_act);
      if (rd_data_act !== t3_act) t3_pass = 0;

      if (t3_pass) $display("[PASS] T3: Namespace switch — both PSUM and ACT correct");
      else         $display("[FAIL] T3: Namespace switch — mismatch");
    end

    // ========== T4: ACT shares lower bits of PSUM storage ==========
    $display("\n===== T4: ACT shares lower bits of PSUM storage =====");
    begin
      int t4_pass;
      logic [LANES*8-1:0]  t4_act;
      logic [LANES*32-1:0] t4_psum_raw;
      t4_pass = 1;

      // Write ACT at addr 20
      t4_act = gen_act_pattern(55);
      do_write_act(9'd20, t4_act);

      // Read as PSUM (full 1024b) — lower 256 bits should match ACT data
      do_read(9'd20, NS_PSUM);
      t4_psum_raw = rd_data_psum;

      // Compare lower LANES*8 bits
      check_act("T4_act_in_lower_psum", t4_act, t4_psum_raw[LANES*8-1:0]);
      if (t4_psum_raw[LANES*8-1:0] !== t4_act) t4_pass = 0;

      if (t4_pass) $display("[PASS] T4: ACT data visible in lower 256 bits of PSUM read");
      else         $display("[FAIL] T4: ACT/PSUM sharing — mismatch in lower bits");
    end

    // ========== T5: Multiple addresses with mixed namespace writes ==========
    $display("\n===== T5: Multiple addresses, mixed namespace writes =====");
    begin
      int t5_pass;
      logic [LANES*32-1:0] t5_psum_golden [10];
      logic [LANES*8-1:0]  t5_act_golden  [10];
      t5_pass = 1;

      // Write PSUM at even addresses, ACT at odd addresses
      for (int a = 0; a < 10; a++) begin
        if (a % 2 == 0) begin
          t5_psum_golden[a] = gen_psum_pattern(1000 + a * 13);
          do_write_psum((9'd100 + a[8:0]), t5_psum_golden[a]);
        end else begin
          t5_act_golden[a] = gen_act_pattern(2000 + a * 17);
          do_write_act((9'd100 + a[8:0]), t5_act_golden[a]);
        end
      end

      // Read back and verify
      for (int a = 0; a < 10; a++) begin
        if (a % 2 == 0) begin
          do_read((9'd100 + a[8:0]), NS_PSUM);
          check_psum($sformatf("T5_psum_addr%0d", 100+a), t5_psum_golden[a], rd_data_psum);
          if (rd_data_psum !== t5_psum_golden[a]) t5_pass = 0;
        end else begin
          do_read((9'd100 + a[8:0]), NS_ACT);
          check_act($sformatf("T5_act_addr%0d", 100+a), t5_act_golden[a], rd_data_act);
          if (rd_data_act !== t5_act_golden[a]) t5_pass = 0;
        end
      end

      if (t5_pass) $display("[PASS] T5: Mixed namespace writes — all addresses correct");
      else         $display("[FAIL] T5: Mixed namespace writes — mismatches detected");
    end

    // ========== Summary ==========
    $display("\n========================================");
    $display("  tb_glb_output_bank SUMMARY");
    $display("  Total checks : %0d", test_count);
    $display("  PASS         : %0d", pass_count);
    $display("  FAIL         : %0d", fail_count);
    if (fail_count == 0)
      $display("  *** ALL TESTS PASSED ***");
    else
      $display("  *** SOME TESTS FAILED ***");
    $display("========================================\n");
    $finish;
  end

endmodule
