`timescale 1ns/1ps

module tb_glb_input_bank;

  import accel_pkg::*;

  // ---------- Parameters ----------
  localparam int LANES         = 32;
  localparam int SUBBANK_DEPTH = 2048;
  localparam int ADDR_W        = $clog2(SUBBANK_DEPTH);
  localparam int CLK_PERIOD    = 10;  // 5ns half-period

  // ---------- DUT signals ----------
  logic                 clk;
  logic                 rst_n;
  logic                 wr_en;
  logic [ADDR_W-1:0]   wr_addr;
  logic [LANES*8-1:0]  wr_data;
  logic [LANES-1:0]    wr_lane_mask;
  logic                 rd_en;
  logic [ADDR_W-1:0]   rd_addr;
  logic [LANES*8-1:0]  rd_data;

  // ---------- DUT ----------
  glb_input_bank #(
    .LANES         (LANES),
    .SUBBANK_DEPTH (SUBBANK_DEPTH)
  ) u_dut (
    .clk          (clk),
    .rst_n        (rst_n),
    .wr_en        (wr_en),
    .wr_addr      (wr_addr),
    .wr_data      (wr_data),
    .wr_lane_mask (wr_lane_mask),
    .rd_en        (rd_en),
    .rd_addr      (rd_addr),
    .rd_data      (rd_data)
  );

  // ---------- Clock ----------
  initial clk = 0;
  always #(CLK_PERIOD/2) clk = ~clk;

  // ---------- Scoreboard ----------
  int test_count;
  int pass_count;
  int fail_count;

  task automatic check(string test_name, logic [LANES*8-1:0] expected, logic [LANES*8-1:0] actual);
    test_count++;
    if (actual === expected) begin
      pass_count++;
    end else begin
      fail_count++;
      $display("[FAIL] %s : expected 0x%064h, got 0x%064h", test_name, expected, actual);
    end
  endtask

  task automatic check_lane(string test_name, int lane, logic [7:0] expected, logic [7:0] actual);
    test_count++;
    if (actual === expected) begin
      pass_count++;
    end else begin
      fail_count++;
      $display("[FAIL] %s lane %0d : expected 0x%02h, got 0x%02h", test_name, lane, expected, actual);
    end
  endtask

  // ---------- Helper tasks ----------
  task automatic do_write(input logic [ADDR_W-1:0] addr, input logic [LANES*8-1:0] data, input logic [LANES-1:0] mask);
    @(posedge clk);
    wr_en        <= 1'b1;
    wr_addr      <= addr;
    wr_data      <= data;
    wr_lane_mask <= mask;
    @(posedge clk);
    wr_en        <= 1'b0;
  endtask

  task automatic do_read(input logic [ADDR_W-1:0] addr, output logic [LANES*8-1:0] data_out);
    @(posedge clk);
    rd_en   <= 1'b1;
    rd_addr <= addr;
    @(posedge clk);
    rd_en   <= 1'b0;
    @(posedge clk);  // registered output available
    data_out = rd_data;
  endtask

  // ---------- Stimulus ----------
  logic [LANES*8-1:0] read_back;
  logic [LANES*8-1:0] golden_mem [101];  // for T3
  logic [LANES*8-1:0] golden_rand [SUBBANK_DEPTH];
  logic [LANES-1:0]   mask_rand   [SUBBANK_DEPTH];
  bit                  addr_written [SUBBANK_DEPTH];

  initial begin
    // Init
    clk           = 0;
    rst_n         = 0;
    wr_en         = 0;
    rd_en         = 0;
    wr_addr       = '0;
    wr_data       = '0;
    wr_lane_mask  = '0;
    rd_addr       = '0;
    test_count    = 0;
    pass_count    = 0;
    fail_count    = 0;

    // Reset
    repeat (4) @(posedge clk);
    rst_n = 1;
    repeat (2) @(posedge clk);

    // ========== T1: Write all lanes, read back ==========
    $display("\n===== T1: Write all lanes, read back =====");
    begin
      logic [LANES*8-1:0] t1_data;
      for (int i = 0; i < LANES; i++)
        t1_data[(i+1)*8-1 -: 8] = 8'hA0 + i[7:0];

      do_write(11'd0, t1_data, {LANES{1'b1}});
      do_read(11'd0, read_back);
      check("T1_all_lanes", t1_data, read_back);
      if (read_back === t1_data) $display("[PASS] T1: Write all lanes, read back — exact match");
    end

    // ========== T2: Lane-masked write ==========
    $display("\n===== T2: Lane-masked write (mask=0x0000FFFF) =====");
    begin
      logic [LANES*8-1:0] bg_data;
      logic [LANES*8-1:0] masked_data;
      logic [LANES*8-1:0] expected;

      // Background: fill all lanes at addr 5
      for (int i = 0; i < LANES; i++)
        bg_data[(i+1)*8-1 -: 8] = 8'hFF;
      do_write(11'd5, bg_data, {LANES{1'b1}});

      // Masked write: only lanes 0-15
      for (int i = 0; i < LANES; i++)
        masked_data[(i+1)*8-1 -: 8] = 8'h55;
      do_write(11'd5, masked_data, 32'h0000FFFF);

      // Expected: lanes 0-15 = 0x55, lanes 16-31 = 0xFF
      for (int i = 0; i < LANES; i++) begin
        if (i < 16)
          expected[(i+1)*8-1 -: 8] = 8'h55;
        else
          expected[(i+1)*8-1 -: 8] = 8'hFF;
      end

      do_read(11'd5, read_back);

      // Check each lane individually
      begin
        int t2_pass;
        t2_pass = 1;
        for (int i = 0; i < LANES; i++) begin
          logic [7:0] exp_lane, act_lane;
          exp_lane = expected[(i+1)*8-1 -: 8];
          act_lane = read_back[(i+1)*8-1 -: 8];
          check_lane("T2_mask", i, exp_lane, act_lane);
          if (act_lane !== exp_lane) t2_pass = 0;
        end
        if (t2_pass) $display("[PASS] T2: Lane-masked write — lanes 16-31 unchanged");
        else         $display("[FAIL] T2: Lane-masked write — mismatch detected");
      end
    end

    // ========== T3: Multiple addresses (0..100) ==========
    $display("\n===== T3: Multiple addresses 0..100 =====");
    begin
      int t3_pass;
      t3_pass = 1;

      for (int a = 0; a <= 100; a++) begin
        for (int i = 0; i < LANES; i++)
          golden_mem[a][(i+1)*8-1 -: 8] = (a[7:0] + i[7:0]) & 8'hFF;
        do_write(a[ADDR_W-1:0], golden_mem[a], {LANES{1'b1}});
      end

      for (int a = 0; a <= 100; a++) begin
        do_read(a[ADDR_W-1:0], read_back);
        if (read_back !== golden_mem[a]) begin
          t3_pass = 0;
          $display("[FAIL] T3 addr %0d : expected 0x%064h, got 0x%064h", a, golden_mem[a], read_back);
        end
        test_count++;
        if (read_back === golden_mem[a]) pass_count++;
        else fail_count++;
      end

      if (t3_pass) $display("[PASS] T3: Multiple addresses 0..100 — all exact");
      else         $display("[FAIL] T3: Multiple addresses — mismatches detected");
    end

    // ========== T4: Banking simulation (h mod 3 pattern) ==========
    $display("\n===== T4: Banking simulation (h mod 3) =====");
    begin
      int t4_pass;
      logic [LANES*8-1:0] bank0_data, bank1_data, bank2_data;
      t4_pass = 1;

      for (int i = 0; i < LANES; i++) begin
        bank0_data[(i+1)*8-1 -: 8] = 8'h10 + i[7:0];
        bank1_data[(i+1)*8-1 -: 8] = 8'h20 + i[7:0];
        bank2_data[(i+1)*8-1 -: 8] = 8'h30 + i[7:0];
      end

      // Write bank0 at addr 200, bank1 at addr 201, bank2 at addr 202
      do_write(11'd200, bank0_data, {LANES{1'b1}});
      do_write(11'd201, bank1_data, {LANES{1'b1}});
      do_write(11'd202, bank2_data, {LANES{1'b1}});

      // Read back and verify isolation
      do_read(11'd200, read_back);
      if (read_back !== bank0_data) begin t4_pass = 0; test_count++; fail_count++; end
      else begin test_count++; pass_count++; end

      do_read(11'd201, read_back);
      if (read_back !== bank1_data) begin t4_pass = 0; test_count++; fail_count++; end
      else begin test_count++; pass_count++; end

      do_read(11'd202, read_back);
      if (read_back !== bank2_data) begin t4_pass = 0; test_count++; fail_count++; end
      else begin test_count++; pass_count++; end

      if (t4_pass) $display("[PASS] T4: Banking simulation — isolation verified");
      else         $display("[FAIL] T4: Banking simulation — data leaked between banks");
    end

    // ========== T5: Random stress (200 writes) ==========
    $display("\n===== T5: Random stress (200 random writes) =====");
    begin
      int t5_pass;
      logic [ADDR_W-1:0]   rnd_addr;
      logic [LANES*8-1:0]  rnd_data;
      logic [LANES-1:0]    rnd_mask;
      t5_pass = 1;

      // Initialize golden model: clear tracking
      for (int a = 0; a < SUBBANK_DEPTH; a++) begin
        golden_rand[a]  = '0;
        mask_rand[a]    = '0;
        addr_written[a] = 0;
      end

      // First: write zeros to a range so the golden model baseline is known
      for (int a = 0; a < 512; a++) begin
        do_write(a[ADDR_W-1:0], '0, {LANES{1'b1}});
        addr_written[a] = 1;
      end

      // 200 random writes
      for (int w = 0; w < 200; w++) begin
        rnd_addr = $urandom_range(0, 511);
        for (int i = 0; i < LANES; i++)
          rnd_data[(i+1)*8-1 -: 8] = $urandom & 8'hFF;
        rnd_mask = $urandom;
        if (rnd_mask == '0) rnd_mask = {LANES{1'b1}};  // avoid empty mask

        do_write(rnd_addr, rnd_data, rnd_mask);

        // Update golden model per lane
        for (int i = 0; i < LANES; i++) begin
          if (rnd_mask[i])
            golden_rand[rnd_addr][(i+1)*8-1 -: 8] = rnd_data[(i+1)*8-1 -: 8];
        end
        addr_written[rnd_addr] = 1;
      end

      // Read back and verify
      for (int a = 0; a < 512; a++) begin
        if (addr_written[a]) begin
          do_read(a[ADDR_W-1:0], read_back);
          test_count++;
          if (read_back !== golden_rand[a]) begin
            t5_pass = 0;
            fail_count++;
            $display("[FAIL] T5 addr %0d : expected 0x%064h, got 0x%064h", a, golden_rand[a], read_back);
          end else begin
            pass_count++;
          end
        end
      end

      if (t5_pass) $display("[PASS] T5: Random stress — all 200 random writes verified");
      else         $display("[FAIL] T5: Random stress — mismatches detected");
    end

    // ========== Summary ==========
    $display("\n========================================");
    $display("  tb_glb_input_bank SUMMARY");
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
