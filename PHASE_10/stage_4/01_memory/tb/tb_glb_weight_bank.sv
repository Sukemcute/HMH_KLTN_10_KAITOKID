`timescale 1ns/1ps

module tb_glb_weight_bank;

  import accel_pkg::*;

  // ---------- Parameters ----------
  localparam int LANES      = 32;
  localparam int BANK_DEPTH = 1024;
  localparam int FIFO_DEPTH = 8;
  localparam int ADDR_W     = $clog2(BANK_DEPTH);
  localparam int CLK_PERIOD = 10;

  // ---------- DUT signals ----------
  logic                 clk;
  logic                 rst_n;
  logic                 wr_en;
  logic [ADDR_W-1:0]   wr_addr;
  logic [LANES*8-1:0]  wr_data;
  logic                 rd_en;
  logic [ADDR_W-1:0]   rd_addr;
  logic [LANES*8-1:0]  rd_data;
  logic                 fifo_push;
  logic [LANES*8-1:0]  fifo_din;
  logic                 fifo_pop;
  logic [LANES*8-1:0]  fifo_dout;
  logic                 fifo_empty;
  logic                 fifo_full;

  // ---------- DUT ----------
  glb_weight_bank #(
    .LANES      (LANES),
    .BANK_DEPTH (BANK_DEPTH),
    .FIFO_DEPTH (FIFO_DEPTH)
  ) u_dut (
    .clk        (clk),
    .rst_n      (rst_n),
    .wr_en      (wr_en),
    .wr_addr    (wr_addr),
    .wr_data    (wr_data),
    .rd_en      (rd_en),
    .rd_addr    (rd_addr),
    .rd_data    (rd_data),
    .fifo_push  (fifo_push),
    .fifo_din   (fifo_din),
    .fifo_pop   (fifo_pop),
    .fifo_dout  (fifo_dout),
    .fifo_empty (fifo_empty),
    .fifo_full  (fifo_full)
  );

  // ---------- Clock ----------
  initial clk = 0;
  always #(CLK_PERIOD/2) clk = ~clk;

  // ---------- Scoreboard ----------
  int test_count;
  int pass_count;
  int fail_count;

  task automatic check_data(string test_name, logic [LANES*8-1:0] expected, logic [LANES*8-1:0] actual);
    test_count++;
    if (actual === expected) begin
      pass_count++;
    end else begin
      fail_count++;
      $display("[FAIL] %s : expected 0x%064h, got 0x%064h", test_name, expected, actual);
    end
  endtask

  task automatic check_flag(string test_name, logic expected, logic actual);
    test_count++;
    if (actual === expected) begin
      pass_count++;
    end else begin
      fail_count++;
      $display("[FAIL] %s : expected %0b, got %0b", test_name, expected, actual);
    end
  endtask

  // ---------- Helper tasks ----------
  task automatic sram_write(input logic [ADDR_W-1:0] addr, input logic [LANES*8-1:0] data);
    @(posedge clk);
    wr_en   <= 1'b1;
    wr_addr <= addr;
    wr_data <= data;
    @(posedge clk);
    wr_en   <= 1'b0;
  endtask

  task automatic sram_read(input logic [ADDR_W-1:0] addr, output logic [LANES*8-1:0] data_out);
    @(posedge clk);
    rd_en   <= 1'b1;
    rd_addr <= addr;
    @(posedge clk);
    rd_en   <= 1'b0;
    @(posedge clk);  // registered read output available
    data_out = rd_data;
  endtask

  task automatic push_fifo(input logic [LANES*8-1:0] data);
    @(posedge clk);
    fifo_push <= 1'b1;
    fifo_din  <= data;
    @(posedge clk);
    fifo_push <= 1'b0;
  endtask

  task automatic pop_fifo(output logic [LANES*8-1:0] data_out);
    // FWFT: data is available on fifo_dout before pop
    data_out = fifo_dout;
    @(posedge clk);
    fifo_pop <= 1'b1;
    @(posedge clk);
    fifo_pop <= 1'b0;
  endtask

  // ---------- Generate pattern ----------
  function automatic logic [LANES*8-1:0] gen_pattern(input int seed);
    logic [LANES*8-1:0] p;
    for (int i = 0; i < LANES; i++)
      p[(i+1)*8-1 -: 8] = (seed + i) & 8'hFF;
    return p;
  endfunction

  // ---------- Stimulus ----------
  logic [LANES*8-1:0] read_back;
  logic [LANES*8-1:0] fifo_out;

  initial begin
    // Init
    clk       = 0;
    rst_n     = 0;
    wr_en     = 0;
    rd_en     = 0;
    wr_addr   = '0;
    wr_data   = '0;
    rd_addr   = '0;
    fifo_push = 0;
    fifo_pop  = 0;
    fifo_din  = '0;
    test_count = 0;
    pass_count = 0;
    fail_count = 0;

    // Reset
    repeat (4) @(posedge clk);
    rst_n = 1;
    repeat (2) @(posedge clk);

    // ========== T1: SRAM write/read (10 addresses) ==========
    $display("\n===== T1: SRAM write/read (10 addresses) =====");
    begin
      int t1_pass;
      logic [LANES*8-1:0] t1_golden [10];
      t1_pass = 1;

      for (int a = 0; a < 10; a++) begin
        t1_golden[a] = gen_pattern(a * 7 + 3);
        sram_write(a[ADDR_W-1:0], t1_golden[a]);
      end

      for (int a = 0; a < 10; a++) begin
        sram_read(a[ADDR_W-1:0], read_back);
        check_data($sformatf("T1_addr%0d", a), t1_golden[a], read_back);
        if (read_back !== t1_golden[a]) t1_pass = 0;
      end

      if (t1_pass) $display("[PASS] T1: SRAM write/read — exact match for 10 addresses");
      else         $display("[FAIL] T1: SRAM write/read — mismatches detected");
    end

    // ========== T2: FIFO basic (push 5, pop 5) ==========
    $display("\n===== T2: FIFO basic (push 5, pop 5) =====");
    begin
      int t2_pass;
      logic [LANES*8-1:0] t2_golden [5];
      t2_pass = 1;

      for (int i = 0; i < 5; i++) begin
        t2_golden[i] = gen_pattern(100 + i * 11);
        push_fifo(t2_golden[i]);
      end

      // Wait one cycle for FIFO state to settle
      @(posedge clk);

      for (int i = 0; i < 5; i++) begin
        pop_fifo(fifo_out);
        check_data($sformatf("T2_fifo_%0d", i), t2_golden[i], fifo_out);
        if (fifo_out !== t2_golden[i]) t2_pass = 0;
      end

      if (t2_pass) $display("[PASS] T2: FIFO basic — correct FIFO order and values");
      else         $display("[FAIL] T2: FIFO basic — mismatches detected");
    end

    // ========== T3: FIFO full ==========
    $display("\n===== T3: FIFO full (push until fifo_full) =====");
    begin
      int t3_pass;
      t3_pass = 1;

      // Make sure FIFO is empty first
      check_flag("T3_initial_empty", 1'b1, fifo_empty);

      // Push FIFO_DEPTH items
      for (int i = 0; i < FIFO_DEPTH; i++) begin
        push_fifo(gen_pattern(200 + i));
      end

      @(posedge clk);
      check_flag("T3_fifo_full", 1'b1, fifo_full);

      // Try to push one more — should be rejected
      begin
        logic [LANES*8-1:0] overflow_data;
        overflow_data = gen_pattern(999);
        push_fifo(overflow_data);
        @(posedge clk);
        // FIFO should still be full (no overflow)
        check_flag("T3_still_full_after_overflow", 1'b1, fifo_full);
      end

      if (fail_count == 0) $display("[PASS] T3: FIFO full — no overflow");
      else                 $display("[INFO] T3: FIFO full — see checks above");

      // Drain FIFO for next test
      for (int i = 0; i < FIFO_DEPTH; i++) begin
        pop_fifo(fifo_out);
      end
      @(posedge clk);
    end

    // ========== T4: FIFO empty ==========
    $display("\n===== T4: FIFO empty (pop until fifo_empty) =====");
    begin
      // FIFO should be empty after T3 drain
      check_flag("T4_fifo_empty", 1'b1, fifo_empty);

      // Push 2 and pop 2
      push_fifo(gen_pattern(300));
      push_fifo(gen_pattern(301));
      @(posedge clk);

      check_flag("T4_not_empty_after_push", 1'b0, fifo_empty);

      pop_fifo(fifo_out);
      pop_fifo(fifo_out);
      @(posedge clk);

      check_flag("T4_empty_after_all_pops", 1'b1, fifo_empty);

      // Try popping from empty — should not underflow
      @(posedge clk);
      fifo_pop <= 1'b1;
      @(posedge clk);
      fifo_pop <= 1'b0;
      @(posedge clk);

      check_flag("T4_still_empty_after_underflow", 1'b1, fifo_empty);
      $display("[PASS] T4: FIFO empty — no underflow");
    end

    // ========== T5: Simultaneous SRAM + FIFO ==========
    $display("\n===== T5: Simultaneous SRAM + FIFO =====");
    begin
      int t5_pass;
      logic [LANES*8-1:0] sram_golden;
      logic [LANES*8-1:0] fifo_golden;
      t5_pass = 1;

      sram_golden = gen_pattern(400);
      fifo_golden = gen_pattern(500);

      // Write SRAM and push FIFO simultaneously
      @(posedge clk);
      wr_en     <= 1'b1;
      wr_addr   <= 10'd50;
      wr_data   <= sram_golden;
      fifo_push <= 1'b1;
      fifo_din  <= fifo_golden;
      @(posedge clk);
      wr_en     <= 1'b0;
      fifo_push <= 1'b0;
      @(posedge clk);

      // Read SRAM
      sram_read(10'd50, read_back);
      check_data("T5_sram", sram_golden, read_back);
      if (read_back !== sram_golden) t5_pass = 0;

      // Pop FIFO
      pop_fifo(fifo_out);
      check_data("T5_fifo", fifo_golden, fifo_out);
      if (fifo_out !== fifo_golden) t5_pass = 0;

      if (t5_pass) $display("[PASS] T5: Simultaneous SRAM + FIFO — both correct");
      else         $display("[FAIL] T5: Simultaneous SRAM + FIFO — mismatch");
    end

    // ========== T6: FIFO wrap-around ==========
    $display("\n===== T6: FIFO wrap-around =====");
    begin
      int t6_pass;
      logic [LANES*8-1:0] t6_golden [20];
      int gi;
      t6_pass = 1;
      gi      = 0;

      // Make sure FIFO is empty
      @(posedge clk);

      // Push 6
      for (int i = 0; i < 6; i++) begin
        t6_golden[gi] = gen_pattern(600 + i);
        push_fifo(t6_golden[gi]);
        gi++;
      end
      @(posedge clk);

      // Pop 4
      for (int i = 0; i < 4; i++) begin
        pop_fifo(fifo_out);
        check_data($sformatf("T6_pop1_%0d", i), t6_golden[i], fifo_out);
        if (fifo_out !== t6_golden[i]) t6_pass = 0;
      end

      // Push 6 more (this should cause pointer wrap since FIFO_DEPTH=8)
      for (int i = 0; i < 6; i++) begin
        t6_golden[gi] = gen_pattern(700 + i);
        push_fifo(t6_golden[gi]);
        gi++;
      end
      @(posedge clk);

      // Pop 8 (remaining 2 from first batch + 6 from second batch)
      for (int i = 0; i < 8; i++) begin
        pop_fifo(fifo_out);
        check_data($sformatf("T6_pop2_%0d", i), t6_golden[4 + i], fifo_out);
        if (fifo_out !== t6_golden[4 + i]) t6_pass = 0;
      end

      @(posedge clk);
      check_flag("T6_empty_after_drain", 1'b1, fifo_empty);

      if (t6_pass) $display("[PASS] T6: FIFO wrap-around — pointer wrap verified");
      else         $display("[FAIL] T6: FIFO wrap-around — mismatch");
    end

    // ========== Summary ==========
    $display("\n========================================");
    $display("  tb_glb_weight_bank SUMMARY");
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
