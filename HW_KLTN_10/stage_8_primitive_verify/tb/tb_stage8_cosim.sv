// ============================================================================
// Testbench : tb_stage8_cosim
// Stage 8 — Byte-exact cosimulation with debug tracing via DUT ports.
// ============================================================================
`timescale 1ns / 1ps

module tb_stage8_cosim;
  import accel_pkg::*;
  import desc_pkg::*;

  localparam int CLK_LANES = accel_pkg::LANES;

  logic clk = 0;
  always #2 clk = ~clk;
  logic rst_n;

  logic          tile_valid;
  layer_desc_t   layer_desc_in;
  tile_desc_t    tile_desc_in;
  logic          tile_accept;

  logic          ext_wr_en;
  logic [1:0]    ext_wr_target;
  logic [1:0]    ext_wr_bank_id;
  logic [11:0]   ext_wr_addr;
  int8_t         ext_wr_data [CLK_LANES];
  logic [CLK_LANES-1:0] ext_wr_mask;

  logic          ext_rd_en;
  logic [1:0]    ext_rd_bank_id;
  logic [11:0]   ext_rd_addr;
  int8_t         ext_rd_act_data  [CLK_LANES];
  int32_t        ext_rd_psum_data [CLK_LANES];

  int32_t        bias_table    [256];
  uint32_t       m_int_table   [256];
  logic [7:0]    shift_table   [256];
  int8_t         zp_out_table  [256];

  logic          barrier_grant;
  logic          barrier_signal;
  tile_state_e   fsm_state;
  logic          tile_done;
  logic [3:0]    dbg_k_pass;
  logic [9:0]    dbg_iter_mp5_ch;

  // Debug ports from DUT
  logic          dbg_seq_ppu_trigger;
  logic          dbg_ppu_trigger_delayed;
  logic          dbg_ppu_act_valid_0;
  logic          dbg_pe_psum_valid;
  logic          dbg_glb_act_wr_en_0;
  logic          dbg_pe_pool_valid;
  logic          dbg_seq_pe_enable;
  logic [3:0]    dbg_fsm_mode_reg;
  logic          dbg_ppu_done_latch;
  logic [9:0]    dbg_seq_ppu_cout_base;
  int32_t        dbg_pe_col_psum_0_0;
  int8_t         dbg_ppu_act_out_0_0;

  subcluster_datapath #(
    .LANES  (CLK_LANES),
    .PE_ROWS(3),
    .PE_COLS(4)
  ) u_dut (
    .clk              (clk),
    .rst_n            (rst_n),
    .tile_valid       (tile_valid),
    .layer_desc_in    (layer_desc_in),
    .tile_desc_in     (tile_desc_in),
    .tile_accept      (tile_accept),
    .ext_wr_en        (ext_wr_en),
    .ext_wr_target    (ext_wr_target),
    .ext_wr_bank_id   (ext_wr_bank_id),
    .ext_wr_addr      (ext_wr_addr),
    .ext_wr_data      (ext_wr_data),
    .ext_wr_mask      (ext_wr_mask),
    .ext_rd_en        (ext_rd_en),
    .ext_rd_bank_id   (ext_rd_bank_id),
    .ext_rd_addr      (ext_rd_addr),
    .ext_rd_act_data  (ext_rd_act_data),
    .ext_rd_psum_data (ext_rd_psum_data),
    .bias_table       (bias_table),
    .m_int_table      (m_int_table),
    .shift_table      (shift_table),
    .zp_out_table     (zp_out_table),
    .barrier_grant    (barrier_grant),
    .barrier_signal   (barrier_signal),
    .fsm_state        (fsm_state),
    .tile_done        (tile_done),
    .dbg_k_pass       (dbg_k_pass),
    .dbg_iter_mp5_ch  (dbg_iter_mp5_ch),
    .dbg_seq_ppu_trigger    (dbg_seq_ppu_trigger),
    .dbg_ppu_trigger_delayed(dbg_ppu_trigger_delayed),
    .dbg_ppu_act_valid_0    (dbg_ppu_act_valid_0),
    .dbg_pe_psum_valid      (dbg_pe_psum_valid),
    .dbg_glb_act_wr_en_0    (dbg_glb_act_wr_en_0),
    .dbg_pe_pool_valid      (dbg_pe_pool_valid),
    .dbg_seq_pe_enable      (dbg_seq_pe_enable),
    .dbg_fsm_mode_reg       (dbg_fsm_mode_reg),
    .dbg_ppu_done_latch     (dbg_ppu_done_latch),
    .dbg_seq_ppu_cout_base  (dbg_seq_ppu_cout_base),
    .dbg_pe_col_psum_0_0    (dbg_pe_col_psum_0_0),
    .dbg_ppu_act_out_0_0    (dbg_ppu_act_out_0_0)
  );

  reg [7:0] mem_buf [0:65535];
  reg [7:0] exp_buf [0:65535];
  integer total_pass;
  integer total_fail;

  // ══════════════════════════════════════════════════════════
  //  DEBUG: concurrent monitor using DUT debug ports only
  // ══════════════════════════════════════════════════════════
  integer cyc_cnt;
  reg dbg_en;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) cyc_cnt <= 0;
    else        cyc_cnt <= cyc_cnt + 1;
  end

  always @(posedge clk) begin
    if (dbg_en && rst_n) begin
      if (dbg_seq_ppu_trigger)
        $display("  [CP-A] cyc=%0d  seq_ppu_trigger  cout_base=%0d",
                 cyc_cnt, dbg_seq_ppu_cout_base);

      if (dbg_ppu_trigger_delayed)
        $display("  [CP-B] cyc=%0d  ppu_trigger_delayed  psum[0][0]=%0d",
                 cyc_cnt, dbg_pe_col_psum_0_0);

      if (dbg_ppu_act_valid_0)
        $display("  [CP-C] cyc=%0d  ppu_act_valid[0]  act_out[0][0]=%0d  wr_en0=%b",
                 cyc_cnt, dbg_ppu_act_out_0_0, dbg_glb_act_wr_en_0);

      if (dbg_pe_psum_valid)
        $display("  [CP-D] cyc=%0d  pe_psum_valid  mode_reg=%0d",
                 cyc_cnt, dbg_fsm_mode_reg);

      if (dbg_pe_pool_valid)
        $display("  [CP-E] cyc=%0d  pe_pool_valid  wr_en0=%b",
                 cyc_cnt, dbg_glb_act_wr_en_0);

      if (dbg_glb_act_wr_en_0 && !ext_wr_en && !ext_rd_en
          && !dbg_ppu_act_valid_0 && !dbg_pe_pool_valid)
        $display("  [CP-F] cyc=%0d  UNEXPECTED act_wr_en[0]=1", cyc_cnt);
    end
  end

  // ══════════════════════════════════════════════════════════
  //  Helper tasks
  // ══════════════════════════════════════════════════════════
  task do_reset;
    integer i;
  begin
    dbg_en = 0;
    rst_n         = 0;
    tile_valid    = 0;
    ext_wr_en     = 0;
    ext_rd_en     = 0;
    ext_wr_mask   = {CLK_LANES{1'b1}};
    barrier_grant = 1;
    layer_desc_in = '0;
    tile_desc_in  = '0;
    for (i = 0; i < CLK_LANES; i = i + 1)
      ext_wr_data[i] = 8'sd0;
    for (i = 0; i < 256; i = i + 1) begin
      bias_table[i]   = 32'sd0;
      m_int_table[i]  = 32'd1;
      shift_table[i]  = 8'd0;
      zp_out_table[i] = 8'sd0;
    end
    repeat (10) @(posedge clk);
    rst_n = 1;
    repeat (5) @(posedge clk);
    #1;
  end
  endtask

  task load_bank;
    input integer target;
    input integer bank;
    input integer num_addr;
    integer a, l;
  begin
    for (a = 0; a < num_addr; a = a + 1) begin
      for (l = 0; l < CLK_LANES; l = l + 1)
        ext_wr_data[l] = mem_buf[a * CLK_LANES + l];
      ext_wr_en      = 1;
      ext_wr_target  = target[1:0];
      ext_wr_bank_id = bank[1:0];
      ext_wr_addr    = a[11:0];
      @(posedge clk); #1;
    end
    ext_wr_en = 0;
    @(posedge clk); #1;
  end
  endtask

  task set_quant_params;
    input integer num_ch;
    input integer shift_val;
    integer ch;
  begin
    for (ch = 0; ch < num_ch; ch = ch + 1) begin
      bias_table[ch]   = 32'sd0;
      m_int_table[ch]  = 32'd1;
      shift_table[ch]  = shift_val[7:0];
      zp_out_table[ch] = 8'sd0;
    end
  end
  endtask

  task fire_tile;
    integer wait_cnt;
    reg [3:0] prev_state;
  begin
    dbg_en = 1;
    $display("  [DESC] pe_mode=%0d cin=%0d cout=%0d hin=%0d win=%0d kh=%0d kw=%0d stride=%0d pad=%0d",
             layer_desc_in.pe_mode, layer_desc_in.cin, layer_desc_in.cout,
             layer_desc_in.hin, layer_desc_in.win,
             layer_desc_in.kh, layer_desc_in.kw, layer_desc_in.stride, layer_desc_in.padding);

    tile_valid = 1;
    @(posedge clk); #1;
    wait_cnt = 0;
    while (fsm_state == 0 && wait_cnt < 100) begin
      @(posedge clk); #1;
      wait_cnt = wait_cnt + 1;
    end
    tile_valid = 0;
    if (wait_cnt >= 100) begin
      $display("ERROR: tile_accept timeout");
      $finish;
    end
    $display("  [ACCEPT] accepted, mode_reg=%0d ppu_done_latch=%b",
             dbg_fsm_mode_reg, dbg_ppu_done_latch);

    prev_state = fsm_state;
    wait_cnt = 0;
    while (!tile_done && wait_cnt < 50000) begin
      @(posedge clk); #1;
      wait_cnt = wait_cnt + 1;
      if (fsm_state !== prev_state) begin
        $display("  [FSM] %0d -> %0d at cycle %0d", prev_state, fsm_state, wait_cnt);
        prev_state = fsm_state;
      end
      if (wait_cnt == 49999)
        $display("  [TIMEOUT] state=%0d k_pass=%0d mp5_ch=%0d mode_reg=%0d ppu_done_latch=%b",
                 fsm_state, dbg_k_pass, dbg_iter_mp5_ch,
                 dbg_fsm_mode_reg, dbg_ppu_done_latch);
    end
    if (wait_cnt >= 50000) begin
      $display("ERROR: tile_done timeout after %0d cycles (state=%0d)", wait_cnt, fsm_state);
      dbg_en = 0;
      $finish;
    end
    $display("  tile_done after %0d cycles", wait_cnt);
    dbg_en = 0;
    repeat (10) @(posedge clk);
  end
  endtask

  task compare_bank;
    input integer bank;
    input integer num_addr;
    integer a, l, mismatches;
    reg signed [7:0] got_val;
    reg signed [7:0] exp_val;
  begin
    mismatches = 0;
    for (a = 0; a < num_addr; a = a + 1) begin
      ext_rd_en      = 1;
      ext_rd_bank_id = bank[1:0];
      ext_rd_addr    = a[11:0];
      @(posedge clk); #1;
      @(posedge clk); #1;
      for (l = 0; l < CLK_LANES; l = l + 1) begin
        got_val = ext_rd_act_data[l];
        exp_val = exp_buf[a * CLK_LANES + l];
        if (got_val !== exp_val) begin
          if (mismatches < 5)
            $display("    MISMATCH bank%0d a=%0d l=%0d  got=%0d(0x%02x) exp=%0d(0x%02x)",
                     bank, a, l, got_val, got_val[7:0], exp_val, exp_val[7:0]);
          mismatches = mismatches + 1;
        end
      end
    end
    ext_rd_en = 0;
    @(posedge clk); #1;
    if (mismatches == 0) begin
      $display("  [PASS] bank%0d — %0d addr verified", bank, num_addr);
      total_pass = total_pass + 1;
    end else begin
      $display("  [FAIL] bank%0d — %0d mismatches / %0d bytes",
               bank, mismatches, num_addr * CLK_LANES);
      total_fail = total_fail + 1;
    end
  end
  endtask

  // ══════════════════════════════════════════════════════════
  //  TEST: RS3
  // ══════════════════════════════════════════════════════════
  task test_rs3;
  begin
    $display("\n═══════════ TEST RS3 ═══════════");
    do_reset;

    $readmemh("vectors/rs3/input_bank0.memh", mem_buf);  load_bank(0, 0, 1);
    $readmemh("vectors/rs3/input_bank1.memh", mem_buf);  load_bank(0, 1, 1);
    $readmemh("vectors/rs3/input_bank2.memh", mem_buf);  load_bank(0, 2, 1);
    $readmemh("vectors/rs3/weight_bank0.memh", mem_buf); load_bank(1, 0, 12);
    $readmemh("vectors/rs3/weight_bank1.memh", mem_buf); load_bank(1, 1, 12);
    $readmemh("vectors/rs3/weight_bank2.memh", mem_buf); load_bank(1, 2, 12);
    set_quant_params(4, 10);

    layer_desc_in            = '0;
    layer_desc_in.pe_mode    = PE_RS3;
    layer_desc_in.activation = ACT_RELU;
    layer_desc_in.cin        = 10'd1;
    layer_desc_in.cout       = 10'd4;
    layer_desc_in.hin        = 10'd3;
    layer_desc_in.win        = 10'd20;
    layer_desc_in.hout       = 10'd3;
    layer_desc_in.wout       = 10'd20;
    layer_desc_in.kh         = 4'd3;
    layer_desc_in.kw         = 4'd3;
    layer_desc_in.stride     = 3'd1;
    layer_desc_in.padding    = 3'd1;
    layer_desc_in.num_cin_pass = 4'd1;
    layer_desc_in.num_k_pass   = 4'd1;

    tile_desc_in             = '0;
    tile_desc_in.first_tile  = 1'b1;
    tile_desc_in.last_tile   = 1'b1;

    fire_tile;

    $readmemh("vectors/rs3/expected_out_bank0.memh", exp_buf); compare_bank(0, 3);
    $readmemh("vectors/rs3/expected_out_bank1.memh", exp_buf); compare_bank(1, 3);
    $readmemh("vectors/rs3/expected_out_bank2.memh", exp_buf); compare_bank(2, 3);
    $readmemh("vectors/rs3/expected_out_bank3.memh", exp_buf); compare_bank(3, 3);

    $display("═══════════ RS3 DONE ═══════════\n");
  end
  endtask

  // ══════════════════════════════════════════════════════════
  //  TEST: OS1
  // ══════════════════════════════════════════════════════════
  task test_os1;
  begin
    $display("\n═══════════ TEST OS1 ═══════════");
    do_reset;

    $readmemh("vectors/os1/input_bank0.memh", mem_buf);  load_bank(0, 0, 4);
    $readmemh("vectors/os1/input_bank1.memh", mem_buf);  load_bank(0, 1, 4);
    $readmemh("vectors/os1/input_bank2.memh", mem_buf);  load_bank(0, 2, 4);
    $readmemh("vectors/os1/weight_bank0.memh", mem_buf); load_bank(1, 0, 16);
    set_quant_params(4, 12);

    layer_desc_in            = '0;
    layer_desc_in.pe_mode    = PE_OS1;
    layer_desc_in.activation = ACT_RELU;
    layer_desc_in.cin        = 10'd4;
    layer_desc_in.cout       = 10'd4;
    layer_desc_in.hin        = 10'd3;
    layer_desc_in.win        = 10'd20;
    layer_desc_in.hout       = 10'd3;
    layer_desc_in.wout       = 10'd20;
    layer_desc_in.kh         = 4'd1;
    layer_desc_in.kw         = 4'd1;
    layer_desc_in.stride     = 3'd1;
    layer_desc_in.padding    = 3'd0;
    layer_desc_in.num_cin_pass = 4'd1;
    layer_desc_in.num_k_pass   = 4'd1;

    tile_desc_in             = '0;
    tile_desc_in.first_tile  = 1'b1;
    tile_desc_in.last_tile   = 1'b1;

    fire_tile;

    $readmemh("vectors/os1/expected_out_bank0.memh", exp_buf); compare_bank(0, 3);
    $readmemh("vectors/os1/expected_out_bank1.memh", exp_buf); compare_bank(1, 3);
    $readmemh("vectors/os1/expected_out_bank2.memh", exp_buf); compare_bank(2, 3);
    $readmemh("vectors/os1/expected_out_bank3.memh", exp_buf); compare_bank(3, 3);

    $display("═══════════ OS1 DONE ═══════════\n");
  end
  endtask

  // ══════════════════════════════════════════════════════════
  //  TEST: DW3
  // ══════════════════════════════════════════════════════════
  task test_dw3;
  begin
    $display("\n═══════════ TEST DW3 ═══════════");
    do_reset;

    $readmemh("vectors/dw3/input_bank0.memh", mem_buf);  load_bank(0, 0, 4);
    $readmemh("vectors/dw3/input_bank1.memh", mem_buf);  load_bank(0, 1, 4);
    $readmemh("vectors/dw3/input_bank2.memh", mem_buf);  load_bank(0, 2, 4);
    $readmemh("vectors/dw3/weight_bank0.memh", mem_buf); load_bank(1, 0, 12);
    $readmemh("vectors/dw3/weight_bank1.memh", mem_buf); load_bank(1, 1, 12);
    $readmemh("vectors/dw3/weight_bank2.memh", mem_buf); load_bank(1, 2, 12);
    set_quant_params(4, 10);

    layer_desc_in            = '0;
    layer_desc_in.pe_mode    = PE_DW3;
    layer_desc_in.activation = ACT_RELU;
    layer_desc_in.cin        = 10'd4;
    layer_desc_in.cout       = 10'd4;
    layer_desc_in.hin        = 10'd3;
    layer_desc_in.win        = 10'd20;
    layer_desc_in.hout       = 10'd3;
    layer_desc_in.wout       = 10'd20;
    layer_desc_in.kh         = 4'd3;
    layer_desc_in.kw         = 4'd3;
    layer_desc_in.stride     = 3'd1;
    layer_desc_in.padding    = 3'd1;
    layer_desc_in.num_cin_pass = 4'd1;
    layer_desc_in.num_k_pass   = 4'd1;

    tile_desc_in             = '0;
    tile_desc_in.first_tile  = 1'b1;
    tile_desc_in.last_tile   = 1'b1;

    fire_tile;

    $readmemh("vectors/dw3/expected_out_bank0.memh", exp_buf); compare_bank(0, 3);
    $readmemh("vectors/dw3/expected_out_bank1.memh", exp_buf); compare_bank(1, 3);
    $readmemh("vectors/dw3/expected_out_bank2.memh", exp_buf); compare_bank(2, 3);
    $readmemh("vectors/dw3/expected_out_bank3.memh", exp_buf); compare_bank(3, 3);

    $display("═══════════ DW3 DONE ═══════════\n");
  end
  endtask

  // ══════════════════════════════════════════════════════════
  //  TEST: MP5
  // ══════════════════════════════════════════════════════════
  task test_mp5;
  begin
    $display("\n═══════════ TEST MP5 ═══════════");
    do_reset;

    $readmemh("vectors/mp5/input_bank0.memh", mem_buf);  load_bank(0, 0, 4);
    $readmemh("vectors/mp5/input_bank1.memh", mem_buf);  load_bank(0, 1, 4);
    $readmemh("vectors/mp5/input_bank2.memh", mem_buf);  load_bank(0, 2, 4);
    set_quant_params(4, 0);

    layer_desc_in            = '0;
    layer_desc_in.pe_mode    = PE_MP5;
    layer_desc_in.activation = ACT_NONE;
    layer_desc_in.cin        = 10'd4;
    layer_desc_in.cout       = 10'd4;
    layer_desc_in.hin        = 10'd3;
    layer_desc_in.win        = 10'd20;
    layer_desc_in.hout       = 10'd3;
    layer_desc_in.wout       = 10'd20;
    layer_desc_in.kh         = 4'd5;
    layer_desc_in.kw         = 4'd5;
    layer_desc_in.stride     = 3'd1;
    layer_desc_in.padding    = 3'd2;
    layer_desc_in.num_cin_pass = 4'd1;
    layer_desc_in.num_k_pass   = 4'd1;

    tile_desc_in             = '0;
    tile_desc_in.first_tile  = 1'b1;
    tile_desc_in.last_tile   = 1'b1;

    fire_tile;

    $readmemh("vectors/mp5/expected_out_bank0.memh", exp_buf); compare_bank(0, 3);
    $readmemh("vectors/mp5/expected_out_bank1.memh", exp_buf); compare_bank(1, 3);
    $readmemh("vectors/mp5/expected_out_bank2.memh", exp_buf); compare_bank(2, 3);
    $readmemh("vectors/mp5/expected_out_bank3.memh", exp_buf); compare_bank(3, 3);

    $display("═══════════ MP5 DONE ═══════════\n");
  end
  endtask

  // ══════════════════════════════════════════════════════════
  //  Main
  // ══════════════════════════════════════════════════════════
  initial begin
    total_pass = 0;
    total_fail = 0;
    dbg_en = 0;

`ifdef RTL_TRACE
    rtl_trace_pkg::rtl_trace_open("rtl_cycle_trace.log");
`endif

    test_rs3;
    test_os1;
    test_dw3;
    test_mp5;

    $display("\n==========================================");
    $display("  TOTAL:  %0d PASS  /  %0d FAIL", total_pass, total_fail);
    $display("==========================================");
    if (total_fail == 0)
      $display("ALL TESTS PASSED");
    else
      $display("SOME TESTS FAILED");
`ifdef RTL_TRACE
    rtl_trace_pkg::rtl_trace_close();
`endif
    $finish;
  end

endmodule
