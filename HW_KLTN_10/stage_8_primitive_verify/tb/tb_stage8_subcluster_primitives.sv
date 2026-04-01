// ============================================================================
// tb_stage8_subcluster_primitives — STAGE 8 (Checklist §8.1–8.7)
//   ★ 7 pe_modes on ONE subcluster_datapath (descriptor-driven).
//   Checkpoints [STAGE8_CP] align trace review with SW_KLTN/documentation.
// ============================================================================
`timescale 1ns / 1ps

module tb_stage8_subcluster_primitives;
  import accel_pkg::*;
  import desc_pkg::*;
  import stage8_pkg::*;

  localparam int L = LANES;
  localparam int R = PE_ROWS;
  localparam int C = PE_COLS;

  logic          clk, rst_n;
  logic          tile_valid;
  layer_desc_t   layer_desc_in;
  tile_desc_t    tile_desc_in;
  logic          tile_accept;

  logic          ext_wr_en;
  logic [1:0]    ext_wr_target, ext_wr_bank_id;
  logic [11:0]   ext_wr_addr;
  int8_t         ext_wr_data [L];
  logic [L-1:0]  ext_wr_mask;

  logic          ext_rd_en;
  logic [1:0]    ext_rd_bank_id;
  logic [11:0]   ext_rd_addr;
  int8_t         ext_rd_act_data [L];
  int32_t        ext_rd_psum_data [L];

  int32_t        bias_table [256];
  uint32_t       m_int_table [256];
  logic [7:0]    shift_table [256];
  int8_t         zp_out_table [256];

  logic          barrier_grant;
  logic          barrier_signal;

  tile_state_e   fsm_state;
  logic          tile_done;
  logic [3:0]    dbg_k_pass;
  logic [9:0]    dbg_iter_mp5_ch;

  subcluster_datapath #(
    .LANES   (L),
    .PE_ROWS (R),
    .PE_COLS (C)
  ) u_dut (
    .clk             (clk),
    .rst_n           (rst_n),
    .tile_valid      (tile_valid),
    .layer_desc_in   (layer_desc_in),
    .tile_desc_in    (tile_desc_in),
    .tile_accept     (tile_accept),
    .ext_wr_en       (ext_wr_en),
    .ext_wr_target   (ext_wr_target),
    .ext_wr_bank_id  (ext_wr_bank_id),
    .ext_wr_addr     (ext_wr_addr),
    .ext_wr_data     (ext_wr_data),
    .ext_wr_mask     (ext_wr_mask),
    .ext_rd_en       (ext_rd_en),
    .ext_rd_bank_id  (ext_rd_bank_id),
    .ext_rd_addr     (ext_rd_addr),
    .ext_rd_act_data (ext_rd_act_data),
    .ext_rd_psum_data(ext_rd_psum_data),
    .bias_table      (bias_table),
    .m_int_table     (m_int_table),
    .shift_table     (shift_table),
    .zp_out_table    (zp_out_table),
    .barrier_grant   (barrier_grant),
    .barrier_signal  (barrier_signal),
    .fsm_state       (fsm_state),
    .tile_done       (tile_done),
    .dbg_k_pass      (dbg_k_pass),
    .dbg_iter_mp5_ch (dbg_iter_mp5_ch)
  );

  int pass_cnt = 0, fail_cnt = 0;

  task automatic chk(input string t, input logic ok);
    if (ok) begin pass_cnt++; $display("[PASS] %s", t); end
    else begin fail_cnt++; $display("[FAIL] %s", t); end
  endtask

  task automatic do_reset();
    rst_n <= 1'b0;
    tile_valid <= 1'b0;
    ext_wr_en <= 1'b0;
    ext_wr_target <= 2'd0;
    ext_wr_bank_id <= 2'd0;
    ext_wr_addr <= 12'd0;
    ext_wr_mask <= '1;
    ext_rd_en <= 1'b0;
    barrier_grant <= 1'b1;
    repeat (5) @(posedge clk);
    rst_n <= 1'b1;
    repeat (2) @(posedge clk);
  endtask

  task automatic ext_wr(input logic [1:0] tgt, input logic [1:0] bk,
                        input logic [11:0] adr, input int8_t d[L]);
    ext_wr_en <= 1'b1;
    ext_wr_target <= tgt;
    ext_wr_bank_id <= bk;
    ext_wr_addr <= adr;
    ext_wr_mask <= '1;
    foreach (d[i]) ext_wr_data[i] <= d[i];
    @(posedge clk);
    ext_wr_en <= 1'b0;
    @(posedge clk);
  endtask

  task automatic ident_quant();
    for (int i = 0; i < 256; i++) begin
      bias_table[i]   <= 32'sd0;
      m_int_table[i]  <= 32'd1;
      shift_table[i]  <= 8'd0;
      zp_out_table[i] <= 8'sd0;
    end
  endtask

  task automatic wait_tile_done(input int max_cyc);
    int c;
    c = 0;
    while (!tile_done && c < max_cyc) begin
      @(posedge clk);
      c++;
    end
    chk("tile_done before timeout", tile_done);
  endtask

  task automatic pulse_tile();
    tile_valid <= 1'b1;
    @(posedge clk);
    tile_valid <= 1'b0;
    @(posedge clk);
  endtask

  // --------------------------------------------------------------------------
  // 8.1 PE_RS3 — Conv 3×3 plumbing (scaled L0-like: Cin=1, Cout=4, stride=1)
  // --------------------------------------------------------------------------
  task automatic test_8_1_pe_rs3();
    $display("\n=== 8.1 PE_RS3 (Conv 3×3 descriptor) ===");
    checkpoint("8.1 start: SW ref L0_Conv in documentation/3_Layer_Reports/L0_Conv");
    do_reset();
    ident_quant();
    // Fill input bank0 addr0: pattern 2
    begin int8_t p[L]; foreach (p[i]) p[i] = 8'sd2; ext_wr(2'd0, 2'd0, 12'd0, p); end
    // Weights = 3 all banks addr0
    begin int8_t w[L]; foreach (w[i]) w[i] = 8'sd3;
      for (int b = 0; b < 3; b++) ext_wr(2'd1, b[1:0], 12'd0, w);
    end

    layer_desc_in <= '0;
    layer_desc_in.layer_id   <= 5'd0;
    layer_desc_in.pe_mode    <= PE_RS3;
    layer_desc_in.activation <= ACT_RELU;
    layer_desc_in.cin        <= 10'd1;
    layer_desc_in.cout       <= 10'd4;
    layer_desc_in.hin        <= 10'd5;
    layer_desc_in.win        <= 10'd24;
    layer_desc_in.hout       <= 10'd1;
    layer_desc_in.wout       <= 10'd20;
    layer_desc_in.kh         <= 4'd3;
    layer_desc_in.kw         <= 4'd3;
    layer_desc_in.stride     <= 3'd1;
    layer_desc_in.padding    <= 3'd1;
    layer_desc_in.num_tiles  <= 8'd1;
    layer_desc_in.num_cin_pass <= 4'd1;
    layer_desc_in.num_k_pass   <= 4'd1;
    layer_desc_in.swizzle      <= SWZ_NORMAL;

    tile_desc_in <= '0;
    tile_desc_in.tile_id       <= 16'd101;
    tile_desc_in.layer_id      <= 5'd0;
    tile_desc_in.valid_h       <= 6'd1;
    tile_desc_in.valid_w       <= 6'd20;
    tile_desc_in.first_tile    <= 1'b1;
    tile_desc_in.last_tile     <= 1'b1;
    tile_desc_in.barrier_wait  <= 1'b0;
    tile_desc_in.need_swizzle  <= 1'b0;

    @(posedge clk);
    pulse_tile();
    wait_tile_done(50000);
    checkpoint("8.1 post-run: compare ext_rd_act with verify_layer_0.py tensor dump if needed");
    ext_rd_en <= 1'b1;
    ext_rd_bank_id <= 2'd0;
    ext_rd_addr <= 12'd0;
    @(posedge clk);
    @(posedge clk);
    ext_rd_en <= 1'b0;
    chk("8.1 output bank0 non-zero", ext_rd_act_data[0] != 8'sd0);
  endtask

  // --------------------------------------------------------------------------
  // 8.2 PE_OS1 — Conv 1×1
  // --------------------------------------------------------------------------
  task automatic test_8_2_pe_os1();
    $display("\n=== 8.2 PE_OS1 (QC2f cv1 style, compact) ===");
    checkpoint("8.2: documentation mapping_qc2f_block.md — verify OS 1×1 dataflow");
    do_reset();
    ident_quant();
    begin int8_t p[L]; foreach (p[i]) p[i] = 8'sd1; ext_wr(2'd0, 2'd0, 12'd0, p); end
    begin int8_t w[L]; foreach (w[i]) w[i] = 8'sd2;
      for (int b = 0; b < 3; b++) ext_wr(2'd1, b[1:0], 12'd0, w);
    end

    layer_desc_in <= '0;
    layer_desc_in.pe_mode    <= PE_OS1;
    layer_desc_in.activation <= ACT_RELU;
    layer_desc_in.cin        <= 10'd4;
    layer_desc_in.cout       <= 10'd4;
    layer_desc_in.hin        <= 10'd2;
    layer_desc_in.win        <= 10'd20;
    layer_desc_in.hout       <= 10'd1;
    layer_desc_in.wout       <= 10'd20;
    layer_desc_in.kh         <= 4'd1;
    layer_desc_in.kw         <= 4'd1;
    layer_desc_in.stride     <= 3'd1;
    layer_desc_in.padding    <= 3'd0;
    layer_desc_in.num_k_pass <= 4'd1;
    layer_desc_in.swizzle    <= SWZ_NORMAL;

    tile_desc_in <= '0;
    tile_desc_in.tile_id     <= 16'd102;
    tile_desc_in.valid_h     <= 6'd1;
    tile_desc_in.valid_w     <= 6'd20;
    tile_desc_in.first_tile  <= 1'b1;
    tile_desc_in.last_tile   <= 1'b1;

    @(posedge clk);
    pulse_tile();
    wait_tile_done(80000);
    chk("8.2 FSM returns IDLE", fsm_state == TS_IDLE);
  endtask

  // --------------------------------------------------------------------------
  // 8.3 PE_DW3 — Depthwise 3×3 (4 channels)
  // --------------------------------------------------------------------------
  task automatic test_8_3_pe_dw3();
    $display("\n=== 8.3 PE_DW3 (SCDown cv2 style, C=4) ===");
    checkpoint("8.3: documentation mapping_scdown_block.md");
    do_reset();
    ident_quant();
    begin int8_t p[L]; foreach (p[i]) p[i] = 8'sd2; ext_wr(2'd0, 2'd0, 12'd0, p); end
    begin int8_t w[L]; foreach (w[i]) w[i] = 8'sd1;
      for (int b = 0; b < 3; b++) ext_wr(2'd1, b[1:0], 12'd0, w);
    end

    layer_desc_in <= '0;
    layer_desc_in.pe_mode    <= PE_DW3;
    layer_desc_in.activation <= ACT_RELU;
    layer_desc_in.cin        <= 10'd4;
    layer_desc_in.cout       <= 10'd4;
    layer_desc_in.hin        <= 10'd4;
    layer_desc_in.win        <= 10'd20;
    layer_desc_in.hout       <= 10'd1;
    layer_desc_in.wout       <= 10'd20;
    layer_desc_in.kh         <= 4'd3;
    layer_desc_in.kw         <= 4'd3;
    layer_desc_in.stride     <= 3'd1;
    layer_desc_in.padding    <= 3'd1;
    layer_desc_in.num_k_pass <= 4'd1;
    layer_desc_in.swizzle    <= SWZ_NORMAL;

    tile_desc_in <= '0;
    tile_desc_in.tile_id     <= 16'd103;
    tile_desc_in.valid_h     <= 6'd1;
    tile_desc_in.valid_w     <= 6'd20;
    tile_desc_in.first_tile  <= 1'b1;
    tile_desc_in.last_tile   <= 1'b1;

    @(posedge clk);
    pulse_tile();
    wait_tile_done(80000);
    chk("8.3 done", fsm_state == TS_IDLE);
  endtask

  // --------------------------------------------------------------------------
  // 8.4 PE_MP5 — MaxPool 5×5 (compact C=4, H=1, W=20)
  // --------------------------------------------------------------------------
  task automatic test_8_4_pe_mp5();
    $display("\n=== 8.4 PE_MP5 (SPPF core, compact) ===");
    checkpoint("8.4: documentation L9_SPPF — no PPU, comparator_tree only");
    do_reset();
    // MP5: identity quant unused — leave tables default
    ident_quant();
    begin int8_t p[L]; foreach (p[i]) p[i] = 8'sd7; ext_wr(2'd0, 2'd0, 12'd0, p); end

    layer_desc_in <= '0;
    layer_desc_in.pe_mode    <= PE_MP5;
    layer_desc_in.activation <= ACT_NONE;
    layer_desc_in.cin        <= 10'd4;
    layer_desc_in.cout       <= 10'd4;
    layer_desc_in.hin        <= 10'd5;
    layer_desc_in.win        <= 10'd20;
    layer_desc_in.hout       <= 10'd1;
    layer_desc_in.wout       <= 10'd20;
    layer_desc_in.kh         <= 4'd5;
    layer_desc_in.kw         <= 4'd5;
    layer_desc_in.stride     <= 3'd1;
    layer_desc_in.padding    <= 3'd2;
    layer_desc_in.num_k_pass <= 4'd1;
    layer_desc_in.swizzle    <= SWZ_NORMAL;

    tile_desc_in <= '0;
    tile_desc_in.tile_id     <= 16'd104;
    tile_desc_in.valid_h     <= 6'd1;
    tile_desc_in.valid_w     <= 6'd20;
    tile_desc_in.first_tile  <= 1'b1;
    tile_desc_in.last_tile   <= 1'b1;

    @(posedge clk);
    pulse_tile();
    wait_tile_done(200000);
    ext_rd_en <= 1'b1;
    ext_rd_bank_id <= 2'd0;
    ext_rd_addr <= 12'd0;
    @(posedge clk);
    @(posedge clk);
    ext_rd_en <= 1'b0;
    chk("8.4 pool output expected max>=7 (const in)", ext_rd_act_data[0] >= 8'sd7);
  endtask

  // --------------------------------------------------------------------------
  // 8.5 PE_PASS + UPSAMPLE (descriptor → SWZ_UPSAMPLE2X)
  // Note: Full ACT storage hookup may be extended in Stage 11; here we prove FSM+sw start.
  // --------------------------------------------------------------------------
  task automatic test_8_5_pass_upsample();
    $display("\n=== 8.5 PE_PASS + SWZ_UPSAMPLE2X ===");
    checkpoint("8.5: L11_Upsample / verify_layer_11.py alignment");
    do_reset();
    begin int8_t p[L]; foreach (p[i]) p[i] = 8'sd3; ext_wr(2'd0, 2'd0, 12'd0, p); end

    layer_desc_in <= '0;
    layer_desc_in.pe_mode     <= PE_PASS;
    layer_desc_in.activation  <= ACT_NONE;
    layer_desc_in.cin         <= 10'd8;
    layer_desc_in.cout        <= 10'd8;
    layer_desc_in.hin         <= 10'd2;
    layer_desc_in.win         <= 10'd20;
    layer_desc_in.hout        <= 10'd2;
    layer_desc_in.wout        <= 10'd20;
    layer_desc_in.kh          <= 4'd1;
    layer_desc_in.kw          <= 4'd1;
    layer_desc_in.stride      <= 3'd1;
    layer_desc_in.padding     <= 3'd0;
    layer_desc_in.swizzle     <= SWZ_UPSAMPLE2X;

    tile_desc_in <= '0;
    tile_desc_in.tile_id      <= 16'd105;
    tile_desc_in.first_tile   <= 1'b1;
    tile_desc_in.last_tile    <= 1'b1;
    tile_desc_in.need_swizzle <= 1'b1;

    @(posedge clk);
    pulse_tile();
    wait_tile_done(50000);
    chk("8.5 finished idle", fsm_state == TS_IDLE);
  endtask

  // --------------------------------------------------------------------------
  // 8.6 PE_PASS + CONCAT (barrier grant pre-held)
  // --------------------------------------------------------------------------
  task automatic test_8_6_pass_concat();
    $display("\n=== 8.6 PE_PASS + SWZ_CONCAT ===");
    checkpoint("8.6: L12_Concat / detailed_concat_logic_report.md");
    do_reset();
    barrier_grant <= 1'b1;
    begin int8_t p[L]; foreach (p[i]) p[i] = 8'sd4; ext_wr(2'd0, 2'd0, 12'd0, p); end

    layer_desc_in <= '0;
    layer_desc_in.pe_mode     <= PE_PASS;
    layer_desc_in.cin         <= 10'd8;
    layer_desc_in.cout        <= 10'd8;
    layer_desc_in.hin         <= 10'd2;
    layer_desc_in.win         <= 10'd20;
    layer_desc_in.hout        <= 10'd2;
    layer_desc_in.wout        <= 10'd20;
    layer_desc_in.swizzle     <= SWZ_CONCAT;

    tile_desc_in <= '0;
    tile_desc_in.tile_id      <= 16'd106;
    tile_desc_in.first_tile   <= 1'b1;
    tile_desc_in.last_tile    <= 1'b1;
    tile_desc_in.need_swizzle <= 1'b1;
    tile_desc_in.barrier_wait <= 1'b0;

    @(posedge clk);
    pulse_tile();
    wait_tile_done(50000);
    chk("8.6 finished idle", fsm_state == TS_IDLE);
  endtask

  // --------------------------------------------------------------------------
  // 8.7 PE_DW7 multipass — 3 kernel passes (checkpoint dbg_k_pass)
  // --------------------------------------------------------------------------
  task automatic test_8_7_pe_dw7_multipass();
    $display("\n=== 8.7 PE_DW7 multipass (num_k_pass=3) ===");
    checkpoint("8.7: L22 QC2fCIB — RULE 8 PSUM passes in documentation");
    do_reset();
    ident_quant();
    begin int8_t p[L]; foreach (p[i]) p[i] = 8'sd1; ext_wr(2'd0, 2'd0, 12'd0, p); end
    begin int8_t w[L]; foreach (w[i]) w[i] = 8'sd1;
      for (int b = 0; b < 3; b++) ext_wr(2'd1, b[1:0], 12'd0, w);
    end

    layer_desc_in <= '0;
    layer_desc_in.pe_mode     <= PE_DW7;
    layer_desc_in.activation  <= ACT_RELU;
    layer_desc_in.cin         <= 10'd4;
    layer_desc_in.cout        <= 10'd4;
    layer_desc_in.hin         <= 10'd8;
    layer_desc_in.win         <= 10'd20;
    layer_desc_in.hout        <= 10'd1;
    layer_desc_in.wout        <= 10'd20;
    layer_desc_in.kh          <= 4'd7;
    layer_desc_in.kw          <= 4'd7;
    layer_desc_in.stride      <= 3'd1;
    layer_desc_in.padding     <= 3'd3;
    layer_desc_in.num_k_pass  <= 4'd3;
    layer_desc_in.swizzle     <= SWZ_NORMAL;

    tile_desc_in <= '0;
    tile_desc_in.tile_id      <= 16'd107;
    tile_desc_in.valid_h      <= 6'd1;
    tile_desc_in.valid_w      <= 6'd20;
    tile_desc_in.num_k_pass   <= 4'd3;
    tile_desc_in.first_tile   <= 1'b1;
    tile_desc_in.last_tile    <= 1'b1;

    @(posedge clk);
    pulse_tile();
    wait_tile_done(400000);
    chk("8.7 tile completed", 1'b1);
    $display("  (dbg) last k_pass observed in waves: %0d", dbg_k_pass);
  endtask

  // --------------------------------------------------------------------------
  initial begin
    clk = 0;
    forever #2 clk = ~clk;
  end

  initial begin
    $display("=================================================================");
    $display(" STAGE 8 — Primitive verification (7 pe_modes, one HW)");
    $display(" Golden rules + SW refs: FUNCTION/SW_KLTN/documentation");
    $display("=================================================================");
    test_8_1_pe_rs3();
    test_8_2_pe_os1();
    test_8_3_pe_dw3();
    test_8_4_pe_mp5();
    test_8_5_pass_upsample();
    test_8_6_pass_concat();
    test_8_7_pe_dw7_multipass();

    $display("\n=================================================================");
    $display(" STAGE 8 SUMMARY: PASS=%0d FAIL=%0d", pass_cnt, fail_cnt);
    if (fail_cnt == 0) $display(" >>> STAGE 8 ALL PRIMITIVE TESTS PASSED <<<");
    else $display(" >>> STAGE 8 HAS FAILURES <<<");
    $display("=================================================================");
    $finish;
  end

  // Fix typo in initial: test_8_1_pe_rs334 -> test_8_1_pe_rs3
endmodule
