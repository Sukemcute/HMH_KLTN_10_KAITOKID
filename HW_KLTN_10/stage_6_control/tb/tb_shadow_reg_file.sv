// ============================================================================
// Testbench : tb_shadow_reg_file
// Project   : YOLOv10n INT8 Accelerator — V4-VC707
// Tests     : T6.2.1  Latch config (outputs match input after latch_en)
//             T6.2.2  Stability (outputs hold after inputs change)
// ============================================================================
`timescale 1ns / 1ps

module tb_shadow_reg_file;
  import accel_pkg::*;
  import desc_pkg::*;

  // ────────────────────────────────────────────────────────────
  // Clock & reset
  // ────────────────────────────────────────────────────────────
  logic clk, rst_n;
  initial clk = 0;
  always #2 clk = ~clk;  // 4 ns => 250 MHz

  // ────────────────────────────────────────────────────────────
  // DUT signals
  // ────────────────────────────────────────────────────────────
  logic          latch_en;
  layer_desc_t   layer_desc_in;
  tile_desc_t    tile_desc_in;

  pe_mode_e      o_pe_mode;
  act_mode_e     o_activation;
  logic [9:0]    o_cin, o_cout;
  logic [9:0]    o_hin, o_win;
  logic [9:0]    o_hout, o_wout;
  logic [3:0]    o_kh, o_kw;
  logic [2:0]    o_stride, o_padding;
  logic [3:0]    o_num_cin_pass, o_num_k_pass;
  swizzle_mode_e o_swizzle;
  int8_t         o_zp_x;
  logic [15:0]   o_tile_id;
  logic [4:0]    o_layer_id;
  logic          o_first_tile, o_last_tile;
  logic          o_hold_skip, o_need_swizzle;

  // ────────────────────────────────────────────────────────────
  // DUT instantiation
  // ────────────────────────────────────────────────────────────
  shadow_reg_file u_dut (
    .clk             (clk),
    .rst_n           (rst_n),
    .latch_en        (latch_en),
    .layer_desc_in   (layer_desc_in),
    .tile_desc_in    (tile_desc_in),
    .o_pe_mode       (o_pe_mode),
    .o_activation    (o_activation),
    .o_cin           (o_cin),
    .o_cout          (o_cout),
    .o_hin           (o_hin),
    .o_win           (o_win),
    .o_hout          (o_hout),
    .o_wout          (o_wout),
    .o_kh            (o_kh),
    .o_kw            (o_kw),
    .o_stride        (o_stride),
    .o_padding       (o_padding),
    .o_num_cin_pass  (o_num_cin_pass),
    .o_num_k_pass    (o_num_k_pass),
    .o_swizzle       (o_swizzle),
    .o_zp_x          (o_zp_x),
    .o_tile_id       (o_tile_id),
    .o_layer_id      (o_layer_id),
    .o_first_tile    (o_first_tile),
    .o_last_tile     (o_last_tile),
    .o_hold_skip     (o_hold_skip),
    .o_need_swizzle  (o_need_swizzle)
  );

  // ────────────────────────────────────────────────────────────
  // Test infrastructure
  // ────────────────────────────────────────────────────────────
  int pass_cnt = 0;
  int fail_cnt = 0;

  task automatic check(input string tag, input logic cond);
    if (cond) begin
      $display("[PASS] %s", tag);
      pass_cnt++;
    end else begin
      $display("[FAIL] %s", tag);
      fail_cnt++;
    end
  endtask

  task automatic do_reset();
    rst_n    <= 1'b0;
    latch_en <= 1'b0;
    layer_desc_in <= '0;
    tile_desc_in  <= '0;
    repeat (4) @(posedge clk);
    rst_n <= 1'b1;
    @(posedge clk);
  endtask

  // ================================================================
  // T6.2.1 — Latch config: set descriptors, pulse latch_en,
  //          verify all outputs match inputs.
  // ================================================================
  task automatic test_T6_2_1();
    $display("\n===== T6.2.1: Latch config =====");
    do_reset();

    // Build descriptors with known values
    layer_desc_in <= '0;
    layer_desc_in.pe_mode      <= PE_DW3;
    layer_desc_in.activation   <= ACT_RELU;
    layer_desc_in.cin          <= 10'd64;
    layer_desc_in.cout         <= 10'd128;
    layer_desc_in.hin          <= 10'd42;
    layer_desc_in.win          <= 10'd22;
    layer_desc_in.hout         <= 10'd40;
    layer_desc_in.wout         <= 10'd20;
    layer_desc_in.kh           <= 4'd3;
    layer_desc_in.kw           <= 4'd3;
    layer_desc_in.stride       <= 3'd1;
    layer_desc_in.padding      <= 3'd1;
    layer_desc_in.num_cin_pass <= 4'd2;
    layer_desc_in.num_k_pass   <= 4'd1;
    layer_desc_in.swizzle      <= SWZ_UPSAMPLE2X;

    tile_desc_in <= '0;
    tile_desc_in.tile_id       <= 16'hBEEF;
    tile_desc_in.layer_id      <= 5'd7;
    tile_desc_in.first_tile    <= 1'b1;
    tile_desc_in.last_tile     <= 1'b0;
    tile_desc_in.hold_skip     <= 1'b1;
    tile_desc_in.need_swizzle  <= 1'b1;

    @(posedge clk);  // Let inputs settle

    // Pulse latch_en for 1 cycle
    latch_en <= 1'b1;
    @(posedge clk);
    latch_en <= 1'b0;
    @(posedge clk);  // Wait for registered output to appear

    // Verify all outputs
    check("T6.2.1-a pe_mode=PE_DW3",       o_pe_mode === PE_DW3);
    check("T6.2.1-b activation=ACT_RELU",   o_activation === ACT_RELU);
    check("T6.2.1-c cin=64",                o_cin === 10'd64);
    check("T6.2.1-d cout=128",              o_cout === 10'd128);
    check("T6.2.1-e hin=42",                o_hin === 10'd42);
    check("T6.2.1-f win=22",                o_win === 10'd22);
    check("T6.2.1-g hout=40",               o_hout === 10'd40);
    check("T6.2.1-h wout=20",               o_wout === 10'd20);
    check("T6.2.1-i kh=3",                  o_kh === 4'd3);
    check("T6.2.1-j kw=3",                  o_kw === 4'd3);
    check("T6.2.1-k stride=1",              o_stride === 3'd1);
    check("T6.2.1-l padding=1",             o_padding === 3'd1);
    check("T6.2.1-m num_cin_pass=2",        o_num_cin_pass === 4'd2);
    check("T6.2.1-n num_k_pass=1",          o_num_k_pass === 4'd1);
    check("T6.2.1-o swizzle=UPSAMPLE2X",    o_swizzle === SWZ_UPSAMPLE2X);
    check("T6.2.1-p zp_x=0 (default)",      o_zp_x === 8'sd0);
    check("T6.2.1-q tile_id=0xBEEF",        o_tile_id === 16'hBEEF);
    check("T6.2.1-r layer_id=7",            o_layer_id === 5'd7);
    check("T6.2.1-s first_tile=1",          o_first_tile === 1'b1);
    check("T6.2.1-t last_tile=0",           o_last_tile === 1'b0);
    check("T6.2.1-u hold_skip=1",           o_hold_skip === 1'b1);
    check("T6.2.1-v need_swizzle=1",        o_need_swizzle === 1'b1);
  endtask

  // ================================================================
  // T6.2.2 — Stability: after latch, change inputs =>
  //          verify outputs DON'T change until next latch.
  // ================================================================
  task automatic test_T6_2_2();
    pe_mode_e    snap_mode;
    logic [9:0]  snap_cin;
    logic [9:0]  snap_cout;
    logic [3:0]  snap_kw;
    logic [15:0] snap_tid;
    $display("\n===== T6.2.2: Stability after latch =====");
    do_reset();

    // --- Phase 1: Latch first set of values ---
    layer_desc_in <= '0;
    layer_desc_in.pe_mode <= PE_RS3;
    layer_desc_in.cin     <= 10'd16;
    layer_desc_in.cout    <= 10'd32;
    layer_desc_in.kw      <= 4'd3;

    tile_desc_in <= '0;
    tile_desc_in.tile_id <= 16'hAAAA;

    @(posedge clk);
    latch_en <= 1'b1;
    @(posedge clk);
    latch_en <= 1'b0;
    @(posedge clk);

    // Snapshot outputs after latch
    snap_mode = o_pe_mode;
    snap_cin  = o_cin;
    snap_cout = o_cout;
    snap_kw   = o_kw;
    snap_tid  = o_tile_id;

    check("T6.2.2-a initial pe_mode=PE_RS3", snap_mode === PE_RS3);
    check("T6.2.2-b initial cin=16",         snap_cin === 10'd16);

    // --- Phase 2: Change inputs WITHOUT latching ---
    layer_desc_in.pe_mode <= PE_OS1;
    layer_desc_in.cin     <= 10'd99;
    layer_desc_in.cout    <= 10'd255;
    layer_desc_in.kw      <= 4'd1;
    tile_desc_in.tile_id  <= 16'h5555;

    // Wait several cycles — outputs should NOT change
    repeat (10) @(posedge clk);

    check("T6.2.2-c pe_mode stable (PE_RS3)", o_pe_mode === PE_RS3);
    check("T6.2.2-d cin stable (16)",          o_cin === 10'd16);
    check("T6.2.2-e cout stable (32)",         o_cout === 10'd32);
    check("T6.2.2-f kw stable (3)",            o_kw === 4'd3);
    check("T6.2.2-g tile_id stable (0xAAAA)",  o_tile_id === 16'hAAAA);

    // --- Phase 3: Latch again — NOW outputs should update ---
    @(posedge clk);
    latch_en <= 1'b1;
    @(posedge clk);
    latch_en <= 1'b0;
    @(posedge clk);

    check("T6.2.2-h pe_mode updated (PE_OS1)", o_pe_mode === PE_OS1);
    check("T6.2.2-i cin updated (99)",          o_cin === 10'd99);
    check("T6.2.2-j cout updated (255)",        o_cout === 10'd255);
    check("T6.2.2-k kw updated (1)",            o_kw === 4'd1);
    check("T6.2.2-l tile_id updated (0x5555)",  o_tile_id === 16'h5555);
  endtask

  // ────────────────────────────────────────────────────────────
  // Test runner
  // ────────────────────────────────────────────────────────────
  initial begin
    $display("========================================");
    $display(" tb_shadow_reg_file — Stage 6 Control");
    $display("========================================");

    test_T6_2_1();
    test_T6_2_2();

    $display("\n========================================");
    $display(" SUMMARY: %0d tests, %0d PASS, %0d FAIL",
             pass_cnt + fail_cnt, pass_cnt, fail_cnt);
    if (fail_cnt == 0)
      $display(" >>> ALL TESTS PASSED <<<");
    else
      $display(" >>> SOME TESTS FAILED <<<");
    $display("========================================");
    $finish;
  end

  // Timeout
  initial begin
    #20000;
    $display("[TIMEOUT] Simulation exceeded 20 us");
    $finish;
  end

endmodule
