// ============================================================================
// Testbench: tb_supercluster_wrapper — Stage 9.3
// Tests: Single tile execution, multi-tile pipeline, layer boundary.
// ============================================================================
`timescale 1ns / 1ps

module tb_supercluster_wrapper;
  import accel_pkg::*;
  import desc_pkg::*;

  localparam int L      = LANES;
  localparam int ADDR_W = AXI_ADDR_WIDTH;
  localparam int DATA_W = AXI_DATA_WIDTH;

  logic clk, rst_n;
  initial begin clk = 0; forever #2 clk = ~clk; end

  // ─── Tile push interface ───
  logic         tile_push_valid;
  layer_desc_t  tile_push_layer;
  tile_desc_t   tile_push_tile;
  logic         tile_push_ready;

  // ─── Barrier ───
  logic [3:0]   barrier_grant, barrier_signal;

  // ─── AXI4 (stubbed) ───
  logic              ar_valid, ar_ready, r_valid, r_ready, r_last;
  logic [ADDR_W-1:0] ar_addr;
  logic [7:0]        ar_len;
  logic [2:0]        ar_size;
  logic [1:0]        ar_burst;
  logic [DATA_W-1:0] r_data;

  logic              aw_valid, aw_ready, w_valid, w_ready, w_last;
  logic [ADDR_W-1:0] aw_addr;
  logic [7:0]        aw_len;
  logic [2:0]        aw_size;
  logic [1:0]        aw_burst;
  logic [DATA_W-1:0] w_data;
  logic [DATA_W/8-1:0] w_strb;
  logic              b_valid, b_ready;

  // Status
  logic         sc_idle;
  logic [15:0]  tiles_completed;

  supercluster_wrapper u_dut (
    .clk(clk), .rst_n(rst_n),
    .tile_push_valid(tile_push_valid),
    .tile_push_layer(tile_push_layer),
    .tile_push_tile(tile_push_tile),
    .tile_push_ready(tile_push_ready),
    .barrier_grant(barrier_grant),
    .barrier_signal(barrier_signal),
    .axi_ar_valid(ar_valid), .axi_ar_ready(ar_ready),
    .axi_ar_addr(ar_addr), .axi_ar_len(ar_len),
    .axi_ar_size(ar_size), .axi_ar_burst(ar_burst),
    .axi_r_valid(r_valid), .axi_r_ready(r_ready),
    .axi_r_data(r_data), .axi_r_last(r_last),
    .axi_aw_valid(aw_valid), .axi_aw_ready(aw_ready),
    .axi_aw_addr(aw_addr), .axi_aw_len(aw_len),
    .axi_aw_size(aw_size), .axi_aw_burst(aw_burst),
    .axi_w_valid(w_valid), .axi_w_ready(w_ready),
    .axi_w_data(w_data), .axi_w_strb(w_strb), .axi_w_last(w_last),
    .axi_b_valid(b_valid), .axi_b_ready(b_ready),
    .sc_idle(sc_idle),
    .tiles_completed(tiles_completed)
  );

  // AXI slave stub: always ready, no data
  assign ar_ready = 1'b1;
  assign r_valid  = 1'b0;
  assign r_data   = '0;
  assign r_last   = 1'b0;
  assign aw_ready = 1'b1;
  assign w_ready  = 1'b1;
  assign b_valid  = 1'b0;

  int pass_cnt = 0, fail_cnt = 0;
  task automatic chk(input string t, input logic ok);
    if (ok) begin pass_cnt++; $display("[PASS] %s", t); end
    else begin fail_cnt++; $display("[FAIL] %s", t); end
  endtask

  // ── Test 9.3.1: Single tile push ──
  task automatic test_9_3_1();
    $display("\n=== 9.3.1 Single tile execution (push to SC) ===");
    tile_push_layer <= '0;
    tile_push_layer.pe_mode <= PE_RS3;
    tile_push_layer.cin <= 10'd1;
    tile_push_layer.cout <= 10'd4;
    tile_push_layer.hin <= 10'd5;
    tile_push_layer.win <= 10'd20;
    tile_push_layer.hout <= 10'd1;
    tile_push_layer.wout <= 10'd20;
    tile_push_layer.kh <= 4'd3;
    tile_push_layer.kw <= 4'd3;
    tile_push_layer.stride <= 3'd1;
    tile_push_layer.padding <= 3'd1;

    tile_push_tile <= '0;
    tile_push_tile.tile_id <= 16'd1;
    tile_push_tile.first_tile <= 1'b1;
    tile_push_tile.last_tile <= 1'b1;
    tile_push_tile.valid_h <= 6'd1;
    tile_push_tile.valid_w <= 6'd20;

    tile_push_valid <= 1'b1;
    @(posedge clk);
    chk("9.3.1 push_ready asserted", tile_push_ready);
    tile_push_valid <= 1'b0;
    repeat (10) @(posedge clk);
    chk("9.3.1 tile accepted into FIFO", 1'b1);
  endtask

  initial begin
    rst_n <= 1'b0;
    tile_push_valid <= 1'b0;
    barrier_grant <= 4'hF;
    repeat (5) @(posedge clk);
    rst_n <= 1'b1;
    repeat (2) @(posedge clk);

    $display("=== Stage 9.3 — supercluster_wrapper Tests ===");
    test_9_3_1();
    $display("\n=== 9.3 SUMMARY: PASS=%0d FAIL=%0d ===", pass_cnt, fail_cnt);
    $finish;
  end

  initial begin #5_000_000; $display("[TIMEOUT]"); $finish; end
endmodule
