// ============================================================================
// Module : tile_ingress_fifo
// Project: YOLOv10n INT8 Accelerator — V4-VC707
//
// 8-deep synchronous FIFO for tile descriptors.
// Buffers tile+layer descriptors between global_scheduler and local_arbiter.
// ============================================================================
`timescale 1ns / 1ps

module tile_ingress_fifo
  import accel_pkg::*;
  import desc_pkg::*;
#(
  parameter int DEPTH = 8
)(
  input  logic          clk,
  input  logic          rst_n,

  // ═══════════ PUSH (from global_scheduler) ═══════════
  input  logic          push_valid,
  input  layer_desc_t   push_layer,
  input  tile_desc_t    push_tile,
  output logic          push_ready,

  // ═══════════ POP (to local_arbiter) ═══════════
  output logic          pop_valid,
  output layer_desc_t   pop_layer,
  output tile_desc_t    pop_tile,
  input  logic          pop_ready,

  // ═══════════ STATUS ═══════════
  output logic [$clog2(DEPTH):0] count,
  output logic          full,
  output logic          empty
);

  localparam int PTR_W = $clog2(DEPTH);

  layer_desc_t layer_mem [DEPTH];
  tile_desc_t  tile_mem  [DEPTH];

  logic [PTR_W:0] wr_ptr, rd_ptr;

  assign full  = (count == DEPTH[PTR_W:0]);
  assign empty = (count == 0);
  assign push_ready = !full;
  assign pop_valid  = !empty;

  assign count = wr_ptr - rd_ptr;

  // Output from read pointer
  assign pop_layer = layer_mem[rd_ptr[PTR_W-1:0]];
  assign pop_tile  = tile_mem[rd_ptr[PTR_W-1:0]];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wr_ptr <= '0;
      rd_ptr <= '0;
    end else begin
      if (push_valid && push_ready) begin
        layer_mem[wr_ptr[PTR_W-1:0]] <= push_layer;
        tile_mem[wr_ptr[PTR_W-1:0]]  <= push_tile;
        wr_ptr <= wr_ptr + 1;
      end
      if (pop_valid && pop_ready) begin
        rd_ptr <= rd_ptr + 1;
      end
    end
  end

endmodule
