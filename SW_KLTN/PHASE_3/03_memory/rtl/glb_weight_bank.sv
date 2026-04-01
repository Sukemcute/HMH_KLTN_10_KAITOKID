`timescale 1ns/1ps

// Weight SRAM Bank — 1 of 3 banks (one per reduction lane / kernel row).
// Main SRAM + staging FIFO for weight prefetch.
module glb_weight_bank #(
  parameter int LANES      = 32,
  parameter int BANK_DEPTH = 1024,
  parameter int FIFO_DEPTH = 8,
  parameter int ADDR_W     = $clog2(BANK_DEPTH)
)(
  input  logic                 clk,
  input  logic                 rst_n,

  // SRAM Write port
  input  logic                 wr_en,
  input  logic [ADDR_W-1:0]   wr_addr,
  input  logic [LANES*8-1:0]  wr_data,

  // SRAM Read port
  input  logic                 rd_en,
  input  logic [ADDR_W-1:0]   rd_addr,
  output logic [LANES*8-1:0]  rd_data,

  // Staging FIFO
  input  logic                 fifo_push,
  input  logic [LANES*8-1:0]  fifo_din,
  input  logic                 fifo_pop,
  output logic [LANES*8-1:0]  fifo_dout,
  output logic                 fifo_empty,
  output logic                 fifo_full
);

  // ═══════════ Main SRAM ═══════════
  logic [LANES*8-1:0] sram [BANK_DEPTH];

  always_ff @(posedge clk) begin
    if (wr_en)
      sram[wr_addr] <= wr_data;
  end

  always_ff @(posedge clk) begin
    if (rd_en)
      rd_data <= sram[rd_addr];
  end

  // ═══════════ Staging FIFO ═══════════
  localparam int FIFO_PTR_W = $clog2(FIFO_DEPTH);

  logic [LANES*8-1:0]  fifo_mem [FIFO_DEPTH];
  logic [FIFO_PTR_W:0] wr_ptr, rd_ptr;

  wire [FIFO_PTR_W:0] count = wr_ptr - rd_ptr;
  assign fifo_empty = (count == 0);
  assign fifo_full  = (count == FIFO_DEPTH[FIFO_PTR_W:0]);

  assign fifo_dout = fifo_mem[rd_ptr[FIFO_PTR_W-1:0]];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wr_ptr <= '0;
      rd_ptr <= '0;
    end else begin
      if (fifo_push && !fifo_full) begin
        fifo_mem[wr_ptr[FIFO_PTR_W-1:0]] <= fifo_din;
        wr_ptr <= wr_ptr + 1;
      end
      if (fifo_pop && !fifo_empty)
        rd_ptr <= rd_ptr + 1;
    end
  end

endmodule
