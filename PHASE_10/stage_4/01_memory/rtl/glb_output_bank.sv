`timescale 1ns/1ps

// Output SRAM Bank — 1 of 4 banks.
// Dual-namespace: PSUM (32×INT32=1024b) or ACT (32×INT8=256b).
module glb_output_bank #(
  parameter int LANES      = 32,
  parameter int BANK_DEPTH = 512,
  parameter int ADDR_W     = $clog2(BANK_DEPTH)
)(
  input  logic                  clk,
  input  logic                  rst_n,

  // Write port
  input  logic                  wr_en,
  input  logic [ADDR_W-1:0]    wr_addr,
  input  accel_pkg::namespace_e wr_ns,
  input  logic [LANES*32-1:0]  wr_data_psum,
  input  logic [LANES*8-1:0]   wr_data_act,

  // Read port
  input  logic                  rd_en,
  input  logic [ADDR_W-1:0]    rd_addr,
  input  accel_pkg::namespace_e rd_ns,
  output logic [LANES*32-1:0]  rd_data_psum,
  output logic [LANES*8-1:0]   rd_data_act
);
  import accel_pkg::*;

  // Wide SRAM stores PSUM width (max width); ACT shares same storage
  logic [LANES*32-1:0] sram [BANK_DEPTH];

  // Write
  always_ff @(posedge clk) begin
    if (wr_en) begin
      if (wr_ns == NS_PSUM)
        sram[wr_addr] <= wr_data_psum;
      else
        sram[wr_addr][LANES*8-1:0] <= wr_data_act;
    end
  end

  // Read (registered)
  logic [LANES*32-1:0] rd_raw;

  always_ff @(posedge clk) begin
    if (rd_en)
      rd_raw <= sram[rd_addr];
  end

  assign rd_data_psum = rd_raw;
  assign rd_data_act  = rd_raw[LANES*8-1:0];

endmodule
