// ============================================================================
// Module : glb_output_bank
// Project: YOLOv10n INT8 Accelerator — V4-VC707
// Description:
//   Dual-namespace output SRAM: PSUM (INT32, multipass accumulation) and
//   ACT (INT8, final PPU output).
//
//   Internal storage is a wide SRAM of PSUM_WIDTH (32 bits) per lane.
//   The ACT namespace shares the lower 8 bits of each 32-bit word.
//   DMA drain port packs the ACT-width (INT8) values into a flat bus
//   for AXI transfer.
//
//   Registered read outputs (1-cycle latency).
// ============================================================================
`timescale 1ns / 1ps

module glb_output_bank #(
  parameter LANES = 20,
  parameter DEPTH = 512
)(
  input  logic                       clk,
  input  logic                       rst_n,

  // PSUM namespace — INT32 × LANES
  input  logic [$clog2(DEPTH)-1:0]   psum_addr,
  input  int32_t                     psum_wr_data [LANES],
  input  logic                       psum_wr_en,
  output int32_t                     psum_rd_data [LANES],
  input  logic                       psum_rd_en,

  // ACT namespace — INT8 × LANES
  input  logic [$clog2(DEPTH)-1:0]   act_addr,
  input  int8_t                      act_wr_data  [LANES],
  input  logic                       act_wr_en,
  output int8_t                      act_rd_data  [LANES],
  input  logic                       act_rd_en,

  // DMA drain — packed flat bus (INT8 per lane)
  input  logic [$clog2(DEPTH)-1:0]   drain_addr,
  output logic [LANES*8-1:0]         drain_data
);

  import accel_pkg::*;

  // --------------------------------------------------------------------------
  //  Wide SRAM — LANES subbanks, each DEPTH × 32-bit
  //  PSUM writes/reads the full 32 bits; ACT writes/reads the lower 8 bits.
  // --------------------------------------------------------------------------
  genvar ln;
  generate
    for (ln = 0; ln < LANES; ln++) begin : gen_lane

      // Each subbank: DEPTH × 32-bit (PSUM width)
      int32_t sram [DEPTH];

      // ------------------------------------------------------------------
      //  Write logic — PSUM has priority over ACT
      // ------------------------------------------------------------------
      always_ff @(posedge clk) begin
        if (psum_wr_en)
          sram[psum_addr] <= psum_wr_data[ln];
        else if (act_wr_en)
          sram[act_addr] <= {{24{act_wr_data[ln][7]}}, act_wr_data[ln]};
      end

      // ------------------------------------------------------------------
      //  PSUM read (registered)
      // ------------------------------------------------------------------
      int32_t psum_rd_reg;

      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
          psum_rd_reg <= 32'sd0;
        else if (psum_rd_en)
          psum_rd_reg <= sram[psum_addr];
      end

      assign psum_rd_data[ln] = psum_rd_reg;

      // ------------------------------------------------------------------
      //  ACT read (registered, lower 8 bits)
      // ------------------------------------------------------------------
      int8_t act_rd_reg;

      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
          act_rd_reg <= 8'sd0;
        else if (act_rd_en)
          act_rd_reg <= int8_t'(sram[act_addr]);
      end

      assign act_rd_data[ln] = act_rd_reg;

      // ------------------------------------------------------------------
      //  DMA drain — read lower 8 bits, registered
      // ------------------------------------------------------------------
      logic [7:0] drain_reg;

      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
          drain_reg <= 8'd0;
        else
          drain_reg <= sram[drain_addr][7:0];
      end

      assign drain_data[ln*8 +: 8] = drain_reg;

    end // gen_lane
  endgenerate

endmodule
