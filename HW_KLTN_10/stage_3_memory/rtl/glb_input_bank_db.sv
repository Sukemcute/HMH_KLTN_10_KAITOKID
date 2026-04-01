// ============================================================================
// Module : glb_input_bank_db
// Project: YOLOv10n INT8 Accelerator — V4-VC707
// Description:
//   Double-buffered input activation SRAM with ping-pong pages.
//   Two pages (A/B), each containing LANES independent subbanks of DEPTH
//   INT8 entries.  Compute reads from the ACTIVE page while DMA writes
//   to the SHADOW page.  page_swap toggles the active page pointer.
//
//   Implementation: 2 × LANES SRAM arrays, each DEPTH deep × 8 bits.
//   Registered read output (1-cycle latency).
//   Lane-masked writes via wr_lane_mask.
// ============================================================================
`timescale 1ns / 1ps

module glb_input_bank_db #(
  parameter LANES = 20,
  parameter DEPTH = 2048
)(
  input  logic                       clk,
  input  logic                       rst_n,
  input  logic                       page_swap,

  // Compute read (active page)
  input  logic [$clog2(DEPTH)-1:0]   rd_addr,
  output int8_t                      rd_data [LANES],

  // DMA write (shadow page)
  input  logic [$clog2(DEPTH)-1:0]   wr_addr,
  input  int8_t                      wr_data [LANES],
  input  logic                       wr_en,
  input  logic [LANES-1:0]           wr_lane_mask
);

  import accel_pkg::*;

  // --------------------------------------------------------------------------
  //  Page pointer — toggles on page_swap pulse
  // --------------------------------------------------------------------------
  logic active_page;  // 0 = page A is active (compute reads), 1 = page B

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      active_page <= 1'b0;
    else if (page_swap)
      active_page <= ~active_page;
  end

  // Shadow page is the inverse of the active page
  wire shadow_page = ~active_page;

  // --------------------------------------------------------------------------
  //  SRAM arrays — 2 pages × LANES subbanks
  //  mem[page][lane][addr]
  // --------------------------------------------------------------------------
  genvar pg, ln;
  generate
    for (pg = 0; pg < 2; pg++) begin : gen_page
      for (ln = 0; ln < LANES; ln++) begin : gen_lane

        // Each subbank: DEPTH × 8-bit signed
        int8_t sram [DEPTH];

        // ---- Write (shadow page only) ------------------------------------
        always_ff @(posedge clk) begin
          if (wr_en && (shadow_page == pg[0]) && wr_lane_mask[ln])
            sram[wr_addr] <= wr_data[ln];
        end

        // ---- Read (active page only, registered output) ------------------
        int8_t rd_reg;

        always_ff @(posedge clk or negedge rst_n) begin
          if (!rst_n)
            rd_reg <= 8'sd0;
          else if (active_page == pg[0])
            rd_reg <= sram[rd_addr];
        end

        // Drive output only from the active-page copy
        if (pg == 0) begin : gen_mux_a
          assign rd_data[ln] = (active_page == 1'b0) ?
                               gen_page[0].gen_lane[ln].rd_reg :
                               gen_page[1].gen_lane[ln].rd_reg;
        end

      end // gen_lane
    end // gen_page
  endgenerate

endmodule
