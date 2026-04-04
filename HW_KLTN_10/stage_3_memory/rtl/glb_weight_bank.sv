// ============================================================================
// Module : glb_weight_bank
// Project: YOLOv10n INT8 Accelerator — V4-VC707
// Description:
//   4-read-port weight SRAM (1 per PE column for per-column weight routing).
//   Implementation: N_READ_PORTS duplicated BRAM arrays, each LANES-wide
//   and DEPTH-deep.  A single write port broadcasts to all copies.
//   Each read port accesses its own copy at an independent address.
//   Registered read output (1-cycle latency).
// ============================================================================
`timescale 1ns / 1ps

module glb_weight_bank
  import accel_pkg::*;
#(
  parameter LANES        = 20,
  parameter DEPTH        = 1024,
  parameter N_READ_PORTS = 4
)(
  input  logic                       clk,
  input  logic                       rst_n,

  // 4 read ports
  input  logic [$clog2(DEPTH)-1:0]   rd_addr  [N_READ_PORTS],
  output int8_t                      rd_data  [N_READ_PORTS][LANES],

  // 1 write port (broadcast to all copies)
  input  logic [$clog2(DEPTH)-1:0]   wr_addr,
  input  int8_t                      wr_data  [LANES],
  input  logic                       wr_en
);

  // --------------------------------------------------------------------------
  //  Duplicated SRAM copies — N_READ_PORTS × LANES subbanks
  // --------------------------------------------------------------------------
  genvar rp, ln;
  generate
    for (rp = 0; rp < N_READ_PORTS; rp++) begin : gen_copy
      for (ln = 0; ln < LANES; ln++) begin : gen_lane

        // Each subbank: DEPTH × 8-bit signed
        int8_t sram [DEPTH];

        // ---- Write broadcast to every copy -------------------------------
        always_ff @(posedge clk) begin
          if (wr_en)
            sram[wr_addr] <= wr_data[ln];
        end

        // ---- Independent read per copy (registered output) ---------------
        int8_t rd_reg;

        always_ff @(posedge clk or negedge rst_n) begin
          if (!rst_n)
            rd_reg <= 8'sd0;
          else
            rd_reg <= sram[rd_addr[rp]];
        end

        assign rd_data[rp][ln] = rd_reg;

      end // gen_lane
    end // gen_copy
  endgenerate

  // synthesis translate_off
`ifdef S8_DBG
  always @(posedge clk) begin
    if (rst_n && wr_en)
      $display("  [WTB] %0t WR addr=%0d d[0]=%0d d[1]=%0d",
               $time, wr_addr, wr_data[0], wr_data[1]);
  end
`endif
`ifdef RTL_TRACE
  always @(posedge clk) begin
    if (rst_n && wr_en)
      rtl_trace_pkg::rtl_trace_line("S3_WTB_WR",
        $sformatf("addr=%0d d0=%0d d1=%0d", wr_addr, wr_data[0], wr_data[1]));
  end
`endif
  // synthesis translate_on

endmodule
