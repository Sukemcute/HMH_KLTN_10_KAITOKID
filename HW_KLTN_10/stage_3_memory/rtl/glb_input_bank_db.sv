// ============================================================================
// Module : glb_input_bank_db
// Project: YOLOv10n INT8 Accelerator — V4-VC707
// Description:
//   Double-buffered input activation SRAM with ping-pong pages.
//   ★ V4 DW3/DW7: N_READ_PORTS parallel read ports (default 4) — duplicate
//   SRAM copies per port; broadcast write keeps all copies identical.
//
//   Active page = compute read; shadow page = DMA write.
// ============================================================================
`timescale 1ns / 1ps

module glb_input_bank_db
  import accel_pkg::*;
#(
  parameter int LANES         = 20,
  parameter int DEPTH         = 2048,
  parameter int N_READ_PORTS  = 4
)(
  input  logic                       clk,
  input  logic                       rst_n,
  input  logic                       page_swap,

  input  logic [$clog2(DEPTH)-1:0]   rd_addr [N_READ_PORTS],
  output int8_t                      rd_data [N_READ_PORTS][LANES],

  input  logic [$clog2(DEPTH)-1:0]   wr_addr,
  input  int8_t                      wr_data [LANES],
  input  logic                       wr_en,
  input  logic [LANES-1:0]           wr_lane_mask
);

  logic active_page;  // 0 = page A compute, 1 = page B compute

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      active_page <= 1'b0;
    else if (page_swap)
      active_page <= ~active_page;
  end

  genvar rp, ln;
  generate
    for (rp = 0; rp < N_READ_PORTS; rp++) begin : gen_rp
      for (ln = 0; ln < LANES; ln++) begin : gen_ln
        int8_t sram_a [DEPTH];
        int8_t sram_b [DEPTH];

        // Write always to shadow page (non-active)
        always_ff @(posedge clk) begin
          if (wr_en && wr_lane_mask[ln]) begin
            if (active_page == 1'b0)
              sram_b[wr_addr] <= wr_data[ln];
            else
              sram_a[wr_addr] <= wr_data[ln];
          end
        end

        int8_t rd_reg;
        always_ff @(posedge clk or negedge rst_n) begin
          if (!rst_n)
            rd_reg <= 8'sd0;
          else if (active_page == 1'b0)
            rd_reg <= sram_a[rd_addr[rp]];
          else
            rd_reg <= sram_b[rd_addr[rp]];
        end

        assign rd_data[rp][ln] = rd_reg;
      end
    end
  endgenerate

  // synthesis translate_off
`ifdef RTL_TRACE
  always @(posedge clk) begin
    if (rst_n && wr_en)
      rtl_trace_pkg::rtl_trace_line("S3_INB_WR",
        $sformatf("ap=%0d addr=%0d d0=%0d d1=%0d",
                  active_page, wr_addr, wr_data[0], wr_data[1]));
  end
`endif
  // synthesis translate_on

endmodule
