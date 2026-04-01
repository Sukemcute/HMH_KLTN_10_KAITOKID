`timescale 1ns/1ps

// Input Activation SRAM Bank — 1 of 3 banks (banking: bank_id = h mod 3).
// Contains 32 subbanks (one per lane), each is independent SRAM.
module glb_input_bank #(
  parameter int LANES         = 32,
  parameter int SUBBANK_DEPTH = 2048,
  parameter int ADDR_W        = $clog2(SUBBANK_DEPTH)
)(
  input  logic                  clk,
  input  logic                  rst_n,

  // Write port (from DMA / swizzle during FILLING)
  input  logic                  wr_en,
  input  logic [ADDR_W-1:0]    wr_addr,
  input  logic [LANES*8-1:0]   wr_data,
  input  logic [LANES-1:0]     wr_lane_mask,

  // Read port (to router → PE during RUNNING)
  input  logic                  rd_en,
  input  logic [ADDR_W-1:0]    rd_addr,
  output logic [LANES*8-1:0]   rd_data
);

  // 32 independent subbanks
  genvar g;
  generate
    for (g = 0; g < LANES; g++) begin : gen_subbank
      logic [7:0] mem [SUBBANK_DEPTH];

      // Write
      always_ff @(posedge clk) begin
        if (wr_en && wr_lane_mask[g])
          mem[wr_addr] <= wr_data[(g+1)*8-1 -: 8];
      end

      // Read (registered output)
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
          rd_data[(g+1)*8-1 -: 8] <= '0;
        else if (rd_en)
          rd_data[(g+1)*8-1 -: 8] <= mem[rd_addr];
      end
    end
  endgenerate

endmodule
