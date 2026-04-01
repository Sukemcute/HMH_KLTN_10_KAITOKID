`timescale 1ns/1ps
// Output Address Generator: maps (h_out, w_out, cout) to (bank_id, addr).
// bank_id = pe_col index (0-3).
module addr_gen_output #(
  parameter int LANES = 32
)(
  input  logic              clk,
  input  logic              rst_n,
  input  logic [3:0]        cfg_stride_h,
  input  logic [3:0]        cfg_q_out,
  input  logic [8:0]        cfg_cout_tile,
  input  logic [3:0]        cfg_pe_cols,

  input  logic              req_valid,
  input  logic [9:0]        req_h_out,
  input  logic [9:0]        req_w_out,
  input  logic [8:0]        req_cout,
  input  logic [1:0]        req_pe_col,

  output logic              out_valid,
  output logic [1:0]        out_bank_id,
  output logic [15:0]       out_addr
);

  wire [5:0] wblk_out = req_w_out / LANES;
  wire [5:0] wblk_total = (req_w_out + LANES - 1) / LANES;  // approximate

  wire [9:0] h_group = req_h_out / (cfg_pe_cols * cfg_stride_h);
  wire [3:0] oslot   = h_group % cfg_q_out;

  wire [15:0] addr = (16'(oslot) * 16'(cfg_cout_tile) + 16'(req_cout))
                    * 16'(wblk_total) + 16'(wblk_out);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_valid   <= 1'b0;
      out_bank_id <= '0;
      out_addr    <= '0;
    end else begin
      out_valid   <= req_valid;
      out_bank_id <= req_pe_col;  // output bank = PE column index
      out_addr    <= addr;
    end
  end

endmodule
