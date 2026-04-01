`timescale 1ns/1ps
// Weight Address Generator: computes SRAM address per PE mode.
// bank_id = kernel_row for RS3/DW3, cin_slice for OS1.
module addr_gen_weight #(
  parameter int LANES = 32
)(
  input  logic              clk,
  input  logic              rst_n,
  input  accel_pkg::pe_mode_e cfg_mode,
  input  logic [8:0]        cfg_cin_tile,
  input  logic [8:0]        cfg_cout_tile,
  input  logic [3:0]        cfg_kw,

  input  logic              req_valid,
  input  logic [2:0]        req_kr,
  input  logic [8:0]        req_cin,
  input  logic [8:0]        req_cout,
  input  logic [2:0]        req_kw_idx,

  output logic              out_valid,
  output logic [1:0]        out_bank_id,
  output logic [15:0]       out_addr
);
  import accel_pkg::*;

  logic [1:0]  bank_comb;
  logic [15:0] addr_comb;

  always_comb begin
    bank_comb = '0;
    addr_comb = '0;

    case (cfg_mode)
      PE_RS3: begin
        bank_comb = req_kr[1:0];
        addr_comb = (16'(req_cout) * 16'(cfg_cin_tile) * 16'(cfg_kw)
                    + 16'(req_cin) * 16'(cfg_kw)
                    + 16'(req_kw_idx));
      end
      PE_OS1: begin
        bank_comb = req_kr[1:0];  // cin_slice index
        addr_comb = 16'(req_cout) * 16'(cfg_cin_tile / 3) + 16'(req_cin);
      end
      PE_DW3: begin
        bank_comb = req_kr[1:0];
        addr_comb = 16'(req_cin) * 16'(cfg_kw) + 16'(req_kw_idx);
      end
      PE_DW7: begin
        bank_comb = req_kr[1:0] % 2'd3;
        addr_comb = 16'(req_cin) * 16'(cfg_kw) + 16'(req_kw_idx);
      end
      PE_GEMM: begin
        bank_comb = req_kr[1:0];
        addr_comb = 16'(req_cout) * 16'(cfg_cin_tile) + 16'(req_cin);
      end
      default: begin
        bank_comb = '0;
        addr_comb = '0;
      end
    endcase
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_valid   <= 1'b0;
      out_bank_id <= '0;
      out_addr    <= '0;
    end else begin
      out_valid   <= req_valid;
      out_bank_id <= bank_comb;
      out_addr    <= addr_comb;
    end
  end

endmodule
