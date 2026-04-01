`timescale 1ns/1ps

// Input Address Generator: computes physical SRAM address from logical (h, w, c).
// Banking rule: bank_id = h mod 3
// CRITICAL: padding positions must output zp_x, NOT zero.
module addr_gen_input #(
  parameter int LANES     = 32,
  parameter int MAX_WIDTH = 640,
  parameter int MAX_CIN   = 256
)(
  input  logic                clk,
  input  logic                rst_n,

  // Configuration from shadow regs
  input  logic [9:0]         cfg_win,
  input  logic [8:0]         cfg_cin_tile,
  input  logic [3:0]         cfg_q_in,
  input  logic [3:0]         cfg_stride,
  input  logic [3:0]         cfg_pad_top,
  input  logic [3:0]         cfg_pad_bot,
  input  logic [3:0]         cfg_pad_left,
  input  logic [3:0]         cfg_pad_right,
  input  logic [9:0]         cfg_hin,
  input  logic signed [7:0]  cfg_zp_x,

  // Request
  input  logic               req_valid,
  input  logic [9:0]         req_h,
  input  logic [9:0]         req_w,
  input  logic [8:0]         req_c,

  // Output (1-cycle latency)
  output logic               out_valid,
  output logic [1:0]         out_bank_id,
  output logic [15:0]        out_addr,
  output logic               out_is_padding,
  output logic signed [7:0]  out_pad_value
);

  // Derived: wblk_total = ceil(cfg_win / LANES)
  wire [5:0] wblk_total = (cfg_win + LANES - 1) / LANES;

  // Padding detection
  wire pad_h_top = (req_h < cfg_pad_top);
  wire pad_h_bot = (req_h >= cfg_hin - cfg_pad_bot);
  wire pad_w_left  = (req_w < cfg_pad_left);
  wire pad_w_right = (req_w >= cfg_win - cfg_pad_right);
  wire is_pad = pad_h_top | pad_h_bot | pad_w_left | pad_w_right;

  // Address calculation
  wire [1:0]  bank_id  = req_h % 3;
  wire [9:0]  h_div3   = req_h / 3;
  wire [3:0]  row_slot = h_div3 % cfg_q_in;
  wire [5:0]  wblk     = req_w / LANES;

  wire [15:0] addr = (16'(row_slot) * 16'(cfg_cin_tile) + 16'(req_c))
                   * 16'(wblk_total) + 16'(wblk);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_valid      <= 1'b0;
      out_bank_id    <= '0;
      out_addr       <= '0;
      out_is_padding <= 1'b0;
      out_pad_value  <= '0;
    end else begin
      out_valid      <= req_valid;
      out_bank_id    <= bank_id;
      out_addr       <= addr;
      out_is_padding <= is_pad;
      out_pad_value  <= cfg_zp_x;  // ALWAYS zp_x for padding
    end
  end

endmodule
