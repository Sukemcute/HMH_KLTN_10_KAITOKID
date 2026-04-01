`timescale 1ns/1ps
// Captures tile/layer descriptor fields into stable pipeline registers.
// Provides constant configuration to PE cluster during entire tile execution.
module shadow_reg_file (
  input  logic              clk,
  input  logic              rst_n,
  input  logic              load,

  input  desc_pkg::tile_desc_t    tile_desc,
  input  desc_pkg::layer_desc_t   layer_desc,
  input  desc_pkg::post_profile_t post_profile,
  input  desc_pkg::router_profile_t router_profile,

  // Stable configuration outputs
  output accel_pkg::pe_mode_e       o_mode,
  output logic [8:0]                o_cin_tile,
  output logic [8:0]                o_cout_tile,
  output logic [9:0]                o_hin, o_win,
  output logic [9:0]                o_hout, o_wout,
  output logic [3:0]                o_kh, o_kw,
  output logic [2:0]                o_sh, o_sw,
  output logic [3:0]                o_pad_top, o_pad_bot, o_pad_left, o_pad_right,
  output logic [3:0]                o_q_in, o_q_out,
  output logic [3:0]                o_num_cin_pass, o_num_k_pass,
  output logic [15:0]               o_tile_flags,
  output desc_pkg::post_profile_t   o_post,
  output desc_pkg::router_profile_t o_router
);
  import accel_pkg::*;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      o_mode         <= PE_RS3;
      o_cin_tile     <= '0;
      o_cout_tile    <= '0;
      o_hin          <= '0;
      o_win          <= '0;
      o_hout         <= '0;
      o_wout         <= '0;
      o_kh           <= '0;
      o_kw           <= '0;
      o_sh           <= '0;
      o_sw           <= '0;
      o_pad_top      <= '0;
      o_pad_bot      <= '0;
      o_pad_left     <= '0;
      o_pad_right    <= '0;
      o_q_in         <= '0;
      o_q_out        <= '0;
      o_num_cin_pass <= '0;
      o_num_k_pass   <= '0;
      o_tile_flags   <= '0;
      o_post         <= '0;
      o_router       <= '0;
    end else if (load) begin
      o_mode         <= pe_mode_e'(layer_desc.template_id);
      o_cin_tile     <= {1'b0, layer_desc.tile_cin};
      o_cout_tile    <= {1'b0, layer_desc.tile_cout};
      o_hin          <= layer_desc.hin;
      o_win          <= layer_desc.win;
      o_hout         <= layer_desc.hout;
      o_wout         <= layer_desc.wout;
      o_kh           <= layer_desc.kh;
      o_kw           <= layer_desc.kw;
      o_sh           <= layer_desc.sh;
      o_sw           <= layer_desc.sw;
      o_pad_top      <= layer_desc.pad_top;
      o_pad_bot      <= layer_desc.pad_bot;
      o_pad_left     <= layer_desc.pad_left;
      o_pad_right    <= layer_desc.pad_right;
      o_q_in         <= layer_desc.q_in;
      o_q_out        <= layer_desc.q_out;
      o_num_cin_pass <= layer_desc.num_cin_pass;
      o_num_k_pass   <= layer_desc.num_k_pass;
      o_tile_flags   <= tile_desc.tile_flags;
      o_post         <= post_profile;
      o_router       <= router_profile;
    end
  end

endmodule
