`timescale 1ns/1ps
package desc_pkg;
  import accel_pkg::*;

  typedef struct packed {
    logic [15:0] magic;
    logic [7:0]  version;
    logic [7:0]  num_layers;
    logic [63:0] layer_table_base;
    logic [63:0] weight_arena_base;
    logic [63:0] act0_arena_base;
    logic [63:0] act1_arena_base;
    logic [63:0] aux_arena_base;
  } net_desc_t;

  typedef struct packed {
    logic [3:0]  template_id;
    logic [4:0]  layer_id;
    logic [8:0]  cin_total;
    logic [8:0]  cout_total;
    logic [9:0]  hin, win;
    logic [9:0]  hout, wout;
    logic [3:0]  kh, kw;
    logic [2:0]  sh, sw;
    logic [3:0]  pad_top, pad_bot, pad_left, pad_right;
    logic [7:0]  tile_cin, tile_cout;
    logic [5:0]  tile_w_blks;
    logic [11:0] num_tile_hw;
    logic [3:0]  r_need, r_new, q_in, q_out;
    logic [3:0]  num_cin_pass, num_k_pass;
    logic [7:0]  router_profile_id;
    logic [7:0]  post_profile_id;
    logic [4:0]  src_in_tid, src_w_tid, src_skip_tid, dst_tid;
    logic [63:0] tile_table_offset;
    logic [15:0] layer_flags;
  } layer_desc_t;

  typedef struct packed {
    logic [15:0] tile_id;
    logic [4:0]  layer_id;
    logic [3:0]  sc_mask;
    logic [9:0]  h_out0, wblk0;
    logic [8:0]  cin0, cout0;
    logic [5:0]  valid_h, valid_w;
    logic [3:0]  halo_top, halo_bot, halo_left, halo_right;
    logic [31:0] src_in_off;
    logic [31:0] src_w_off;
    logic [31:0] src_skip_off;
    logic [31:0] dst_off;
    logic [9:0]  in_base_h, in_base_c;
    logic [9:0]  out_base_h, out_base_c;
    logic [3:0]  first_cin_pass, num_cin_pass;
    logic [3:0]  first_k_pass, num_k_pass;
    logic [15:0] tile_flags;
  } tile_desc_t;

  typedef struct packed {
    logic        bias_en;
    quant_mode_e quant_mode;
    act_mode_e   act_mode;
    logic        ewise_en;
    logic [31:0] bias_scale_offset;
    logic [7:0]  concat_ch_offset;
    logic [1:0]  upsample_factor;
  } post_profile_t;

  typedef struct packed {
    logic [2:0][2:0] rin_src;
    logic [2:0][3:0] rin_dst_mask;
    logic [2:0][2:0] rwt_src;
    logic            rwt_h_multicast;
    logic [1:0]      rps_accum_mode;
    logic            concat_offset_mode;
    logic            upsample_dup_mode;
} router_profile_t;

endpackage
