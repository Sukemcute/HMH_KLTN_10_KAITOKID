// ============================================================================
// Package: stage11_pkg — Stage 11 Block Verification Helpers
//
// Provides:
//   1. Descriptor builder functions for each YOLOv10n block type
//   2. Golden-reference computation helpers (SW-equivalent)
//   3. Checkpoint infrastructure for debug-friendly tracing
//   4. Match percentage calculator
//
// Block Types Covered (11.1–11.7):
//   Conv, QC2f, SCDown, SPPF, QConcat, Upsample, QC2fCIB
//
// NOTE: Test dimensions are SCALED DOWN from real model for simulation
//       feasibility. The descriptor STRUCTURE is identical to real inference.
// ============================================================================
`timescale 1ns / 1ps

package stage11_pkg;
  import accel_pkg::*;
  import desc_pkg::*;

  // ═══════════════════════════════════════════════════════════════════
  // CHECKPOINT INFRASTRUCTURE
  // ═══════════════════════════════════════════════════════════════════
  int unsigned cp_id = 0;

  task automatic checkpoint(input string block_name, input string phase);
    cp_id++;
    $display("[S11_CP %04d] %0t | %-10s | %s", cp_id, $time, block_name, phase);
  endtask

  task automatic checkpoint_data(input string block_name, input string label,
                                  input int value);
    $display("[S11_DATA]   %0t | %-10s | %s = %0d", $time, block_name, label, value);
  endtask

  // ═══════════════════════════════════════════════════════════════════
  // GOLDEN REFERENCE HELPERS (bit-accurate SW equivalents)
  // ═══════════════════════════════════════════════════════════════════

  // Half-up requant: acc*m_int rounded and shifted
  function automatic int32_t sw_requant(
    input int64_t acc,
    input uint32_t m_int,
    input logic [7:0] sh
  );
    int64_t prod, rnd;
    prod = acc * int64_t'(unsigned'(m_int));
    if (sh == 0) return int32_t'(prod);
    rnd = 64'sd1 <<< (sh - 1);
    return int32_t'((prod + rnd) >>> sh);
  endfunction

  // ReLU + ZP_out + clamp
  function automatic int8_t sw_relu_zp_clamp(input int32_t x, input int8_t zp);
    int32_t v;
    v = (x > 0) ? x : 32'sd0;
    v = v + int32_t'($signed({{24{zp[7]}}, zp}));
    if (v > 127) return 8'sd127;
    if (v < -128) return -8'sd128;
    return int8_t'(v);
  endfunction

  // No-activation + ZP + clamp
  function automatic int8_t sw_none_zp_clamp(input int32_t x, input int8_t zp);
    int32_t v;
    v = x + int32_t'($signed({{24{zp[7]}}, zp}));
    if (v > 127) return 8'sd127;
    if (v < -128) return -8'sd128;
    return int8_t'(v);
  endfunction

  // Max of 25 signed INT8 values (MaxPool 5x5)
  function automatic int8_t sw_max25(input int8_t v[25]);
    int8_t m;
    m = v[0];
    for (int i = 1; i < 25; i++)
      if (v[i] > m) m = v[i];
    return m;
  endfunction

  // Domain alignment requant (single value)
  function automatic int8_t sw_domain_align(
    input int8_t x, input int8_t zp_in,
    input uint32_t m_int, input logic [7:0] sh,
    input int8_t zp_out
  );
    int32_t sub, shifted;
    int64_t prod;
    sub = int32_t'(x) - int32_t'(zp_in);
    prod = int64_t'(sub) * int64_t'(unsigned'(m_int));
    if (sh > 0)
      shifted = int32_t'((prod + (64'sd1 <<< (sh-1))) >>> sh);
    else
      shifted = int32_t'(prod);
    shifted = shifted + int32_t'(zp_out);
    if (shifted > 127) return 8'sd127;
    if (shifted < -128) return -8'sd128;
    return int8_t'(shifted);
  endfunction

  // ═══════════════════════════════════════════════════════════════════
  // DESCRIPTOR BUILDER HELPERS
  // Each function fills a layer_desc_t for a specific primitive step.
  // ═══════════════════════════════════════════════════════════════════

  function automatic layer_desc_t make_conv_rs3_desc(
    input logic [4:0]  layer_id,
    input logic [9:0]  cin, cout, hin, win, hout, wout,
    input logic [2:0]  stride,
    input act_mode_e   act
  );
    layer_desc_t d;
    d = '0;
    d.layer_id   = layer_id;
    d.pe_mode    = PE_RS3;
    d.activation = act;
    d.cin = cin; d.cout = cout;
    d.hin = hin; d.win = win;
    d.hout = hout; d.wout = wout;
    d.kh = 4'd3; d.kw = 4'd3;
    d.stride = stride;
    d.padding = 3'd1;
    d.num_tiles = 8'd1;
    d.num_cin_pass = 4'd1;
    d.num_k_pass = 4'd1;
    d.swizzle = SWZ_NORMAL;
    return d;
  endfunction

  function automatic layer_desc_t make_conv_os1_desc(
    input logic [4:0]  layer_id,
    input logic [9:0]  cin, cout, hin, win, hout, wout,
    input logic [2:0]  stride,
    input act_mode_e   act
  );
    layer_desc_t d;
    d = '0;
    d.layer_id   = layer_id;
    d.pe_mode    = PE_OS1;
    d.activation = act;
    d.cin = cin; d.cout = cout;
    d.hin = hin; d.win = win;
    d.hout = hout; d.wout = wout;
    d.kh = 4'd1; d.kw = 4'd1;
    d.stride = stride;
    d.padding = 3'd0;
    d.num_tiles = 8'd1;
    d.num_cin_pass = 4'd1;
    d.num_k_pass = 4'd1;
    d.swizzle = SWZ_NORMAL;
    return d;
  endfunction

  function automatic layer_desc_t make_dw3_desc(
    input logic [4:0]  layer_id,
    input logic [9:0]  ch, hin, win, hout, wout,
    input logic [2:0]  stride,
    input act_mode_e   act
  );
    layer_desc_t d;
    d = '0;
    d.layer_id   = layer_id;
    d.pe_mode    = PE_DW3;
    d.activation = act;
    d.cin = ch; d.cout = ch;
    d.hin = hin; d.win = win;
    d.hout = hout; d.wout = wout;
    d.kh = 4'd3; d.kw = 4'd3;
    d.stride = stride;
    d.padding = 3'd1;
    d.num_tiles = 8'd1;
    d.num_cin_pass = 4'd1;
    d.num_k_pass = 4'd1;
    d.swizzle = SWZ_NORMAL;
    return d;
  endfunction

  function automatic layer_desc_t make_mp5_desc(
    input logic [4:0]  layer_id,
    input logic [9:0]  ch, hin, win, hout, wout
  );
    layer_desc_t d;
    d = '0;
    d.layer_id   = layer_id;
    d.pe_mode    = PE_MP5;
    d.activation = ACT_NONE;
    d.cin = ch; d.cout = ch;
    d.hin = hin; d.win = win;
    d.hout = hout; d.wout = wout;
    d.kh = 4'd5; d.kw = 4'd5;
    d.stride = 3'd1;
    d.padding = 3'd2;
    d.num_tiles = 8'd1;
    d.num_cin_pass = 4'd1;
    d.num_k_pass = 4'd1;
    d.swizzle = SWZ_NORMAL;
    return d;
  endfunction

  function automatic layer_desc_t make_upsample_desc(
    input logic [4:0]  layer_id,
    input logic [9:0]  ch, hin, win
  );
    layer_desc_t d;
    d = '0;
    d.layer_id   = layer_id;
    d.pe_mode    = PE_PASS;
    d.activation = ACT_NONE;
    d.cin = ch; d.cout = ch;
    d.hin = hin; d.win = win;
    d.hout = hin * 2; d.wout = win * 2;
    d.kh = 4'd1; d.kw = 4'd1;
    d.stride = 3'd1;
    d.padding = 3'd0;
    d.num_tiles = 8'd1;
    d.swizzle = SWZ_UPSAMPLE2X;
    return d;
  endfunction

  function automatic layer_desc_t make_concat_desc(
    input logic [4:0]  layer_id,
    input logic [9:0]  ch_a, ch_b, h, w
  );
    layer_desc_t d;
    d = '0;
    d.layer_id   = layer_id;
    d.pe_mode    = PE_PASS;
    d.activation = ACT_NONE;
    d.cin = ch_a + ch_b; d.cout = ch_a + ch_b;
    d.hin = h; d.win = w;
    d.hout = h; d.wout = w;
    d.kh = 4'd1; d.kw = 4'd1;
    d.stride = 3'd1;
    d.swizzle = SWZ_CONCAT;
    return d;
  endfunction

  function automatic layer_desc_t make_ewise_add_desc(
    input logic [4:0]  layer_id,
    input logic [9:0]  ch, h, w
  );
    layer_desc_t d;
    d = '0;
    d.layer_id   = layer_id;
    d.pe_mode    = PE_PASS;
    d.activation = ACT_NONE;
    d.cin = ch; d.cout = ch;
    d.hin = h; d.win = w;
    d.hout = h; d.wout = w;
    d.kh = 4'd1; d.kw = 4'd1;
    d.stride = 3'd1;
    d.swizzle = SWZ_EWISE_ADD;
    return d;
  endfunction

  function automatic layer_desc_t make_dw7_multipass_desc(
    input logic [4:0]  layer_id,
    input logic [9:0]  ch, hin, win, hout, wout,
    input act_mode_e   act,
    input logic [3:0]  num_k_pass
  );
    layer_desc_t d;
    d = '0;
    d.layer_id   = layer_id;
    d.pe_mode    = PE_DW7;
    d.activation = act;
    d.cin = ch; d.cout = ch;
    d.hin = hin; d.win = win;
    d.hout = hout; d.wout = wout;
    d.kh = 4'd7; d.kw = 4'd7;
    d.stride = 3'd1;
    d.padding = 3'd3;
    d.num_tiles = 8'd1;
    d.num_cin_pass = 4'd1;
    d.num_k_pass = num_k_pass;
    d.swizzle = SWZ_NORMAL;
    return d;
  endfunction

  // ═══════════════════════════════════════════════════════════════════
  // TILE DESCRIPTOR BUILDER (generic)
  // ═══════════════════════════════════════════════════════════════════
  function automatic tile_desc_t make_tile(
    input logic [15:0] tile_id,
    input logic [4:0]  layer_id,
    input logic [5:0]  valid_h, valid_w,
    input logic        first, last,
    input logic        need_swizzle,
    input logic        barrier_wait,
    input logic [3:0]  barrier_id,
    input logic [3:0]  num_k_pass
  );
    tile_desc_t t;
    t = '0;
    t.tile_id      = tile_id;
    t.layer_id     = layer_id;
    t.valid_h      = valid_h;
    t.valid_w      = valid_w;
    t.first_tile   = first;
    t.last_tile    = last;
    t.need_swizzle = need_swizzle;
    t.barrier_wait = barrier_wait;
    t.barrier_id   = barrier_id;
    t.num_k_pass   = num_k_pass;
    return t;
  endfunction

endpackage
