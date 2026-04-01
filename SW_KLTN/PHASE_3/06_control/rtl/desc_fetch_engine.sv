`timescale 1ns/1ps
// Descriptor fetch engine: reads NET_DESC → LAYER_DESC → TILE_DESC from DDR via AXI.
// Parses 64-byte descriptors and dispatches them downstream.
module desc_fetch_engine (
  input  logic               clk,
  input  logic               rst_n,
  input  logic               start,

  // AXI4 read master interface (simplified)
  output logic [39:0]        axi_araddr,
  output logic [7:0]         axi_arlen,
  output logic               axi_arvalid,
  input  logic               axi_arready,
  input  logic [255:0]       axi_rdata,
  input  logic               axi_rvalid,
  input  logic               axi_rlast,
  output logic               axi_rready,

  // Configuration from CSR
  input  logic [63:0]        net_desc_base,
  input  logic [7:0]         layer_start,
  input  logic [7:0]         layer_end,

  // Output: parsed descriptors
  output desc_pkg::net_desc_t   net_desc,
  output logic                   net_desc_valid,
  output desc_pkg::layer_desc_t layer_desc,
  output logic                   layer_desc_valid,
  output desc_pkg::tile_desc_t  tile_desc,
  output logic                   tile_desc_valid,
  input  logic                   tile_desc_ready,

  // Status
  output logic [7:0]         current_layer,
  output logic               all_layers_done
);
  import desc_pkg::*;

  typedef enum logic [3:0] {
    DF_IDLE,
    DF_FETCH_NET_AR,
    DF_FETCH_NET_R,
    DF_PARSE_NET,
    DF_FETCH_LAYER_AR,
    DF_FETCH_LAYER_R,
    DF_PARSE_LAYER,
    DF_FETCH_TILE_AR,
    DF_FETCH_TILE_R,
    DF_DISPATCH_TILE,
    DF_NEXT_TILE,
    DF_NEXT_LAYER,
    DF_DONE
  } df_state_e;

  df_state_e state, nstate;

  logic [7:0]  layer_id;
  logic [15:0] tile_cnt, tile_total;
  logic [63:0] layer_table_base;
  logic [63:0] tile_table_addr;

  // Accumulate 64-byte descriptor from 256-bit (32-byte) AXI beats
  logic [511:0] desc_buf;
  logic         beat_cnt;

  // VRFC/xvlog: cannot use layer_desc_t'(...).num_tile_hw inline in case — split cast + field
  layer_desc_t lyr_from_buf_comb;
  logic [11:0] lyr_num_tile_hw_comb;
  always_comb begin
    lyr_from_buf_comb    = layer_desc_t'(desc_buf[$bits(layer_desc_t)-1:0]);
    lyr_num_tile_hw_comb = lyr_from_buf_comb.num_tile_hw;
  end

  assign current_layer   = layer_id;
  assign all_layers_done = (state == DF_DONE);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) state <= DF_IDLE;
    else        state <= nstate;
  end

  always_comb begin
    nstate = state;
    case (state)
      DF_IDLE:          if (start) nstate = DF_FETCH_NET_AR;
      DF_FETCH_NET_AR:  if (axi_arready) nstate = DF_FETCH_NET_R;
      DF_FETCH_NET_R:   if (axi_rvalid && axi_rlast) nstate = DF_PARSE_NET;
      DF_PARSE_NET:     nstate = DF_FETCH_LAYER_AR;
      DF_FETCH_LAYER_AR: if (axi_arready) nstate = DF_FETCH_LAYER_R;
      DF_FETCH_LAYER_R: if (axi_rvalid && axi_rlast) nstate = DF_PARSE_LAYER;
      // BUGFIX: do not use registered tile_total here — it updates same posedge as state,
      // so nstate still sees OLD value (0 on first layer) → skip all tiles → deadlock / wrong counts.
      DF_PARSE_LAYER:   nstate = (lyr_num_tile_hw_comb != 12'd0) ? DF_FETCH_TILE_AR : DF_NEXT_LAYER;
      DF_FETCH_TILE_AR: if (axi_arready) nstate = DF_FETCH_TILE_R;
      DF_FETCH_TILE_R:  if (axi_rvalid && axi_rlast) nstate = DF_DISPATCH_TILE;
      DF_DISPATCH_TILE: if (tile_desc_ready) nstate = DF_NEXT_TILE;
      // After dispatching tile at index tile_cnt, next index would be tile_cnt+1.
      // Using (tile_cnt >= tile_total) is off-by-one: for 25 tiles (idx 0..24) when
      // tile_cnt==24 comb still sees 24 < 25 → spurious FETCH tile 25 → tc>tt and GS drift.
      DF_NEXT_TILE:     nstate = ((tile_total != 16'd0) && (tile_cnt + 16'd1 >= tile_total))
                          ? DF_NEXT_LAYER : DF_FETCH_TILE_AR;
      // Inclusive layer_end (TB L0-only uses end=0; old `>` wrongly continued to L1).
      DF_NEXT_LAYER:    nstate = (layer_id >= layer_end) ? DF_DONE : DF_FETCH_LAYER_AR;
      DF_DONE:          nstate = DF_IDLE;
      default:          nstate = DF_IDLE;
    endcase
  end

  // ───── AXI AR Channel ─────
  always_comb begin
    axi_arvalid = 1'b0;
    axi_araddr  = '0;
    axi_arlen   = 8'd1;  // 2 beats x 32B = 64B

    case (state)
      DF_FETCH_NET_AR: begin
        axi_arvalid = 1'b1;
        axi_araddr  = net_desc_base[39:0];
      end
      DF_FETCH_LAYER_AR: begin
        axi_arvalid = 1'b1;
        axi_araddr  = layer_table_base[39:0] + {layer_id, 6'd0}; // layer_id * 64
      end
      DF_FETCH_TILE_AR: begin
        axi_arvalid = 1'b1;
        axi_araddr  = tile_table_addr[39:0] + {tile_cnt[9:0], 6'd0};
      end
      default: ;
    endcase
  end

  // ───── AXI R Channel ─────
  // RREADY must be high on the AR-handshake cycle as well: state is still *_AR until the
  // posedge, but some DDR/interconnect models assert RVALID (even RLAST) in that same
  // cycle — if RREADY stays low, the read channel deadlocks.
  wire df_ar_hs_net   = (state == DF_FETCH_NET_AR)   && axi_arvalid && axi_arready;
  wire df_ar_hs_layer = (state == DF_FETCH_LAYER_AR) && axi_arvalid && axi_arready;
  wire df_ar_hs_tile  = (state == DF_FETCH_TILE_AR)  && axi_arvalid && axi_arready;
  assign axi_rready = (state == DF_FETCH_NET_R) ||
                       (state == DF_FETCH_LAYER_R) ||
                       (state == DF_FETCH_TILE_R) ||
                       df_ar_hs_net || df_ar_hs_layer || df_ar_hs_tile;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      desc_buf <= '0;
      beat_cnt <= 1'b0;
    end else if (axi_rvalid && axi_rready) begin
      if (!beat_cnt)
        desc_buf[255:0] <= axi_rdata;
      else
        desc_buf[511:256] <= axi_rdata;
      beat_cnt <= beat_cnt + 1'b1;
    end else if (state == DF_IDLE || state == DF_FETCH_NET_AR ||
                 state == DF_FETCH_LAYER_AR || state == DF_FETCH_TILE_AR) begin
      beat_cnt <= 1'b0;
    end
  end

  // ───── Descriptor Parsing ─────
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      net_desc         <= '0;
      net_desc_valid   <= 1'b0;
      layer_desc       <= '0;
      layer_desc_valid <= 1'b0;
      tile_desc        <= '0;
      tile_desc_valid  <= 1'b0;
      layer_id         <= '0;
      tile_cnt         <= '0;
      tile_total       <= '0;
      layer_table_base <= '0;
      tile_table_addr  <= '0;
    end else begin
      net_desc_valid   <= 1'b0;
      layer_desc_valid <= 1'b0;
      tile_desc_valid  <= 1'b0;

      case (state)
        DF_IDLE: if (start) begin
          layer_id <= layer_start;
          tile_cnt <= '0;
        end
        // Use packed-struct cast — old hard-coded slices (e.g. [139:128]) did not match
        // desc_pkg / generate_descriptors.py → tile_total often 0 → no tiles → no DDR writes.
        DF_PARSE_NET: begin
          automatic net_desc_t nd;
          nd = net_desc_t'(desc_buf[$bits(net_desc_t)-1:0]);
          net_desc <= nd;
          net_desc_valid <= 1'b1;
          layer_table_base <= nd.layer_table_base;
`ifdef ACCEL_DEBUG
          $display("[%t] %m [CHK-FETCH] PARSE_NET num_layers=%0d layer_tbl=%h wgt=%h act0=%h",
                   $time, nd.num_layers, nd.layer_table_base, nd.weight_arena_base,
                   nd.act0_arena_base);
`endif
        end
        DF_PARSE_LAYER: begin
          automatic layer_desc_t ld;
          ld = layer_desc_t'(desc_buf[$bits(layer_desc_t)-1:0]);
          layer_desc <= ld;
          layer_desc_valid <= 1'b1;
          tile_total <= ld.num_tile_hw;
          tile_table_addr <= ld.tile_table_offset;
          tile_cnt <= '0;
`ifdef ACCEL_DEBUG
          $display("[%t] %m [CHK-FETCH] PARSE_LAYER idx=%0d num_tiles=%0d tile_tbl=%h tmpl=%h",
                   $time, layer_id, ld.num_tile_hw, ld.tile_table_offset, ld.template_id);
`endif
        end
        DF_DISPATCH_TILE: begin
          tile_desc <= desc_buf[$bits(tile_desc_t)-1:0];
          tile_desc_valid <= 1'b1;
        end
        DF_NEXT_TILE: begin
          tile_cnt <= tile_cnt + 1;
        end
        DF_NEXT_LAYER: begin
          // Only advance when another layer remains; do not increment on the hop to DF_DONE
          // (else layer_end=0 still bumps layer_id to 1 and TB/PPU see a fake L1).
          if (layer_id < layer_end)
            layer_id <= layer_id + 1;
        end
        default: ;
      endcase
    end
  end

`ifdef ACCEL_DEBUG
  // Throttle: first 32 tile dispatches + every 512th (full model có rất nhiều tile)
  integer dbg_disp_cnt;
  df_state_e dbg_prev_st;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      dbg_disp_cnt <= 0;
      dbg_prev_st  <= DF_IDLE;
    end else begin
      dbg_prev_st <= state;
      if (state == DF_IDLE && start)
        dbg_disp_cnt <= 0;
      if (state == DF_DISPATCH_TILE && tile_desc_ready) begin
        automatic tile_desc_t td;
        td = tile_desc_t'(desc_buf[$bits(tile_desc_t)-1:0]);
        if (dbg_disp_cnt < 32 || dbg_disp_cnt[8:0] == 9'd0)
          $display("[%t] %m [CHK-FETCH] DISPATCH #%0d layer_idx=%0d tile_id=%0h dst_off=%h flags=%h",
                   $time, dbg_disp_cnt, layer_id, td.tile_id, td.dst_off, td.tile_flags);
        dbg_disp_cnt <= dbg_disp_cnt + 1;
      end
      if (state == DF_DONE && dbg_prev_st != DF_DONE)
        $display("[%t] %m [CHK-FETCH] DF_DONE (descriptor walk finished)", $time);
    end
  end
`endif

endmodule
