`timescale 1ns/1ps
// Global scheduler: receives tile descriptors from desc_fetch_engine and
// dispatches them to 4 SuperClusters based on sc_mask routing.
module global_scheduler (
  input  logic               clk,
  input  logic               rst_n,

  // From desc_fetch_engine
  input  desc_pkg::layer_desc_t layer_desc,
  input  logic                   layer_valid,
  input  desc_pkg::tile_desc_t  tile_desc_in,
  input  logic                   tile_valid,
  output logic                   tile_accept,

  // To 4 SuperClusters
  output desc_pkg::tile_desc_t  sc_tile [4],
  output logic                   sc_tile_valid [4],
  input  logic                   sc_tile_accept [4],

  // Layer tracking
  output logic [4:0]            current_layer_id,
  output logic                   layer_complete,
  output logic                   inference_complete
);
  import desc_pkg::*;

  typedef enum logic [1:0] {
    GS_IDLE,
    GS_DISPATCH,
    GS_WAIT_ACCEPT,
    GS_DONE
  } gs_state_e;

  gs_state_e state, nstate;

  tile_desc_t  tile_reg;
  logic [3:0]  sc_mask;
  logic [3:0]  sc_dispatched;  // which SCs have accepted
  logic [3:0]  accepted_this_cycle;
  logic [15:0] tiles_dispatched, tiles_total;
  logic [4:0]  layer_id_reg;
  layer_desc_t layer_reg;

  assign current_layer_id = layer_id_reg;

  // SC accepted handshake this cycle (before nstate comb uses it)
  always_comb begin
    for (int i = 0; i < 4; i++)
      accepted_this_cycle[i] = sc_tile_accept[i] && sc_tile_valid[i];
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) state <= GS_IDLE;
    else        state <= nstate;
  end

  always_comb begin
    nstate = state;
    case (state)
      GS_IDLE: begin
        if (tile_valid)
          nstate = GS_DISPATCH;
      end
      GS_DISPATCH: begin
        // Fast-path: if all target SCs accepted during this same cycle,
        // skip WAIT_ACCEPT entirely.
        if (accepted_this_cycle == sc_mask)
          nstate = GS_IDLE;
        else
          nstate = GS_WAIT_ACCEPT;
      end
      GS_WAIT_ACCEPT: begin
        if ((sc_dispatched | accepted_this_cycle) == sc_mask)
          nstate = GS_IDLE;
      end
      GS_DONE:
        nstate = GS_IDLE;
      default: nstate = GS_IDLE;
    endcase
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      tile_reg         <= '0;
      sc_mask          <= '0;
      sc_dispatched    <= '0;
      tiles_dispatched <= '0;
      tiles_total      <= '0;
      layer_id_reg     <= '0;
      layer_reg        <= '0;
      layer_complete   <= 1'b0;
      inference_complete <= 1'b0;
    end else begin
      layer_complete     <= 1'b0;
      inference_complete <= 1'b0;

      if (layer_valid) begin
        layer_reg    <= layer_desc;
        layer_id_reg <= layer_desc.layer_id;
        tiles_total  <= {4'd0, layer_desc.num_tile_hw};
        tiles_dispatched <= '0;
      end

      case (state)
        GS_IDLE: begin
          if (tile_valid) begin
            tile_reg      <= tile_desc_in;
            sc_mask       <= tile_desc_in.sc_mask;
            sc_dispatched <= '0;
          end
        end
        GS_DISPATCH: begin
          // Capture any SC acceptances that occur during the DISPATCH cycle.
          // Without this, SCs that accept combinationally in DISPATCH are
          // missed because sc_dispatched was only updated in WAIT_ACCEPT,
          // by which time the SC has already dropped tile_accept.
          sc_dispatched <= accepted_this_cycle;
          if (accepted_this_cycle == sc_mask) begin
            tiles_dispatched <= tiles_dispatched + 1;
            if (tiles_dispatched + 1 >= tiles_total)
              layer_complete <= 1'b1;
          end
        end
        GS_WAIT_ACCEPT: begin
          sc_dispatched <= sc_dispatched | accepted_this_cycle;
          if ((sc_dispatched | accepted_this_cycle) == sc_mask) begin
            tiles_dispatched <= tiles_dispatched + 1;
            if (tiles_dispatched + 1 >= tiles_total)
              layer_complete <= 1'b1;
          end
        end
        default: ;
      endcase
    end
  end

  // ───── Output to SCs ─────
  always_comb begin
    tile_accept = (state == GS_IDLE) && tile_valid;
    for (int i = 0; i < 4; i++) begin
      sc_tile[i]       = tile_reg;
      sc_tile_valid[i] = (state == GS_DISPATCH || state == GS_WAIT_ACCEPT)
                          && sc_mask[i] && !sc_dispatched[i];
    end
  end

`ifdef ACCEL_DEBUG
  logic        dbg_prev_layer_complete;
  integer      dbg_gs_tile_in;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      dbg_prev_layer_complete <= 1'b0;
      dbg_gs_tile_in          <= 0;
    end else begin
      if (layer_valid)
        $display("[%t] %m [CHK-GS] LAYER_DESC layer_id=%0d num_tiles=%0d tmpl=%h",
                 $time, layer_desc.layer_id, layer_desc.num_tile_hw,
                 layer_desc.template_id);
      if (state == GS_IDLE && tile_valid) begin
        if (dbg_gs_tile_in < 32 || dbg_gs_tile_in[9:0] == 10'd0)
          $display("[%t] %m [CHK-GS] TILE_IN tile_id=%0h sc_mask=%b dst_off=%h",
                   $time, tile_desc_in.tile_id, tile_desc_in.sc_mask,
                   tile_desc_in.dst_off);
        dbg_gs_tile_in <= dbg_gs_tile_in + 1;
      end
      if (layer_complete && !dbg_prev_layer_complete)
        $display("[%t] %m [CHK-GS] LAYER_COMPLETE layer_id=%0d tiles_total=%0d",
                 $time, layer_id_reg, tiles_total);
      dbg_prev_layer_complete <= layer_complete;
    end
  end
`endif

endmodule
