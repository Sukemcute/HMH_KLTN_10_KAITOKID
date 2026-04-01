`timescale 1ns/1ps
// Dual-RUNNING 4-phase scheduler for 4 subclusters within one SuperCluster.
// Assigns roles: 2xRUNNING + 1xFILLING + 1xDRAINING/HOLD.
// Arbitrates shared external memory port access.
module local_arbiter #(
  parameter int NUM_SUBS = 4
)(
  input  logic              clk,
  input  logic              rst_n,

  // Tile queue (from global_scheduler)
  input  logic              tile_available,
  input  desc_pkg::tile_desc_t next_tile,
  output logic              tile_consumed,
  // At least one subcluster slot free (ROLE_IDLE); used to backpressure GS ingress.
  output logic              has_idle_sub,

  // Per-subcluster status
  input  accel_pkg::tile_state_e sub_state [NUM_SUBS],
  input  logic              sub_tile_done [NUM_SUBS],
  input  logic              sub_dma_wr_req [NUM_SUBS],

  // Role assignment output
  output accel_pkg::sc_role_e sub_role [NUM_SUBS],

  // External port arbitration
  input  logic              ext_port_ready,
  output logic [1:0]        ext_port_grant_sub,
  output logic              ext_port_is_read,

  // Tile dispatch to subclusters
  output logic              sub_tile_valid [NUM_SUBS],
  output desc_pkg::tile_desc_t sub_tile [NUM_SUBS]
);
  import accel_pkg::*;

  sc_role_e role_reg [NUM_SUBS];
  logic [1:0] fill_target;
  logic       dispatch_pending;

  assign sub_role = role_reg;

  always_comb begin
    has_idle_sub = 1'b0;
    for (int i = 0; i < NUM_SUBS; i++)
      if (role_reg[i] == ROLE_IDLE)
        has_idle_sub = 1'b1;
  end

  // ───── Role Management ─────
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < NUM_SUBS; i++) begin
        role_reg[i] <= ROLE_IDLE;
      end
      fill_target      <= '0;
      dispatch_pending <= 1'b0;
    end else begin
      dispatch_pending <= 1'b0;

      for (int i = 0; i < NUM_SUBS; i++) begin
        // Promote FILLING → RUNNING when sub leaves TILE_IDLE (started this tile).
        // PE_PASS / template bypass skips TILE_RUN_COMPUTE (LOAD_CFG → SWIZZLE_STORE);
        // only checking RUN_COMPUTE leaves ROLE_FILLING through DONE→IDLE so no sub
        // ever looks ROLE_IDLE → no further tile_consumed → supercluster ingress deadlock.
        if (role_reg[i] == ROLE_FILLING && sub_state[i] != TILE_IDLE)
          role_reg[i] <= ROLE_RUNNING;

        // RUNNING → DRAINING when tile is done
        if (role_reg[i] == ROLE_RUNNING && sub_tile_done[i])
          role_reg[i] <= ROLE_DRAINING;

        // DRAINING → IDLE when sub returns to idle
        if (role_reg[i] == ROLE_DRAINING && sub_state[i] == TILE_IDLE)
          role_reg[i] <= ROLE_IDLE;
      end

      // Assign FILLING to an idle sub when a tile is available
      if (tile_available && !dispatch_pending) begin
        for (int i = 0; i < NUM_SUBS; i++) begin
          if (role_reg[i] == ROLE_IDLE) begin
            role_reg[i]      <= ROLE_FILLING;
            fill_target      <= i[1:0];
            dispatch_pending <= 1'b1;
            break;
          end
        end
      end
    end
  end

  // ───── Tile Dispatch ─────
  always_comb begin
    tile_consumed = 1'b0;
    for (int i = 0; i < NUM_SUBS; i++) begin
      sub_tile_valid[i] = 1'b0;
      sub_tile[i]       = next_tile;
    end

    if (dispatch_pending) begin
      sub_tile_valid[fill_target] = 1'b1;
      tile_consumed               = 1'b1;
    end
  end

  // ───── External Port Arbitration ─────
  logic any_swizzle_wr;
  always_comb begin
    any_swizzle_wr = 1'b0;
    for (int j = 0; j < NUM_SUBS; j++)
      if (sub_state[j] == TILE_SWIZZLE_STORE && sub_dma_wr_req[j])
        any_swizzle_wr = 1'b1;
  end

  // Priority: SWIZZLE_STORE + dma_wr wins; else FILLING then DRAINING (DR overwrites).
  always_comb begin
    ext_port_grant_sub = '0;
    ext_port_is_read   = 1'b1;

    if (ext_port_ready) begin
      if (any_swizzle_wr) begin
        for (int j = 0; j < NUM_SUBS; j++) begin
          if (sub_state[j] == TILE_SWIZZLE_STORE && sub_dma_wr_req[j]) begin
            ext_port_grant_sub = j[1:0];
            ext_port_is_read   = 1'b0;
            break;
          end
        end
      end else begin
        for (int j = 0; j < NUM_SUBS; j++) begin
          if (role_reg[j] == ROLE_FILLING) begin
            ext_port_grant_sub = j[1:0];
            ext_port_is_read   = 1'b1;
            break;
          end
        end
        for (int j = 0; j < NUM_SUBS; j++) begin
          if (role_reg[j] == ROLE_DRAINING) begin
            ext_port_grant_sub = j[1:0];
            ext_port_is_read   = 1'b0;
            break;
          end
        end
      end
    end
  end

endmodule
