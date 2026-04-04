// ============================================================================
// Module : local_arbiter_v2
// Project: YOLOv10n INT8 Accelerator — V4-VC707
//
// Triple-RUNNING scheduler for 4 subclusters within one SuperCluster.
// At any instant: 3 subs COMPUTE, 1 sub FILL+DRAIN.
// Roles rotate when the FILL sub finishes loading and a COMPUTE sub finishes.
// ============================================================================
`timescale 1ns / 1ps

module local_arbiter_v2
  import accel_pkg::*;
  import desc_pkg::*;
#(
  parameter int N_SUBS = N_SUBS_PER_SC  // 4
)(
  input  logic          clk,
  input  logic          rst_n,

  // ═══════════ TILE INPUT (from SuperCluster ingress FIFO) ═══════════
  input  logic          tile_valid,
  input  layer_desc_t   layer_desc_in,
  input  tile_desc_t    tile_desc_in,
  output logic          tile_accept,

  // ═══════════ PER-SUB DESCRIPTOR OUTPUTS ═══════════
  output logic          sub_tile_valid [N_SUBS],
  output layer_desc_t   sub_layer_desc [N_SUBS],
  output tile_desc_t    sub_tile_desc  [N_SUBS],
  input  logic          sub_tile_accept[N_SUBS],
  input  logic          sub_tile_done  [N_SUBS],

  // ═══════════ DMA GRANT (one sub at a time for fill/drain) ═══════════
  output logic [1:0]    dma_grant_id,
  output logic          dma_fill_grant,
  output logic          dma_drain_grant,

  // ═══════════ STATUS ═══════════
  output sub_role_e     sub_roles  [N_SUBS],
  output logic          all_idle
);

  // Role tracking per sub
  sub_role_e role_r [N_SUBS];

  // Rotation pointer: which sub is currently in FILL/DRAIN role
  logic [1:0] fill_ptr;

  // Pending tile buffer for the sub that will switch from FILL to COMPUTE
  logic          pending_valid;
  layer_desc_t   pending_layer;
  tile_desc_t    pending_tile;

  // Track which sub just finished compute
  logic [1:0] done_sub_id;
  logic       any_compute_done;

  always_comb begin
    any_compute_done = 1'b0;
    done_sub_id = 2'd0;
    for (int i = 0; i < N_SUBS; i++) begin
      if (sub_tile_done[i] && role_r[i] == ROLE_COMPUTE) begin
        any_compute_done = 1'b1;
        done_sub_id = i[1:0];
      end
    end
  end

  // DMA grant: fill_ptr sub gets DMA access
  assign dma_grant_id   = fill_ptr;
  assign dma_fill_grant  = (role_r[fill_ptr] == ROLE_FILL);
  assign dma_drain_grant = (role_r[fill_ptr] == ROLE_DRAIN);

  // Accept tile when we can buffer it for the fill sub
  assign tile_accept = tile_valid && !pending_valid &&
                       (role_r[fill_ptr] == ROLE_FILL || role_r[fill_ptr] == ROLE_IDLE);

  // All idle detection
  logic all_idle_comb;
  always_comb begin
    all_idle_comb = 1'b1;
    for (int i = 0; i < N_SUBS; i++)
      if (role_r[i] != ROLE_IDLE) all_idle_comb = 1'b0;
  end
  assign all_idle = all_idle_comb;

  // Forward descriptors to subs
  always_comb begin
    for (int i = 0; i < N_SUBS; i++) begin
      sub_tile_valid[i] = 1'b0;
      sub_layer_desc[i] = '0;
      sub_tile_desc[i]  = '0;
    end
    if (pending_valid) begin
      sub_tile_valid[fill_ptr] = 1'b1;
      sub_layer_desc[fill_ptr] = pending_layer;
      sub_tile_desc[fill_ptr]  = pending_tile;
    end
  end

  // Role output
  always_comb begin
    for (int i = 0; i < N_SUBS; i++)
      sub_roles[i] = role_r[i];
  end

  // Main FSM
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      fill_ptr      <= 2'd3;  // Sub-3 starts as FILL
      pending_valid <= 1'b0;
      pending_layer <= '0;
      pending_tile  <= '0;
      for (int i = 0; i < N_SUBS; i++)
        role_r[i] <= ROLE_IDLE;
    end else begin
      // Accept new tile into pending buffer
      if (tile_accept) begin
        pending_valid <= 1'b1;
        pending_layer <= layer_desc_in;
        pending_tile  <= tile_desc_in;
        if (role_r[fill_ptr] == ROLE_IDLE)
          role_r[fill_ptr] <= ROLE_FILL;
      end

      // Fill sub accepts the descriptor → transition to COMPUTE
      if (pending_valid && sub_tile_accept[fill_ptr]) begin
        pending_valid <= 1'b0;
        role_r[fill_ptr] <= ROLE_COMPUTE;
      end

      // Rotation: when a COMPUTE sub finishes, it becomes the new FILL sub
      if (any_compute_done) begin
        role_r[done_sub_id] <= ROLE_FILL;
        fill_ptr <= done_sub_id;
      end

      // Initial bootstrap: assign first 3 subs to COMPUTE, sub-3 to FILL
      // This happens naturally via tile dispatch
    end
  end

endmodule
