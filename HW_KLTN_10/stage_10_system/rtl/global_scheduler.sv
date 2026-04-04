// ============================================================================
// Module : global_scheduler
// Project: YOLOv10n INT8 Accelerator — V4-VC707
//
// Dispatches tile descriptors from desc_fetch_engine to 4 SuperClusters
// using round-robin scheduling with backpressure.
// ============================================================================
`timescale 1ns / 1ps

module global_scheduler
  import accel_pkg::*;
  import desc_pkg::*;
#(
  parameter int N_SC = N_SUPER_CLUSTERS  // 4
)(
  input  logic          clk,
  input  logic          rst_n,
  input  logic          enable,

  // ═══════════ INPUT: from desc_fetch_engine ═══════════
  input  logic          desc_valid,
  input  layer_desc_t   desc_layer,
  input  tile_desc_t    desc_tile,
  output logic          desc_ready,

  // ═══════════ OUTPUT: to 4 SuperClusters ═══════════
  output logic          sc_tile_valid [N_SC],
  output layer_desc_t   sc_layer_desc [N_SC],
  output tile_desc_t    sc_tile_desc  [N_SC],
  input  logic          sc_tile_ready [N_SC],

  // ═══════════ STATUS ═══════════
  output logic [15:0]   total_tiles_dispatched,
  output logic          all_dispatched,
  input  logic          sc_idle [N_SC]
);

  // Round-robin pointer
  logic [$clog2(N_SC)-1:0] rr_ptr;

  // Dispatch state
  typedef enum logic [1:0] {
    S_IDLE, S_DISPATCH, S_WAIT_ACCEPT
  } sched_state_e;
  sched_state_e state;

  // Find next ready SC (round-robin with skip)
  logic [$clog2(N_SC)-1:0] next_sc;
  logic found;

  always_comb begin
    found = 1'b0;
    next_sc = rr_ptr;
    for (int i = 0; i < N_SC; i++) begin
      automatic int idx = (rr_ptr + i) % N_SC;
      if (!found && sc_tile_ready[idx]) begin
        next_sc = idx[$clog2(N_SC)-1:0];
        found = 1'b1;
      end
    end
  end

  // Default outputs
  always_comb begin
    for (int i = 0; i < N_SC; i++) begin
      sc_tile_valid[i] = 1'b0;
      sc_layer_desc[i] = desc_layer;
      sc_tile_desc[i]  = desc_tile;
    end
    desc_ready = 1'b0;

    if (state == S_DISPATCH && found) begin
      sc_tile_valid[next_sc] = desc_valid;
      desc_ready = sc_tile_ready[next_sc];
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state                  <= S_IDLE;
      rr_ptr                 <= '0;
      total_tiles_dispatched <= 16'd0;
      all_dispatched         <= 1'b0;
    end else begin
      case (state)
        S_IDLE: begin
          all_dispatched <= 1'b0;
          if (enable && desc_valid)
            state <= S_DISPATCH;
        end

        S_DISPATCH: begin
          if (!desc_valid) begin
            all_dispatched <= 1'b1;
            state <= S_IDLE;
          end else if (found && sc_tile_ready[next_sc]) begin
            total_tiles_dispatched <= total_tiles_dispatched + 16'd1;
            rr_ptr <= (next_sc + 1) % N_SC;
            // Stay in DISPATCH for next tile
          end
        end

        default: state <= S_IDLE;
      endcase
    end
  end

endmodule
