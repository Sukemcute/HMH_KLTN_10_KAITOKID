// ============================================================================
// Module : barrier_manager
// Project: YOLOv10n INT8 Accelerator — V4-VC707
//
// Manages 4 skip-connection barriers for YOLOv10n:
//   barrier_0: L3→L12 (F6)
//   barrier_1: L7→L15 (F12)
//   barrier_2: L14→L18 (not in original, but reserved)
//   barrier_3: L8→L21 (F8)
//
// Protocol:
//   Producer layer signals barrier_signal[id] when skip data is ready.
//   Consumer layer requests barrier_grant[id]; manager asserts grant
//   only when the corresponding signal has been set.
//   Barrier is auto-cleared after grant is consumed.
// ============================================================================
`timescale 1ns / 1ps

module barrier_manager
  import accel_pkg::*;
#(
  parameter int N_BARRIERS = BARRIER_COUNT  // 4
)(
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic                    soft_reset,

  // ═══════════ PRODUCER: skip data ready signals ═══════════
  input  logic [N_BARRIERS-1:0]   barrier_signal,

  // ═══════════ CONSUMER: request barrier grant ═══════════
  input  logic [N_BARRIERS-1:0]   barrier_request,
  output logic [N_BARRIERS-1:0]   barrier_grant,

  // ═══════════ STATUS ═══════════
  output logic [N_BARRIERS-1:0]   barrier_pending
);

  logic [N_BARRIERS-1:0] ready_r;

  assign barrier_pending = ready_r;

  generate
    for (genvar i = 0; i < N_BARRIERS; i++) begin : gen_barrier
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          ready_r[i] <= 1'b0;
        end else if (soft_reset) begin
          ready_r[i] <= 1'b0;
        end else begin
          // Set on signal from producer
          if (barrier_signal[i])
            ready_r[i] <= 1'b1;
          // Clear when consumer consumes the grant
          if (barrier_request[i] && ready_r[i])
            ready_r[i] <= 1'b0;
        end
      end

      // Grant = ready AND requested
      assign barrier_grant[i] = ready_r[i] && barrier_request[i];
    end
  endgenerate

endmodule
