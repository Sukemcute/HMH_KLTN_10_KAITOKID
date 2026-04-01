`timescale 1ns/1ps
// Skip dependency barrier scoreboard.
// 32 barrier points; producer signals completion, consumer checks readiness.
// YOLOv10n uses 4 barriers: L6→L12, L4→L15, L13→L18, L8→L21.
module barrier_manager #(
  parameter int NUM_BARRIERS = 32
)(
  input  logic              clk,
  input  logic              rst_n,
  input  logic              clear_all,

  // Producer signals barrier completion
  input  logic              signal_valid,
  input  logic [4:0]        signal_barrier_id,

  // Consumer checks barrier readiness
  input  logic              wait_valid,
  input  logic [4:0]        wait_barrier_id,
  output logic              wait_grant,

  output logic [NUM_BARRIERS-1:0] scoreboard
);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n || clear_all)
      scoreboard <= '0;
    else if (signal_valid)
      scoreboard[signal_barrier_id] <= 1'b1;
  end

  assign wait_grant = wait_valid & scoreboard[wait_barrier_id];

endmodule
