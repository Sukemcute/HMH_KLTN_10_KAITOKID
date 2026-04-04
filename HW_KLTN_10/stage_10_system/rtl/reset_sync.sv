// ============================================================================
// Module : reset_sync
// Project: YOLOv10n INT8 Accelerator — V4-VC707
//
// 2-stage synchronizer for asynchronous active-low reset.
// Ensures clean, metastability-free reset de-assertion in the clock domain.
// ============================================================================
`timescale 1ns / 1ps

module reset_sync (
  input  logic clk,
  input  logic rst_async_n,   // asynchronous active-low reset
  output logic rst_sync_n     // synchronous active-low reset (safe)
);

  logic [1:0] sync_r;

  always_ff @(posedge clk or negedge rst_async_n) begin
    if (!rst_async_n)
      sync_r <= 2'b00;
    else
      sync_r <= {sync_r[0], 1'b1};
  end

  assign rst_sync_n = sync_r[1];

endmodule
