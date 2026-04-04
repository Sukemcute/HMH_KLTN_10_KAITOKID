// Minimal probe: isolate Vivado xelab crash on desc_pkg structs + tile_fsm
`timescale 1ns / 1ps
module tb_tile_fsm_probe;
  import accel_pkg::*;
  import desc_pkg::*;
  logic clk, rst_n;
  logic tile_valid;
  layer_desc_t layer_desc;
  tile_desc_t tile_desc;
  logic tile_accept, shadow_latch, dma_start, dma_is_write, dma_done;
  logic seq_start, seq_done, ppu_start, ppu_done;
  logic swizzle_start, swizzle_done;
  logic barrier_wait_req, barrier_grant, barrier_signal;
  logic page_swap;
  tile_state_e state;
  logic tile_done;
  logic [3:0] cur_k_pass_idx;

  assign dma_done = dma_start;

  tile_fsm u_dut (.*);

  initial clk = 0;
  always #2 clk = ~clk;
  initial begin
    #10;
    $finish;
  end
endmodule
