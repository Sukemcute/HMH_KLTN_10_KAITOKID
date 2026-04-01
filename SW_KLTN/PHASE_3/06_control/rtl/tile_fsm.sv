`timescale 1ns/1ps
// Tile execution FSM: orchestrates the full lifecycle of one tile.
//
// Normal conv: IDLE → LOAD_CFG → PREFILL_WT → PREFILL_IN → [PREFILL_SKIP] →
//              WAIT_READY → RUN_COMPUTE (multi-cycle) ↔ ACCUMULATE →
//              POST_PROCESS (multi-cycle) → SWIZZLE_STORE (multi-cycle) → DONE
//
// PE_PASS (upsample/concat/identity): IDLE → LOAD_CFG → PREFILL_IN →
//              SWIZZLE_STORE → DONE
module tile_fsm (
  input  logic              clk,
  input  logic              rst_n,

  // Tile descriptor input (from local_arbiter)
  input  logic              tile_valid,
  input  desc_pkg::tile_desc_t  tile_desc,
  input  desc_pkg::layer_desc_t layer_desc,
  output logic              tile_accept,

  // GLB control
  output logic              glb_wr_en,
  output logic              glb_rd_en,
  output logic              glb_wr_is_weight,
  output logic              glb_wr_is_skip,

  // PE cluster control
  output logic              pe_en,
  output logic              pe_clear_psum,
  output accel_pkg::pe_mode_e pe_mode,

  // PPU control
  output logic              ppu_en,
  output logic              ppu_last_pass,

  // Swizzle control
  output logic              swizzle_start,
  input  logic              swizzle_done,

  // Subcluster completion handshakes
  input  logic              compute_done,
  input  logic              ppu_done,

  // DMA read request
  output logic              dma_rd_req,
  output logic [39:0]       dma_rd_addr,
  output logic [15:0]       dma_rd_len,
  input  logic              dma_rd_done,

  // DMA write request
  output logic              dma_wr_req,
  output logic [39:0]       dma_wr_addr,
  output logic [15:0]       dma_wr_len,
  input  logic              dma_wr_done,

  // Barrier interface
  output logic              barrier_wait_req,
  input  logic              barrier_grant,
  output logic              barrier_signal,

  // Status
  output accel_pkg::tile_state_e state,
  output logic              tile_done,
  output logic              layer_done
);
  import accel_pkg::*;
  import desc_pkg::*;

  tile_state_e cur_state, nxt_state;
  assign state = cur_state;

  // Captured descriptor fields
  logic [15:0] flags;
  logic [3:0]  cin_pass_cnt, cin_pass_max;
  logic [3:0]  k_pass_cnt, k_pass_max;
  pe_mode_e    mode_reg;

  // Tile flag bit definitions
  wire flag_first_tile     = flags[0];
  wire flag_last_tile      = flags[1];
  wire flag_has_skip       = flags[3];
  wire flag_need_swizzle   = flags[4];
  wire flag_need_spill     = flags[5];
  wire flag_barrier_before = flags[6];
  wire flag_barrier_after  = flags[7];

  // If num_*_pass==0, (max-1) underflows unsigned → last_* never true → infinite
  // RUN_COMPUTE/ACCUMULATE and SC ingress FIFO fills (GS_WAIT_ACCEPT).
  wire last_cin = (cin_pass_max == 4'd0) || (cin_pass_cnt >= cin_pass_max - 4'd1);
  wire last_k   = (k_pass_max == 4'd0) || (k_pass_cnt >= k_pass_max - 4'd1);
  wire all_passes_done = last_cin && last_k;

  // DMA byte-length computation from descriptor fields
  // Input region for conv: (valid_h-1)*stride + kh rows, (valid_w-1)*stride + kw cols
  wire [15:0] in_h_actual = (16'(tile_desc.valid_h) > 16'd0)
      ? (16'(tile_desc.valid_h) - 16'd1) * 16'(layer_desc.sh) + 16'(layer_desc.kh)
      : 16'd0;
  wire [15:0] in_w_actual = (16'(tile_desc.valid_w) > 16'd0)
      ? (16'(tile_desc.valid_w) - 16'd1) * 16'(layer_desc.sw) + 16'(layer_desc.kw)
      : 16'd0;
  wire [15:0] wt_byte_len    = 16'(layer_desc.tile_cout) * 16'(layer_desc.tile_cin)
                                * 16'(layer_desc.kh) * 16'(layer_desc.kw);
  wire [15:0] in_byte_len    = in_h_actual * in_w_actual * 16'(layer_desc.tile_cin);
  wire [15:0] skip_byte_len  = 16'(tile_desc.valid_h) * 16'(tile_desc.valid_w)
                                * 16'(layer_desc.tile_cout);
  wire [15:0] out_byte_len   = 16'(tile_desc.valid_h) * 16'(tile_desc.valid_w)
                                * 16'(layer_desc.tile_cout);

  // ───── State Register ─────
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      cur_state <= TILE_IDLE;
    else
      cur_state <= nxt_state;
  end

  // ───── Next-State Logic ─────
  always_comb begin
    nxt_state = cur_state;
    case (cur_state)
      TILE_IDLE:
        if (tile_valid) nxt_state = TILE_LOAD_CFG;

      TILE_LOAD_CFG:
        nxt_state = (mode_reg == PE_PASS) ? TILE_PREFILL_IN : TILE_PREFILL_WT;

      TILE_PREFILL_WT:
        if (dma_rd_done) nxt_state = TILE_PREFILL_IN;

      TILE_PREFILL_IN:
        if (dma_rd_done) begin
          if (mode_reg == PE_PASS)
            nxt_state = TILE_SWIZZLE_STORE;
          else if (flag_has_skip)
            nxt_state = TILE_PREFILL_SKIP;
          else
            nxt_state = TILE_WAIT_READY;
        end

      TILE_PREFILL_SKIP:
        if (dma_rd_done) nxt_state = TILE_WAIT_READY;

      TILE_WAIT_READY: begin
        if (flag_barrier_before) begin
          if (barrier_grant) nxt_state = TILE_RUN_COMPUTE;
        end else
          nxt_state = TILE_RUN_COMPUTE;
      end

      TILE_RUN_COMPUTE:
        if (compute_done) nxt_state = TILE_ACCUMULATE;

      TILE_ACCUMULATE: begin
        if (all_passes_done)
          nxt_state = TILE_POST_PROCESS;
        else
          nxt_state = TILE_RUN_COMPUTE;
      end

      TILE_POST_PROCESS:
        if (ppu_done) nxt_state = TILE_SWIZZLE_STORE;

      TILE_SWIZZLE_STORE: begin
        if (flag_need_swizzle && !swizzle_done)
          nxt_state = TILE_SWIZZLE_STORE;
        else if (flag_need_spill && !dma_wr_done)
          nxt_state = TILE_SWIZZLE_STORE;
        else
          nxt_state = TILE_DONE;
      end

      TILE_DONE:
        nxt_state = TILE_IDLE;

      default: nxt_state = TILE_IDLE;
    endcase
  end

  // ───── Pass Counters ─────
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      cin_pass_cnt <= '0;
      k_pass_cnt   <= '0;
      cin_pass_max <= '0;
      k_pass_max   <= '0;
      flags        <= '0;
      mode_reg     <= PE_RS3;
    end else begin
      case (cur_state)
        TILE_IDLE: if (tile_valid) begin
          flags        <= tile_desc.tile_flags;
          cin_pass_max <= layer_desc.num_cin_pass;
          k_pass_max   <= layer_desc.num_k_pass;
          mode_reg     <= pe_mode_e'(layer_desc.template_id);
          cin_pass_cnt <= '0;
          k_pass_cnt   <= '0;
        end
        TILE_ACCUMULATE: begin
          if (!all_passes_done) begin
            if (!last_cin)
              cin_pass_cnt <= cin_pass_cnt + 1;
            else begin
              cin_pass_cnt <= '0;
              k_pass_cnt   <= k_pass_cnt + 1;
            end
          end
        end
        default: ;
      endcase
    end
  end

  // ───── Output Control Signals ─────
  always_comb begin
    tile_accept      = (cur_state == TILE_IDLE) && tile_valid;
    glb_wr_en        = (cur_state == TILE_PREFILL_WT)
                     || (cur_state == TILE_PREFILL_IN)
                     || (cur_state == TILE_PREFILL_SKIP);
    glb_wr_is_weight = (cur_state == TILE_PREFILL_WT);
    glb_wr_is_skip   = (cur_state == TILE_PREFILL_SKIP);
    glb_rd_en        = (cur_state == TILE_RUN_COMPUTE);
    pe_en            = (cur_state == TILE_RUN_COMPUTE);
    pe_clear_psum    = (cur_state == TILE_RUN_COMPUTE) && flag_first_tile
                       && (cin_pass_cnt == 0) && (k_pass_cnt == 0);
    pe_mode          = mode_reg;
    ppu_en           = (cur_state == TILE_POST_PROCESS);
    ppu_last_pass    = (cur_state == TILE_POST_PROCESS);
    swizzle_start    = (cur_state == TILE_SWIZZLE_STORE) && flag_need_swizzle;
    barrier_wait_req = (cur_state == TILE_WAIT_READY) && flag_barrier_before;
    barrier_signal   = (cur_state == TILE_POST_PROCESS) && flag_barrier_after;
    tile_done        = (cur_state == TILE_DONE);
    layer_done       = (cur_state == TILE_DONE) && flag_last_tile;

    dma_rd_req = (cur_state == TILE_PREFILL_WT)
               || (cur_state == TILE_PREFILL_IN)
               || (cur_state == TILE_PREFILL_SKIP);

    case (cur_state)
      TILE_PREFILL_WT: begin
        dma_rd_addr = {8'd0, tile_desc.src_w_off};
        dma_rd_len  = (wt_byte_len != '0) ? wt_byte_len : 16'd32;
      end
      TILE_PREFILL_SKIP: begin
        dma_rd_addr = {8'd0, tile_desc.src_skip_off};
        dma_rd_len  = (skip_byte_len != '0) ? skip_byte_len : 16'd32;
      end
      default: begin
        dma_rd_addr = {8'd0, tile_desc.src_in_off};
        dma_rd_len  = (in_byte_len != '0) ? in_byte_len : 16'd32;
      end
    endcase

    dma_wr_req  = (cur_state == TILE_SWIZZLE_STORE) && flag_need_spill;
    dma_wr_addr = {8'd0, tile_desc.dst_off};
    dma_wr_len  = (out_byte_len != '0) ? out_byte_len : 16'd32;
  end

`ifdef ACCEL_DEBUG
`ifndef ACCEL_TILE_LOG_MAX
  `define ACCEL_TILE_LOG_MAX 64
`endif
  integer dbg_tile_issue_n;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      dbg_tile_issue_n <= 0;
    else if (cur_state == TILE_IDLE && tile_valid) begin
      if (dbg_tile_issue_n < `ACCEL_TILE_LOG_MAX
          || dbg_tile_issue_n[9:0] == 10'd0)
        $display("[%t] %m [CHK-TILE] ISSUE tile_id=%0h layer_id=%0d dst_off=%h spill=%b swz=%b",
                 $time, tile_desc.tile_id, tile_desc.layer_id, tile_desc.dst_off,
                 tile_desc.tile_flags[5], tile_desc.tile_flags[4]);
      dbg_tile_issue_n <= dbg_tile_issue_n + 1;
    end
  end
`endif

endmodule
