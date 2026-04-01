// ============================================================================
// Module : tile_fsm
// Project: YOLOv10n INT8 Accelerator — V4-VC707
//
// Description:
//   Phase-level controller: "KHI NÀO làm gì" cho tile execution.
//   Orchestrates the high-level sequence of operations for one tile.
//
//   10 states:
//     TS_IDLE       — Waiting for tile descriptor
//     TS_LOAD_DESC  — Latch descriptor into shadow_reg_file (1 cycle)
//     TS_PREFILL_WT — DMA weights from DDR3 → GLB weight banks
//     TS_PREFILL_IN — DMA input activations from DDR3 → GLB input banks
//     TS_COMPUTE    — compute_sequencer runs inner loops
//     TS_PE_DRAIN   — Wait for PE pipeline to flush (5 cycles)
//     TS_PPU_RUN    — Wait for PPU pipeline to flush (5 cycles)
//     TS_SWIZZLE    — swizzle_engine transforms layout (if needed)
//     TS_WRITEBACK  — DMA output from GLB → DDR3 (if spill needed)
//     TS_DONE       — Signal completion, return to IDLE
//
//   Special paths:
//     PE_PASS mode: IDLE → LOAD_DESC → PREFILL_IN → SWIZZLE → DONE
//       (Skip weights, compute, and PPU entirely.)
//
//     DW7x7 multipass: COMPUTE loops num_k_pass times:
//       COMPUTE → PE_DRAIN → COMPUTE → PE_DRAIN → ... → PPU_RUN → DONE
//       (PPU fires only on final pass.)
//
//     Barrier wait: If barrier_wait=1, stay in TS_LOAD_DESC until grant.
// ============================================================================
`timescale 1ns / 1ps

module tile_fsm
  import accel_pkg::*;
  import desc_pkg::*;
(
  input  logic          clk,
  input  logic          rst_n,

  // ── Descriptor input (from scheduler or testbench) ──
  input  logic          tile_valid,        // New tile available
  input  layer_desc_t   layer_desc,        // Layer configuration
  input  tile_desc_t    tile_desc,         // Tile-specific config
  output logic          tile_accept,       // FSM accepted tile (1-cycle pulse)

  // ── Shadow reg file control ──
  output logic          shadow_latch,      // Pulse to latch descriptor

  // ── DMA control (simplified handshake) ──
  output logic          dma_start,         // Start DMA transfer
  output logic          dma_is_write,      // 0=read(DDR→GLB), 1=write(GLB→DDR)
  input  logic          dma_done,          // DMA transfer complete

  // ── Compute sequencer control ──
  output logic          seq_start,         // Start compute iteration
  input  logic          seq_done,          // All iterations complete

  // ── PPU control ──
  output logic          ppu_start,         // Trigger PPU pipeline
  input  logic          ppu_done,          // PPU pipeline drained

  // ── Swizzle control ──
  output logic          swizzle_start,     // Start layout transform
  input  logic          swizzle_done,      // Transform complete

  // ── Barrier interface ──
  output logic          barrier_wait_req,  // Request barrier grant
  input  logic          barrier_grant,     // Barrier released
  output logic          barrier_signal,    // Signal barrier completion

  // ── Page control (double-buffer) ──
  output logic          page_swap,         // Toggle active/shadow page

  // ── Status ──
  output tile_state_e   state,
  output logic          tile_done,         // Pulse: tile execution complete
  // ★ DW7 multipass: which k pass (0 .. num_k_pass-1) for PSUM namespace / accum
  output logic [3:0]    cur_k_pass_idx
);

  // ═══════════════════════════════════════════════════════════════
  // State register
  // ═══════════════════════════════════════════════════════════════
  tile_state_e cur_state, nxt_state;
  assign state = cur_state;
  assign cur_k_pass_idx = k_pass_cnt;

  // ── Latched descriptor fields ──
  pe_mode_e   mode_reg;
  logic [3:0] k_pass_cnt, k_pass_max;
  logic       is_pass_mode;
  logic       need_swizzle_reg;
  logic       need_writeback_reg;
  logic       barrier_wait_reg;
  logic [3:0] barrier_id_reg;

  // ── Pipeline drain counter (5-stage pipeline) ──
  logic [3:0] drain_cnt;

  // ═══════════════════════════════════════════════════════════════
  // State register
  // ═══════════════════════════════════════════════════════════════
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      cur_state <= TS_IDLE;
    else
      cur_state <= nxt_state;
  end

  // ═══════════════════════════════════════════════════════════════
  // Next-state logic
  // ═══════════════════════════════════════════════════════════════
  always_comb begin
    nxt_state = cur_state;

    case (cur_state)
      TS_IDLE: begin
        if (tile_valid) begin
          // Check barrier BEFORE accepting
          if (tile_desc.barrier_wait && !barrier_grant)
            nxt_state = TS_IDLE;  // Wait for barrier
          else
            nxt_state = TS_LOAD_DESC;
        end
      end

      TS_LOAD_DESC: begin
        // 1 cycle to latch descriptor into shadow regs
        if (is_pass_mode)
          nxt_state = TS_PREFILL_IN;  // PE_PASS: skip weights
        else
          nxt_state = TS_PREFILL_WT;
      end

      TS_PREFILL_WT: begin
        if (dma_done) nxt_state = TS_PREFILL_IN;
      end

      TS_PREFILL_IN: begin
        if (dma_done) begin
          if (is_pass_mode)
            nxt_state = TS_SWIZZLE;  // PE_PASS: skip compute+PPU
          else
            nxt_state = TS_COMPUTE;
        end
      end

      TS_COMPUTE: begin
        if (seq_done) nxt_state = TS_PE_DRAIN;
      end

      TS_PE_DRAIN: begin
        if (drain_cnt >= DSP_PIPE_DEPTH[3:0]) begin
          // ★ PE_MP5: no PPU — max goes straight to ACT during compute (Rule: SPPF)
          if (mode_reg == PE_MP5) begin
            nxt_state = need_swizzle_reg ? TS_SWIZZLE
                       : (need_writeback_reg ? TS_WRITEBACK : TS_DONE);
          end else if (k_pass_cnt < k_pass_max - 4'd1) begin
            nxt_state = TS_COMPUTE;  // DW7x7: next k pass
          end else begin
            nxt_state = TS_PPU_RUN;  // Final pass → PPU
          end
        end
      end

      TS_PPU_RUN: begin
        if (ppu_done) begin
          if (need_swizzle_reg)
            nxt_state = TS_SWIZZLE;
          else if (need_writeback_reg)
            nxt_state = TS_WRITEBACK;
          else
            nxt_state = TS_DONE;
        end
      end

      TS_SWIZZLE: begin
        if (swizzle_done) begin
          if (need_writeback_reg)
            nxt_state = TS_WRITEBACK;
          else
            nxt_state = TS_DONE;
        end
      end

      TS_WRITEBACK: begin
        if (dma_done) nxt_state = TS_DONE;
      end

      TS_DONE: begin
        nxt_state = TS_IDLE;
      end

      default: nxt_state = TS_IDLE;
    endcase
  end

  // ═══════════════════════════════════════════════════════════════
  // Descriptor latch + pass counter
  // ═══════════════════════════════════════════════════════════════
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      mode_reg           <= PE_RS3;
      k_pass_cnt         <= 4'd0;
      k_pass_max         <= 4'd1;
      is_pass_mode       <= 1'b0;
      need_swizzle_reg   <= 1'b0;
      need_writeback_reg <= 1'b0;
      barrier_wait_reg   <= 1'b0;
      barrier_id_reg     <= 4'd0;
      drain_cnt          <= 4'd0;
    end else begin
      case (cur_state)
        TS_IDLE: begin
          if (tile_valid && (nxt_state == TS_LOAD_DESC)) begin
            mode_reg           <= layer_desc.pe_mode;
            k_pass_max         <= (layer_desc.num_k_pass == 4'd0) ? 4'd1 : layer_desc.num_k_pass;
            k_pass_cnt         <= 4'd0;
            is_pass_mode       <= (layer_desc.pe_mode == PE_PASS);
            need_swizzle_reg   <= tile_desc.need_swizzle;
            need_writeback_reg <= tile_desc.last_tile;  // Writeback on last tile
            barrier_wait_reg   <= tile_desc.barrier_wait;
            barrier_id_reg     <= tile_desc.barrier_id;
          end
        end

        TS_PE_DRAIN: begin
          drain_cnt <= drain_cnt + 4'd1;
          // Advance pass counter when draining completes
          if (drain_cnt >= DSP_PIPE_DEPTH[3:0] && k_pass_cnt < k_pass_max - 4'd1)
            k_pass_cnt <= k_pass_cnt + 4'd1;
        end

        TS_COMPUTE: begin
          drain_cnt <= 4'd0;  // Reset drain counter at start of compute
        end

        default: ;
      endcase
    end
  end

  // ═══════════════════════════════════════════════════════════════
  // Output signals (combinational from state)
  // ═══════════════════════════════════════════════════════════════
  always_comb begin
    // Defaults
    tile_accept    = 1'b0;
    shadow_latch   = 1'b0;
    dma_start      = 1'b0;
    dma_is_write   = 1'b0;
    seq_start      = 1'b0;
    ppu_start      = 1'b0;
    swizzle_start  = 1'b0;
    barrier_wait_req = 1'b0;
    barrier_signal = 1'b0;
    page_swap      = 1'b0;
    tile_done      = 1'b0;

    case (cur_state)
      TS_IDLE: begin
        tile_accept      = tile_valid && (nxt_state == TS_LOAD_DESC);
        barrier_wait_req = tile_valid && tile_desc.barrier_wait;
      end

      TS_LOAD_DESC: begin
        shadow_latch = 1'b1;  // Latch descriptor → shadow_reg_file
        page_swap    = 1'b1;  // Swap double-buffer page
      end

      TS_PREFILL_WT: begin
        dma_start    = 1'b1;
        dma_is_write = 1'b0;  // Read DDR → GLB weight
      end

      TS_PREFILL_IN: begin
        dma_start    = 1'b1;
        dma_is_write = 1'b0;  // Read DDR → GLB input
      end

      TS_COMPUTE: begin
        seq_start    = 1'b1;  // Start compute_sequencer
      end

      TS_PPU_RUN: begin
        ppu_start    = 1'b1;  // Trigger PPU
      end

      TS_SWIZZLE: begin
        swizzle_start = 1'b1; // Start swizzle engine
      end

      TS_WRITEBACK: begin
        dma_start    = 1'b1;
        dma_is_write = 1'b1;  // Write GLB → DDR
      end

      TS_DONE: begin
        tile_done      = 1'b1;
        barrier_signal = 1'b1;  // Signal barrier completion
      end

      default: ;
    endcase
  end

endmodule
