// ============================================================================
// Module : compute_sequencer
// Project: YOLOv10n INT8 Accelerator — V4-VC707
//
// Description:
//   Cycle-level iteration controller: "CÁI GÌ mỗi cycle".
//   Generates the (h_out, wblk, cout_group, cin, kw) loop counters
//   and drives addr_gens + PE enable/clear + PPU trigger.
//
//   ★ CORE V4: cout_group increments by 4 (PE_COLS) → 4 columns parallel
//   ★ CORE V4: LANES=20 → W_out / 20 = exact integer for ALL YOLOv10n layers!
//
//   Loop structures:
//     PE_RS3 (Conv 3×3):
//       for h_out:
//         for wblk = 0 .. Wout/20 - 1:
//           for cout_group = 0 .. Cout/4 - 1:     ★ 4 columns
//             pe_clear_acc
//             for cin = 0 .. Cin - 1:
//               for kw = 0 .. 2:
//                 pe_enable
//             ppu_trigger
//
//     PE_OS1 (Conv 1×1):
//       Same but no kw loop (kw always 0)
//
//     PE_DW3 (DW 3×3):
//       for h_out, wblk:
//         for ch_group = 0 .. C/4 - 1:             ★ 4 channels
//           pe_clear_acc
//           for kw = 0 .. 2:
//             pe_enable
//           ppu_trigger
//
//     PE_MP5 (MaxPool 5×5):
//       AGI+GLB read latency → window_gen shift uses mp5_shift_en delayed 1 cyc (subcluster).
//       sub 0 flush; 1..5 fetch kh=0..4 (agi_iter_kh_mux = sub-1); 6 bubble; 7 pool; 8..13 bubble;
//       14 ch adv → iter_mp5_ch++, sub=15 (cin settle); 15→0 next ch.
//
//     PE_PASS: seq_done immediately (no compute)
// ============================================================================
`timescale 1ns / 1ps

module compute_sequencer
  import accel_pkg::*;
(
  input  logic          clk,
  input  logic          rst_n,

  // ── Control (from tile_fsm) ──
  input  logic          seq_start,         // Pulse: begin compute sequence
  output logic          seq_done,          // Pulse: all iterations complete

  // ── Configuration (from shadow_reg_file) ──
  input  pe_mode_e      cfg_pe_mode,
  input  logic [9:0]    cfg_cin,
  input  logic [9:0]    cfg_cout,
  input  logic [9:0]    cfg_hout,
  input  logic [9:0]    cfg_wout,
  input  logic [3:0]    cfg_kh,
  input  logic [3:0]    cfg_kw,
  input  logic [2:0]    cfg_stride,

  // ── Iteration outputs (drive addr_gens) ──
  output logic [9:0]    iter_h,            // Current output row
  output logic [9:0]    iter_wblk,         // Current width block (0 .. Wout/LANES-1)
  output logic [9:0]    iter_cin,          // Current input channel
  output logic [9:0]    iter_cout_group,   // Current cout group (×4 for PE_COLS)
  output logic [3:0]    iter_kw,           // Current kernel column (0..cfg_kw-1)
  output logic [3:0]    iter_kh_row,       // Current kernel row (for addr_gen bank sel)

  // ── PE control ──
  output logic          pe_enable,         // Feed data to PE this cycle
  output logic          pe_clear_acc,      // Clear PE accumulators (start of new output)
  output logic          pe_acc_valid,      // PE accumulation complete (ready for PPU)

  // ── PPU control ──
  output logic          ppu_trigger,       // Trigger PPU for current cout_group
  output logic [9:0]    ppu_cout_base,     // cout_base = cout_group × 4

  // ── MaxPool control ──
  output logic          pool_enable,       // 1-cycle pulse → comparator_tree
  output logic          mp5_shift_en,      // Shift window_gen during MP5 row scan
  output logic          mp5_win_flush,     // 1-cycle: clear line-buffer before new (h,w,ch0)
  // ── Address mux helpers (MP5 uses per-channel cin × 5 rows) ──
  output logic [9:0]    agi_iter_cin_mux,
  output logic [3:0]    agi_iter_kh_mux,
  output logic [9:0]    ago_iter_cout_grp_mux,
  output logic [9:0]    dbg_iter_mp5_ch,      // current MP5 output channel (for GLB ACT write sel)

  // Window flush: fires during SEQ_CLEAR (prefetch cycle, pe_enable=0)
  // so that window_gen can reset taps without colliding with shift_in_valid.
  output logic          seq_window_flush
);

  // ═══════════════════════════════════════════════════════════════
  // Sequencer FSM
  // ═══════════════════════════════════════════════════════════════
  typedef enum logic [3:0] {
    SEQ_IDLE,
    SEQ_INIT,          // Compute derived values (1 cycle)
    SEQ_CLEAR,         // Assert pe_clear_acc (1 cycle)
    SEQ_FEED,          // pe_enable = 1 (kw × cin iterations)
    SEQ_NEXT_KW,       // Advance kw
    SEQ_NEXT_CIN,      // Advance cin
    SEQ_ACC_DONE,      // Accumulation complete → trigger PPU
    SEQ_NEXT_COUT,     // Advance cout_group (dense) or ch_group (DW)
    SEQ_NEXT_WBLK,     // Advance wblk
    SEQ_NEXT_H,        // Advance h_out
    SEQ_POOL,          // MaxPool: 1 cycle per ch_group
    SEQ_DONE,
    SEQ_ADDR_SETTLE,       // 1-cycle prefetch after kw change (weight addr comb.; input may need +1)
    SEQ_CIN_PRE_SETTLE    // OS1/GEMM: addr_gen_input is registered → extra cycle before ADDR_SETTLE
  } seq_state_e;

  seq_state_e ss;

  // ── Derived values (latched at INIT) ──
  logic [9:0] num_wblk;       // Wout / LANES (exact division for V4!)
  logic [9:0] num_cout_grp;   // Cout / PE_COLS
  logic [3:0] kw_max;         // cfg_kw - 1

  logic [3:0] mp_sub;
  logic [9:0] iter_mp5_ch;

  // FIX: first_feed flag ensures pe_clear_acc is asserted during the first
  // SEQ_FEED cycle (not during SEQ_CLEAR where pe_enable=0). The DSP pipeline
  // requires en_s4 && clear_s4 simultaneously at stage 4 to reset the accumulator.
  logic first_feed;

  localparam int MP5_SUB_FIRE      = 4'd7;
  localparam int MP5_SUB_BUBBLE_HI = 4'd13;
  localparam int MP5_SUB_CH_ADV    = 4'd14;
  // After CH_ADV, bump iter_mp5_ch then hold one cycle so addr_gen_input samples
  // the new cin *before* mp_sub returns to 0 (flush). Same-edge {ch++,sub<=0}
  // leaves AGI registered addr one cycle behind (cin still old) — breaks ch1..3 MP5.
  localparam int MP5_SUB_CIN_SETTLE = 4'd15;

  // ═══════════════════════════════════════════════════════════════
  // Main FSM
  // ═══════════════════════════════════════════════════════════════
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ss              <= SEQ_IDLE;
      seq_done        <= 1'b0;
      iter_h          <= 10'd0;
      iter_wblk       <= 10'd0;
      iter_cin        <= 10'd0;
      iter_cout_group <= 10'd0;
      iter_kw         <= 4'd0;
      iter_kh_row     <= 4'd0;
      num_wblk        <= 10'd1;
      num_cout_grp    <= 10'd1;
      kw_max          <= 4'd0;
      mp_sub          <= 4'd0;
      iter_mp5_ch     <= 10'd0;
      first_feed      <= 1'b0;
    end else begin
      // Default: single-cycle pulses off
      seq_done <= 1'b0;

      case (ss)
        // ────────────────────────────────────────
        SEQ_IDLE: begin
          if (seq_start) begin
            if (cfg_pe_mode == PE_PASS) begin
              // PE_PASS: no compute needed
              ss       <= SEQ_DONE;
            end else begin
              ss       <= SEQ_INIT;
            end
          end
        end

        // ────────────────────────────────────────
        SEQ_INIT: begin
          // ★ LANES=20 → exact division for all YOLOv10n widths
          num_wblk     <= cfg_wout / LANES[9:0];
          num_cout_grp <= (cfg_pe_mode == PE_DW3 || cfg_pe_mode == PE_DW7)
                        ? cfg_cin / PE_COLS[9:0]    // DW: channel groups
                        : cfg_cout / PE_COLS[9:0];  // Dense: cout groups
          kw_max       <= cfg_kw - 4'd1;

          // Reset all counters
          iter_h          <= 10'd0;
          iter_wblk       <= 10'd0;
          iter_cin        <= 10'd0;
          iter_cout_group <= 10'd0;
          iter_kw         <= 4'd0;
          iter_kh_row     <= 4'd0;
          mp_sub          <= 4'd0;
          iter_mp5_ch     <= 10'd0;
          first_feed      <= 1'b0;

          if (cfg_pe_mode == PE_MP5)
            ss <= SEQ_POOL;
          else
            ss <= SEQ_CLEAR;
        end

        // ────────────────────────────────────────
        SEQ_CLEAR: begin
          // Prefetch cycle: memory read starts. Mark first_feed so that the
          // next SEQ_FEED asserts pe_clear_acc alongside pe_enable.
          first_feed <= 1'b1;
          ss <= SEQ_FEED;
        end

        // ────────────────────────────────────────
        SEQ_FEED: begin
          // pe_enable active → data feeds into PE this cycle.
          // If first_feed, pe_clear_acc is also asserted (see output logic).
          first_feed <= 1'b0;
          ss <= SEQ_NEXT_KW;
        end

        // ────────────────────────────────────────
        SEQ_NEXT_KW: begin
          if (cfg_pe_mode == PE_OS1 || cfg_pe_mode == PE_GEMM) begin
            ss <= SEQ_NEXT_CIN;
          end else if (iter_kw >= kw_max) begin
            iter_kw <= 4'd0;
            ss      <= SEQ_NEXT_CIN;
          end else begin
            iter_kw <= iter_kw + 4'd1;
            ss      <= SEQ_ADDR_SETTLE;
          end
        end

        // ────────────────────────────────────────
        SEQ_ADDR_SETTLE: begin
          ss <= SEQ_FEED;
        end

        // ────────────────────────────────────────
        SEQ_NEXT_CIN: begin
          if (cfg_pe_mode == PE_DW3 || cfg_pe_mode == PE_DW7) begin
            ss <= SEQ_ACC_DONE;
          end else if (iter_cin >= cfg_cin - 10'd1) begin
            iter_cin <= 10'd0;
            ss       <= SEQ_ACC_DONE;
          end else begin
            iter_cin <= iter_cin + 10'd1;
            iter_kw  <= 4'd0;
            if (cfg_pe_mode == PE_OS1 || cfg_pe_mode == PE_GEMM)
              ss <= SEQ_CIN_PRE_SETTLE;
            else
              ss <= SEQ_ADDR_SETTLE;
          end
        end

        // ────────────────────────────────────────
        SEQ_CIN_PRE_SETTLE: begin
          ss <= SEQ_ADDR_SETTLE;
        end

        // ────────────────────────────────────────
        SEQ_ACC_DONE: begin
          // Accumulation complete for this cout_group → PPU
          ss <= SEQ_NEXT_COUT;
        end

        // ────────────────────────────────────────
        SEQ_NEXT_COUT: begin
          if (iter_cout_group >= num_cout_grp - 10'd1) begin
            iter_cout_group <= 10'd0;
            ss              <= SEQ_NEXT_WBLK;
          end else begin
            iter_cout_group <= iter_cout_group + 10'd1;
            iter_cin        <= 10'd0;
            iter_kw         <= 4'd0;
            ss              <= SEQ_CLEAR;  // New cout_group → clear + restart
          end
        end

        // ────────────────────────────────────────
        SEQ_NEXT_WBLK: begin
          if (iter_wblk >= num_wblk - 10'd1) begin
            iter_wblk <= 10'd0;
            ss        <= SEQ_NEXT_H;
          end else begin
            iter_wblk       <= iter_wblk + 10'd1;
            iter_cout_group <= 10'd0;
            iter_cin        <= 10'd0;
            iter_kw         <= 4'd0;
            if (cfg_pe_mode == PE_MP5) begin
              iter_mp5_ch <= 10'd0;
              mp_sub      <= 4'd0;
              ss          <= SEQ_POOL;
            end else
              ss <= SEQ_CLEAR;
          end
        end

        // ────────────────────────────────────────
        SEQ_NEXT_H: begin
          if (iter_h >= cfg_hout - 10'd1) begin
            ss <= SEQ_DONE;  // All spatial positions computed
          end else begin
            iter_h          <= iter_h + 10'd1;
            iter_wblk       <= 10'd0;
            iter_cout_group <= 10'd0;
            iter_cin        <= 10'd0;
            iter_kw         <= 4'd0;
            if (cfg_pe_mode == PE_MP5) begin
              iter_mp5_ch <= 10'd0;
              mp_sub      <= 4'd0;
              ss          <= SEQ_POOL;
            end else
              ss <= SEQ_CLEAR;
          end
        end

        // ────────────────────────────────────────
        SEQ_POOL: begin
          // ★ MP5: structured micro-sequence per channel @ (h, wblk)
          if (mp_sub == MP5_SUB_CIN_SETTLE) begin
            mp_sub <= 4'd0;
          end else if (mp_sub == 4'd0) begin
            mp_sub <= 4'd1;
          end else if (mp_sub < MP5_SUB_FIRE) begin
            mp_sub <= mp_sub + 4'd1;
          end else if (mp_sub == MP5_SUB_FIRE) begin
            mp_sub <= mp_sub + 4'd1;
          end else if (mp_sub < MP5_SUB_CH_ADV) begin
            mp_sub <= mp_sub + 4'd1;
          end else begin
            // mp_sub == 13 → finished one channel @ this spatial location
            if (iter_mp5_ch >= cfg_cout - 10'd1) begin
              iter_mp5_ch <= 10'd0;
              mp_sub      <= 4'd0;
              ss          <= SEQ_NEXT_WBLK;
            end else begin
              iter_mp5_ch <= iter_mp5_ch + 10'd1;
              mp_sub      <= MP5_SUB_CIN_SETTLE;
            end
          end
        end

        // ────────────────────────────────────────
        SEQ_DONE: begin
          seq_done <= 1'b1;
          ss       <= SEQ_IDLE;
        end

        default: ss <= SEQ_IDLE;
      endcase
    end
  end

  // ═══════════════════════════════════════════════════════════════
  // Output signals (combinational from state + counters)
  // ═══════════════════════════════════════════════════════════════

  // PE control — FIX: pe_clear_acc now fires during SEQ_FEED (first cycle of
  // each cout_group) so that en and clear enter the DSP pipeline simultaneously.
  assign pe_enable    = (ss == SEQ_FEED);
  assign pe_clear_acc = (ss == SEQ_FEED) & first_feed;
  assign pe_acc_valid = (ss == SEQ_ACC_DONE);

  // PPU trigger: fires at SEQ_ACC_DONE (once per cout_group completion)
  assign ppu_trigger  = (ss == SEQ_ACC_DONE);

  // PPU cout base: which 4 output channels are being processed
  // ★ 4 PE columns → cout_base = cout_group × 4
  assign ppu_cout_base = iter_cout_group * PE_COLS[9:0];

  assign pool_enable    = (ss == SEQ_POOL) && (mp_sub == MP5_SUB_FIRE);
  assign mp5_shift_en   = (ss == SEQ_POOL) && (mp_sub > 4'd0) && (mp_sub < 4'd6);
  assign mp5_win_flush  = (ss == SEQ_POOL) && (mp_sub == 4'd0);

  // Addr mux: mp_sub 1..5 are the five window row shifts after flush; each shift must
  // fetch iter_kh_row = 0..4 (ho + kh - pad), not 1..4 — first shift used kh=1 and
  // dropped the true top row of the 5×5 stencil (golden hi = ho + kh - pad).
  always_comb begin
    if (cfg_pe_mode == PE_MP5 && ss == SEQ_POOL) begin
      agi_iter_cin_mux = iter_mp5_ch;
      if (mp_sub == 4'd0)
        agi_iter_kh_mux = 4'd0;
      else if (mp_sub <= 4'd5)
        agi_iter_kh_mux = mp_sub - 4'd1;
      else
        agi_iter_kh_mux = iter_kh_row;
    end else begin
      agi_iter_cin_mux = iter_cin;
      agi_iter_kh_mux    = iter_kh_row;
    end
  end

  always_comb begin
    if (cfg_pe_mode == PE_MP5 && ss == SEQ_POOL)
      ago_iter_cout_grp_mux = iter_mp5_ch >> 2;
    else
      ago_iter_cout_grp_mux = iter_cout_group;
  end

  assign dbg_iter_mp5_ch = iter_mp5_ch;

  assign seq_window_flush = (ss == SEQ_CLEAR);

  // synthesis translate_off
`ifdef S8_DBG
  `define SEQ_DBG_PREV 1
`endif
`ifdef RTL_TRACE
  `define SEQ_DBG_PREV 1
`endif
`ifdef SEQ_DBG_PREV
  seq_state_e ss_prev;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) ss_prev <= SEQ_IDLE;
    else        ss_prev <= ss;
  end
`endif
`ifdef S8_DBG
  always @(posedge clk) begin
    if (rst_n && ss != ss_prev)
      $display("  [SEQ] %0t ss=%0d->%0d h=%0d wblk=%0d cout_grp=%0d cin=%0d kw=%0d pe_en=%b clr=%b ppu_trig=%b",
               $time, ss_prev, ss, iter_h, iter_wblk, iter_cout_group, iter_cin, iter_kw,
               pe_enable, pe_clear_acc, ppu_trigger);
  end
`endif
`ifdef RTL_TRACE
  always @(posedge clk) begin
    if (rst_n && ss != ss_prev)
      rtl_trace_pkg::rtl_trace_line("S6_SEQ",
        $sformatf("ss=%0d->%0d h=%0d wb=%0d cg=%0d ci=%0d kw=%0d pe=%b clr=%b ppu=%b pool=%b",
                  ss_prev, ss, iter_h, iter_wblk, iter_cout_group, iter_cin, iter_kw,
                  pe_enable, pe_clear_acc, ppu_trigger, pool_enable));
  end
`endif
`ifdef SEQ_DBG_PREV
  `undef SEQ_DBG_PREV
`endif
  // synthesis translate_on

endmodule
