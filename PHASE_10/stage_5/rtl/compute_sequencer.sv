`timescale 1ns/1ps
// ============================================================================
// COMPUTE SEQUENCER — Inner-loop address & control generator
//
// tile_fsm quyết định KHI NÀO compute (phase-level).
// compute_sequencer quyết định TÍNH GÌ ở mỗi cycle (cycle-level).
//
// Generates the (h_out, wblk, cout, cin, kh, kw) iteration sequence and
// drives addr_gen_input, addr_gen_weight, addr_gen_output, PE enable/clear,
// PPU trigger, and output write signals.
//
// Supports ALL pe_modes through a unified loop structure:
//   PE_RS3: for h_out, wblk, cout: for cin: for kw=0..2 → PE(3 rows) → PPU
//   PE_OS1: for h_out, wblk, cout: for cin: 1 cycle    → PE(1 unit) → PPU
//   PE_DW3: for h_out, wblk:       for c: for kw=0..2  → PE(3 rows) → PPU
//   PE_DW7: for h_out, wblk:       for c: 3 passes(kh) × 7 kw → PSUM → PPU
//   PE_MP5: for h_out, wblk:       for c: 1 cycle      → comparator → output
//   PE_PASS: swizzle_engine handles (no compute loop)
// ============================================================================
module compute_sequencer
  import accel_pkg::*;
  import desc_pkg::*;
#(
  parameter int LANES   = 32,
  parameter int PE_ROWS = 3,
  parameter int PE_COLS = 4
)(
  input  logic        clk,
  input  logic        rst_n,

  // ═══════════ Control from tile_fsm ═══════════
  input  logic        start,          // Pulse: begin compute sequence
  input  tile_state_e fsm_state,      // Current tile FSM state
  output logic        compute_done,   // All spatial/channel iterations complete
  output logic        ppu_done,       // PPU output written for current tile

  // ═══════════ Configuration from shadow_reg_file ═══════════
  input  pe_mode_e    cfg_mode,
  input  logic [8:0]  cfg_cin_tile,   // Input channels for this tile
  input  logic [8:0]  cfg_cout_tile,  // Output channels for this tile
  input  logic [9:0]  cfg_hout,       // Output height
  input  logic [9:0]  cfg_wout,       // Output width
  input  logic [3:0]  cfg_kh, cfg_kw, // Kernel size
  input  logic [2:0]  cfg_sh, cfg_sw, // Stride
  input  logic [3:0]  cfg_pad_top, cfg_pad_left,
  input  logic [9:0]  cfg_hin, cfg_win,
  input  logic [3:0]  cfg_q_in, cfg_q_out,
  input  logic [3:0]  cfg_num_cin_pass, cfg_num_k_pass,
  input  logic signed [7:0] cfg_zp_x, // Zero-point for padding

  // ═══════════ Address Generator: Input ═══════════
  output logic              agi_req_valid,
  output logic [9:0]        agi_req_h,
  output logic [9:0]        agi_req_w,
  output logic [8:0]        agi_req_c,

  // ═══════════ Address Generator: Weight ═══════════
  output logic              agw_req_valid,
  output logic [2:0]        agw_req_kr,
  output logic [8:0]        agw_req_cin,
  output logic [8:0]        agw_req_cout,
  output logic [2:0]        agw_req_kw_idx,

  // ═══════════ Address Generator: Output ═══════════
  output logic              ago_req_valid,
  output logic [9:0]        ago_req_h_out,
  output logic [9:0]        ago_req_w_out,
  output logic [8:0]        ago_req_cout,
  output logic [1:0]        ago_req_pe_col,

  // ═══════════ PE Cluster Control ═══════════
  output logic              pe_en,
  output logic              pe_clear,

  // ═══════════ PPU Control ═══════════
  output logic              ppu_trigger,    // Start PPU pipeline
  output logic [8:0]        ppu_cout_idx,   // Current cout for bias/quant select

  // ═══════════ Output Write Control ═══════════
  output logic              out_wr_trigger, // Write PPU result to output bank
  output logic [1:0]        out_wr_bank,
  output logic [15:0]       out_wr_addr,
  output logic              out_wr_is_pool, // MaxPool output (bypass PPU)

  // ═══════════ Pool Control ═══════════
  output logic              pool_en
);

  // ═══════════════════════════════════════════════════════════════════
  // SEQUENCER FSM
  // ═══════════════════════════════════════════════════════════════════
  typedef enum logic [3:0] {
    SEQ_IDLE,
    SEQ_INIT,           // Latch config, compute derived values
    SEQ_FEED_PE,        // Feed activation + weight to PE each cycle
    SEQ_PE_DRAIN,       // Wait PE pipeline latency (4 cycles)
    SEQ_PPU_RUN,        // PPU processes psum → INT8 (4 cycles)
    SEQ_WRITE_OUT,      // Write result to output bank
    SEQ_NEXT_KW,        // Advance kw counter
    SEQ_NEXT_CIN,       // Advance cin counter
    SEQ_NEXT_COUT,      // Advance cout counter (dense conv only)
    SEQ_NEXT_WBLK,      // Advance wblk counter
    SEQ_NEXT_HOUT,      // Advance h_out counter
    SEQ_POOL_FEED,      // Feed maxpool window
    SEQ_POOL_DRAIN,     // Wait comparator pipeline (5 cycles)
    SEQ_DONE
  } seq_state_e;

  seq_state_e seq_state;

  // ─────── Loop Counters ───────
  logic [9:0] cnt_h_out;
  logic [5:0] cnt_wblk;
  logic [8:0] cnt_cout;
  logic [8:0] cnt_cin;
  logic [2:0] cnt_kw;
  logic [2:0] cnt_kh_pass;  // For DW7 multipass: 0,1,2 = pass 1,2,3
  logic [3:0] drain_cnt;

  // ─────── Derived Values (latched at INIT) ───────
  logic [5:0] num_wblk_out;   // ceil(wout / LANES)
  logic [2:0] kw_max;         // cfg_kw - 1
  logic       is_depthwise;   // DW3 or DW7
  logic       is_pool;        // MP5
  logic       is_pass;        // PE_PASS (upsample/concat)

  // ─────── Compute current input coordinates ───────
  // For conv: h_in = h_out * stride + kh_offset + current_kh_row
  // w_base = wblk * LANES * stride + kw_offset
  logic [9:0] h_in_base;
  logic [9:0] w_in_base;

  always_comb begin
    h_in_base = cnt_h_out * cfg_sh;
    w_in_base = 10'(cnt_wblk) * LANES[9:0] * cfg_sw;
  end

  // ═══════════════════════════════════════════════════════════════════
  // MAIN SEQUENCER FSM
  // ═══════════════════════════════════════════════════════════════════
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      seq_state     <= SEQ_IDLE;
      compute_done  <= 1'b0;
      ppu_done      <= 1'b0;
      cnt_h_out     <= '0;
      cnt_wblk      <= '0;
      cnt_cout      <= '0;
      cnt_cin       <= '0;
      cnt_kw        <= '0;
      cnt_kh_pass   <= '0;
      drain_cnt     <= '0;
      num_wblk_out  <= '0;
      kw_max        <= '0;
      is_depthwise  <= 1'b0;
      is_pool       <= 1'b0;
      is_pass       <= 1'b0;
    end else begin

      // Default: deassert single-cycle pulses
      compute_done  <= 1'b0;
      ppu_done      <= 1'b0;

      case (seq_state)
        // ─────────────────────────────────────────────
        SEQ_IDLE: begin
          if (start && fsm_state == TILE_RUN_COMPUTE) begin
            seq_state <= SEQ_INIT;
          end
        end

        // ─────────────────────────────────────────────
        SEQ_INIT: begin
          // Latch derived values
          num_wblk_out <= (cfg_wout + LANES - 1) / LANES;
          kw_max       <= cfg_kw[2:0] - 3'd1;
          is_depthwise <= (cfg_mode == PE_DW3) || (cfg_mode == PE_DW7);
          is_pool      <= (cfg_mode == PE_MP5);
          is_pass      <= (cfg_mode == PE_PASS);

          // Reset all counters
          cnt_h_out   <= '0;
          cnt_wblk    <= '0;
          cnt_cout    <= '0;
          cnt_cin     <= '0;
          cnt_kw      <= '0;
          cnt_kh_pass <= '0;

          if (cfg_mode == PE_PASS) begin
            // PE_PASS: no compute, signal done immediately
            seq_state    <= SEQ_DONE;
          end else if (cfg_mode == PE_MP5) begin
            seq_state    <= SEQ_POOL_FEED;
          end else begin
            seq_state    <= SEQ_FEED_PE;
          end
        end

        // ─────────────────────────────────────────────
        // FEED_PE: Drive addr_gen + PE for one (kw, cin) iteration
        // PE accumulates internally across kw and cin
        // ─────────────────────────────────────────────
        SEQ_FEED_PE: begin
          // PE is fed this cycle (outputs computed combinationally by addr_gen + router)
          // Advance to next kw
          if (cnt_kw == kw_max) begin
            cnt_kw <= '0;
            // Done with all kw for this cin → next cin
            seq_state <= SEQ_NEXT_CIN;
          end else begin
            cnt_kw <= cnt_kw + 1;
          end
        end

        // ─────────────────────────────────────────────
        SEQ_NEXT_CIN: begin
          if (cnt_cin == cfg_cin_tile - 1) begin
            cnt_cin <= '0;
            // All cin done → drain PE pipeline then PPU
            seq_state <= SEQ_PE_DRAIN;
            drain_cnt <= '0;
          end else begin
            cnt_cin   <= cnt_cin + 1;
            cnt_kw    <= '0;
            seq_state <= SEQ_FEED_PE;
          end
        end

        // ─────────────────────────────────────────────
        SEQ_PE_DRAIN: begin
          // Wait 4 cycles for PE pipeline to flush
          drain_cnt <= drain_cnt + 1;
          if (drain_cnt == 4'd4) begin
            seq_state <= SEQ_PPU_RUN;
            drain_cnt <= '0;
          end
        end

        // ─────────────────────────────────────────────
        SEQ_PPU_RUN: begin
          // PPU takes 4 cycles (pipeline)
          drain_cnt <= drain_cnt + 1;
          if (drain_cnt == 4'd4) begin
            seq_state <= SEQ_WRITE_OUT;
          end
        end

        // ─────────────────────────────────────────────
        SEQ_WRITE_OUT: begin
          // Write PPU output to output bank
          // Then advance to next output position
          if (is_depthwise) begin
            // DW: no cout loop, advance wblk
            seq_state <= SEQ_NEXT_WBLK;
          end else begin
            // Dense: advance cout
            seq_state <= SEQ_NEXT_COUT;
          end
        end

        // ─────────────────────────────────────────────
        SEQ_NEXT_COUT: begin
          if (cnt_cout == cfg_cout_tile - 1) begin
            cnt_cout  <= '0;
            seq_state <= SEQ_NEXT_WBLK;
          end else begin
            cnt_cout  <= cnt_cout + 1;
            cnt_cin   <= '0;
            cnt_kw    <= '0;
            seq_state <= SEQ_FEED_PE;
          end
        end

        // ─────────────────────────────────────────────
        SEQ_NEXT_WBLK: begin
          if (cnt_wblk == num_wblk_out - 1) begin
            cnt_wblk  <= '0;
            seq_state <= SEQ_NEXT_HOUT;
          end else begin
            cnt_wblk  <= cnt_wblk + 1;
            cnt_cout  <= '0;
            cnt_cin   <= '0;
            cnt_kw    <= '0;
            seq_state <= is_pool ? SEQ_POOL_FEED : SEQ_FEED_PE;
          end
        end

        // ─────────────────────────────────────────────
        SEQ_NEXT_HOUT: begin
          if (cnt_h_out == cfg_hout - 1) begin
            // ALL output positions computed
            seq_state <= SEQ_DONE;
          end else begin
            cnt_h_out <= cnt_h_out + 1;
            cnt_wblk  <= '0;
            cnt_cout  <= '0;
            cnt_cin   <= '0;
            cnt_kw    <= '0;
            seq_state <= is_pool ? SEQ_POOL_FEED : SEQ_FEED_PE;
          end
        end

        // ─────────────────────────────────────────────
        // MAXPOOL path: feed 25 values from window, wait comparator
        // ─────────────────────────────────────────────
        SEQ_POOL_FEED: begin
          // Comparator tree needs 25 values loaded → 1 cycle feed
          // (window_gen should have 5 rows ready)
          seq_state <= SEQ_POOL_DRAIN;
          drain_cnt <= '0;
        end

        SEQ_POOL_DRAIN: begin
          drain_cnt <= drain_cnt + 1;
          if (drain_cnt == 4'd5) begin  // 5-stage comparator pipeline
            seq_state <= SEQ_WRITE_OUT;
          end
        end

        // ─────────────────────────────────────────────
        SEQ_DONE: begin
          compute_done <= 1'b1;
          ppu_done     <= 1'b1;
          seq_state    <= SEQ_IDLE;
        end

        default: seq_state <= SEQ_IDLE;
      endcase
    end
  end

  // ═══════════════════════════════════════════════════════════════════
  // OUTPUT SIGNALS — Drive addr_gen and PE controls
  // ═══════════════════════════════════════════════════════════════════

  wire feeding = (seq_state == SEQ_FEED_PE);

  // ── Address Generator: Input ──
  always_comb begin
    agi_req_valid = feeding;
    // For each PE row r: input row = h_out*stride + r (for RS3/DW3)
    // We request row 0 here; router handles row selection
    agi_req_h = h_in_base + {6'b0, cnt_kw};  // Simplified: kw as h offset
    agi_req_w = w_in_base;
    agi_req_c = is_depthwise ? cnt_cin : cnt_cin;  // Same for both
  end

  // ── Address Generator: Weight ──
  always_comb begin
    agw_req_valid  = feeding;
    agw_req_kr     = '0;  // Router selects per-row weight
    agw_req_cin    = cnt_cin;
    agw_req_cout   = is_depthwise ? cnt_cin : cnt_cout;  // DW: cout=cin
    agw_req_kw_idx = cnt_kw;
  end

  // ── Address Generator: Output ──
  assign ago_req_valid = (seq_state == SEQ_WRITE_OUT);
  assign ago_req_h_out = cnt_h_out;
  assign ago_req_w_out = {4'b0, cnt_wblk} * LANES;
  assign ago_req_cout  = is_depthwise ? cnt_cin : cnt_cout;
  assign ago_req_pe_col = 2'd0;  // Simplified: use column 0

  // ── PE Control ──
  assign pe_en    = feeding;
  assign pe_clear = feeding && (cnt_cin == 0) && (cnt_kw == 0);

  // ── PPU Control ──
  assign ppu_trigger = (seq_state == SEQ_PPU_RUN) && (drain_cnt == 0);
  assign ppu_cout_idx = is_depthwise ? cnt_cin : cnt_cout;

  // ── Output Write ──
  assign out_wr_trigger = (seq_state == SEQ_WRITE_OUT);
  assign out_wr_bank    = 2'd0;  // Simplified
  assign out_wr_addr    = '0;    // Driven by addr_gen_output
  assign out_wr_is_pool = is_pool;

  // ── Pool Control ──
  assign pool_en = (seq_state == SEQ_POOL_FEED);

endmodule
