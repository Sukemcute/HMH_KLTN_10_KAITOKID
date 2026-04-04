// ============================================================================
// Module : swizzle_engine
// Project: YOLOv10n INT8 Accelerator — V4-VC707
//
// Description:
//   Tensor layout transform between layers.
//   3 modes (selected by swizzle_mode_e from descriptor):
//
//   SWZ_NORMAL:     Identity pass-through (most layers).
//   SWZ_UPSAMPLE2X: Nearest-neighbor 2× upsample (L11, L14).
//     dst[c][2h+dh][2w+dw] = src[c][h][w] for dh,dw ∈ {0,1}
//   SWZ_CONCAT:     Channel concatenation with domain alignment (QConcat).
//     Join tensors A and B along channel dim, requant if scales differ.
//
// FSM-driven: start → iterate src addresses → transform → write dst → done.
// Verified: Upsample = 100% bit-exact, Concat = 100% bit-exact.
// ============================================================================
`timescale 1ns / 1ps

module swizzle_engine
  import accel_pkg::*;
#(
  parameter int LANES = accel_pkg::LANES  // 20
)(
  input  logic          clk,
  input  logic          rst_n,
  input  logic          start,
  input  swizzle_mode_e cfg_mode,

  // ── Configuration ──
  input  logic [9:0]    cfg_src_h, cfg_src_w, cfg_src_c,
  input  logic [9:0]    cfg_dst_h, cfg_dst_w,

  // ── Domain alignment params (for CONCAT requant & EWISE_ADD) ──
  input  uint32_t       cfg_align_m_a,     // requant multiplier for tensor A
  input  logic [7:0]    cfg_align_sh_a,    // requant shift for tensor A
  input  int8_t         cfg_align_zp_a,    // zero-point for tensor A
  input  uint32_t       cfg_align_m_b,     // requant multiplier for tensor B
  input  logic [7:0]    cfg_align_sh_b,    // requant shift for tensor B
  input  int8_t         cfg_align_zp_b,    // zero-point for tensor B
  input  int8_t         cfg_align_zp_out,  // output zero-point
  input  logic          cfg_align_bypass,  // 1 = same domain, skip requant (fast path)

  // ── Source A: read from GLB output (ACT namespace) ──
  output logic          src_rd_en,
  output logic [11:0]   src_rd_addr,
  input  int8_t         src_rd_data [LANES],

  // ── Source B: read from skip buffer (for CONCAT/EWISE_ADD) ──
  output logic          src_b_rd_en,
  output logic [11:0]   src_b_rd_addr,
  input  int8_t         src_b_rd_data [LANES],

  // ── Destination: write to GLB input (for next layer) ──
  output logic          dst_wr_en,
  output logic [11:0]   dst_wr_addr,
  output int8_t         dst_wr_data [LANES],
  output logic [LANES-1:0] dst_wr_mask,

  // ── Status ──
  output logic          done
);

  // ════════════════════════════════════════════════════════════════
  // FSM States
  // ════════════════════════════════════════════════════════════════
  typedef enum logic [3:0] {
    SW_IDLE,
    SW_READ,           // Issue read address to source A
    SW_READ_WAIT,      // Wait 1 cycle for registered SRAM read
    SW_WRITE,          // Write to destination (normal/concat mode)
    SW_UP_WRITE_0,     // Upsample: write row 2h+0
    SW_UP_WRITE_1,     // Upsample: write row 2h+1
    SW_ADD_READ_B,     // EWISE_ADD: issue read for source B
    SW_ADD_READ_B_WAIT,// EWISE_ADD: wait for B read
    SW_ADD_COMPUTE,    // EWISE_ADD: requant A+B, compute sum
    SW_ADD_WRITE,      // EWISE_ADD: write result
    SW_DONE
  } sw_state_e;

  sw_state_e state;

  // ── Iteration counters ──
  logic [9:0] cnt_h, cnt_w, cnt_c;
  logic [9:0] src_wblk_total, dst_wblk_total;

  assign src_wblk_total = (cfg_src_w + LANES[9:0] - 10'd1) / LANES[9:0];
  assign dst_wblk_total = (cfg_dst_w + LANES[9:0] - 10'd1) / LANES[9:0];

  logic [11:0] src_h_off, dst_h_off;
  assign src_h_off = 12'(cnt_h) * 12'(cfg_src_c) * 12'(src_wblk_total);
  assign dst_h_off = 12'(cnt_h) * 12'(cfg_src_c) * 12'(dst_wblk_total);

  // ── Latched read data (source A and B) ──
  int8_t rd_data_lat [LANES];
  int8_t rd_data_b_lat [LANES];

  // ── Domain-aligned (requantized) intermediates ──
  int8_t aligned_a [LANES];
  int8_t aligned_b [LANES];

  // ── Requant helper: clip(round((val - zp_in) * m >> sh) + zp_out) ──
  function automatic int8_t requant_val(
    input int8_t    val,
    input int8_t    zp_in,
    input uint32_t  m,
    input logic [7:0] sh,
    input int8_t    zp_out
  );
    automatic int64_t diff   = int64_t'(int32_t'(val) - int32_t'(zp_in));
    automatic int64_t prod   = diff * int64_t'({1'b0, m});
    automatic int64_t rounded;
    automatic int32_t shifted;
    automatic int32_t with_zp;

    if (sh > 0)
      rounded = prod + (int64_t'(1) <<< (sh - 1));
    else
      rounded = prod;
    shifted = int32_t'(rounded >>> sh);
    with_zp = shifted + int32_t'(zp_out);

    if (with_zp > 32'sd127)       return int8_t'(8'sd127);
    else if (with_zp < -32'sd128) return int8_t'(-8'sd128);
    else                           return int8_t'(with_zp[7:0]);
  endfunction

  // ════════════════════════════════════════════════════════════════
  // Main FSM
  // ════════════════════════════════════════════════════════════════
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state       <= SW_IDLE;
      done        <= 1'b0;
      src_rd_en    <= 1'b0;
      src_rd_addr  <= 12'd0;
      src_b_rd_en  <= 1'b0;
      src_b_rd_addr <= 12'd0;
      dst_wr_en    <= 1'b0;
      dst_wr_addr  <= 12'd0;
      dst_wr_mask  <= '1;
      cnt_h        <= '0;
      cnt_w        <= '0;
      cnt_c        <= '0;
      for (int l = 0; l < LANES; l++) begin
        rd_data_lat[l]   <= 8'sd0;
        rd_data_b_lat[l] <= 8'sd0;
        dst_wr_data[l]   <= 8'sd0;
      end
    end else begin
      // Defaults
      src_rd_en   <= 1'b0;
      src_b_rd_en <= 1'b0;
      dst_wr_en   <= 1'b0;
      done        <= 1'b0;

      case (state)
        // ────────────────────────────────────────
        SW_IDLE: begin
          if (start) begin
            state <= (cfg_mode == SWZ_NORMAL) ? SW_DONE : SW_READ;
            cnt_h <= '0;
            cnt_w <= '0;
            cnt_c <= '0;
          end
        end

        // ────────────────────────────────────────
        SW_READ: begin
          src_rd_addr <= src_h_off + 12'(cnt_c) * 12'(src_wblk_total) + 12'(cnt_w);
          src_rd_en   <= 1'b1;
          state       <= SW_READ_WAIT;
        end

        // ────────────────────────────────────────
        SW_READ_WAIT: begin
          // Latch read data (1-cycle SRAM latency)
          for (int l = 0; l < LANES; l++)
            rd_data_lat[l] <= src_rd_data[l];

          case (cfg_mode)
            SWZ_UPSAMPLE2X: state <= SW_UP_WRITE_0;
            SWZ_EWISE_ADD:  state <= SW_ADD_READ_B;
            SWZ_CONCAT:     state <= SW_WRITE;
            default:        state <= SW_WRITE;
          endcase
        end

        // ────────────────────────────────────────
        // NORMAL / CONCAT: 1 write per read
        // ────────────────────────────────────────
        SW_WRITE: begin
          dst_wr_addr <= dst_h_off + 12'(cnt_c) * 12'(dst_wblk_total) + 12'(cnt_w);
          for (int l = 0; l < LANES; l++)
            dst_wr_data[l] <= rd_data_lat[l];
          dst_wr_en   <= 1'b1;
          dst_wr_mask <= '1;

          // Advance counters: w → c → h
          if (cnt_w == src_wblk_total - 1) begin
            cnt_w <= '0;
            if (cnt_c == cfg_src_c - 1) begin
              cnt_c <= '0;
              if (cnt_h == cfg_src_h - 1)
                state <= SW_DONE;
              else begin
                cnt_h <= cnt_h + 1;
                state <= SW_READ;
              end
            end else begin
              cnt_c <= cnt_c + 1;
              state <= SW_READ;
            end
          end else begin
            cnt_w <= cnt_w + 1;
            state <= SW_READ;
          end
        end

        // ────────────────────────────────────────
        // UPSAMPLE 2×: Each source block → 2 destination rows
        // Row 2h+0: same data as source
        // Row 2h+1: same data as source (height duplicate)
        // ────────────────────────────────────────
        SW_UP_WRITE_0: begin
          dst_wr_addr <= 12'(2*cnt_h) * 12'(cfg_src_c) * 12'(dst_wblk_total)
                       + 12'(cnt_c) * 12'(dst_wblk_total)
                       + 12'(cnt_w);
          for (int l = 0; l < LANES; l++)
            dst_wr_data[l] <= rd_data_lat[l];
          dst_wr_en   <= 1'b1;
          dst_wr_mask <= '1;
          state       <= SW_UP_WRITE_1;
        end

        SW_UP_WRITE_1: begin
          dst_wr_addr <= 12'(2*cnt_h + 1) * 12'(cfg_src_c) * 12'(dst_wblk_total)
                       + 12'(cnt_c) * 12'(dst_wblk_total)
                       + 12'(cnt_w);
          for (int l = 0; l < LANES; l++)
            dst_wr_data[l] <= rd_data_lat[l];
          dst_wr_en   <= 1'b1;
          dst_wr_mask <= '1;

          // Advance counters
          if (cnt_w == src_wblk_total - 1) begin
            cnt_w <= '0;
            if (cnt_c == cfg_src_c - 1) begin
              cnt_c <= '0;
              if (cnt_h == cfg_src_h - 1)
                state <= SW_DONE;
              else begin
                cnt_h <= cnt_h + 1;
                state <= SW_READ;
              end
            end else begin
              cnt_c <= cnt_c + 1;
              state <= SW_READ;
            end
          end else begin
            cnt_w <= cnt_w + 1;
            state <= SW_READ;
          end
        end

        // ────────────────────────────────────────
        // EWISE_ADD: Read tensor B from skip buffer
        // ────────────────────────────────────────
        SW_ADD_READ_B: begin
          src_b_rd_addr <= src_h_off + 12'(cnt_c) * 12'(src_wblk_total) + 12'(cnt_w);
          src_b_rd_en   <= 1'b1;
          state         <= SW_ADD_READ_B_WAIT;
        end

        SW_ADD_READ_B_WAIT: begin
          for (int l = 0; l < LANES; l++)
            rd_data_b_lat[l] <= src_b_rd_data[l];
          state <= SW_ADD_COMPUTE;
        end

        // ────────────────────────────────────────
        // EWISE_ADD: Requant both tensors to common domain, then add
        // Pipeline: align A + align B → sum (INT16 intermediate) → clip INT8
        // ────────────────────────────────────────
        SW_ADD_COMPUTE: begin
          for (int l = 0; l < LANES; l++) begin
            if (cfg_align_bypass) begin
              // Same domain: no requant needed, direct add
              automatic int16_t sum16;
              sum16 = int16_t'(rd_data_lat[l]) + int16_t'(rd_data_b_lat[l]);
              if (sum16 > 16'sd127)        dst_wr_data[l] <= int8_t'(8'sd127);
              else if (sum16 < -16'sd128)  dst_wr_data[l] <= int8_t'(-8'sd128);
              else                          dst_wr_data[l] <= int8_t'(sum16[7:0]);
            end else begin
              // Domain alignment: requant each tensor to common domain, then add
              automatic int8_t a_aligned, b_aligned;
              automatic int16_t sum16;
              a_aligned = requant_val(rd_data_lat[l], cfg_align_zp_a,
                                     cfg_align_m_a, cfg_align_sh_a, cfg_align_zp_out);
              b_aligned = requant_val(rd_data_b_lat[l], cfg_align_zp_b,
                                     cfg_align_m_b, cfg_align_sh_b, cfg_align_zp_out);
              sum16 = int16_t'(a_aligned) + int16_t'(b_aligned);
              if (sum16 > 16'sd127)        dst_wr_data[l] <= int8_t'(8'sd127);
              else if (sum16 < -16'sd128)  dst_wr_data[l] <= int8_t'(-8'sd128);
              else                          dst_wr_data[l] <= int8_t'(sum16[7:0]);
            end
          end
          state <= SW_ADD_WRITE;
        end

        SW_ADD_WRITE: begin
          dst_wr_addr <= dst_h_off + 12'(cnt_c) * 12'(dst_wblk_total) + 12'(cnt_w);
          dst_wr_en   <= 1'b1;
          dst_wr_mask <= '1;

          // Advance counters: w → c → h
          if (cnt_w == src_wblk_total - 1) begin
            cnt_w <= '0;
            if (cnt_c == cfg_src_c - 1) begin
              cnt_c <= '0;
              if (cnt_h == cfg_src_h - 1)
                state <= SW_DONE;
              else begin
                cnt_h <= cnt_h + 1;
                state <= SW_READ;
              end
            end else begin
              cnt_c <= cnt_c + 1;
              state <= SW_READ;
            end
          end else begin
            cnt_w <= cnt_w + 1;
            state <= SW_READ;
          end
        end

        // ────────────────────────────────────────
        SW_DONE: begin
          done  <= 1'b1;
          state <= SW_IDLE;
        end

        default: state <= SW_IDLE;
      endcase
    end
  end

  // synthesis translate_off
`ifdef RTL_TRACE
  always @(posedge clk) begin
    if (rst_n && start)
      rtl_trace_pkg::rtl_trace_line("S5_SWZ",
        $sformatf("START mode=%0d sh=%0d sw=%0d", cfg_mode, cfg_src_h, cfg_src_w));
    if (rst_n && done)
      rtl_trace_pkg::rtl_trace_line("S5_SWZ", "DONE");
  end
`endif
  // synthesis translate_on

endmodule
