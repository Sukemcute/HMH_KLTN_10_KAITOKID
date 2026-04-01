`timescale 1ns/1ps
// ============================================================================
// EWISE_ADD Primitive Engine (P7)
// Element-wise addition of two tensors A and B with domain alignment
// (requantization) to a common output scale/zero-point.
//
// Golden Path (from detailed_add_logic_report.md):
//   For each position (h, w, c):
//     A_float = (A_int8 - zp_A) * scale_A
//     B_float = (B_int8 - zp_B) * scale_B
//     sum_float = A_float + B_float
//     Y_int8 = clamp(round(sum_float / scale_out) + zp_out, -128, 127)
//
// Integer-only implementation:
//   M_a = round(scale_A / scale_out * 2^shift)
//   M_b = round(scale_B / scale_out * 2^shift)
//   a_aligned = (A_int8 - zp_A) * M_a
//   b_aligned = (B_int8 - zp_B) * M_b
//   sum = a_aligned + b_aligned + (1 << (shift - 1))   // rounding bias
//   Y = clamp((sum >> shift) + zp_out, -128, 127)
//
// Both inputs share the same spatial and channel dimensions.
//
// FSM: IDLE → for each (h, c, wblk): READ_AB → COMPUTE_WRITE → DONE
//
// Covers: Residual addition in QC2fCIB blocks
// ============================================================================
module ewise_add_engine
  import yolo_accel_pkg::*;
#(
  parameter int LANES = 32
)(
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  output logic        done,
  output logic        busy,

  // ═══════════ Configuration ═══════════
  input  logic [9:0]  cfg_h,          // Spatial height
  input  logic [9:0]  cfg_w,          // Spatial width
  input  logic [8:0]  cfg_channels,   // Number of channels (same for A, B, output)

  // Quantization parameters (float64 scale for golden model reference)
  input  real         cfg_scale_a,    // Scale for input A (float64)
  input  logic signed [7:0] cfg_zp_a, // Zero-point for input A (int8)
  input  real         cfg_scale_b,    // Scale for input B (float64)
  input  logic signed [7:0] cfg_zp_b, // Zero-point for input B (int8)
  input  real         cfg_scale_out,  // Scale for output (float64)
  input  logic signed [7:0] cfg_zp_out, // Zero-point for output (int8)

  // Pre-computed integer requantization multipliers & shift
  input  logic signed [31:0] cfg_m_a, // M_a = round(scale_A / scale_out * 2^shift)
  input  logic signed [31:0] cfg_m_b, // M_b = round(scale_B / scale_out * 2^shift)
  input  logic [5:0]         cfg_shift, // Common shift amount

  // ═══════════ Input A SRAM (LANES-wide read) ═══════════
  // Layout: [H][C][num_wblk], word = LANES INT8 values
  output logic [23:0]       a_rd_addr,
  output logic              a_rd_en,
  input  logic signed [7:0] a_rd_data [LANES],

  // ═══════════ Input B SRAM (LANES-wide read) ═══════════
  // Layout: [H][C][num_wblk], word = LANES INT8 values
  output logic [23:0]       b_rd_addr,
  output logic              b_rd_en,
  input  logic signed [7:0] b_rd_data [LANES],

  // ═══════════ Output SRAM (LANES-wide write) ═══════════
  // Layout: [H][C][num_wblk], word = LANES INT8 values
  output logic [23:0]       ofm_wr_addr,
  output logic              ofm_wr_en,
  output logic signed [7:0] ofm_wr_data [LANES]
);

  // ═══════════════════════════════════════════════════════════════════
  // FSM STATES
  // ═══════════════════════════════════════════════════════════════════
  typedef enum logic [2:0] {
    ST_IDLE      = 3'd0,
    ST_READ_AB   = 3'd1,  // Issue simultaneous reads from A and B SRAMs
    ST_COMPUTE   = 3'd2,  // Capture read data, compute requant add, write output
    ST_DONE      = 3'd3
  } add_state_e;

  add_state_e state;

  // ═══════════════════════════════════════════════════════════════════
  // LATCHED CONFIGURATION
  // ═══════════════════════════════════════════════════════════════════
  logic [9:0]  h_val, w_val;
  logic [8:0]  ch_val;
  logic signed [31:0] m_a_val, m_b_val;
  logic [5:0]  shift_val;
  logic signed [7:0]  zp_a_val, zp_b_val, zp_out_val;
  logic [5:0]  num_wblk;          // ceil(W / LANES)

  // ═══════════════════════════════════════════════════════════════════
  // LOOP COUNTERS
  // ═══════════════════════════════════════════════════════════════════
  logic [9:0]  cnt_h;             // Current row
  logic [8:0]  cnt_c;             // Current channel
  logic [5:0]  cnt_wblk;          // Current width block

  // ═══════════════════════════════════════════════════════════════════
  // CAPTURED INPUT DATA
  // ═══════════════════════════════════════════════════════════════════
  logic signed [7:0] a_buf [LANES];
  logic signed [7:0] b_buf [LANES];

  // ═══════════════════════════════════════════════════════════════════
  // ELEMENT-WISE ADD + REQUANT DATAPATH (combinational)
  // Integer-only path:
  //   a_aligned = (A_int8 - zp_A) * M_a
  //   b_aligned = (B_int8 - zp_B) * M_b
  //   sum = a_aligned + b_aligned + rounding_bias
  //   Y = clamp((sum >> shift) + zp_out, -128, 127)
  // ═══════════════════════════════════════════════════════════════════
  logic signed [7:0] add_out [LANES];

  always_comb begin
    for (int l = 0; l < LANES; l++) begin
      automatic logic signed [15:0] a_dq;       // A - zp_A (16-bit to avoid overflow)
      automatic logic signed [15:0] b_dq;       // B - zp_B
      automatic logic signed [47:0] a_aligned;  // (A - zp_A) * M_a
      automatic logic signed [47:0] b_aligned;  // (B - zp_B) * M_b
      automatic logic signed [47:0] sum_raw;    // a_aligned + b_aligned + rounding
      automatic logic signed [31:0] shifted;    // sum >> shift
      automatic logic signed [15:0] with_zp;    // shifted + zp_out

      // Step 1: Subtract zero-points
      a_dq = 16'(a_buf[l]) - 16'(zp_a_val);
      b_dq = 16'(b_buf[l]) - 16'(zp_b_val);

      // Step 2: Scale to output domain
      a_aligned = 48'(a_dq) * 48'(m_a_val);
      b_aligned = 48'(b_dq) * 48'(m_b_val);

      // Step 3: Sum with rounding bias (half-up)
      if (shift_val > 0)
        sum_raw = a_aligned + b_aligned + (48'sd1 <<< (shift_val - 1));
      else
        sum_raw = a_aligned + b_aligned;

      // Step 4: Arithmetic right shift
      shifted = 32'(sum_raw >>> shift_val);

      // Step 5: Add output zero-point and clamp to int8
      with_zp = shifted[15:0] + 16'(zp_out_val);
      if (with_zp > 16'sd127)
        add_out[l] = 8'sd127;
      else if (with_zp < -16'sd128)
        add_out[l] = -8'sd128;
      else
        add_out[l] = with_zp[7:0];
    end
  end

  // ═══════════════════════════════════════════════════════════════════
  // SRAM ADDRESS (shared for both A and B, since layouts are identical)
  // addr = h * ch_val * num_wblk + c * num_wblk + wblk
  // ═══════════════════════════════════════════════════════════════════
  logic [23:0] sram_addr;
  assign sram_addr = 24'(cnt_h) * 24'(ch_val) * 24'(num_wblk)
                   + 24'(cnt_c) * 24'(num_wblk)
                   + 24'(cnt_wblk);

  // ═══════════════════════════════════════════════════════════════════
  // MAIN FSM
  // ═══════════════════════════════════════════════════════════════════
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state      <= ST_IDLE;
      done       <= 1'b0;
      busy       <= 1'b0;
      a_rd_en    <= 1'b0;
      b_rd_en    <= 1'b0;
      ofm_wr_en  <= 1'b0;
      cnt_h      <= '0;
      cnt_c      <= '0;
      cnt_wblk   <= '0;
    end else begin

      // Default: deassert strobes
      a_rd_en   <= 1'b0;
      b_rd_en   <= 1'b0;
      ofm_wr_en <= 1'b0;

      case (state)

        // ─────────────────────────────────────────────
        ST_IDLE: begin
          done <= 1'b0;
          if (start) begin
            state       <= ST_READ_AB;
            busy        <= 1'b1;
            // Latch configuration
            h_val       <= cfg_h;
            w_val       <= cfg_w;
            ch_val      <= cfg_channels;
            m_a_val     <= cfg_m_a;
            m_b_val     <= cfg_m_b;
            shift_val   <= cfg_shift;
            zp_a_val    <= cfg_zp_a;
            zp_b_val    <= cfg_zp_b;
            zp_out_val  <= cfg_zp_out;
            num_wblk    <= (cfg_w + LANES - 1) / LANES;
            // Reset loop counters
            cnt_h       <= '0;
            cnt_c       <= '0;
            cnt_wblk    <= '0;
          end
        end

        // ─────────────────────────────────────────────
        // ST_READ_AB: Issue simultaneous reads from A and B SRAMs
        // Both share the same (h, c, wblk) address since dimensions are identical
        // ─────────────────────────────────────────────
        ST_READ_AB: begin
          a_rd_addr <= sram_addr;
          b_rd_addr <= sram_addr;
          a_rd_en   <= 1'b1;
          b_rd_en   <= 1'b1;
          state     <= ST_COMPUTE;
        end

        // ─────────────────────────────────────────────
        // ST_COMPUTE: Capture read data into buffers, compute add, write output
        // (1 cycle SRAM latency: data available this cycle from previous read)
        // ─────────────────────────────────────────────
        ST_COMPUTE: begin
          // Capture SRAM read data into buffers
          for (int l = 0; l < LANES; l++) begin
            a_buf[l] <= a_rd_data[l];
            b_buf[l] <= b_rd_data[l];
          end

          // Write computed result to output SRAM
          // Output uses same address layout as inputs
          ofm_wr_addr <= sram_addr;
          ofm_wr_en   <= 1'b1;
          for (int l = 0; l < LANES; l++)
            ofm_wr_data[l] <= add_out[l];

          // Advance loop: wblk → c → h
          if (cnt_wblk == num_wblk - 1) begin
            cnt_wblk <= '0;
            if (cnt_c == ch_val - 1) begin
              cnt_c <= '0;
              if (cnt_h == h_val - 1) begin
                state <= ST_DONE;
              end else begin
                cnt_h <= cnt_h + 1'b1;
                state <= ST_READ_AB;
              end
            end else begin
              cnt_c <= cnt_c + 1'b1;
              state <= ST_READ_AB;
            end
          end else begin
            cnt_wblk <= cnt_wblk + 1'b1;
            state    <= ST_READ_AB;
          end
        end

        // ─────────────────────────────────────────────
        ST_DONE: begin
          done  <= 1'b1;
          busy  <= 1'b0;
          state <= ST_IDLE;
        end

        default: state <= ST_IDLE;
      endcase
    end
  end

endmodule
