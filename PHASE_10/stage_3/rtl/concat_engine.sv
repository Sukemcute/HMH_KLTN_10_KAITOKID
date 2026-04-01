`timescale 1ns/1ps
// ============================================================================
// CONCAT Primitive Engine (P5)
// Concatenates two input tensors A and B along the channel dimension with
// domain alignment (requantization) to a common output scale/zero-point.
//
// Golden Path (from detailed_concat_logic_report.md):
//   For channels from A:
//     A_float = (A_int8 - zp_A) * scale_A
//     A_aligned = clamp(round(A_float / scale_out) + zp_out, -128, 127)
//   For channels from B:
//     B_float = (B_int8 - zp_B) * scale_B
//     B_aligned = clamp(round(B_float / scale_out) + zp_out, -128, 127)
//
// Integer-only requantization:
//   M_x = round(scale_X / scale_out * 2^shift)
//   val_aligned = clamp(( (val - zp_in) * M_x + (1 << (shift-1)) ) >> shift
//                       + zp_out, -128, 127)
//
// Pass-through optimisation: if scale_A == scale_out && zp_A == zp_out,
// skip requant for tensor A (likewise for B).
//
// FSM: IDLE → ALIGN_A → ALIGN_B → DONE
//
// Covers: FPN/PAN QConcat layers (L8, L12, etc.)
// ============================================================================
module concat_engine
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
  input  logic [9:0]  cfg_h,          // Spatial height (same for A, B, and output)
  input  logic [9:0]  cfg_w,          // Spatial width  (same for A, B, and output)
  input  logic [8:0]  cfg_c_a,        // Channel count for input A
  input  logic [8:0]  cfg_c_b,        // Channel count for input B

  // Quantization parameters (float64 scale stored as 64-bit real for golden model;
  // integer-only multiplier/shift used in actual datapath)
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

  // Pass-through flags (set by host when scale/zp match exactly)
  input  logic        cfg_passthru_a, // 1 = skip requant for A
  input  logic        cfg_passthru_b, // 1 = skip requant for B

  // ═══════════ Input A SRAM (LANES-wide read) ═══════════
  // Layout: [H][C_a][num_wblk], word = LANES INT8 values
  output logic [23:0]       a_rd_addr,
  output logic              a_rd_en,
  input  logic signed [7:0] a_rd_data [LANES],

  // ═══════════ Input B SRAM (LANES-wide read) ═══════════
  // Layout: [H][C_b][num_wblk], word = LANES INT8 values
  output logic [23:0]       b_rd_addr,
  output logic              b_rd_en,
  input  logic signed [7:0] b_rd_data [LANES],

  // ═══════════ Output SRAM (LANES-wide write) ═══════════
  // Layout: [H][(C_a+C_b)][num_wblk], word = LANES INT8 values
  output logic [23:0]       ofm_wr_addr,
  output logic              ofm_wr_en,
  output logic signed [7:0] ofm_wr_data [LANES]
);

  // ═══════════════════════════════════════════════════════════════════
  // FSM STATES
  // ═══════════════════════════════════════════════════════════════════
  typedef enum logic [2:0] {
    ST_IDLE    = 3'd0,
    ST_ALIGN_A = 3'd1,  // Read A, requant, write to output channels [0..C_a-1]
    ST_ALIGN_B = 3'd2,  // Read B, requant, write to output channels [C_a..C_a+C_b-1]
    ST_DONE    = 3'd3
  } concat_state_e;

  concat_state_e state;

  // ═══════════════════════════════════════════════════════════════════
  // LATCHED CONFIGURATION
  // ═══════════════════════════════════════════════════════════════════
  logic [9:0]  h_val, w_val;
  logic [8:0]  c_a_val, c_b_val;
  logic signed [31:0] m_a_val, m_b_val;
  logic [5:0]  shift_val;
  logic signed [7:0]  zp_a_val, zp_b_val, zp_out_val;
  logic        passthru_a, passthru_b;
  logic [5:0]  num_wblk;          // ceil(W / LANES)

  // ═══════════════════════════════════════════════════════════════════
  // LOOP COUNTERS
  // ═══════════════════════════════════════════════════════════════════
  logic [9:0]  cnt_h;             // Current row
  logic [8:0]  cnt_c;             // Current channel within current tensor
  logic [5:0]  cnt_wblk;          // Current width block
  logic        rd_phase;          // 0: issue SRAM read, 1: capture & write

  // ═══════════════════════════════════════════════════════════════════
  // REQUANTIZATION DATAPATH (combinational)
  // Computes LANES aligned output values from SRAM read data
  // ═══════════════════════════════════════════════════════════════════
  logic signed [7:0] requant_out [LANES];

  // Current requant parameters muxed by state (A or B)
  logic signed [31:0] cur_m;
  logic signed [7:0]  cur_zp_in;
  logic               cur_passthru;
  logic signed [7:0]  cur_rd_data [LANES];

  always_comb begin
    if (state == ST_ALIGN_A) begin
      cur_m        = m_a_val;
      cur_zp_in    = zp_a_val;
      cur_passthru = passthru_a;
      for (int l = 0; l < LANES; l++)
        cur_rd_data[l] = a_rd_data[l];
    end else begin
      cur_m        = m_b_val;
      cur_zp_in    = zp_b_val;
      cur_passthru = passthru_b;
      for (int l = 0; l < LANES; l++)
        cur_rd_data[l] = b_rd_data[l];
    end
  end

  // Requant each lane:
  //   val_aligned = clamp(((val - zp_in) * M + (1 << (shift-1))) >> shift + zp_out, -128, 127)
  always_comb begin
    for (int l = 0; l < LANES; l++) begin
      if (cur_passthru) begin
        // Pass-through: no requant needed
        requant_out[l] = cur_rd_data[l];
      end else begin
        automatic logic signed [15:0] dequant;
        automatic logic signed [31:0] requanted;
        automatic logic signed [15:0] with_zp;

        // Subtract input zero-point
        dequant = 16'(cur_rd_data[l]) - 16'(cur_zp_in);

        // Fixed-point multiply and round-shift
        requanted = requant_fixed(32'(dequant), cur_m, shift_val);

        // Add output zero-point and clamp to int8
        with_zp = requanted[15:0] + 16'(zp_out_val);
        if (with_zp > 16'sd127)
          requant_out[l] = 8'sd127;
        else if (with_zp < -16'sd128)
          requant_out[l] = -8'sd128;
        else
          requant_out[l] = with_zp[7:0];
      end
    end
  end

  // ═══════════════════════════════════════════════════════════════════
  // OUTPUT CHANNEL OFFSET: B channels are placed after A channels
  // ═══════════════════════════════════════════════════════════════════
  logic [8:0] out_c_offset;
  always_comb begin
    out_c_offset = (state == ST_ALIGN_B) ? c_a_val : 9'd0;
  end

  // Total output channels
  logic [9:0] c_total;
  assign c_total = {1'b0, c_a_val} + {1'b0, c_b_val};

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
      rd_phase   <= 1'b0;
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
            state       <= ST_ALIGN_A;
            busy        <= 1'b1;
            // Latch configuration
            h_val       <= cfg_h;
            w_val       <= cfg_w;
            c_a_val     <= cfg_c_a;
            c_b_val     <= cfg_c_b;
            m_a_val     <= cfg_m_a;
            m_b_val     <= cfg_m_b;
            shift_val   <= cfg_shift;
            zp_a_val    <= cfg_zp_a;
            zp_b_val    <= cfg_zp_b;
            zp_out_val  <= cfg_zp_out;
            passthru_a  <= cfg_passthru_a;
            passthru_b  <= cfg_passthru_b;
            num_wblk    <= (cfg_w + LANES - 1) / LANES;
            // Reset loop counters
            cnt_h       <= '0;
            cnt_c       <= '0;
            cnt_wblk    <= '0;
            rd_phase    <= 1'b0;
          end
        end

        // ─────────────────────────────────────────────
        // ST_ALIGN_A: Iterate over (h, c_a, wblk), read from A, requant, write
        // ─────────────────────────────────────────────
        ST_ALIGN_A: begin
          if (!rd_phase) begin
            // Phase 0: Issue SRAM read for input A
            // A address = h * C_a * num_wblk + c * num_wblk + wblk
            a_rd_addr <= 24'(cnt_h) * 24'(c_a_val) * 24'(num_wblk)
                       + 24'(cnt_c) * 24'(num_wblk)
                       + 24'(cnt_wblk);
            a_rd_en   <= 1'b1;
            rd_phase  <= 1'b1;
          end else begin
            // Phase 1: Write requantized result to output SRAM
            // Output address = h * C_total * num_wblk + (c_offset + c) * num_wblk + wblk
            ofm_wr_addr <= 24'(cnt_h) * 24'(c_total) * 24'(num_wblk)
                         + 24'(out_c_offset + cnt_c) * 24'(num_wblk)
                         + 24'(cnt_wblk);
            ofm_wr_en <= 1'b1;
            for (int l = 0; l < LANES; l++)
              ofm_wr_data[l] <= requant_out[l];

            rd_phase <= 1'b0;

            // Advance loop: wblk → c → h
            if (cnt_wblk == num_wblk - 1) begin
              cnt_wblk <= '0;
              if (cnt_c == c_a_val - 1) begin
                cnt_c <= '0;
                if (cnt_h == h_val - 1) begin
                  // All of A aligned → move to B
                  cnt_h    <= '0;
                  state    <= ST_ALIGN_B;
                end else begin
                  cnt_h <= cnt_h + 1'b1;
                end
              end else begin
                cnt_c <= cnt_c + 1'b1;
              end
            end else begin
              cnt_wblk <= cnt_wblk + 1'b1;
            end
          end
        end

        // ─────────────────────────────────────────────
        // ST_ALIGN_B: Iterate over (h, c_b, wblk), read from B, requant, write
        // ─────────────────────────────────────────────
        ST_ALIGN_B: begin
          if (!rd_phase) begin
            // Phase 0: Issue SRAM read for input B
            // B address = h * C_b * num_wblk + c * num_wblk + wblk
            b_rd_addr <= 24'(cnt_h) * 24'(c_b_val) * 24'(num_wblk)
                       + 24'(cnt_c) * 24'(num_wblk)
                       + 24'(cnt_wblk);
            b_rd_en   <= 1'b1;
            rd_phase  <= 1'b1;
          end else begin
            // Phase 1: Write requantized result to output SRAM
            // Output address = h * C_total * num_wblk + (C_a + c) * num_wblk + wblk
            ofm_wr_addr <= 24'(cnt_h) * 24'(c_total) * 24'(num_wblk)
                         + 24'(out_c_offset + cnt_c) * 24'(num_wblk)
                         + 24'(cnt_wblk);
            ofm_wr_en <= 1'b1;
            for (int l = 0; l < LANES; l++)
              ofm_wr_data[l] <= requant_out[l];

            rd_phase <= 1'b0;

            // Advance loop: wblk → c → h
            if (cnt_wblk == num_wblk - 1) begin
              cnt_wblk <= '0;
              if (cnt_c == c_b_val - 1) begin
                cnt_c <= '0;
                if (cnt_h == h_val - 1) begin
                  // All of B aligned → done
                  state <= ST_DONE;
                end else begin
                  cnt_h <= cnt_h + 1'b1;
                end
              end else begin
                cnt_c <= cnt_c + 1'b1;
              end
            end else begin
              cnt_wblk <= cnt_wblk + 1'b1;
            end
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
