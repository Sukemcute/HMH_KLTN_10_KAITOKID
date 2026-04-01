`timescale 1ns/1ps
// ============================================================================
// UPSAMPLE_NEAREST 2x Primitive Engine (P6)
// Pure address remapping with no arithmetic computation.
//
// For each input position (c, h, w):
//   output[c][2*h  ][2*w  ] = input[c][h][w]
//   output[c][2*h  ][2*w+1] = input[c][h][w]
//   output[c][2*h+1][2*w  ] = input[c][h][w]
//   output[c][2*h+1][2*w+1] = input[c][h][w]
//
// Scale and zero-point pass through unchanged:
//   scale_out = scale_in, zp_out = zp_in
//
// Output dimensions: (cfg_channels, 2*cfg_h, 2*cfg_w)
//
// FSM: IDLE → for each (h, c, wblk): READ → WRITE×4 → DONE
//
// Covers: Neck upsample layers (L7, L11)
// ============================================================================
module upsample_engine
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
  input  logic [9:0]  cfg_h,          // Input height
  input  logic [9:0]  cfg_w,          // Input width
  input  logic [8:0]  cfg_channels,   // Number of channels

  // ═══════════ Input SRAM (LANES-wide read) ═══════════
  // Layout: [H][C][num_wblk_in], word = LANES INT8 values
  output logic [23:0]       ifm_rd_addr,
  output logic              ifm_rd_en,
  input  logic signed [7:0] ifm_rd_data [LANES],

  // ═══════════ Output SRAM (LANES-wide write) ═══════════
  // Layout: [2*H][C][num_wblk_out], word = LANES INT8 values
  // Output width = 2*W, so num_wblk_out = ceil(2*W / LANES)
  output logic [23:0]       ofm_wr_addr,
  output logic              ofm_wr_en,
  output logic signed [7:0] ofm_wr_data [LANES]
);

  // ═══════════════════════════════════════════════════════════════════
  // FSM STATES
  // ═══════════════════════════════════════════════════════════════════
  typedef enum logic [2:0] {
    ST_IDLE    = 3'd0,
    ST_READ    = 3'd1,  // Issue SRAM read for input block
    ST_WRITE0  = 3'd2,  // Write to output row 2h,   columns 2w..2w+2*LANES-1 (even)
    ST_WRITE1  = 3'd3,  // Write to output row 2h,   columns (odd half)
    ST_WRITE2  = 3'd4,  // Write to output row 2h+1, columns 2w..2w+2*LANES-1 (even)
    ST_WRITE3  = 3'd5,  // Write to output row 2h+1, columns (odd half)
    ST_DONE    = 3'd6
  } upsample_state_e;

  upsample_state_e state;

  // ═══════════════════════════════════════════════════════════════════
  // LATCHED CONFIGURATION
  // ═══════════════════════════════════════════════════════════════════
  logic [9:0]  h_val, w_val;
  logic [8:0]  ch_val;
  logic [5:0]  num_wblk_in;       // ceil(W / LANES)
  logic [5:0]  num_wblk_out;      // ceil(2*W / LANES)

  // ═══════════════════════════════════════════════════════════════════
  // LOOP COUNTERS
  // ═══════════════════════════════════════════════════════════════════
  logic [9:0]  cnt_h;             // Current input row
  logic [8:0]  cnt_c;             // Current channel
  logic [5:0]  cnt_wblk;          // Current input width block

  // ═══════════════════════════════════════════════════════════════════
  // CAPTURED INPUT DATA BUFFER
  // Holds LANES values read from input for 2x duplication
  // ═══════════════════════════════════════════════════════════════════
  logic signed [7:0] in_buf [LANES];

  // ═══════════════════════════════════════════════════════════════════
  // 2x HORIZONTAL EXPANSION LOGIC
  // Input lane l contains pixel at w_base + l (w_base = cnt_wblk * LANES).
  // Output needs pixel at w_out = 2*(w_base + l):
  //   For even output block (offset 0): out_lane[2l]   = in[l], out_lane[2l+1] = in[l]
  //   This produces 2*LANES output pixels from LANES input pixels.
  //   We split into two LANES-wide writes:
  //     Write0/2: output lanes 0..LANES-1 → in_buf[l/2] duplicated
  //     Write1/3: output lanes 0..LANES-1 → in_buf[LANES/2 + l/2] duplicated
  //
  // Mapping (for write to output wblk = 2*cnt_wblk + sub):
  //   sub=0: out[lane] = in_buf[lane/2]           (first LANES outputs)
  //   sub=1: out[lane] = in_buf[LANES/2 + lane/2] (second LANES outputs)
  // ═══════════════════════════════════════════════════════════════════
  logic signed [7:0] expanded_lo [LANES]; // For sub=0 writes (WRITE0, WRITE2)
  logic signed [7:0] expanded_hi [LANES]; // For sub=1 writes (WRITE1, WRITE3)

  always_comb begin
    for (int l = 0; l < LANES; l++) begin
      // Each input pixel duplicated to two adjacent output pixels
      // sub=0 covers input pixels [0 .. LANES/2 - 1] → output pixels [0..LANES-1]
      expanded_lo[l] = in_buf[l / 2];
      // sub=1 covers input pixels [LANES/2 .. LANES-1] → output pixels [0..LANES-1]
      expanded_hi[l] = in_buf[LANES/2 + l/2];
    end
  end

  // ═══════════════════════════════════════════════════════════════════
  // OUTPUT ADDRESS COMPUTATION
  // Output layout: [2*H][C][num_wblk_out]
  // For row_out, channel c, output wblk w_out:
  //   addr = row_out * ch_val * num_wblk_out + c * num_wblk_out + w_out
  // ═══════════════════════════════════════════════════════════════════
  logic [9:0]  out_row;           // 2*cnt_h or 2*cnt_h+1
  logic [5:0]  out_wblk;          // 2*cnt_wblk or 2*cnt_wblk+1

  // ═══════════════════════════════════════════════════════════════════
  // MAIN FSM
  // ═══════════════════════════════════════════════════════════════════
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state      <= ST_IDLE;
      done       <= 1'b0;
      busy       <= 1'b0;
      ifm_rd_en  <= 1'b0;
      ofm_wr_en  <= 1'b0;
      cnt_h      <= '0;
      cnt_c      <= '0;
      cnt_wblk   <= '0;
    end else begin

      // Default: deassert strobes
      ifm_rd_en <= 1'b0;
      ofm_wr_en <= 1'b0;

      case (state)

        // ─────────────────────────────────────────────
        ST_IDLE: begin
          done <= 1'b0;
          if (start) begin
            state        <= ST_READ;
            busy         <= 1'b1;
            // Latch configuration
            h_val        <= cfg_h;
            w_val        <= cfg_w;
            ch_val       <= cfg_channels;
            num_wblk_in  <= (cfg_w + LANES - 1) / LANES;
            num_wblk_out <= (2 * cfg_w + LANES - 1) / LANES;
            // Reset loop counters
            cnt_h        <= '0;
            cnt_c        <= '0;
            cnt_wblk     <= '0;
          end
        end

        // ─────────────────────────────────────────────
        // ST_READ: Read LANES input values for current (h, c, wblk)
        // ─────────────────────────────────────────────
        ST_READ: begin
          // Input address = h * ch_val * num_wblk_in + c * num_wblk_in + wblk
          ifm_rd_addr <= 24'(cnt_h) * 24'(ch_val) * 24'(num_wblk_in)
                       + 24'(cnt_c) * 24'(num_wblk_in)
                       + 24'(cnt_wblk);
          ifm_rd_en   <= 1'b1;
          state       <= ST_WRITE0;
        end

        // ─────────────────────────────────────────────
        // ST_WRITE0: Capture read data; write first half to row 2h
        // Output wblk = 2*cnt_wblk, row = 2*cnt_h
        // ─────────────────────────────────────────────
        ST_WRITE0: begin
          // Capture SRAM read data into buffer (1 cycle latency)
          for (int l = 0; l < LANES; l++)
            in_buf[l] <= ifm_rd_data[l];

          // Write expanded_lo to output row 2h, output wblk = 2*cnt_wblk
          out_row  = {cnt_h[8:0], 1'b0};      // 2 * cnt_h
          out_wblk = {cnt_wblk[4:0], 1'b0};   // 2 * cnt_wblk

          ofm_wr_addr <= 24'(out_row) * 24'(ch_val) * 24'(num_wblk_out)
                       + 24'(cnt_c) * 24'(num_wblk_out)
                       + 24'(out_wblk);
          ofm_wr_en   <= 1'b1;
          for (int l = 0; l < LANES; l++)
            ofm_wr_data[l] <= expanded_lo[l];

          state <= ST_WRITE1;
        end

        // ─────────────────────────────────────────────
        // ST_WRITE1: Write second half to row 2h
        // Output wblk = 2*cnt_wblk + 1, row = 2*cnt_h
        // ─────────────────────────────────────────────
        ST_WRITE1: begin
          out_row  = {cnt_h[8:0], 1'b0};              // 2 * cnt_h
          out_wblk = {cnt_wblk[4:0], 1'b1};           // 2 * cnt_wblk + 1

          ofm_wr_addr <= 24'(out_row) * 24'(ch_val) * 24'(num_wblk_out)
                       + 24'(cnt_c) * 24'(num_wblk_out)
                       + 24'(out_wblk);
          ofm_wr_en   <= 1'b1;
          for (int l = 0; l < LANES; l++)
            ofm_wr_data[l] <= expanded_hi[l];

          state <= ST_WRITE2;
        end

        // ─────────────────────────────────────────────
        // ST_WRITE2: Write first half to row 2h+1
        // Output wblk = 2*cnt_wblk, row = 2*cnt_h + 1
        // ─────────────────────────────────────────────
        ST_WRITE2: begin
          out_row  = {cnt_h[8:0], 1'b0} + 10'd1;     // 2 * cnt_h + 1
          out_wblk = {cnt_wblk[4:0], 1'b0};           // 2 * cnt_wblk

          ofm_wr_addr <= 24'(out_row) * 24'(ch_val) * 24'(num_wblk_out)
                       + 24'(cnt_c) * 24'(num_wblk_out)
                       + 24'(out_wblk);
          ofm_wr_en   <= 1'b1;
          for (int l = 0; l < LANES; l++)
            ofm_wr_data[l] <= expanded_lo[l];

          state <= ST_WRITE3;
        end

        // ─────────────────────────────────────────────
        // ST_WRITE3: Write second half to row 2h+1, then advance loop
        // Output wblk = 2*cnt_wblk + 1, row = 2*cnt_h + 1
        // ─────────────────────────────────────────────
        ST_WRITE3: begin
          out_row  = {cnt_h[8:0], 1'b0} + 10'd1;     // 2 * cnt_h + 1
          out_wblk = {cnt_wblk[4:0], 1'b1};           // 2 * cnt_wblk + 1

          ofm_wr_addr <= 24'(out_row) * 24'(ch_val) * 24'(num_wblk_out)
                       + 24'(cnt_c) * 24'(num_wblk_out)
                       + 24'(out_wblk);
          ofm_wr_en   <= 1'b1;
          for (int l = 0; l < LANES; l++)
            ofm_wr_data[l] <= expanded_hi[l];

          // Advance loop: wblk → c → h
          if (cnt_wblk == num_wblk_in - 1) begin
            cnt_wblk <= '0;
            if (cnt_c == ch_val - 1) begin
              cnt_c <= '0;
              if (cnt_h == h_val - 1) begin
                state <= ST_DONE;
              end else begin
                cnt_h <= cnt_h + 1'b1;
                state <= ST_READ;
              end
            end else begin
              cnt_c <= cnt_c + 1'b1;
              state <= ST_READ;
            end
          end else begin
            cnt_wblk <= cnt_wblk + 1'b1;
            state    <= ST_READ;
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
