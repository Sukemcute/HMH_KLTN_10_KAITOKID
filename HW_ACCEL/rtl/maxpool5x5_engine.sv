`timescale 1ns/1ps
// ============================================================================
// MAXPOOL_5x5 Primitive Engine (P3)
// Computes: Y[h][w][c] = max_{kh,kw in 5x5} X[h+kh-2][w+kw-2][c]
//
// Kernel 5x5, padding=2, stride=1 -> output same size as input.
// Pure comparison engine -- no MAC, no weights, no bias, no PPU.
// Uses comparator_tree (25 inputs -> 1 max per lane, 5-stage pipeline).
// Output scale/zp = input scale/zp, so no requantization needed.
//
// Used 3x in series within SPPF (Layer 9):
//   P1 = MP5(X1),  P2 = MP5(P1),  P3 = MP5(P2)
//
// Loop order: for h_out -> for channel -> for wblk
//   1. LOAD_ROWS:  Load 5 padded input rows into row_buf
//   2. COMPUTE:    Extract 5x5=25 values per lane, feed comparator_tree
//   3. WAIT_PIPE:  Wait 5 cycles for comparator_tree pipeline to drain
//   4. WRITE:      Write LANES max results to output SRAM
// ============================================================================
module maxpool5x5_engine
  import yolo_accel_pkg::*;
#(
  parameter int LANES     = 32,
  parameter int MAX_W_PAD = 672    // Max padded input row width (includes pad=2 on each side)
)(
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  output logic        done,
  output logic        busy,

  // =========== Configuration ===========
  input  logic [9:0]  cfg_h,         // Input height (= output height, stride=1)
  input  logic [9:0]  cfg_w,         // Input width  (= output width,  stride=1)
  input  logic [8:0]  cfg_channels,  // Cin = Cout for maxpool

  // =========== Input Feature Map SRAM (LANES-wide read) ===========
  // Layout: [H_pad][C][num_wblk_pad], word = LANES INT8 values
  // H_pad = H + 2*pad = H + 4;  W_pad = W + 4 (padded to LANES boundary)
  // Address = h_pad * channels * num_wblk_pad + ch * num_wblk_pad + w_word
  output logic [23:0]       ifm_rd_addr,
  output logic              ifm_rd_en,
  input  logic signed [7:0] ifm_rd_data [LANES],

  // =========== Output Feature Map SRAM (LANES-wide write) ===========
  // Layout: [Hout][C][num_wblk_out], word = LANES INT8 values
  // Address = h_out * channels * num_wblk_out + ch * num_wblk_out + w_word
  output logic [23:0]       ofm_wr_addr,
  output logic              ofm_wr_en,
  output logic signed [7:0] ofm_wr_data [LANES]
);

  // ═══════════════════════════════════════════════════════════════════
  // INTERNAL SIGNALS
  // ═══════════════════════════════════════════════════════════════════

  // Row buffers: 5 kernel rows x MAX_W_PAD spatial positions
  // row_buf[kh][spatial_idx] holds one INT8 value
  logic signed [7:0] row_buf [5][MAX_W_PAD];

  // Comparator tree interface
  logic signed [7:0] ct_data_in [25][LANES]; // 25 inputs per lane
  logic              ct_en;
  logic signed [7:0] ct_max_out [LANES];
  logic              ct_max_valid;

  // FSM states (subset of eng_state_e)
  typedef enum logic [3:0] {
    ST_IDLE       = 4'h0,
    ST_LOAD_ROWS  = 4'h1,  // Load 5 padded input rows into row_buf
    ST_COMPUTE    = 4'h2,  // Feed 25 values per lane to comparator_tree
    ST_WAIT_PIPE  = 4'h3,  // Wait for comparator_tree 5-stage pipeline
    ST_WRITE      = 4'h4,  // Write LANES results to output SRAM
    ST_NEXT_WBLK  = 4'h5,  // Advance wblk counter
    ST_NEXT_CH    = 4'h6,  // Advance channel counter
    ST_NEXT_HOUT  = 4'h7,  // Advance h_out counter
    ST_DONE       = 4'hF
  } mp_state_e;

  mp_state_e state;

  // Loop counters
  logic [9:0] cnt_hout;       // Current output row
  logic [8:0] cnt_ch;         // Current channel
  logic [5:0] cnt_wblk;       // Current output width block

  // Row loading counters
  logic [5:0] load_w_idx;     // Word index within row
  logic [2:0] load_kh_idx;    // Kernel row being loaded (0..4)
  logic       load_rd_phase;  // 0: issue SRAM read, 1: capture data

  // Pipeline drain counter
  logic [2:0] pipe_cnt;

  // Latched configuration
  logic [9:0]  h_val;          // Input/output height
  logic [9:0]  w_val;          // Input/output width
  logic [9:0]  h_pad_val;      // H + 4 (padded height)
  logic [9:0]  w_pad_val;      // W + 4 (padded width, before rounding to LANES)
  logic [8:0]  ch_val;         // Number of channels
  logic [5:0]  num_wblk_pad;   // ceil(W_pad / LANES) -- padded input width blocks
  logic [5:0]  num_wblk_out;   // ceil(W / LANES)     -- output width blocks

  // ═══════════════════════════════════════════════════════════════════
  // COMPARATOR TREE INSTANTIATION
  // 25 inputs -> 1 max per lane, 5-stage pipeline
  // ═══════════════════════════════════════════════════════════════════
  comparator_tree #(
    .LANES      (LANES),
    .NUM_INPUTS (25)
  ) u_comp_tree (
    .clk       (clk),
    .rst_n     (rst_n),
    .en        (ct_en),
    .data_in   (ct_data_in),
    .max_out   (ct_max_out),
    .max_valid (ct_max_valid)
  );

  // ═══════════════════════════════════════════════════════════════════
  // COMPARATOR TREE INPUT MUX
  // For each lane l, extract a 5x5 window from row_buf:
  //   ct_data_in[kh*5 + kw][l] = row_buf[kh][cnt_wblk*LANES + l + kw]
  //
  // The padded input already has padding=2 on each side, so
  // for output pixel at w_out, the window starts at w_out in the
  // padded coordinate (pad=2 means offset 0 into padded buffer).
  // Since stride=1, output wblk maps to the same spatial position
  // in the padded input, with an extra 4 columns for the kernel tail.
  // ═══════════════════════════════════════════════════════════════════
  always_comb begin
    for (int kh = 0; kh < 5; kh++) begin
      for (int kw = 0; kw < 5; kw++) begin
        for (int l = 0; l < LANES; l++) begin
          automatic int sidx;
          // Spatial index into padded row buffer
          sidx = int'(cnt_wblk) * LANES + l + kw;
          if (state == ST_COMPUTE && sidx >= 0 && sidx < MAX_W_PAD)
            ct_data_in[kh * 5 + kw][l] = row_buf[kh][sidx];
          else
            ct_data_in[kh * 5 + kw][l] = -8'sd128; // -128 = INT8 min (identity for max)
        end
      end
    end
  end

  // ═══════════════════════════════════════════════════════════════════
  // MAIN FSM
  // ═══════════════════════════════════════════════════════════════════
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state         <= ST_IDLE;
      done          <= 1'b0;
      busy          <= 1'b0;
      ct_en         <= 1'b0;
      ifm_rd_en     <= 1'b0;
      ofm_wr_en     <= 1'b0;
      cnt_hout      <= '0;
      cnt_ch        <= '0;
      cnt_wblk      <= '0;
      load_w_idx    <= '0;
      load_kh_idx   <= '0;
      load_rd_phase <= 1'b0;
      pipe_cnt      <= '0;
    end else begin

      // Default: deassert one-shot controls
      ifm_rd_en <= 1'b0;
      ofm_wr_en <= 1'b0;
      ct_en     <= 1'b0;

      case (state)
        // ─────────────────────────────────────────────
        // IDLE: Wait for start pulse, latch configuration
        // ─────────────────────────────────────────────
        ST_IDLE: begin
          done <= 1'b0;
          if (start) begin
            state        <= ST_LOAD_ROWS;
            busy         <= 1'b1;

            // Latch configuration
            h_val        <= cfg_h;
            w_val        <= cfg_w;
            h_pad_val    <= cfg_h + 10'd4;          // H + 2*pad
            w_pad_val    <= cfg_w + 10'd4;           // W + 2*pad
            ch_val       <= cfg_channels;
            num_wblk_pad <= 6'((int'(cfg_w) + 4 + LANES - 1) / LANES);
            num_wblk_out <= 6'((int'(cfg_w) + LANES - 1) / LANES);

            // Reset all loop counters
            cnt_hout      <= '0;
            cnt_ch        <= '0;
            cnt_wblk      <= '0;
            load_w_idx    <= '0;
            load_kh_idx   <= '0;
            load_rd_phase <= 1'b0;
          end
        end

        // ─────────────────────────────────────────────
        // LOAD_ROWS: Load 5 padded input rows into row_buf
        // Each row is read in num_wblk_pad words of LANES bytes.
        // Uses 2-phase protocol: phase 0 = issue read, phase 1 = capture.
        //
        // Row in padded input for kernel row kh:
        //   h_pad_row = cnt_hout + kh  (h_out maps to h_pad = h_out since pad=2
        //                                and kernel offset -2..+2 covers 0..4)
        //
        // SRAM address = h_pad_row * ch_val * num_wblk_pad
        //              + cnt_ch * num_wblk_pad + load_w_idx
        // ─────────────────────────────────────────────
        ST_LOAD_ROWS: begin
          if (!load_rd_phase) begin
            // Phase 0: Issue SRAM read request
            automatic logic [9:0]  h_pad_row;
            automatic logic [23:0] addr;

            h_pad_row = cnt_hout + {7'b0, load_kh_idx};

            addr = 24'(h_pad_row) * 24'(ch_val) * 24'(num_wblk_pad)
                 + 24'(cnt_ch)    * 24'(num_wblk_pad)
                 + 24'(load_w_idx);

            ifm_rd_addr   <= addr;
            ifm_rd_en     <= 1'b1;
            load_rd_phase <= 1'b1;
          end else begin
            // Phase 1: Capture SRAM data into row_buf
            for (int l = 0; l < LANES; l++) begin
              automatic int widx;
              widx = int'(load_w_idx) * LANES + l;
              if (widx < MAX_W_PAD)
                row_buf[load_kh_idx][widx] <= ifm_rd_data[l];
            end
            load_rd_phase <= 1'b0;

            // Advance word counter within row
            if (load_w_idx == num_wblk_pad - 1) begin
              load_w_idx <= '0;
              // Advance kernel row counter
              if (load_kh_idx == 3'd4) begin
                // All 5 rows loaded -> feed comparator tree
                load_kh_idx <= '0;
                state       <= ST_COMPUTE;
              end else begin
                load_kh_idx <= load_kh_idx + 1'b1;
              end
            end else begin
              load_w_idx <= load_w_idx + 1'b1;
            end
          end
        end

        // ─────────────────────────────────────────────
        // COMPUTE: Assert ct_en for one cycle to push the 25 values
        // (already wired combinationally from row_buf) into the
        // comparator_tree pipeline.
        // ─────────────────────────────────────────────
        ST_COMPUTE: begin
          ct_en    <= 1'b1;
          state    <= ST_WAIT_PIPE;
          pipe_cnt <= '0;
        end

        // ─────────────────────────────────────────────
        // WAIT_PIPE: Wait for comparator_tree 5-stage pipeline
        // ct_max_valid asserts when result is ready.
        // We also count cycles as a safety measure.
        // ─────────────────────────────────────────────
        ST_WAIT_PIPE: begin
          pipe_cnt <= pipe_cnt + 1'b1;
          if (ct_max_valid) begin
            state <= ST_WRITE;
          end
        end

        // ─────────────────────────────────────────────
        // WRITE: Write LANES max values to output SRAM
        // No PPU / requantization -- output = comparator_tree result directly.
        //
        // Output address = h_out * ch_val * num_wblk_out
        //                + cnt_ch * num_wblk_out + cnt_wblk
        // ─────────────────────────────────────────────
        ST_WRITE: begin
          ofm_wr_en   <= 1'b1;
          ofm_wr_addr <= 24'(cnt_hout) * 24'(ch_val) * 24'(num_wblk_out)
                       + 24'(cnt_ch)   * 24'(num_wblk_out)
                       + 24'(cnt_wblk);

          for (int l = 0; l < LANES; l++)
            ofm_wr_data[l] <= ct_max_out[l];

          state <= ST_NEXT_WBLK;
        end

        // ─────────────────────────────────────────────
        // LOOP CONTROL: wblk -> channel -> h_out
        // ─────────────────────────────────────────────
        ST_NEXT_WBLK: begin
          if (cnt_wblk == num_wblk_out - 1) begin
            cnt_wblk <= '0;
            state    <= ST_NEXT_CH;
          end else begin
            cnt_wblk <= cnt_wblk + 1'b1;
            // Same 5 rows, same channel, next spatial block
            // Row buffers already loaded for this (h_out, ch),
            // so go directly to COMPUTE.
            state    <= ST_COMPUTE;
          end
        end

        ST_NEXT_CH: begin
          if (cnt_ch == ch_val - 1) begin
            cnt_ch <= '0;
            state  <= ST_NEXT_HOUT;
          end else begin
            cnt_ch   <= cnt_ch + 1'b1;
            cnt_wblk <= '0;
            // Need to reload rows for the new channel
            state         <= ST_LOAD_ROWS;
            load_w_idx    <= '0;
            load_kh_idx   <= '0;
            load_rd_phase <= 1'b0;
          end
        end

        ST_NEXT_HOUT: begin
          if (cnt_hout == h_val - 1) begin
            state <= ST_DONE;
          end else begin
            cnt_hout <= cnt_hout + 1'b1;
            cnt_ch   <= '0;
            cnt_wblk <= '0;
            // Need to reload rows for the new output row
            state         <= ST_LOAD_ROWS;
            load_w_idx    <= '0;
            load_kh_idx   <= '0;
            load_rd_phase <= 1'b0;
          end
        end

        // ─────────────────────────────────────────────
        // DONE: Signal completion, return to IDLE
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
