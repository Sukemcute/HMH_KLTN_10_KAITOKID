`timescale 1ns/1ps
// ============================================================================
// DW_7x7_MULTIPASS Primitive Engine (P8)
// Computes: Y[h][w][c] = Sum_{kh=0..6, kw=0..6} X[h+kh-3][w+kw-3][c] * W[kh][kw][c]
//           + bias[c] -> requant -> activation -> clamp -> INT8
//
// Multipass strategy using 3 pe_unit instances (same 3-PE-row architecture):
//   Pass 1 (kernel rows 0-2): PEs handle kh=0,1,2, each accumulates kw=0..6
//                              Result stored to PSUM buffer as partial sum
//   Pass 2 (kernel rows 3-5): PEs handle kh=3,4,5, each accumulates kw=0..6
//                              Accumulated with Pass 1 PSUM -> store PSUM buffer
//   Pass 3 (kernel row 6):    PE[0] handles kh=6, accumulates kw=0..6
//                              PE[1],PE[2] contribute 0 (inactive)
//                              Accumulated with Pass 2 PSUM -> PPU -> INT8 output
//
// Depthwise: Cin = Cout, one channel processed at a time.
// Weight layout in SRAM: [C][7][7] (49 weights per channel)
//
// Covers: Layer 22 (QC2fCIB) depthwise 7x7 convolution
// ============================================================================
module dwconv7x7_engine
  import yolo_accel_pkg::*;
#(
  parameter int LANES     = 32,
  parameter int MAX_W_PAD = 672    // Max padded input row width (Win + 2*pad)
)(
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  output logic        done,
  output logic        busy,

  // =========== Configuration ===========
  input  logic [8:0]  cfg_channels,    // Depthwise: Cin = Cout
  input  logic [9:0]  cfg_h,           // Input height (unpadded)
  input  logic [9:0]  cfg_w,           // Input width (unpadded)
  input  logic [9:0]  cfg_hout,        // Output height
  input  logic [9:0]  cfg_wout,        // Output width
  input  logic [9:0]  cfg_w_pad,       // Padded input width (Win + 2*pad)
  input  act_mode_e   cfg_act_mode,    // Activation: NONE/SILU/RELU/CLAMP
  input  logic signed [7:0] cfg_zp_out,// Output zero-point

  // =========== Input Feature Map SRAM (LANES-wide read) ===========
  // Layout: [H_pad][C][num_w_words], word = LANES INT8 values
  // Address = h_pad * channels * num_wblk_in + ch * num_wblk_in + w_word
  output logic [23:0]       ifm_rd_addr,
  output logic              ifm_rd_en,
  input  logic signed [7:0] ifm_rd_data [LANES],

  // =========== Weight SRAM (byte read) ===========
  // Layout: [C][7][7] -- 49 weights per channel
  // Address = ch * 49 + kh * 7 + kw
  output logic [23:0]       wgt_rd_addr,
  output logic              wgt_rd_en,
  input  logic signed [7:0] wgt_rd_data,

  // =========== Bias & Quantization Params (per channel, preloaded) ===========
  input  logic signed [31:0] bias_arr  [MAX_COUT],
  input  logic signed [31:0] m_int_arr [MAX_COUT],
  input  logic [5:0]         shift_arr [MAX_COUT],

  // =========== SiLU LUT ===========
  input  logic signed [7:0]  silu_lut [256],

  // =========== Output Feature Map SRAM (LANES-wide write) ===========
  // Layout: [Hout][C][num_w_out_words]
  // Address = h_out * channels * num_wblk_out + ch * num_wblk_out + w_word
  output logic [23:0]       ofm_wr_addr,
  output logic              ofm_wr_en,
  output logic signed [7:0] ofm_wr_data [LANES]
);

  // =======================================================================
  // FSM States -- extended from eng_state_e with multipass-specific states
  // =======================================================================
  typedef enum logic [3:0] {
    ST_IDLE       = 4'h0,
    ST_LOAD_ROWS  = 4'h1,   // Load 3 (or 1) input rows into row buffers
    ST_LOAD_WGT   = 4'h2,   // Load 7 (or 21) weights for current pass
    ST_COMPUTE    = 4'h3,   // Feed PEs with 7 kw cycles
    ST_DRAIN      = 4'h4,   // Wait for PE pipeline drain
    ST_PSUM_ACC   = 4'h5,   // Column-reduce + accumulate with PSUM buffer
    ST_PPU        = 4'h6,   // Post-processing (bias + requant + act + clamp)
    ST_WRITE      = 4'h7,   // Write LANES INT8 outputs
    ST_NEXT_WBLK  = 4'h8,   // Advance wblk loop
    ST_NEXT_PASS  = 4'h9,   // Advance to next pass (1->2->3)
    ST_NEXT_CH    = 4'hA,   // Advance channel loop
    ST_NEXT_HOUT  = 4'hB,   // Advance hout loop
    ST_DONE       = 4'hF
  } dw7_state_e;

  // =======================================================================
  // INTERNAL SIGNALS
  // =======================================================================

  // Row buffers: 3 rows x MAX_W_PAD spatial positions
  // Re-loaded per pass for different kernel rows
  logic signed [7:0] row_buf [3][MAX_W_PAD];

  // Weight buffer: [pe_row][kw] -- 3 PEs x 7 kw values per pass
  logic signed [7:0] wgt_buf [3][7];

  // PE control signals
  logic              pe_en;
  logic              pe_clear;
  logic signed [7:0] pe_x_in  [3][LANES];
  logic signed [7:0] pe_w_in  [3][LANES];
  logic signed [31:0] pe_psum [3][LANES];
  logic              pe_valid [3];

  // Column-reduced partial sum (sum of 3 PE rows)
  logic signed [31:0] reduced_psum [LANES];

  // PSUM buffer: stores INT32 partial sums between passes
  // Indexed by [wblk][lane]
  logic signed [31:0] psum_buf [MAX_WBLK][LANES];

  // PPU results
  logic signed [7:0]  ppu_out [LANES];

  // FSM state
  dw7_state_e state;

  // Loop counters
  logic [9:0] cnt_hout;         // Output row counter
  logic [8:0] cnt_ch;           // Channel counter (depthwise: cin=cout)
  logic [5:0] cnt_wblk;         // Width-block counter
  logic [2:0] cnt_kw;           // kw counter (0..6)
  logic [1:0] cnt_pass;         // Pass counter: 0=Pass1, 1=Pass2, 2=Pass3

  // Row loading counters
  logic [5:0] load_w_idx;       // Word index within row
  logic [1:0] load_row_idx;     // Which of the 3 (or 1) rows being loaded (0,1,2)
  logic       load_rd_phase;    // 0: issue read, 1: capture data

  // Weight loading counters
  logic [1:0] wgt_pe_idx;       // Which PE's weights (0,1,2)
  logic [2:0] wgt_kw_idx;       // kw index (0..6)
  logic       wgt_rd_phase;     // 0: issue read, 1: capture data
  logic       wgt_first_read;   // Flag: first read has no data to capture

  // Pipeline drain counter
  logic [2:0] drain_cnt;

  // Derived configuration (latched at start)
  logic [5:0] num_wblk_in;      // ceil(W_pad / LANES)
  logic [5:0] num_wblk_out;     // ceil(Wout / LANES)
  logic [8:0] ch_val;           // Latched channel count
  logic [9:0] hout_val;         // Latched output height
  logic [9:0] wout_val;         // Latched output width
  logic [9:0] w_pad_val;        // Latched padded width

  // Number of active PEs per pass: 3 for Pass1/2, 1 for Pass3
  logic [1:0] active_pe_cnt;    // 3 or 1
  // Number of rows to load per pass: 3 for Pass1/2, 1 for Pass3
  logic [1:0] rows_to_load;

  // Base kernel-row offset for current pass
  // Pass 0: kh_base=0 (rows 0,1,2)
  // Pass 1: kh_base=3 (rows 3,4,5)
  // Pass 2: kh_base=6 (row 6 only)
  logic [2:0] kh_base;

  // =======================================================================
  // PE UNIT INSTANTIATION (3 units, one per kernel row within a pass)
  // =======================================================================
  genvar pe;
  generate
    for (pe = 0; pe < 3; pe++) begin : gen_pe
      pe_unit #(.LANES(LANES)) u_pe (
        .clk       (clk),
        .rst_n     (rst_n),
        .en        (pe_en),
        .clear_psum(pe_clear),
        .mode      (PE_DW7),
        .x_in      (pe_x_in[pe]),
        .w_in      (pe_w_in[pe]),
        .psum_out  (pe_psum[pe]),
        .psum_valid(pe_valid[pe])
      );
    end
  endgenerate

  // =======================================================================
  // PE INPUT MUX: Extract data from row buffers for current kw
  //
  // Depthwise stride=1 with pad=3:
  //   For output pixel w, the input window is [w+kw-3 .. w+kw-3] = [w-3 .. w+3]
  //   For a LANES-wide block at wblk, output w = wblk*LANES + l
  //   Input x position = wblk*LANES + l + kw   (padding already in h_pad dim)
  //
  // PE[p] gets row_buf[p], weight wgt_buf[p][kw]
  // For Pass 3: PE[1] and PE[2] get zero inputs (inactive)
  // =======================================================================
  always_comb begin
    for (int p = 0; p < 3; p++) begin
      for (int l = 0; l < LANES; l++) begin
        if (state == ST_COMPUTE) begin
          // Check if this PE is active in the current pass
          if (p[1:0] < active_pe_cnt) begin
            automatic int base_w;
            // Input position = wblk*LANES + lane + kw
            // (padding offset is handled at row-load time by loading from h_pad rows)
            base_w = int'(cnt_wblk) * LANES + l + int'(cnt_kw);
            if (base_w >= 0 && base_w < MAX_W_PAD)
              pe_x_in[p][l] = row_buf[p][base_w];
            else
              pe_x_in[p][l] = 8'sd0;

            // Broadcast this PE's weight for current kw to all lanes (depthwise)
            pe_w_in[p][l] = wgt_buf[p][cnt_kw];
          end else begin
            // Inactive PE: zero input so it contributes 0
            pe_x_in[p][l] = 8'sd0;
            pe_w_in[p][l] = 8'sd0;
          end
        end else begin
          pe_x_in[p][l] = 8'sd0;
          pe_w_in[p][l] = 8'sd0;
        end
      end
    end
  end

  // =======================================================================
  // COLUMN REDUCE: Sum 3 PE rows -> 1 result per lane
  // For Pass 3, PE[1] and PE[2] output 0 so the sum is just PE[0].
  // =======================================================================
  always_comb begin
    for (int l = 0; l < LANES; l++) begin
      reduced_psum[l] = pe_psum[0][l] + pe_psum[1][l] + pe_psum[2][l];
    end
  end

  // =======================================================================
  // INLINE PPU: Bias + Requant + Activation + Clamp
  // Applied after Pass 3 accumulation with PSUM buffer
  // Input: psum_buf[cnt_wblk][l] (already accumulated across all 3 passes)
  // =======================================================================
  always_comb begin
    for (int l = 0; l < LANES; l++) begin
      automatic logic signed [31:0] biased;
      automatic logic signed [31:0] requanted;
      automatic logic signed [15:0] act_in;
      automatic logic signed [7:0]  act_val;
      automatic logic signed [15:0] final_val;

      // Stage 1: Bias add (psum_buf already has full 7x7 accumulation)
      biased = psum_buf[cnt_wblk][l] + bias_arr[cnt_ch];

      // Stage 2: Fixed-point requantization (x m_int >>> shift, half_up rounding)
      requanted = requant_fixed(biased, m_int_arr[cnt_ch], shift_arr[cnt_ch]);

      // Clamp to 16-bit range for activation indexing
      if (requanted > 32'sd32767)
        act_in = 16'sd32767;
      else if (requanted < -32'sd32768)
        act_in = -16'sd32768;
      else
        act_in = requanted[15:0];

      // Stage 3: Activation function
      case (cfg_act_mode)
        ACT_SILU: begin
          act_val = silu_lut[silu_index(act_in)];
        end
        ACT_RELU: begin
          act_val = (act_in > 0) ? clamp_int8(32'(act_in)) : 8'sd0;
        end
        default: begin  // ACT_NONE, ACT_CLAMP
          act_val = clamp_int8(32'(act_in));
        end
      endcase

      // Stage 4: Add output zero-point and final clamp
      final_val = 16'(act_val) + 16'(cfg_zp_out);
      if (final_val > 16'sd127)
        ppu_out[l] = 8'sd127;
      else if (final_val < -16'sd128)
        ppu_out[l] = -8'sd128;
      else
        ppu_out[l] = final_val[7:0];
    end
  end

  // =======================================================================
  // PASS CONFIGURATION HELPER
  // Derives kh_base, active_pe_cnt, rows_to_load from cnt_pass
  // =======================================================================
  always_comb begin
    case (cnt_pass)
      2'd0: begin  // Pass 1: kernel rows 0,1,2
        kh_base        = 3'd0;
        active_pe_cnt  = 2'd3;
        rows_to_load   = 2'd3;
      end
      2'd1: begin  // Pass 2: kernel rows 3,4,5
        kh_base        = 3'd3;
        active_pe_cnt  = 2'd3;
        rows_to_load   = 2'd3;
      end
      default: begin  // Pass 3: kernel row 6 only
        kh_base        = 3'd6;
        active_pe_cnt  = 2'd1;
        rows_to_load   = 2'd1;
      end
    endcase
  end

  // =======================================================================
  // MAIN FSM
  // =======================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state          <= ST_IDLE;
      done           <= 1'b0;
      busy           <= 1'b0;
      pe_en          <= 1'b0;
      pe_clear       <= 1'b0;
      ifm_rd_en      <= 1'b0;
      wgt_rd_en      <= 1'b0;
      ofm_wr_en      <= 1'b0;
      cnt_hout       <= '0;
      cnt_ch         <= '0;
      cnt_wblk       <= '0;
      cnt_kw         <= '0;
      cnt_pass       <= '0;
      load_w_idx     <= '0;
      load_row_idx   <= '0;
      load_rd_phase  <= 1'b0;
      wgt_pe_idx     <= '0;
      wgt_kw_idx     <= '0;
      wgt_rd_phase   <= 1'b0;
      wgt_first_read <= 1'b1;
      drain_cnt      <= '0;
    end else begin

      // Default: deassert control signals each cycle
      ifm_rd_en <= 1'b0;
      wgt_rd_en <= 1'b0;
      ofm_wr_en <= 1'b0;
      pe_en     <= 1'b0;
      pe_clear  <= 1'b0;

      case (state)

        // ---------------------------------------------------------
        // IDLE: Wait for start signal, latch configuration
        // ---------------------------------------------------------
        ST_IDLE: begin
          done <= 1'b0;
          if (start) begin
            state       <= ST_LOAD_ROWS;
            busy        <= 1'b1;
            // Latch configuration
            ch_val      <= cfg_channels;
            hout_val    <= cfg_hout;
            wout_val    <= cfg_wout;
            w_pad_val   <= cfg_w_pad;
            num_wblk_in  <= (cfg_w_pad + LANES[9:0] - 10'd1) / LANES[9:0];
            num_wblk_out <= (cfg_wout  + LANES[9:0] - 10'd1) / LANES[9:0];
            // Reset all loop counters
            cnt_hout      <= '0;
            cnt_ch        <= '0;
            cnt_wblk      <= '0;
            cnt_pass      <= '0;
            // Start loading rows for Pass 1
            load_w_idx    <= '0;
            load_row_idx  <= '0;
            load_rd_phase <= 1'b0;
          end
        end

        // ---------------------------------------------------------
        // LOAD_ROWS: Load input rows into row_buf for current pass
        //   Pass 1/2: Load 3 rows (kh_base+0, kh_base+1, kh_base+2)
        //   Pass 3:   Load 1 row  (kh_base = 6)
        //
        // Input address: h_in * channels * num_wblk_in + ch * num_wblk_in + w_word
        // where h_in = cnt_hout + (kh_base + load_row_idx) for stride=1, pad=3
        //       (padded input already includes padding rows)
        // ---------------------------------------------------------
        ST_LOAD_ROWS: begin
          if (!load_rd_phase) begin
            // Phase 0: Issue SRAM read request
            automatic logic [9:0] h_in_row;
            automatic logic [23:0] addr;

            // h_in = hout + kh_base + load_row_idx
            // (padding is already baked into the padded input feature map)
            h_in_row = cnt_hout + {7'b0, kh_base} + {8'b0, load_row_idx};

            addr = 24'(h_in_row) * 24'(ch_val) * 24'(num_wblk_in)
                 + 24'(cnt_ch) * 24'(num_wblk_in)
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
                row_buf[load_row_idx][widx] <= ifm_rd_data[l];
            end
            load_rd_phase <= 1'b0;

            // Advance word counter within current row
            if (load_w_idx == num_wblk_in - 1) begin
              load_w_idx <= '0;
              // Check if all rows for this pass are loaded
              if (load_row_idx == rows_to_load - 2'd1) begin
                // All rows loaded -> move to weight loading
                load_row_idx   <= '0;
                state          <= ST_LOAD_WGT;
                wgt_pe_idx     <= '0;
                wgt_kw_idx     <= '0;
                wgt_rd_phase   <= 1'b0;
                wgt_first_read <= 1'b1;
              end else begin
                load_row_idx <= load_row_idx + 1'b1;
              end
            end else begin
              load_w_idx <= load_w_idx + 1'b1;
            end
          end
        end

        // ---------------------------------------------------------
        // LOAD_WGT: Load weights for current pass's kernel rows
        //   Pass 1: wgt_buf[0][0..6], wgt_buf[1][0..6], wgt_buf[2][0..6]
        //           = W[ch][0][0..6], W[ch][1][0..6], W[ch][2][0..6]
        //   Pass 2: W[ch][3][0..6], W[ch][4][0..6], W[ch][5][0..6]
        //   Pass 3: W[ch][6][0..6] into wgt_buf[0][0..6] only
        //
        // Weight SRAM address = ch * 49 + (kh_base + wgt_pe_idx) * 7 + wgt_kw_idx
        // Uses 2-phase read: issue addr, then capture data next cycle
        // ---------------------------------------------------------
        ST_LOAD_WGT: begin
          if (!wgt_rd_phase) begin
            // Phase 0: Issue weight SRAM read
            automatic logic [23:0] w_addr;
            automatic logic [2:0]  actual_kh;
            actual_kh = kh_base + {1'b0, wgt_pe_idx};
            w_addr = 24'(cnt_ch) * 24'd49
                   + 24'(actual_kh) * 24'd7
                   + 24'(wgt_kw_idx);
            wgt_rd_addr  <= w_addr;
            wgt_rd_en    <= 1'b1;
            wgt_rd_phase <= 1'b1;
          end else begin
            // Phase 1: Capture weight data
            wgt_buf[wgt_pe_idx][wgt_kw_idx] <= wgt_rd_data;
            wgt_rd_phase <= 1'b0;

            // Advance weight loading counters
            if (wgt_kw_idx == 3'd6) begin
              wgt_kw_idx <= '0;
              // Check if all PEs' weights for this pass are loaded
              if (wgt_pe_idx == active_pe_cnt - 2'd1) begin
                // All weights loaded -> start compute
                wgt_pe_idx <= '0;
                state      <= ST_COMPUTE;
                cnt_kw     <= '0;
                // Clear PE accumulators for fresh accumulation
                pe_clear   <= 1'b1;
                pe_en      <= 1'b1;
              end else begin
                wgt_pe_idx <= wgt_pe_idx + 1'b1;
              end
            end else begin
              wgt_kw_idx <= wgt_kw_idx + 1'b1;
            end
          end
        end

        // ---------------------------------------------------------
        // COMPUTE: Feed PEs with 7 kw cycles (kw = 0..6)
        // PEs internally accumulate the MAC results across kw.
        // For Pass 3: only PE[0] is active; PE[1],PE[2] get zeros.
        // ---------------------------------------------------------
        ST_COMPUTE: begin
          pe_en    <= 1'b1;
          pe_clear <= (cnt_kw == 3'd0) ? 1'b1 : 1'b0;

          if (cnt_kw == 3'd6) begin
            // Last kw cycle -> drain PE pipeline
            cnt_kw    <= '0;
            state     <= ST_DRAIN;
            drain_cnt <= '0;
          end else begin
            cnt_kw <= cnt_kw + 3'd1;
          end
        end

        // ---------------------------------------------------------
        // DRAIN: Wait DSP_LATENCY cycles for PE pipeline to flush
        // ---------------------------------------------------------
        ST_DRAIN: begin
          drain_cnt <= drain_cnt + 3'd1;
          if (drain_cnt == DSP_LATENCY[2:0]) begin
            state <= ST_PSUM_ACC;
          end
        end

        // ---------------------------------------------------------
        // PSUM_ACC: Column-reduce PE outputs + accumulate with PSUM buffer
        //   If Pass 1 (cnt_pass==0):
        //     psum_buf[wblk][l] = reduced_psum[l]  (first partial sum)
        //   If Pass 2 (cnt_pass==1):
        //     psum_buf[wblk][l] += reduced_psum[l]  (add to Pass 1 result)
        //   If Pass 3 (cnt_pass==2):
        //     psum_buf[wblk][l] += reduced_psum[l]  (complete sum -> PPU)
        // ---------------------------------------------------------
        ST_PSUM_ACC: begin
          for (int l = 0; l < LANES; l++) begin
            if (cnt_pass == 2'd0) begin
              // Pass 1: store fresh partial sum
              psum_buf[cnt_wblk][l] <= reduced_psum[l];
            end else begin
              // Pass 2 or 3: accumulate with previous passes
              psum_buf[cnt_wblk][l] <= psum_buf[cnt_wblk][l] + reduced_psum[l];
            end
          end

          if (cnt_pass == 2'd2) begin
            // Pass 3 complete: proceed to PPU
            state <= ST_PPU;
          end else begin
            // Not last pass: advance to next pass
            state <= ST_NEXT_PASS;
          end
        end

        // ---------------------------------------------------------
        // NEXT_PASS: Advance to next pass for same (hout, ch, wblk)
        // Reload rows and weights for the next kernel-row group
        // ---------------------------------------------------------
        ST_NEXT_PASS: begin
          cnt_pass      <= cnt_pass + 2'd1;
          // Reset load counters for next pass
          load_w_idx    <= '0;
          load_row_idx  <= '0;
          load_rd_phase <= 1'b0;
          state         <= ST_LOAD_ROWS;
        end

        // ---------------------------------------------------------
        // PPU: Apply bias + requant + activation + clamp
        // ppu_out is combinationally computed from psum_buf
        // (psum_buf was updated in ST_PSUM_ACC on previous cycle)
        // ---------------------------------------------------------
        ST_PPU: begin
          state <= ST_WRITE;
        end

        // ---------------------------------------------------------
        // WRITE: Write LANES INT8 output values to output SRAM
        // Address = h_out * channels * num_wblk_out + ch * num_wblk_out + wblk
        // ---------------------------------------------------------
        ST_WRITE: begin
          ofm_wr_en <= 1'b1;
          ofm_wr_addr <= 24'(cnt_hout) * 24'(ch_val) * 24'(num_wblk_out)
                       + 24'(cnt_ch) * 24'(num_wblk_out)
                       + 24'(cnt_wblk);
          for (int l = 0; l < LANES; l++)
            ofm_wr_data[l] <= ppu_out[l];

          state <= ST_NEXT_WBLK;
        end

        // ---------------------------------------------------------
        // LOOP CONTROL: wblk -> ch -> hout
        // ---------------------------------------------------------
        ST_NEXT_WBLK: begin
          if (cnt_wblk == num_wblk_out - 1) begin
            cnt_wblk <= '0;
            state    <= ST_NEXT_CH;
          end else begin
            cnt_wblk <= cnt_wblk + 1'b1;
            // Restart from Pass 1 for new wblk
            cnt_pass      <= '0;
            load_w_idx    <= '0;
            load_row_idx  <= '0;
            load_rd_phase <= 1'b0;
            state         <= ST_LOAD_ROWS;
          end
        end

        ST_NEXT_CH: begin
          if (cnt_ch == ch_val - 1) begin
            cnt_ch <= '0;
            state  <= ST_NEXT_HOUT;
          end else begin
            cnt_ch   <= cnt_ch + 1'b1;
            cnt_wblk <= '0;
            cnt_pass <= '0;
            load_w_idx    <= '0;
            load_row_idx  <= '0;
            load_rd_phase <= 1'b0;
            state         <= ST_LOAD_ROWS;
          end
        end

        ST_NEXT_HOUT: begin
          if (cnt_hout == hout_val - 1) begin
            state <= ST_DONE;
          end else begin
            cnt_hout <= cnt_hout + 1'b1;
            cnt_ch   <= '0;
            cnt_wblk <= '0;
            cnt_pass <= '0;
            load_w_idx    <= '0;
            load_row_idx  <= '0;
            load_rd_phase <= 1'b0;
            state         <= ST_LOAD_ROWS;
          end
        end

        // ---------------------------------------------------------
        // DONE: Signal completion, return to idle
        // ---------------------------------------------------------
        ST_DONE: begin
          done <= 1'b1;
          busy <= 1'b0;
          state <= ST_IDLE;
        end

        default: state <= ST_IDLE;
      endcase
    end
  end

endmodule
