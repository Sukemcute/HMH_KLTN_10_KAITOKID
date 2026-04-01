`timescale 1ns/1ps
// ============================================================================
// DW_3x3 Depthwise Convolution Primitive Engine (P2)
// Computes: Y[hout][wout][c] = Sigma_{kh,kw} X[h*s+kh][w*s+kw][c] * W[kh][kw][c]
//           + bias[c] -> requant -> activation -> clamp -> INT8
//
// Key differences from RS_DENSE_3x3 (conv3x3_engine):
//   - Depthwise: groups = Cin, Cout = Cin (each channel independent)
//   - Weight shape: [C][3][3] -- per-channel, NOT [Cout][Cin][3][3]
//   - Per-channel bias and quantization parameters
//   - No cross-channel accumulation (no cin loop; cin=1 per output channel)
//   - Each PE lane processes one spatial output position for ONE channel
//   - LANES output positions computed per cycle for the same channel
//   - Only 3 kw cycles needed (no cin accumulation across multiple cin values)
//
// Architecture:
//   - 3 pe_unit instances (kh=0,1,2), mode PE_DW3
//   - Row buffers: 3 x MAX_W_PAD for spatial window
//   - For each channel c:
//       1. Load 3 input rows for channel c into row_buf
//       2. Load 9 weights (3x3) for channel c
//       3. Feed PE: 3 kw cycles (kh handled by 3 PEs in parallel)
//       4. Drain pipeline -> column reduce (sum 3 kh PEs) -> PPU
//       5. PPU: bias[c] + requant(m_int[c], shift[c]) + act + clamp
//       6. Write output for this wblk of channel c
//   - Outer loops: hout -> ch -> wblk
//
// Covers: SCDown depthwise 3x3, other DW3x3 layers in YOLOv10n
// ============================================================================
module dwconv3x3_engine
  import yolo_accel_pkg::*;
#(
  parameter int LANES     = 32,
  parameter int MAX_W_PAD = 672    // Max padded input row width
)(
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  output logic        done,
  output logic        busy,

  // =========== Configuration ===========
  input  logic [9:0]  cfg_w_pad,     // Padded input width (Win + 2*pad)
  input  logic [9:0]  cfg_hout,      // Output height
  input  logic [9:0]  cfg_wout,      // Output width
  input  logic [8:0]  cfg_channels,  // Number of channels (= Cin = Cout for DW)
  input  logic [1:0]  cfg_stride,    // 1 or 2
  input  act_mode_e   cfg_act_mode,  // Activation: NONE/SILU/RELU/CLAMP
  input  logic signed [7:0] cfg_zp_out,  // Output zero-point

  // =========== Input Feature Map SRAM (LANES-wide read) ===========
  // Layout: [H_pad][C][num_w_words], word = LANES INT8 values
  // Address = h_pad * C * num_w_words + ch * num_w_words + w_word
  output logic [23:0]       ifm_rd_addr,
  output logic              ifm_rd_en,
  input  logic signed [7:0] ifm_rd_data [LANES],

  // =========== Weight SRAM (byte read) ===========
  // Layout: [C][3][3]  -- per-channel depthwise weights
  // Address = ch * 9 + kh * 3 + kw
  output logic [23:0]       wgt_rd_addr,
  output logic              wgt_rd_en,
  input  logic signed [7:0] wgt_rd_data,

  // =========== Bias & Quantization Params (per channel, preloaded) ===========
  input  logic signed [31:0] bias_arr  [MAX_CIN],
  input  logic signed [31:0] m_int_arr [MAX_CIN],
  input  logic [5:0]         shift_arr [MAX_CIN],

  // =========== SiLU LUT ===========
  input  logic signed [7:0]  silu_lut [256],

  // =========== Output Feature Map SRAM (LANES-wide write) ===========
  // Layout: [Hout][C][num_w_out_words]
  // Address = h_out * C * num_w_out_words + ch * num_w_out_words + w_word
  output logic [23:0]       ofm_wr_addr,
  output logic              ofm_wr_en,
  output logic signed [7:0] ofm_wr_data [LANES]
);

  // =================================================================
  // INTERNAL SIGNALS
  // =================================================================

  // Row buffers: 3 kernel rows x MAX_W_PAD spatial positions
  // Holds one channel's worth of input data for the current 3 rows
  logic signed [7:0] row_buf [3][MAX_W_PAD];

  // Weight buffer: [kh][kw] for current channel c
  logic signed [7:0] wgt_buf [3][3];

  // PE control signals
  logic              pe_en;
  logic              pe_clear;
  logic signed [7:0] pe_x_in  [3][LANES];
  logic signed [7:0] pe_w_in  [3][LANES];
  logic signed [31:0] pe_psum [3][LANES];
  logic              pe_valid [3];

  // Column-reduced partial sum (sum of 3 PE rows for kh=0,1,2)
  logic signed [31:0] reduced_psum [LANES];

  // PPU results
  logic signed [7:0]  ppu_out [LANES];

  // FSM
  eng_state_e state;

  // Loop counters
  logic [9:0] cnt_hout;       // Output row index
  logic [8:0] cnt_ch;         // Channel index (single loop, since cin=cout for DW)
  logic [5:0] cnt_wblk;       // Output width block index
  logic [1:0] cnt_kw;         // Kernel column (0,1,2) during compute

  // Row loading counters
  logic [5:0] load_w_idx;     // Word index within row
  logic [1:0] load_kh_idx;    // Kernel row (0,1,2)
  logic       load_rd_phase;  // 0: issue read, 1: capture data

  // Weight loading counters
  logic [1:0] wgt_kh_idx, wgt_kw_idx;

  // Pipeline drain counter
  logic [2:0] drain_cnt;

  // Derived configuration (latched on start)
  logic [5:0] num_wblk_in;    // ceil(W_pad / LANES)
  logic [5:0] num_wblk_out;   // ceil(Wout / LANES)
  logic [8:0] ch_val;         // Number of channels
  logic [9:0] hout_val, wout_val, w_pad_val;
  logic [1:0] stride_val;

  // =================================================================
  // PE UNIT INSTANTIATION (3 units, one per kernel row kh=0,1,2)
  // Each PE runs in PE_DW3 mode for depthwise convolution.
  // =================================================================
  genvar kh;
  generate
    for (kh = 0; kh < 3; kh++) begin : gen_pe
      pe_unit #(.LANES(LANES)) u_pe (
        .clk       (clk),
        .rst_n     (rst_n),
        .en        (pe_en),
        .clear_psum(pe_clear),
        .mode      (PE_DW3),
        .x_in      (pe_x_in[kh]),
        .w_in      (pe_w_in[kh]),
        .psum_out  (pe_psum[kh]),
        .psum_valid(pe_valid[kh])
      );
    end
  endgenerate

  // =================================================================
  // PE INPUT MUX: Extract data from row buffers for current kw
  //
  // Depthwise: all LANES lanes process the SAME channel but different
  // spatial (width) positions.
  //   PE[kh].x_in[l] = row_buf[kh][ wblk*LANES*stride + l*stride + kw ]
  //   PE[kh].w_in[l] = wgt_buf[kh][kw]  (broadcast same weight to all lanes)
  // =================================================================
  always_comb begin
    for (int k = 0; k < 3; k++) begin
      for (int l = 0; l < LANES; l++) begin
        if (state == ENG_COMPUTE) begin
          automatic int base_w;
          base_w = int'(cnt_wblk) * LANES * int'(stride_val)
                 + l * int'(stride_val)
                 + int'(cnt_kw);
          if (base_w >= 0 && base_w < MAX_W_PAD)
            pe_x_in[k][l] = row_buf[k][base_w];
          else
            pe_x_in[k][l] = 8'sd0;  // Zero for OOB (safety guard)

          // Broadcast the same per-channel weight to all spatial lanes
          pe_w_in[k][l] = wgt_buf[k][cnt_kw];
        end else begin
          pe_x_in[k][l] = 8'sd0;
          pe_w_in[k][l] = 8'sd0;
        end
      end
    end
  end

  // =================================================================
  // COLUMN REDUCE: Sum 3 PE rows (kh=0,1,2) -> 1 result per lane
  // For depthwise 3x3, this is the full accumulation since there is
  // no cin dimension to iterate over (each channel is independent).
  // =================================================================
  always_comb begin
    for (int l = 0; l < LANES; l++) begin
      reduced_psum[l] = pe_psum[0][l] + pe_psum[1][l] + pe_psum[2][l];
    end
  end

  // =================================================================
  // INLINE PPU: Bias + Requant + Activation + Clamp
  // Uses per-channel parameters indexed by cnt_ch.
  // =================================================================
  always_comb begin
    for (int l = 0; l < LANES; l++) begin
      automatic logic signed [31:0] biased;
      automatic logic signed [31:0] requanted;
      automatic logic signed [15:0] act_in;
      automatic logic signed [7:0]  act_val;
      automatic logic signed [15:0] final_val;

      // Stage 1: Per-channel bias add
      biased = reduced_psum[l] + bias_arr[cnt_ch];

      // Stage 2: Per-channel fixed-point requantization (x m_int >>> shift)
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

      // Stage 4: Add output zero-point and final clamp to INT8
      final_val = 16'(act_val) + 16'(cfg_zp_out);
      if (final_val > 16'sd127)
        ppu_out[l] = 8'sd127;
      else if (final_val < -16'sd128)
        ppu_out[l] = -8'sd128;
      else
        ppu_out[l] = final_val[7:0];
    end
  end

  // =================================================================
  // MAIN FSM
  //
  // Loop order (outermost first):
  //   hout -> ch -> wblk -> [load_rows -> load_wgt -> compute(3 kw) -> drain -> ppu -> write]
  //
  // Unlike conv3x3_engine, there is NO inner cin loop because depthwise
  // convolution has groups=C (each channel is independent, cin=1 per group).
  // The "channel" loop here replaces the cout x cin double loop.
  // =================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state         <= ENG_IDLE;
      done          <= 1'b0;
      busy          <= 1'b0;
      pe_en         <= 1'b0;
      pe_clear      <= 1'b0;
      ifm_rd_en     <= 1'b0;
      wgt_rd_en     <= 1'b0;
      ofm_wr_en     <= 1'b0;
      cnt_hout      <= '0;
      cnt_ch        <= '0;
      cnt_wblk      <= '0;
      cnt_kw        <= '0;
      load_w_idx    <= '0;
      load_kh_idx   <= '0;
      load_rd_phase <= 1'b0;
      wgt_kh_idx    <= '0;
      wgt_kw_idx    <= '0;
      drain_cnt     <= '0;
    end else begin

      // Default: deassert enables each cycle
      ifm_rd_en <= 1'b0;
      wgt_rd_en <= 1'b0;
      ofm_wr_en <= 1'b0;
      pe_en     <= 1'b0;
      pe_clear  <= 1'b0;

      case (state)
        // ---------------------------------------------------------
        // IDLE: Wait for start pulse, latch configuration
        // ---------------------------------------------------------
        ENG_IDLE: begin
          done <= 1'b0;
          if (start) begin
            state      <= ENG_LOAD_ROWS;
            busy       <= 1'b1;
            // Latch configuration
            ch_val     <= cfg_channels;
            hout_val   <= cfg_hout;
            wout_val   <= cfg_wout;
            w_pad_val  <= cfg_w_pad;
            stride_val <= cfg_stride;
            num_wblk_in  <= (cfg_w_pad + LANES - 1) / LANES;
            num_wblk_out <= (cfg_wout  + LANES - 1) / LANES;
            // Reset all loop counters
            cnt_hout      <= '0;
            cnt_ch        <= '0;
            cnt_wblk      <= '0;
            // Start loading first rows
            load_w_idx    <= '0;
            load_kh_idx   <= '0;
            load_rd_phase <= 1'b0;
          end
        end

        // ---------------------------------------------------------
        // LOAD_ROWS: Load 3 input rows into row_buf for current
        // channel (cnt_ch) and output row (cnt_hout).
        // Reads LANES bytes per cycle from input SRAM.
        //
        // For depthwise: we load rows for a single channel c.
        // Address = h_in * C * num_wblk_in + ch * num_wblk_in + w_word
        // where h_in = cnt_hout * stride + kh
        // ---------------------------------------------------------
        ENG_LOAD_ROWS: begin
          if (!load_rd_phase) begin
            // Phase 0: Issue SRAM read request
            automatic logic [9:0]  h_in;
            automatic logic [23:0] addr;
            h_in = cnt_hout * stride_val + {8'b0, load_kh_idx};

            addr = 24'(h_in) * 24'(ch_val) * 24'(num_wblk_in)
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
                row_buf[load_kh_idx][widx] <= ifm_rd_data[l];
            end
            load_rd_phase <= 1'b0;

            // Advance word counter
            if (load_w_idx == num_wblk_in - 1) begin
              load_w_idx <= '0;
              if (load_kh_idx == 2'd2) begin
                // All 3 kernel rows loaded -> proceed to weight load
                load_kh_idx <= '0;
                state       <= ENG_LOAD_WGT;
                wgt_kh_idx  <= '0;
                wgt_kw_idx  <= '0;
              end else begin
                load_kh_idx <= load_kh_idx + 1'b1;
              end
            end else begin
              load_w_idx <= load_w_idx + 1'b1;
            end
          end
        end

        // ---------------------------------------------------------
        // LOAD_WGT: Load 9 per-channel weights for current channel.
        // Weight SRAM layout: [C][3][3]
        // Address = ch * 9 + kh * 3 + kw
        //
        // Unlike conv3x3_engine (which indexes [cout][cin][kh][kw]),
        // depthwise has no cin dimension in weights.
        // ---------------------------------------------------------
        ENG_LOAD_WGT: begin
          // Issue weight SRAM read for current (kh, kw)
          wgt_rd_addr <= 24'(cnt_ch) * 24'd9
                       + 24'(wgt_kh_idx) * 24'd3
                       + 24'(wgt_kw_idx);
          wgt_rd_en   <= 1'b1;

          // Store previous read result (1-cycle SRAM latency, skip first beat)
          if (wgt_kh_idx != 0 || wgt_kw_idx != 0) begin
            automatic logic [1:0] prev_kh, prev_kw;
            if (wgt_kw_idx == 0) begin
              prev_kh = wgt_kh_idx - 1'b1;
              prev_kw = 2'd2;
            end else begin
              prev_kh = wgt_kh_idx;
              prev_kw = wgt_kw_idx - 1'b1;
            end
            wgt_buf[prev_kh][prev_kw] <= wgt_rd_data;
          end

          // Advance weight counter
          if (wgt_kw_idx == 2'd2) begin
            wgt_kw_idx <= '0;
            if (wgt_kh_idx == 2'd2) begin
              // All 9 weights issued; last one captured in COMPUTE entry
              state  <= ENG_COMPUTE;
              cnt_kw <= '0;
            end else begin
              wgt_kh_idx <= wgt_kh_idx + 1'b1;
            end
          end else begin
            wgt_kw_idx <= wgt_kw_idx + 1'b1;
          end
        end

        // ---------------------------------------------------------
        // COMPUTE: Feed PE with 3 kw cycles.
        //
        // Depthwise difference: there is NO cin loop. We only need
        // 3 clock cycles (kw=0,1,2) to accumulate the full 3x3
        // kernel for a single channel. The 3 PE rows handle kh
        // in parallel, so the total MAC count per output position
        // is 3 kw x 1 = 3 cycles (vs 3 kw x cin for dense conv).
        // ---------------------------------------------------------
        ENG_COMPUTE: begin
          // Capture last weight from LOAD_WGT pipeline (1-cycle latency)
          if (cnt_kw == 2'd0) begin
            wgt_buf[2][2] <= wgt_rd_data;
          end

          pe_en    <= 1'b1;
          pe_clear <= (cnt_kw == 2'd0) ? 1'b1 : 1'b0;  // Clear accum on first kw

          if (cnt_kw == 2'd2) begin
            // Last kw done -> drain PE pipeline
            cnt_kw    <= '0;
            state     <= ENG_DRAIN;
            drain_cnt <= '0;
          end else begin
            cnt_kw <= cnt_kw + 1'b1;
          end
        end

        // ---------------------------------------------------------
        // DRAIN: Wait for PE pipeline to fully drain.
        // Need DSP_LATENCY cycles after last en=1 assertion.
        // ---------------------------------------------------------
        ENG_DRAIN: begin
          drain_cnt <= drain_cnt + 1'b1;
          if (drain_cnt == DSP_LATENCY[2:0]) begin
            state <= ENG_PPU;
          end
        end

        // ---------------------------------------------------------
        // PPU: Column reduce + bias + requant + activation + clamp
        // Result available combinationally in ppu_out (driven above).
        // Transition to WRITE on next cycle.
        // ---------------------------------------------------------
        ENG_PPU: begin
          state <= ENG_WRITE;
        end

        // ---------------------------------------------------------
        // WRITE: Write LANES output bytes to output SRAM.
        // Address = h_out * C * num_wblk_out + ch * num_wblk_out + wblk
        // ---------------------------------------------------------
        ENG_WRITE: begin
          ofm_wr_en <= 1'b1;
          ofm_wr_addr <= 24'(cnt_hout) * 24'(ch_val) * 24'(num_wblk_out)
                       + 24'(cnt_ch) * 24'(num_wblk_out)
                       + 24'(cnt_wblk);
          for (int l = 0; l < LANES; l++)
            ofm_wr_data[l] <= ppu_out[l];

          state <= ENG_NEXT_WBLK;
        end

        // ---------------------------------------------------------
        // LOOP CONTROL: wblk -> ch -> hout
        //
        // Depthwise loop structure (no cout/cin split):
        //   for hout in [0, Hout):
        //     for ch in [0, C):
        //       for wblk in [0, num_wblk_out):
        //         load_rows -> load_wgt -> compute -> drain -> ppu -> write
        //
        // Note: rows and weights are reloaded for each wblk of the
        // same (hout, ch). A future optimization could cache rows
        // and weights across wblk iterations when they share the
        // same channel and row set.
        // ---------------------------------------------------------
        ENG_NEXT_WBLK: begin
          if (cnt_wblk == num_wblk_out - 1) begin
            cnt_wblk <= '0;
            state    <= ENG_NEXT_COUT;  // Reusing ENG_NEXT_COUT for channel advance
          end else begin
            cnt_wblk <= cnt_wblk + 1'b1;
            // Same channel, next spatial block -> reload rows and weights
            state         <= ENG_LOAD_ROWS;
            load_w_idx    <= '0;
            load_kh_idx   <= '0;
            load_rd_phase <= 1'b0;
          end
        end

        // ---------------------------------------------------------
        // ENG_NEXT_COUT: Advance channel counter.
        // In depthwise conv, this replaces the separate cout/cin
        // loops of dense conv3x3. Each channel c produces exactly
        // one output channel c.
        // ---------------------------------------------------------
        ENG_NEXT_COUT: begin
          if (cnt_ch == ch_val - 1) begin
            cnt_ch <= '0;
            state  <= ENG_NEXT_HOUT;
          end else begin
            cnt_ch   <= cnt_ch + 1'b1;
            cnt_wblk <= '0;
            state    <= ENG_LOAD_ROWS;
            load_w_idx    <= '0;
            load_kh_idx   <= '0;
            load_rd_phase <= 1'b0;
          end
        end

        ENG_NEXT_HOUT: begin
          if (cnt_hout == hout_val - 1) begin
            state <= ENG_DONE;
          end else begin
            cnt_hout <= cnt_hout + 1'b1;
            cnt_ch   <= '0;
            cnt_wblk <= '0;
            state    <= ENG_LOAD_ROWS;
            load_w_idx    <= '0;
            load_kh_idx   <= '0;
            load_rd_phase <= 1'b0;
          end
        end

        // ---------------------------------------------------------
        // DONE: Signal completion, return to idle
        // ---------------------------------------------------------
        ENG_DONE: begin
          done  <= 1'b1;
          busy  <= 1'b0;
          state <= ENG_IDLE;
        end

        default: state <= ENG_IDLE;
      endcase
    end
  end

endmodule
