`timescale 1ns/1ps
// ============================================================================
// RS_DENSE_3x3 Primitive Engine (P0)
// Computes: Y[hout][wout][cout] = Σ_{kh,kw,cin} X[h*s+kh][w*s+kw][cin] * W[kh][kw][cin][cout]
//           + bias[cout] → requant → activation → clamp → INT8
//
// Uses 3 pe_unit instances (one per kernel row kh=0,1,2).
// PE accumulates across kw and cin iterations internally.
// Column-reduce sums across 3 kernel rows after all cin done.
// Inline PPU performs bias + requant + activation + clamp.
//
// Covers: L0(3→16,s2), L1(16→32,s2), L3(32→64,s2), L17(64→64,s2)
//         QC2f internal RS3 steps
// ============================================================================
module conv3x3_engine
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

  // ═══════════ Configuration ═══════════
  input  logic [9:0]  cfg_w_pad,     // Padded input width (Win + 2*pad)
  input  logic [9:0]  cfg_hout,      // Output height
  input  logic [9:0]  cfg_wout,      // Output width
  input  logic [8:0]  cfg_cin,       // Input channels
  input  logic [8:0]  cfg_cout,      // Output channels
  input  logic [1:0]  cfg_stride,    // 1 or 2
  input  act_mode_e   cfg_act_mode,  // Activation: NONE/SILU/RELU/CLAMP
  input  logic signed [7:0] cfg_zp_out,  // Output zero-point

  // ═══════════ Input Feature Map SRAM (LANES-wide read) ═══════════
  // Layout: [H_pad][Cin][num_w_words], word = LANES INT8 values
  // Address = h_pad * Cin * num_w_words + cin * num_w_words + w_word
  output logic [23:0]       ifm_rd_addr,
  output logic              ifm_rd_en,
  input  logic signed [7:0] ifm_rd_data [LANES],

  // ═══════════ Weight SRAM (byte read) ═══════════
  // Layout: [Cout][Cin][3][3]
  // Address = cout * Cin * 9 + cin * 9 + kh * 3 + kw
  output logic [23:0]       wgt_rd_addr,
  output logic              wgt_rd_en,
  input  logic signed [7:0] wgt_rd_data,

  // ═══════════ Bias & Quantization Params (per channel, preloaded) ═══════════
  input  logic signed [31:0] bias_arr  [MAX_COUT],
  input  logic signed [31:0] m_int_arr [MAX_COUT],
  input  logic [5:0]         shift_arr [MAX_COUT],

  // ═══════════ SiLU LUT ═══════════
  input  logic signed [7:0]  silu_lut [256],

  // ═══════════ Output Feature Map SRAM (LANES-wide write) ═══════════
  // Layout: [Hout][Cout][num_w_out_words]
  // Address = h_out * Cout * num_w_out_words + cout * num_w_out_words + w_word
  output logic [23:0]       ofm_wr_addr,
  output logic              ofm_wr_en,
  output logic signed [7:0] ofm_wr_data [LANES]
);

  // ═══════════════════════════════════════════════════════════════════
  // INTERNAL SIGNALS
  // ═══════════════════════════════════════════════════════════════════

  // Row buffers: 3 kernel rows × MAX_W_PAD spatial positions
  logic signed [7:0] row_buf [3][MAX_W_PAD];

  // Weight buffer: [kh][kw] for current (cout, cin)
  logic signed [7:0] wgt_buf [3][3];

  // PE control signals
  logic              pe_en;
  logic              pe_clear;
  logic signed [7:0] pe_x_in  [3][LANES];
  logic signed [7:0] pe_w_in  [3][LANES];
  logic signed [31:0] pe_psum [3][LANES];
  logic              pe_valid [3];

  // Column-reduced partial sum (sum of 3 PE rows)
  logic signed [31:0] reduced_psum [LANES];

  // PPU results
  logic signed [7:0]  ppu_out [LANES];

  // FSM
  eng_state_e state;

  // Loop counters
  logic [9:0] cnt_hout;
  logic [8:0] cnt_cout;
  logic [5:0] cnt_wblk;
  logic [8:0] cnt_cin;
  logic [1:0] cnt_kw;

  // Row loading counters
  logic [5:0] load_w_idx;     // Word index within row
  logic [1:0] load_kh_idx;    // Kernel row (0,1,2)
  logic       load_rd_phase;  // 0: issue read, 1: capture data

  // Weight loading counters
  logic [1:0] wgt_kh_idx, wgt_kw_idx;

  // Pipeline drain counter
  logic [2:0] drain_cnt;

  // Derived configuration
  logic [5:0] num_wblk_in;    // ceil(W_pad / LANES)
  logic [5:0] num_wblk_out;   // ceil(Wout / LANES)
  logic [8:0] cin_val, cout_val;
  logic [9:0] hout_val, wout_val, w_pad_val;
  logic [1:0] stride_val;

  // ═══════════════════════════════════════════════════════════════════
  // PE UNIT INSTANTIATION (3 units, one per kernel row)
  // ═══════════════════════════════════════════════════════════════════
  genvar kh;
  generate
    for (kh = 0; kh < 3; kh++) begin : gen_pe
      pe_unit #(.LANES(LANES)) u_pe (
        .clk       (clk),
        .rst_n     (rst_n),
        .en        (pe_en),
        .clear_psum(pe_clear),
        .mode      (PE_RS3),
        .x_in      (pe_x_in[kh]),
        .w_in      (pe_w_in[kh]),
        .psum_out  (pe_psum[kh]),
        .psum_valid(pe_valid[kh])
      );
    end
  endgenerate

  // ═══════════════════════════════════════════════════════════════════
  // PE INPUT MUX: Extract data from row buffers for current kw
  // For each lane l: PE[kh].x_in[l] = row_buf[kh][wblk*LANES*stride + l*stride + kw]
  // Weight broadcast: PE[kh].w_in[l] = wgt_buf[kh][kw] for all l
  // ═══════════════════════════════════════════════════════════════════
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
            pe_x_in[k][l] = 8'sd0;  // Zero for OOB (shouldn't happen with proper padding)

          pe_w_in[k][l] = wgt_buf[k][cnt_kw];  // Broadcast weight to all lanes
        end else begin
          pe_x_in[k][l] = 8'sd0;  // Drive zero when not computing
          pe_w_in[k][l] = 8'sd0;
        end
      end
    end
  end

  // ═══════════════════════════════════════════════════════════════════
  // COLUMN REDUCE: Sum 3 PE rows → 1 result per lane
  // ═══════════════════════════════════════════════════════════════════
  always_comb begin
    for (int l = 0; l < LANES; l++) begin
      reduced_psum[l] = pe_psum[0][l] + pe_psum[1][l] + pe_psum[2][l];
    end
  end

  // ═══════════════════════════════════════════════════════════════════
  // INLINE PPU: Bias + Requant + Activation + Clamp
  // ═══════════════════════════════════════════════════════════════════
  always_comb begin
    for (int l = 0; l < LANES; l++) begin
      automatic logic signed [31:0] biased;
      automatic logic signed [31:0] requanted;
      automatic logic signed [15:0] act_in;
      automatic logic signed [7:0]  act_val;
      automatic logic signed [15:0] final_val;

      // Stage 1: Bias add
      biased = reduced_psum[l] + bias_arr[cnt_cout];

      // Stage 2: Fixed-point requantization (× m_int >>> shift, half_up)
      requanted = requant_fixed(biased, m_int_arr[cnt_cout], shift_arr[cnt_cout]);

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

  // ═══════════════════════════════════════════════════════════════════
  // MAIN FSM
  // ═══════════════════════════════════════════════════════════════════
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
      cnt_cout      <= '0;
      cnt_wblk      <= '0;
      cnt_cin       <= '0;
      cnt_kw        <= '0;
      load_w_idx    <= '0;
      load_kh_idx   <= '0;
      load_rd_phase <= 1'b0;
      wgt_kh_idx    <= '0;
      wgt_kw_idx    <= '0;
      drain_cnt     <= '0;
    end else begin

      // Default: deassert write/read enables
      ifm_rd_en <= 1'b0;
      wgt_rd_en <= 1'b0;
      ofm_wr_en <= 1'b0;
      pe_en     <= 1'b0;
      pe_clear  <= 1'b0;

      case (state)
        // ─────────────────────────────────────────────
        ENG_IDLE: begin
          done <= 1'b0;
          if (start) begin
            state      <= ENG_LOAD_ROWS;
            busy       <= 1'b1;
            // Latch configuration
            cin_val    <= cfg_cin;
            cout_val   <= cfg_cout;
            hout_val   <= cfg_hout;
            wout_val   <= cfg_wout;
            w_pad_val  <= cfg_w_pad;
            stride_val <= cfg_stride;
            num_wblk_in  <= (cfg_w_pad + LANES - 1) / LANES;
            num_wblk_out <= (cfg_wout + LANES - 1) / LANES;
            // Reset loop counters
            cnt_hout     <= '0;
            cnt_cout     <= '0;
            cnt_wblk     <= '0;
            cnt_cin      <= '0;
            // Start loading first rows
            load_w_idx   <= '0;
            load_kh_idx  <= '0;
            load_rd_phase <= 1'b0;
          end
        end

        // ─────────────────────────────────────────────
        // LOAD_ROWS: Load 3 input rows into row_buf for current (h_in, cin)
        // Read LANES bytes per cycle from input SRAM
        // ─────────────────────────────────────────────
        ENG_LOAD_ROWS: begin
          if (!load_rd_phase) begin
            // Phase 0: Issue SRAM read
            automatic logic [9:0]  h_in;
            automatic logic [23:0] addr;
            h_in = cnt_hout * stride_val + {8'b0, load_kh_idx};

            // Address = h_in * cin_val * num_wblk_in + cnt_cin * num_wblk_in + load_w_idx
            addr = 24'(h_in) * 24'(cin_val) * 24'(num_wblk_in)
                 + 24'(cnt_cin) * 24'(num_wblk_in)
                 + 24'(load_w_idx);

            ifm_rd_addr   <= addr;
            ifm_rd_en     <= 1'b1;
            load_rd_phase <= 1'b1;  // Next cycle: capture data
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
                // All 3 rows loaded → move to weight loading
                load_kh_idx <= '0;
                state <= ENG_LOAD_WGT;
                wgt_kh_idx <= '0;
                wgt_kw_idx <= '0;
              end else begin
                load_kh_idx <= load_kh_idx + 1'b1;
              end
            end else begin
              load_w_idx <= load_w_idx + 1'b1;
            end
          end
        end

        // ─────────────────────────────────────────────
        // LOAD_WGT: Load 9 weights for current (cout, cin, kh, kw)
        // ─────────────────────────────────────────────
        ENG_LOAD_WGT: begin
          // Issue weight read
          wgt_rd_addr <= 24'(cnt_cout) * 24'(cin_val) * 24'd9
                       + 24'(cnt_cin) * 24'd9
                       + 24'(wgt_kh_idx) * 24'd3
                       + 24'(wgt_kw_idx);
          wgt_rd_en   <= 1'b1;

          // Store previous read result (1-cycle latency, skip first)
          if (wgt_kh_idx != 0 || wgt_kw_idx != 0) begin
            // Store previous weight
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
              // All 9 weights issued; need 1 more cycle to capture last
              state <= ENG_COMPUTE;
              cnt_kw <= '0;
            end else begin
              wgt_kh_idx <= wgt_kh_idx + 1'b1;
            end
          end else begin
            wgt_kw_idx <= wgt_kw_idx + 1'b1;
          end
        end

        // ─────────────────────────────────────────────
        // COMPUTE: Feed PE with 3 kw values per cin iteration
        // PE accumulates across kw and cin
        // ─────────────────────────────────────────────
        ENG_COMPUTE: begin
          // Capture last weight from LOAD_WGT (1-cycle latency)
          if (cnt_kw == 0) begin
            wgt_buf[2][2] <= wgt_rd_data;
          end

          if (cnt_kw == 2'd0 && cnt_cin == '0) begin
            // Capture last weight then start PE
            if (cnt_kw == 0) begin
              // First cycle after weight load: wait for last weight capture
              // Actually feed PE starting next cycle
              pe_en    <= 1'b1;
              pe_clear <= 1'b1;  // Clear accumulator for new output
              cnt_kw   <= cnt_kw + 1'b1;
            end
          end else begin
            pe_en    <= 1'b1;
            pe_clear <= (cnt_cin == '0 && cnt_kw == 2'd0) ? 1'b1 : 1'b0;

            if (cnt_kw == 2'd2) begin
              // Last kw for this cin
              cnt_kw <= '0;
              state  <= ENG_NEXT_CIN;
            end else begin
              cnt_kw <= cnt_kw + 1'b1;
            end
          end
        end

        // ─────────────────────────────────────────────
        // NEXT_CIN: Advance cin counter
        // ─────────────────────────────────────────────
        ENG_NEXT_CIN: begin
          if (cnt_cin == cin_val - 1) begin
            // All cin done → drain PE pipeline
            cnt_cin   <= '0;
            state     <= ENG_DRAIN;
            drain_cnt <= '0;
          end else begin
            cnt_cin <= cnt_cin + 1'b1;
            state   <= ENG_LOAD_ROWS;
            // Reset load counters for next cin
            load_w_idx    <= '0;
            load_kh_idx   <= '0;
            load_rd_phase <= 1'b0;
          end
        end

        // ─────────────────────────────────────────────
        // DRAIN: Wait for PE pipeline to fully drain
        // Need DSP_LATENCY cycles after last en=1
        // ─────────────────────────────────────────────
        ENG_DRAIN: begin
          drain_cnt <= drain_cnt + 1'b1;
          if (drain_cnt == DSP_LATENCY[2:0]) begin
            state <= ENG_PPU;
          end
        end

        // ─────────────────────────────────────────────
        // PPU: Read PE psum, column reduce, bias+requant+act+clamp
        // Result available combinationally in ppu_out
        // ─────────────────────────────────────────────
        ENG_PPU: begin
          state <= ENG_WRITE;
        end

        // ─────────────────────────────────────────────
        // WRITE: Write LANES output bytes to output SRAM
        // ─────────────────────────────────────────────
        ENG_WRITE: begin
          ofm_wr_en <= 1'b1;
          // Address = h_out * cout_val * num_wblk_out + cout * num_wblk_out + wblk
          ofm_wr_addr <= 24'(cnt_hout) * 24'(cout_val) * 24'(num_wblk_out)
                       + 24'(cnt_cout) * 24'(num_wblk_out)
                       + 24'(cnt_wblk);
          for (int l = 0; l < LANES; l++)
            ofm_wr_data[l] <= ppu_out[l];

          state <= ENG_NEXT_WBLK;
        end

        // ─────────────────────────────────────────────
        // LOOP CONTROL: Advance wblk → cout → hout
        // ─────────────────────────────────────────────
        ENG_NEXT_WBLK: begin
          if (cnt_wblk == num_wblk_out - 1) begin
            cnt_wblk <= '0;
            state    <= ENG_NEXT_COUT;
          end else begin
            cnt_wblk <= cnt_wblk + 1'b1;
            // Reload rows for new wblk, restart cin from 0
            cnt_cin       <= '0;
            state         <= ENG_LOAD_ROWS;
            load_w_idx    <= '0;
            load_kh_idx   <= '0;
            load_rd_phase <= 1'b0;
          end
        end

        ENG_NEXT_COUT: begin
          if (cnt_cout == cout_val - 1) begin
            cnt_cout <= '0;
            state    <= ENG_NEXT_HOUT;
          end else begin
            cnt_cout <= cnt_cout + 1'b1;
            cnt_wblk <= '0;
            cnt_cin  <= '0;
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
            cnt_cout <= '0;
            cnt_wblk <= '0;
            cnt_cin  <= '0;
            state    <= ENG_LOAD_ROWS;
            load_w_idx    <= '0;
            load_kh_idx   <= '0;
            load_rd_phase <= 1'b0;
          end
        end

        // ─────────────────────────────────────────────
        ENG_DONE: begin
          done <= 1'b1;
          busy <= 1'b0;
          state <= ENG_IDLE;
        end

        default: state <= ENG_IDLE;
      endcase
    end
  end

endmodule
