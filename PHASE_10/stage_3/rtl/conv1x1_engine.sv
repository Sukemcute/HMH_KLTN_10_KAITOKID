`timescale 1ns/1ps
// ============================================================================
// OS_1x1 Primitive Engine (P1) — Output-Stationary 1x1 Convolution
// Computes: Y[hout][wout][cout] = Σ_{cin} X[h][w][cin] * W[cin][cout]
//           + bias[cout] → requant → activation → clamp → INT8
//
// Key differences from conv3x3_engine:
//   - Kernel 1×1: no spatial window, no row buffers
//   - Only 1 pe_unit instance (no kernel-row parallelism)
//   - PE operates in PE_OS1 mode: weight broadcast to all LANES
//   - No padding, stride always 1
//   - Input read directly from SRAM (LANES bytes per read)
//   - For each cin: 1 cycle to feed PE (vs 3 kw cycles in 3×3)
//
// Covers: QC2f cv1/cv2, SCDown 1×1 path, SPPF 1×1
// ============================================================================
module conv1x1_engine
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
  input  logic [9:0]  cfg_h,           // Input / output height (stride=1, no pad)
  input  logic [9:0]  cfg_w,           // Input / output width
  input  logic [8:0]  cfg_cin,         // Input channels
  input  logic [8:0]  cfg_cout,        // Output channels
  input  act_mode_e   cfg_act_mode,    // Activation: NONE / SILU / RELU / CLAMP
  input  logic signed [7:0] cfg_zp_out, // Output zero-point

  // ═══════════ Input Feature Map SRAM (LANES-wide read) ═══════════
  // Layout: [H][Cin][num_wblk], word = LANES INT8 values
  // Address = h * Cin * num_wblk + cin * num_wblk + wblk
  output logic [23:0]       ifm_rd_addr,
  output logic              ifm_rd_en,
  input  logic signed [7:0] ifm_rd_data [LANES],

  // ═══════════ Weight SRAM (byte read) ═══════════
  // Layout: [Cout][Cin]
  // Address = cout * Cin + cin
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
  // Layout: [Hout][Cout][num_wblk]
  // Address = h_out * Cout * num_wblk + cout * num_wblk + wblk
  output logic [23:0]       ofm_wr_addr,
  output logic              ofm_wr_en,
  output logic signed [7:0] ofm_wr_data [LANES]
);

  // ═══════════════════════════════════════════════════════════════════
  // INTERNAL SIGNALS
  // ═══════════════════════════════════════════════════════════════════

  // PE control signals — single PE, LANES wide
  logic              pe_en;
  logic              pe_clear;
  logic signed [7:0] pe_x_in [LANES];
  logic signed [7:0] pe_w_in [LANES];
  logic signed [31:0] pe_psum [LANES];
  logic              pe_valid;

  // Weight latch: holds 1 weight byte for the current (cout, cin)
  logic signed [7:0] wgt_lat;

  // PPU results (combinational)
  logic signed [7:0] ppu_out [LANES];

  // FSM
  eng_state_e state;

  // Loop counters
  logic [9:0] cnt_hout;       // Output row     [0 .. h-1]
  logic [8:0] cnt_cout;       // Output channel [0 .. cout-1]
  logic [5:0] cnt_wblk;       // Width block    [0 .. num_wblk-1]
  logic [8:0] cnt_cin;        // Input channel  [0 .. cin-1]

  // SRAM read pipeline tracking
  logic       ifm_rd_pending; // 1 = read issued, awaiting data next cycle
  logic       wgt_rd_pending; // 1 = weight read issued, awaiting data next cycle

  // Pipeline drain counter
  logic [2:0] drain_cnt;

  // Latched configuration
  logic [9:0] h_val, w_val;
  logic [8:0] cin_val, cout_val;
  logic [5:0] num_wblk;       // ceil(W / LANES)

  // ═══════════════════════════════════════════════════════════════════
  // PE UNIT INSTANTIATION — single unit in OS1 mode
  // Weight is broadcast to all LANES inside pe_unit when mode=PE_OS1:
  //   w_sel[l] = w_in[0] for all l
  // ═══════════════════════════════════════════════════════════════════
  pe_unit #(.LANES(LANES)) u_pe (
    .clk       (clk),
    .rst_n     (rst_n),
    .en        (pe_en),
    .clear_psum(pe_clear),
    .mode      (PE_OS1),
    .x_in      (pe_x_in),
    .w_in      (pe_w_in),
    .psum_out  (pe_psum),
    .psum_valid(pe_valid)
  );

  // ═══════════════════════════════════════════════════════════════════
  // PE INPUT MUX
  // In ENG_COMPUTE:  x_in = SRAM data, w_in[0] = weight byte
  // Otherwise:       drive zeros to avoid garbage accumulation
  // ═══════════════════════════════════════════════════════════════════
  always_comb begin
    for (int l = 0; l < LANES; l++) begin
      if (state == ENG_COMPUTE && pe_en) begin
        pe_x_in[l] = ifm_rd_data[l];  // LANES input activations
        pe_w_in[l] = wgt_lat;         // Broadcast weight (PE_OS1 uses w_in[0])
      end else begin
        pe_x_in[l] = 8'sd0;           // Zero when not computing
        pe_w_in[l] = 8'sd0;
      end
    end
  end

  // ═══════════════════════════════════════════════════════════════════
  // INLINE PPU: Bias + Requant + Activation + Clamp + ZP
  // Purely combinational; result consumed during ENG_PPU / ENG_WRITE
  // ═══════════════════════════════════════════════════════════════════
  always_comb begin
    for (int l = 0; l < LANES; l++) begin
      automatic logic signed [31:0] biased;
      automatic logic signed [31:0] requanted;
      automatic logic signed [15:0] act_in;
      automatic logic signed [7:0]  act_val;
      automatic logic signed [15:0] final_val;

      // Stage 1: Bias add
      biased = pe_psum[l] + bias_arr[cnt_cout];

      // Stage 2: Fixed-point requantization (× m_int >>> shift, half-up)
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
  //
  // Loop nest (outermost → innermost):
  //   for hout  in [0 .. H-1]
  //     for wblk in [0 .. num_wblk-1]
  //       for cout in [0 .. Cout-1]
  //         for cin in [0 .. Cin-1]        ← ENG_COMPUTE (1 cycle each)
  //         drain PE pipeline              ← ENG_DRAIN
  //         PPU                            ← ENG_PPU
  //         write output                   ← ENG_WRITE
  //
  // Inside the cin loop:
  //   Cycle N-1: issue IFM + WGT reads
  //   Cycle N  : data arrives, feed PE (en=1)
  //
  // ═══════════════════════════════════════════════════════════════════
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state          <= ENG_IDLE;
      done           <= 1'b0;
      busy           <= 1'b0;
      pe_en          <= 1'b0;
      pe_clear       <= 1'b0;
      ifm_rd_en      <= 1'b0;
      wgt_rd_en      <= 1'b0;
      ofm_wr_en      <= 1'b0;
      cnt_hout       <= '0;
      cnt_cout       <= '0;
      cnt_wblk       <= '0;
      cnt_cin        <= '0;
      drain_cnt      <= '0;
      ifm_rd_pending <= 1'b0;
      wgt_rd_pending <= 1'b0;
      wgt_lat        <= 8'sd0;
      h_val          <= '0;
      w_val          <= '0;
      cin_val        <= '0;
      cout_val       <= '0;
      num_wblk       <= '0;
    end else begin

      // ── Defaults: deassert one-shot signals ──
      ifm_rd_en  <= 1'b0;
      wgt_rd_en  <= 1'b0;
      ofm_wr_en  <= 1'b0;
      pe_en      <= 1'b0;
      pe_clear   <= 1'b0;

      case (state)

        // ─────────────────────────────────────────────
        // IDLE: Wait for start pulse
        // ─────────────────────────────────────────────
        ENG_IDLE: begin
          done <= 1'b0;
          if (start) begin
            state    <= ENG_LOAD_WGT;
            busy     <= 1'b1;

            // Latch configuration
            h_val    <= cfg_h;
            w_val    <= cfg_w;
            cin_val  <= cfg_cin;
            cout_val <= cfg_cout;
            num_wblk <= (cfg_w + LANES[9:0] - 10'd1) / LANES[9:0];

            // Reset all loop counters
            cnt_hout <= '0;
            cnt_cout <= '0;
            cnt_wblk <= '0;
            cnt_cin  <= '0;

            ifm_rd_pending <= 1'b0;
            wgt_rd_pending <= 1'b0;
          end
        end

        // ─────────────────────────────────────────────
        // LOAD_WGT: Issue weight read for current (cout, cin)
        // Weight SRAM has 1-cycle read latency.
        // Address = cout * Cin + cin
        // Also issue IFM read for the first cin beat.
        // ─────────────────────────────────────────────
        ENG_LOAD_WGT: begin
          // Issue weight read
          wgt_rd_addr <= 24'(cnt_cout) * 24'(cin_val) + 24'(cnt_cin);
          wgt_rd_en   <= 1'b1;

          // Issue IFM read in parallel
          // IFM address = h * Cin * num_wblk + cin * num_wblk + wblk
          ifm_rd_addr <= 24'(cnt_hout) * 24'(cin_val) * 24'(num_wblk)
                       + 24'(cnt_cin) * 24'(num_wblk)
                       + 24'(cnt_wblk);
          ifm_rd_en   <= 1'b1;

          ifm_rd_pending <= 1'b1;
          wgt_rd_pending <= 1'b1;

          state <= ENG_COMPUTE;
        end

        // ─────────────────────────────────────────────
        // COMPUTE: Feed PE one (cin) at a time
        //
        // On entry from LOAD_WGT:
        //   - IFM data and WGT data arrive this cycle (1-cycle latency)
        //   - Latch weight, assert pe_en to feed PE
        //   - If more cin remain, pre-issue next reads
        //
        // For cin=0: pe_clear=1 to reset accumulator
        // For cin>0: pe_clear=0, accumulates into running psum
        // ─────────────────────────────────────────────
        ENG_COMPUTE: begin
          // Latch weight data arriving from SRAM
          if (wgt_rd_pending) begin
            wgt_lat        <= wgt_rd_data;
            wgt_rd_pending <= 1'b0;
          end

          if (ifm_rd_pending) begin
            ifm_rd_pending <= 1'b0;
          end

          // Feed PE: activations from ifm_rd_data, weight from wgt_lat
          pe_en    <= 1'b1;
          pe_clear <= (cnt_cin == '0) ? 1'b1 : 1'b0;

          // Check if more cin iterations remain
          if (cnt_cin == cin_val - 1) begin
            // Last cin: no more reads needed. Move to drain.
            cnt_cin <= '0;
            state   <= ENG_DRAIN;
            drain_cnt <= '0;
          end else begin
            // Pre-issue reads for next cin
            cnt_cin <= cnt_cin + 1'b1;

            // Weight read for next cin
            wgt_rd_addr <= 24'(cnt_cout) * 24'(cin_val) + 24'(cnt_cin) + 24'd1;
            wgt_rd_en   <= 1'b1;
            wgt_rd_pending <= 1'b1;

            // IFM read for next cin
            ifm_rd_addr <= 24'(cnt_hout) * 24'(cin_val) * 24'(num_wblk)
                         + (24'(cnt_cin) + 24'd1) * 24'(num_wblk)
                         + 24'(cnt_wblk);
            ifm_rd_en   <= 1'b1;
            ifm_rd_pending <= 1'b1;

            // Stay in ENG_COMPUTE for next cin
            state <= ENG_COMPUTE;
          end
        end

        // ─────────────────────────────────────────────
        // DRAIN: Wait for PE pipeline to fully drain
        // DSP pipeline has DSP_LATENCY stages (4 cycles).
        // pe_en=0, inputs zeroed → no spurious accumulation.
        // ─────────────────────────────────────────────
        ENG_DRAIN: begin
          drain_cnt <= drain_cnt + 1'b1;
          if (drain_cnt == DSP_LATENCY[2:0]) begin
            state <= ENG_PPU;
          end
        end

        // ─────────────────────────────────────────────
        // PPU: Post-processing (combinational)
        // ppu_out is valid this cycle from pe_psum
        // ─────────────────────────────────────────────
        ENG_PPU: begin
          state <= ENG_WRITE;
        end

        // ─────────────────────────────────────────────
        // WRITE: Write LANES output bytes to output SRAM
        // Address = h_out * Cout * num_wblk + cout * num_wblk + wblk
        // ─────────────────────────────────────────────
        ENG_WRITE: begin
          ofm_wr_en   <= 1'b1;
          ofm_wr_addr <= 24'(cnt_hout) * 24'(cout_val) * 24'(num_wblk)
                       + 24'(cnt_cout) * 24'(num_wblk)
                       + 24'(cnt_wblk);
          for (int l = 0; l < LANES; l++)
            ofm_wr_data[l] <= ppu_out[l];

          state <= ENG_NEXT_COUT;
        end

        // ─────────────────────────────────────────────
        // LOOP CONTROL: Advance cout → wblk → hout
        // ─────────────────────────────────────────────

        ENG_NEXT_COUT: begin
          if (cnt_cout == cout_val - 1) begin
            cnt_cout <= '0;
            state    <= ENG_NEXT_WBLK;
          end else begin
            cnt_cout <= cnt_cout + 1'b1;
            cnt_cin  <= '0;
            state    <= ENG_LOAD_WGT;
          end
        end

        ENG_NEXT_WBLK: begin
          if (cnt_wblk == num_wblk - 1) begin
            cnt_wblk <= '0;
            state    <= ENG_NEXT_HOUT;
          end else begin
            cnt_wblk <= cnt_wblk + 1'b1;
            cnt_cout <= '0;
            cnt_cin  <= '0;
            state    <= ENG_LOAD_WGT;
          end
        end

        ENG_NEXT_HOUT: begin
          if (cnt_hout == h_val - 1) begin
            state <= ENG_DONE;
          end else begin
            cnt_hout <= cnt_hout + 1'b1;
            cnt_cout <= '0;
            cnt_wblk <= '0;
            cnt_cin  <= '0;
            state    <= ENG_LOAD_WGT;
          end
        end

        // ─────────────────────────────────────────────
        // DONE: Signal completion, return to idle
        // ─────────────────────────────────────────────
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
