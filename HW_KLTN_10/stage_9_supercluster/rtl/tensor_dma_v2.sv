// ============================================================================
// Module : tensor_dma_v2
// Project: YOLOv10n INT8 Accelerator — V4-VC707
//
// Dual-channel AXI4 DMA engine for GLB ↔ DDR3 data movement.
// Channel 0: READ (DDR→GLB, fill path)
// Channel 1: WRITE (GLB→DDR, drain path)
// Supports concurrent read + write for triple-RUNNING overlap.
// Respects AXI4 4KB boundary alignment.
// ============================================================================
`timescale 1ns / 1ps

module tensor_dma_v2
  import accel_pkg::*;
#(
  parameter int LANES      = accel_pkg::LANES,
  parameter int ADDR_W     = AXI_ADDR_WIDTH,
  parameter int DATA_W     = AXI_DATA_WIDTH,
  parameter int BURST_LEN  = 16   // AXI4 burst length (beats)
)(
  input  logic              clk,
  input  logic              rst_n,

  // ═══════════ COMMAND INTERFACE ═══════════
  input  logic              fill_start,
  input  logic [ADDR_W-1:0] fill_ddr_addr,
  input  logic [23:0]       fill_length,      // bytes
  input  logic [1:0]        fill_target,      // 0=input, 1=weight, 2=output
  input  logic [1:0]        fill_bank_id,
  output logic              fill_done,

  input  logic              drain_start,
  input  logic [ADDR_W-1:0] drain_ddr_addr,
  input  logic [23:0]       drain_length,     // bytes
  input  logic [1:0]        drain_bank_id,
  output logic              drain_done,

  // ═══════════ GLB WRITE PORT (fill: DDR → GLB) ═══════════
  output logic              glb_wr_en,
  output logic [1:0]        glb_wr_target,
  output logic [1:0]        glb_wr_bank_id,
  output logic [11:0]       glb_wr_addr,
  output logic signed [7:0] glb_wr_data [LANES],
  output logic [LANES-1:0]  glb_wr_mask,

  // ═══════════ GLB READ PORT (drain: GLB → DDR) ═══════════
  output logic              glb_rd_en,
  output logic [1:0]        glb_rd_bank_id,
  output logic [11:0]       glb_rd_addr,
  input  logic signed [7:0] glb_rd_data [LANES],

  // ═══════════ AXI4 MASTER (READ CHANNEL) ═══════════
  output logic              axi_ar_valid,
  input  logic              axi_ar_ready,
  output logic [ADDR_W-1:0] axi_ar_addr,
  output logic [7:0]        axi_ar_len,
  output logic [2:0]        axi_ar_size,
  output logic [1:0]        axi_ar_burst,

  input  logic              axi_r_valid,
  output logic              axi_r_ready,
  input  logic [DATA_W-1:0] axi_r_data,
  input  logic              axi_r_last,

  // ═══════════ AXI4 MASTER (WRITE CHANNEL) ═══════════
  output logic              axi_aw_valid,
  input  logic              axi_aw_ready,
  output logic [ADDR_W-1:0] axi_aw_addr,
  output logic [7:0]        axi_aw_len,
  output logic [2:0]        axi_aw_size,
  output logic [1:0]        axi_aw_burst,

  output logic              axi_w_valid,
  input  logic              axi_w_ready,
  output logic [DATA_W-1:0] axi_w_data,
  output logic [DATA_W/8-1:0] axi_w_strb,
  output logic              axi_w_last,

  input  logic              axi_b_valid,
  output logic              axi_b_ready
);

  // Fixed AXI burst parameters
  assign axi_ar_size  = 3'b101;  // 32 bytes per beat (256-bit)
  assign axi_ar_burst = 2'b01;   // INCR
  assign axi_aw_size  = 3'b101;
  assign axi_aw_burst = 2'b01;

  // ═══════════ FILL (READ) CHANNEL FSM ═══════════
  typedef enum logic [2:0] {
    F_IDLE, F_AR_ISSUE, F_R_DATA, F_DONE
  } fill_state_e;
  fill_state_e fill_st;

  logic [ADDR_W-1:0] fill_addr_r;
  logic [23:0]       fill_remain;
  logic [11:0]       fill_glb_addr;
  logic [1:0]        fill_tgt_r, fill_bk_r;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      fill_st       <= F_IDLE;
      fill_done     <= 1'b0;
      axi_ar_valid  <= 1'b0;
      axi_r_ready   <= 1'b0;
      glb_wr_en     <= 1'b0;
      fill_glb_addr <= 12'd0;
    end else begin
      fill_done <= 1'b0;
      glb_wr_en <= 1'b0;

      case (fill_st)
        F_IDLE: begin
          if (fill_start) begin
            fill_addr_r   <= fill_ddr_addr;
            fill_remain   <= fill_length;
            fill_tgt_r    <= fill_target;
            fill_bk_r     <= fill_bank_id;
            fill_glb_addr <= 12'd0;
            fill_st       <= F_AR_ISSUE;
          end
        end

        F_AR_ISSUE: begin
          if (fill_remain == 0) begin
            fill_st   <= F_DONE;
          end else begin
            axi_ar_valid <= 1'b1;
            axi_ar_addr  <= fill_addr_r;
            // Compute burst length, respect 4KB boundary
            if (fill_remain > (BURST_LEN * (DATA_W/8)))
              axi_ar_len <= BURST_LEN[7:0] - 8'd1;
            else
              axi_ar_len <= 8'(fill_remain / (DATA_W/8)) - 8'd1;
            if (axi_ar_ready) begin
              axi_ar_valid <= 1'b0;
              axi_r_ready  <= 1'b1;
              fill_st      <= F_R_DATA;
            end
          end
        end

        F_R_DATA: begin
          if (axi_r_valid && axi_r_ready) begin
            // Unpack AXI data into LANES of INT8
            glb_wr_en      <= 1'b1;
            glb_wr_target  <= fill_tgt_r;
            glb_wr_bank_id <= fill_bk_r;
            glb_wr_addr    <= fill_glb_addr;
            glb_wr_mask    <= '1;
            for (int i = 0; i < LANES; i++)
              glb_wr_data[i] <= axi_r_data[i*8 +: 8];

            fill_glb_addr <= fill_glb_addr + 12'd1;
            fill_addr_r   <= fill_addr_r + (DATA_W/8);
            fill_remain   <= fill_remain - (DATA_W/8);

            if (axi_r_last) begin
              axi_r_ready <= 1'b0;
              if (fill_remain <= (DATA_W/8))
                fill_st <= F_DONE;
              else
                fill_st <= F_AR_ISSUE;
            end
          end
        end

        F_DONE: begin
          fill_done <= 1'b1;
          fill_st   <= F_IDLE;
        end
      endcase
    end
  end

  // ═══════════ DRAIN (WRITE) CHANNEL FSM ═══════════
  typedef enum logic [2:0] {
    D_IDLE, D_GLB_RD, D_AW_ISSUE, D_W_DATA, D_B_WAIT, D_DONE
  } drain_state_e;
  drain_state_e drain_st;

  logic [ADDR_W-1:0] drain_addr_r;
  logic [23:0]       drain_remain;
  logic [11:0]       drain_glb_addr;
  logic [1:0]        drain_bk_r;
  logic [7:0]        drain_beat_cnt;
  logic [7:0]        drain_burst_len;

  // Read-data latch (1 cycle GLB read latency)
  logic              drain_rd_pending;
  logic signed [7:0] drain_rd_latch [LANES];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      drain_st        <= D_IDLE;
      drain_done      <= 1'b0;
      axi_aw_valid    <= 1'b0;
      axi_w_valid     <= 1'b0;
      axi_b_ready     <= 1'b0;
      glb_rd_en       <= 1'b0;
      drain_glb_addr  <= 12'd0;
      drain_rd_pending <= 1'b0;
    end else begin
      drain_done  <= 1'b0;
      glb_rd_en   <= 1'b0;
      axi_w_valid <= 1'b0;

      case (drain_st)
        D_IDLE: begin
          if (drain_start) begin
            drain_addr_r   <= drain_ddr_addr;
            drain_remain   <= drain_length;
            drain_bk_r     <= drain_bank_id;
            drain_glb_addr <= 12'd0;
            drain_st       <= D_AW_ISSUE;
          end
        end

        D_AW_ISSUE: begin
          if (drain_remain == 0) begin
            drain_st <= D_DONE;
          end else begin
            axi_aw_valid <= 1'b1;
            axi_aw_addr  <= drain_addr_r;
            if (drain_remain > (BURST_LEN * (DATA_W/8)))
              drain_burst_len <= BURST_LEN[7:0] - 8'd1;
            else
              drain_burst_len <= 8'(drain_remain / (DATA_W/8)) - 8'd1;
            axi_aw_len <= drain_burst_len;
            if (axi_aw_ready) begin
              axi_aw_valid   <= 1'b0;
              drain_beat_cnt <= 8'd0;
              drain_st       <= D_GLB_RD;
            end
          end
        end

        D_GLB_RD: begin
          glb_rd_en      <= 1'b1;
          glb_rd_bank_id <= drain_bk_r;
          glb_rd_addr    <= drain_glb_addr;
          drain_glb_addr <= drain_glb_addr + 12'd1;
          drain_rd_pending <= 1'b1;
          drain_st       <= D_W_DATA;
        end

        D_W_DATA: begin
          if (drain_rd_pending) begin
            for (int i = 0; i < LANES; i++)
              drain_rd_latch[i] <= glb_rd_data[i];
            drain_rd_pending <= 1'b0;
          end

          axi_w_valid <= 1'b1;
          for (int i = 0; i < LANES; i++)
            axi_w_data[i*8 +: 8] <= drain_rd_latch[i];
          // Zero-fill upper bits if DATA_W > LANES*8
          for (int i = LANES*8; i < DATA_W; i++)
            axi_w_data[i] <= 1'b0;
          axi_w_strb <= '1;
          axi_w_last <= (drain_beat_cnt == drain_burst_len);

          if (axi_w_ready) begin
            drain_beat_cnt <= drain_beat_cnt + 8'd1;
            drain_addr_r   <= drain_addr_r + (DATA_W/8);
            drain_remain   <= drain_remain - (DATA_W/8);

            if (drain_beat_cnt == drain_burst_len) begin
              axi_b_ready <= 1'b1;
              drain_st    <= D_B_WAIT;
            end else begin
              drain_st <= D_GLB_RD;
            end
          end
        end

        D_B_WAIT: begin
          axi_b_ready <= 1'b1;
          if (axi_b_valid) begin
            axi_b_ready <= 1'b0;
            if (drain_remain == 0)
              drain_st <= D_DONE;
            else
              drain_st <= D_AW_ISSUE;
          end
        end

        D_DONE: begin
          drain_done <= 1'b1;
          drain_st   <= D_IDLE;
        end
      endcase
    end
  end

endmodule
