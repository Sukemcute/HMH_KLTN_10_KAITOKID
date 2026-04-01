`timescale 1ns/1ps
// Tensor layout transform: bank_output → bank_input for next layer.
// Handles: normal layer chaining, UPSAMPLE_NEAREST (2x), CONCAT channel offset.
module swizzle_engine #(
  parameter int LANES = 32
)(
  input  logic                clk,
  input  logic                rst_n,
  input  logic                start,
  input  accel_pkg::pe_mode_e mode,

  // Configuration
  input  logic [1:0]          cfg_upsample_factor,   // 0=none, 1=2x
  input  logic [8:0]          cfg_concat_ch_offset,
  input  logic [9:0]          cfg_src_h, cfg_src_w, cfg_src_c,
  input  logic [9:0]          cfg_dst_h, cfg_dst_w,
  input  logic [3:0]          cfg_dst_q_in,
  input  logic [8:0]          cfg_dst_cin_tile,

  // Source: read from bank_output (ACT namespace)
  output logic                src_rd_en,
  output logic [15:0]         src_rd_addr,
  output logic [1:0]          src_rd_bank,
  input  logic [LANES*8-1:0]  src_rd_data,

  // Destination: write to bank_input (next layer)
  output logic                dst_wr_en,
  output logic [15:0]         dst_wr_addr,
  output logic [1:0]          dst_wr_bank,
  output logic [LANES*8-1:0]  dst_wr_data,
  output logic [LANES-1:0]    dst_wr_mask,

  output logic                done
);
  import accel_pkg::*;

  typedef enum logic [2:0] {
    SW_IDLE,
    SW_READ,
    SW_READ_WAIT,
    SW_WRITE,
    SW_UP_WRITE,
    SW_DONE
  } sw_state_e;

  sw_state_e state, nstate;

  logic [9:0] h_cnt, w_cnt, c_cnt;
  logic [1:0] up_phase;  // 0..3 for 2x upsample
  logic       rd_data_valid;

  // wblk counters
  wire [5:0] src_wblk = w_cnt[9:5]; // w_cnt / LANES (assuming LANES=32)
  wire [5:0] src_wblk_total = (cfg_src_w + LANES - 1) / LANES;
  wire [5:0] dst_wblk_total = (cfg_dst_w + LANES - 1) / LANES;

  // Source address: simple sequential read from output bank
  wire [1:0] src_bank = h_cnt[1:0]; // pe_col bank
  wire [15:0] src_addr_calc = 16'(c_cnt) * 16'(src_wblk_total) + 16'(src_wblk);

  // Destination address calculation
  logic [9:0] dst_h, dst_w;
  logic [8:0] dst_c;
  wire [1:0] dst_bank_id = dst_h % 3;
  wire [3:0] dst_slot    = (dst_h / 3) % cfg_dst_q_in;
  wire [5:0] dst_wblk    = dst_w[9:5];
  wire [15:0] dst_addr_calc = (16'(dst_slot) * 16'(cfg_dst_cin_tile) + 16'(dst_c))
                              * 16'(dst_wblk_total) + 16'(dst_wblk);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      state <= SW_IDLE;
    else
      state <= nstate;
  end

  always_comb begin
    nstate = state;
    case (state)
      SW_IDLE:      if (start) nstate = SW_READ;
      SW_READ:      nstate = SW_READ_WAIT;
      SW_READ_WAIT: nstate = SW_WRITE;
      SW_WRITE: begin
        if (cfg_upsample_factor == 2'd1)
          nstate = SW_UP_WRITE;
        else begin
          if (c_cnt >= cfg_src_c - 1 && w_cnt >= cfg_src_w - LANES && h_cnt >= cfg_src_h - 1)
            nstate = SW_DONE;
          else
            nstate = SW_READ;
        end
      end
      SW_UP_WRITE: begin
        if (up_phase == 2'd3) begin
          if (c_cnt >= cfg_src_c - 1 && w_cnt >= cfg_src_w - LANES && h_cnt >= cfg_src_h - 1)
            nstate = SW_DONE;
          else
            nstate = SW_READ;
        end
      end
      SW_DONE:      nstate = SW_IDLE;
      default:      nstate = SW_IDLE;
    endcase
  end

  // Counters: iterate (h, w_blk, c)
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      h_cnt    <= '0;
      w_cnt    <= '0;
      c_cnt    <= '0;
      up_phase <= '0;
    end else if (state == SW_IDLE && start) begin
      h_cnt    <= '0;
      w_cnt    <= '0;
      c_cnt    <= '0;
      up_phase <= '0;
    end else if (state == SW_WRITE && cfg_upsample_factor == 2'd0) begin
      if (c_cnt < cfg_src_c - 1)
        c_cnt <= c_cnt + 1;
      else begin
        c_cnt <= '0;
        if (w_cnt < cfg_src_w - LANES)
          w_cnt <= w_cnt + LANES;
        else begin
          w_cnt <= '0;
          h_cnt <= h_cnt + 1;
        end
      end
    end else if (state == SW_UP_WRITE && up_phase == 2'd3) begin
      up_phase <= '0;
      if (c_cnt < cfg_src_c - 1)
        c_cnt <= c_cnt + 1;
      else begin
        c_cnt <= '0;
        if (w_cnt < cfg_src_w - LANES)
          w_cnt <= w_cnt + LANES;
        else begin
          w_cnt <= '0;
          h_cnt <= h_cnt + 1;
        end
      end
    end else if (state == SW_UP_WRITE || (state == SW_WRITE && cfg_upsample_factor == 2'd1)) begin
      up_phase <= up_phase + 1;
    end
  end

  // Upsample destination mapping: phase 0→(2h,2w), 1→(2h,2w+L), 2→(2h+1,2w), 3→(2h+1,2w+L)
  always_comb begin
    dst_c = cfg_concat_ch_offset + c_cnt;
    if (cfg_upsample_factor == 2'd1) begin
      case (up_phase)
        2'd0: begin dst_h = {h_cnt[8:0], 1'b0};     dst_w = {w_cnt[8:0], 1'b0};     end
        2'd1: begin dst_h = {h_cnt[8:0], 1'b0};     dst_w = {w_cnt[8:0], 1'b0} + LANES; end
        2'd2: begin dst_h = {h_cnt[8:0], 1'b0} + 1; dst_w = {w_cnt[8:0], 1'b0};     end
        2'd3: begin dst_h = {h_cnt[8:0], 1'b0} + 1; dst_w = {w_cnt[8:0], 1'b0} + LANES; end
      endcase
    end else begin
      dst_h = h_cnt;
      dst_w = w_cnt;
    end
  end

  // Output control
  always_comb begin
    src_rd_en   = (state == SW_READ);
    src_rd_bank = src_bank;
    src_rd_addr = src_addr_calc;

    dst_wr_en   = (state == SW_WRITE) || (state == SW_UP_WRITE);
    dst_wr_bank = dst_bank_id;
    dst_wr_addr = dst_addr_calc;
    dst_wr_data = src_rd_data;
    dst_wr_mask = {LANES{1'b1}};

    done = (state == SW_DONE);
  end

endmodule
