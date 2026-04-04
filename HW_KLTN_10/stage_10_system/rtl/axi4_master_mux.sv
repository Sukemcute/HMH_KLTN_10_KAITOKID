// ============================================================================
// Module : axi4_master_mux
// Project: YOLOv10n INT8 Accelerator — V4-VC707
//
// Round-robin AXI4 arbiter: multiplexes 4 SC AXI masters + 1 desc_fetch
// into a single AXI4 master port to DDR3 controller.
// Arbitration is per-transaction (AR/AW level).
// ============================================================================
`timescale 1ns / 1ps

module axi4_master_mux
  import accel_pkg::*;
#(
  parameter int N_MASTERS = N_SUPER_CLUSTERS + 1,  // 4 SC + 1 desc_fetch = 5
  parameter int ADDR_W    = AXI_ADDR_WIDTH,
  parameter int DATA_W    = AXI_DATA_WIDTH
)(
  input  logic              clk,
  input  logic              rst_n,

  // ═══════════ SLAVE PORTS (from masters: 4 SCs + desc_fetch) ═══════════
  // Read address
  input  logic              m_ar_valid [N_MASTERS],
  output logic              m_ar_ready [N_MASTERS],
  input  logic [ADDR_W-1:0] m_ar_addr  [N_MASTERS],
  input  logic [7:0]        m_ar_len   [N_MASTERS],
  input  logic [2:0]        m_ar_size  [N_MASTERS],
  input  logic [1:0]        m_ar_burst [N_MASTERS],

  // Read data
  output logic              m_r_valid  [N_MASTERS],
  input  logic              m_r_ready  [N_MASTERS],
  output logic [DATA_W-1:0] m_r_data   [N_MASTERS],
  output logic              m_r_last   [N_MASTERS],

  // Write address
  input  logic              m_aw_valid [N_MASTERS],
  output logic              m_aw_ready [N_MASTERS],
  input  logic [ADDR_W-1:0] m_aw_addr  [N_MASTERS],
  input  logic [7:0]        m_aw_len   [N_MASTERS],
  input  logic [2:0]        m_aw_size  [N_MASTERS],
  input  logic [1:0]        m_aw_burst [N_MASTERS],

  // Write data
  input  logic              m_w_valid  [N_MASTERS],
  output logic              m_w_ready  [N_MASTERS],
  input  logic [DATA_W-1:0] m_w_data   [N_MASTERS],
  input  logic [DATA_W/8-1:0] m_w_strb [N_MASTERS],
  input  logic              m_w_last   [N_MASTERS],

  // Write response
  output logic              m_b_valid  [N_MASTERS],
  input  logic              m_b_ready  [N_MASTERS],

  // ═══════════ MASTER PORT (to DDR3 controller) ═══════════
  output logic              ddr_ar_valid,
  input  logic              ddr_ar_ready,
  output logic [ADDR_W-1:0] ddr_ar_addr,
  output logic [7:0]        ddr_ar_len,
  output logic [2:0]        ddr_ar_size,
  output logic [1:0]        ddr_ar_burst,

  input  logic              ddr_r_valid,
  output logic              ddr_r_ready,
  input  logic [DATA_W-1:0] ddr_r_data,
  input  logic              ddr_r_last,

  output logic              ddr_aw_valid,
  input  logic              ddr_aw_ready,
  output logic [ADDR_W-1:0] ddr_aw_addr,
  output logic [7:0]        ddr_aw_len,
  output logic [2:0]        ddr_aw_size,
  output logic [1:0]        ddr_aw_burst,

  output logic              ddr_w_valid,
  input  logic              ddr_w_ready,
  output logic [DATA_W-1:0] ddr_w_data,
  output logic [DATA_W/8-1:0] ddr_w_strb,
  output logic              ddr_w_last,

  input  logic              ddr_b_valid,
  output logic              ddr_b_ready
);

  localparam int IDX_W = $clog2(N_MASTERS);

  // ─── Read arbitration ───
  logic [IDX_W-1:0] rd_grant;
  logic             rd_locked;  // transaction in progress
  logic [IDX_W-1:0] rd_rr;

  // Round-robin read arbiter
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rd_grant  <= '0;
      rd_locked <= 1'b0;
      rd_rr     <= '0;
    end else begin
      if (!rd_locked) begin
        // Find next requesting master
        for (int i = 0; i < N_MASTERS; i++) begin
          automatic int idx = (rd_rr + i) % N_MASTERS;
          if (m_ar_valid[idx]) begin
            rd_grant  <= idx[IDX_W-1:0];
            rd_locked <= 1'b1;
            break;
          end
        end
      end else begin
        // Unlock when read burst completes
        if (ddr_r_valid && ddr_r_ready && ddr_r_last) begin
          rd_locked <= 1'b0;
          rd_rr     <= (rd_grant + 1) % N_MASTERS;
        end
      end
    end
  end

  // Read channel mux
  assign ddr_ar_valid = rd_locked ? m_ar_valid[rd_grant] : 1'b0;
  assign ddr_ar_addr  = m_ar_addr[rd_grant];
  assign ddr_ar_len   = m_ar_len[rd_grant];
  assign ddr_ar_size  = m_ar_size[rd_grant];
  assign ddr_ar_burst = m_ar_burst[rd_grant];
  assign ddr_r_ready  = rd_locked ? m_r_ready[rd_grant] : 1'b0;

  always_comb begin
    for (int i = 0; i < N_MASTERS; i++) begin
      m_ar_ready[i] = (rd_locked && rd_grant == i[IDX_W-1:0]) ? ddr_ar_ready : 1'b0;
      m_r_valid[i]  = (rd_locked && rd_grant == i[IDX_W-1:0]) ? ddr_r_valid  : 1'b0;
      m_r_data[i]   = ddr_r_data;
      m_r_last[i]   = ddr_r_last;
    end
  end

  // ─── Write arbitration ───
  logic [IDX_W-1:0] wr_grant;
  logic             wr_locked;
  logic [IDX_W-1:0] wr_rr;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wr_grant  <= '0;
      wr_locked <= 1'b0;
      wr_rr     <= '0;
    end else begin
      if (!wr_locked) begin
        for (int i = 0; i < N_MASTERS; i++) begin
          automatic int idx = (wr_rr + i) % N_MASTERS;
          if (m_aw_valid[idx]) begin
            wr_grant  <= idx[IDX_W-1:0];
            wr_locked <= 1'b1;
            break;
          end
        end
      end else begin
        if (ddr_b_valid && ddr_b_ready) begin
          wr_locked <= 1'b0;
          wr_rr     <= (wr_grant + 1) % N_MASTERS;
        end
      end
    end
  end

  // Write channel mux
  assign ddr_aw_valid = wr_locked ? m_aw_valid[wr_grant] : 1'b0;
  assign ddr_aw_addr  = m_aw_addr[wr_grant];
  assign ddr_aw_len   = m_aw_len[wr_grant];
  assign ddr_aw_size  = m_aw_size[wr_grant];
  assign ddr_aw_burst = m_aw_burst[wr_grant];
  assign ddr_w_valid  = wr_locked ? m_w_valid[wr_grant]  : 1'b0;
  assign ddr_w_data   = m_w_data[wr_grant];
  assign ddr_w_strb   = m_w_strb[wr_grant];
  assign ddr_w_last   = m_w_last[wr_grant];
  assign ddr_b_ready  = wr_locked ? m_b_ready[wr_grant]  : 1'b0;

  always_comb begin
    for (int i = 0; i < N_MASTERS; i++) begin
      m_aw_ready[i] = (wr_locked && wr_grant == i[IDX_W-1:0]) ? ddr_aw_ready : 1'b0;
      m_w_ready[i]  = (wr_locked && wr_grant == i[IDX_W-1:0]) ? ddr_w_ready  : 1'b0;
      m_b_valid[i]  = (wr_locked && wr_grant == i[IDX_W-1:0]) ? ddr_b_valid  : 1'b0;
    end
  end

endmodule
