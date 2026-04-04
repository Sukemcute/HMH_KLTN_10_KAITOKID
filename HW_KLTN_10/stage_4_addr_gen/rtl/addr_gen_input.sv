// ============================================================================
// Module : addr_gen_input
// Project: YOLOv10n INT8 Accelerator — V4-VC707
//
// Description:
//   Computes GLB input bank address from logical coordinates (h_in, w, cin).
//   Banking rule: bank_id = h_in mod 3 (3 input banks)
//
//   ★ PE_DW3 / PE_DW7 (Eyeriss-style): 4 columns = 4 input channels in parallel.
//   bank_rd_addr_col[b][col] = address for physical bank b, PE column col,
//   using cin = iter_cout_group×4+col (requires glb_input_bank_db ×4 read ports).
//
//   ★ RULE 5: Padding → cfg_zp_x (NOT literal 0).
//
//   Combinational address logic + 1-cycle registered output.
// ============================================================================
`timescale 1ns / 1ps

module addr_gen_input
  import accel_pkg::*;
#(
  parameter int LANES   = accel_pkg::LANES,   // 20
  parameter int PE_COLS = accel_pkg::PE_COLS  // 4 — must be a module param (not pkg-only) for xelab port sizing
)(
  input  logic          clk,
  input  logic          rst_n,

  input  pe_mode_e      cfg_pe_mode,
  input  logic [9:0]    cfg_hin,
  input  logic [9:0]    cfg_win,
  input  logic [9:0]    cfg_cin,
  input  logic [2:0]    cfg_stride,
  input  logic [2:0]    cfg_padding,
  input  int8_t         cfg_zp_x,

  input  logic [9:0]    iter_h_out,
  input  logic [9:0]    iter_wblk,
  input  logic [9:0]    iter_cin,
  input  logic [9:0]    iter_cout_group,
  input  logic [3:0]    iter_kh_row,

  output logic [1:0]    bank_id,
  output logic [11:0]   sram_addr,
  output logic [11:0]   sram_addr_row [3],
  output logic          is_padding,
  output int8_t         pad_value,
  output logic [1:0]    bank_id_row [3],
  output logic          is_padding_row [3],
  // Legacy single addr per physical bank (= col 0); RS3: all cols match
  output logic [11:0]   bank_rd_addr [3],
  // ★ Per-column read addresses for physical banks (DW3/DW7 + RS3 broadcast cin)
  output logic [11:0]   bank_rd_addr_col [3][PE_COLS]
);

  logic signed [10:0] h_in_raw;
  logic [9:0]         w_in_base;

  always_comb begin
    h_in_raw  = 11'(signed'({1'b0, iter_h_out})) * 11'(signed'({1'b0, cfg_stride}))
              + 11'(signed'({1'b0, iter_kh_row}))
              - 11'(signed'({1'b0, cfg_padding}));
    w_in_base = iter_wblk * LANES[9:0];
  end

  logic pad_h, pad_w;
  always_comb begin
    pad_h = (h_in_raw < 0) || (h_in_raw >= signed'({1'b0, cfg_hin}));
    pad_w = (w_in_base >= cfg_win);
  end

  logic [1:0]  bank_comb;
  logic [11:0] addr_comb;
  logic [11:0] addr_row_comb [3];

  function automatic logic [11:0] addr_from_h_in_cin(
    input logic signed [10:0] hir,
    input logic [9:0]         cin_idx
  );
    automatic logic [9:0] wblk_total = (cfg_win + LANES[9:0] - 10'd1) / LANES[9:0];
    automatic logic        pad_h_i = (hir < 0) || (hir >= signed'({1'b0, cfg_hin}));
    automatic logic        pad_w_i = (iter_wblk * LANES[9:0] >= cfg_win);
    automatic logic [9:0]  hp;
    if (pad_h_i || pad_w_i)
      return 12'd0;
    hp = hir[9:0];
    return 12'((hp / 10'd3) * wblk_total * cfg_cin[9:0])
         + 12'(cin_idx * wblk_total)
         + 12'(iter_wblk);
  endfunction

  // Non-RS3/DW path (incl. PE_MP5): never use h_in_raw[9:0] when h_in_raw is negative —
  // the low bits of a negative two's-complement value become a huge unsigned h_pos and
  // drive SRAM addresses past depth → X on read data while pad_h is still asserted.
  always_comb begin
    automatic logic [9:0] h_pos;
    automatic logic [9:0] wblk_total = (cfg_win + LANES[9:0] - 10'd1) / LANES[9:0];

    if (pad_h || pad_w)
      h_pos = 10'd0;
    else
      h_pos = h_in_raw[9:0];

    bank_comb = h_pos % 10'd3;

    addr_comb = 12'((h_pos / 10'd3) * wblk_total * cfg_cin[9:0])
              + 12'(iter_cin * wblk_total)
              + 12'(iter_wblk);

    for (int r = 0; r < 3; r++) begin
      if (cfg_pe_mode == PE_RS3 || cfg_pe_mode == PE_DW3 || cfg_pe_mode == PE_DW7) begin
        automatic logic signed [10:0] hir_r;
        hir_r = 11'(signed'({1'b0, iter_h_out})) * 11'(signed'({1'b0, cfg_stride}))
              + 11'(signed'(r))
              - 11'(signed'({1'b0, cfg_padding}));
        addr_row_comb[r] = addr_from_h_in_cin(hir_r, iter_cin);
      end else
        addr_row_comb[r] = addr_comb;
    end
  end

  logic [1:0]  bank_id_row_comb [3];
  logic        is_padding_row_comb [3];
  logic [11:0] bank_rd_addr_col_comb [3][PE_COLS];

  always_comb begin
    for (int r = 0; r < 3; r++) begin
      if (cfg_pe_mode == PE_RS3 || cfg_pe_mode == PE_DW3 || cfg_pe_mode == PE_DW7) begin
        automatic logic signed [10:0] hir_r;
        hir_r = 11'(signed'({1'b0, iter_h_out})) * 11'(signed'({1'b0, cfg_stride}))
              + 11'(signed'(r))
              - 11'(signed'({1'b0, cfg_padding}));
        is_padding_row_comb[r] = (hir_r < 0) || (hir_r >= signed'({1'b0, cfg_hin}))
                                 || (iter_wblk * LANES[9:0] >= cfg_win);
        bank_id_row_comb[r] = is_padding_row_comb[r] ? 2'd0 : (hir_r[9:0] % 10'd3);
      end else begin
        is_padding_row_comb[r] = pad_h | pad_w;
        bank_id_row_comb[r] = bank_comb;
      end
    end

    for (int b = 0; b < 3; b++) begin
      for (int col = 0; col < PE_COLS; col++) begin
        bank_rd_addr_col_comb[b][col] = 12'd0;
        for (int r = 0; r < 3; r++) begin
          if ((cfg_pe_mode == PE_RS3 || cfg_pe_mode == PE_DW3 || cfg_pe_mode == PE_DW7)
              && !is_padding_row_comb[r] && (bank_id_row_comb[r] == b[1:0])) begin
            automatic logic signed [10:0] hir_r;
            automatic logic [9:0] ecin;
            hir_r = 11'(signed'({1'b0, iter_h_out})) * 11'(signed'({1'b0, cfg_stride}))
                  + 11'(signed'(r))
                  - 11'(signed'({1'b0, cfg_padding}));
            if (cfg_pe_mode == PE_DW3 || cfg_pe_mode == PE_DW7)
              ecin = iter_cout_group * PE_COLS[9:0] + col[9:0];
            else
              ecin = iter_cin;
            bank_rd_addr_col_comb[b][col] = addr_from_h_in_cin(hir_r, ecin);
          end
        end
      end
    end
  end

  logic [11:0] bank_rd_addr_comb [3];
  always_comb begin
    for (int b = 0; b < 3; b++)
      bank_rd_addr_comb[b] = bank_rd_addr_col_comb[b][0];
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      bank_id    <= 2'd0;
      sram_addr  <= 12'd0;
      is_padding <= 1'b0;
      pad_value  <= 8'sd0;
      for (int r = 0; r < 3; r++) begin
        sram_addr_row[r]  <= 12'd0;
        bank_id_row[r]    <= 2'd0;
        is_padding_row[r] <= 1'b0;
        bank_rd_addr[r]   <= 12'd0;
        for (int c = 0; c < PE_COLS; c++)
          bank_rd_addr_col[r][c] <= 12'd0;
      end
    end else begin
      bank_id    <= bank_comb;
      sram_addr  <= addr_comb;
      is_padding <= pad_h | pad_w;
      pad_value  <= cfg_zp_x;
      for (int r = 0; r < 3; r++) begin
        sram_addr_row[r]  <= addr_row_comb[r];
        bank_id_row[r]    <= bank_id_row_comb[r];
        is_padding_row[r] <= is_padding_row_comb[r];
        bank_rd_addr[r]   <= bank_rd_addr_comb[r];
        for (int c = 0; c < PE_COLS; c++)
          bank_rd_addr_col[r][c] <= bank_rd_addr_col_comb[r][c];
      end
    end
  end

  // synthesis translate_off
`ifdef RTL_TRACE
  always @(posedge clk) begin
    if (rst_n)
      rtl_trace_pkg::rtl_trace_line("S4_AGI",
        $sformatf("mode=%0d bank=%0d addr=%0d ar0=%0d ar1=%0d pad=%b zp=%0d kh=%0d",
                  cfg_pe_mode, bank_id, sram_addr, sram_addr_row[0], sram_addr_row[1],
                  is_padding, pad_value, iter_kh_row));
  end
`endif
  // synthesis translate_on

endmodule
