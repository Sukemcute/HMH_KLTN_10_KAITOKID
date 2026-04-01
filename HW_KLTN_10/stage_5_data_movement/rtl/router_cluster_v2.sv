// ============================================================================
// Module : router_cluster_v2
// Project: YOLOv10n INT8 Accelerator — V4-VC707
//
// Description:
//   ★ EYERISS V2-INSPIRED Data Routing Hub.
//   3 independent routing networks + bypass path:
//
//   RIN (Activation Router):
//     GLB input banks → PE rows. MULTICAST: same activation to all 3 rows.
//     (Each row handles a different kernel row kh=0,1,2 but same spatial data.)
//
//   RWT (Weight Router):
//     GLB weight banks → PE rows × PE columns. ★ PER-COLUMN WEIGHT.
//     Each of 4 columns receives DIFFERENT weight = different output channel.
//     This is the CORE optimization from Eyeriss v2.
//
//   RPS (Psum Router):
//     PE columns → GLB output banks. Direct column-to-bank mapping.
//
//   BYPASS (for PE_PASS modes):
//     Input → Output directly (upsample, concat, move).
//     PE cluster is bypassed entirely.
//
// All routing is combinational (registered at source/destination).
// ============================================================================
`timescale 1ns / 1ps

module router_cluster_v2
  import accel_pkg::*;
#(
  parameter int LANES   = accel_pkg::LANES,     // 20
  parameter int PE_ROWS = accel_pkg::PE_ROWS,   // 3
  parameter int PE_COLS = accel_pkg::PE_COLS     // 4
)(
  input  logic          clk,
  input  logic          rst_n,
  input  pe_mode_e      cfg_pe_mode,
  // ★ MP5 / streaming: which input bank holds the row read (h mod 3)
  input  logic [1:0]    rin_bank_sel,

  // ═══ RIN: Activation Router (GLB Input → PE Rows) ═══
  // 3 input banks provide data; router selects and multicasts
  input  int8_t         glb_in_data  [3][LANES],     // 3 banks × 20 lanes
  output int8_t         pe_act       [PE_ROWS][LANES], // 3 rows × 20 lanes

  // ═══ RWT: Weight Router (GLB Weight → PE Rows × Columns) ═══
  // ★ V4: 4 different weight streams per row (1 per column)
  // glb_wgt_data[bank][col][lane]: bank selects kh row, col selects cout
  input  int8_t         glb_wgt_data [3][PE_COLS][LANES], // 3 banks × 4 cols × 20 lanes
  output int8_t         pe_wgt       [PE_ROWS][PE_COLS][LANES], // ★ 3×4×20

  // ═══ RPS: Psum/Output Router (PE Columns → GLB Output) ═══
  input  int32_t        pe_psum_in   [PE_COLS][LANES], // 4 cols × 20 lanes
  input  logic          psum_valid,
  output int32_t        glb_out_psum [PE_COLS][LANES], // 4 banks × 20 lanes
  output logic          glb_out_wr_en [PE_COLS],

  // ═══ BYPASS Path (PE_PASS modes: upsample, concat, move) ═══
  input  int8_t         bypass_in    [LANES],
  output int8_t         bypass_out   [LANES],
  input  logic          bypass_en
);

  // ────────────────────────────────────────────────────────
  // RIN: Activation Routing — MULTICAST to all PE rows
  //
  // For conv: all 3 PE rows see the SAME activation data
  // (each row handles different kh, but same spatial position)
  // Bank selection: bank = iter_kh_row mod 3 (from addr_gen_input)
  //
  // For RS3: bank 0 → row 0 (kh=0), bank 1 → row 1 (kh=1), bank 2 → row 2 (kh=2)
  // For OS1: all rows get bank 0 (only 1 kernel row)
  // For DW:  same as RS3
  // For MP5: not used (comparator tree instead)
  // ────────────────────────────────────────────────────────
  always_comb begin
    for (int row = 0; row < PE_ROWS; row++) begin
      for (int ln = 0; ln < LANES; ln++) begin
        case (cfg_pe_mode)
          PE_OS1:  pe_act[row][ln] = glb_in_data[0][ln]; // 1×1: all from bank 0
          PE_PASS: pe_act[row][ln] = 8'sd0;              // Bypass: PE inactive
          PE_MP5:  pe_act[row][ln] = glb_in_data[rin_bank_sel][ln];  // Stream rows → window_gen
          default: pe_act[row][ln] = glb_in_data[row][ln]; // RS3/DW: row-matched
        endcase
      end
    end
  end

  // ────────────────────────────────────────────────────────
  // RWT: Weight Routing — ★ PER-COLUMN (Eyeriss v2 core idea)
  //
  // Each PE column receives a DIFFERENT weight for a DIFFERENT cout.
  // Bank selection: bank = kh_row (same as RIN for rows)
  // Column selection: col 0..3 each get different cout weight
  //
  // glb_wgt_data[bank][col] was loaded at different SRAM addresses
  // by addr_gen_weight (4 per-column addresses).
  //
  // For DW modes: 4 columns = 4 different INPUT channels
  // ────────────────────────────────────────────────────────
  always_comb begin
    for (int row = 0; row < PE_ROWS; row++) begin
      for (int col = 0; col < PE_COLS; col++) begin
        for (int ln = 0; ln < LANES; ln++) begin
          case (cfg_pe_mode)
            PE_RS3, PE_OS1, PE_DW3, PE_DW7, PE_GEMM: begin
              // ★ Each column gets its OWN weight from bank[row]
              pe_wgt[row][col][ln] = glb_wgt_data[row][col][ln];
            end
            default: begin
              // PE_MP5, PE_PASS: weights not used
              pe_wgt[row][col][ln] = 8'sd0;
            end
          endcase
        end
      end
    end
  end

  // ────────────────────────────────────────────────────────
  // RPS: Psum → Output Bank (direct column-to-bank mapping)
  //
  // Column c → Bank c. No crossbar needed.
  // PE_COLS = OUTPUT_BANKS = 4 by design.
  // ────────────────────────────────────────────────────────
  always_comb begin
    for (int col = 0; col < PE_COLS; col++) begin
      glb_out_wr_en[col] = psum_valid;
      for (int ln = 0; ln < LANES; ln++) begin
        glb_out_psum[col][ln] = pe_psum_in[col][ln];
      end
    end
  end

  // ────────────────────────────────────────────────────────
  // BYPASS: Direct passthrough for PE_PASS modes
  //
  // Used for: UPSAMPLE_NEAREST, CONCAT, MOVE
  // PE cluster and PPU are completely bypassed.
  // ────────────────────────────────────────────────────────
  always_comb begin
    for (int ln = 0; ln < LANES; ln++) begin
      bypass_out[ln] = bypass_en ? bypass_in[ln] : 8'sd0;
    end
  end

endmodule
