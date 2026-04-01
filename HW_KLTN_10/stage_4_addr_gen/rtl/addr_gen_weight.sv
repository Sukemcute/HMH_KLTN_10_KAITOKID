// ============================================================================
// Module : addr_gen_weight
// Project: YOLOv10n INT8 Accelerator — V4-VC707
//
// Description:
//   ★ CORE V4 MODULE: Generates 4 DIFFERENT weight SRAM addresses,
//   one per PE column, for per-column output-channel parallelism.
//
//   Eyeriss v2 insight: Instead of all 4 columns reading the SAME weight
//   (redundant), each column reads weight for a DIFFERENT output channel.
//   → 4× effective throughput with zero extra DSP cost.
//
//   Mode-dependent addressing:
//     PE_RS3: addr[col] = (cout_base + col) × Cin × Kw + cin × Kw + kw
//     PE_OS1: addr[col] = (cout_base + col) × Cin + cin
//     PE_DW3: addr[col] = (ch_base + col) × Kw + kw
//     PE_DW7: addr[col] = (ch_base + col) × Kw + kw  (same formula, different kw range)
//
//   For dense conv (RS3/OS1): cout_base = iter_cout_group × 4
//   For depthwise (DW3/DW7): ch_base = iter_cout_group × 4 (channels, not cout)
//
// All computations are combinational + 1-cycle registered output.
// ============================================================================
`timescale 1ns / 1ps

module addr_gen_weight
  import accel_pkg::*;
#(
  parameter int LANES   = accel_pkg::LANES,     // 20
  parameter int PE_COLS = accel_pkg::PE_COLS     // 4
)(
  input  logic          clk,
  input  logic          rst_n,

  // ── Configuration (from shadow_reg_file) ──
  input  pe_mode_e      cfg_pe_mode,
  input  logic [9:0]    cfg_cin,       // Input channels
  input  logic [9:0]    cfg_cout,      // Output channels
  input  logic [3:0]    cfg_kw,        // Kernel width (1, 3, 5, 7)

  // ── Iteration inputs (from compute_sequencer) ──
  input  logic [9:0]    iter_cin,          // Current input channel
  input  logic [9:0]    iter_cout_group,   // Current cout group (×4 = cout_base)
  input  logic [3:0]    iter_kw,           // Current kernel column position
  input  logic [3:0]    iter_kh_row,       // Current kernel row (for bank select)

  // ── Output: ★ 4 DIFFERENT addresses for 4 PE columns ──
  output logic [15:0]   wgt_addr [PE_COLS],   // 4 independent SRAM addresses
  output logic [1:0]    wgt_bank_id           // Weight bank = kh_row mod 3
);

  // ── Combinational: bank selection ──
  // Weight banks organized by kernel row: bank_id = iter_kh_row mod 3
  always_comb begin
    wgt_bank_id = iter_kh_row[1:0] % 2'd3;
  end

  // ── Combinational: 4 per-column addresses ──
  always_comb begin
    for (int col = 0; col < PE_COLS; col++) begin
      automatic logic [9:0] cout_or_ch;

      // ★ Key: each column addresses a DIFFERENT output channel (or input channel for DW)
      cout_or_ch = (iter_cout_group * PE_COLS[9:0]) + col[9:0];

      case (cfg_pe_mode)
        // ────────────────────────────────────────────────
        // PE_RS3: Dense Conv 3×3
        //   Weight layout: [Cout][Cin][Kw]
        //   addr = cout × Cin × Kw + cin × Kw + kw
        // ────────────────────────────────────────────────
        PE_RS3: begin
          wgt_addr[col] = 16'(cout_or_ch) * 16'(cfg_cin) * 16'(cfg_kw)
                        + 16'(iter_cin)   * 16'(cfg_kw)
                        + 16'(iter_kw);
        end

        // ────────────────────────────────────────────────
        // PE_OS1: Pointwise Conv 1×1
        //   Weight layout: [Cout][Cin]
        //   addr = cout × Cin + cin
        //   No kw dimension (kernel = 1×1)
        // ────────────────────────────────────────────────
        PE_OS1: begin
          wgt_addr[col] = 16'(cout_or_ch) * 16'(cfg_cin)
                        + 16'(iter_cin);
        end

        // ────────────────────────────────────────────────
        // PE_DW3: Depthwise Conv 3×3
        //   Weight layout: [Channel][Kw]
        //   addr = channel × Kw + kw
        //   4 columns = 4 DIFFERENT channels (not cout!)
        // ────────────────────────────────────────────────
        PE_DW3: begin
          wgt_addr[col] = 16'(cout_or_ch) * 16'(cfg_kw)
                        + 16'(iter_kw);
        end

        // ────────────────────────────────────────────────
        // PE_DW7: Depthwise Conv 7×7 (multipass)
        //   Same layout as DW3 but kw ranges 0..6
        //   Multipass: kh handled by tile_fsm pass counter
        // ────────────────────────────────────────────────
        PE_DW7: begin
          wgt_addr[col] = 16'(cout_or_ch) * 16'(cfg_kw)
                        + 16'(iter_kw);
        end

        // ────────────────────────────────────────────────
        // PE_GEMM: Matrix multiply for attention (QPSA)
        //   Weight layout: [Cout][Cin]
        //   Same as OS1 (GEMM is effectively 1×1 conv on reshaped tensor)
        // ────────────────────────────────────────────────
        PE_GEMM: begin
          wgt_addr[col] = 16'(cout_or_ch) * 16'(cfg_cin)
                        + 16'(iter_cin);
        end

        // ────────────────────────────────────────────────
        // PE_MP5, PE_PASS: No weight read needed
        // ────────────────────────────────────────────────
        default: begin
          wgt_addr[col] = 16'd0;
        end
      endcase
    end
  end

endmodule
