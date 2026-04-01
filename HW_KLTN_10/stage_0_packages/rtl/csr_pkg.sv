// ============================================================================
// Module : csr_pkg
// Project: YOLOv10n INT8 Accelerator — V4-VC707
// Description:
//   CSR (Control/Status Register) address map for CPU ↔ Accelerator
//   communication via AXI-Lite interface.
//
//   CPU writes CSR_CTRL.start → Accelerator processes L0-L22 → IRQ assert
// ============================================================================
`timescale 1ns / 1ps

package csr_pkg;

  // ═══════════════════════════════════════════════════════════════════
  //  CSR ADDRESS MAP (12-bit byte-address, 32-bit data width)
  // ═══════════════════════════════════════════════════════════════════

  // — Control / Status —
  parameter logic [11:0] CSR_CTRL          = 12'h000;  // [R/W] start, soft_reset, irq_clear
  parameter logic [11:0] CSR_STATUS        = 12'h004;  // [R]   busy, done, error, state

  // — Descriptor Pointers —
  parameter logic [11:0] CSR_NET_DESC_LO   = 12'h010;  // [R/W] Net descriptor base [31:0]
  parameter logic [11:0] CSR_NET_DESC_HI   = 12'h014;  // [R/W] Net descriptor base [63:32]
  parameter logic [11:0] CSR_LAYER_START   = 12'h018;  // [R/W] First layer to execute (0-22)
  parameter logic [11:0] CSR_LAYER_END     = 12'h01C;  // [R/W] Last layer to execute (0-22)

  // — Performance Counters —
  parameter logic [11:0] CSR_PERF_CYCLES   = 12'h020;  // [R] Total clock cycles
  parameter logic [11:0] CSR_PERF_STALLS   = 12'h024;  // [R] Stall cycle counter
  parameter logic [11:0] CSR_PERF_TILES    = 12'h028;  // [R] Completed tile counter

  // — Interrupt —
  parameter logic [11:0] CSR_IRQ_MASK      = 12'h02C;  // [R/W] Interrupt enable mask

  // — Info (Read-Only) —
  parameter logic [11:0] CSR_VERSION       = 12'h030;  // [R] IP version (V4 = 0x0004_0000)
  parameter logic [11:0] CSR_CONFIG        = 12'h034;  // [R] {LANES, N_SUBS, CLK_MHZ}

  // ═══════════════════════════════════════════════════════════════════
  //  CSR_CTRL BIT FIELDS
  // ═══════════════════════════════════════════════════════════════════
  parameter int CTRL_BIT_START      = 0;   // Write 1 to start inference
  parameter int CTRL_BIT_SOFT_RST   = 1;   // Write 1 for soft reset
  parameter int CTRL_BIT_IRQ_CLEAR  = 2;   // Write 1 to clear IRQ

  // ═══════════════════════════════════════════════════════════════════
  //  CSR_STATUS BIT FIELDS
  // ═══════════════════════════════════════════════════════════════════
  parameter int STAT_BIT_BUSY       = 0;   // 1 = inference in progress
  parameter int STAT_BIT_DONE       = 1;   // 1 = inference complete
  parameter int STAT_BIT_ERROR      = 2;   // 1 = error occurred

  // ═══════════════════════════════════════════════════════════════════
  //  VERSION / CONFIG CONSTANTS
  // ═══════════════════════════════════════════════════════════════════
  parameter logic [31:0] IP_VERSION = 32'h0004_0000;  // V4.0
  parameter logic [31:0] IP_CONFIG  = {8'(accel_pkg::LANES),
                                       8'(accel_pkg::N_TOTAL_SUBS),
                                       16'(accel_pkg::TARGET_CLOCK_MHZ)};

endpackage
