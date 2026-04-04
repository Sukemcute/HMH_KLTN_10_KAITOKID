// ============================================================================
// Module : addr_gen_output
// Project: YOLOv10n INT8 Accelerator — V4-VC707
//
// Description:
//   Maps output coordinates → GLB output bank address.
//   Banking: bank_id = PE column index (0-3).
//   Column c writes to bank c → no conflict between columns.
//
//   Address within bank: combines h_out, wblk, and cout information
//   to linearize the 3D output tensor into 1D SRAM address.
//
// Combinational logic (no pipeline needed — output timing not critical).
// ============================================================================
`timescale 1ns / 1ps

module addr_gen_output
  import accel_pkg::*;
#(
  parameter int LANES   = accel_pkg::LANES,     // 20
  parameter int PE_COLS = accel_pkg::PE_COLS     // 4
)(
  input  logic          clk,
  input  logic          rst_n,

  // ── Configuration (from shadow_reg_file) ──
  input  logic [9:0]    cfg_wout,          // Output width
  input  logic [9:0]    cfg_cout,          // Output channels (total for layer)

  // ── Iteration inputs (from compute_sequencer) ──
  input  logic [9:0]    iter_h_out,        // Current output row
  input  logic [9:0]    iter_wblk,         // Current width block
  input  logic [9:0]    iter_cout_group,   // Current cout group (×4)

  // ── Output: per-column bank and address ──
  output logic [1:0]    out_bank_id [PE_COLS],  // bank[col] = col
  output logic [11:0]   out_addr    [PE_COLS]   // SRAM address per column
);

  // ── Derived: total width blocks ──
  logic [9:0] wblk_total;
  assign wblk_total = (cfg_wout + LANES[9:0] - 10'd1) / LANES[9:0];

  // Total cout groups (for address stride calculation)
  logic [9:0] cout_groups_total;
  assign cout_groups_total = (cfg_cout + PE_COLS[9:0] - 10'd1) / PE_COLS[9:0];

  // ── Per-column bank and address ──
  always_comb begin
    for (int col = 0; col < PE_COLS; col++) begin
      // Bank = column index (direct 1:1 mapping, no conflict)
      out_bank_id[col] = col[1:0];

      // Address: linearize (h_out, cout_group, wblk) within bank
      // Each bank stores outputs for 1 PE column
      // addr = h_out × cout_groups_total × wblk_total + cout_group × wblk_total + wblk
      out_addr[col] = 12'(iter_h_out) * 12'(cout_groups_total) * 12'(wblk_total)
                    + 12'(iter_cout_group) * 12'(wblk_total)
                    + 12'(iter_wblk);
    end
  end

  // synthesis translate_off
`ifdef RTL_TRACE
  always @(posedge clk) begin
    if (rst_n)
      rtl_trace_pkg::rtl_trace_line("S4_AGO",
        $sformatf("h=%0d wb=%0d cg=%0d o0=%0d o1=%0d o2=%0d o3=%0d",
                  iter_h_out, iter_wblk, iter_cout_group,
                  out_addr[0], out_addr[1], out_addr[2], out_addr[3]));
  end
`endif
  // synthesis translate_on

endmodule
