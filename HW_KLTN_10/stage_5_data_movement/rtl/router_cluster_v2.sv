// ============================================================================
// Module : router_cluster_v2
// Project: YOLOv10n INT8 Accelerator — V4-VC707
//
// Description:
//   ★ EYERISS V2-INSPIRED Data Routing Hub.
//
//   RIN (Activation Router):
//     glb_in_data[bank][col][lane]: ★ 4 read ports per input bank (DW3/DW7).
//     PE_RS3/DW3/DW7: pe_act[row][col] = glb_in_data[row][col] (row=kh bank, col=channel/cin).
//     PE_OS1/GEMM: row 0 only, all cols (same act broadcast from subcluster).
//
//   RWT: per-column weight (unchanged).
// ============================================================================
`timescale 1ns / 1ps

module router_cluster_v2
  import accel_pkg::*;
#(
  parameter int LANES   = accel_pkg::LANES,
  parameter int PE_ROWS = accel_pkg::PE_ROWS,
  parameter int PE_COLS = accel_pkg::PE_COLS
)(
  input  logic          clk,
  input  logic          rst_n,
  input  pe_mode_e      cfg_pe_mode,
  input  logic [1:0]    rin_bank_sel,

  // ★ 3 banks × 4 column-read ports × LANES (duplicate data per col for OS1/RS3)
  input  int8_t         glb_in_data  [3][PE_COLS][LANES],
  output int8_t         pe_act       [PE_ROWS][PE_COLS][LANES],

  input  int8_t         glb_wgt_data [3][PE_COLS][LANES],
  output int8_t         pe_wgt       [PE_ROWS][PE_COLS][LANES],

  input  int32_t        pe_psum_in   [PE_COLS][LANES],
  input  logic          psum_valid,
  output int32_t        glb_out_psum [PE_COLS][LANES],
  output logic          glb_out_wr_en [PE_COLS],

  input  int8_t         bypass_in    [LANES],
  output int8_t         bypass_out   [LANES],
  input  logic          bypass_en
);

  always_comb begin
    for (int row = 0; row < PE_ROWS; row++) begin
      for (int col = 0; col < PE_COLS; col++) begin
        for (int ln = 0; ln < LANES; ln++) begin
          case (cfg_pe_mode)
            PE_OS1, PE_GEMM: begin
              if (row == 0)
                pe_act[row][col][ln] = glb_in_data[0][col][ln];
              else
                pe_act[row][col][ln] = 8'sd0;
            end
            PE_PASS:
              pe_act[row][col][ln] = 8'sd0;
            PE_MP5:
              // Stream: same spatial row for all rows/cols (window uses row 0, col 0)
              pe_act[row][col][ln] = glb_in_data[rin_bank_sel][0][ln];
            default:
              // PE_RS3, PE_DW3, PE_DW7 — row = kh slice, col = parallel cin/channel
              pe_act[row][col][ln] = glb_in_data[row][col][ln];
          endcase
        end
      end
    end
  end

  always_comb begin
    for (int row = 0; row < PE_ROWS; row++) begin
      for (int col = 0; col < PE_COLS; col++) begin
        for (int ln = 0; ln < LANES; ln++) begin
          case (cfg_pe_mode)
            PE_OS1, PE_GEMM: begin
              if (row == 0)
                pe_wgt[row][col][ln] = glb_wgt_data[0][col][ln];
              else
                pe_wgt[row][col][ln] = 8'sd0;
            end
            PE_RS3, PE_DW3, PE_DW7: begin
              pe_wgt[row][col][ln] = glb_wgt_data[row][col][ln];
            end
            default: begin
              pe_wgt[row][col][ln] = 8'sd0;
            end
          endcase
        end
      end
    end
  end

  always_comb begin
    for (int col = 0; col < PE_COLS; col++) begin
      glb_out_wr_en[col] = psum_valid;
      for (int ln = 0; ln < LANES; ln++) begin
        glb_out_psum[col][ln] = pe_psum_in[col][ln];
      end
    end
  end

  always_comb begin
    for (int ln = 0; ln < LANES; ln++) begin
      bypass_out[ln] = bypass_en ? bypass_in[ln] : 8'sd0;
    end
  end

  // synthesis translate_off
`ifdef RTL_TRACE
  always @(posedge clk) begin
    if (rst_n)
      rtl_trace_pkg::rtl_trace_line("S5_RTR",
        $sformatf("mode=%0d rin=%0d a00=%0d a10=%0d a20=%0d w000=%0d w100=%0d w200=%0d",
                  cfg_pe_mode, rin_bank_sel,
                  pe_act[0][0][0], pe_act[1][0][0], pe_act[2][0][0],
                  pe_wgt[0][0][0], pe_wgt[1][0][0], pe_wgt[2][0][0]));
  end
`endif
  // synthesis translate_on

endmodule
