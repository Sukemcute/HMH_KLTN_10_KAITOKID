// ============================================================================
// Module : clk_wiz_250
// Project: YOLOv10n INT8 Accelerator — V4-VC707
//
// MMCM wrapper: 200 MHz board clock → 250 MHz accelerator clock.
// For simulation: simple pass-through (no PLL model needed).
// For synthesis: instantiates Xilinx MMCME2_ADV.
// ============================================================================
`timescale 1ns / 1ps

module clk_wiz_250 (
  input  logic clk_in_200,
  input  logic rst_n,
  output logic clk_out_250,
  output logic locked
);

`ifdef SYNTHESIS
  // Xilinx 7-Series MMCM
  wire clk_fb, clk_mmcm;

  MMCME2_ADV #(
    .CLKFBOUT_MULT_F  (5.0),    // VCO = 200 * 5 = 1000 MHz
    .CLKOUT0_DIVIDE_F (4.0),    // 1000 / 4 = 250 MHz
    .CLKIN1_PERIOD     (5.0),   // 200 MHz = 5ns
    .DIVCLK_DIVIDE     (1)
  ) u_mmcm (
    .CLKIN1    (clk_in_200),
    .CLKFBIN   (clk_fb),
    .CLKFBOUT  (clk_fb),
    .CLKOUT0   (clk_mmcm),
    .RST       (!rst_n),
    .LOCKED    (locked),
    .PWRDWN    (1'b0),
    .CLKIN2    (1'b0),
    .CLKINSEL  (1'b1),
    .DADDR     (7'd0),
    .DCLK      (1'b0),
    .DEN       (1'b0),
    .DI        (16'd0),
    .DWE       (1'b0),
    .PSCLK     (1'b0),
    .PSEN      (1'b0),
    .PSINCDEC  (1'b0)
  );

  BUFG u_bufg (.I(clk_mmcm), .O(clk_out_250));

`else
  // Simulation: pass-through
  assign clk_out_250 = clk_in_200;
  logic [1:0] lock_cnt;
  always_ff @(posedge clk_in_200 or negedge rst_n) begin
    if (!rst_n)
      lock_cnt <= 2'd0;
    else if (lock_cnt < 2'd3)
      lock_cnt <= lock_cnt + 2'd1;
  end
  assign locked = (lock_cnt == 2'd3);
`endif

endmodule
