// ============================================================================
// Module : pe_unit
// Project: YOLOv10n INT8 Accelerator — V4-VC707
// Description:
//   Single Processing Element with LANES=20 parallel MAC lanes.
//   Contains 10 dsp_pair_int8 instances, each handling 2 adjacent lanes.
//
//   Weight selection by pe_mode:
//     PE_OS1: Broadcast w_in[0] to ALL lanes (1×1 conv)
//     Others: Per-lane weight from w_in array (but within each DSP pair,
//             both lanes share the same weight — guaranteed by architecture)
//
//   In ALL pe_modes, weight is uniform across spatial lanes within a PE:
//     - RS3/DW3/DW7/GEMM: same weight for all 20 spatial positions
//     - OS1: explicitly broadcast
//   This is WHY the 2-MAC packed DSP (shared weight) always works.
//
//   Latency: 5 cycles (matches dsp_pair_int8 pipeline depth)
//   Throughput: 20 MACs per cycle (sustained after fill)
//   Resources: 10 DSP48E1 + ~250 LUT + ~400 FF per instance
//
// Instances: 12 per PE cluster × 16 subclusters = 192 total
// ============================================================================
`timescale 1ns / 1ps

module pe_unit
  import accel_pkg::*;
#(
  parameter int LANES = accel_pkg::LANES,  // 20
  parameter bit TRACE_CLUSTER = 1'b0     // simulation: 1 = log this PE only
)(
  input  logic              clk,
  input  logic              rst_n,

  // — Configuration —
  input  pe_mode_e          pe_mode,   // Selects weight routing

  // — Data inputs —
  input  logic signed [7:0] x_in [LANES],  // 20 activation values (per-lane)
  input  logic signed [7:0] w_in [LANES],  // 20 weight values (see note below)
  //   NOTE: w_in[0..19] are identical in ALL modes for this PE.
  //   In RS3/DW3/DW7: weight is per (cout,cin,kh,kw), same for all spatial positions.
  //   In OS1: weight is per (cout,cin), broadcast to all positions.
  //   The per-lane array exists for interface uniformity with router_cluster.

  // — Control —
  input  logic              pe_enable,   // Enable MAC accumulation
  input  logic              clear_acc,   // Reset accumulators (new tile/cout_group)

  // — Accumulated outputs —
  output logic signed [31:0] psum_out [LANES],  // 20 accumulated partial sums
  output logic               psum_valid          // Delayed valid from pipeline
);

  // ════════════════════════════════════════════════════════════════
  //  VALID PIPELINE TRACKER
  //  Track when first valid data exits the 5-stage DSP pipeline.
  // ════════════════════════════════════════════════════════════════
  localparam int PIPE_DEPTH = DSP_PIPE_DEPTH;  // 5
  logic [PIPE_DEPTH:0] valid_sr;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      valid_sr <= '0;
    else
      valid_sr <= {valid_sr[PIPE_DEPTH-1:0], pe_enable};
  end

  assign psum_valid = valid_sr[PIPE_DEPTH];

  // ════════════════════════════════════════════════════════════════
  //  WEIGHT SELECTION LOGIC
  //  In OS1 mode, broadcast w_in[0] to all lanes.
  //  In all other modes, use per-lane weight (which is already uniform
  //  within the PE — router_cluster ensures this).
  //  For each DSP pair, we select the weight of the even lane (2*g).
  // ════════════════════════════════════════════════════════════════
  logic signed [7:0] w_sel [LANES];

  always_comb begin
    case (pe_mode)
      PE_OS1: begin
        // 1×1 conv: broadcast weight[0] to all lanes
        for (int l = 0; l < LANES; l++)
          w_sel[l] = w_in[0];
      end
      default: begin
        // RS3, DW3, DW7, MP5, GEMM, PASS: use per-lane weight
        for (int l = 0; l < LANES; l++)
          w_sel[l] = w_in[l];
      end
    endcase
  end

  // ════════════════════════════════════════════════════════════════
  //  DSP PAIR INSTANTIATION — 10 pairs for 20 lanes
  //  Each pair (g) handles lanes [2g] and [2g+1].
  //  Both lanes in a pair share the weight from lane [2g].
  // ════════════════════════════════════════════════════════════════
  localparam int NUM_PAIRS = LANES / 2;  // 10

  genvar g;
  generate
    for (g = 0; g < NUM_PAIRS; g++) begin : gen_dsp_pair
      //
      // DSP pair #g: handles lane 2g (psum_a) and lane 2g+1 (psum_b)
      // Weight is shared: w_sel[2g] == w_sel[2g+1] by architecture guarantee
      //
      dsp_pair_int8
`ifdef RTL_TRACE
      #(
        .TRACE_EN((g == 0) ? 1'b1 : 1'b0)
      )
`endif
      u_dsp_pair (
        .clk    (clk),
        .rst_n  (rst_n),
        .x_a    (x_in[2*g]),         // Activation for even lane
        .x_b    (x_in[2*g + 1]),     // Activation for odd lane
        .w      (w_sel[2*g]),         // Shared weight for both lanes
        .en     (pe_enable),
        .clear  (clear_acc),
        .psum_a (psum_out[2*g]),      // Accumulated result for even lane
        .psum_b (psum_out[2*g + 1])   // Accumulated result for odd lane
      );
    end
  endgenerate

  // synthesis translate_off
`ifdef RTL_TRACE
  always @(posedge clk) begin
    if (rst_n && TRACE_CLUSTER && pe_enable)
      rtl_trace_pkg::rtl_trace_line("S1_PEU",
        $sformatf("mode=%0d x0=%0d x1=%0d w0=%0d clr=%b psum_v=%b p0=%0d",
                  pe_mode, x_in[0], x_in[1], w_sel[0], clear_acc, psum_valid, psum_out[0]));
  end
`endif
  // synthesis translate_on

endmodule
