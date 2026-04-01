`timescale 1ns/1ps
// ============================================================================
// YOLOv10n INT8 Accelerator Package
// Extends PHASE_3 accel_pkg with primitive engine definitions
// Target: Xilinx Virtex-7 XC7VX690T @ 200-250 MHz
// ============================================================================
package yolo_accel_pkg;

  // Import PE and activation types from PHASE_3 accel_pkg for compatibility
  // with pe_unit, column_reduce, and other PHASE_3 leaf modules.
  import accel_pkg::pe_mode_e;
  import accel_pkg::PE_RS3;
  import accel_pkg::PE_OS1;
  import accel_pkg::PE_DW3;
  import accel_pkg::PE_DW7;
  import accel_pkg::PE_MP5;
  import accel_pkg::PE_GEMM;
  import accel_pkg::PE_PASS;
  import accel_pkg::act_mode_e;
  import accel_pkg::ACT_NONE;
  import accel_pkg::ACT_SILU;
  import accel_pkg::ACT_RELU;
  import accel_pkg::ACT_CLAMP;
  import accel_pkg::namespace_e;
  import accel_pkg::NS_PSUM;
  import accel_pkg::NS_ACT;
  import accel_pkg::quant_mode_e;
  import accel_pkg::QMODE_PER_TENSOR;
  import accel_pkg::QMODE_PER_CHANNEL;
  import accel_pkg::QMODE_NONE;

  // ═══════════ Compute Array Parameters (V2 Architecture) ═══════════
  parameter int LANES          = 32;         // Spatial parallelism per PE
  parameter int PE_ROWS        = 3;          // Kernel-row parallelism
  parameter int PE_COLS        = 4;          // Height parallelism
  parameter int NUM_PES        = PE_ROWS * PE_COLS;  // 12 PEs per subcluster
  parameter int MACS_PER_SUB   = PE_ROWS * PE_COLS * LANES;  // 384
  parameter int DSP_PAIRS      = LANES / 2;  // 16 DSP48E1 pairs per PE

  // ═══════════ Data Widths ═══════════
  parameter int ACT_WIDTH      = 8;          // INT8 activation
  parameter int WEIGHT_WIDTH   = 8;          // INT8 weight
  parameter int PSUM_WIDTH     = 32;         // INT32 partial sum
  parameter int BIAS_WIDTH     = 32;         // INT32 bias (fused BN)
  parameter int MULT_WIDTH     = 64;         // INT64 for requant multiply

  // ═══════════ Memory Parameters ═══════════
  parameter int INPUT_BANKS    = 3;          // h mod 3 banking
  parameter int OUTPUT_BANKS   = 4;          // out_row mod 4 banking
  parameter int WEIGHT_BANKS   = 3;          // Per kernel row

  // ═══════════ System Parameters ═══════════
  parameter int SUPER_CLUSTERS = 4;
  parameter int SUBS_PER_SC    = 4;
  parameter int ACTIVE_PER_SC  = 2;          // Dual-RUNNING mode
  parameter int EXT_PORT_WIDTH = 256;        // 32 bytes per beat
  parameter int EXT_PORT_BYTES = EXT_PORT_WIDTH / 8;

  // ═══════════ Pipeline Latencies ═══════════
  parameter int DSP_LATENCY    = 4;          // dsp_pair_int8: 4 pipeline stages
  parameter int PE_LATENCY     = DSP_LATENCY;
  parameter int COL_RED_LATENCY = 1;         // column_reduce: 1 cycle
  parameter int PPU_LATENCY    = 4;          // PPU: 4 pipeline stages
  parameter int POOL_LATENCY   = 5;          // comparator_tree: 5 stages

  // ═══════════ SiLU LUT ═══════════
  parameter int SILU_LUT_DEPTH = 256;
  parameter int SILU_LUT_WIDTH = 8;

  // ═══════════ Max Tile Dimensions ═══════════
  parameter int MAX_W_PAD      = 672;        // ceil(640+2, 32) padded width
  parameter int MAX_H_PAD      = 642;        // 640 + 2 padded height
  parameter int MAX_CIN        = 384;        // Max input channels (L13 QConcat)
  parameter int MAX_COUT       = 256;        // Max output channels
  parameter int MAX_WBLK       = 21;         // ceil(672/32)

  // pe_mode_e and act_mode_e are imported from accel_pkg above
  // (PE_RS3, PE_OS1, PE_DW3, PE_DW7, PE_MP5, PE_GEMM, PE_PASS)
  // (ACT_NONE, ACT_SILU, ACT_RELU, ACT_CLAMP)

  // ═══════════ Primitive ID (P0-P9) ═══════════
  typedef enum logic [3:0] {
    PRIM_RS_DENSE_3x3       = 4'd0,   // Conv 3x3 dense (L0,1,3,17, QC2f内部)
    PRIM_OS_1x1             = 4'd1,   // Conv 1x1 (QC2f cv1/cv2, SCDown, SPPF)
    PRIM_DW_3x3             = 4'd2,   // Depthwise 3x3 (SCDown)
    PRIM_MAXPOOL_5x5        = 4'd3,   // MaxPool 5x5 (SPPF)
    PRIM_MOVE               = 4'd4,   // Buffer move (skip connections)
    PRIM_CONCAT             = 4'd5,   // Channel concat (FPN/PAN)
    PRIM_UPSAMPLE_NEAREST   = 4'd6,   // 2x nearest upsample (Neck)
    PRIM_EWISE_ADD          = 4'd7,   // Element-wise add (residual)
    PRIM_DW_7x7_MULTIPASS   = 4'd8,   // Depthwise 7x7 multipass (QC2fCIB)
    PRIM_GEMM_ATTN          = 4'd9    // GEMM for attention (QPSA)
  } prim_id_e;

  // quant_mode_e and namespace_e imported from accel_pkg above

  // ═══════════ Engine FSM States ═══════════
  typedef enum logic [3:0] {
    ENG_IDLE          = 4'h0,
    ENG_LOAD_ROWS     = 4'h1,   // Load input rows into row buffers
    ENG_LOAD_WGT      = 4'h2,   // Load weights for current (cout, cin)
    ENG_COMPUTE       = 4'h3,   // Feed PE with kw iterations
    ENG_DRAIN         = 4'h4,   // Wait PE pipeline drain
    ENG_COL_REDUCE    = 4'h5,   // Sum PE outputs across kernel rows
    ENG_PPU           = 4'h6,   // Post-processing
    ENG_WRITE         = 4'h7,   // Write output
    ENG_NEXT_CIN      = 4'h8,   // Advance cin loop
    ENG_NEXT_WBLK     = 4'h9,   // Advance wblk loop
    ENG_NEXT_COUT     = 4'hA,   // Advance cout loop
    ENG_NEXT_HOUT     = 4'hB,   // Advance hout loop
    ENG_DONE          = 4'hF
  } eng_state_e;

  // namespace_e (NS_PSUM, NS_ACT) imported from accel_pkg above

  // ═══════════ Helper Functions ═══════════

  // Saturating clamp to INT8 [-128, 127]
  function automatic logic signed [7:0] clamp_int8(input logic signed [31:0] val);
    if (val > 32'sd127) return 8'sd127;
    if (val < -32'sd128) return -8'sd128;
    return val[7:0];
  endfunction

  // Fixed-point requantization with half-up rounding
  // result = (acc * m_int + round_bias) >>> shift
  function automatic logic signed [31:0] requant_fixed(
    input logic signed [31:0] acc,
    input logic signed [31:0] m_int,
    input logic [5:0]         shift_val
  );
    automatic logic signed [63:0] mult;
    automatic logic signed [63:0] rounded;
    automatic logic signed [31:0] shifted;

    mult = 64'(acc) * 64'(m_int);
    if (shift_val > 0)
      rounded = mult + (64'sd1 <<< (shift_val - 1));  // half_up
    else
      rounded = mult;
    shifted = 32'(rounded >>> shift_val);
    return shifted;
  endfunction

  // SiLU LUT index from signed value
  function automatic logic [7:0] silu_index(input logic signed [15:0] val);
    automatic int idx;
    idx = int'(val) + 128;
    if (idx < 0)   return 8'd0;
    if (idx > 255) return 8'd255;
    return idx[7:0];
  endfunction

  // ═══════════ YOLOv10n Layer Configuration Table ═══════════
  // Layer 0-22 primitive decomposition
  // (used for testbench configuration, not synthesized)

  typedef struct packed {
    prim_id_e    prim_id;
    logic [8:0]  cin, cout;
    logic [9:0]  hin, win, hout, wout;
    logic [2:0]  kh, kw;
    logic [1:0]  stride;
    logic [3:0]  pad;
    act_mode_e   act_mode;
  } layer_config_t;

endpackage
