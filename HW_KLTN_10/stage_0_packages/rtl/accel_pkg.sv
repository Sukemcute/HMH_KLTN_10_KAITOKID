// ============================================================================
// Module : accel_pkg
// Project: YOLOv10n INT8 Accelerator — V4-VC707
// Description:
//   Global package containing ALL parameters, types, and enums for the
//   entire design.  Every RTL module imports this package.
//
//   V4 Key Changes vs V3:
//     - LANES           = 20  (from 32, perfect spatial alignment for YOLOv10n)
//     - N_SUBS_PER_SC   = 4   (from 3, enables Triple-RUNNING)
//     - TARGET_CLOCK_MHZ = 250 (from 200, enabled by narrower datapath)
//     - DSP_PIPE_DEPTH   = 5  (from 4, deeper pipeline for 250 MHz)
//     - GLB_INPUT_PAGES  = 2  (double-buffer for overlap fill/compute)
//
// Golden Rules enforced here:
//   RULE 1: Signed INT8 [-128, 127] — see int8_t typedef
//   RULE 3: INT32 accumulator, INT64 PPU multiply — see int32_t, int64_t
//   RULE 4: ACT_RELU as default activation
// ============================================================================
`timescale 1ns / 1ps

package accel_pkg;

  // ═══════════════════════════════════════════════════════════════════
  //  DATAPATH PARAMETERS
  // ═══════════════════════════════════════════════════════════════════
  parameter int LANES             = 20;   // ★ Spatial parallelism (magic number)
  parameter int PE_ROWS           = 3;    // kh parallelism (rows 0,1,2 → kh=0,1,2)
  parameter int PE_COLS           = 4;    // cout parallelism (per-column weight)
  parameter int NUM_PES           = PE_ROWS * PE_COLS;    // 12 PEs per cluster
  parameter int DSP_PAIRS_PER_PE  = LANES / 2;            // 10
  parameter int MACS_PER_PE       = LANES;                 // 20
  parameter int MACS_PER_SUB      = PE_ROWS * PE_COLS * MACS_PER_PE;  // 240

  // ═══════════════════════════════════════════════════════════════════
  //  HIERARCHY PARAMETERS
  // ═══════════════════════════════════════════════════════════════════
  parameter int N_SUPER_CLUSTERS  = 4;
  parameter int N_SUBS_PER_SC     = 4;    // ★ V4: 4 subs per SC (Triple-RUNNING)
  parameter int N_ACTIVE_PER_SC   = 3;    // ★ V4: 3 compute + 1 fill/drain
  parameter int N_TOTAL_SUBS      = N_SUPER_CLUSTERS * N_SUBS_PER_SC;  // 16
  parameter int N_TOTAL_ACTIVE    = N_SUPER_CLUSTERS * N_ACTIVE_PER_SC; // 12

  // ═══════════════════════════════════════════════════════════════════
  //  DATA WIDTH PARAMETERS
  // ═══════════════════════════════════════════════════════════════════
  parameter int ACT_WIDTH         = 8;    // Activation bit-width (signed INT8)
  parameter int WEIGHT_WIDTH      = 8;    // Weight bit-width (signed INT8)
  parameter int PSUM_WIDTH        = 32;   // Accumulator bit-width (signed INT32)
  parameter int BIAS_WIDTH        = 32;   // Bias bit-width (signed INT32)
  parameter int PPU_MULT_WIDTH    = 64;   // PPU multiply width (signed INT64)

  // ═══════════════════════════════════════════════════════════════════
  //  MEMORY PARAMETERS
  // ═══════════════════════════════════════════════════════════════════
  parameter int INPUT_BANKS       = 3;    // h mod 3 banking
  parameter int WEIGHT_BANKS      = 3;    // Per kernel row
  parameter int OUTPUT_BANKS      = 4;    // Per PE column
  parameter int WEIGHT_READ_PORTS = PE_COLS; // 4 (per-column weight)

  parameter int GLB_INPUT_DEPTH   = 2048; // Per bank per page
  parameter int GLB_INPUT_PAGES   = 2;    // ★ V4: double-buffer (ping-pong)
  parameter int GLB_WEIGHT_DEPTH  = 1024; // Per bank
  parameter int GLB_OUTPUT_DEPTH  = 512;  // Per bank (PSUM + ACT)

  // ═══════════════════════════════════════════════════════════════════
  //  PIPELINE PARAMETERS
  // ═══════════════════════════════════════════════════════════════════
  parameter int DSP_PIPE_DEPTH    = 5;    // ★ V4: 5-stage (from V3's 4)
  parameter int PPU_PIPE_DEPTH    = 5;    // ★ V4: 5-stage (from V3's 4)

  // ═══════════════════════════════════════════════════════════════════
  //  CLOCK / SYSTEM
  // ═══════════════════════════════════════════════════════════════════
  parameter int TARGET_CLOCK_MHZ  = 250;  // ★ V4: 250 MHz (from 200)
  parameter int AXI_ADDR_WIDTH    = 40;
  parameter int AXI_DATA_WIDTH    = 256;

  // ═══════════════════════════════════════════════════════════════════
  //  DESCRIPTOR / TILING LIMITS
  // ═══════════════════════════════════════════════════════════════════
  parameter int MAX_LAYERS        = 32;
  parameter int MAX_TILES         = 4096;
  parameter int BARRIER_COUNT     = 4;    // 4 skip-connection barriers
  parameter int WBLK_MAX          = 16;   // max(W_out / LANES) = 320/20 = 16
  parameter int CIN_TILE_MAX      = 384;  // Max Cin in any layer
  parameter int COUT_TILE_MAX     = 256;  // Max Cout in any layer
  parameter int H_TILE_MAX        = 80;   // Max tile height

  // ═══════════════════════════════════════════════════════════════════
  //  PE MODE ENUMERATION
  //  Selects which dataflow the subcluster executes for a given tile.
  //  All modes run on the SAME hardware — only descriptor config changes.
  // ═══════════════════════════════════════════════════════════════════
  typedef enum logic [3:0] {
    PE_RS3  = 4'd0,   // Conv 3×3 row-stationary   (P0: RS_DENSE_3x3)
    PE_OS1  = 4'd1,   // Conv 1×1 output-stationary (P1: OS_1x1)
    PE_DW3  = 4'd2,   // Depthwise 3×3             (P2: DW_3x3)
    PE_DW7  = 4'd3,   // Depthwise 7×7 multi-pass  (P8: DW_7x7_MULTIPASS)
    PE_MP5  = 4'd4,   // MaxPool 5×5               (P3: MAXPOOL_5x5)
    PE_PASS = 4'd5,   // Bypass (Move/Concat/Upsample) (P4/P5/P6)
    PE_GEMM = 4'd6    // GEMM for attention          (P9/P10: QPSA matmul)
  } pe_mode_e;

  // ═══════════════════════════════════════════════════════════════════
  //  ACTIVATION MODE ENUMERATION
  // ═══════════════════════════════════════════════════════════════════
  typedef enum logic [1:0] {
    ACT_NONE  = 2'd0,  // Identity (no activation)
    ACT_RELU  = 2'd1,  // y = max(0, x)  — RULE 4: model uses ReLU
    ACT_SILU  = 2'd2,  // LUT-based SiLU (unused for current model)
    ACT_RELU6 = 2'd3   // y = clamp(x, 0, 6_quant)
  } act_mode_e;

  // ═══════════════════════════════════════════════════════════════════
  //  SWIZZLE MODE ENUMERATION
  // ═══════════════════════════════════════════════════════════════════
  typedef enum logic [1:0] {
    SWZ_NORMAL     = 2'd0,  // Identity pass-through
    SWZ_UPSAMPLE2X = 2'd1,  // Nearest-neighbor 2× (P6)
    SWZ_CONCAT     = 2'd2,  // Channel concatenation (P5)
    SWZ_EWISE_ADD  = 2'd3   // Element-wise add with domain alignment (P7)
  } swizzle_mode_e;

  // ═══════════════════════════════════════════════════════════════════
  //  TILE FSM STATES (10 states)
  // ═══════════════════════════════════════════════════════════════════
  typedef enum logic [3:0] {
    TS_IDLE       = 4'd0,
    TS_LOAD_DESC  = 4'd1,
    TS_PREFILL_WT = 4'd2,
    TS_PREFILL_IN = 4'd3,
    TS_COMPUTE    = 4'd4,
    TS_PE_DRAIN   = 4'd5,
    TS_PPU_RUN    = 4'd6,
    TS_SWIZZLE    = 4'd7,
    TS_WRITEBACK  = 4'd8,
    TS_DONE       = 4'd9
  } tile_state_e;

  // ═══════════════════════════════════════════════════════════════════
  //  SUBCLUSTER ROLE (for arbiter scheduling)
  // ═══════════════════════════════════════════════════════════════════
  typedef enum logic [1:0] {
    ROLE_IDLE    = 2'd0,
    ROLE_COMPUTE = 2'd1,
    ROLE_FILL    = 2'd2,
    ROLE_DRAIN   = 2'd3
  } sub_role_e;

  // ═══════════════════════════════════════════════════════════════════
  //  PSUM / ACT NAMESPACE (dual namespace for GLB output bank)
  // ═══════════════════════════════════════════════════════════════════
  typedef enum logic {
    NS_PSUM = 1'b0,   // INT32 partial sums (multipass accumulation)
    NS_ACT  = 1'b1    // INT8 final activations (PPU output)
  } namespace_e;

  // ═══════════════════════════════════════════════════════════════════
  //  QUANTIZATION MODE
  // ═══════════════════════════════════════════════════════════════════
  typedef enum logic [1:0] {
    QMODE_PER_TENSOR  = 2'd0,
    QMODE_PER_CHANNEL = 2'd1,
    QMODE_NONE        = 2'd2
  } quant_mode_e;

  // ═══════════════════════════════════════════════════════════════════
  //  CONVENIENCE TYPEDEFS
  //  RULE 1: Signed INT8 everywhere.
  //  RULE 3: INT32 accumulator, INT64 PPU multiply.
  // ═══════════════════════════════════════════════════════════════════
  typedef logic signed [7:0]   int8_t;
  typedef logic signed [15:0]  int16_t;
  typedef logic signed [31:0]  int32_t;
  typedef logic signed [63:0]  int64_t;
  typedef logic        [7:0]   uint8_t;
  typedef logic        [31:0]  uint32_t;

endpackage
