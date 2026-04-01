`timescale 1ns/1ps
package accel_pkg;

  // ═══════════ Compute Array Parameters ═══════════
  parameter int LANES          = 32;
  parameter int PE_ROWS        = 3;
  parameter int PE_COLS        = 4;
  parameter int NUM_PES        = PE_ROWS * PE_COLS;  // 12
  parameter int MACS_PER_SUB   = PE_ROWS * PE_COLS * LANES;  // 384
  parameter int DSP_PAIRS      = LANES / 2;  // 16

  // ═══════════ Memory Parameters ═══════════
  parameter int INPUT_BANKS    = 3;
  parameter int OUTPUT_BANKS   = 4;
  parameter int WEIGHT_BANKS   = 3;
  parameter int PSUM_WIDTH     = 32;
  parameter int ACT_WIDTH      = 8;
  parameter int WEIGHT_WIDTH   = 8;
  parameter int BIAS_WIDTH     = 32;

  // ═══════════ System Parameters ═══════════
  parameter int SUPER_CLUSTERS   = 4;
  parameter int SUBS_PER_SC      = 4;
  parameter int ACTIVE_PER_SC    = 2;
  parameter int EXT_PORT_WIDTH   = 256;
  parameter int EXT_PORT_BYTES   = EXT_PORT_WIDTH / 8;  // 32
  parameter int AXI_ADDR_WIDTH   = 40;
  parameter int AXI_DATA_WIDTH   = 256;

  // ═══════════ Descriptor Parameters ═══════════
  parameter int DESC_WIDTH       = 512;
  parameter int MAX_LAYERS       = 32;
  parameter int MAX_TILES        = 4096;
  parameter int BARRIER_BITS     = 32;

  // ═══════════ SiLU LUT ═══════════
  parameter int SILU_LUT_DEPTH   = 256;
  parameter int SILU_LUT_WIDTH   = 8;

  // ═══════════ Derived ═══════════
  parameter int WBLK_MAX         = 20;
  parameter int CIN_TILE_MAX     = 256;
  parameter int COUT_TILE_MAX    = 256;
  parameter int H_TILE_MAX       = 80;

  // ═══════════ PE Mode Enum ═══════════
  typedef enum logic [3:0] {
    PE_RS3     = 4'h0,
    PE_OS1     = 4'h1,
    PE_DW3     = 4'h2,
    PE_DW7     = 4'h3,
    PE_MP5     = 4'h4,
    PE_GEMM    = 4'h5,
    PE_PASS    = 4'h6
  } pe_mode_e;

  // ═══════════ Activation Mode ═══════════
  typedef enum logic [1:0] {
    ACT_NONE   = 2'h0,
    ACT_SILU   = 2'h1,
    ACT_RELU   = 2'h2,
    ACT_CLAMP  = 2'h3
  } act_mode_e;

  // ═══════════ Subcluster Role ═══════════
  typedef enum logic [2:0] {
    ROLE_IDLE     = 3'h0,
    ROLE_RUNNING  = 3'h1,
    ROLE_FILLING  = 3'h2,
    ROLE_DRAINING = 3'h3
  } sc_role_e;

  // ═══════════ Tile FSM States ═══════════
  typedef enum logic [3:0] {
    TILE_IDLE          = 4'h0,
    TILE_LOAD_CFG      = 4'h1,
    TILE_PREFILL_WT    = 4'h2,
    TILE_PREFILL_IN    = 4'h3,
    TILE_PREFILL_SKIP  = 4'hA,
    TILE_WAIT_READY    = 4'h4,
    TILE_RUN_COMPUTE   = 4'h5,
    TILE_ACCUMULATE    = 4'h6,
    TILE_POST_PROCESS  = 4'h7,
    TILE_SWIZZLE_STORE = 4'h8,
    TILE_DONE          = 4'h9
  } tile_state_e;

  // ═══════════ Quant Mode ═══════════
  typedef enum logic [1:0] {
    QMODE_PER_TENSOR   = 2'h0,
    QMODE_PER_CHANNEL  = 2'h1,
    QMODE_NONE         = 2'h2
  } quant_mode_e;

  // ═══════════ PSUM/ACT Namespace ═══════════
  typedef enum logic {
    NS_PSUM = 1'b0,
    NS_ACT  = 1'b1
  } namespace_e;

endpackage
