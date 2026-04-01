// ============================================================================
// Module : desc_pkg
// Project: YOLOv10n INT8 Accelerator — V4-VC707
// Description:
//   Descriptor struct definitions for the 3-level descriptor hierarchy:
//     1. net_desc_t   — 1 per inference run
//     2. layer_desc_t — 23 per inference (L0-L22)
//     3. tile_desc_t  — ~60 total per inference
//
//   These structs are fetched from DDR3 by desc_fetch_engine and dispatched
//   to subclusters via global_scheduler.
// ============================================================================
`timescale 1ns / 1ps

package desc_pkg;
  import accel_pkg::*;

  // ═══════════════════════════════════════════════════════════════════
  //  NET DESCRIPTOR — 1 per inference, describes the whole model
  // ═══════════════════════════════════════════════════════════════════
  typedef struct packed {
    logic [31:0] magic;               // 0xACC10004 (V4 identifier)
    logic [15:0] version;             // Descriptor format version
    logic [7:0]  num_layers;          // Number of layers (23 for L0-L22)
    logic [7:0]  reserved0;
    logic [63:0] layer_table_base;    // DDR3 base address for layer descriptors
    logic [63:0] weight_arena_base;   // DDR3 base address for weight data
    logic [63:0] act0_arena_base;     // DDR3 base for activation buffer 0
    logic [63:0] act1_arena_base;     // DDR3 base for activation buffer 1
  } net_desc_t;

  // ═══════════════════════════════════════════════════════════════════
  //  LAYER DESCRIPTOR — 1 per layer (23 total for L0-L22)
  // ═══════════════════════════════════════════════════════════════════
  typedef struct packed {
    // — Identity —
    logic [4:0]       layer_id;       // 0..22
    pe_mode_e         pe_mode;        // PE_RS3, PE_OS1, PE_DW3, ...
    act_mode_e        activation;     // ACT_RELU, ACT_NONE, ...
    // — Dimensions —
    logic [9:0]       cin;            // Input channels
    logic [9:0]       cout;           // Output channels
    logic [9:0]       hin, win;       // Input spatial size
    logic [9:0]       hout, wout;     // Output spatial size
    // — Kernel —
    logic [3:0]       kh, kw;         // Kernel height/width (1,3,5,7)
    logic [2:0]       stride;         // 1 or 2
    logic [2:0]       padding;        // 0, 1, 2, or 3
    // — Tiling —
    logic [7:0]       num_tiles;      // Number of tiles for this layer
    logic [3:0]       num_cin_pass;   // Multi-pass cin accumulation
    logic [3:0]       num_k_pass;     // DW7x7 multipass (=3 for 7x7)
    // — Routing & Post —
    swizzle_mode_e    swizzle;        // Post-compute layout transform
    logic [7:0]       router_profile_id;
    logic [7:0]       post_profile_id;
    // — Flags —
    logic [15:0]      layer_flags;    // Reserved flags
  } layer_desc_t;

  // ═══════════════════════════════════════════════════════════════════
  //  TILE DESCRIPTOR — N per layer (~60 total for L0-L22)
  // ═══════════════════════════════════════════════════════════════════
  typedef struct packed {
    // — Identity —
    logic [15:0]      tile_id;
    logic [4:0]       layer_id;
    logic [3:0]       sc_mask;        // Which SuperClusters handle this tile
    // — Tile origin in output space —
    logic [9:0]       h_out0;         // Starting output row
    logic [9:0]       wblk0;          // Starting output column block
    logic [9:0]       cin0;           // Starting input channel
    logic [9:0]       cout0;          // Starting output channel
    // — DDR3 memory offsets —
    logic [31:0]      src_in_off;     // Input activation offset
    logic [31:0]      src_w_off;      // Weight offset
    logic [31:0]      src_skip_off;   // Skip connection offset
    logic [31:0]      dst_off;        // Output destination offset
    // — Tile extent —
    logic [5:0]       valid_h;        // Valid output rows in this tile
    logic [5:0]       valid_w;        // Valid output cols in this tile
    // — Multi-pass —
    logic [3:0]       first_cin_pass;
    logic [3:0]       num_cin_pass;
    logic [3:0]       first_k_pass;
    logic [3:0]       num_k_pass;
    // — Flags —
    logic             first_tile;     // First tile of this layer
    logic             last_tile;      // Last tile of this layer
    logic             hold_skip;      // Hold output for skip connection
    logic             need_swizzle;   // Apply swizzle_engine post-compute
    logic             barrier_wait;   // Wait for barrier before starting
    logic [3:0]       barrier_id;     // Which barrier to wait/signal (0-3)
    logic [7:0]       tile_flags;     // Reserved flags
  } tile_desc_t;

  // ═══════════════════════════════════════════════════════════════════
  //  POST-PROCESSING PROFILE (PPU configuration)
  // ═══════════════════════════════════════════════════════════════════
  typedef struct packed {
    logic             bias_en;
    quant_mode_e      quant_mode;
    act_mode_e        act_mode;
    logic             ewise_en;       // Element-wise add enable
    logic [31:0]      bias_scale_offset; // DDR offset to bias/scale arrays
    logic [7:0]       concat_ch_offset;  // Channel offset for concat
    logic [1:0]       upsample_factor;   // 2x upsample
  } post_profile_t;

  // ═══════════════════════════════════════════════════════════════════
  //  ROUTER PROFILE (data movement configuration)
  // ═══════════════════════════════════════════════════════════════════
  typedef struct packed {
    logic [2:0]       rin_src   [3];  // Input bank → PE row mapping
    logic [2:0]       rwt_src   [3];  // Weight bank → PE row mapping
    logic             rwt_h_multicast; // Multicast same activation to all rows
    logic [1:0]       rps_accum_mode;  // Accumulation mode
    logic             bypass_en;       // Bypass PE cluster
  } router_profile_t;

endpackage
