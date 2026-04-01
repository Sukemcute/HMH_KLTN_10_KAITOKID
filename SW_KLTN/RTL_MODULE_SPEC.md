# RTL MODULE SPECIFICATION – YOLOv10n INT8 Accelerator V2
## Danh Sách Module, Interface, Logic Chi Tiết Cho Prompt Verilog

> **Mục tiêu**: Mô tả ĐẦY ĐỦ mọi module RTL cần xây dựng, đủ chi tiết để prompt sinh Verilog.
> **Kiến trúc**: V2 (LANES=32, dual-RUNNING, 3,072 active MACs)
> **Tham chiếu**: `HW_ARCHITECTURE_V2_100FPS.md`, `HW_MAPPING_RESEARCH.md`

---

# PHẦN A: TỔNG QUAN CÂY MODULE

```
accel_top
├── controller_system
│   ├── csr_mmio
│   ├── desc_fetch_engine
│   │   └── axi_read_master
│   ├── barrier_manager
│   └── global_scheduler
│
├── supercluster_wrapper [×4]
│   ├── local_arbiter
│   └── subcluster_wrapper [×4 per SC]
│       ├── tile_fsm
│       ├── shadow_reg_file
│       ├── glb_bank
│       │   ├── glb_input_bank [×3]
│       │   ├── glb_weight_bank [×3]
│       │   ├── glb_output_bank [×4]
│       │   ├── metadata_ram
│       │   ├── addr_gen_input
│       │   ├── addr_gen_weight
│       │   └── addr_gen_output
│       │
│       ├── router_cluster
│       │   ├── rin_router [×3]
│       │   ├── rwt_router [×3]
│       │   └── rps_router [×4]
│       │
│       ├── window_gen
│       ├── pe_cluster
│       │   ├── dsp_pair_int8 [×12 PEs × 16 DSP pairs]
│       │   ├── column_reduce
│       │   └── comparator_tree (MAXPOOL mode)
│       │
│       ├── ppu
│       │   ├── bias_add_unit
│       │   ├── requant_unit
│       │   ├── silu_lut
│       │   ├── clamp_unit
│       │   └── ewise_add_unit
│       │
│       └── swizzle_engine
│
├── tensor_dma
│   ├── axi_read_master
│   └── axi_write_master
│
└── perf_mon
```

**Tổng số module files cần viết: ~35 files**

---

# PHẦN B: LEVEL 0 – PACKAGE DEFINITIONS

---

## M00. `accel_pkg.sv` — Hằng số và kiểu dữ liệu chung

**Mục đích**: Định nghĩa tất cả tham số toàn cục, kiểu enum, struct dùng chung bởi mọi module.

```systemverilog
package accel_pkg;

  // ═══════════ Compute Array Parameters ═══════════
  parameter int LANES          = 32;    // spatial positions parallel per PE
  parameter int PE_ROWS        = 3;     // reduction rows (matches 3×3 kernel)
  parameter int PE_COLS        = 4;     // output rows parallel
  parameter int NUM_PES        = PE_ROWS * PE_COLS; // = 12
  parameter int MACS_PER_SUB   = PE_ROWS * PE_COLS * LANES; // = 384
  parameter int DSP_PAIRS      = LANES / 2; // = 16 DSP48E1 per PE

  // ═══════════ Memory Parameters ═══════════
  parameter int INPUT_BANKS    = 3;     // h mod 3 banking
  parameter int OUTPUT_BANKS   = 4;     // PE_COLS output streams
  parameter int WEIGHT_BANKS   = 3;     // 3 reduction lanes
  parameter int PSUM_WIDTH     = 32;    // INT32 accumulator
  parameter int ACT_WIDTH      = 8;     // INT8 activation
  parameter int WEIGHT_WIDTH   = 8;     // INT8 weight
  parameter int BIAS_WIDTH     = 32;    // INT32 bias

  // ═══════════ System Parameters ═══════════
  parameter int SUPER_CLUSTERS   = 4;
  parameter int SUBS_PER_SC      = 4;
  parameter int ACTIVE_PER_SC    = 2;   // dual-RUNNING
  parameter int EXT_PORT_WIDTH   = 256; // bits, external memory port
  parameter int EXT_PORT_BYTES   = EXT_PORT_WIDTH / 8; // = 32
  parameter int AXI_ADDR_WIDTH   = 40;
  parameter int AXI_DATA_WIDTH   = 256;

  // ═══════════ Descriptor Parameters ═══════════
  parameter int DESC_WIDTH       = 512; // 64 bytes = 512 bits per descriptor
  parameter int MAX_LAYERS       = 32;
  parameter int MAX_TILES        = 4096;
  parameter int BARRIER_BITS     = 32;

  // ═══════════ SiLU LUT ═══════════
  parameter int SILU_LUT_DEPTH   = 256;
  parameter int SILU_LUT_WIDTH   = 8;

  // ═══════════ Derived ═══════════
  parameter int WBLK_MAX         = 20;  // ceil(640/32)
  parameter int CIN_TILE_MAX     = 256;
  parameter int COUT_TILE_MAX    = 256;
  parameter int H_TILE_MAX       = 80;

  // ═══════════ PE Mode Enum ═══════════
  typedef enum logic [3:0] {
    PE_RS3     = 4'h0,  // RS_DENSE_3x3 (conv 3×3, stride 1 or 2)
    PE_OS1     = 4'h1,  // OS_1x1 (pointwise conv)
    PE_DW3     = 4'h2,  // DW_3x3 (depthwise 3×3)
    PE_DW7     = 4'h3,  // DW_7x7_MULTIPASS (3 passes)
    PE_MP5     = 4'h4,  // MAXPOOL_5x5 (comparator tree)
    PE_GEMM    = 4'h5,  // GEMM_ATTN (matrix multiply)
    PE_PASS    = 4'h6   // bypass (MOVE, CONCAT, UPSAMPLE)
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
    ROLE_DRAINING = 3'h3,
    ROLE_HOLD     = 3'h4
  } sc_role_e;

  // ═══════════ Tile FSM States ═══════════
  typedef enum logic [3:0] {
    TILE_IDLE          = 4'h0,
    TILE_LOAD_CFG      = 4'h1,
    TILE_PREFILL_WT    = 4'h2,
    TILE_PREFILL_IN    = 4'h3,
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
    NS_PSUM = 1'b0,  // INT32 partial sum (intermediate)
    NS_ACT  = 1'b1   // INT8 activation (final after PPU)
  } namespace_e;

endpackage
```

---

## M01. `desc_pkg.sv` — Descriptor Struct Definitions

**Mục đích**: Định nghĩa các struct cho NET_DESC, LAYER_DESC, TILE_DESC, POST_PROFILE, ROUTER_PROFILE.

```systemverilog
package desc_pkg;
  import accel_pkg::*;

  typedef struct packed {
    logic [15:0] magic;          // 0x594F
    logic [7:0]  version;
    logic [7:0]  num_layers;
    logic [63:0] layer_table_base;
    logic [63:0] weight_arena_base;
    logic [63:0] act0_arena_base;  // ping
    logic [63:0] act1_arena_base;  // pong
    logic [63:0] aux_arena_base;   // skip buffer
  } net_desc_t;                    // 48 bytes used / 64 bytes padded

  typedef struct packed {
    logic [3:0]  template_id;      // pe_mode_e
    logic [4:0]  layer_id;
    logic [8:0]  cin_total;
    logic [8:0]  cout_total;
    logic [9:0]  hin, win;
    logic [9:0]  hout, wout;
    logic [3:0]  kh, kw;
    logic [2:0]  sh, sw;           // stride
    logic [3:0]  pad_top, pad_bot, pad_left, pad_right;
    logic [7:0]  tile_cin, tile_cout;
    logic [5:0]  tile_w_blks;
    logic [11:0] num_tile_hw;
    logic [3:0]  r_need, r_new, q_in, q_out;
    logic [3:0]  num_cin_pass, num_k_pass;
    logic [7:0]  router_profile_id;
    logic [7:0]  post_profile_id;
    logic [4:0]  src_in_tid, src_w_tid, src_skip_tid, dst_tid;
    logic [63:0] tile_table_offset;
    logic [15:0] layer_flags;       // [0]keep_on_chip [1]need_barrier [2]hold_skip_after
  } layer_desc_t;                   // 64 bytes

  typedef struct packed {
    logic [15:0] tile_id;
    logic [4:0]  layer_id;
    logic [3:0]  sc_mask;           // which SC processes this tile
    logic [9:0]  h_out0, wblk0;
    logic [8:0]  cin0, cout0;
    logic [5:0]  valid_h, valid_w;
    logic [3:0]  halo_top, halo_bot, halo_left, halo_right;
    logic [31:0] src_in_off;
    logic [31:0] src_w_off;
    logic [31:0] src_skip_off;
    logic [31:0] dst_off;
    logic [9:0]  in_base_h, in_base_c;
    logic [9:0]  out_base_h, out_base_c;
    logic [3:0]  first_cin_pass, num_cin_pass;
    logic [3:0]  first_k_pass, num_k_pass;
    logic [15:0] tile_flags;
    // tile_flags bits:
    // [0] FIRST_TILE      → reset psum
    // [1] LAST_TILE       → signal layer done
    // [2] EDGE_TILE       → needs padding
    // [3] HAS_SKIP        → read skip tensor
    // [4] NEED_SWIZZLE    → output → swizzle → next bank_input
    // [5] NEED_SPILL      → output spill to DDR
    // [6] BARRIER_BEFORE  → wait barrier before run
    // [7] BARRIER_AFTER   → signal barrier when done
    // [10] HOLD_SKIP_ROLE → this sub holds skip data
  } tile_desc_t;

  typedef struct packed {
    logic        bias_en;
    quant_mode_e quant_mode;
    act_mode_e   act_mode;
    logic        ewise_en;
    logic [31:0] bias_scale_offset;    // offset in weight arena to bias/scale tables
    logic [7:0]  concat_ch_offset;
    logic [1:0]  upsample_factor;
  } post_profile_t;

  typedef struct packed {
    logic [2:0]  rin_src [3];          // source select per RIN channel
    logic [3:0]  rin_dst_mask [3];
    logic [2:0]  rwt_src [3];
    logic        rwt_h_multicast;
    logic [1:0]  rps_accum_mode;       // none/local/vertical/writeback
    logic        concat_offset_mode;
    logic        upsample_dup_mode;
  } router_profile_t;

endpackage
```

---

## M02. `csr_pkg.sv` — CSR Register Map

**Mục đích**: Định nghĩa địa chỉ và cấu trúc các Control/Status Registers.

```systemverilog
package csr_pkg;
  parameter int CSR_CTRL          = 12'h000;  // [0]start [1]soft_reset [2]irq_clr
  parameter int CSR_STATUS        = 12'h004;  // [0]busy [1]done [2]irq [3]error
  parameter int CSR_VERSION       = 12'h008;
  parameter int CSR_CAP0          = 12'h00C;  // num_sc, subcl/sc, pe_rows, pe_cols, lanes
  parameter int CSR_NET_DESC_LO   = 12'h010;
  parameter int CSR_NET_DESC_HI   = 12'h014;
  parameter int CSR_LAYER_START   = 12'h018;
  parameter int CSR_LAYER_END     = 12'h01C;
  parameter int CSR_IRQ_MASK      = 12'h020;
  parameter int CSR_PERF_CTRL     = 12'h030;
  parameter int CSR_PERF_CYCLE_LO = 12'h034;
  parameter int CSR_PERF_CYCLE_HI = 12'h038;
  parameter int CSR_PERF_TILE_DONE= 12'h03C;
  parameter int CSR_PERF_STALL    = 12'h040;
  parameter int CSR_BARRIER_STATUS= 12'h050;
  parameter int CSR_DBG_LAYER_ID  = 12'h060;
  parameter int CSR_DBG_TILE_ID   = 12'h064;
endpackage
```

---

# PHẦN C: LEVEL 1 – MEMORY LEAF MODULES

---

## M03. `glb_input_bank.sv` — Input Activation SRAM Bank

**Mục đích**: 1 trong 3 input banks. Lưu activation INT8. Banking rule: `bank_id = h mod 3`.
Mỗi bank chứa 32 subbanks (1 per lane), mỗi subbank là SRAM đơn cổng.

**Parameters**:
```
LANES           = 32
SUBBANK_DEPTH   = 2048    // max entries per subbank
DATA_WIDTH      = 8       // INT8
```

**Ports**:
```systemverilog
module glb_input_bank #(
  parameter int LANES         = 32,
  parameter int SUBBANK_DEPTH = 2048,
  parameter int ADDR_W        = $clog2(SUBBANK_DEPTH)
)(
  input  logic                  clk, rst_n,

  // Write port (from DMA / swizzle engine during FILLING)
  input  logic                  wr_en,
  input  logic [ADDR_W-1:0]    wr_addr,         // shared across all 32 subbanks
  input  logic [LANES*8-1:0]   wr_data,         // 32 × INT8 = 256 bits
  input  logic [LANES-1:0]     wr_lane_mask,    // per-lane write enable (for edge tiles)

  // Read port (to router → window_gen → PE during RUNNING)
  input  logic                  rd_en,
  input  logic [ADDR_W-1:0]    rd_addr,
  output logic [LANES*8-1:0]   rd_data          // 32 × INT8 = 256 bits
);
```

**Logic xử lý**:
1. **32 subbank instances**: Mỗi subbank là `SRAM[SUBBANK_DEPTH][8]`.
2. **Write**: Khi `wr_en=1`, ghi `wr_data[(lane+1)*8-1:lane*8]` vào `subbank[lane][wr_addr]` nếu `wr_lane_mask[lane]=1`.
3. **Read**: Khi `rd_en=1`, đọc `subbank[lane][rd_addr]` cho tất cả 32 lanes.
4. **Xilinx implementation**: Mỗi subbank map lên 1-2 BRAM36K (tuỳ SUBBANK_DEPTH).

---

## M04. `glb_weight_bank.sv` — Weight SRAM Bank

**Mục đích**: 1 trong 3 weight banks. Mỗi bank lưu weight cho 1 kernel row (reduction lane).
Có staging FIFO 8-entry để prefetch weight pass kế tiếp.

**Ports**:
```systemverilog
module glb_weight_bank #(
  parameter int LANES         = 32,
  parameter int BANK_DEPTH    = 1024,
  parameter int FIFO_DEPTH    = 8
)(
  input  logic                 clk, rst_n,

  // Write port (from DMA during FILLING)
  input  logic                 wr_en,
  input  logic [$clog2(BANK_DEPTH)-1:0] wr_addr,
  input  logic [LANES*8-1:0]  wr_data,          // 32 × INT8 weights

  // Read port (to PE via router)
  input  logic                 rd_en,
  input  logic [$clog2(BANK_DEPTH)-1:0] rd_addr,
  output logic [LANES*8-1:0]  rd_data,

  // Staging FIFO interface (prefetch for next pass)
  input  logic                 fifo_push,
  input  logic [LANES*8-1:0]  fifo_din,
  input  logic                 fifo_pop,
  output logic [LANES*8-1:0]  fifo_dout,
  output logic                 fifo_empty, fifo_full
);
```

**Logic**:
1. SRAM `[BANK_DEPTH][LANES*8]` dùng BRAM.
2. FIFO 8-entry dùng distributed RAM hoặc SRL16.
3. Trong mode RS_DENSE_3x3: bank[0] = kernel row 0, bank[1] = row 1, bank[2] = row 2.
4. Trong mode OS_1x1: bank[0..2] = 3 Cin slices parallel.
5. Trong mode DW_7x7 pass k: load 3 kernel rows per pass vào 3 banks.

---

## M05. `glb_output_bank.sv` — Output SRAM Bank (Dual-mode PSUM/ACT)

**Mục đích**: 1 trong 4 output banks. Dual-mode: lưu INT32 partial sums HOẶC INT8 activations.

**Ports**:
```systemverilog
module glb_output_bank #(
  parameter int LANES      = 32,
  parameter int BANK_DEPTH = 512
)(
  input  logic                  clk, rst_n,

  // Write port
  input  logic                  wr_en,
  input  logic [$clog2(BANK_DEPTH)-1:0] wr_addr,
  input  accel_pkg::namespace_e wr_ns,           // PSUM (32b) or ACT (8b)
  input  logic [LANES*32-1:0]  wr_data_psum,     // 32 × INT32 = 1024 bits (PSUM mode)
  input  logic [LANES*8-1:0]   wr_data_act,      // 32 × INT8  = 256 bits  (ACT mode)

  // Read port
  input  logic                  rd_en,
  input  logic [$clog2(BANK_DEPTH)-1:0] rd_addr,
  input  accel_pkg::namespace_e rd_ns,
  output logic [LANES*32-1:0]  rd_data_psum,
  output logic [LANES*8-1:0]   rd_data_act
);
```

**Logic**:
1. **PSUM mode**: SRAM `[BANK_DEPTH][LANES*32]` — 32 INT32 values = 128 bytes per entry. Sử dụng cho các pass trung gian (accumulate partial sums qua Cin pass / kernel pass).
2. **ACT mode**: SRAM `[BANK_DEPTH][LANES*8]` — 32 INT8 values = 32 bytes per entry. Sử dụng cho kết quả cuối cùng sau PPU.
3. Hai namespace dùng chung address space nhưng khác data width → implement bằng BRAM wide + byte-enable.

---

## M06. `metadata_ram.sv` — Slot Validity & Ring Pointers

**Mục đích**: Quản lý valid bits cho input/output slots, ring buffer pointers.

**Ports**:
```systemverilog
module metadata_ram #(
  parameter int NUM_SLOTS = 16,  // max slots in ring buffer
  parameter int META_BITS = 32   // per-slot metadata
)(
  input  logic                           clk, rst_n,
  input  logic                           clear_all,
  // Write
  input  logic                           set_valid,
  input  logic [$clog2(NUM_SLOTS)-1:0]  set_slot_id,
  input  logic [META_BITS-1:0]          set_meta,     // base_h, base_c
  // Read
  input  logic [$clog2(NUM_SLOTS)-1:0]  query_slot_id,
  output logic                           query_valid,
  output logic [META_BITS-1:0]          query_meta,
  // Ring management
  input  logic                           advance_ring,
  output logic [$clog2(NUM_SLOTS)-1:0]  ring_head,
  output logic [$clog2(NUM_SLOTS)-1:0]  ring_tail,
  output logic                           ring_full, ring_empty
);
```

**Logic**: Circular buffer management cho row-slot rotation. Khi `advance_ring`: `head = (head + 1) mod NUM_SLOTS`, invalidate oldest slot.

---

# PHẦN D: LEVEL 2 – ADDRESS GENERATORS

---

## M07. `addr_gen_input.sv` — Input Address Generator

**Mục đích**: Tính physical address từ logical (h, w, c) cho bank_input.

**Ports**:
```systemverilog
module addr_gen_input #(
  parameter int LANES     = 32,
  parameter int MAX_WIDTH = 640,
  parameter int MAX_CIN   = 256
)(
  input  logic                clk, rst_n,

  // Configuration (from shadow regs / tile descriptor)
  input  logic [9:0]         cfg_win,           // input width
  input  logic [8:0]         cfg_cin_tile,      // Cin tile size
  input  logic [3:0]         cfg_q_in,          // Q_in (circular slots)
  input  logic [3:0]         cfg_stride,
  input  logic [3:0]         cfg_pad_mode,
  input  logic signed [7:0]  cfg_zp_x,          // zero-point for padding

  // Request
  input  logic               req_valid,
  input  logic [9:0]         req_h,             // row in input
  input  logic [9:0]         req_w,             // column (start of lane block)
  input  logic [8:0]         req_c,             // channel

  // Output
  output logic               out_valid,
  output logic [1:0]         out_bank_id,        // h mod 3 → {0,1,2}
  output logic [15:0]        out_addr,           // physical address in subbank
  output logic               out_is_padding,     // true if (h,w) is in padding region
  output logic signed [7:0]  out_pad_value       // zp_x for padding positions
);
```

**Logic**:
```
bank_id    = h mod 3
row_slot   = (h / 3) mod q_in
wblk       = w / LANES
addr       = (row_slot × cfg_cin_tile + c) × wblk_total + wblk

is_padding = (h < pad_top) OR (h >= hin - pad_bot) OR (w < pad_left) OR (w >= win - pad_right)
pad_value  = cfg_zp_x   // CRITICAL: phải pad bằng zp_x, KHÔNG PHẢI 0
```

---

## M08. `addr_gen_weight.sv` — Weight Address Generator

**Mục đích**: Tính address cho weight data trong bank_weight.

**Ports**:
```systemverilog
module addr_gen_weight #(
  parameter int LANES = 32
)(
  input  logic              clk, rst_n,
  input  accel_pkg::pe_mode_e cfg_mode,
  input  logic [8:0]        cfg_cin_tile, cfg_cout_tile,

  input  logic              req_valid,
  input  logic [1:0]        req_kr,           // kernel row (0-2 for RS3/DW3, 0-6 for DW7)
  input  logic [8:0]        req_cin,
  input  logic [8:0]        req_cout,

  output logic              out_valid,
  output logic [1:0]        out_bank_id,      // kr mod 3 for reduction lane
  output logic [15:0]       out_addr
);
```

**Logic per mode**:
```
RS_DENSE_3x3:
  bank_id = kr (0,1,2 for 3 kernel rows)
  addr    = cout × cin_tile × kw_total + cin × kw_total + kw
  
OS_1x1:
  bank_id = cin_slice (0,1,2 for 3 Cin chunks)
  addr    = cout × cin_per_slice + cin_within_slice

DW_3x3:
  bank_id = kr (0,1,2)
  addr    = channel × kw + kw_idx     (groups=Cin, no cross-channel)

DW_7x7_MULTIPASS:
  pass 0: bank[0..2] = kernel rows 0-2
  pass 1: bank[0..2] = kernel rows 3-5
  pass 2: bank[0]    = kernel row 6, bank[1..2] unused
```

---

## M09. `addr_gen_output.sv` — Output Address Generator

**Mục đích**: Tính address cho write results vào bank_output.

**Ports**:
```systemverilog
module addr_gen_output #(
  parameter int LANES = 32
)(
  input  logic              clk, rst_n,
  input  logic [3:0]        cfg_stride_h,
  input  logic [3:0]        cfg_q_out,
  input  logic [8:0]        cfg_cout_tile,

  input  logic              req_valid,
  input  logic [9:0]        req_h_out,
  input  logic [9:0]        req_w_out,
  input  logic [8:0]        req_cout,

  output logic              out_valid,
  output logic [1:0]        out_bank_id,       // pe_col index (0-3)
  output logic [15:0]       out_addr
);
```

**Logic**:
```
obank  = pe_col                                    // output bank = PE column index
oslot  = (h_out / (PE_COLS × stride_h)) mod q_out
addr   = (oslot × cout_tile + cout) × wblk_out + wblk
```

---

# PHẦN E: LEVEL 3 – COMPUTE PRIMITIVES

---

## M10. `dsp_pair_int8.sv` — Dual INT8 MAC in 1 DSP48E1

**Mục đích**: Module cơ bản nhất — 2 phép INT8×INT8→INT32 MAC dùng 1 DSP48E1.

**Ports**:
```systemverilog
module dsp_pair_int8 (
  input  logic              clk, rst_n,
  input  logic              en,            // compute enable
  input  logic              clear,         // reset accumulator (FIRST_TILE)
  input  logic signed [7:0] x_a,           // activation lane 2i
  input  logic signed [7:0] x_b,           // activation lane 2i+1
  input  logic signed [7:0] w,             // shared weight
  output logic signed [31:0] psum_a,       // accumulated result lane 2i
  output logic signed [31:0] psum_b        // accumulated result lane 2i+1
);
```

**Logic chi tiết**:
```
Pipeline stage 1 (unsigned conversion):
  x_a_u = x_a + 128    // signed [-128,127] → unsigned [0,255]
  x_b_u = x_b + 128
  w_u   = w   + 128

Pipeline stage 2 (DSP48E1 multiply):
  dsp_A[24:0] = {x_b_u[7:0], 9'b0, x_a_u[7:0]}    // pack 2 activations
  dsp_B[17:0] = {10'b0, w_u[7:0]}
  dsp_P[42:0] = dsp_A × dsp_B                        // single DSP48E1 multiply

Pipeline stage 3 (extract & correct):
  prod_a_u = dsp_P[15:0]                              // a_u × w_u
  prod_b_u = dsp_P[32:17]                             // b_u × w_u
  
  // Reverse unsigned offset: (a+128)(w+128) = a×w + 128a + 128w + 16384
  signed_prod_a = prod_a_u - 128×(x_a_u + w_u) + 16384   // = x_a × w
  signed_prod_b = prod_b_u - 128×(x_b_u + w_u) + 16384   // = x_b × w

Pipeline stage 4 (accumulate):
  if (clear) psum_a <= signed_prod_a;  else psum_a <= psum_a + signed_prod_a;
  if (clear) psum_b <= signed_prod_b;  else psum_b <= psum_b + signed_prod_b;
```

**Overflow safety**: Max accumulation: `9 × 256 × 127 × 127 = 37,064,529 < 2^31` ✓

---

## M11. `pe_unit.sv` — Single Processing Element (32 lanes)

**Mục đích**: 1 PE = 16 DSP pairs = 32 lanes. Thực hiện 32 MAC operations per cycle.

**Ports**:
```systemverilog
module pe_unit #(
  parameter int LANES = 32
)(
  input  logic              clk, rst_n,
  input  logic              en,
  input  logic              clear_psum,       // from FIRST_TILE flag
  input  accel_pkg::pe_mode_e mode,

  // Activation input (32 INT8 values per cycle)
  input  logic signed [7:0] x_in [LANES],

  // Weight input (shared per DSP pair for RS3/DW/GEMM; broadcast in OS1)
  input  logic signed [7:0] w_in [LANES],

  // Partial sum output (32 INT32 values)
  output logic signed [31:0] psum_out [LANES],
  output logic               psum_valid
);
```

**Logic**:
```
Instantiate 16 dsp_pair_int8:
  for i in 0..15:
    dsp_pair[i].x_a = x_in[2*i]
    dsp_pair[i].x_b = x_in[2*i + 1]
    dsp_pair[i].w   = w_in[2*i]      // RS3/DW/GEMM: shared by lanes [2*i] and [2*i+1]
                                       // OS1: w broadcast to all lanes
    psum_out[2*i]   = dsp_pair[i].psum_a
    psum_out[2*i+1] = dsp_pair[i].psum_b

Mode-specific behavior:
  RS_DENSE_3x3: x_in = spatial window tap, w_in[2*i] shared across lanes [2*i] and [2*i+1]
  OS_1x1:       x_in = activation, w_in = same weight broadcast to all lanes
  DW_3x3:       x_in = spatial tap, w_in = per-channel weight (groups=C)
  MAXPOOL:      MAC disabled, use comparator_tree instead
  GEMM:         x_in = matrix row tile, w_in = matrix col tile (transposed)
```

---

## M12. `window_gen.sv` — Spatial Window Tap Generator

**Mục đích**: Tạo sliding window K=1/3/5/7 từ input vector stream. Mỗi cycle nhận 1 vector 32-wide, output K vectors (taps).

**Ports**:
```systemverilog
module window_gen #(
  parameter int LANES = 32,
  parameter int K_MAX = 7      // max kernel width
)(
  input  logic                clk, rst_n,
  input  logic [2:0]          cfg_kw,          // kernel width: 1,3,5,7
  input  logic                shift_in_valid,
  input  logic signed [7:0]  shift_in [LANES], // new input vector (1 row, 32 cols)

  output logic                taps_valid,
  output logic signed [7:0]  taps [K_MAX][LANES]  // K tap vectors, each 32-wide
);
```

**Logic**:
```
Internal: shift_reg[K_MAX][LANES] — shift register chain

Every cycle when shift_in_valid:
  shift_reg[K_MAX-1] = shift_reg[K_MAX-2]
  ...
  shift_reg[1] = shift_reg[0]
  shift_reg[0] = shift_in

Output taps selection based on cfg_kw:
  K1: taps[0] = shift_reg[0]
  K3: taps[0..2] = shift_reg[0..2]
  K5: taps[0..4] = shift_reg[0..4]
  K7: taps[0..6] = shift_reg[0..6]

taps_valid asserted when pipeline has accumulated enough rows (≥ cfg_kw entries).
```

---

## M13. `column_reduce.sv` — Cross-Row Partial Sum Reduction

**Mục đích**: Cộng kết quả từ 3 PE rows → 1 psum vector per PE column.

**Ports**:
```systemverilog
module column_reduce #(
  parameter int LANES   = 32,
  parameter int PE_ROWS = 3,
  parameter int PE_COLS = 4
)(
  input  logic               clk, rst_n,
  input  logic               en,
  input  accel_pkg::pe_mode_e mode,

  // From PE array: [PE_ROWS][PE_COLS] × LANES INT32 values
  input  logic signed [31:0] pe_psum [PE_ROWS][PE_COLS][LANES],

  // Reduced output: [PE_COLS] × LANES INT32 values
  output logic signed [31:0] col_psum [PE_COLS][LANES],
  output logic               col_valid
);
```

**Logic**:
```
For RS_DENSE_3x3, OS_1x1, GEMM:
  col_psum[col][lane] = Σ_{row=0..2} pe_psum[row][col][lane]
  // Sum 3 kernel rows → 1 psum per output position

For DW_3x3, DW_7x7:
  col_psum[col][lane] = Σ_{row=0..2} pe_psum[row][col][lane]
  // Same sum but per-CHANNEL (groups mode, no cross-channel)

For MAXPOOL_5x5:
  col_psum bypassed → use comparator_tree output instead
```

---

## M14. `comparator_tree.sv` — Max Comparator for MAXPOOL

**Mục đích**: Tìm max trong 25 input (5×5 window) per lane. Dùng cho MAXPOOL_5x5.

**Ports**:
```systemverilog
module comparator_tree #(
  parameter int LANES      = 32,
  parameter int NUM_INPUTS = 25    // 5×5 window
)(
  input  logic               clk, rst_n,
  input  logic               en,
  input  logic signed [7:0] data_in [NUM_INPUTS][LANES],  // 25 taps × 32 lanes
  output logic signed [7:0] max_out [LANES],               // 32 max values
  output logic               max_valid
);
```

**Logic**:
```
Tree reduction (pipelined 3 stages for 25→1):
  Stage 1: 25 inputs → 13 (compare pairs, 1 odd passes through)
  Stage 2: 13 → 7
  Stage 3: 7 → 4
  Stage 4: 4 → 2
  Stage 5: 2 → 1

Per lane: independent max-tree (no cross-lane interaction).
Latency: 5 cycles (pipelined).
Scale/zp: pass-through (maxpool preserves quantization domain).
```

---

## M15. `pe_cluster.sv` — Full PE Array (3×4×32)

**Mục đích**: Wrapper quanh 12 PE units + column_reduce + comparator_tree. Module tính toán chính của mỗi subcluster.

**Ports**:
```systemverilog
module pe_cluster #(
  parameter int LANES   = 32,
  parameter int PE_ROWS = 3,
  parameter int PE_COLS = 4
)(
  input  logic                clk, rst_n,
  input  logic                en,
  input  logic                clear_psum,
  input  accel_pkg::pe_mode_e mode,

  // Activation taps from window_gen (per PE row)
  input  logic signed [7:0]  act_taps [PE_ROWS][LANES],

  // Weight data from router (per PE row, broadcast to PE cols)
  input  logic signed [7:0]  wgt_data [PE_ROWS][LANES],

  // Psum input (for multi-pass accumulation from bank_output)
  input  logic signed [31:0] psum_in [PE_COLS][LANES],
  input  logic               psum_in_valid,

  // Output
  output logic signed [31:0] psum_out [PE_COLS][LANES],
  output logic               psum_out_valid,
  output logic               last_pass,         // from tile_fsm

  // MAXPOOL output (bypass psum path)
  output logic signed [7:0]  pool_out [LANES],
  output logic               pool_out_valid
);
```

**Logic interne**:
```
Instantiate PE_ROWS × PE_COLS = 12 pe_unit instances.

Weight routing per mode:
  RS3:  wgt_data[row] broadcast to all PE_COLS in that row
  OS1:  wgt_data[row] = Cin_slice weights, broadcast to PE_COLS
  DW3:  wgt_data[row] = per-channel weight, each col = different channel group
  GEMM: similar to OS1 but matrix tiles

Psum accumulation:
  if (psum_in_valid): load psum_in to PE accumulators (multi-pass)
  Column reduce: sum 3 rows → col_psum[PE_COLS][LANES]

  if (mode == PE_MP5):
    Gather 25 taps from window_gen → comparator_tree → pool_out
    psum path inactive
  else:
    psum_out = col_psum (after cross-row reduction)
```

---

# PHẦN F: LEVEL 3 – POST-PROCESSING UNIT

---

## M16. `ppu.sv` — Post-Processing Unit (Top Wrapper)

**Mục đích**: Xử lý hậu kỳ: bias + requant + activation + clamp + ewise_add. Pipeline 4-stage.

**Ports**:
```systemverilog
module ppu #(
  parameter int LANES = 32
)(
  input  logic                  clk, rst_n,
  input  logic                  en,

  // Configuration
  input  desc_pkg::post_profile_t cfg_post,
  input  accel_pkg::pe_mode_e     cfg_mode,

  // PSUM input (INT32 from PE cluster, per PE column, process 1 column at a time)
  input  logic signed [31:0]   psum_in [LANES],
  input  logic                  psum_valid,

  // Per-channel parameters (loaded from weight arena via DMA)
  input  logic signed [31:0]   bias_val [LANES],     // INT32 bias per output channel
  input  logic signed [31:0]   m_int [LANES],        // fixed-point multiplier M_int
  input  logic [5:0]           shift [LANES],         // right shift amount
  input  logic signed [7:0]    zp_out,               // output zero-point

  // SiLU LUT (preloaded)
  input  logic signed [7:0]    silu_lut [256],

  // Element-wise add input (for skip connection / residual)
  input  logic signed [7:0]    ewise_in [LANES],
  input  logic                  ewise_valid,

  // Output (INT8)
  output logic signed [7:0]    act_out [LANES],
  output logic                  act_valid
);
```

**Pipeline stages**:
```
Stage 1 — Bias Add:
  biased[lane] = psum_in[lane] + bias_val[lane]      // INT32 + INT32 → INT32

Stage 2 — Fixed-Point Requant:
  y_raw[lane] = (biased[lane] × m_int[lane]) >>> shift[lane]   // arithmetic right shift
  // Rounding: half_up → add (1 << (shift-1)) before shift

Stage 3 — Activation:
  switch (cfg_post.act_mode):
    ACT_SILU: idx = clamp(y_raw + 128, 0, 255); y_act = silu_lut[idx]
    ACT_RELU: y_act = (y_raw > 0) ? y_raw : 0
    ACT_NONE: y_act = y_raw
    ACT_CLAMP: y_act = y_raw (clamped in stage 4)

Stage 4 — Clamp + Ewise Add:
  if (cfg_post.ewise_en):
    y_add = y_act + ewise_in[lane]     // element-wise add (skip connection)
  else:
    y_add = y_act
  act_out[lane] = clamp(y_add + zp_out, -128, 127)
```

---

## M17. `silu_lut.sv` — SiLU Lookup Table ROM

**Mục đích**: 256-entry INT8 ROM preloaded với SiLU(x) values. Hỗ trợ 32 concurrent reads.

**Ports**:
```systemverilog
module silu_lut #(
  parameter int LANES = 32
)(
  input  logic                clk,
  // Load interface (preload from descriptor/DMA)
  input  logic                load_en,
  input  logic [7:0]          load_addr,
  input  logic signed [7:0]  load_data,
  // Lookup interface (32 parallel reads)
  input  logic [7:0]          idx [LANES],
  output logic signed [7:0]  out [LANES]
);
```

**Logic**:
```
ROM[256] of INT8, precomputed:
  For each q ∈ [-128..127]:
    x_float = (q - zp_out) × scale_out
    silu_float = x_float × sigmoid(x_float)
    ROM[q + 128] = clamp(round(silu_float / scale_out) + zp_out, -128, 127)

32 parallel reads: dùng 32-port distributed RAM
  hoặc: 2 cycles × 16-port BRAM (latency tradeoff)
  hoặc: replicate ROM 32 lần (LUT-based, ~8K LUTs)
```

---

# PHẦN G: LEVEL 5 – DATA MOVEMENT

---

## M18. `router_cluster.sv` — Data Routing Hub

**Mục đích**: Điều hướng dữ liệu giữa GLB banks ↔ PE cluster ↔ PPU ↔ swizzle.
3 sub-routers: RIN (activation), RWT (weight), RPS (psum/output).

**Ports**:
```systemverilog
module router_cluster #(
  parameter int LANES = 32
)(
  input  logic                     clk, rst_n,
  input  desc_pkg::router_profile_t cfg_profile,
  input  accel_pkg::pe_mode_e       cfg_mode,

  // ═══ RIN: Activation Router (3 channels → 3 PE rows) ═══
  input  logic [LANES*8-1:0]      bank_input_rd [3],     // from 3 input banks
  input  logic [LANES*8-1:0]      neighbor_in [4],        // N/S/E/W neighbors
  input  logic [LANES*8-1:0]      swizzle_in,             // from swizzle engine
  output logic signed [7:0]       pe_act [3][LANES],      // to 3 PE rows

  // ═══ RWT: Weight Router (3 channels → 3 PE rows) ═══
  input  logic [LANES*8-1:0]      bank_weight_rd [3],
  output logic signed [7:0]       pe_wgt [3][LANES],      // to 3 PE rows

  // ═══ RPS: Psum/Output Router (4 channels from PE cols) ═══
  input  logic signed [31:0]      pe_psum [4][LANES],     // from PE cluster
  output logic [LANES*32-1:0]     bank_output_wr [4],     // to 4 output banks
  output logic [LANES*8-1:0]      ppu_in,                  // to PPU
  output logic [LANES*8-1:0]      neighbor_out [4],        // to N/S/E/W neighbors

  // ═══ Bypass paths (MOVE, CONCAT, UPSAMPLE) ═══
  input  logic                     bypass_en,
  input  logic [LANES*8-1:0]      bypass_data,
  output logic [LANES*8-1:0]      bypass_out
);
```

**Logic per mode**:
```
RS_DENSE_3x3 / OS_1x1:
  RIN: bank_input[0..2] → pe_act[0..2] (1-to-1 per row)
  RWT: bank_weight[0..2] → pe_wgt[0..2] → broadcast to 4 PE cols
  RPS: pe_psum[0..3] → bank_output[0..3]

DW_3x3 / DW_7x7:
  RIN: same as RS3
  RWT: same but per-channel (no cross-channel broadcast)
  RPS: same as RS3

CONCAT:
  bypass_en = 1
  Route: bank_input (tensor A channels) → bypass → bank_output
         then bank_input (tensor B channels, offset) → bypass → bank_output
  No PE/PPU involvement

UPSAMPLE:
  bypass_en = 1
  Route: bank_input → swizzle_engine (address duplication) → bank_output

MOVE:
  bypass_en = 1
  Direct: bank_input → bypass → bank_output (or → external DMA)
```

---

## M19. `swizzle_engine.sv` — Tensor Layout Transform

**Mục đích**: Biến đổi layout giữa bank_output → bank_input cho layer kế tiếp.
Xử lý UPSAMPLE_NEAREST, CONCAT channel offset, re-layout.

**Ports**:
```systemverilog
module swizzle_engine #(
  parameter int LANES = 32
)(
  input  logic                clk, rst_n,
  input  logic                en,
  input  accel_pkg::pe_mode_e mode,

  // Configuration
  input  logic [1:0]          cfg_upsample_factor,  // 0=none, 1=2×, 2=4× (unused)
  input  logic [8:0]          cfg_concat_ch_offset,  // channel offset for B in concat
  input  logic [9:0]          cfg_src_h, cfg_src_w, cfg_src_c,
  input  logic [9:0]          cfg_dst_h, cfg_dst_w, cfg_dst_c,
  input  logic [3:0]          cfg_dst_q_in,         // Q_in for next layer's bank_input

  // Source: read from bank_output
  output logic                src_rd_en,
  output logic [15:0]         src_rd_addr,
  output logic [1:0]          src_rd_bank,
  input  logic [LANES*8-1:0]  src_rd_data,

  // Destination: write to bank_input (for next layer)
  output logic                dst_wr_en,
  output logic [15:0]         dst_wr_addr,
  output logic [1:0]          dst_wr_bank,    // h_dst mod 3
  output logic [LANES*8-1:0]  dst_wr_data,
  output logic [LANES-1:0]    dst_wr_mask,

  output logic                done
);
```

**Logic per mode**:
```
NORMAL (layer chaining):
  for each output element (h_out, w, c_out):
    ib_next   = h_out mod 3
    slot_next = (h_out / 3) mod cfg_dst_q_in
    wblk      = w / LANES
    addr      = (slot_next × c_tile_next + c_out) × wblk_total_next + wblk
    bank_input_next[ib_next][addr] = bank_output_current[element]

UPSAMPLE_NEAREST (scale=2):
  for each src element (h, w, c):
    write to 4 dst positions:
      (2h,   2w)   → bank_input[(2h) mod 3][addr(2h, 2w, c)]
      (2h,   2w+1) → bank_input[(2h) mod 3][addr(2h, 2w+1, c)]
      (2h+1, 2w)   → bank_input[(2h+1) mod 3][addr(2h+1, 2w, c)]
      (2h+1, 2w+1) → bank_input[(2h+1) mod 3][addr(2h+1, 2w+1, c)]
  → 4 writes per source read → throughput = 4 cycles/element

CONCAT:
  Tensor A: write channels [0..C_A-1] normally
  Tensor B: write channels [C_A..C_A+C_B-1] with cfg_concat_ch_offset = C_A
  Address: same (h,w) but channel offset applied
```

---

# PHẦN H: LEVEL 6 – CONTROL MODULES

---

## M20. `shadow_reg_file.sv` — Tile Descriptor Shadow Registers

**Mục đích**: Capture tile descriptor fields vào pipeline registers. Cung cấp cấu hình ổn định cho PE cluster suốt thời gian compute 1 tile.

**Ports**:
```systemverilog
module shadow_reg_file (
  input  logic              clk, rst_n,
  input  logic              load,           // pulse: capture from tile_desc
  input  desc_pkg::tile_desc_t  tile_desc,
  input  desc_pkg::layer_desc_t layer_desc,
  input  desc_pkg::post_profile_t post_profile,
  input  desc_pkg::router_profile_t router_profile,

  // Output: stable configuration signals
  output accel_pkg::pe_mode_e   o_mode,
  output logic [8:0]            o_cin_tile, o_cout_tile,
  output logic [9:0]            o_hin, o_win, o_hout, o_wout,
  output logic [3:0]            o_kh, o_kw, o_sh, o_sw,
  output logic [3:0]            o_pad_top, o_pad_bot, o_pad_left, o_pad_right,
  output logic [3:0]            o_r_need, o_q_in, o_q_out,
  output logic [3:0]            o_num_cin_pass, o_num_k_pass,
  output logic [15:0]           o_tile_flags,
  output logic signed [7:0]     o_zp_x,
  output desc_pkg::post_profile_t   o_post,
  output desc_pkg::router_profile_t o_router
);
```

---

## M21. `tile_fsm.sv` — Tile Execution FSM

**Mục đích**: Điều khiển toàn bộ quá trình thực thi 1 tile: load config → fill data → compute passes → post-process → store output.

**Ports**:
```systemverilog
module tile_fsm (
  input  logic              clk, rst_n,

  // Tile descriptor input (from global_scheduler)
  input  logic              tile_valid,
  input  desc_pkg::tile_desc_t  tile_desc,
  input  desc_pkg::layer_desc_t layer_desc,
  output logic              tile_accept,

  // GLB control
  output logic              glb_wr_en,        // enable writes to GLB during PREFILL
  output logic              glb_rd_en,        // enable reads from GLB during COMPUTE
  
  // PE cluster control
  output logic              pe_en,
  output logic              pe_clear_psum,
  output accel_pkg::pe_mode_e pe_mode,

  // PPU control
  output logic              ppu_en,
  output logic              ppu_last_pass,     // trigger PPU after final accumulation

  // Swizzle control
  output logic              swizzle_start,
  input  logic              swizzle_done,

  // DMA requests (for external data movement)
  output logic              dma_rd_req,        // request DMA read (fill)
  output logic [39:0]       dma_rd_addr,
  output logic [15:0]       dma_rd_len,
  input  logic              dma_rd_done,

  output logic              dma_wr_req,        // request DMA write (drain)
  output logic [39:0]       dma_wr_addr,
  output logic [15:0]       dma_wr_len,
  input  logic              dma_wr_done,

  // Barrier interface
  output logic              barrier_wait_req,  // request barrier check
  input  logic              barrier_grant,
  output logic              barrier_signal,    // signal barrier done

  // Status
  output accel_pkg::tile_state_e state,
  output logic              tile_done,
  output logic              layer_done         // when LAST_TILE flag
);
```

**FSM Logic chi tiết**:
```
TILE_IDLE:
  Wait for tile_valid → accept → goto TILE_LOAD_CFG

TILE_LOAD_CFG:
  shadow_reg_file.load = 1
  Configure pe_mode, addr_gen, router_profile, post_profile
  goto TILE_PREFILL_WT

TILE_PREFILL_WT:
  if (mode != PE_PASS):
    Issue dma_rd_req for weight data (src_w_off in weight_arena)
    Wait for weight data → write to glb_weight_bank[0..2]
  goto TILE_PREFILL_IN

TILE_PREFILL_IN:
  Issue dma_rd_req for input activation (src_in_off in act_arena)
  Write R_need resident rows to glb_input_bank[0..2]
  if (tile_flags.HAS_SKIP): also load skip tensor (src_skip_off)
  goto TILE_WAIT_READY

TILE_WAIT_READY:
  if (tile_flags.BARRIER_BEFORE): issue barrier_wait_req; wait barrier_grant
  Check: weight_loaded AND input_loaded AND router_configured
  goto TILE_RUN_COMPUTE

TILE_RUN_COMPUTE:
  pe_en = 1
  pe_clear_psum = tile_flags.FIRST_TILE
  
  Run compute loop:
    for each output row group (PE_COLS rows at a time):
      for each Wblk (0 to wblk_total-1):
        for each Cin pass (0 to num_cin_pass-1):
          for each K pass (0 to num_k_pass-1):
            addr_gen generates addresses
            window_gen produces taps
            PE computes MACs
            column_reduce produces psum
  goto TILE_ACCUMULATE

TILE_ACCUMULATE:
  if NOT (last_cin AND last_kernel):
    Write psum to bank_output (PSUM namespace)
    goto TILE_RUN_COMPUTE (next pass)
  else:
    ppu_last_pass = 1
    goto TILE_POST_PROCESS

TILE_POST_PROCESS:
  PPU processes: bias + requant + SiLU + clamp
  Write INT8 result to bank_output (ACT namespace)
  if (tile_flags.BARRIER_AFTER): barrier_signal = 1
  goto TILE_SWIZZLE_STORE

TILE_SWIZZLE_STORE:
  if (tile_flags.NEED_SWIZZLE):
    swizzle_start = 1; wait swizzle_done
  if (tile_flags.NEED_SPILL):
    dma_wr_req for output data → DDR
    wait dma_wr_done
  goto TILE_DONE

TILE_DONE:
  tile_done = 1
  if (tile_flags.LAST_TILE): layer_done = 1
  goto TILE_IDLE
```

---

## M22. `barrier_manager.sv` — Skip Dependency Barrier

**Mục đích**: Quản lý 4 skip dependencies cho YOLOv10n (L4→L15, L6→L12, L8→L21, L13→L18).

**Ports**:
```systemverilog
module barrier_manager #(
  parameter int NUM_BARRIERS = 32
)(
  input  logic              clk, rst_n,
  input  logic              clear_all,

  // Signal interface (producer signals completion)
  input  logic              signal_valid,
  input  logic [4:0]        signal_barrier_id,

  // Wait interface (consumer checks readiness)
  input  logic              wait_valid,
  input  logic [4:0]        wait_barrier_id,
  output logic              wait_grant,

  // Debug readback
  output logic [NUM_BARRIERS-1:0] scoreboard
);
```

**Logic**:
```
scoreboard[32]: 1 bit per barrier point

signal: scoreboard[signal_barrier_id] = 1
wait:   wait_grant = scoreboard[wait_barrier_id]
clear:  scoreboard = 0 (at start of new inference)

YOLOv10n barrier mapping:
  barrier[0]: L6_done  → enables L12
  barrier[1]: L4_done  → enables L15
  barrier[2]: L13_done → enables L18
  barrier[3]: L8_done  → enables L21
```

---

## M23. `local_arbiter.sv` — Dual-RUNNING 4-Phase Scheduler

**Mục đích**: Quản lý 4 subclusters per SuperCluster. Gán roles (2×RUNNING + 1×FILLING + 1×DRAINING/HOLD). Arbitrate external port access.

**Ports**:
```systemverilog
module local_arbiter #(
  parameter int NUM_SUBS = 4
)(
  input  logic              clk, rst_n,
  
  // Tile queue (from global_scheduler)
  input  logic              tile_available,
  input  desc_pkg::tile_desc_t next_tile,
  output logic              tile_consumed,

  // Per-subcluster status
  input  accel_pkg::tile_state_e sub_state [NUM_SUBS],
  input  logic              sub_tile_done [NUM_SUBS],

  // Role assignment output
  output accel_pkg::sc_role_e sub_role [NUM_SUBS],

  // External port arbitration
  input  logic              ext_port_ready,
  output logic [1:0]        ext_port_grant_sub,  // which sub gets ext port
  output logic              ext_port_is_read,     // read (fill) vs write (drain)

  // Tile dispatch to subclusters
  output logic              sub_tile_valid [NUM_SUBS],
  output desc_pkg::tile_desc_t sub_tile [NUM_SUBS]
);
```

**Logic**:
```
Role rotation (dual-RUNNING):
  Maintain role_reg[4] — each sub's current role

  When sub[i] finishes tile (sub_tile_done[i] = 1):
    sub[i].role → DRAINING (needs to output its result)
    Find sub in FILLING state with data ready → promote to RUNNING
    Find IDLE/DRAINED sub → assign FILLING (load next tile)

  Priority: RUNNING > RUNNING > FILLING > DRAINING/HOLD

External port arbitration:
  if (any sub in FILLING state AND ext_port_ready):
    Grant to FILLING sub (read operations)
  elif (any sub in DRAINING state AND ext_port_ready):
    Grant to DRAINING sub (write operations)
  
  Time-multiplex: alternate FILL/DRAIN bursts when both pending
```

---

## M24. `desc_fetch_engine.sv` — Descriptor Fetch from DDR

**Mục đích**: Đọc NET_DESC → LAYER_DESC → TILE_DESC từ DDR qua DMA. Parse và dispatch.

**Ports**:
```systemverilog
module desc_fetch_engine (
  input  logic               clk, rst_n,
  input  logic               start,           // from CSR.CTRL.start

  // AXI read master interface
  output logic [39:0]        axi_araddr,
  output logic               axi_arvalid,
  input  logic               axi_arready,
  input  logic [255:0]       axi_rdata,
  input  logic               axi_rvalid,
  output logic               axi_rready,

  // Configuration from CSR
  input  logic [63:0]        net_desc_base,
  input  logic [4:0]         layer_start, layer_end,

  // Output: parsed descriptors
  output desc_pkg::net_desc_t   net_desc,
  output logic                   net_desc_valid,
  output desc_pkg::layer_desc_t layer_desc,
  output logic                   layer_desc_valid,
  output desc_pkg::tile_desc_t  tile_desc,
  output logic                   tile_desc_valid,

  // Status
  output logic [4:0]         current_layer,
  output logic               all_layers_done
);
```

**FSM**:
```
IDLE → FETCH_NET → PARSE_NET → FETCH_LAYER[layer_id] → PARSE_LAYER →
FETCH_TILES → DISPATCH_TILES → NEXT_LAYER → DONE

FETCH_NET: DMA read 64B from net_desc_base → parse into net_desc_t
FETCH_LAYER: DMA read 64B from layer_table_base + layer_id × 64
FETCH_TILES: DMA read N × 64B tile descriptors from tile_table_offset
DISPATCH_TILES: push tile_desc_t to global_scheduler → local_arbiter → subcluster
NEXT_LAYER: increment layer_id; if layer_id > layer_end → DONE
```

---

## M25. `global_scheduler.sv` — Layer/Tile Dispatcher

**Mục đích**: Nhận layer_desc và tile_desc từ desc_fetch_engine, phân phối tới 4 SuperClusters theo sc_mask.

**Ports**:
```systemverilog
module global_scheduler (
  input  logic               clk, rst_n,

  // From desc_fetch_engine
  input  desc_pkg::layer_desc_t layer_desc,
  input  logic                   layer_valid,
  input  desc_pkg::tile_desc_t  tile_desc,
  input  logic                   tile_valid,

  // To 4 SuperClusters
  output desc_pkg::tile_desc_t  sc_tile [4],
  output logic                   sc_tile_valid [4],
  input  logic                   sc_tile_accept [4],

  // Barrier interface
  output logic                   barrier_signal,
  output logic [4:0]            barrier_id,
  input  logic                   barrier_grant,

  // Status
  output logic                   layer_complete,
  output logic                   inference_complete
);
```

**Logic**:
```
For each tile_desc received:
  sc_mask = tile_desc.sc_mask     // bits [3:0] indicate which SC processes this tile
  for sc in 0..3:
    if (sc_mask[sc]):
      push tile to sc_tile[sc] queue
      wait sc_tile_accept[sc]

Track: tiles_dispatched, tiles_completed per layer
When all tiles of current layer done: signal layer_complete
When all layers done: signal inference_complete → assert IRQ
```

---

## M26. `controller_system.sv` — Top Control Wrapper

**Mục đích**: Glue module: CSR + desc_fetch + barrier_manager + global_scheduler.

**Ports**:
```systemverilog
module controller_system (
  input  logic               clk, rst_n,

  // AXI-Lite MMIO (CPU control)
  input  logic [11:0]        mmio_addr,
  input  logic [31:0]        mmio_wdata,
  input  logic               mmio_we, mmio_re,
  output logic [31:0]        mmio_rdata,
  output logic               irq,

  // AXI4 DMA read (for descriptor fetch)
  output logic [39:0]        axi_araddr,
  output logic               axi_arvalid,
  input  logic               axi_arready,
  input  logic [255:0]       axi_rdata,
  input  logic               axi_rvalid,
  output logic               axi_rready,

  // To 4 SuperClusters
  output desc_pkg::tile_desc_t  sc_tile [4],
  output desc_pkg::layer_desc_t sc_layer_desc,
  output logic                   sc_tile_valid [4],
  input  logic                   sc_tile_accept [4],
  input  logic                   sc_tile_done [4],
  input  logic                   sc_layer_done [4],

  // Barrier net
  input  logic                   barrier_signal [4],
  input  logic [4:0]            barrier_signal_id [4],
  output logic [31:0]           barrier_scoreboard
);
```

---

# PHẦN I: LEVEL 7 – TOP WRAPPERS

---

## M27. `subcluster_wrapper.sv` — Full Compute Unit

**Mục đích**: Đơn vị tính toán hoàn chỉnh. Gồm: tile_fsm + shadow_regs + GLB + router + window_gen + PE_cluster + PPU + swizzle.

**Ports**:
```systemverilog
module subcluster_wrapper #(
  parameter int LANES = 32
)(
  input  logic               clk, rst_n,

  // Tile input (from local_arbiter)
  input  logic               tile_valid,
  input  desc_pkg::tile_desc_t  tile_desc,
  input  desc_pkg::layer_desc_t layer_desc,
  output logic               tile_accept,

  // External memory port (shared via arbiter)
  output logic               ext_rd_req,
  output logic [39:0]        ext_rd_addr,
  output logic [15:0]        ext_rd_len,
  input  logic               ext_rd_grant,
  input  logic [255:0]       ext_rd_data,
  input  logic               ext_rd_valid,

  output logic               ext_wr_req,
  output logic [39:0]        ext_wr_addr,
  input  logic               ext_wr_grant,
  output logic [255:0]       ext_wr_data,
  output logic               ext_wr_valid,

  // Barrier
  output logic               barrier_signal,
  output logic [4:0]         barrier_signal_id,
  input  logic               barrier_grant,

  // Status
  output accel_pkg::tile_state_e state,
  output logic               tile_done,
  output logic               layer_done
);
```

**Internal instantiation**:
```
shadow_reg_file      shadow_regs (.tile_desc, .layer_desc, ...)

glb_input_bank       glb_in[3]   (...)
glb_weight_bank      glb_wt[3]   (...)
glb_output_bank      glb_out[4]  (...)
metadata_ram         meta        (...)

addr_gen_input       agi         (...)
addr_gen_weight      agw         (...)
addr_gen_output      ago         (...)

router_cluster       router      (...)
window_gen           wgen        (...)
pe_cluster           pe          (...)
ppu                  ppu_inst    (...)
swizzle_engine       swiz        (...)

tile_fsm             fsm         (...)
```

---

## M28. `supercluster_wrapper.sv` — 4 Subclusters + Arbiter

**Ports**:
```systemverilog
module supercluster_wrapper #(
  parameter int NUM_SUBS = 4,
  parameter int LANES    = 32
)(
  input  logic               clk, rst_n,

  // From global_scheduler
  input  desc_pkg::tile_desc_t  tile_in,
  input  desc_pkg::layer_desc_t layer_desc,
  input  logic                   tile_valid,
  output logic                   tile_accept,

  // External DDR port (256b)
  output logic [39:0]        axi_araddr,  output logic axi_arvalid,
  input  logic               axi_arready,
  input  logic [255:0]       axi_rdata,   input  logic axi_rvalid,
  output logic               axi_rready,
  output logic [39:0]        axi_awaddr,  output logic axi_awvalid,
  input  logic               axi_awready,
  output logic [255:0]       axi_wdata,   output logic axi_wvalid,
  input  logic               axi_wready,

  // Barrier
  output logic               barrier_signal,
  output logic [4:0]         barrier_signal_id,
  input  logic               barrier_grant,

  // Status
  output logic               layer_done,
  output logic [15:0]        tiles_completed
);
```

**Internal**: 4 `subcluster_wrapper` + 1 `local_arbiter` + ext port mux.

---

## M29. `tensor_dma.sv` — AXI4 DMA Master

**Mục đích**: DMA engine cho load tensor/weight/descriptor từ DDR và store output.

**Ports**:
```systemverilog
module tensor_dma #(
  parameter int AXI_DATA_W = 256,
  parameter int AXI_ADDR_W = 40
)(
  input  logic                  clk, rst_n,

  // AXI4 Master Read
  output logic [AXI_ADDR_W-1:0] m_axi_araddr,
  output logic [7:0]            m_axi_arlen,
  output logic                  m_axi_arvalid,
  input  logic                  m_axi_arready,
  input  logic [AXI_DATA_W-1:0] m_axi_rdata,
  input  logic [1:0]            m_axi_rresp,
  input  logic                  m_axi_rlast,
  input  logic                  m_axi_rvalid,
  output logic                  m_axi_rready,

  // AXI4 Master Write
  output logic [AXI_ADDR_W-1:0] m_axi_awaddr,
  output logic [7:0]            m_axi_awlen,
  output logic                  m_axi_awvalid,
  input  logic                  m_axi_awready,
  output logic [AXI_DATA_W-1:0] m_axi_wdata,
  output logic                  m_axi_wlast,
  output logic                  m_axi_wvalid,
  input  logic                  m_axi_wready,
  input  logic [1:0]            m_axi_bresp,
  input  logic                  m_axi_bvalid,
  output logic                  m_axi_bready,

  // Internal request interface (from subclusters via arbiter)
  input  logic                  rd_req,
  input  logic [AXI_ADDR_W-1:0] rd_addr,
  input  logic [15:0]           rd_byte_len,
  output logic                  rd_data_valid,
  output logic [AXI_DATA_W-1:0] rd_data,
  output logic                  rd_done,

  input  logic                  wr_req,
  input  logic [AXI_ADDR_W-1:0] wr_addr,
  input  logic [15:0]           wr_byte_len,
  input  logic [AXI_DATA_W-1:0] wr_data,
  input  logic                  wr_data_valid,
  output logic                  wr_done
);
```

**Logic**:
```
Read path:
  Split rd_byte_len into AXI bursts (max ARLEN=15 → 16 beats × 32B = 512B per burst)
  Issue AXI AR, collect R data, push to rd_data output

Write path:
  Split wr_byte_len into AXI bursts
  Issue AXI AW, stream W data with WLAST on final beat
  Wait BRESP

Burst parameters:
  ARSIZE = 5 (32 bytes = 256 bits)
  ARBURST = INCR
  ARLEN = min(15, remaining_beats - 1)
```

---

## M30. `accel_top.sv` — Top-Level Module

**Ports**:
```systemverilog
module accel_top (
  input  logic        clk, rst_n,

  // AXI-Lite Slave (CPU MMIO)
  input  logic [11:0] s_axil_awaddr, input logic s_axil_awvalid,
  output logic        s_axil_awready,
  input  logic [31:0] s_axil_wdata,  input logic s_axil_wvalid,
  output logic        s_axil_wready,
  output logic [1:0]  s_axil_bresp,  output logic s_axil_bvalid,
  input  logic        s_axil_bready,
  input  logic [11:0] s_axil_araddr, input logic s_axil_arvalid,
  output logic        s_axil_arready,
  output logic [31:0] s_axil_rdata,  output logic s_axil_rvalid,
  input  logic        s_axil_rready,

  // AXI4 Master (DDR DMA — 256-bit)
  output logic [39:0]  m_axi_araddr,  output logic [7:0] m_axi_arlen,
  output logic         m_axi_arvalid, input  logic m_axi_arready,
  input  logic [255:0] m_axi_rdata,   input  logic m_axi_rvalid,
  input  logic         m_axi_rlast,   output logic m_axi_rready,
  output logic [39:0]  m_axi_awaddr,  output logic [7:0] m_axi_awlen,
  output logic         m_axi_awvalid, input  logic m_axi_awready,
  output logic [255:0] m_axi_wdata,   output logic m_axi_wvalid,
  output logic         m_axi_wlast,   input  logic m_axi_wready,
  input  logic [1:0]   m_axi_bresp,   input  logic m_axi_bvalid,
  output logic         m_axi_bready,

  // Interrupt
  output logic         irq
);
```

**Internal instantiation**:
```
controller_system       ctrl (...)
supercluster_wrapper    sc[4] (...)
tensor_dma              dma (...)
perf_mon                perf (...)

AXI interconnect:
  - ctrl.axi (desc fetch) + dma.axi (tensor load/store) → AXI arbiter → m_axi
  - 4 SC external ports → dma request mux → dma
```

---

# PHẦN J: MODULE → PRIMITIVE MAPPING

## Bảng: Primitive nào dùng module nào

```
┌─────────────────────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│ Module              │ RS3  │ OS1  │ DW3  │ DW7  │ MP5  │ GEMM │ MOVE │ CAT  │ UP   │ EADD │
├─────────────────────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ glb_input_bank      │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │
│ glb_weight_bank     │  ✓   │  ✓   │  ✓   │  ✓   │  -   │  ✓   │  -   │  -   │  -   │  -   │
│ glb_output_bank     │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │
│ addr_gen_input      │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │
│ addr_gen_weight     │  ✓   │  ✓   │  ✓   │  ✓   │  -   │  ✓   │  -   │  -   │  -   │  -   │
│ addr_gen_output     │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │
│ router_cluster      │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  -   │  ✓   │
│ window_gen          │  ✓K3 │  ✓K1 │  ✓K3 │  ✓K7 │  ✓K5 │  ✓K1 │  -   │  -   │  -   │  -   │
│ dsp_pair_int8       │  ✓   │  ✓   │  ✓   │  ✓   │  -   │  ✓   │  -   │  -   │  -   │  -   │
│ pe_unit             │  ✓   │  ✓   │  ✓   │  ✓   │  -   │  ✓   │  -   │  -   │  -   │  -   │
│ pe_cluster          │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  -   │  -   │  -   │  -   │
│ column_reduce       │  ✓   │  ✓   │  ✓   │  ✓   │  -   │  ✓   │  -   │  -   │  -   │  -   │
│ comparator_tree     │  -   │  -   │  -   │  -   │  ✓   │  -   │  -   │  -   │  -   │  -   │
│ ppu (bias+requant)  │  ✓   │  ✓   │  ✓   │  ✓   │  -   │  ✓   │  -   │  ?   │  -   │  ✓   │
│ silu_lut            │  ✓   │  ✓   │  -   │  -   │  -   │  -   │  -   │  -   │  -   │  -   │
│ swizzle_engine      │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  -   │  ✓   │  ✓   │  -   │
│ barrier_manager     │  -   │  -   │  -   │  -   │  -   │  -   │  -   │  ✓   │  -   │  -   │
│ tile_fsm            │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │
└─────────────────────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘

RS3 = RS_DENSE_3x3, OS1 = OS_1x1, DW3 = DW_3x3, DW7 = DW_7x7_MULTIPASS
MP5 = MAXPOOL_5x5, GEMM = GEMM_ATTN, CAT = CONCAT, UP = UPSAMPLE, EADD = EWISE_ADD
? = mini-requant khi scale_A ≠ scale_B
```

---

# PHẦN K: BLOCK → PRIMITIVE → LAYER DECOMPOSITION

## YOLOv10n L0–L22: Mỗi layer cần chạy primitive nào

```
┌───────┬───────────┬──────────────────────────────────────────────────────────────┬──────────────┐
│ Layer │ Block     │ Primitive Sequence (thứ tự thực thi)                         │ Output       │
├───────┼───────────┼──────────────────────────────────────────────────────────────┼──────────────┤
│  L0   │ Conv(s2)  │ RS_DENSE_3x3(3→16, s=2, SiLU)                              │[16,320,320]  │
│  L1   │ Conv(s2)  │ RS_DENSE_3x3(16→32, s=2, SiLU)                             │[32,160,160]  │
│  L2   │ QC2f      │ OS_1x1(32→32) → RS3(16→16) → RS3(16→16) → CAT → OS_1x1   │[32,160,160]  │
│  L3   │ Conv(s2)  │ RS_DENSE_3x3(32→64, s=2, SiLU)                             │[64,80,80]    │
│  L4   │ QC2f      │ OS_1x1(64→64) → RS3(32→32) → RS3(32→32) → CAT → OS_1x1   │[64,80,80] ★  │
│  L5   │ SCDown    │ OS_1x1(64→64) → DW_3x3(64, s=2) → OS_1x1(64→64) →        │[128,40,40]   │
│       │           │ DW_3x3(64, s=2) → CONCAT(64+64)                            │              │
│  L6   │ QC2f      │ OS_1x1 → RS3 → RS3 → CAT → OS_1x1 (Cin=Cout=128)          │[128,40,40] ★ │
│  L7   │ SCDown    │ OS_1x1 → DW_3x3(s=2) → OS_1x1 → DW_3x3(s=2) → CONCAT    │[256,20,20]   │
│  L8   │ QC2f      │ OS_1x1 → RS3 → RS3 → CAT → OS_1x1 (Cin=Cout=256)          │[256,20,20] ★ │
│  L9   │ SPPF      │ OS_1x1(256→128) → MP5 → MP5 → MP5 → CAT(128×4) →         │[256,20,20]   │
│       │           │ OS_1x1(512→256)                                             │              │
│  L10  │ QPSA      │ OS_1x1(split) → GEMM_ATTN(Q,K,V proj→QK^T→soft→×V) →     │[256,20,20]   │
│       │           │ CAT(attn+pass) → OS_1x1                                    │              │
│  L11  │ Upsample  │ UPSAMPLE_NEAREST(×2)                                        │[256,40,40]   │
│  L12  │ QConcat   │ CONCAT(L11 + L6★) [barrier wait L6]                         │[384,40,40]   │
│  L13  │ QC2f      │ OS_1x1 → RS3 → RS3 → CAT → OS_1x1 (384→128)               │[128,40,40] ★ │
│  L14  │ Upsample  │ UPSAMPLE_NEAREST(×2)                                        │[128,80,80]   │
│  L15  │ QConcat   │ CONCAT(L14 + L4★) [barrier wait L4]                         │[192,80,80]   │
│  L16  │ QC2f      │ OS_1x1 → RS3 → RS3 → CAT → OS_1x1 (192→64)                │[64,80,80] P3 │
│  L17  │ Conv(s2)  │ RS_DENSE_3x3(64→64, s=2, SiLU)                             │[64,40,40]    │
│  L18  │ QConcat   │ CONCAT(L17 + L13★) [barrier wait L13]                       │[192,40,40]   │
│  L19  │ QC2f      │ OS_1x1 → RS3 → RS3 → CAT → OS_1x1 (192→128)               │[128,40,40] P4│
│  L20  │ SCDown    │ OS_1x1 → DW_3x3(s=2)                                       │[128,20,20]   │
│  L21  │ QConcat   │ CONCAT(L20 + L8★) [barrier wait L8]                         │[384,20,20]   │
│  L22  │ QC2fCIB   │ OS_1x1(384→256) → DW_7x7×3pass → OS_1x1 → CAT →          │[256,20,20] P5│
│       │           │ OS_1x1(256)                                                 │              │
└───────┴───────────┴──────────────────────────────────────────────────────────────┴──────────────┘

★ = output phải lưu cho skip connection (HOLD_SKIP buffer)
P3/P4/P5 = final outputs gửi về CPU
```

---

# PHẦN L: THỨ TỰ XÂY DỰNG VÀ VERIFY

## Chiến thuật implementation (Bottom-Up)

```
PHASE 1 — Compute Leaf (2 tuần):
  ① dsp_pair_int8.sv       → unit test: all 65536 INT8 pairs, bit-exact
  ② pe_unit.sv              → test: 32-lane MAC accumulate
  ③ comparator_tree.sv      → test: max of 25 random values
  ④ column_reduce.sv        → test: sum 3 rows
  ⑤ silu_lut.sv             → test: preload + 32-parallel lookup

PHASE 2 — PPU (1 tuần):
  ⑥ ppu.sv                  → test: bias + requant + SiLU vs Golden Python
     (pipeline: bias_add → requant → silu → clamp)
     Verify: per-channel M_int/shift correctness

PHASE 3 — Memory (2 tuần):
  ⑦ glb_input_bank.sv       → test: modulo-3 banking, 32 subbanks
  ⑧ glb_weight_bank.sv      → test: 3 reduction lanes + FIFO
  ⑨ glb_output_bank.sv      → test: dual PSUM/ACT mode
  ⑩ addr_gen_input.sv       → test: no address overlap, padding=zp_x
  ⑪ addr_gen_weight.sv      → test: per-mode address patterns
  ⑫ addr_gen_output.sv      → test: output bank mapping

PHASE 4 — Data Movement (2 tuần):
  ⑬ window_gen.sv           → test: K1/K3/K5/K7 tap generation
  ⑭ router_cluster.sv       → test: per-mode routing patterns
  ⑮ swizzle_engine.sv       → test: upsample 20→40, concat offset

PHASE 5 — Integration (2 tuần):
  ⑯ pe_cluster.sv           → test: RS3/OS1/DW3/DW7/MP5/GEMM all modes
  ⑰ subcluster_wrapper.sv   → test: 1 tile RS_DENSE_3x3 end-to-end
  ⑱ shadow_reg_file.sv      → test: descriptor field capture

PHASE 6 — Control (2 tuần):
  ⑲ tile_fsm.sv             → test: FSM transitions, multi-pass accumulation
  ⑳ barrier_manager.sv      → test: 4 YOLOv10n barriers
  ㉑ local_arbiter.sv        → test: dual-RUNNING rotation
  ㉒ desc_fetch_engine.sv    → test: parse NET/LAYER/TILE descriptors
  ㉓ global_scheduler.sv     → test: tile dispatch to 4 SCs

PHASE 7 — System (2 tuần):
  ㉔ supercluster_wrapper.sv → test: 4 subclusters, role rotation
  ㉕ tensor_dma.sv           → test: AXI4 burst read/write
  ㉖ controller_system.sv    → test: CSR read/write, start→done flow
  ㉗ accel_top.sv            → test: full L0 inference, compare Golden Python

PHASE 8 — End-to-End (2 tuần):
  ㉘ Full L0→L22: input X_int8 → P3, P4, P5
     Compare bit-exact with Phase 1 Golden Python
     Verify: all 4 barrier points, skip tensors correct
     Performance: measure cycle count, compute utilization
```

---

# PHẦN M: VERIFICATION CHECKLIST

```
☐ dsp_pair_int8:    65536 pairs, max error = 0
☐ pe_cluster RS3:   Conv3×3 s=1 [3,16,16] 8×8 tile → match Golden Python
☐ pe_cluster RS3:   Conv3×3 s=2 [3,16,16] 16×16 tile → match
☐ pe_cluster OS1:   Conv1×1 [32,64] 8×8 → match
☐ pe_cluster DW3:   DW3×3 s=1 [64] 8×8 → match
☐ pe_cluster DW7:   DW7×7 3-pass [256] 20×20 → match monolithic
☐ pe_cluster MP5:   MaxPool5×5 [128] 20×20 → match
☐ pe_cluster GEMM:  MatMul [400,64]×[400,64]^T → match
☐ ppu:              bias+requant+SiLU per-channel → match Golden Python
☐ ppu:              ewise_add with saturation → match
☐ window_gen:       K7 taps from 32-wide stream → correct spatial values
☐ addr_gen_input:   padding positions output zp_x (NOT zero)
☐ addr_gen_input:   no address collision across (h,w,c) space
☐ swizzle:          upsample 20×20→40×40 content replicated correctly
☐ swizzle:          concat [128ch + 64ch] → [192ch] channel offset correct
☐ barrier:          L6→L12 dependency holds until L6 done
☐ barrier:          L4→L15, L13→L18, L8→L21 all correct
☐ tile_fsm:         multi-pass Cin accumulation: 3 passes → correct final psum
☐ tile_fsm:         DW7 3 passes: psum persists across passes
☐ local_arbiter:    dual-RUNNING: 2 subs compute simultaneously, no corruption
☐ end-to-end L0:    X_int8[1,3,640,640] → [1,16,320,320] bit-exact
☐ end-to-end L0-L4: sequential layers, data passing through GLB/swizzle
☐ end-to-end L0-L22: P3/P4/P5 bit-exact with Golden Python
```

---

*Tài liệu này cung cấp đầy đủ interface, logic, và thứ tự xây dựng cho toàn bộ RTL.
Mỗi module description đủ chi tiết để prompt sinh SystemVerilog code.
Tính đúng đắn được đảm bảo bởi: bottom-up verification, bit-exact comparison với Golden Python Phase 1.*
