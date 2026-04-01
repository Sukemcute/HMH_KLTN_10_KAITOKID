# CHECKLIST XÂY DỰNG RTL V4-VC707 — STEP BY STEP
## YOLOv10n INT8 Accelerator | LANES=20 | 250 MHz | 16 Subclusters

> **Ngày**: 2026-03-30
> **Target**: VC707 (XC7VX485T), 250 MHz, ~192 FPS
> **Kiến trúc**: V4-VC707 (LANES=20, 4SC×4Sub, Triple-RUNNING)
> **Nguyên tắc**: Mỗi step phải PASS 100% trước khi qua step kế tiếp.
> **Ngôn ngữ**: SystemVerilog (IEEE 1800-2017)
> **Tham chiếu**: FPS_RESOURCE_INFERENCE_ANALYSIS.md (Phần II+III)

---

## MỤC LỤC

```
STAGE 0: Nền tảng — Packages & Golden Rules
STAGE 1: Compute Atoms — dsp_pair, pe_unit, column_reduce, comparator_tree
STAGE 2: Post-Processing — PPU (bias + requant + ReLU + clamp)
STAGE 3: Memory — GLB banks (double-buffer), metadata_ram
STAGE 4: Address Generation — addr_gen_input, addr_gen_weight, addr_gen_output
STAGE 5: Data Movement — router_cluster_v2, window_gen, swizzle_engine
STAGE 6: Control — tile_fsm, shadow_reg_file, compute_sequencer
STAGE 7: Subcluster Integration — subcluster_datapath (wire-up all)
STAGE 8: Primitive Verification — 7 pe_modes trên CÙNG 1 HW
STAGE 9: SuperCluster — arbiter_v2, tensor_dma_v2, SC wrapper
STAGE 10: System — scheduler, desc_fetch, barrier, CSR, accel_top
STAGE 11: Block Verification — QC2f, SCDown, SPPF, QConcat, Upsample, QC2fCIB, QPSA
STAGE 12: Layer-by-Layer Verification — L0 → L22
STAGE 13: Full Inference — L0-L22 chain, P3/P4/P5 output
STAGE 14: FPGA Deployment — Synthesis, P&R, Board Test
```

---

# ════════════════════════════════════════════════════════════════
# 10 QUY TẮC VÀNG — ĐỌC TRƯỚC KHI VIẾT BẤT KỲ DÒNG RTL NÀO
# ════════════════════════════════════════════════════════════════

```
RULE 1:  SIGNED INT8 [-128, 127] mọi nơi.
         ZP_hw = ZP_pytorch - 128. Weight ZP = 0 (symmetric).

RULE 2:  HALF-UP ROUNDING:
         y = (acc × M_int + (1 << (shift-1))) >> shift
         KHÔNG BAO GIỜ dùng floor: y = (acc × M_int) >> shift

RULE 3:  INT32 cho MAC accumulator. INT64 cho PPU multiply.
         mult = int64(biased) × int64(M_int)

RULE 4:  Model dùng ReLU: y = max(0, x). KHÔNG dùng SiLU.

RULE 5:  Padding fill = zero_point_x (KHÔNG PHẢI 0).
         Trong INT8 quantized, "true zero" = ZP, không phải literal 0.

RULE 6:  Per-output-channel: bias[cout], m_int[cout], shift[cout].
         Per-input-channel cho DW: bias[ch], m_int[ch], shift[ch].

RULE 7:  CONCAT cần domain alignment: requant_to_common nếu scale khác.

RULE 8:  DW_7x7 multipass: PSUM namespace pass 1,2. PPU chỉ ở pass 3.

RULE 9:  Barrier sync cho 4 skip connections: L12, L15, L18, L21.

RULE 10: Output = clamp(activated + zp_out, -128, 127).
         ZP_out thêm SAU activation, TRƯỚC final clamp.
```

---

# ════════════════════════════════════════════════════════════════
# V4 PARAMETERS — THAM SỐ XUYÊN SUỐT TOÀN BỘ THIẾT KẾ
# ════════════════════════════════════════════════════════════════

```
LANES               = 20       // ★ Magic number: chia hết 320,160,80,40,20
PE_ROWS              = 3        // kh parallelism
PE_COLS              = 4        // cout parallelism (per-column weight)
DSP_PAIRS_PER_PE     = 10       // = LANES/2
MACS_PER_PE          = 20       // = LANES
MACS_PER_SUB         = 240      // = PE_ROWS × PE_COLS × MACS_PER_PE
DSP_PER_SUB          = 120      // = PE_ROWS × PE_COLS × DSP_PAIRS_PER_PE

N_SUPER_CLUSTERS     = 4
N_SUBS_PER_SC        = 4        // ★ Triple-RUNNING
N_ACTIVE_PER_SC      = 3
N_TOTAL_SUBS         = 16
N_TOTAL_ACTIVE       = 12

TARGET_CLOCK_MHZ     = 250      // ★ High-freq (LANES=20 cho phép)
DSP_PIPELINE_DEPTH   = 5        // ★ Sâu hơn V3 (4) cho Fmax
PPU_PIPELINE_DEPTH   = 5

GLB_INPUT_PAGES      = 2        // ★ Double-buffer (ping-pong)
GLB_INPUT_DEPTH      = 2048     // Per bank per page
GLB_WEIGHT_DEPTH     = 1024     // Per bank
GLB_OUTPUT_DEPTH     = 512      // Per bank
WEIGHT_READ_PORTS    = 4        // Per-column weight
```

---
---

# STAGE 0: NỀN TẢNG — PACKAGES
**Mục tiêu**: Định nghĩa types, enums, structs dùng chung. Không có logic.

---

## ☐ 0.1 `accel_pkg.sv`

**File**: `rtl/packages/accel_pkg.sv`

| Thuộc tính | Mô tả |
|---|---|
| **Chức năng** | Package chứa TẤT CẢ parameters, types, enums cho toàn design |
| **Input** | Không (package, không phải module) |
| **Output** | Không |
| **Import** | `import accel_pkg::*;` ở MỌI module khác |

**Nội dung cần define**:
```
☐ Parameters: LANES=20, PE_ROWS=3, PE_COLS=4, DSP_PAIRS_PER_PE=10, ...
☐ Hierarchy: N_SUPER_CLUSTERS=4, N_SUBS_PER_SC=4, N_ACTIVE_PER_SC=3, ...
☐ Memory: GLB_INPUT_DEPTH=2048, GLB_INPUT_PAGES=2, GLB_WEIGHT_DEPTH=1024, ...
☐ Pipeline: DSP_PIPELINE_DEPTH=5, PPU_PIPELINE_DEPTH=5
☐ pe_mode_t enum: PE_RS3, PE_OS1, PE_DW3, PE_DW7, PE_MP5, PE_PASS, PE_GEMM
☐ act_type_t enum: ACT_NONE, ACT_RELU, ACT_SILU, ACT_RELU6
☐ swizzle_mode_t enum: SWZ_NORMAL, SWZ_UPSAMPLE2X, SWZ_CONCAT
☐ tile_state_t enum: TS_IDLE → TS_LOAD_DESC → ... → TS_DONE (10 states)
☐ Data types: int8_t, int32_t, int64_t (typedef logic signed)
```

**Test**: Compile-only (no simulation). Verify: `import accel_pkg::*;` works.
**Pass criteria**: Zero compilation errors.

---

## ☐ 0.2 `desc_pkg.sv`

**File**: `rtl/packages/desc_pkg.sv`

| Thuộc tính | Mô tả |
|---|---|
| **Chức năng** | Descriptor struct definitions (3-level hierarchy) |

**Nội dung cần define**:
```
☐ net_desc_t:   magic, version, num_layers, weight_arena_base, act_arena_base
☐ layer_desc_t: pe_mode, activation, cin/cout, hin/win/hout/wout, kh/kw,
                 stride, padding, num_tiles, num_cin_pass, num_k_pass,
                 swizzle, router_profile_id, post_profile_id
☐ tile_desc_t:  tile_id, layer_id, sc_mask, h_out0, wblk0, cin0, cout0,
                 src_in_off, src_w_off, src_skip_off, dst_off,
                 first_tile, last_tile, hold_skip, need_swizzle,
                 barrier_wait, barrier_id
```

**Pass criteria**: Zero compilation errors. Struct sizes match spec.

---

## ☐ 0.3 `csr_pkg.sv`

**File**: `rtl/packages/csr_pkg.sv`

**Nội dung**: CSR address map (CSR_CTRL=0x000, CSR_STATUS=0x004, ...).
**Pass criteria**: Compile clean.

---
---

# STAGE 1: COMPUTE ATOMS
**Mục tiêu**: Chứng minh phép tính MAC bit-exact. PHẢI 100% đúng.
**Nguyên tắc**: KHÔNG tolerance. 0 errors.

---

## ☐ 1.1 `dsp_pair_int8.sv` — 2-MAC DSP48E1 Packing

**File**: `rtl/compute/dsp_pair_int8.sv`

| Thuộc tính | Mô tả |
|---|---|
| **Chức năng** | 2 phép signed INT8 × INT8 → INT32 accumulate trong 1 DSP48E1 |
| **Pipeline** | ★ 5 stages (V4, tăng từ V3's 4 cho 250 MHz) |
| **Instances** | 10 per PE × 12 PEs/sub × 16 subs = **1,920 total** |
| **Resources** | 1 DSP48E1 + ~25 LUT + ~40 FF per instance |

**Ports**:
```systemverilog
module dsp_pair_int8 (
  input  logic                clk,
  input  logic                rst_n,
  input  logic signed [7:0]   a0, b0,       // MAC pair 0: a0 × b0
  input  logic signed [7:0]   a1, b1,       // MAC pair 1: a1 × b1
  input  logic                enable,        // Accumulate when high
  input  logic                clear,         // Reset accumulator to 0
  output logic signed [31:0]  acc_out0,      // Running sum pair 0
  output logic signed [31:0]  acc_out1       // Running sum pair 1
);
```

**5-Stage Pipeline** (★ V4: +1 stage vs V3):
```
Stage 1: Input register latch (a0_r, b0_r, a1_r, b1_r)
Stage 2: Unsigned offset: a0_u = a0_r + 128, b0_u = b0_r + 128  [0..255]
Stage 3: DSP48E1 multiply: P = packed_A × packed_B (25×18 unsigned)
Stage 4: Correction + accumulate: signed_prod = raw - 128×(a_u+b_u) + 16384
Stage 5: Output register (★ V4 extra stage for timing)
```

**Công thức correction (unsigned-offset trick)**:
```
a_s × b_s = (a_u - 128)(b_u - 128) = a_u × b_u - 128×a_u - 128×b_u + 16384
→ correction_term = 128 × (a_u + b_u) - 16384
→ signed_product = unsigned_product - correction_term
```

**Test Plan**:
```
☐ Test 1.1.1: Exhaustive corners (17 pairs)
   (-128)×(-128)=16384, 127×127=16129, (-128)×127=-16256,
   0×0=0, 1×1=1, (-1)×(-1)=1, (-1)×1=-1, ...
   CÙNG LÚC cho cả pair0 và pair1.
   Pass: ALL exact match.

☐ Test 1.1.2: Full exhaustive 256×256 = 65,536 products (per pair)
   Iterate a ∈ [-128..127], b ∈ [-128..127].
   Verify: acc_out = a × b (single cycle, clear before each).
   Pass: 0 errors out of 65,536.

☐ Test 1.1.3: 9-cycle accumulation (Conv 3×3 pattern)
   Feed 9 (a,b) pairs with clear at start, enable for 9 cycles.
   Verify: acc_out = Σ_{i=0..8} a[i] × b[i] vs golden sum.
   Test 20 random sequences.
   Pass: ALL exact match.

☐ Test 1.1.4: 49-cycle accumulation (DW 7×7 pattern)
   Feed 49 (a,b) pairs. Verify extended accumulation.
   Pass: ALL exact match.

☐ Test 1.1.5: Random stress 10,000 products
   Random (a,b) pairs, random enable/clear patterns.
   Pass: 0 errors.

☐ Test 1.1.6: Pipeline timing verification
   Verify: 5-cycle latency from input to valid output.
   Verify: enable=0 → accumulator holds value.
   Verify: clear → accumulator resets to 0 on next cycle.
```

**PASS CRITERIA**: 0 errors. Không có tolerance. Đây là nền tảng.

---

## ☐ 1.2 `pe_unit.sv` — 20-Lane Processing Element

**File**: `rtl/compute/pe_unit.sv`

| Thuộc tính | Mô tả |
|---|---|
| **Chức năng** | 20 parallel MACs = 10 dsp_pair_int8 instances |
| **LANES** | ★ 20 (V4, giảm từ V3's 32) |
| **Instances** | 12 per cluster × 16 subs = **192 total** |
| **Resources** | 10 DSP + ~250 LUT + ~400 FF per instance |

**Ports**:
```systemverilog
module pe_unit #(
  parameter LANES = 20
)(
  input  logic                clk,
  input  logic                rst_n,
  input  pe_mode_t            pe_mode,
  input  logic signed [7:0]   x_in  [LANES],   // 20 activation values
  input  logic signed [7:0]   w_in  [LANES],   // 20 weight values
  input  logic                pe_enable,
  input  logic                clear_acc,
  output logic signed [31:0]  psum_out [LANES]  // 20 accumulated results
);
```

**Internal**: generate 10 dsp_pair_int8, connecting lanes [0,1], [2,3], ..., [18,19].

**Test Plan**:
```
☐ Test 1.2.1: PE_RS3 mode (per-lane weight)
   Feed 20 distinct (x,w) pairs per cycle, 9 cycles (3×3 kernel).
   Verify: psum_out[l] = Σ_{i=0..8} x_in[l][i] × w_in[l][i]  per lane.
   Pass: 0 errors across 20 lanes.

☐ Test 1.2.2: PE_OS1 mode (broadcast weight)
   Feed 20 distinct x_in, BUT w_in[0..19] = same value (broadcast).
   Verify: each lane accumulates with same weight.
   Pass: 0 errors.

☐ Test 1.2.3: PE_DW3 mode (per-channel)
   Feed per-channel data: lane = channel.
   Verify: each lane independently accumulates.
   Pass: 0 errors.

☐ Test 1.2.4: Enable gating
   Set pe_enable=0 mid-accumulation → verify psum holds.
   Re-enable → verify accumulation resumes.
   Pass: psum unchanged when disabled.

☐ Test 1.2.5: Clear + immediate start
   Clear → enable on next cycle → verify fresh accumulation.
   Pass: no residual from previous tile.
```

**PASS CRITERIA**: 0 errors across all modes and all 20 lanes.

---

## ☐ 1.3 `column_reduce.sv` — 3-Row Summation

**File**: `rtl/compute/column_reduce.sv`

| Thuộc tính | Mô tả |
|---|---|
| **Chức năng** | Sum 3 PE row outputs per column → kh dimension reduction |
| **Instances** | 4 per cluster × 16 subs = **64 total** |
| **Latency** | 1 cycle (registered output for timing) |

**Ports**:
```systemverilog
module column_reduce #(
  parameter LANES = 20,
  parameter N_ROWS = 3
)(
  input  logic                clk,
  input  logic signed [31:0]  row_psum [N_ROWS][LANES],  // 3 × 20 INT32
  input  logic                valid_in,
  output logic signed [31:0]  col_sum  [LANES],           // 20 INT32
  output logic                valid_out
);
```

**Công thức**: `col_sum[l] = row_psum[0][l] + row_psum[1][l] + row_psum[2][l]`

**Test Plan**:
```
☐ Test 1.3.1: Known values
   row[0]=[100, -50, ...], row[1]=[200, 30, ...], row[2]=[-150, 20, ...]
   Verify: col=[150, 0, ...]
   Pass: Exact match all 20 lanes.

☐ Test 1.3.2: Overflow boundary
   row[0]=[MAX_INT32/3], row[1]=[MAX_INT32/3], row[2]=[MAX_INT32/3]
   Verify: no overflow within INT32 (guaranteed by design constraints).

☐ Test 1.3.3: Random 200 vectors
   Random INT32 values across 3×20 → verify sum.
   Pass: 0 errors.
```

---

## ☐ 1.4 `comparator_tree.sv` — MaxPool 5×5

**File**: `rtl/compute/comparator_tree.sv`

| Thuộc tính | Mô tả |
|---|---|
| **Chức năng** | 25 signed INT8 inputs → 1 maximum per lane, 5-stage pipeline |
| **Sử dụng** | PE_MP5 mode only (SPPF L9) |
| **Latency** | 5 cycles |

**Ports**:
```systemverilog
module comparator_tree #(
  parameter LANES = 20,
  parameter K = 5          // 5×5 = 25 inputs
)(
  input  logic                clk,
  input  logic                rst_n,
  input  logic signed [7:0]   window [K*K][LANES],  // 25 × 20
  input  logic                valid_in,
  output logic signed [7:0]   max_out [LANES],       // 20 maximums
  output logic                valid_out
);
```

**Test Plan**:
```
☐ Test 1.4.1: Signed boundary tests
   All inputs = -128 → max = -128. All = 127 → max = 127.
   One = 127 rest = -128 → max = 127 (verify position doesn't matter).
   Pass: Exact.

☐ Test 1.4.2: Gradient test
   Inputs = [0, 1, 2, ..., 24] per lane → max = 24.
   Pass: Exact.

☐ Test 1.4.3: Per-lane independence
   Different max positions per lane → verify each lane independent.
   Pass: Exact.

☐ Test 1.4.4: Pipeline timing
   Verify: 5-cycle latency, valid_out tracks valid_in with delay.
   Pass: Timing correct.

☐ Test 1.4.5: Random stress 500 windows
   Random signed INT8, compare vs behavioral max().
   Pass: 0 errors.
```

**STAGE 1 SIGN-OFF**: Mọi compute atom PHẢI 100% bit-exact. Không tolerance.

---
---

# STAGE 2: POST-PROCESSING — PPU
**Mục tiêu**: Bias + Requant + Activation đúng bit-exact với Golden Python.
**★ ĐÂY LÀ MODULE QUAN TRỌNG NHẤT cho accuracy toàn hệ thống.**

---

## ☐ 2.1 `ppu.sv` — Post-Processing Unit

**File**: `rtl/compute/ppu.sv`

| Thuộc tính | Mô tả |
|---|---|
| **Chức năng** | Bias → INT64 Requant → Activation → Clamp → INT8 output |
| **Pipeline** | ★ 5 stages (V4) |
| **Instances** | 4 per sub (1 per PE column) × 16 subs = **64 total** |
| **★ CRITICAL** | Rounding, overflow, activation order — MỌI thứ phải đúng |

**Ports**:
```systemverilog
module ppu #(
  parameter LANES = 20
)(
  input  logic                    clk,
  input  logic                    rst_n,
  // Input (from column_reduce)
  input  logic signed [31:0]      psum_in [LANES],    // 20 INT32 partial sums
  input  logic                    psum_valid,
  // Quantization parameters (from shadow_reg_file, per-cout)
  input  logic signed [31:0]      bias_val,           // B_int32[cout]
  input  logic        [31:0]      m_int,              // M_int[cout] (unsigned)
  input  logic        [7:0]       shift_val,          // shift[cout]
  input  logic signed [7:0]       zp_out,             // ZP_out[cout]
  input  act_type_t               activation,          // ACT_RELU / ACT_NONE
  // Output
  output logic signed [7:0]       act_out [LANES],    // 20 INT8 results
  output logic                    act_valid
);
```

**5-Stage Pipeline (★ PHẢI ĐÚNG CHÍNH XÁC)**:
```
Stage 1 — BIAS ADD:
  biased[l] = psum_in[l] + bias_val             // INT32 + INT32 → INT32

Stage 2 — INT64 MULTIPLY:
  product[l] = int64(biased[l]) × int64(m_int)  // ★ PHẢI là INT64!
  // Nếu dùng INT32 → overflow → sai hoàn toàn

Stage 3 — HALF-UP ROUNDING + SHIFT:
  if (shift_val > 0)
    rounded[l] = product[l] + (64'sd1 <<< (shift_val - 1))   // ★ Half-up!
  else
    rounded[l] = product[l]
  shifted[l] = int32(rounded[l] >>> shift_val)
  // ★ KHÔNG ĐƯỢC dùng: shifted = product >> shift  (đó là FLOOR, SAI!)

Stage 4 — ACTIVATION:
  if (activation == ACT_RELU)
    activated[l] = (shifted[l] > 0) ? shifted[l] : 32'sd0   // max(0, x)
  else // ACT_NONE
    activated[l] = shifted[l]

Stage 5 — CLAMP + ZP_OUT:
  with_zp[l] = activated[l] + int32(zp_out)
  act_out[l] = clamp(with_zp[l], -128, 127)
  // clamp: if (with_zp > 127) return 127;
  //        if (with_zp < -128) return -128;
  //        else return with_zp[7:0];
```

**Test Plan (★ MỌI test PHẢI pass)**:
```
☐ Test 2.1.1: Half-up rounding vs Floor
   Input: biased=15, M_int=100, shift=3
   Floor: (15×100) >> 3 = 1500 >> 3 = 187
   Half-up: (15×100 + 4) >> 3 = 1504 >> 3 = 188  ← PHẢI ra 188
   Test 100 cases where half-up ≠ floor.
   Pass: ALL match half-up.

☐ Test 2.1.2: INT64 overflow protection
   Input: biased = 2,000,000,000 (near INT32 max), M_int = 1,500,000,000
   Product = 3×10^18 > INT32_MAX (2.1×10^9) → PHẢI dùng INT64.
   Verify: no truncation/wrap.
   Pass: Exact INT64 result.

☐ Test 2.1.3: ReLU activation
   shifted = [-100, -1, 0, 1, 50, 127, 200]
   Expected after ReLU: [0, 0, 0, 1, 50, 127, 200]
   (Before clamp+ZP).
   Pass: Exact.

☐ Test 2.1.4: ZP_out + final clamp
   activated = 120, zp_out = 10 → with_zp = 130 → clamp → 127
   activated = -50, zp_out = -80 → with_zp = -130 → clamp → -128
   activated = 50, zp_out = 20 → with_zp = 70 → output = 70
   Pass: Exact boundary behavior.

☐ Test 2.1.5: Golden Python vector comparison
   Extract from L0 verification: 100 real psum values + real quant params.
   Run through PPU → compare vs Golden Python output.
   Pass: 100% bit-exact.

☐ Test 2.1.6: Per-lane independence
   Feed different psum_in per lane with SAME quant params.
   Verify: each lane processes independently.
   Pass: 0 errors.

☐ Test 2.1.7: ACT_NONE mode (no activation)
   Verify: activation stage is identity (pass-through).
   Used in: intermediate PSUM accumulation, some concat paths.
   Pass: output = clamp(shifted + zp_out).

☐ Test 2.1.8: Stress 1,000 random vectors
   Random: psum ∈ [-10^8, 10^8], M_int ∈ [1, 2^31], shift ∈ [0, 31],
           zp_out ∈ [-128, 127], activation ∈ {RELU, NONE}.
   Compare vs Python golden: ppu_golden(psum, bias, m, sh, zp, act).
   Pass: 100% bit-exact.
```

**STAGE 2 SIGN-OFF**: PPU 100% bit-exact với Golden Python. Đây là gate quan trọng nhất.

---
---

# STAGE 3: MEMORY SUBSYSTEM
**Mục tiêu**: GLB banks đọc/ghi đúng, double-buffer hoạt động, dual-namespace PSUM/ACT.

---

## ☐ 3.1 `glb_input_bank_db.sv` — Double-Buffered Input SRAM

**File**: `rtl/memory/glb_input_bank_db.sv`

| Thuộc tính | Mô tả |
|---|---|
| **Chức năng** | ★ Double-buffered input activation SRAM. Page A/B ping-pong. |
| **Instances** | 3 per sub × 16 subs = **48 total** |
| **V4 Key** | Double-buffer loại bỏ fill stall |
| **Resources** | ~150 LUT, ~80 FF, ~2.5 BRAM per bank |

**Ports**:
```systemverilog
module glb_input_bank_db #(
  parameter LANES = 20,
  parameter DEPTH = 2048
)(
  input  logic                     clk,
  input  logic                     rst_n,
  // Page control
  input  logic                     page_swap,           // Toggle active page
  // Compute read port (active page)
  input  logic [$clog2(DEPTH)-1:0] rd_addr,
  output logic signed [7:0]        rd_data [LANES],     // 20 INT8
  // DMA write port (shadow page)
  input  logic [$clog2(DEPTH)-1:0] wr_addr,
  input  logic signed [7:0]        wr_data [LANES],
  input  logic                     wr_en,
  input  logic [LANES-1:0]         wr_lane_mask          // Per-lane write enable
);
```

**Test Plan**:
```
☐ Test 3.1.1: Basic write → read
   Write pattern to page A → swap → read from page A → verify data.
   Pass: All data matches.

☐ Test 3.1.2: Double-buffer independence
   Write "AAA" to page A, "BBB" to page B (via swap + write).
   Read page A → "AAA". Swap. Read page B (now active) → "BBB".
   Pass: Pages independent.

☐ Test 3.1.3: Concurrent read + write
   While reading page A (compute), write page B (DMA fill).
   Verify: read data from A unaffected by writes to B.
   Pass: No cross-page interference.

☐ Test 3.1.4: Lane mask
   Write with wr_lane_mask = 20'b0000_0000_0011_1111_1111 (lanes 0-9 only).
   Verify: lanes 0-9 updated, lanes 10-19 unchanged.
   Pass: Exact lane control.

☐ Test 3.1.5: Address sweep (full depth)
   Write all 2048 addresses → read back → verify.
   Pass: 0 errors.
```

---

## ☐ 3.2 `glb_weight_bank.sv` — 4-Read-Port Weight SRAM

**File**: `rtl/memory/glb_weight_bank.sv`

| Thuộc tính | Mô tả |
|---|---|
| **Chức năng** | Weight SRAM với 4 read ports (1 per PE column = per-col weight) |
| **Instances** | 3 per sub × 16 subs = **48 total** |
| **V4 Key** | 4 independent read addresses → 4 khác cout/channel cùng lúc |
| **Implementation** | 4× BRAM duplicate: write broadcast, read independent |

**Ports**:
```systemverilog
module glb_weight_bank #(
  parameter LANES = 20,
  parameter DEPTH = 1024,
  parameter N_READ_PORTS = 4
)(
  input  logic                          clk,
  input  logic                          rst_n,
  // 4 independent read ports (per PE column)
  input  logic [$clog2(DEPTH)-1:0]      rd_addr [N_READ_PORTS],
  output logic signed [7:0]             rd_data [N_READ_PORTS][LANES],
  // 1 write port (DMA fill, broadcast to all 4 copies)
  input  logic [$clog2(DEPTH)-1:0]      wr_addr,
  input  logic signed [7:0]             wr_data [LANES],
  input  logic                          wr_en
);
```

**Test Plan**:
```
☐ Test 3.2.1: Write broadcast → 4 independent reads
   Write data at addr=0. Read addr=0 from all 4 ports → same data.
   Pass: All 4 ports return identical data.

☐ Test 3.2.2: Different addresses per port
   Write data at addr 0, 10, 20, 30.
   Read: port0=addr0, port1=addr10, port2=addr20, port3=addr30.
   Verify: each port gets its own data.
   Pass: All 4 ports return correct data.

☐ Test 3.2.3: Simultaneous read + write
   Write to addr X while reading from addr Y → no interference.
   Pass: Read data stable.
```

---

## ☐ 3.3 `glb_output_bank.sv` — Dual-Namespace PSUM/ACT

**File**: `rtl/memory/glb_output_bank.sv`

| Thuộc tính | Mô tả |
|---|---|
| **Chức năng** | Dual namespace: PSUM (INT32, for multipass) + ACT (INT8, final) |
| **Instances** | 4 per sub (1 per PE column) × 16 subs = **64 total** |

**Ports**:
```systemverilog
module glb_output_bank #(
  parameter LANES = 20,
  parameter DEPTH = 512
)(
  input  logic                          clk,
  input  logic                          rst_n,
  // PSUM namespace (INT32, multipass accumulation)
  input  logic [$clog2(DEPTH)-1:0]      psum_addr,
  input  logic signed [31:0]            psum_wr_data [LANES],
  input  logic                          psum_wr_en,
  output logic signed [31:0]            psum_rd_data [LANES],
  input  logic                          psum_rd_en,
  // ACT namespace (INT8, final PPU output)
  input  logic [$clog2(DEPTH)-1:0]      act_addr,
  input  logic signed [7:0]             act_wr_data [LANES],
  input  logic                          act_wr_en,
  output logic signed [7:0]             act_rd_data [LANES],
  input  logic                          act_rd_en,
  // DMA drain port
  input  logic [$clog2(DEPTH)-1:0]      drain_addr,
  output logic [LANES*8-1:0]            drain_data     // Packed for DMA
);
```

**Test Plan**:
```
☐ Test 3.3.1: PSUM write → read cycle
   Write INT32 PSUM at addr 0 → read back → verify.

☐ Test 3.3.2: ACT write → read cycle
   Write INT8 ACT at addr 0 → read back → verify.

☐ Test 3.3.3: Namespace independence
   Write PSUM at addr 5 → write ACT at addr 5.
   Read PSUM addr 5 → original PSUM (not ACT). Read ACT addr 5 → correct ACT.
   Pass: Namespaces don't interfere.

☐ Test 3.3.4: DW7x7 multipass simulation
   Pass 1: Write PSUM. Pass 2: Read PSUM + add + write PSUM.
   Pass 3: Read PSUM + add → PPU → write ACT.
   Verify: final ACT = sum of all 3 passes.
```

---

## ☐ 3.4 `metadata_ram.sv` — Slot Validity Manager

**File**: `rtl/memory/metadata_ram.sv`

| Thuộc tính | Mô tả |
|---|---|
| **Chức năng** | Ring buffer pointers + slot validity for multipass PSUM |
| **Resources** | ~50 LUT, ~30 FF, 1 BRAM |

**Test Plan**:
```
☐ Alloc/free cycle: allocate slot → check valid → free → check invalid.
☐ Ring buffer wrap: allocate all 16 slots → verify wrap behavior.
```

---
---

# STAGE 4: ADDRESS GENERATION
**Mục tiêu**: Mỗi mode (RS3/OS1/DW3/DW7/MP5) sinh đúng địa chỉ SRAM.

---

## ☐ 4.1 `addr_gen_input.sv`

**File**: `rtl/addr_gen/addr_gen_input.sv`

| Thuộc tính | Mô tả |
|---|---|
| **Chức năng** | (h_in, w, cin) → (bank_id, sram_addr, is_padding, pad_value) |
| **★ CRITICAL** | Padding fill = zp_x (KHÔNG PHẢI 0!) |

**Ports**:
```systemverilog
module addr_gen_input #(
  parameter LANES = 20
)(
  input  logic                clk, rst_n,
  input  pe_mode_t            cfg_pe_mode,
  input  logic [9:0]          cfg_hin, cfg_win, cfg_cin,
  input  logic [2:0]          cfg_stride, cfg_padding,
  input  logic signed [7:0]   cfg_zp_x,
  // From compute_sequencer
  input  logic [9:0]          iter_h, iter_w, iter_cin,
  input  logic [3:0]          iter_kh_row,
  // Output
  output logic [1:0]          bank_id,           // h_in mod 3
  output logic [11:0]         sram_addr,
  output logic                is_padding,
  output logic signed [7:0]   pad_value          // = cfg_zp_x
);
```

**Test Plan**:
```
☐ Test 4.1.1: Banking pattern
   h_in=0 → bank 0, h_in=1 → bank 1, h_in=2 → bank 2, h_in=3 → bank 0.
   Pass: Exact.

☐ Test 4.1.2: Padding detection
   Config: hin=8, padding=1. h_in = -1 → is_padding=1, pad_value=zp_x.
   h_in = 8 → is_padding=1. h_in = 0..7 → is_padding=0.
   Pass: ★ pad_value = cfg_zp_x (NOT 0).

☐ Test 4.1.3: L0 sweep
   Config: hin=640, win=640, stride=2, padding=1.
   Sweep all (h,w) for tile → verify banking + padding boundaries.
   Pass: No out-of-bounds access.
```

---

## ☐ 4.2 `addr_gen_weight.sv` — ★ Per-Column Weight Addressing

**File**: `rtl/addr_gen/addr_gen_weight.sv`

| Thuộc tính | Mô tả |
|---|---|
| **Chức năng** | ★ 4 KHÁC addresses cho 4 PE columns = 4 khác cout/channel |
| **V4 Key** | Core của Eyeriss-v2 per-column routing |

**Ports**:
```systemverilog
module addr_gen_weight #(
  parameter LANES = 20,
  parameter PE_COLS = 4
)(
  input  pe_mode_t            cfg_pe_mode,
  input  logic [9:0]          cfg_cin, cfg_cout,
  input  logic [3:0]          cfg_kw,
  // From compute_sequencer
  input  logic [9:0]          iter_cin, iter_cout_group,
  input  logic [3:0]          iter_kw, iter_kh_row,
  // Output: ★ 4 DIFFERENT addresses
  output logic [9:0]          wgt_addr [PE_COLS],
  output logic [1:0]          wgt_bank_id
);
```

**Per-mode formulas**:
```
PE_RS3:  wgt_addr[col] = (cout_base+col) × Cin × Kw + cin × Kw + kw
         cout_base = iter_cout_group × 4
PE_OS1:  wgt_addr[col] = (cout_base+col) × Cin + cin
PE_DW3:  wgt_addr[col] = (ch_base+col) × Kw + kw
         ch_base = iter_cout_group × 4 (channels, not cout)
PE_DW7:  Similar to DW3 with kh offset per pass
```

**Test Plan**:
```
☐ Test 4.2.1: PE_RS3 — 4 different cout addresses
   cout_group=0 → addr[0] for cout=0, addr[1] for cout=1, etc.
   Verify: 4 addresses are DIFFERENT and correspond to 4 consecutive couts.
   Pass: Exact formula match.

☐ Test 4.2.2: PE_OS1 — 4 different cout addresses (simpler)
   No kw dimension. Verify: addr[col] = (base+col) × Cin + cin.
   Pass: Exact.

☐ Test 4.2.3: PE_DW3 — 4 different CHANNEL addresses
   In DW mode, 4 columns = 4 khác channels (NOT cout).
   addr[col] = (ch_base+col) × 3 + kw.
   Pass: Exact.

☐ Test 4.2.4: Full L0 weight address sweep
   Config: L0 (Cin=3, Cout=16, kh=3, kw=3).
   Sweep all (cout_group, cin, kw) → verify no overlap, full coverage.
   Pass: All weight addresses correct and complete.
```

---

## ☐ 4.3 `addr_gen_output.sv`

**File**: `rtl/addr_gen/addr_gen_output.sv`

| Thuộc tính | Mô tả |
|---|---|
| **Chức năng** | PE column index → output bank_id (direct mapping) |

**Key**: `bank_id[col] = col` (column 0 → bank 0, column 1 → bank 1, etc.)

**Test**: Verify mapping for all (h_out, wblk, cout_group) combinations.

---
---

# STAGE 5: DATA MOVEMENT
**Mục tiêu**: Router multicast + per-col weight + bypass. Window sliding. Swizzle.

---

## ☐ 5.1 `router_cluster_v2.sv` — ★ Eyeriss-Inspired Router

**File**: `rtl/data_movement/router_cluster_v2.sv`

| Thuộc tính | Mô tả |
|---|---|
| **Chức năng** | 3 routing networks: RIN (multicast act), RWT (per-col wgt), RPS (output), Bypass |
| **★ V4 Key** | Per-column weight routing = core of Eyeriss optimization |

**Ports**:
```systemverilog
module router_cluster_v2 #(
  parameter LANES = 20,
  parameter PE_ROWS = 3,
  parameter PE_COLS = 4
)(
  input  logic                 clk, rst_n,
  input  pe_mode_t             cfg_pe_mode,
  // RIN: Input activation routing (GLB → PE rows, MULTICAST)
  input  logic signed [7:0]    glb_in_data   [3][LANES],
  output logic signed [7:0]    pe_act        [PE_ROWS][LANES],
  // RWT: Weight routing (GLB → PE rows × columns, ★ PER-COL)
  input  logic signed [7:0]    glb_wgt_data  [3][PE_COLS][LANES],
  output logic signed [7:0]    pe_wgt        [PE_ROWS][PE_COLS][LANES],
  // RPS: Output routing (PE columns → GLB output banks)
  input  logic signed [31:0]   pe_psum       [PE_COLS][LANES],
  output logic signed [31:0]   glb_out_psum  [PE_COLS][LANES],
  // Bypass path (PE_PASS modes)
  input  logic signed [7:0]    bypass_in     [LANES],
  output logic signed [7:0]    bypass_out    [LANES],
  input  logic                 bypass_en
);
```

**Test Plan**:
```
☐ Test 5.1.1: RIN multicast
   Set glb_in_data[bank0] = pattern_A.
   Verify: pe_act[row0] = pe_act[row1] = pe_act[row2] = pattern_A.
   (Same activation for all 3 rows = kh parallelism.)

☐ Test 5.1.2: RWT per-column weight (★ KEY TEST)
   Set 4 DIFFERENT weight patterns for 4 columns.
   Verify: pe_wgt[row][col=0] ≠ pe_wgt[row][col=1] ≠ ... ≠ pe_wgt[row][col=3].
   Each column receives its own weight = its own cout.
   Pass: 4 independent weight streams.

☐ Test 5.1.3: Bypass path
   Set bypass_en=1, feed bypass_in. Verify: bypass_out = bypass_in.
   PE data paths should be isolated.
   Pass: Exact passthrough.

☐ Test 5.1.4: Mode switching
   Switch from PE_RS3 → PE_PASS → PE_DW3 → verify correct routing each time.
   Pass: No state leakage between modes.
```

---

## ☐ 5.2 `window_gen.sv` — Sliding Window Register

**File**: `rtl/data_movement/window_gen.sv`

| Thuộc tính | Mô tả |
|---|---|
| **Chức năng** | Shift register producing K tap outputs. K=1,3,5,7 configurable. |
| **V4** | 7 × 20 = 140 registers (vs 7×32=224 in V3, smaller!) |

**Ports**:
```systemverilog
module window_gen #(
  parameter LANES = 20,
  parameter K_MAX = 7
)(
  input  logic                 clk, rst_n,
  input  logic [3:0]           cfg_k,           // Active kernel width: 1,3,5,7
  input  logic                 shift_in_valid,
  input  logic signed [7:0]    shift_in [LANES],
  output logic signed [7:0]    taps [K_MAX][LANES],  // Up to 7 taps
  output logic                 taps_valid,
  input  logic                 flush
);
```

**Test Plan**:
```
☐ Test 5.2.1: K=3 (Conv 3×3)
   Feed 3 columns → verify taps[0..2] contain correct sliding data.
☐ Test 5.2.2: K=5 (MaxPool 5×5)
   Feed 5 columns → verify taps[0..4].
☐ Test 5.2.3: K=7 (DW 7×7)
   Feed 7 columns → verify taps[0..6].
☐ Test 5.2.4: K=1 (Conv 1×1)
   Feed 1 column → tap[0] = input (bypass shift).
☐ Test 5.2.5: Flush + refill
   Flush mid-stream → verify all taps cleared → refill → correct data.
```

---

## ☐ 5.3 `swizzle_engine.sv` — Layout Transform

**File**: `rtl/data_movement/swizzle_engine.sv`

| Thuộc tính | Mô tả |
|---|---|
| **Chức năng** | SWZ_NORMAL (identity), SWZ_UPSAMPLE2X (2× nearest), SWZ_CONCAT (channel join) |

**Test Plan**:
```
☐ Test 5.3.1: SWZ_UPSAMPLE2X
   Input 10×10 → output 20×20. Each pixel duplicated to 2×2 block.
   Compare vs golden P6 (UPSAMPLE_NEAREST).
   Pass: 100% bit-exact (verified = 100.00%).

☐ Test 5.3.2: SWZ_CONCAT
   2 input tensors with DIFFERENT scales → domain-aligned concat.
   Compare vs golden P5 (CONCAT).
   Pass: 100% bit-exact (verified = 100.00%).

☐ Test 5.3.3: SWZ_NORMAL
   Input = output (identity).
   Pass: Exact passthrough.
```

---
---

# STAGE 6: CONTROL
**Mục tiêu**: FSM logic điều khiển datapath đúng sequence, đúng timing.

---

## ☐ 6.1 `tile_fsm.sv` — Phase-Level Controller

**File**: `rtl/control/tile_fsm.sv`

| Thuộc tính | Mô tả |
|---|---|
| **Chức năng** | "KHI NÀO làm gì": Phase sequencing cho tile execution |
| **States** | 10: IDLE → LOAD_DESC → PREFILL_WT → PREFILL_IN → COMPUTE → PE_DRAIN → PPU_RUN → SWIZZLE → WRITEBACK → DONE |

**Test Plan**:
```
☐ Test 6.1.1: Normal Conv tile flow
   IDLE → LOAD_DESC → PREFILL_WT → PREFILL_IN → COMPUTE → PE_DRAIN → PPU_RUN → WRITEBACK → DONE
   Verify: correct state transitions + handshake signals.

☐ Test 6.1.2: DW7x7 multipass loop
   COMPUTE → PE_DRAIN → COMPUTE → PE_DRAIN → COMPUTE → PE_DRAIN → PPU_RUN → DONE
   (3 passes, PPU only on last pass.)
   Verify: num_k_pass=3 → loops 3 times.

☐ Test 6.1.3: PE_PASS mode (no compute)
   IDLE → LOAD_DESC → SWIZZLE → WRITEBACK → DONE
   (Skip PREFILL_WT, COMPUTE, PPU entirely.)

☐ Test 6.1.4: Barrier wait
   If tile_desc.barrier_wait=1: FSM waits at IDLE until barrier_grant.
   Verify: doesn't proceed without grant.
```

---

## ☐ 6.2 `shadow_reg_file.sv` — Config Latch

**File**: `rtl/control/shadow_reg_file.sv`

| Thuộc tính | Mô tả |
|---|---|
| **Chức năng** | Latch descriptor → stable config during compute |

**Test**: Latch config → verify outputs stable → latch new config → verify update.

---

## ☐ 6.3 `compute_sequencer.sv` — Cycle-Level Iteration

**File**: `rtl/control/compute_sequencer.sv`

| Thuộc tính | Mô tả |
|---|---|
| **Chức năng** | "CÁI GÌ mỗi cycle": (h, w, cin, kw, cout_group) inner loops |
| **★ CRITICAL** | cout_group steps by 4 (4 PE columns parallel) |
| **★ V4** | Wblk = W_out / LANES = W_out / 20 (exact division!) |

**Ports**:
```systemverilog
module compute_sequencer (
  input  logic                clk, rst_n,
  input  logic                seq_start,
  output logic                seq_done,
  // Config
  input  pe_mode_t            cfg_pe_mode,
  input  logic [9:0]          cfg_cin, cfg_cout, cfg_hout, cfg_wout,
  input  logic [3:0]          cfg_kh, cfg_kw,
  input  logic [2:0]          cfg_stride,
  // Iteration outputs (drive addr_gens + PE control)
  output logic [9:0]          iter_h, iter_w, iter_cin, iter_cout_group,
  output logic [3:0]          iter_kw, iter_kh_row,
  output logic                pe_enable, pe_clear_acc, pe_acc_valid,
  output logic                ppu_trigger,
  output logic [9:0]          ppu_cout_base
);
```

**Loop structure per mode**:
```
PE_RS3 (Conv 3×3):
  for h_out = 0..Hout-1:
    for wblk = 0..ceil(Wout/20)-1:        // ★ LANES=20 → exact divisions
      for cout_group = 0..ceil(Cout/4)-1:  // ★ 4 columns parallel
        pe_clear_acc
        for cin = 0..Cin-1:
          for kw = 0..2:
            pe_enable (feed data)
        pe_acc_valid → ppu_trigger

PE_OS1 (Conv 1×1):
  for h_out:
    for wblk:
      for cout_group:
        pe_clear_acc
        for cin:           // No kw loop (kw=0 only)
          pe_enable
        ppu_trigger

PE_DW3 (DW 3×3):
  for h_out:
    for wblk:
      for ch_group = 0..ceil(C/4)-1:    // 4 channels per iteration
        pe_clear_acc
        for kw = 0..2:
          pe_enable
        ppu_trigger (per-channel params)
```

**Test Plan**:
```
☐ Test 6.3.1: PE_RS3 iteration count
   Config: L0 (Cin=3, Cout=16, Hout=320, Wout=320, kw=3).
   Expected iterations: 320 × 16 × 4 × 3 × 3 = 184,320 PE cycles.
   Cout_groups = 16/4 = 4. Wblks = 320/20 = 16.
   Verify: seq_done asserts after exact cycle count.

☐ Test 6.3.2: PE_OS1 iteration count
   Config: QC2f cv1 (Cin=32, Cout=64, H=160, W=160).
   Wblks = 160/20 = 8 (exact!). Cout_groups = 64/4 = 16.
   PE cycles = 160 × 8 × 16 × 32 = 6,553,600.
   Verify: exact count.

☐ Test 6.3.3: PE_DW3 iteration count
   Config: SCDown (C=128, H=40, W=40, kw=3).
   Ch_groups = 128/4 = 32. Wblks = 40/20 = 2 (exact!).
   PE cycles = 40 × 2 × 32 × 3 = 7,680.

☐ Test 6.3.4: ppu_trigger timing
   Verify: ppu_trigger fires exactly once per cout_group completion.
   Verify: ppu_cout_base = cout_group × 4.

☐ Test 6.3.5: ★ LANES=20 exact division
   W_out=320: 320/20 = 16 blocks (exact, no padding waste).
   W_out=160: 160/20 = 8 blocks (exact).
   W_out=80:  80/20  = 4 blocks (exact).
   W_out=40:  40/20  = 2 blocks (exact).
   W_out=20:  20/20  = 1 block  (exact).
   Verify: NO partial blocks for ANY YOLOv10n layer.
```

---
---

# STAGE 7: SUBCLUSTER INTEGRATION
**Mục tiêu**: Wire-up all 24 instances → 1 working subcluster.

---

## ☐ 7.1 `pe_cluster_v4.sv` — 3×4×20 PE Array

**File**: `rtl/compute/pe_cluster_v4.sv`

| Thuộc tính | Mô tả |
|---|---|
| **Chức năng** | ★ 3 rows × 4 columns × 20 lanes = 240 MACs. Per-col weight. |
| **★ Core V4** | Mỗi column tính KHÁC output channel → 4× throughput |

**Test Plan**:
```
☐ Test 7.1.1: 4 independent cout outputs
   Feed SAME activation but 4 DIFFERENT weights to 4 columns.
   Verify: 4 DIFFERENT col_psum outputs = 4 different cout results.
   ★ This is THE key test for V4 architecture.

☐ Test 7.1.2: DW mode — 4 independent channels
   In PE_DW3/DW7: 4 columns = 4 DIFFERENT input channels.
   Verify: independent channel processing.

☐ Test 7.1.3: MaxPool bypass
   In PE_MP5 mode: PE cluster bypassed, comparator_tree active.
   Verify: maxpool output, PE psum ignored.
```

---

## ☐ 7.2 `subcluster_datapath.sv` — ★ COMPLETE INTEGRATION

**File**: `rtl/core/subcluster_datapath.sv`

| Thuộc tính | Mô tả |
|---|---|
| **Chức năng** | Wire-up ALL modules: control + memory + addr_gen + router + PE + PPU |
| **Instances** | 24+ sub-module instances per subcluster |
| **★ MOST COMPLEX** | This is where all pieces come together |

**Internal module instances**:
```
CONTROL (3):     tile_fsm, shadow_reg_file, compute_sequencer
MEMORY (10):     glb_input_bank_db ×3, glb_weight_bank ×3,
                 glb_output_bank ×4, metadata_ram
ADDR_GEN (3):    addr_gen_input, addr_gen_weight, addr_gen_output
DATA_MOVE (3):   router_cluster_v2, window_gen, swizzle_engine
COMPUTE (1+):    pe_cluster_v4 (contains 12 pe_units + column_reduce + comp_tree)
POST_PROC (4):   ppu ×4
TOTAL: ~24 instances
```

**Wiring verification checklist**:
```
☐ tile_fsm outputs → shadow_reg_file latch_en
☐ shadow_reg_file outputs → compute_sequencer config inputs
☐ compute_sequencer → addr_gen_input (iter_h, iter_w, iter_cin, iter_kh_row)
☐ compute_sequencer → addr_gen_weight (iter_cin, iter_cout_group, iter_kw, iter_kh_row)
☐ compute_sequencer → addr_gen_output (iter_h, iter_wblk, iter_cout_group)
☐ addr_gen_input → glb_input_bank_db[bank_id] read
☐ addr_gen_weight → glb_weight_bank[wgt_bank_id] read (★ 4 addresses per column)
☐ glb_input_bank_db → router_cluster_v2 (RIN)
☐ glb_weight_bank → router_cluster_v2 (RWT, ★ per-column)
☐ router_cluster_v2 → window_gen (activation taps)
☐ router_cluster_v2 → pe_cluster_v4 (weights, ★ per-column)
☐ window_gen → pe_cluster_v4 (activation taps)
☐ pe_cluster_v4 col_psum → ppu[0..3] (★ 4 parallel PPUs)
☐ compute_sequencer ppu_trigger → ppu[0..3]
☐ ppu[0..3] act_out → glb_output_bank[0..3] ACT write
☐ comparator_tree maxpool_out → glb_output_bank (PE_MP5 mode bypass)
☐ bypass path: router bypass → swizzle_engine → glb_output_bank
```

---
---

# STAGE 8: PRIMITIVE VERIFICATION — 7 PE_MODES
**Mục tiêu**: CÙNG 1 subcluster chạy đúng MỌI primitive qua descriptor config.
**★ Gate quan trọng: chứng minh descriptor-driven execution hoạt động.**

---

## ☐ 8.1 Test PE_RS3 (P0: Conv 3×3) — L0 Golden Vectors

```
Config:  pe_mode=PE_RS3, Cin=3, Cout=16, stride=2, kh=3, kw=3, act=RELU
Input:   X_int8[3, 8, 64] (8 rows × 64 cols × 3 channels, tile from L0)
Weight:  W_int8[16, 3, 3, 3] + bias[16] + m_int[16] + shift[16]
Output:  Y_int8[16, 4, 32] (stride=2 halves spatial)

Verify:
  ☐ Pre-fill GLB weight banks via DMA interface
  ☐ Pre-fill GLB input banks (page B) via DMA interface
  ☐ Page swap → tile_fsm start
  ☐ tile_fsm: PREFILL_WT → PREFILL_IN → COMPUTE → PE_DRAIN → PPU_RUN → DONE
  ☐ Read GLB output banks (ACT namespace)
  ☐ Compare vs golden Python L0 output tile

Pass: ≥99.99% bit-exact (verified = 99.99% in SW)
```

## ☐ 8.2 Test PE_OS1 (P1: Conv 1×1) — QC2f cv1 Vectors

```
Config:  pe_mode=PE_OS1, Cin=32, Cout=64, stride=1, kh=1, kw=1, act=RELU
Pass: ≥99% bit-exact
```

## ☐ 8.3 Test PE_DW3 (P2: DW Conv 3×3) — SCDown cv2 Vectors

```
Config:  pe_mode=PE_DW3, C=128, stride=2, kh=3, kw=3, act=RELU
★ 4 columns = 4 KHÁC channels (not cout)
★ Per-channel bias/m_int/shift
Pass: ≥99.9% bit-exact
```

## ☐ 8.4 Test PE_MP5 (P3: MaxPool 5×5) — SPPF Vectors

```
Config:  pe_mode=PE_MP5, C=128, K=5, stride=1, pad=2, NO PPU
★ PE cluster BYPASSED. comparator_tree active.
★ Output scale/zp = input scale/zp (unchanged).
Pass: ≥99.9% bit-exact
```

## ☐ 8.5 Test PE_PASS + UPSAMPLE (P6) — L11 Vectors

```
Config:  pe_mode=PE_PASS, swizzle=SWZ_UPSAMPLE2X
★ PE + PPU BYPASSED. swizzle_engine only.
★ 20×20 input → 40×40 output (address remap)
Pass: 100% bit-exact (verified = 100.00%)
```

## ☐ 8.6 Test PE_PASS + CONCAT (P5) — L12 Vectors

```
Config:  pe_mode=PE_PASS, swizzle=SWZ_CONCAT
★ Domain alignment: requant_to_common if scales differ.
★ Barrier wait (skip connection F6 ready).
Pass: 100% bit-exact (verified = 100.00%)
```

## ☐ 8.7 Test PE_DW7 Multipass (P8: DW 7×7) — L22 Vectors

```
Config:  pe_mode=PE_DW7, C=256, K=7, stride=1, pad=3, num_k_pass=3
★ 3 passes: Pass1→PSUM, Pass2→PSUM, Pass3→PPU→INT8
★ PSUM namespace in glb_output_bank holds intermediate results.
Pass: ≥99.9% bit-exact (verified = 99.96%)
```

**STAGE 8 SIGN-OFF**: Cùng 1 HW xử lý đúng 7 pe_modes. Descriptor-driven execution proven.

---
---

# STAGE 9: SUPERCLUSTER LEVEL
**Mục tiêu**: 4 subclusters + Triple-RUNNING + DMA hoạt động.

---

## ☐ 9.1 `local_arbiter_v2.sv` — ★ Triple-RUNNING Scheduler

**File**: `rtl/system/local_arbiter_v2.sv`

| Thuộc tính | Mô tả |
|---|---|
| **Chức năng** | ★ V4: 4-sub Triple-RUNNING: 3 compute + 1 fill/drain |
| **Rotation** | Sub-0,1,2 compute; Sub-3 fill+drain → rotate when tile done |

**Test Plan**:
```
☐ Test 9.1.1: Normal rotation
   Sub-3 finishes fill → Sub-0 finishes compute → rotate.
   Verify: roles shift correctly, 3 always computing.

☐ Test 9.1.2: DMA grant arbitration
   When Sub-X in FILL role: gets DMA read grant.
   When Sub-X in DRAIN role: gets DMA write grant.
   Verify: no grant conflict.

☐ Test 9.1.3: Stall handling
   All 3 COMPUTE subs done but FILL not ready → stall gracefully.
   Verify: no tile dropped.
```

---

## ☐ 9.2 `tensor_dma_v2.sv` — Dual-Channel AXI4 DMA

**File**: `rtl/system/tensor_dma_v2.sv`

**Test Plan**:
```
☐ Test 9.2.1: Burst read DDR → GLB (fill path)
☐ Test 9.2.2: Burst write GLB → DDR (drain path)
☐ Test 9.2.3: Concurrent read + write (★ V4 dual-channel)
☐ Test 9.2.4: Address alignment (AXI4 4KB boundary respect)
☐ Test 9.2.5: Data integrity (write → read back → compare)
```

---

## ☐ 9.3 `supercluster_wrapper.sv`

**Test Plan**:
```
☐ Test 9.3.1: Single tile execution
   Feed 1 tile_desc → 1 subcluster processes → output correct.

☐ Test 9.3.2: Multi-tile pipeline
   Feed 4 tiles → Triple-RUNNING: 3 computing + 1 loading at all times.
   Verify: all 4 tiles produce correct output.

☐ Test 9.3.3: Layer boundary (multiple tiles → 1 layer)
   L0 = multiple tiles → all processed → layer output assembled.
```

---
---

# STAGE 10: SYSTEM INTEGRATION
**Mục tiêu**: 4 SuperClusters + Controller + DDR3 interface.

---

## ☐ 10.1 `global_scheduler.sv` — Tile Dispatch

**Test**: Dispatch tiles round-robin to 4 SCs. Verify: all tiles processed.

## ☐ 10.2 `desc_fetch_engine.sv` — Descriptor Parser

**Test**: Load descriptor blob from DDR → parse NET/LAYER/TILE structs correctly.

## ☐ 10.3 `barrier_manager.sv` — Skip Connection Sync

**Test Plan**:
```
☐ Test 10.3.1: barrier_0 (L6→L12)
   L6 signals barrier_0 → L12 waits → grant → proceed.
☐ Test 10.3.2: All 4 barriers
   barrier_0, barrier_1, barrier_2, barrier_3 all work independently.
☐ Test 10.3.3: Out-of-order signal
   L4 barrier_1 signals before L15 waits → grant immediately when L15 reaches.
```

## ☐ 10.4 `csr_register_bank.sv` — CPU Interface

**Test**: AXI-Lite write CSR_CTRL.start → read CSR_STATUS.done after completion.

## ☐ 10.5 `accel_top.sv` — System Top-Level

**Test**: CPU start → descriptors fetched → tiles dispatched → computed → IRQ asserted.

---
---

# STAGE 11: BLOCK VERIFICATION — YOLOv10n Blocks
**Mục tiêu**: Mỗi block type đúng = chuỗi descriptors đúng trên CÙNG HW.

---

## ☐ 11.1 Conv Block (L0, L1, L3, L17)

```
Descriptors: 1 (PE_RS3 with ReLU)
Test L0:  X[3,640,640] → Y[16,320,320]. Golden: ≥99.99%.
Test L17: X[64,80,80]  → Y[64,40,40].  Golden: ≥99.99%.
```

## ☐ 11.2 QC2f Block (L2, L4, L6, L8, L13, L16, L19)

```
Descriptors: 5 (OS1 → RS3 → RS3 → CONCAT → OS1)
Test L2: X[32,160,160] → Y[32,160,160]. Golden: ≥97%.
Test L8: X[256,20,20]  → Y[256,20,20].  Golden: ≥99.2%.
```

## ☐ 11.3 SCDown Block (L5, L7, L20)

```
Descriptors: 2 (OS1 → DW3)
Test L5: X[64,80,80] → Y[128,40,40]. Golden: ≥99.9%.
```

## ☐ 11.4 SPPF Block (L9)

```
Descriptors: 6 (OS1 → MP5 → MP5 → MP5 → CONCAT → OS1)
Test L9: X[256,20,20] → Y[256,20,20]. Golden: ≥99.93%.
```

## ☐ 11.5 QConcat (L12, L15, L18, L21)

```
Descriptors: 1 (PE_PASS + CONCAT + barrier)
Test L12: F11[256,40,40] + F6[128,40,40] → Y[384,40,40]. Golden: 100%.
```

## ☐ 11.6 Upsample (L11, L14)

```
Descriptors: 1 (PE_PASS + UPSAMPLE2X)
Test L11: X[256,20,20] → Y[256,40,40]. Golden: 100%.
```

## ☐ 11.7 QC2fCIB (L22)

```
Descriptors: ~9 (OS1→DW3→OS1→DW7(3-pass)→OS1→DW3→ADD→CONCAT→OS1)
Test L22: X[384,20,20] → Y[256,20,20]. Golden: ≥99.96%.
```

## ☐ 11.8 QPSA (L10) — Optional/Deferred

```
Descriptors: ~14 (OS1 projections + GEMM + softmax + DW3 + OS1 FFN + ADD)
Test L10: X[256,20,20] → Y[256,20,20]. Golden: ≥83.5%.
★ Defer: implement after L0-L22 (except L10) all pass.
```

---
---

# STAGE 12: LAYER-BY-LAYER VERIFICATION
**Mục tiêu**: Mỗi layer L0-L22 output khớp golden Python.

```
☐ 12.1  L0  Conv:      ≥99.99%   [3,640,640]   → [16,320,320]
☐ 12.2  L1  Conv:      ≥99.96%   [16,320,320]  → [32,160,160]
☐ 12.3  L2  QC2f:      ≥99.09%   [32,160,160]  → [32,160,160]
☐ 12.4  L3  Conv:      ≥99.98%   [32,160,160]  → [64,80,80]
☐ 12.5  L4  QC2f:      ≥96.52%   [64,80,80]    → [64,80,80]
☐ 12.6  L5  SCDown:    ≥99.90%   [64,80,80]    → [128,40,40]
☐ 12.7  L6  QC2f:      ≥94.46%   [128,40,40]   → [128,40,40]
☐ 12.8  L7  SCDown:    ≥99.92%   [128,40,40]   → [256,20,20]
☐ 12.9  L8  QC2f:      ≥99.20%   [256,20,20]   → [256,20,20]
☐ 12.10 L9  SPPF:      ≥99.93%   [256,20,20]   → [256,20,20]
☐ 12.11 L10 QPSA:      ≥83.52%   [256,20,20]   → [256,20,20]   (deferred)
☐ 12.12 L11 Upsample:  100.00%   [256,20,20]   → [256,40,40]
☐ 12.13 L12 QConcat:   100.00%   +F6           → [384,40,40]   barrier_0
☐ 12.14 L13 QC2f:      ≥98.83%   [384,40,40]   → [128,40,40]
☐ 12.15 L14 Upsample:  100.00%   [128,40,40]   → [128,80,80]
☐ 12.16 L15 QConcat:   100.00%   +F4           → [192,80,80]   barrier_1
☐ 12.17 L16 QC2f:      ≥98.83%   [192,80,80]   → [64,80,80]    ★ P3 output
☐ 12.18 L17 Conv:      ≥99.99%   [64,80,80]    → [64,40,40]
☐ 12.19 L18 QConcat:   100.00%   +F13          → [192,40,40]   barrier_2
☐ 12.20 L19 QC2f:      ≥98.83%   [192,40,40]   → [128,40,40]   ★ P4 output
☐ 12.21 L20 SCDown:    ≥99.90%   [128,40,40]   → [128,20,20]
☐ 12.22 L21 QConcat:   100.00%   +F8           → [384,20,20]   barrier_3
☐ 12.23 L22 QC2fCIB:   ≥99.96%   [384,20,20]   → [256,20,20]   ★ P5 output
```

**Pass criteria**: Mỗi layer ≥ baseline % từ SW golden verification.
**Skip connections verified**: F4(L4→L15), F6(L6→L12), F8(L8→L21), F13(L13→L18).

---
---

# STAGE 13: FULL INFERENCE
**Mục tiêu**: L0→L22 chain hoàn chỉnh, P3/P4/P5 output khớp golden.

---

## ☐ 13.1 Generate Descriptor Blob

```
Python script: generate_descriptors.py
Output: binary blob chứa:
  - 1 NET_DESC (64 bytes)
  - 23 LAYER_DESC (23 × 32 bytes)
  - ~60 TILE_DESC (~60 × 32 bytes)
Total: ~2.7 KB
```

## ☐ 13.2 Generate Weight Blob

```
Python script: generate_weights.py
Pack: weights + bias + m_int + shift cho L0-L22 → binary DDR image.
Total: ~5.2 MB (YOLOv10n INT8)
```

## ☐ 13.3 End-to-End Simulation

```
Flow:
  1. Load descriptors + weights + input image → DDR model (SystemVerilog)
  2. CPU writes CSR_START → accelerator begins
  3. Accelerator: desc_fetch → scheduler → 4 SCs → L0→L22 (auto via descriptors)
  4. Accelerator: IRQ assert when L22 last tile done
  5. Read P3/P4/P5 from DDR model

Compare:
  ☐ P3 [1, 64, 80, 80]:  ≥97% bit-exact vs golden
  ☐ P4 [1, 128, 40, 40]: ≥97% bit-exact vs golden
  ☐ P5 [1, 256, 20, 20]: ≥97% bit-exact vs golden
```

## ☐ 13.4 Detection Head Validation

```
Feed P3/P4/P5 vào CPU Qv10Detect head (Python float32):
  - Dequantize INT8 → float32
  - Detection head convolutions
  - Decode bounding boxes
  - NMS (Non-Maximum Suppression)

☐ mAP50 ≥ 0.92 (< 1% degradation from golden 0.9302)
☐ mAP50-95 ≥ 0.71
```

## ☐ 13.5 Multi-Image Validation

```
☐ Run 100 images from ALPR dataset
☐ Per-image: compare P3/P4/P5 vs golden
☐ Summary: average match %, max LSB diff distribution
☐ Final mAP50 ≥ 0.92
```

---
---

# STAGE 14: FPGA DEPLOYMENT
**Mục tiêu**: Chạy thực trên VC707, đo FPS thực tế.

---

## ☐ 14.1 Synthesis

```
Tool: Vivado 2023.2+
Target: XC7VX485T-2FFG1761
Clock constraint: 250 MHz (4.0ns period)

Resource check:
  ☐ DSP48E1 ≤ 1,920  (68.6% of 2,800)
  ☐ BRAM36K ≤ 544    (52.8% of 1,030)
  ☐ LUT6    ≤ 187K   (61.5% of 303,600)
  ☐ FF      ≤ 198K   (32.6% of 607,200)
```

## ☐ 14.2 Implementation (Place & Route)

```
☐ Timing closure @ 250 MHz (WNS ≥ 0)
☐ If timing fails: reduce to 220 MHz (still 170+ FPS)
☐ Power estimation: ≤ 15W
```

## ☐ 14.3 Bitstream & Board Test

```
☐ Generate bitstream (.bit)
☐ Program VC707 via JTAG
☐ CPU driver: load data via AXI-Lite/PCIe
☐ Run inference on real hardware
☐ Measure T_hw via CSR_PERF_CYCLES → calculate FPS
☐ Target: ≥ 180 FPS (T_hw ≤ 5.5ms)
☐ Compare output vs golden → FINAL VALIDATION
```

---
---

# TỔNG HỢP: FILE RTL CẦN XÂY DỰNG — FULL LIST

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  #  │ File                        │ Stage │ ★ Notes                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║     │ PACKAGES                     │       │                                ║
║  1  │ accel_pkg.sv                 │ 0     │ Types, params, enums           ║
║  2  │ desc_pkg.sv                  │ 0     │ Descriptor structs             ║
║  3  │ csr_pkg.sv                   │ 0     │ CSR address map                ║
║     │                              │       │                                ║
║     │ COMPUTE ATOMS                │       │                                ║
║  4  │ dsp_pair_int8.sv             │ 1     │ ★ 5-stage pipe, 2-MAC DSP     ║
║  5  │ pe_unit.sv                   │ 1     │ ★ LANES=20, 10 dsp_pairs      ║
║  6  │ column_reduce.sv             │ 1     │ 3→1 sum, 20 lanes             ║
║  7  │ comparator_tree.sv           │ 1     │ 25→1 max, 5-stage, 20 lanes   ║
║     │                              │       │                                ║
║     │ POST-PROCESSING              │       │                                ║
║  8  │ ppu.sv                       │ 2     │ ★★ CRITICAL: half-up, INT64   ║
║  9  │ silu_lut.sv                  │ 2     │ Optional, unused for ReLU      ║
║     │                              │       │                                ║
║     │ MEMORY                       │       │                                ║
║ 10  │ glb_input_bank_db.sv         │ 3     │ ★ Double-buffer (V4)          ║
║ 11  │ glb_weight_bank.sv           │ 3     │ ★ 4-read-port (per-col)      ║
║ 12  │ glb_output_bank.sv           │ 3     │ Dual namespace PSUM/ACT       ║
║ 13  │ metadata_ram.sv              │ 3     │ Slot validity                  ║
║     │                              │       │                                ║
║     │ ADDRESS GENERATION           │       │                                ║
║ 14  │ addr_gen_input.sv            │ 4     │ ★ Padding = zp_x              ║
║ 15  │ addr_gen_weight.sv           │ 4     │ ★★ 4-col addresses (V4 key)  ║
║ 16  │ addr_gen_output.sv           │ 4     │ bank = PE column               ║
║     │                              │       │                                ║
║     │ DATA MOVEMENT                │       │                                ║
║ 17  │ router_cluster_v2.sv         │ 5     │ ★★ Multicast + per-col wgt   ║
║ 18  │ window_gen.sv                │ 5     │ K=1,3,5,7 sliding window      ║
║ 19  │ swizzle_engine.sv            │ 5     │ Upsample/Concat/Normal        ║
║     │                              │       │                                ║
║     │ CONTROL                      │       │                                ║
║ 20  │ tile_fsm.sv                  │ 6     │ 10-state phase FSM             ║
║ 21  │ shadow_reg_file.sv           │ 6     │ Config latch                   ║
║ 22  │ compute_sequencer.sv         │ 6     │ ★ Cycle-level iteration       ║
║     │                              │       │                                ║
║     │ COMPUTE (CLUSTER)            │       │                                ║
║ 23  │ pe_cluster_v4.sv             │ 7     │ ★★ 3×4×20, per-col weight    ║
║     │                              │       │                                ║
║     │ INTEGRATION                  │       │                                ║
║ 24  │ subcluster_datapath.sv       │ 7     │ ★★★ Core: 24 instances wired ║
║     │                              │       │                                ║
║     │ SUPERCLUSTER                 │       │                                ║
║ 25  │ local_arbiter_v2.sv          │ 9     │ ★ Triple-RUNNING (V4)        ║
║ 26  │ tensor_dma_v2.sv             │ 9     │ ★ Dual-channel (V4)          ║
║ 27  │ tile_ingress_fifo.sv         │ 9     │ 8-deep tile FIFO              ║
║ 28  │ supercluster_wrapper.sv      │ 9     │ 4 subs + arbiter + DMA        ║
║     │                              │       │                                ║
║     │ SYSTEM                       │       │                                ║
║ 29  │ global_scheduler.sv          │ 10    │ Tile dispatch → 4 SCs         ║
║ 30  │ desc_fetch_engine.sv         │ 10    │ DDR → descriptor parser        ║
║ 31  │ barrier_manager.sv           │ 10    │ 4-point skip sync              ║
║ 32  │ csr_register_bank.sv         │ 10    │ AXI-Lite CPU interface         ║
║ 33  │ axi_lite_slave.sv            │ 10    │ AXI-Lite protocol              ║
║ 34  │ axi4_master_mux.sv           │ 10    │ 4 SC → 1 DDR arbiter          ║
║ 35  │ axi_interconnect.sv          │ 10    │ AXI routing fabric             ║
║ 36  │ accel_top.sv                 │ 10    │ ★ System top-level            ║
║     │                              │       │                                ║
║     │ CLOCK/RESET                  │       │                                ║
║ 37  │ clk_wiz_250.sv              │ 10    │ MMCM: 200→250 MHz             ║
║ 38  │ reset_sync.sv               │ 10    │ Async reset synchronizer       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  TOTAL: 38 files │ 15 stages │ Target: ~192 FPS @ 250 MHz, 65% resources  ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---
---

# MILESTONE SUMMARY

```
╔══════════════════════════════════════════════════════════════════════════╗
║  Milestone │ Stage  │ Gate Test                      │ Pass Criteria    ║
╠══════════════════════════════════════════════════════════════════════════╣
║  M1        │ 0-1    │ Compute atoms bit-exact        │ 0 errors         ║
║  M2        │ 2      │ PPU bit-exact vs Golden Python │ 100% match       ║
║  M3        │ 3-6    │ Memory + Addr + Control compile│ 0 compile errors ║
║  M4        │ 7      │ Subcluster integration clean   │ All wires correct║
║  M5        │ 8      │ 7 pe_modes PASS golden vectors │ ≥ baseline %     ║
║  M6        │ 9-10   │ System integration IRQ works   │ CPU start→done   ║
║  M7        │ 11     │ All block types PASS golden    │ ≥ baseline %     ║
║  M8        │ 12     │ All 23 layers PASS golden      │ ≥ baseline %     ║
║  M9        │ 13     │ Full inference P3/P4/P5 correct│ mAP50 ≥ 0.92    ║
║  M10       │ 14     │ VC707 board running            │ ≥ 180 FPS        ║
╚══════════════════════════════════════════════════════════════════════════╝

★ Mỗi milestone PHẢI pass trước khi tiến hành milestone kế tiếp.
★ Nếu fail: debug tại level đó, KHÔNG skip lên level cao hơn.
★ Nguyên tắc: ĐÚng TRƯỚC, nhanh SAU.
```
