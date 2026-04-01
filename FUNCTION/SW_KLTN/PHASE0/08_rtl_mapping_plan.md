# 08 – RTL Mapping Plan (Freeze Spec)
## qYOLOv10n INT8 Accelerator – Primitive/Layer → RTL Module Mapping

> **Trạng thái**: FREEZE | **Version**: v1.0 | **Ngày**: 2026-03-16

---

## 1. Mục Đích

Chốt đường đi từ primitive/layer xuống RTL module. Sau khi Golden Python pass, file này là bản đồ implement RTL theo thứ tự dependency.

---

## 2. RTL Module Hierarchy

```
accel_top.sv
  ├── desc_fetch_engine.sv        ← Fetch & parse NET/LAYER/TILE desc
  ├── barrier_manager.sv          ← Sync producer→consumer skip
  ├── tile_fsm.sv                 ← Tile loop control
  ├── subcluster_wrapper.sv       ← Block-level composition
  │     ├── pe_cluster.sv         ← Dense conv (RS_DENSE_3x3, OS_1x1)
  │     │     ├── window_gen.sv
  │     │     ├── pe_lane_mac.sv
  │     │     └── column_reduce.sv
  │     ├── pe_cluster_dw.sv      ← Depthwise conv (DW_3x3, DW_7x7)
  │     │     ├── window_gen.sv   (shared)
  │     │     └── pe_lane_mac_dw.sv
  │     ├── pool_engine.sv        ← MaxPool (MAXPOOL_5x5)
  │     │     ├── window_gen.sv   (shared)
  │     │     └── max_compare_tree.sv
  │     ├── router_cluster.sv     ← CONCAT, MOVE, routing
  │     ├── swizzle_engine.sv     ← UPSAMPLE_NEAREST, transpose
  │     └── tensor_post_engine.sv ← Address remapping DMA
  ├── ppu_lite.sv                 ← Bias + Requant + Act + Clamp
  ├── glb_input_bank.sv (×3)
  ├── glb_output_bank.sv (×4)
  ├── glb_weight_bank.sv
  ├── psum_buffer.sv
  ├── addr_gen_input.sv
  ├── addr_gen_weight.sv
  ├── addr_gen_output.sv
  └── row_slot_manager.sv
```

---

## 3. Primitive → RTL Module Mapping

| Primitive | RTL Module Chính | RTL Module Phụ |
|---|---|---|
| P0 RS_DENSE_3x3 | `pe_cluster.sv` | `window_gen`, `pe_lane_mac`, `column_reduce`, `ppu_lite` |
| P1 OS_1x1 | `pe_cluster.sv` (1×1 mode) | `pe_lane_mac`, `ppu_lite` |
| P2 DW_3x3 | `pe_cluster_dw.sv` | `window_gen`, `pe_lane_mac_dw`, `ppu_lite` |
| P3 MAXPOOL_5x5 | `pool_engine.sv` | `window_gen`, `max_compare_tree` |
| P4 MOVE | `tensor_post_engine.sv` | DMA controller |
| P5 CONCAT | `router_cluster.sv` | `ppu_lite` (mini, nếu domain align) |
| P6 UPSAMPLE_NEAREST | `swizzle_engine.sv` | `tensor_post_engine.sv` |
| P7 EWISE_ADD | `router_cluster.sv` | `ppu_lite` (add + requant) |
| P8 DW_7x7_MULTIPASS | `pe_cluster_dw.sv` (multipass mode) | `psum_buffer`, `ppu_lite` |
| P9 GEMM_ATTN_BASIC | `pe_cluster.sv` (GEMM mode) | `ppu_lite`, softmax LUT |

---

## 4. Đặc Tả Chi Tiết Từng RTL Module

### 4.1. window_gen.sv

```
Input:
  glb_input_banks[3]  ← 3-bank GLB input
  tile_params         ← h_out, w_blk, kernel_size, stride, padding

Output:
  window_data[K×K][Cin_chunk][LANES]  ← window data per cycle
  valid                               ← valid window data

Chức năng:
  - Sinh cửa sổ K×K pixel từ GLB theo bank model (h mod 3)
  - Xử lý edge padding (zeros) khi edge_tile flag set
  - Hỗ trợ kernel 1×1, 3×3, 5×5, 7×7 (partial cho multipass)
  - Output: LANES=16 columns đồng thời

Parameters: kernel_size ∈ {1,3,5}, stride ∈ {1,2}
Multipass param: kh_start, kh_end cho từng pass của DW_7x7
```

### 4.2. pe_lane_mac.sv (Dense mode)

```
Input:
  x_lane[LANES][Cin_chunk]   ← INT8 activation, 16 lanes × Cin
  w_col[Cin_chunk][Cout_chunk] ← INT8 weight, Cin × Cout
  psum_in[Cout_chunk][LANES]  ← INT32 partial sum carry-in

Output:
  psum_out[Cout_chunk][LANES]  ← INT32 updated PSUM

Operation:
  for lane in 0..LANES-1:
    for cout in 0..Cout_chunk-1:
      psum_out[cout][lane] = psum_in[cout][lane]
        + Σ_{cin} x_lane[lane][cin] × w_col[cin][cout]
  
  INT8 × INT8 → INT16 per multiply
  Σ Cin terms → INT32 accumulate
  
Clock budget: Cin_chunk term per cycle (systolic approach) OR
              All Cin in parallel (combinational MAC tree per lane)
```

### 4.3. pe_lane_mac_dw.sv (Depthwise mode)

```
Input:
  x_lane[LANES][1]      ← 1 channel per lane (depthwise)
  w_dw[1][1]            ← weight scalar per channel (broadcast)
  psum_dw_in[LANES]

Output:
  psum_dw_out[LANES]

Operation:
  for lane in 0..LANES-1:
    psum_dw_out[lane] = psum_dw_in[lane] + x_lane[lane][0] × w_dw[0][0]

  No cross-channel reduction.
  Per-channel weight loaded from glb_weight_bank with different offsets.
```

### 4.4. column_reduce.sv

```
Input:
  psum_partial[Cout][LANES]  ← partial sum từ current kernel position
  psum_acc[Cout][LANES]      ← accumulated PSUM từ previous Cin chunks

Output:
  psum_acc_new[Cout][LANES]  ← updated PSUM

Operation:
  psum_acc_new = psum_acc + psum_partial
  (Cin dimension reduce: cộng dồn qua các Cin chunks)

Triggered only when last_cin=0 or intermediate passes.
```

### 4.5. ppu_lite.sv

```
Input:
  psum[Cout][LANES]  ← INT32 sau last_pass
  bias[Cout]         ← INT32 bias values (từ POST_PROFILE)
  M_int[Cout]        ← INT32 fixed-point multiplier
  shift              ← Shift amount
  zp_out             ← Output zero point
  act_mode           ← 0=none, 1=SiLU_LUT
  clamp_min/max      ← -128/127

Output:
  y_int8[Cout][LANES]  ← INT8 final output

Pipeline stages:
  Stage 1: bias_add   → acc_biased = psum + bias[cout]  (INT32)
  Stage 2: multiply   → y_scaled = acc_biased × M_int[cout]  (INT64 intermediate)
  Stage 3: shift      → y_shifted = y_scaled >> shift  (INT32)
  Stage 4: offset     → y_off = y_shifted + zp_out  (INT32)
  Stage 5: act        → y_act = SiLU_LUT[y_off] if act_mode=1 else y_off
  Stage 6: clamp      → y_int8 = clamp(y_act, -128, 127)  (INT8)

SiLU LUT:
  - 256-entry ROM (or BRAM)
  - LUT[i] = SiLU value at index i (precomputed from software)
  - Address: y_off + 128 (shift to [0,255])
```

### 4.6. pool_engine.sv (MAXPOOL_5x5)

```
Input:
  window[5×5][LANES]  ← 25 INT8 values per lane từ window_gen_5x5

Output:
  max_val[LANES]  ← INT8 max per lane

Operation:
  max_val[lane] = max(window[0..24][lane])  ← 25-way INT8 max
  
Implementation: binary tree (5 levels), pipelined
  Level 0: 25 → 13 (one input passed through)
  Level 1: 13 → 7
  Level 2: 7  → 4
  Level 3: 4  → 2
  Level 4: 2  → 1 = max

No PPU, no requant. scale/zp metadata pass-through.
```

### 4.7. router_cluster.sv

```
Chức năng:
  1. Route GLB_INPUT → PE (standard conv path)
  2. Route HOLD_SKIP + GLB_INPUT → CONCAT output (QConcat path)
  3. Route GLB_OUTPUT → next layer GLB_INPUT (ping-pong)

For CONCAT mode:
  - Read A_channels từ GLB region A
  - Read B_channels từ HOLD_SKIP region
  - If domain_align_en: pass B through mini_ppu_requant
  - Write [A_channels, B_channels] interleaved to GLB_OUTPUT

mini_ppu_requant (inline in router):
  y = clamp(round((x-zp_in) * (scale_in/scale_out)) + zp_out, -128, 127)
  Implementation: fixed-point multiply-shift, same as ppu_lite but simpler
```

### 4.8. swizzle_engine.sv (UPSAMPLE_NEAREST)

```
Input:
  src_tensor[C][H][W]  ← INT8 từ GLB
  upsample_scale = 2

Output:
  dst_tensor[C][2H][2W]  ← Written to GLB_INPUT for next layer

Address pattern:
  for h in 0..H-1, w in 0..W-1:
    src_addr = addr_gen_input(h, w, ...)
    val = GLB.read(src_addr)
    GLB.write(addr_gen_output(2h,   2w,   ...), val)
    GLB.write(addr_gen_output(2h,   2w+1, ...), val)
    GLB.write(addr_gen_output(2h+1, 2w,   ...), val)
    GLB.write(addr_gen_output(2h+1, 2w+1, ...), val)

No arithmetic, no PPU. scale/zp unchanged.
```

### 4.9. addr_gen_input.sv

```
Input: h, x, cin, layer_params (stride, Q_in, Wblk_total, Cin)
Output: (bank_id[2], offset[32])

Logic:
  bank_id = h[1:0] % 3     (2-bit modulo 3)
  slot    = (h >> 2) % Q_in
  lane    = x[3:0]          (x mod 16)
  Wblk    = x >> 4          (x div 16)
  offset  = slot*(Wblk_total*Cin*16) + Wblk*(Cin*16) + cin*16 + lane
```

### 4.10. row_slot_manager.sv

```
Tham số:  K_eff, stride
Computed: Q_in = ceil((K_eff + 3*stride) / 3)

State:
  slot_reg[3]  ← current slot for each bank (3-entry array)
  
Operations:
  advance_slot(bank):       slot_reg[bank] = (slot_reg[bank] + 1) % Q_in
  get_slot(bank):           return slot_reg[bank]
  reset_all():              slot_reg = [0,0,0]
```

### 4.11. barrier_manager.sv

```
Registers:
  done_reg[23]      ← done_reg[i]=1 when layer i finishes
  hold_ready_reg[4] ← {F4_ready, F6_ready, F8_ready, F13_ready}

Outputs combinational:
  L12_start_en = done_reg[11] & hold_ready_reg[F6]
  L15_start_en = done_reg[14] & hold_ready_reg[F4]
  L18_start_en = done_reg[17] & hold_ready_reg[F13]
  L21_start_en = done_reg[20] & hold_ready_reg[F8]

Update rules:
  done_reg[i] ← set by tile_fsm when layer i last_tile completes
  hold_ready_reg[Fx] ← set when Fx producer layer last_tile completes
  hold_ready_reg[Fx] ← clear when Fx consumer layer starts

Timeout:
  Timeout counter per barrier: if stalled > N cycles → error_interrupt
```

### 4.12. tile_fsm.sv

```
State machine điều khiển tiling loop:

States:
  IDLE → FETCH_LAYER_DESC → FETCH_TILE_DESC → CHECK_BARRIER
       → EXECUTE_TILE → WAIT_TILE_DONE → NEXT_TILE → ...
       → LAYER_DONE → NEXT_LAYER → ... → INFERENCE_DONE

Key actions:
  FETCH_LAYER_DESC: desc_fetch_engine nhận LAYER_DESC[layer_idx]
  FETCH_TILE_DESC:  nhận TILE_DESC[tile_idx]
  CHECK_BARRIER:    nếu layer là QConcat → đợi barrier release
  EXECUTE_TILE:     dispatch tile sang pe_cluster/pool/router/swizzle
  LAYER_DONE:       set done_reg[layer_idx], update hold states
  NEXT_LAYER:       layer_idx++, tile_idx=0
```

---

## 5. Block → RTL Composition

| Block | RTL Composition |
|---|---|
| Conv (L0,1,3,17) | pe_cluster + addr_gen + ppu_lite |
| QC2f (L2,4...) | OS_1x1+RS_DENSE_3x3 sequences via pe_cluster, router concat nội bộ |
| SCDown (L5,7,20) | 2× pe_cluster_dw sequences + router_cluster CONCAT |
| SPPF (L9) | pe_cluster (OS_1x1) + pool_engine×3 + router_cluster (CONCAT) + pe_cluster |
| QPSA (L10) | pe_cluster (OS_1x1 Q/K/V proj) + pe_cluster (GEMM) + ppu_lite |
| Upsample (L11,14) | swizzle_engine |
| QConcat (L12,15,18,21) | router_cluster + barrier_manager + optional mini-PPU |
| QC2fCIB (L22) | pe_cluster + pe_cluster_dw (multipass) + ppu_lite + router_cluster |

---

## 6. RTL Implementation Order (Dependency)

```
Level 0 – Packages (no circuit):
  accel_pkg.sv      ← primitive IDs, constants, type definitions
  desc_pkg.sv       ← descriptor structs
  csr_pkg.sv        ← CSR register map

Level 1 – Memory primitives:
  glb_input_bank.sv
  glb_output_bank.sv
  glb_weight_bank.sv
  psum_buffer.sv

Level 2 – Address generation:
  addr_gen_input.sv
  addr_gen_weight.sv
  addr_gen_output.sv
  row_slot_manager.sv

Level 3 – Compute atoms:
  window_gen.sv       ← depends on addr_gen_input, glb_input_bank
  pe_lane_mac.sv      ← depends on nothing (pure combinational)
  pe_lane_mac_dw.sv
  column_reduce.sv
  max_compare_tree.sv

Level 4 – Post-processing:
  ppu_lite.sv         ← SiLU LUT, requant, clamp

Level 5 – Engines:
  pe_cluster.sv       ← window_gen + pe_lane_mac + column_reduce
  pe_cluster_dw.sv    ← window_gen + pe_lane_mac_dw (+ psum_buffer cho multipass)
  pool_engine.sv      ← window_gen + max_compare_tree
  tensor_post_engine.sv ← DMA + addr remapping

Level 6 – Data movement:
  router_cluster.sv   ← depends on glb banks, ppu_lite (mini)
  swizzle_engine.sv   ← depends on tensor_post_engine

Level 7 – Control:
  desc_fetch_engine.sv ← depends on desc_pkg, glb_weight_bank
  barrier_manager.sv   ← simple registers + combinational logic
  tile_fsm.sv          ← depends on all Level 6 engines

Level 8 – Top:
  subcluster_wrapper.sv ← composes Level 5+6 engines
  accel_top.sv          ← all modules + DMA + AXI interface
```

---

## 7. Verification Strategy

### RTL vs Golden Python

```
For each RTL module at Level 3+:
  1. Generate test vectors from Golden Python:
     python: input=test_vector, expected=golden_output
  2. Write SystemVerilog testbench:
     tb_pe_cluster.sv: drive same test_vector, capture output
  3. Simulate (VCS/Icarus): compare output vs expected
  4. Require: bit-exact match for all 100 random vectors

For top-level integration:
  tb_accel_top.sv: drive full model_forward
  Compare P3/P4/P5 output vs oracle_P3.npy, oracle_P4.npy, oracle_P5.npy
  Require: bit-exact match
```

### Golden Vector Format

```python
# Generate từ Python golden
test_vec = {
    "input": X_int8.tolist(),
    "scale_in": float(scale_in),
    "zp_in": int(zp_in),
    "weight": W_int8.tolist(),
    "bias": B_int32.tolist(),
    "expected_output": Y_int8.tolist(),
    "scale_out": float(scale_out),
    "zp_out": int(zp_out),
}
json.dump(test_vec, open("test_vec_layer0.json", "w"))

# Đọc từ SV testbench qua $readmemh hoặc DPI-C
```

---

## 8. RTL Packages Spec

### accel_pkg.sv (phải viết TRƯỚC tất cả RTL module)

```systemverilog
package accel_pkg;
  // Architecture constants
  parameter int LANES       = 16;
  parameter int INPUT_BANKS = 3;
  parameter int OUTPUT_BANKS = 4;
  parameter int PSUM_W      = 32;   // PSUM width bits
  parameter int ACT_W       = 8;    // Activation width bits
  parameter int WEIGHT_W    = 8;    // Weight width bits
  
  // Primitive IDs
  typedef enum logic [3:0] {
    P_RS_DENSE_3x3    = 4'd0,
    P_OS_1x1          = 4'd1,
    P_DW_3x3          = 4'd2,
    P_MAXPOOL_5x5     = 4'd3,
    P_MOVE            = 4'd4,
    P_CONCAT          = 4'd5,
    P_UPSAMPLE_NEAREST = 4'd6,
    P_EWISE_ADD       = 4'd7,
    P_DW_7x7_MULTIPASS = 4'd8,
    P_GEMM_ATTN_BASIC = 4'd9
  } primitive_id_t;

  // Activation mode
  typedef enum logic [1:0] {
    ACT_NONE = 2'd0,
    ACT_SILU = 2'd1,
    ACT_RELU = 2'd2
  } act_mode_t;

  // Partition mode
  typedef enum logic [1:0] {
    PART_HW   = 2'd0,
    PART_COUT = 2'd1,
    PART_CIN  = 2'd2
  } partition_mode_t;

  // Tile flags
  typedef struct packed {
    logic first_tile;
    logic edge_tile_h;
    logic edge_tile_w;
    logic hold_skip;
    logic need_swizzle;
    logic psum_carry_in;
    logic [9:0] reserved;
  } tile_flags_t;

  // Last flags
  typedef struct packed {
    logic last_cin;
    logic last_kernel;
    logic last_reduce;
    logic last_pass_kernel;
    logic [11:0] reserved;
  } last_flags_t;

endpackage
```

---

## 9. Sign-off Checklist

```
PACKAGES:
☐ accel_pkg.sv: primitive_id_t, act_mode_t, tile_flags_t, last_flags_t defined
☐ desc_pkg.sv: NET_DESC, LAYER_DESC, TILE_DESC, ROUTER_PROFILE, POST_PROFILE structs
☐ csr_pkg.sv: CSR map defined

RTL MODULES (theo Level):
Level 1:
☐ glb_input_bank.sv: 3-bank, h%3 addressing
☐ glb_output_bank.sv: 4-bank, out_row%4 addressing
☐ psum_buffer.sv: INT32 store/load per tile

Level 2:
☐ addr_gen_input.sv: bank+slot+Wblk+lane formula correct
☐ row_slot_manager.sv: Q_in compute correct

Level 3:
☐ window_gen.sv: 1×1/3×3/5×5/7×7 all tested
☐ pe_lane_mac.sv: INT8×INT8→INT32, 16 lanes parallel
☐ pe_lane_mac_dw.sv: per-channel independent

Level 4:
☐ ppu_lite.sv: bias→multiply→shift→act_lut→clamp pipeline correct
☐ SiLU LUT loaded correctly (256 entries)

Level 5+:
☐ pe_cluster.sv: golden vector match (conv3x3 s1, s2, conv1x1)
☐ pe_cluster_dw.sv: dw3x3 and dw7x7 multipass match
☐ pool_engine.sv: maxpool5x5 result match
☐ router_cluster.sv: concat output bit-exact with golden

Level 7:
☐ barrier_manager.sv: 4 barriers correct
☐ tile_fsm.sv: fsm steps correct

Integration:
☐ accel_top.sv: P3/P4/P5 bit-exact với oracle_P3/P4/P5.npy
```

*Sau khi 8 file Phase 0 được sign-off, bắt đầu viết accel_pkg.sv → Level 1 → ... theo thứ tự.*
