# 06 – Execution Semantics (Freeze Spec)
## qYOLOv10n INT8 Accelerator – Tile/Layer Execution Semantics

> **Trạng thái**: FREEZE | **Version**: v1.0 | **Ngày**: 2026-03-16

---

## 1. Mục Đích

Chốt ý nghĩa chính xác của last_pass, PSUM/ACT namespace, HOLD_SKIP, barrier, và UPSAMPLE/CONCAT/MOVE path. RTL `tile_fsm.sv` và `barrier_manager.sv` phải tuân thủ file này.

---

## 2. Last Pass Semantics

### 2.1. Định Nghĩa

```
last_pass = last_cin AND last_kernel AND last_reduce

last_cin    : Cin chunk hiện tại là chunk cuối cùng để tích lũy PSUM
              → Reduction theo chiều channel hoàn tất
              
last_kernel : Kernel position hiện tại là vị trí cuối cùng trong tích chập
              → Reduction theo kernel hoàn tất (kH×kW done)
              
last_reduce : Đây là operation cuối trong toàn bộ reduction loop
              → Cả Cin và kernel position đều done
```

### 2.2. Flow Điều Khiển

```
if NOT last_pass:
  PSUM_accumulator += mac_result
  output_namespace = PSUM_BUFFER (INT32)
  PPU không kích hoạt
  Không write ra GLB_OUTPUT
  
if last_pass:
  PSUM_accumulator += mac_result    ← lần cộng cuối
  PPU kích hoạt:
    step 1: bias_add: PSUM += B_int32[cout]
    step 2: requant:  y_raw = (PSUM * M_int) >> shift
    step 3: offset:   y_off = y_raw + zp_out
    step 4: act:      y_act = activation_fn(y_off)   (SiLU LUT hoặc identity)
    step 5: clamp:    y_int8 = clamp(y_act, -128, 127)
  Write y_int8 → GLB_OUTPUT (ACT_namespace)
  Reset PSUM_accumulator = 0 cho output position tiếp theo

Lỗi nếu:
  last_pass=1 nhưng PPU chưa load bias/scale params → undefined behavior
  last_pass=0 nhưng PPU kích hoạt → wrong output
```

### 2.3. Ví Dụ: Conv3x3 Cin=64, Cout=16 với Cin_chunk=16

```
Tile 1 (cin=0..15,  last_cin=0): MAC → PSUM, last_pass=0 → accumulate
Tile 2 (cin=16..31, last_cin=0): MAC → PSUM, last_pass=0 → accumulate
Tile 3 (cin=32..47, last_cin=0): MAC → PSUM, last_pass=0 → accumulate
Tile 4 (cin=48..63, last_cin=1): MAC → PSUM, last_pass=1 → PPU → INT8

Nếu kernel cũng tích lũy (ít dùng cho Conv3x3 vì kernel nhỏ):
  last_kernel=1 chỉ sau khi toàn bộ 3×3=9 kernel positions đã xử lý
  Thông thường trong 1 tile: tất cả 9 kernel pos xử lý → last_kernel=1 luôn true
```

### 2.4. DW_7x7_MULTIPASS Last Pass

```
Pass 1 (rows 0-2): last_kernel=0, last_pass=0 → PSUM accumulate
Pass 2 (rows 3-5): last_kernel=0, last_pass=0 → PSUM accumulate
Pass 3 (row  6):   last_kernel=1, last_pass=1 → PPU kích hoạt

Chú ý: DW không có cross-channel Cin → last_cin luôn=1 (per-channel)
Vì last_pass = last_cin AND last_kernel AND last_reduce = 1 AND last_kernel AND 1
→ last_pass = last_kernel

Signal last_pass_kernel trong TILE_DESC bit 3 dùng để distinguish multipass cases.
```

---

## 3. PSUM / ACT Namespace

### 3.1. Phân Vùng Bộ Nhớ

```
┌──────────────────────────────────────────────────────────────┐
│                     GLB Memory Map                           │
│                                                              │
│  0x0000_0000 ┌─────────────────┐                           │
│              │  PSUM Buffer    │  (INT32, tối đa tile size) │
│  0x0002_0000 ├─────────────────┤                           │
│              │  ACT Buffer A   │  (INT8, ping-pong)        │
│  0x0010_0000 ├─────────────────┤                           │
│              │  ACT Buffer B   │  (INT8, ping-pong)        │
│  0x0020_0000 ├─────────────────┤                           │
│              │  HOLD_SKIP      │  (~900KB cho 4 skips)     │
│  0x0030_0000 ├─────────────────┤                           │
│              │  Weight SRAM    │                            │
│  0x0080_0000 └─────────────────┘                           │
└──────────────────────────────────────────────────────────────┘
```

### 3.2. Quy Tắc Truy Cập

```
PSUM_namespace (INT32):
  - Ghi: PE cluster → column_reduce → thẳng vào PSUM_Buffer
  - Đọc: PPU đọc từ PSUM_Buffer tại last_pass
  - Không expose ra ngoài accelerator
  - Giá trị INT32, ký hiệu: PSUM[cout, h_out, w_out]

ACT_namespace (INT8):
  - Ghi: PPU output → GLB_OUTPUT banks
  - Đọc: Layer tiếp theo đọc từ GLB_INPUT banks (ping-pong)
  - Có thể expose ra CPU interface (P3/P4/P5 outputs)
  - Giá trị INT8, ký hiệu: Y_int8[cout, h_out, w_out]

Ping-pong mechanism:
  Layer L viết vào ACT_Buffer_A
  Layer L+1 đọc từ ACT_Buffer_A và viết vào ACT_Buffer_B
  Layer L+2 đọc từ ACT_Buffer_B và viết vào ACT_Buffer_A
  (trừ HOLD_SKIP buffer – không ping-pong)
```

---

## 4. HOLD_SKIP Semantics

### 4.1. Định Nghĩa

HOLD_SKIP: Giữ lại tensor output trong bộ nhớ GLB sau khi layer compute xong, không giải phóng, cho đến khi consumer layer (QConcat) đọc xong.

### 4.2. State Machine

```
States:
  INACTIVE   : Không được sử dụng
  FILLING    : Layer producer đang compute, dần ghi vào HOLD_SKIP region
  READY      : Layer producer done, toàn bộ tensor đã ghi, sẵn sàng cho consumer
  CONSUMING  : Consumer (QConcat) đang đọc từ HOLD_SKIP region
  RELEASED   : Consumer done, vùng nhớ được giải phóng

Transitions:
  INACTIVE  → FILLING   : Khi first_tile của producer layer bắt đầu
  FILLING   → READY     : Khi last_tile của producer layer kết thúc
                          → Signal barrier: F{idx}_hold_ready = 1
  READY     → CONSUMING : Khi barrier release cho consumer layer
  CONSUMING → RELEASED  : Khi consumer layer last_tile done
                          → Giải phóng HOLD_SKIP region cho mục đích khác
```

### 4.3. Mapping Cụ Thể

```
HOLD_SKIP_A (F4, SKIP-A):
  Producer: L4 (QC2f)   → FILLING từ L4_tile0 đến L4_lastTile
  Ready signal: F4_hold_ready ← L4_done
  Consumer: L15 (QConcat source_layer[1])
  Region: skip_buf_base + 0, size=409,600 bytes

HOLD_SKIP_B (F6, SKIP-B):
  Producer: L6 (QC2f)   → FILLING trong L6 compute
  Ready signal: F6_hold_ready ← L6_done
  Consumer: L12 (QConcat source_layer[1])
  Region: skip_buf_base + 409,600, size=204,800 bytes

HOLD_SKIP_C (F8, SKIP-C):
  Producer: L8 (QC2f)   → FILLING trong L8 compute
  Ready signal: F8_hold_ready ← L8_done
  Consumer: L21 (QConcat source_layer[1])
  Region: skip_buf_base + 614,400, size=102,400 bytes

HOLD_SKIP_D (F13, SKIP-D):
  Producer: L13 (QC2f)  → FILLING trong L13 compute
  Ready signal: F13_hold_ready ← L13_done
  Consumer: L18 (QConcat source_layer[1])
  Region: skip_buf_base + 716,800, size=204,800 bytes
  Note: F6 đã được RELEASED tại L12_done trước khi F13 cần space
        → Vẫn dùng region của F6 (reuse) hoặc region riêng
```

---

## 5. Barrier Semantics

### 5.1. Định Nghĩa Barrier

Barrier = Synchronization point: một layer phải đợi tất cả producer dependencies hoàn tất trước khi bắt đầu compute.

### 5.2. Bốn Barrier trong Model (Critical Points)

```
BARRIER_L12:
  Wait condition: (L11_done == TRUE) AND (F6_hold_ready == TRUE)
  Action: release L12_first_tile_start
  Timeout: hardware timer (nếu quá lâu → error interrupt)
  
  Nếu L11 done nhưng F6 chưa READY: L12 phải stall
  Nếu F6 READY nhưng L11 chưa done: L12 phải stall
  Cả hai phải TRUE mới release

BARRIER_L15:
  Wait condition: (L14_done == TRUE) AND (F4_hold_ready == TRUE)
  Action: release L15_first_tile_start
  Note: F4 sinh từ L4, hold từ lúc L4 done, sẵn sàng rất lâu trước L15

BARRIER_L18:
  Wait condition: (L17_done == TRUE) AND (F13_hold_ready == TRUE)
  Action: release L18_first_tile_start

BARRIER_L21:
  Wait condition: (L20_done == TRUE) AND (F8_hold_ready == TRUE)
  Action: release L21_first_tile_start
  Note: F8 sinh SỚMEST nhất (L8), hold rất lâu, nên F8_hold_ready=1 từ rất sớm
```

### 5.3. Barrier Implementation

```
Hardware `barrier_manager.sv` duy trì:
  done_register[23]   : bit array, done_register[i]=1 khi layer i hoàn tất
  hold_ready[4]       : {F4_ready, F6_ready, F8_ready, F13_ready}

Logic:
  L12_start_en = done_register[11] AND hold_ready[F6]  ← BARRIER_L12
  L15_start_en = done_register[14] AND hold_ready[F4]  ← BARRIER_L15
  L18_start_en = done_register[17] AND hold_ready[F13] ← BARRIER_L18
  L21_start_en = done_register[20] AND hold_ready[F8]  ← BARRIER_L21

  done_register[i] ← set by tile_fsm khi last_tile của layer i complete
  hold_ready[Fx]   ← set bởi HOLD_SKIP state machine

desc_fetch_engine không fetch TILE_DESC[0] của QConcat layer
cho đến khi barrier release.
```

---

## 6. UPSAMPLE / CONCAT / MOVE Path

### 6.1. Các Primitive Không Qua PE

```
Primitives không cần PE compute:
  - UPSAMPLE_NEAREST (P6)
  - MOVE (P4)

Primitives cần routing nhưng không cần MAC:
  - CONCAT (P5) – chỉ routing, có thể cần mini-requant
  - MAXPOOL (P3) – cần max-compare tree, không phải MAC
```

### 6.2. UPSAMPLE_NEAREST Path

```
Thực hiện bởi tensor_post_engine (không qua PE cluster):

Input: F_in [C, H, W] được lưu trong GLB_OUTPUT của layer trước
Output: F_out [C, 2H, 2W] phải được ghi vào GLB_INPUT của layer tiếp theo

Mechanism: DMA với address pattern repeat:
  for h in 0..H-1:
    for w in 0..W-1:
      src_addr = compute_input_addr(h, w, ...)
      val = GLB.read(src_addr)
      
      dst_addr_00 = compute_output_addr(2h,   2w,   ...)
      dst_addr_01 = compute_output_addr(2h,   2w+1, ...)
      dst_addr_10 = compute_output_addr(2h+1, 2w,   ...)
      dst_addr_11 = compute_output_addr(2h+1, 2w+1, ...)
      
      GLB.write(dst_addr_00, val)  ← 4 writes cùng value
      GLB.write(dst_addr_01, val)
      GLB.write(dst_addr_10, val)
      GLB.write(dst_addr_11, val)

Scale/ZP: tensor_post_engine chuyển metadata unchanged.
No PPU involved.
```

### 6.3. CONCAT Path

```
Thực hiện bởi router_cluster + optional mini-PPU:

Case 1: Same domain (scale_A == scale_Y, zp_A == zp_Y):
  router_cluster routing:
    for h in 0..H-1:
      for w_blk in 0..Wblk_total-1:
        Read A_channels [Cin_A] từ GLB region A
        Write A_channels to output GLB at [cout=0..Cin_A-1]
        
        Read B_channels [Cin_B] từ HOLD_SKIP region B
        Write B_channels to output GLB at [cout=Cin_A..Cin_A+Cin_B-1]

Case 2: Domain mismatch (cần requant):
  router_cluster với mini-PPU:
    for each channel A:
      val_A = read from GLB region A
      val_A_aligned = mini_ppu_requant(val_A, scale_A, zp_A, scale_Y, zp_Y)
      write val_A_aligned to output GLB
    
    for each channel B:
      val_B = read from HOLD_SKIP region B
      val_B_aligned = mini_ppu_requant(val_B, scale_B, zp_B, scale_Y, zp_Y)
      write val_B_aligned to output GLB

mini_ppu_requant:
  y = clamp(round((val - zp_in) * (scale_in/scale_out)) + zp_out, -128, 127)
  Implemented với fixed-point multiply-shift
```

### 6.4. MOVE Path

```
Simple DMA copy:
  src_region → dst_region
  Metadata (scale, zp) pass-through từ LAYER_DESC
  Không qua PE, PPU, hay router routing logic
```

---

## 7. Tile Execution Order

### 7.1. General Tiling Policy

```
Outer loop: layer (L0 → L22)
  Middle loops: tiling dimensions (tùy partition_mode)
    partition_mode = TILE_HW:    outer=h_tile, inner=w_tile, innermost=cout_chunk
    partition_mode = TILE_COUT:  outer=cout_tile, inner=h_tile, innermost=cin_chunk
    partition_mode = TILE_CIN:   outer=h_tile, inner=cin_chunk, innermost=cout_chunk

  Innermost: PE execution per tile
```

### 7.2. SPPF Tile Order

```
L9 SPPF = OS_1x1 + MaxPool + MaxPool + MaxPool + CONCAT + OS_1x1

Execution:
  Sub-layer 9.1: OS_1x1 (Cin=256→128)  → output X1 [128,20,20]
  Sub-layer 9.2: MaxPool on X1          → P1 [128,20,20]  (hold X1, P1)
  Sub-layer 9.3: MaxPool on P1          → P2 [128,20,20]  (hold X1,P1,P2)
  Sub-layer 9.4: MaxPool on P2          → P3 [128,20,20]  (hold X1,P1,P2,P3)
  Sub-layer 9.5: CONCAT(X1,P1,P2,P3)   → Ycat [512,20,20] (release holds)
  Sub-layer 9.6: OS_1x1 (Cin=512→256)  → F9 [256,20,20]

Note: X1, P1, P2 cần được buffer đồng thời → total buffer cho SPPF tại 9.4:
  3 × 128 × 20 × 20 = 153,600 bytes ≈ 150KB internal SPPF buffer
```

### 7.3. QC2fCIB Tile Order (DW_7x7_MULTIPASS)

```
L22 QC2fCIB:
  Sub 22.1: OS_1x1(384→256)               → X1 [256,20,20]
  Sub 22.2: DW_7x7_MULTIPASS pass 1 (rows 0-2) → PSUM accumulate
  Sub 22.3: DW_7x7_MULTIPASS pass 2 (rows 3-5) → PSUM accumulate
  Sub 22.4: DW_7x7_MULTIPASS pass 3 (row 6)    → PPU → Y_dw [128,20,20]
  Sub 22.5: OS_1x1(128→128)               → Y_cib [128,20,20]
  Sub 22.6: CONCAT(Y_cib, X1_split)       → Ycat [256,20,20]
  Sub 22.7: OS_1x1(256→256)               → P5 [256,20,20]

Note: X1 một nửa channels (128) được giữ làm skip cho CONCAT trong Sub 22.6
```

---

## 8. Error và Exception Semantics

```
BARRIER TIMEOUT:
  Nếu barrier condition không được thỏa mãn trong N cycles:
  → Set error_interrupt
  → Halt execution
  → Report: which barrier, which dependency missing

PSUM OVERFLOW:
  PSUM INT32 cộng tất cả Cin×kH×kW terms
  INT32 có thể overflow nếu:
    Cin=256, kH=kW=3: 256×9=2304 terms × max(INT8×INT8)=127×127=16129
    max_psum = 2304 × 16129 = 37,161,216  ← fit INT32 vẫn OK
  Monitor: if psum > 2^31 - 1 → saturation flag (report, không crash)

HOLD_SKIP CONFLICT:
  Nếu producer cố ghi vào HOLD_SKIP region đang CONSUMING:
  → Error: producer layer chưa được cleared
  → Cần re-check layer HOLD_SKIP assignment không bị overlap

WRONG LAST_PASS:
  Nếu last_pass=1 nhưng không phải tile cuối → partial output to ACT
  Nếu last_pass=0 nhưng là tile cuối → PSUM never flushed
  → Cả hai: output sai nhưng không crash → phải catch trong test
```

---

## 9. Sign-off Checklist

```
LAST PASS:
☐ last_pass = last_cin AND last_kernel AND last_reduce: định nghĩa khóa
☐ PPU chỉ kích hoạt khi last_pass=1: verified trong psum_act_model.py
☐ DW_7x7: last_pass=1 chỉ tại pass 3 (row 6): verified
☐ PSUM reset sau mỗi last_pass: verified (không leak sang output tiếp)

HOLD_SKIP:
☐ 4 HOLD_SKIP regions định nghĩa đúng: F4, F6, F8, F13
☐ State machine INACTIVE→FILLING→READY→CONSUMING→RELEASED: correct
☐ Region sizes: F4=409600, F6=204800, F8=102400, F13=204800 bytes
☐ No overlap giữa 4 regions

BARRIER:
☐ 4 barriers: L12, L15, L18, L21 được implement trong barrier_manager
☐ Condition AND semantics: cả hai producer phải done
☐ Stall mechanism khi barrier chưa cleared

UPSAMPLE PATH:
☐ Không qua PE, không qua PPU
☐ scale/zp pass-through metadata
☐ 4 writes per source pixel (×2 repetition each dimension)

CONCAT PATH:
☐ router_cluster handle channel interleaving
☐ mini-PPU kích hoạt khi domain_align_en=1 trong POST_PROFILE
☐ Thứ tự channels: A_channels first, B_channels second

TILE ORDER:
☐ SPPF sub-layer order: OS_1x1 → Pool×3 → CONCAT → OS_1x1
☐ DW_7x7_MULTIPASS order: pass1 → pass2 → pass3(last_pass)
☐ first_tile flag reset PSUM: verified
```

*Execution semantics là "luật" mà tile_fsm.sv và barrier_manager.sv phải implement.*
