# Nghiên cứu Mapping Khối Tính Toán YOLOv10n lên Phần Cứng qua Primitive Set

> **Phạm vi**: Layer 0–22 (Backbone + Neck) của qYOLOv10n PTQ  
> **Mục tiêu**: Xây dựng bản đồ chi tiết từ model block → hardware primitive → RTL module  
> **Phiên bản**: Phase 0 Freeze Spec

---

## 1. Tổng quan kiến trúc phân cấp

```
┌─────────────────────────────────────────────────────────────────┐
│                    qYOLOv10n Model Block                        │
│  (Conv, QC2f, SCDown, SPPF, QPSA, Upsample, QConcat, QC2fCIB)  │
└──────────────────────────┬──────────────────────────────────────┘
                           │  decompose
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Primitive Set                              │
│  RS_DENSE_3x3 | OS_1x1 | DW_3x3 | MAXPOOL_5x5 | MOVE          │
│  CONCAT | UPSAMPLE_NEAREST | EWISE_ADD | DW_7x7_MULTIPASS      │
│  GEMM_ATTN_BASIC                                                │
└──────────────────────────┬──────────────────────────────────────┘
                           │  map to RTL
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RTL Hardware                               │
│  PE Cluster | GLB Banks | Router | PPU | Swizzle Engine        │
│  Descriptor Stack | Tile FSM | Barrier Manager                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Primitive Set chính thức & Đặc tả

### 2.1. Bảng Primitive Matrix

| Primitive ID | Tên chuẩn | Input Shape | Output Shape | Stride | Padding | Quant Domain | PPU | Dùng cho |
|---|---|---|---|---|---|---|---|---|
| P0 | RS_DENSE_3x3 | [H,W,Cin] | [Hout,Wout,Cout] | 1,2 | same | scale_in×scale_w→scale_out | Yes | Conv layers 0,1,3,17 |
| P1 | OS_1x1 | [H,W,Cin] | [H,W,Cout] | 1 | 0 | scale_in×scale_w→scale_out | Yes | Projection trong C2f, SPPF |
| P2 | DW_3x3 | [H,W,C] | [Hout,Wout,C] | 1,2 | same | per-channel weight | Yes | SCDown nhánh DW |
| P3 | MAXPOOL_5x5 | [H,W,C] | [H,W,C] | 1 | 2 | giữ nguyên (INT8 compare) | No | SPPF (×3 lặp) |
| P4 | MOVE | [H,W,C] | [H,W,C] | - | - | giữ nguyên | No | Skip lưu buffer |
| P5 | CONCAT | [H,W,C1],[H,W,C2] | [H,W,C1+C2] | - | - | common-domain requant | No | FPN/PAN neck |
| P6 | UPSAMPLE_NEAREST | [H,W,C] | [2H,2W,C] | - | - | giữ nguyên scale/zp | No | Neck upsample |
| P7 | EWISE_ADD | [H,W,C],[H,W,C] | [H,W,C] | - | - | common-domain requant | Yes | Residual (dự phòng) |
| P8 | DW_7x7_MULTIPASS | [H,W,C] | [H,W,C] | 1 | 3 | per-channel, bias ở pass cuối | Yes | QC2fCIB large kernel |
| P9 | GEMM_ATTN_BASIC | [N,HW,C] | [N,HW,C] | - | - | INT8 GEMM→requant | Yes | QPSA attention |

### 2.2. Quantization Rule cho từng Primitive

```
┌──────────────────┬──────────────────────────────────────────────────────────────┐
│ Primitive        │ Quantization Rule                                            │
├──────────────────┼──────────────────────────────────────────────────────────────┤
│ RS_DENSE_3x3     │ acc[i] = Σ (x-zp_x)(w-zp_w)·scale_x·scale_w + bias        │
│                  │ y_int8 = clamp(round(acc/scale_y + zp_y), -128, 127)        │
├──────────────────┼──────────────────────────────────────────────────────────────┤
│ OS_1x1           │ Tương tự RS_DENSE_3x3 với kernel 1×1                        │
├──────────────────┼──────────────────────────────────────────────────────────────┤
│ DW_3x3           │ Per-channel weight: scale_w[cout] riêng từng channel        │
│                  │ bias[cout] = fused_bn với scale_bias=scale_x·scale_w[cout]  │
├──────────────────┼──────────────────────────────────────────────────────────────┤
│ MAXPOOL_5x5      │ max(x_int8) → không đổi scale/zp, chỉ so sánh số nguyên   │
├──────────────────┼──────────────────────────────────────────────────────────────┤
│ CONCAT           │ Nếu scale A ≠ scale B: requant nhánh có scale nhỏ hơn       │
│                  │ về common scale trước khi concat                             │
├──────────────────┼──────────────────────────────────────────────────────────────┤
│ UPSAMPLE_NEAREST │ y[i,j] = x[i//2, j//2] → giữ nguyên scale/zp              │
├──────────────────┼──────────────────────────────────────────────────────────────┤
│ EWISE_ADD        │ align cả hai nhánh về common_scale trước khi add            │
│                  │ common_scale = max(scale_A, scale_B)                        │
├──────────────────┼──────────────────────────────────────────────────────────────┤
│ DW_7x7_MULTIPASS │ Pass 1,2: acc lưu PSUM; Pass cuối: +bias → requant → INT8 │
├──────────────────┼──────────────────────────────────────────────────────────────┤
│ GEMM_ATTN_BASIC  │ Q,K,V projection (OS_1x1) → QK^T/sqrt(d) → softmax(INT8-  │
│                  │ approx) → ×V → output projection                           │
└──────────────────┴──────────────────────────────────────────────────────────────┘
```

---

## 3. Layer-to-Primitive Mapping chi tiết (Layer 0–22)

### 3.1. Backbone – Đường xuống (Layer 0–10)

#### Layer 0 – `Conv` [3,640,640] → [16,320,320]

```
Block: Conv (k=3, s=2, BN_fuse, SiLU)
─────────────────────────────────────────────
Primitive:  RS_DENSE_3x3(stride=2)
  Input:    X_int8[1,3,640,640]
  Weight:   W_int8[3,3,3,16], B_int32[16]
  Output:   Y_int8[1,16,320,320]
  scale_in: 0.00392 (= 1/255), zp_in: 0
  
Execution steps:
  1. MAC: acc[cout] = Σ_{kh,kw,cin} (x-zp_x)(w-zp_w)   →  int32 psum
  2. Bias: acc += B_int32[cout]
  3. Requant: y_raw = round(acc * M) >> shift             →  int32 clip
     where M = scale_in * scale_w[cout] / scale_out
  4. Activation: SiLU approximation (LUT) hoặc dequant→SiLU→requant
  5. Clamp: y_int8 = clamp(y_raw + zp_out, -128, 127)
  
Hardware path: GLB_IN → window_gen_3x3 → PE_MAC × LANES=16 → PSUM_buf → PPU → GLB_OUT
Tile size: tile_h × tile_w × Cin_chunk → Cout_chunk per tile
Memory: GLB_bank_input (h mod 3), bank_output (out_row mod 4)
```

#### Layer 1 – `Conv` [16,320,320] → [32,160,160]

```
Primitive:  RS_DENSE_3x3(stride=2)
  Input:    F0_out: INT8[1,16,320,320]
  Weight:   W_int8[3,3,16,32], B_int32[32]
  Output:   INT8[1,32,160,160]

Notes: Giống layer 0, chỉ khác Cin=16, Cout=32, H/W=320→160
```

#### Layer 2 – `QC2f` [32,160,160] → [32,160,160]

```
Block: QC2f (n=1 bottleneck)
─────────────────────────────────────────────────────────────
Primitive sequence (OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1):

  Step 1: cv1 = OS_1x1
    Input:  F1_out [1,32,160,160]
    Weight: W[1,1,32,32], B[32]
    Output: X1_int8 [1,32,160,160]   ← nhánh split

  Step 2: bottleneck_cv1 = RS_DENSE_3x3 (internal)
    Input:  X1_int8 [1,32,160,160]  (low-half channels = 16ch actual)
    Weight: W[3,3,16,16]
    Output: Y_tmp [1,16,160,160]

  Step 3: bottleneck_cv2 = RS_DENSE_3x3 (internal)
    Input:  Y_tmp [1,16,160,160]
    Output: Y_branch [1,16,160,160]

  Step 4: CONCAT (Y_branch, X1_nhánh_giữ)
    Inputs: [1,16,160,160] + [1,16,160,160] = [1,32,160,160]
    → common-domain requant nếu scale khác nhau

  Step 5: cv2 = OS_1x1
    Input:  [1,32,160,160]  (sau concat)
    → OS_1x1 gom về C_out=32

  Output: F2_out [1,32,160,160]

Intermediate buffers cần lưu:
  - X1_int8 (nhánh skip nội bộ C2f)
  - Y_tmp  (kết quả giữa chừng trong bottleneck)
```

#### Layer 3 – `Conv` [32,160,160] → [64,80,80]

```
Primitive: RS_DENSE_3x3(stride=2), Cin=32, Cout=64
```

#### Layer 4 – `QC2f` [64,80,80] → [64,80,80]

```
Tương tự Layer 2, scale tăng lên Cin=Cout=64
Primitive sequence: OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1
Output: F4_out [1,64,80,80]   ⚠️ CẦN LƯU - skip đến L15
```

#### Layer 5 – `SCDown` [64,80,80] → [128,40,40]

```
Block: SCDown (Spatial Channel Downsample)
─────────────────────────────────────────────────────────────
Primitive sequence (OS_1x1 + DW_3x3(stride=2)):

  Phương án hardware-friendly (theo spec freeze):
  
  Step 1: OS_1x1  ← channel adjustment
    Input:  F4_out [1,64,80,80]
    Output: tmp [1,128,80,80]  (hoặc [1,64,80,80] rồi split)

  Step 2: DW_3x3(stride=2)  ← spatial downsample, per-channel
    Input:  tmp [1,...,80,80]
    Output: F5_out [1,128,40,40]

Notes về SCDown trong YOLOv10:
  - Thực tế có 2 nhánh: nhánh conv3x3 s2 + nhánh DW_3x3 s2 sau OS_1x1
  - 2 nhánh xử lý C/2 kênh mỗi nhánh, sau đó CONCAT
  
Primitive sequence đầy đủ:
  Branch A: OS_1x1(Cin→Cout/2) → DW_3x3(s2)
  Branch B: OS_1x1(Cin→Cout/2) → DW_3x3(s2)
  CONCAT(A, B) → F5_out [1,128,40,40]
```

#### Layer 6 – `QC2f` [128,40,40] → [128,40,40]

```
Primitive sequence: OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1
Output: F6_out [1,128,40,40]   ⚠️ CẦN LƯU - skip đến L12
```

#### Layer 7 – `SCDown` [128,40,40] → [256,20,20]

```
Tương tự Layer 5, scale lên Cin=128, Cout=256
```

#### Layer 8 – `QC2f` [256,20,20] → [256,20,20]

```
Primitive sequence: OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1
Output: F8_out [1,256,20,20]   ⚠️ CẦN LƯU - skip đến L21
```

#### Layer 9 – `SPPF` [256,20,20] → [256,20,20]

```
Block: SPPF (Spatial Pyramid Pooling Fast)
─────────────────────────────────────────────────────────────
Primitive sequence:
  Step 1: cv1 = OS_1x1
    Input:  F8_out [1,256,20,20]
    Output: X1 [1,128,20,20]   (giảm kênh xuống c2=128)

  Step 2: MAXPOOL_5x5 × 3 (lặp)
    P1 = MAXPOOL_5x5(X1)  → [1,128,20,20]
    P2 = MAXPOOL_5x5(P1)  → [1,128,20,20]
    P3 = MAXPOOL_5x5(P2)  → [1,128,20,20]

  Step 3: CONCAT(X1, P1, P2, P3)
    [1,128], [1,128], [1,128], [1,128] → [1,512,20,20]
    ⚠️ Tất cả 4 nhánh từ cùng qconfig → scale/zp tương tự → concat đơn giản

  Step 4: cv2 = OS_1x1
    Input:  [1,512,20,20]
    Output: F9_out [1,256,20,20]

Lưu ý hardware:
  - MAXPOOL_5x5 chỉ so sánh INT8, không cần PPU hay requant
  - X1, P1, P2 phải được buffer đồng thời trước khi CONCAT
```

#### Layer 10 – `QPSA` [256,20,20] → [256,20,20]

```
Block: QPSA (Quantized Pixel/Position Sensitive Attention)
─────────────────────────────────────────────────────────────
Primitive sequence (OS_1x1 + GEMM_ATTN_BASIC + OS_1x1):

  Step 1: split = OS_1x1
    Input:  F9_out [1,256,20,20]
    → split thành nhánh attn và nhánh pass-through
    QAttn_in [1,128,20,20], Pass [1,128,20,20]

  Step 2: GEMM_ATTN_BASIC
    Input:  QAttn_in [1,128,20,20]
    → reshape: [1, 400, 128]  (HW=20×20=400, C=128)
    → Q proj OS_1x1 → [1,400,64]
    → K proj OS_1x1 → [1,400,64]
    → V proj OS_1x1 → [1,400,128]
    → Attn = Q×K^T / sqrt(64)  →  [1,400,400]  (INT8 GEMM)
    → Attn_soft = softmax_approx(Attn)  →  requant → INT8
    → Out = Attn_soft × V →  [1,400,128]
    → reshape back: [1,128,20,20]

  Step 3: CONCAT / EWISE_ADD (merge với Pass)
    [1,128,20,20] + [1,128,20,20] → concat → [1,256,20,20]

  Step 4: output proj = OS_1x1
    [1,256,20,20] → [1,256,20,20]

  Output: F10_out [1,256,20,20]

⚠️ GEMM_ATTN_BASIC là primitive phức tạp nhất:
   - GEMM thực hiện bằng PE MAC với output accumulate INT32
   - softmax approximation: dùng INT8 LUT hoặc log-sum-exp phân đoạn
   - Tại stage 20×20, tensor nhỏ: throughput không phải ưu tiên số 1
```

---

### 3.2. Neck – FPN (Layer 11–16)

#### Layer 11 – `Upsample` [256,20,20] → [256,40,40]

```
Primitive: UPSAMPLE_NEAREST(scale=2)
  Input:   F10_out [1,256,20,20]
  Output:  F11_out [1,256,40,40]
  
Rule: y[h,w,c] = x[h//2, w//2, c]
  → scale_out = scale_in, zp_out = zp_in  (không thay đổi)
  
Hardware path: swizzle_engine / tensor_post_engine
  - Phát địa chỉ source theo pattern ×2 repetition
  - Không cần PE MAC hay PPU
  - Chỉ là router/DMA với address remapping
```

#### Layer 12 – `QConcat` [256,40,40]+[128,40,40] → [384,40,40]

```
Primitive: CONCAT
  Input A: F11_out [1,256,40,40]  (scale_A, zp_A) từ upsample
  Input B: F6_out  [1,128,40,40]  (scale_B, zp_B) từ backbone  ⚠️ SKIP DEPENDENCY
  Output:  F12_out [1,384,40,40]  (scale_Y, zp_Y)

Common-domain alignment:
  if scale_A ≠ scale_B:
    Chọn scale_Y = scale_B (thường scale backbone là reference)
    Requant A → scale_Y: A' = round((A - zp_A) × (scale_A/scale_Y)) + zp_Y
    Concat(A', B) theo chiều channel
  else:
    Concat trực tiếp

Hardware path:
  - Đọc F11_out từ GLB bank A
  - Đọc F6_out (đã được MOVE/HOLD từ bước trước) từ GLB bank B
  - Router chuyển luân phiên: 256 channel A, rồi 128 channel B → concatenated output
  
⚠️ F6_out phải được lưu trong GLB (HOLD_SKIP) từ thời điểm compute layer 6
   cho đến khi layer 12 sẵn sàng consume nó.
```

#### Layer 13 – `QC2f` [384,40,40] → [128,40,40]

```
Primitive sequence: OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1
  Input Cin=384, Output Cout=128
  Đây là tầng FPN feature trung bình (P4 vùng)
Output: F13_out [1,128,40,40]   ⚠️ CẦN LƯU - skip đến L18
```

#### Layer 14 – `Upsample` [128,40,40] → [128,80,80]

```
Primitive: UPSAMPLE_NEAREST(scale=2)
  Input:  F13_out [1,128,40,40]
  Output: F14_out [1,128,80,80]
  (scale, zp giữ nguyên)
```

#### Layer 15 – `QConcat` [128,80,80]+[64,80,80] → [192,80,80]

```
Primitive: CONCAT
  Input A: F14_out [1,128,80,80] từ upsample
  Input B: F4_out  [1,64,80,80]  từ backbone  ⚠️ SKIP DEPENDENCY
  Output:  F15_out [1,192,80,80]

⚠️ F4_out phải được lưu HOLD_SKIP từ layer 4 → đây là skip dài nhất trong flow
```

#### Layer 16 – `QC2f` [192,80,80] → [64,80,80]

```
Primitive sequence: OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1
  Cin=192, Cout=64, H=W=80  ← spatial size lớn nhất
  → Đây là P3 feature (80×80) → OUTPUT P3
Output: F16_out = P3_int8 [1,64,80,80]  ✅ OUTPUT P3
```

---

### 3.3. Neck – PAN (Layer 17–22)

#### Layer 17 – `Conv` [64,80,80] → [64,40,40]

```
Primitive: RS_DENSE_3x3(stride=2)
  Cin=64, Cout=64, H=80→40
  Đây là nhánh downsample để kết nối P3→P4 trong PAN
```

#### Layer 18 – `QConcat` [64,40,40]+[128,40,40] → [192,40,40]

```
Primitive: CONCAT
  Input A: F17_out [1,64,40,40]  từ conv downsample
  Input B: F13_out [1,128,40,40] từ tầng trung FPN  ⚠️ SKIP DEPENDENCY
  Output:  F18_out [1,192,40,40]

⚠️ F13_out phải được lưu HOLD_SKIP từ layer 13 → consume tại layer 18
```

#### Layer 19 – `QC2f` [192,40,40] → [128,40,40]

```
Primitive sequence: OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1
  Cin=192, Cout=128
Output: F19_out = P4_int8 [1,128,40,40]  ✅ OUTPUT P4
```

#### Layer 20 – `SCDown` [128,40,40] → [128,20,20]

```
Primitive sequence: OS_1x1 + DW_3x3(stride=2)
  Cin=Cout=128, downsample 40→20
```

#### Layer 21 – `QConcat` [128,20,20]+[256,20,20] → [384,20,20]

```
Primitive: CONCAT
  Input A: F20_out [1,128,20,20] từ SCDown
  Input B: F8_out  [1,256,20,20] từ deep backbone  ⚠️ SKIP DEPENDENCY
  Output:  F21_out [1,384,20,20]

⚠️ F8_out phải được lưu HOLD_SKIP từ layer 8 → đây là skip dài nhất toàn flow
   Giữ đến tận layer 21 → cần allocation GLB space xuyên suốt
```

#### Layer 22 – `QC2fCIB` [384,20,20] → [256,20,20]

```
Block: QC2fCIB (C2f with CIB - large kernel)
─────────────────────────────────────────────────────────────
Primitive sequence:
  OS_1x1 + DW_7x7_MULTIPASS + OS_1x1 + (RS_DENSE_3x3 | DW_3x3) + CONCAT + OS_1x1

  Step 1: cv1 = OS_1x1
    Input:  [1,384,20,20] → [1,256,20,20]

  Step 2: CIB bottleneck:
    a. DW_7x7_MULTIPASS
       Kernel 7×7 chia làm 3 pass: [3 rows] + [3 rows] + [1 row]
       Pass 1,2: lưu PSUM namespace
       Pass 3 (last_pass): PSUM + bias → requant →INT8
       
    b. OS_1x1 compression
    
    c. (Optional) RS_DENSE_3x3 thêm

  Step 3: CONCAT (nhánh CIB + skip) theo chiều channel

  Step 4: cv2 = OS_1x1
    Output: P5_int8 [1,256,20,20]  ✅ OUTPUT P5

Output: F22_out = P5_int8 [1,256,20,20]

⚠️ DW_7x7_MULTIPASS cần trace pass-by-pass cho RTL verification
```

---

## 4. Dependency Graph và Buffer Management

### 4.1. Sơ đồ phụ thuộc Skip Connection

```
L0 → L1 → L2 → L3 → L4 ─────────────────────────────────────────────► L15 (concat)
                      │                                                   ▲
                      └─► L5 → L6 ──────────────────────────────────► L12 (concat)
                                 │                                       ▲
                                 └─► L7 → L8 ────────────────────────► L21 (concat)
                                           │                             ▲
                                           └─► L9 → L10 → L11 ─────────┘
                                                              │
                                                              └─► L12 (nhánh upsample)
                                                                    │
                                                                    ▼
                                                              L13 ─────────────────► L18 (concat)
                                                               │                      ▲
                                                               └─► L14 → L15          │
                                                                           │           │
                                                                           ▼           │
P3 = L16 ◄───────────────────────────────────────────────────── L16          L17 ────┘
P4 = L19 ◄── L18 ← L13(skip) + L17
P5 = L22 ◄── L22 ← L21 ← L20 + L8(skip)
```

### 4.2. Bảng HOLD_SKIP Buffer Requirements

| Skip tensor | Sinh ra tại | Tiêu thụ tại | Khoảng cách | Kích thước buffer |
|---|---|---|---|---|
| F4_out | L4 | L15 | 11 layer | INT8 [1,64,80,80] = 409,600 bytes |
| F6_out | L6 | L12 | 6 layer | INT8 [1,128,40,40] = 204,800 bytes |
| F8_out | L8 | L21 | 13 layer | INT8 [1,256,20,20] = 102,400 bytes |
| F13_out | L13 | L18 | 5 layer | INT8 [1,128,40,40] = 204,800 bytes |

**Tổng GLB buffer cần dự trữ cho SKIP**: ~921,600 bytes (~900 KB)

### 4.3. Barrier Management

```
barrier_L12: wait (L11_done AND L6_hold_ready) → release L12_start
barrier_L15: wait (L14_done AND L4_hold_ready) → release L15_start
barrier_L18: wait (L17_done AND L13_hold_ready) → release L18_start
barrier_L21: wait (L20_done AND L8_hold_ready) → release L21_start
```

---

## 5. Hardware Datapath theo Primitive

### 5.1. RS_DENSE_3x3 / OS_1x1 – Compute Path

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Compute Pipeline                                 │
│                                                                      │
│  GLB_INPUT ──► window_gen_3x3/1x1                                   │
│                    │                                                 │
│                    ▼                                                 │
│  GLB_WEIGHT ──► PE_LANE_MAC × 16 (LANES=16)                        │
│                    │ INT8×INT8 → INT32 MAC                          │
│                    ▼                                                 │
│               column_reduce (Cin tích lũy, horizontal)              │
│                    │                                                 │
│                    ▼                                                 │
│               PSUM_BUF → last_pass?                                 │
│                   No: lưu PSUM namespace                            │
│                   Yes: → PPU                                        │
│                         ├─ bias_add (INT32)                         │
│                         ├─ requant (scale_mul × shift)              │
│                         ├─ activation (SiLU LUT / approx)          │
│                         └─ clamp → INT8                             │
│                              │                                      │
│                              ▼                                      │
│               GLB_OUTPUT (bank_output = out_row mod 4)             │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.2. DW_3x3 – Depthwise Path

```
Khác biệt so với RS_DENSE_3x3:
- groups = Cin → mỗi channel xử lý độc lập
- weight shape: [3,3,1,Cout] với Cout = Cin
- Mỗi PE_LANE chỉ MAC trong 1 channel → không cần column_reduce
- Per-channel bias và scale_w[c] riêng từng channel
- last_pass luôn = true (không accumulate cross-channel)
```

### 5.3. MAXPOOL_5x5 – Pool Path

```
window_gen_5x5 ──► max_compare_tree (INT8 max, 25 inputs)
                        │
                        ▼ (không qua PPU)
                   GLB_OUTPUT (cùng scale/zp với input)

- Kernel 5×5, padding=2, stride=1
- Lặp 3 lần trong SPPF → output của lần trước là input lần sau
- GLB cần buffer P1, P2 song song với X1 để CONCAT sau
```

### 5.4. UPSAMPLE_NEAREST – Tensor Post Path

```
Không qua PE. Thực hiện qua tensor_post_engine hoặc swizzle_engine.

Address remapping:
  src_addr(c, h, w) → dst_addr(c, h//2, w//2)  ← đọc cùng src addr 4 lần (2×2 block)

Hoặc DMA với stride pattern:
  dst[c][2h  ][2w  ] = src[c][h][w]
  dst[c][2h  ][2w+1] = src[c][h][w]
  dst[c][2h+1][2w  ] = src[c][h][w]
  dst[c][2h+1][2w+1] = src[c][h][w]
```

### 5.5. CONCAT – Router Path

```
Input A: [H,W,C_A] với (scale_A, zp_A)
Input B: [H,W,C_B] với (scale_B, zp_B)

Case 1: scale_A = scale_B (ideal)
  → router_cluster chuyển A_channels, rồi B_channels → output interleaved by channel

Case 2: scale_A ≠ scale_B (cần requant)
  → nhánh có scale khác phải qua PPU mini-requant trước khi vào concat
  → common_scale = được chọn offline (thường là scale của backbone branch)
  → requant_params được kiểm tra tại LAYER_DESC
```

### 5.6. DW_7x7_MULTIPASS – Multi-pass Pipeline

```
Kernel 7×7 chia làm passes: [rows 0-2] + [rows 3-5] + [row 6]
(đảm bảo tương đương kết quả monolithic 7×7)

Pass 1 (rows 0-2):
  window_gen partial (3 kernel rows) + DW_MAC → PSUM_namespace (INT32)
  last_kernel = 0, last_pass = 0

Pass 2 (rows 3-5):
  window_gen partial (3 kernel rows) + DW_MAC → accumulate PSUM_namespace
  last_kernel = 0, last_pass = 0

Pass 3 (row 6, last_kernel=1, last_pass=1):
  window_gen partial (1 kernel row) + DW_MAC → ACC to PSUM
  → PPU: bias_add + requant + clamp → INT8 → ACT_namespace

Debug trace: xuất PSUM sau mỗi pass để verify golden vs RTL
```

### 5.7. GEMM_ATTN_BASIC – Attention Path

```
QPSA tại 20×20 → HW_seq_len = 400 (nhỏ, phù hợp hardware)

Q = OS_1x1(X_attn)   → [1,Hq,20*20]
K = OS_1x1(X_attn)   → [1,Hk,20*20]
V = OS_1x1(X_attn)   → [1,Hv,20*20]

Attn = Q × K^T  → [1,400,400]  INT8 GEMM
  - Dùng PE MAC với K^T được transpose trong GLB
  - Scale: M_attn = scale_Q × scale_K / scale_Attn

Attn_soft = softmax_approx(Attn / sqrt(Hq))
  - INT8 softmax approx: dùng lookup table hoặc piecewise linear
  - Output requant về INT8

Out = Attn_soft × V → [1,400,Hv]
  - INT8 GEMM
  - reshape → [1,Hv,20,20]
  
Output proj = OS_1x1 → F10_out [1,256,20,20]
```

---

## 6. Layout/Addressing Rules cho từng Primitive

### 6.1. Banking Model

```
bank_input  = h mod 3           → 3 input banks: bank0, bank1, bank2
bank_output = out_row mod 4     → 4 output banks

Ý nghĩa: Conv3x3 stride1 cần row h-1, h, h+1 → 3 banks xoay vòng
Conv3x3 stride2 cần row h, h+1, h+1 → vẫn 3 banks
```

### 6.2. Row Slot Model

```
Q_in    = ceil((K_eff + 3*stride) / 3)
row_slot = floor(h / 3) mod Q_in

Ví dụ Conv3x3 stride=1:
  K_eff = 3, stride = 1
  Q_in = ceil((3 + 3*1) / 3) = ceil(6/3) = 2
  row_slot ∈ {0, 1} → 2 slot positions per bank

Ví dụ Conv3x3 stride=2:
  K_eff = 3, stride = 2
  Q_in = ceil((3 + 3*2) / 3) = ceil(9/3) = 3
  row_slot ∈ {0, 1, 2}
```

### 6.3. Lane Packing

```
LANES = 16
lane  = x mod 16          → column trong warp
Wblk  = floor(x / 16)     → horizontal block index

pack16(data[W, C]):
  → packed[W//16, C, 16]   (16 values per lane per channel)

unpack16(packed[W//16, C, 16]):
  → data[W, C]

Áp dụng: khi load input từ GLB vào PE lane array
  - 16 lanes × Cin channels × 1 spatial point per cycle
```

### 6.4. Address Mapping

```
Physical address (logical):
  addr = bank_base
       + row_slot * (Wblk_total * Cin * LANES)
       + Wblk * (Cin * LANES)
       + cin_idx * LANES
       + lane_id

Với:
  bank_base   = GLB_BANK[bank_input]  phần bắt đầu cho bank số đó
  Wblk_total  = ceil(W / LANES)
  lane_id     = x mod LANES
  Wblk        = x // LANES
```

### 6.5. PSUM_MODE vs ACT_MODE

```
if NOT last_pass:
    output → PSUM_namespace (INT32, địa chỉ trong GLB output riêng)
    Không qua PPU bias/requant
    
if last_pass:
    PSUM_namespace + bias_add → requant → activation → clamp
    output → ACT_namespace (INT8, địa chỉ normal GLB output)
    
Điều kiện last_pass:
    last_pass = last_cin AND last_kernel AND last_reduce
```

---

## 7. Descriptor Mapping

### 7.1. Mỗi Primitive cần các Descriptor

| Descriptor | Nội dung bắt buộc | Ví dụ cho RS_DENSE_3x3 |
|---|---|---|
| NET_DESC | version, num_layers, weight_base, act_base | v1, 23 layers, 0x0000, 0x40000 |
| LAYER_DESC | primitive_id, in/out shape, kernel, stride | P0, [1,3,640,640],[1,16,320,320], 3, 2 |
| TILE_DESC | tile bounds, cin_chunk, cout_chunk, flags | h=0..31, Wblk=0..39, cin=0..2, cout=0..15 |
| ROUTER_PROFILE | route source→dest, broadcast mask | GLB_IN→PE, no broadcast |
| POST_PROFILE | bias_en, scale_mul, scale_shift, zp_out, clamp | en=1, M=..., shift=..., zp=0, [-128,127] |

### 7.2. Flags Semantics

```
flags:
  first_tile  : tile đầu tiên của layer → reset PSUM accumulator
  edge_tile   : tile chạm biên ảnh → padding zeros
  hold_skip   : tensor output cần giữ lại trong GLB cho skip đến sau
  need_swizzle: output cần qua swizzle engine (transpose, upsample, concat)

last_flags:
  last_cin    : đây là Cin chunk cuối → reduce channel dimension
  last_kernel : đây là kernel position cuối → reduce kernel dimension
  last_reduce : đây là tile cuối trong reduction → trigger PPU
```

---

## 8. Block → RTL Module Mapping

### 8.1. Bảng Mapping

| Block | Primitives | RTL Modules chính |
|---|---|---|
| Conv (L0,1,3,17) | RS_DENSE_3x3(s2) | window_gen, pe_lane_mac, column_reduce, ppu_lite |
| QC2f (L2,4,6,8,13,16,19) | OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1 | pe_cluster, glb_banks, router_cluster, ppu_lite |
| SCDown (L5,7,20) | OS_1x1 + DW_3x3(s2) | pe_cluster (dw mode), glb_banks, ppu_lite |
| SPPF (L9) | OS_1x1 + MAXPOOL_5x5×3 + CONCAT + OS_1x1 | pool_engine, router_cluster, glb_banks |
| QPSA (L10) | OS_1x1 + GEMM_ATTN_BASIC + OS_1x1 | gemm_attn_engine, pe_cluster, ppu_lite |
| Upsample (L11,14) | UPSAMPLE_NEAREST | swizzle_engine / tensor_post_engine |
| QConcat (L12,15,18,21) | CONCAT | router_cluster, (mini PPU cho requant nếu cần) |
| QC2fCIB (L22) | OS_1x1 + DW_7x7_MULTIPASS + OS_1x1 + CONCAT | pe_cluster (dw multipass), ppu_lite |

### 8.2. RTL Module Roles

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Module               │ Chức năng                                        │
├─────────────────────────────────────────────────────────────────────────┤
│ window_gen           │ Sinh cửa sổ 3×3/1×1/5×5 từ GLB input           │
│ pe_lane_mac          │ 16-lane INT8×INT8→INT32 MAC                     │
│ column_reduce        │ Sum across kernel positions (Cin tích lũy)      │
│ pe_cluster           │ Bao gồm window_gen + pe_lane_mac + column_reduce │
│ ppu_lite             │ bias_add + requant (scale×shift) + clamp + act  │
│ pool_engine          │ max_compare_tree cho MAXPOOL                    │
│ gemm_attn_engine     │ Matrix multiplication cho attention             │
│ router_cluster       │ Routing dữ liệu giữa GLB banks và PE          │
│ swizzle_engine       │ Tensor reshape/upsample/transpose               │
│ tensor_post_engine   │ UPSAMPLE_NEAREST, MOVE operations               │
│ glb_*_bank           │ Global Line Buffer banks (input/output/weight)  │
│ addr_gen_input       │ Tạo địa chỉ load input theo banking model       │
│ addr_gen_weight      │ Tạo địa chỉ load weight                        │
│ addr_gen_output      │ Tạo địa chỉ write output                       │
│ row_slot_manager     │ Quản lý row_slot vòng xoay trong GLB          │
│ tile_fsm             │ Điều khiển tiling loop                         │
│ desc_fetch_engine    │ Fetch và parse descriptor stack                │
│ barrier_manager      │ Đồng bộ producer→consumer cho skip/concat     │
│ subcluster_wrapper   │ Wrapper ghép các module thành block-level      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Golden Python → RTL Verification Flow

### 9.1. Thứ tự xây dựng Golden Python

```
Phase 1: Core primitives
  config.py + types.py
  quant_affine.py (quantize, dequantize, requant, clamp)
  quant_domain_align.py (common-domain cho concat/add)
  
Phase 2: Primitives
  primitive_conv.py   (RS_DENSE_3x3, OS_1x1)
  primitive_dw.py     (DW_3x3, DW_7x7_MULTIPASS)
  primitive_pool.py   (MAXPOOL_5x5)
  primitive_tensor.py (MOVE, CONCAT, UPSAMPLE_NEAREST, EWISE_ADD)
  primitive_psa.py    (GEMM_ATTN_BASIC)
  
Phase 3: Layout models
  banking_model.py    (bank_input, bank_output)
  row_slot_model.py   (Q_in, row_slot)
  lane_packing.py     (pack16/unpack16)
  address_model.py    (logical → physical)
  psum_act_model.py   (PSUM/ACT namespace semantics)
  
Phase 4: Block models
  block_qc2f.py
  block_scdown.py
  block_sppf.py
  block_qpsa.py
  block_qc2fcib.py
  
Phase 5: Model runner
  layer_specs.py      (bảng layer 0–22)
  model_forward_runner.py (chạy end-to-end)
  
Phase 6: Tests
  test_primitives.py, test_quant.py, test_layout.py
  test_blocks.py, test_model_forward.py
```

### 9.2. Sign-off Criteria

```
☐ test_primitives.py PASS:
    conv3x3 s1/s2, conv1x1, dw3x3, dw7x7 multipass, pool, upsample, concat, add

☐ test_quant.py PASS:
    quantize/dequantize round-trip, requant accuracy
    common-domain concat với domain mismatch
    common-domain add với saturation

☐ test_layout.py PASS:
    bank_input: h=0→bank0, h=1→bank1, h=2→bank2, h=3→bank0
    row_slot: Conv3x3 s1: Q_in=2; Conv3x3 s2: Q_in=3
    pack16/unpack16 round-trip
    address mapping không overlap

☐ test_blocks.py PASS:
    QC2f: shape [1,32,160,160]→[1,32,160,160], int8 output
    SCDown: shape [1,64,80,80]→[1,128,40,40]
    SPPF: shape [1,256,20,20]→[1,256,20,20], 3× pool đúng
    QPSA: shape [1,256,20,20]→[1,256,20,20], attention correct
    QC2fCIB: DW7x7 multipass == monolithic 7x7

☐ test_model_forward.py PASS:
    Layer 0–22 end-to-end: P3[1,64,80,80], P4[1,128,40,40], P5[1,256,20,20]
    Stage outputs cho tất cả 23 stages
    Quant metadata đúng (scale, zp cho P3/P4/P5)
    Dump traces: layout/address traces, PSUM/ACT traces, DW7x7 pass traces
```

---

## 10. Critical Path và Risk Analysis

### 10.1. Rủi ro cao nhất

| Rủi ro | Mô tả | Giải pháp |
|---|---|---|
| **Concat domain mismatch** | scale_A ≠ scale_B tại L12,15,18,21 | common-domain requant, kiểm tra qua test_quant.py |
| **DW_7x7 multipass accuracy** | Pass split phải cho kết quả bằng monolithic | Test bắt buộc: multipass == monolithic 7×7 |
| **GLB skip buffer pressure** | F4, F6, F8 phải sống lâu trong GLB | Tính toán tổng 900KB dự trữ, verify không overlap |
| **QPSA softmax approximation** | INT8 softmax thiếu chính xác | Kiểm tra accuracy drop so với float reference |
| **SiLU INT8 approximation** | SiLU không natural cho INT8 | LUT 8-bit với 256 entries hoặc piecewise linear |
| **PSUM/ACT namespace** | last_pass sai → output sai stage | Verify flag logic qua psum_act_model.py trước RTL |

### 10.2. Thứ tự ưu tiên implement

```
1. quant_affine.py         ← nền tảng cho mọi thứ
2. quant_domain_align.py   ← risk số 1 của neck
3. primitive_conv.py       ← RS_DENSE_3x3 chiếm ~70% compute
4. primitive_tensor.py     ← CONCAT, UPSAMPLE critical path
5. primitive_dw.py         ← DW_7x7_MULTIPASS risk
6. block_qc2f.py           ← most repeated block (7 lần)
7. model_forward_runner.py ← integration test
```

---

## 11. Ví dụ Tính Toán Chi tiết: RS_DENSE_3x3

### 11.1. Math Reference (Layer 0)

```python
# Input đã quantize từ CPU
X_int8 = tensor(shape=[1,3,640,640], dtype=int8)  # int8 thực tế [-128,127]
scale_x = 0.003921568627  # ≈ 1/255
zp_x = 0  # quant8: y = round(x/scale + zp), zp=0 cho unsigned

# Weight (per-output-channel quantize)
W_int8 = tensor(shape=[16,3,3,3], dtype=int8)  # [Cout, Cin, kH, kW]
scale_w = tensor(shape=[16], dtype=float32)  # per-channel
zp_w = 0  # weight INT8 thường zp=0 (symmetric)

# Bias (đã fuse BN offline)
B_int32 = tensor(shape=[16], dtype=int32)
# B_int32[cout] = BN_fused_bias / (scale_x * scale_w[cout])

# Output scale (học từ PTQ calibration)
scale_y = tensor(scalar, dtype=float32)
zp_y = 0

# Tính toán (floor model):
for cout in range(16):
    for h_out in range(320):
        for w_out in range(320):
            acc = 0  # int32
            for cin in range(3):
                for kh in range(3):
                    for kw in range(3):
                        h_in = 2*h_out + kh - 1  # stride=2, padding=1
                        w_in = 2*w_out + kw - 1
                        if 0 <= h_in < 640 and 0 <= w_in < 640:
                            acc += int32(X_int8[0,cin,h_in,w_in] - zp_x) * \
                                   int32(W_int8[cout,cin,kh,kw] - zp_w)
                        # else: padding (value 0 - zp_x)*w → cộng zp correction
            
            acc += B_int32[cout]  # bias add
            
            # Requant: M = scale_x * scale_w[cout] / scale_y
            M = scale_x * scale_w[cout] / scale_y
            # Hardware: M được biểu diễn bằng (M_int, shift) fixed-point
            # M_int ≈ round(M * 2^shift), chọn shift để M_int fit INT16/INT32
            
            y_raw = round(M * acc) + zp_y
            Y_int8[0,cout,h_out,w_out] = clamp(y_raw, -128, 127)
```

### 11.2. Zero-Point Correction (Optimization)

```python
# Với zp_w = 0 (symmetric weight):
# acc = Σ (x-zp_x)(w-0) = Σ x*w - zp_x * Σ w

# Tách: acc = Σ x*w - zp_x * partial_sum_w

# partial_sum_w[cout] = Σ_{cin,kh,kw} W_int8[cout,cin,kh,kw]
#   → precomputed offline, stored as INT32 constant

# Optimized hardware:
acc_raw = mac(X_int8, W_int8)  # INT8*INT8 → INT32, 16 lanes parallel
acc = acc_raw - zp_x * partial_sum_w[cout] + B_int32[cout]
```

---

## 12. Kết luận

Bảng mapping hoàn chỉnh Layer 0–22 → Primitive → RTL được tóm tắt:

| Layer | Block | Primitive decomposition | RTL chính |
|---|---|---|---|
| 0 | Conv(s2) | RS_DENSE_3x3(s=2) | pe_cluster, ppu_lite |
| 1 | Conv(s2) | RS_DENSE_3x3(s=2) | pe_cluster, ppu_lite |
| 2 | QC2f | OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1 | pe_cluster, router_cluster |
| 3 | Conv(s2) | RS_DENSE_3x3(s=2) | pe_cluster, ppu_lite |
| 4 | QC2f | OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1 | pe_cluster, router_cluster |
| 5 | SCDown | OS_1x1 + DW_3x3(s=2) | pe_cluster(dw), ppu_lite |
| 6 | QC2f | OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1 | pe_cluster, router_cluster |
| 7 | SCDown | OS_1x1 + DW_3x3(s=2) | pe_cluster(dw), ppu_lite |
| 8 | QC2f | OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1 | pe_cluster, router_cluster |
| 9 | SPPF | OS_1x1 + MAXPOOL_5x5×3 + CONCAT + OS_1x1 | pool_engine, router_cluster |
| 10 | QPSA | OS_1x1 + GEMM_ATTN_BASIC + OS_1x1 | gemm_attn_engine |
| 11 | Upsample | UPSAMPLE_NEAREST | tensor_post_engine |
| 12 | QConcat | CONCAT (skip: L6) | router_cluster, barrier_manager |
| 13 | QC2f | OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1 | pe_cluster, router_cluster |
| 14 | Upsample | UPSAMPLE_NEAREST | tensor_post_engine |
| 15 | QConcat | CONCAT (skip: L4) | router_cluster, barrier_manager |
| 16 | QC2f | OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1 | pe_cluster, router_cluster → **P3** |
| 17 | Conv(s2) | RS_DENSE_3x3(s=2) | pe_cluster, ppu_lite |
| 18 | QConcat | CONCAT (skip: L13) | router_cluster, barrier_manager |
| 19 | QC2f | OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1 | pe_cluster, router_cluster → **P4** |
| 20 | SCDown | OS_1x1 + DW_3x3(s=2) | pe_cluster(dw), ppu_lite |
| 21 | QConcat | CONCAT (skip: L8) | router_cluster, barrier_manager |
| 22 | QC2fCIB | OS_1x1 + DW_7x7_MULTIPASS + OS_1x1 + CONCAT | pe_cluster(dw_mp), ppu_lite → **P5** |

**Outputs cuối:**
- `P3_int8 = F16_out [1,64,80,80]`
- `P4_int8 = F19_out [1,128,40,40]`
- `P5_int8 = F22_out [1,256,20,20]`

---

*Tài liệu này là kết quả tổng hợp từ MODEL_BLOCKS_INT8_DETAIL.md, MODEL_FORWARD_FLOW.md, MODEL_LAYER_DEPENDENCIES.md và MODEL_LAYERS_INT8_FLOW.md.*
