# 02 – Layer Mapping (Freeze Spec)
## qYOLOv10n INT8 PTQ – Layer 0–22 → Primitive Decomposition

> **Trạng thái**: FREEZE | **Version**: v1.0 | **Ngày**: 2026-03-16

---

## 1. Mục Đích

Chốt mapping từng layer (0–22) của qYOLOv10n PTQ sang chuỗi primitive tương ứng. Đây là cầu nối trực tiếp giữa model và hardware. `model_forward_runner.py` và RTL `desc_fetch_engine` phải tuân thủ file này.

---

## 2. Bảng Trace Thực Tế (Đo Từ PyTorch)

```
Layer  Module     Input shape          Output shape         Dtype
0      Conv       [1, 3,  640, 640]    [1, 16, 320, 320]    quint8→quint8
1      Conv       [1, 16, 320, 320]    [1, 32, 160, 160]    quint8→quint8
2      QC2f       [1, 32, 160, 160]    [1, 32, 160, 160]    quint8→quint8
3      Conv       [1, 32, 160, 160]    [1, 64,  80,  80]    quint8→quint8
4      QC2f       [1, 64,  80,  80]    [1, 64,  80,  80]    quint8→quint8
5      SCDown     [1, 64,  80,  80]    [1,128,  40,  40]    quint8→quint8
6      QC2f       [1,128,  40,  40]    [1,128,  40,  40]    quint8→quint8
7      SCDown     [1,128,  40,  40]    [1,256,  20,  20]    quint8→quint8
8      QC2f       [1,256,  20,  20]    [1,256,  20,  20]    quint8→quint8
9      SPPF       [1,256,  20,  20]    [1,256,  20,  20]    quint8→quint8
10     QPSA       [1,256,  20,  20]    [1,256,  20,  20]    quint8→quint8
11     Upsample   [1,256,  20,  20]    [1,256,  40,  40]    quint8→quint8
12     QConcat    [1,256+128,40, 40]   [1,384,  40,  40]    quint8→quint8
13     QC2f       [1,384,  40,  40]    [1,128,  40,  40]    quint8→quint8
14     Upsample   [1,128,  40,  40]    [1,128,  80,  80]    quint8→quint8
15     QConcat    [1,128+64, 80, 80]   [1,192,  80,  80]    quint8→quint8
16     QC2f       [1,192,  80,  80]    [1, 64,  80,  80]    quint8→quint8  ← P3
17     Conv       [1, 64,  80,  80]    [1, 64,  40,  40]    quint8→quint8
18     QConcat    [1, 64+128,40, 40]   [1,192,  40,  40]    quint8→quint8
19     QC2f       [1,192,  40,  40]    [1,128,  40,  40]    quint8→quint8  ← P4
20     SCDown     [1,128,  40,  40]    [1,128,  20,  20]    quint8→quint8
21     QConcat    [1,128+256,20, 20]   [1,384,  20,  20]    quint8→quint8
22     QC2fCIB    [1,384,  20,  20]    [1,256,  20,  20]    quint8→quint8  ← P5
```

---

## 3. Layer → Primitive Mapping Chi Tiết

### BACKBONE – Đường Xuống (L0–L10)

---

#### L0 – Conv [3,640,640] → [16,320,320]

```
Block type  : Conv (Conv2d + BN_fuse + SiLU)
Primitive   : RS_DENSE_3x3(stride=2)
Sources     : X_int8 từ CPU (input quantize)
Hold output : No

Primitive params:
  Cin=3, Cout=16, H=640, W=640, stride=2, padding=1
  W_int8[16,3,3,3], B_int32[16]
  scale_in=1/255, zp_in=0
  scale_out=s_L0, zp_out=zp_L0

Output: F0 [1,16,320,320]
```

---

#### L1 – Conv [16,320,320] → [32,160,160]

```
Primitive   : RS_DENSE_3x3(stride=2)
Sources     : F0
Hold output : No

Primitive params:
  Cin=16, Cout=32, H=320, W=320, stride=2

Output: F1 [1,32,160,160]
```

---

#### L2 – QC2f [32,160,160] → [32,160,160]

```
Block type  : QC2f (n=1 bottleneck)
Sources     : F1
Hold output : No

Primitive sequence:
  Step 1: OS_1x1(Cin=32, Cout=32)              → X1 [1,32,160,160]
  Step 2: RS_DENSE_3x3(Cin=16, Cout=16, s=1)   → Ytmp [1,16,160,160]    ← half channel
  Step 3: RS_DENSE_3x3(Cin=16, Cout=16, s=1)   → Ybranch [1,16,160,160]
  Step 4: CONCAT([Ybranch, X1_upper_half])      → Ycat [1,32,160,160]
  Step 5: OS_1x1(Cin=32, Cout=32)              → F2 [1,32,160,160]

Note: X1 được split: nửa dưới vào bottleneck, nửa trên giữ lại cho CONCAT
      (theo kiến trúc C2f: channels = [c1//2, c2//2, ...]

Output: F2 [1,32,160,160]
```

---

#### L3 – Conv [32,160,160] → [64,80,80]

```
Primitive   : RS_DENSE_3x3(stride=2)
Sources     : F2
Hold output : No

Primitive params: Cin=32, Cout=64, stride=2

Output: F3 [1,64,80,80]
```

---

#### L4 – QC2f [64,80,80] → [64,80,80]

```
Block type  : QC2f (n=1)
Sources     : F3
Hold output : YES → hold đến L15 (QConcat)

Primitive sequence: OS_1x1 + RS_DENSE_3x3 + RS_DENSE_3x3 + CONCAT + OS_1x1
  (Tương tự L2, với Cin=Cout=64)

Output: F4 [1,64,80,80]
⚠️  HOLD_SKIP = True, hold_until = L15
```

---

#### L5 – SCDown [64,80,80] → [128,40,40]

```
Block type  : SCDown
Sources     : F4
Hold output : No

Primitive sequence (2 nhánh song song):
  Branch A:
    OS_1x1(Cin=64, Cout=64)      → tmpA [1,64,80,80]
    DW_3x3(C=64, stride=2)       → A_out [1,64,40,40]

  Branch B:
    OS_1x1(Cin=64, Cout=64)      → tmpB [1,64,80,80]
    DW_3x3(C=64, stride=2)       → B_out [1,64,40,40]

  CONCAT(A_out, B_out)           → F5 [1,128,40,40]

Note: Cin_total=64, Cout_total=128
      Mỗi nhánh xử lý 64→64 channels, sau CONCAT ra 128

Output: F5 [1,128,40,40]
```

---

#### L6 – QC2f [128,40,40] → [128,40,40]

```
Block type  : QC2f (n=1)
Sources     : F5
Hold output : YES → hold đến L12 (QConcat)

Primitive sequence: OS_1x1 + RS_DENSE_3x3 + RS_DENSE_3x3 + CONCAT + OS_1x1
  (Cin=Cout=128)

Output: F6 [1,128,40,40]
⚠️  HOLD_SKIP = True, hold_until = L12
```

---

#### L7 – SCDown [128,40,40] → [256,20,20]

```
Block type  : SCDown
Sources     : F6
Hold output : No

Primitive sequence: (tương tự L5 với Cin=128, Cout=256)
  Branch A: OS_1x1(128→128) → DW_3x3(s2) → [1,128,20,20]
  Branch B: OS_1x1(128→128) → DW_3x3(s2) → [1,128,20,20]
  CONCAT → F7 [1,256,20,20]

Output: F7 [1,256,20,20]
```

---

#### L8 – QC2f [256,20,20] → [256,20,20]

```
Block type  : QC2f (n=1)
Sources     : F7
Hold output : YES → hold đến L21 (QConcat) ← SKIP DÀI NHẤT

Primitive sequence: OS_1x1 + RS_DENSE_3x3 + RS_DENSE_3x3 + CONCAT + OS_1x1
  (Cin=Cout=256)

Output: F8 [1,256,20,20]
⚠️  HOLD_SKIP = True, hold_until = L21  (giữ qua 13 layer!)
```

---

#### L9 – SPPF [256,20,20] → [256,20,20]

```
Block type  : SPPF
Sources     : F8
Hold output : No

Primitive sequence:
  Step 1: OS_1x1(Cin=256, Cout=128)     → X1 [1,128,20,20]  (giảm kênh)
  Step 2: MAXPOOL_5x5(X1)               → P1 [1,128,20,20]
  Step 3: MAXPOOL_5x5(P1)               → P2 [1,128,20,20]
  Step 4: MAXPOOL_5x5(P2)               → P3 [1,128,20,20]
  Step 5: CONCAT(X1, P1, P2, P3)        → Ycat [1,512,20,20]
          (4 nhánh cùng qconfig → scale/zp tương đồng → concat đơn giản)
  Step 6: OS_1x1(Cin=512, Cout=256)     → F9 [1,256,20,20]

Note: X1 phải được buffer đồng thời với P1,P2,P3 trước khi CONCAT

Output: F9 [1,256,20,20]
```

---

#### L10 – QPSA [256,20,20] → [256,20,20]

```
Block type  : QPSA (Quantized Position Sensitive Attention)
Sources     : F9
Hold output : No

Primitive sequence:
  Step 1: OS_1x1(Cin=256, Cout=256)     → X_split, phân thành:
          X_attn [1,128,20,20]  (nhánh attention)
          X_pass [1,128,20,20]  (nhánh pass-through)

  Step 2: GEMM_ATTN_BASIC(X_attn)
          reshape [1,128,20,20] → [1,400,128]
          Q = OS_1x1_proj(X) → [1,400,64]
          K = OS_1x1_proj(X) → [1,400,64]
          V = OS_1x1_proj(X) → [1,400,128]
          Attn = Q×K^T/sqrt(64) → softmax_approx → ×V
          Y_attn = reshape → [1,128,20,20]

  Step 3: CONCAT(Y_attn, X_pass)        → Ymerge [1,256,20,20]
  Step 4: OS_1x1(Cin=256, Cout=256)     → F10 [1,256,20,20]

Output: F10 [1,256,20,20]
```

---

### NECK – FPN (L11–L16)

---

#### L11 – Upsample [256,20,20] → [256,40,40]

```
Primitive   : UPSAMPLE_NEAREST(scale=2)
Sources     : F10
Hold output : No

Y[c,2h,2w]=Y[c,2h,2w+1]=Y[c,2h+1,2w]=Y[c,2h+1,2w+1] = X[c,h,w]
scale/zp giữ nguyên từ F10

Output: F11 [1,256,40,40]
```

---

#### L12 – QConcat [256,40,40]+[128,40,40] → [384,40,40]

```
Primitive   : CONCAT
Sources     : [F11, F6]  ← SKIP DEPENDENCY: F6 từ L6
Hold output : No

Input A: F11 [1,256,40,40] (scale_A=scale_F11, zp_A=zp_F11) ← từ upsample
Input B: F6  [1,128,40,40] (scale_B=scale_F6,  zp_B=zp_F6)  ← từ backbone skip

Common-domain alignment:
  scale_Y, zp_Y được chọn offline từ PTQ (scale của QConcat output layer 12)
  Nếu scale_A ≠ scale_Y → requant F11 về scale_Y
  Nếu scale_B ≠ scale_Y → requant F6 về scale_Y
  CONCAT(F11_aligned, F6_aligned) theo chiều channel

BARRIER: L12 phải đợi BOTH L11_done AND F6_hold_ready

Output: F12 [1,384,40,40]
```

---

#### L13 – QC2f [384,40,40] → [128,40,40]

```
Block type  : QC2f (n=1)
Sources     : F12
Hold output : YES → hold đến L18

Primitive sequence: OS_1x1(384→192) + RS_DENSE_3x3 + CONCAT + OS_1x1(→128)
  (Cin=384, Cout=128 – thu hẹp channels)

Output: F13 [1,128,40,40]
⚠️  HOLD_SKIP = True, hold_until = L18
```

---

#### L14 – Upsample [128,40,40] → [128,80,80]

```
Primitive   : UPSAMPLE_NEAREST(scale=2)
Sources     : F13
Hold output : No

Output: F14 [1,128,80,80] (scale/zp từ F13)
```

---

#### L15 – QConcat [128,80,80]+[64,80,80] → [192,80,80]

```
Primitive   : CONCAT
Sources     : [F14, F4]  ← SKIP DEPENDENCY: F4 từ L4 (xa nhất, 11 layer)
Hold output : No

Input A: F14 [1,128,80,80] ← từ upsample
Input B: F4  [1, 64,80,80] ← từ backbone L4 skip

BARRIER: L15 phải đợi BOTH L14_done AND F4_hold_ready

Output: F15 [1,192,80,80]
```

---

#### L16 – QC2f [192,80,80] → [64,80,80]

```
Block type  : QC2f (n=1)
Sources     : F15
Hold output : No  ← ĐÂY LÀ OUTPUT P3

Primitive sequence: OS_1x1(192→96) + RS_DENSE_3x3 + CONCAT + OS_1x1(→64)
  (Cin=192, Cout=64 – spatial 80×80 LỚN NHẤT trong toàn model)

Output: F16 = P3_int8 [1,64,80,80]   ✅ P3 OUTPUT
```

---

### NECK – PAN (L17–L22)

---

#### L17 – Conv [64,80,80] → [64,40,40]

```
Primitive   : RS_DENSE_3x3(stride=2)
Sources     : F16 (= P3)
Hold output : No

Primitive params: Cin=64, Cout=64, stride=2, H=80→40

Output: F17 [1,64,40,40]
```

---

#### L18 – QConcat [64,40,40]+[128,40,40] → [192,40,40]

```
Primitive   : CONCAT
Sources     : [F17, F13]  ← SKIP DEPENDENCY: F13 từ L13
Hold output : No

Input A: F17 [1, 64,40,40] ← từ conv downsample P3
Input B: F13 [1,128,40,40] ← từ FPN mid L13 skip

BARRIER: L18 phải đợi BOTH L17_done AND F13_hold_ready

Output: F18 [1,192,40,40]
```

---

#### L19 – QC2f [192,40,40] → [128,40,40]

```
Block type  : QC2f (n=1)
Sources     : F18
Hold output : No  ← ĐÂY LÀ OUTPUT P4

Primitive sequence: OS_1x1(192→128) + RS_DENSE_3x3 + CONCAT + OS_1x1(→128)

Output: F19 = P4_int8 [1,128,40,40]   ✅ P4 OUTPUT
```

---

#### L20 – SCDown [128,40,40] → [128,20,20]

```
Block type  : SCDown
Sources     : F19 (= P4)
Hold output : No

Primitive sequence: (tương tự L5, Cin=Cout=128)
  Branch A: OS_1x1(128→64) → DW_3x3(s2) → [1,64,20,20]
  Branch B: OS_1x1(128→64) → DW_3x3(s2) → [1,64,20,20]
  CONCAT → F20 [1,128,20,20]

Output: F20 [1,128,20,20]
```

---

#### L21 – QConcat [128,20,20]+[256,20,20] → [384,20,20]

```
Primitive   : CONCAT
Sources     : [F20, F8]  ← SKIP DEPENDENCY: F8 từ L8 (dài nhất, 13 layer)
Hold output : No

Input A: F20 [1,128,20,20] ← từ SCDown
Input B: F8  [1,256,20,20] ← từ deep backbone L8 skip

BARRIER: L21 phải đợi BOTH L20_done AND F8_hold_ready

Output: F21 [1,384,20,20]
```

---

#### L22 – QC2fCIB [384,20,20] → [256,20,20]

```
Block type  : QC2fCIB (C2f with CIB – large kernel)
Sources     : F21
Hold output : No  ← ĐÂY LÀ OUTPUT P5

Primitive sequence:
  Step 1: OS_1x1(Cin=384, Cout=256)       → X1 [1,256,20,20]

  Step 2: CIB bottleneck (nhánh branch):
    DW_7x7_MULTIPASS(C=128)               → Y_dw [1,128,20,20]
    OS_1x1(Cin=128, Cout=128)             → Y_cib [1,128,20,20]

  Step 3: CONCAT(Y_cib, X1_split_half)    → Ycat [1,256,20,20]

  Step 4: OS_1x1(Cin=256, Cout=256)       → F22 [1,256,20,20]

Note: DW_7x7_MULTIPASS với 3 pass (rows 0-2, 3-5, 6)
      Trace PSUM sau mỗi pass bắt buộc

Output: F22 = P5_int8 [1,256,20,20]   ✅ P5 OUTPUT
```

---

## 4. Bốn Skip Dependencies Bắt Buộc

```
┌────────────────────────────────────────────────────────────────────┐
│  Skip    │ Source │ Hold from │ Destination │ Hold to │ Dist │ Size│
├────────────────────────────────────────────────────────────────────┤
│ SKIP-A   │  F4    │    L4     │   L15       │   L15   │  11  │400K│
│ SKIP-B   │  F6    │    L6     │   L12       │   L12   │   6  │205K│
│ SKIP-C   │  F8    │    L8     │   L21       │   L21   │  13  │102K│  ← Dài nhất
│ SKIP-D   │  F13   │   L13     │   L18       │   L18   │   5  │205K│
└────────────────────────────────────────────────────────────────────┘
Total GLB skip buffer: ~912 KB cần dự trữ đồng thời
```

### Đồ thị phụ thuộc đầy đủ:

```
Input X_int8
    │
    L0→L1→L2→L3→L4 ──────────────────────────────────────── SKIP-A ──► L15
                    │                                                      ▲
                    └─► L5→L6 ──────────────────────── SKIP-B ──► L12    │
                               │                                   ▲      │
                               └─► L7→L8 ──── SKIP-C ──► L21     │      │
                                           │              ▲       │      │
                                           └─► L9→L10→L11─────────┘      │
                                                          │               │
                                                         L12              │
                                                          │               │
                                                         L13 ── SKIP-D ──► L18
                                                          │               ▲
                                                         L14              │
                                                          │               │
                                                         L15              │
                                                          │               │
                P3 ◄────────────────────────────────── L16        L17 ───┘
                                                                    │
                                                                   L18
                                                                    │
                P4 ◄─────────────────────────────────────────── L19
                                                                    │
                                                                   L20
                                                                    │
                                                                   L21
                                                                    │
                P5 ◄─────────────────────────────────────────── L22
```

---

## 5. Barrier Logic (4 điểm đồng bộ)

```
barrier_L12:
  precondition: (L11 == DONE) AND (F6_hold == READY)
  action: release L12_compute_start
  error_case: timeout nếu F6 không được hold đúng

barrier_L15:
  precondition: (L14 == DONE) AND (F4_hold == READY)
  action: release L15_compute_start
  error_case: timeout nếu F4 không được hold đúng

barrier_L18:
  precondition: (L17 == DONE) AND (F13_hold == READY)
  action: release L18_compute_start
  error_case: timeout nếu F13 không được hold đúng

barrier_L21:
  precondition: (L20 == DONE) AND (F8_hold == READY)
  action: release L21_compute_start
  error_case: timeout nếu F8 không được hold đúng
```

---

## 6. Output Mapping

| Output | Layer | Shape | Role |
|---|---|---|---|
| P3_int8 | L16 | [1, 64, 80, 80] | Small object feature (1/8 scale) |
| P4_int8 | L19 | [1, 128, 40, 40] | Medium object feature (1/16 scale) |
| P5_int8 | L22 | [1, 256, 20, 20] | Large object feature (1/32 scale) |

Quant metadata đi kèm: `(scale_P3, zp_P3)`, `(scale_P4, zp_P4)`, `(scale_P5, zp_P5)`

---

## 7. Sign-off Checklist

```
☐ Layer count: 23 layers (L0–L22) confirmed từ trace
☐ Shape của mỗi layer match với trace thực tế
☐ Hold_skip flags đúng cho: L4, L6, L8, L13
☐ 4 barrier conditions được xác nhận
☐ 3 output layers xác nhận: L16=P3, L19=P4, L22=P5
☐ DW_7x7_MULTIPASS tại L22 được annotate với pass split
☐ SPPF tại L9: OS_1x1 + MAXPOOL×3 + CONCAT×4 + OS_1x1 được confirm
☐ QPSA tại L10: OS_1x1 + GEMM_ATTN_BASIC + CONCAT + OS_1x1 được confirm
☐ Total skip buffer ~912KB được accepted trong GLB capacity planning
```

*File này là nguồn chân lý cho layer-to-primitive mapping. Không được hardcode sequence trong runner.*
