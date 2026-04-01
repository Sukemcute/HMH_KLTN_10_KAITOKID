# 01 – Primitive Matrix (Freeze Spec)
## qYOLOv10n INT8 Accelerator – Phase 0

> **Trạng thái**: FREEZE | **Version**: v1.0 | **Ngày**: 2026-03-16

---

## 1. Mục Đích

Chốt tập primitive chính thức của accelerator V1 để cover qYOLOv10n PTQ layer 0–22. Mọi RTL, Golden Python và test plan đều tham chiếu file này.

---

## 2. Bảng Primitive Chính Thức

| ID | Tên | Loại | Status | Layer dùng |
|---|---|---|---|---|
| P0 | RS_DENSE_3x3 | Conv 3×3 | **BẮT BUỘC** | L0,1,3,17; nội bộ QC2f |
| P1 | OS_1x1 | Conv 1×1 | **BẮT BUỘC** | Tất cả QC2f, SPPF, QPSA, SCDown |
| P2 | DW_3x3 | Depthwise 3×3 | **BẮT BUỘC** | SCDown L5,7,20; QC2fCIB |
| P3 | MAXPOOL_5x5 | Pooling | **BẮT BUỘC** | SPPF L9 (×3 lần) |
| P4 | MOVE | Tensor copy | **BẮT BUỘC** | Skip buffer HOLD |
| P5 | CONCAT | Tensor join | **BẮT BUỘC** | QConcat L12,15,18,21 |
| P6 | UPSAMPLE_NEAREST | Tensor upsample | **BẮT BUỘC** | L11, L14 |
| P7 | EWISE_ADD | Element-wise add | Nên giữ | Residual dự phòng |
| P8 | DW_7x7_MULTIPASS | Depthwise 7×7 | Nên giữ | QC2fCIB L22 |
| P9 | GEMM_ATTN_BASIC | Attention GEMM | Nên giữ | QPSA L10 |

---

## 3. Đặc Tả Chi Tiết

### P0 – RS_DENSE_3x3

| Trường | Giá trị |
|---|---|
| Input | [N, Cin, H, W] INT8 per-tensor (scale_x, zp_x) |
| Weight | [Cout, Cin, 3, 3] INT8 per-output-channel (scale_w[cout], zp_w=0) |
| Bias | [Cout] INT32, scale_bias[cout] = scale_x × scale_w[cout] |
| Output | [N, Cout, Hout, Wout] INT8 per-tensor (scale_y, zp_y) |
| Stride | {1, 2} |
| Padding | same: pad=1, Hout = ceil(H/stride) |
| PPU | YES: bias_add → requant → SiLU_LUT → clamp |
| PSUM | INT32 accumulator nội bộ |

**Golden Math**:
```
acc[cout,h,w] = Σ_{cin,kh,kw} X[cin,h*s+kh-1,w*s+kw-1] × W[cout,cin,kh,kw]
              - zp_x × Σ_{cin,kh,kw} W[cout,cin,kh,kw]   ← zp_correction (precomputed)
              + B[cout]

M[cout] = scale_x × scale_w[cout] / scale_y  → decompose: (M_int, shift)

y_raw = round(M_int × acc >> shift) + zp_y
y_int8 = SiLU_LUT[clamp(y_raw, -128, 127)]   (nếu có activation)
```

**Tests sign-off**: stride=1, stride=2, edge padding, Cin=128 Cout=256, zp_x≠0, random×100 vs PyTorch float conv (≤1 LSB)

---

### P1 – OS_1x1

| Trường | Giá trị |
|---|---|
| Input | [N, Cin, H, W] INT8 |
| Output | [N, Cout, H, W] INT8 (spatial giữ nguyên) |
| Stride | 1, Padding 0 |
| PPU | YES |

**Golden Math**: Như P0 với kh=kw=0 (kernel 1 element), không padding.

**Tests**: expand (Cin<Cout), compress (Cin>Cout), projection (Cin=Cout)

---

### P2 – DW_3x3

| Trường | Giá trị |
|---|---|
| Input | [N, C, H, W] INT8 |
| Output | [N, C, Hout, Wout] INT8 |
| Groups | C (mỗi channel độc lập) |
| Weight | [C, 1, 3, 3] INT8, per-channel: scale_w[c], zp_w[c]=0 |
| Bias | [C] INT32 per-channel |
| PPU | YES, per-channel requant |
| last_pass | LUÔN True (không cross-channel) |

**Golden Math**:
```
for c in C:
  acc[c,h,w] = Σ_{kh,kw} X[c,h*s+kh-1,w*s+kw-1] × W[c,0,kh,kw]
             - zp_x × Σ_{kh,kw} W[c,0,kh,kw] + B[c]
  y[c,h,w] = clamp(round(M[c] × acc) + zp_y, -128, 127)
  where M[c] = scale_x × scale_w[c] / scale_y
```

**Tests**: stride=1, stride=2, per-channel weight khác nhau giữa các channels

---

### P3 – MAXPOOL_5x5

| Trường | Giá trị |
|---|---|
| Kernel | 5×5, stride=1, padding=2 |
| Quant | **Pass-through**: scale_out=scale_in, zp_out=zp_in |
| PPU | KHÔNG – so sánh INT8 thuần |

**Golden Math**: `Y[c,h,w] = max(X[c, h+dh-2, w+dw-2] for dh,dw in 0..4)`

**Tests**: shape [1,128,20,20]→[1,128,20,20], lặp 3×, scale/zp unchanged

---

### P4 – MOVE

| Trường | Giá trị |
|---|---|
| Chức năng | Copy tensor, giữ nguyên (scale, zp) |
| PPU | KHÔNG |
| Use | HOLD_SKIP buffer, skip connection management |

---

### P5 – CONCAT

| Trường | Giá trị |
|---|---|
| Input | A[N,C_A,H,W](scale_A,zp_A) + B[N,C_B,H,W](scale_B,zp_B) |
| Output | [N,C_A+C_B,H,W](scale_Y,zp_Y) |
| Axis | Channel (dim=1) |
| Quant | Common-domain alignment nếu scale_A≠scale_B |

**Common-domain alignment**:
```
scale_Y, zp_Y = được chọn offline từ PTQ calibration

if scale_A ≠ scale_Y:
  A_float = (A_int8 - zp_A) × scale_A
  A_aligned = clamp(round(A_float/scale_Y) + zp_Y, -128, 127)

concat(A_aligned, B_aligned) theo chiều channel
```

**Tests**: same domain (no requant), diff domain (scale×0.5), 4-way concat (SPPF)

---

### P6 – UPSAMPLE_NEAREST

| Trường | Giá trị |
|---|---|
| Scale | ×2 spatial |
| Quant | Pass-through |
| Hardware | tensor_post_engine, address remapping |

**Golden Math**:
```
Y[n,c,2h,2w]=Y[n,c,2h,2w+1]=Y[n,c,2h+1,2w]=Y[n,c,2h+1,2w+1] = X[n,c,h,w]
```

---

### P7 – EWISE_ADD (Optional)

Common-domain alignment → INT16 add → requant → INT8.
Dự phòng residual. Chưa dùng trong L0–22 baseline.

---

### P8 – DW_7x7_MULTIPASS

| Trường | Giá trị |
|---|---|
| Pass split | 3-3-1 (rows 0-2, 3-5, 6) |
| PSUM | INT32, giữ qua 2 pass đầu |
| Bias+PPU | Chỉ tại last_pass (pass 3) |
| Invariant | **Output == monolithic DW_7x7** |

**Thuật toán**:
```
PSUM = 0
Pass 1 (kh=0,1,2): PSUM += Σ_{kh in [0,1,2], kw} X×W; last_pass=False
Pass 2 (kh=3,4,5): PSUM += Σ_{kh in [3,4,5], kw} X×W; last_pass=False
Pass 3 (kh=6):     PSUM += Σ_{kh=6, kw} X×W
                   PSUM -= zp_correction_full; PSUM += B_int32
                   Y = clamp(round(M×PSUM)+zp_y); last_pass=True
```

**Trace bắt buộc**: PSUM sau mỗi pass để RTL debug.

---

### P9 – GEMM_ATTN_BASIC

```
Input: [N,C,H,W] → reshape [N,HW,C] (HW=400 tại 20×20)
Q=OS_1x1(X), K=OS_1x1(X), V=OS_1x1(X)
Attn = Q×K^T → requant → softmax_approx → ×V → requant → reshape → OS_1x1
Output: [N,C,H,W]
```

---

## 4. Constraints Bất Biến

1. INT8 activation range: [-128, 127]
2. Weight zero-point: zp_w = 0 (symmetric)
3. PSUM width: INT32 (min 24-bit logic)
4. Requant path duy nhất: `y = clamp(round(M_int × acc >> shift) + zp_out)`
5. UPSAMPLE/MAXPOOL: scale_out = scale_in, zp_out = zp_in (không đổi)
6. scale > 0 tại mọi điểm

---

## 5. Sign-off Checklist

```
☐ P0 RS_DENSE_3x3 : test pass (stride 1/2, padding, multi-channel, random×100)
☐ P1 OS_1x1       : test pass (expand, compress, projection)
☐ P2 DW_3x3       : test pass (stride 1/2, per-channel)
☐ P3 MAXPOOL_5x5  : test pass (shape, value, 3×loop)
☐ P4 MOVE         : test pass (copy, metadata identical)
☐ P5 CONCAT       : test pass (same domain, diff domain, 4-way)
☐ P6 UPSAMPLE     : test pass (shape×2, content correct, metadata unchanged)
☐ P7 EWISE_ADD    : test pass (basic, saturation, domain mismatch)
☐ P8 DW_7x7       : test pass (BẮT BUỘC multipass == monolithic)
☐ P9 GEMM_ATTN    : test pass (shape, deterministic)
```

*Nguồn chân lý cho primitive set. Mọi thay đổi cần sign-off trước khi áp dụng.*
