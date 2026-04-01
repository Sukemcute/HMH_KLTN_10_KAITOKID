# 03 – Quantization Policy (Freeze Spec)
## qYOLOv10n INT8 PTQ – Quant Policy cho Model-Forward

> **Trạng thái**: FREEZE | **Version**: v1.0 | **Ngày**: 2026-03-16

---

## 1. Mục Đích

Chốt toàn bộ chính sách quantization: cách biểu diễn số, cách requant, cách xử lý CONCAT/ADD domain. Đây là file quan trọng nhất để tránh sai ở neck.

---

## 2. Quantization Scheme Cơ Bản

### 2.1. Affine Quantization Formula

```
Quantize:    x_int = clamp(round(x_float / scale) + zp, min_val, max_val)
Dequantize:  x_float = (x_int - zp) * scale

Trong đó:
  scale > 0   (float32)
  zp          (int32)
  min_val = -128, max_val = 127  (INT8 signed)
```

### 2.2. Bảng Policy Theo Đối Tượng

| Đối tượng | Scheme | Granularity | ZP |
|---|---|---|---|
| Activation input | INT8 | Per-tensor (1 scale, 1 zp cho toàn tensor) | Tự do |
| Activation output | INT8 | Per-tensor | Tự do |
| Weight conv/dw | INT8 | Per-output-channel (1 scale cho mỗi Cout) | **zp_w = 0** (symmetric) |
| Bias | INT32 | Per-output-channel | Không áp dụng |
| PSUM accumulator | INT32 | Nội bộ hardware | N/A |

---

## 3. Bias Fusion (BN Offline)

Batch Normalization được fuse vào Conv weight và bias trước khi deploy:

```
Fused weight: W_fused[cout, cin, kh, kw] = W_orig * (gamma / sqrt(var + eps))
Fused bias:   B_fused[cout] = beta - gamma * mean / sqrt(var + eps)

Sau khi quantize weight → W_int8, bias được scale thành:
B_int32[cout] = round(B_fused[cout] / (scale_x * scale_w[cout]))

Trong đó:
  scale_x     = scale của activation input
  scale_w[cout] = scale của weight channel cout
  B_int32 là INT32, không bị quantize cắt precision
```

---

## 4. Requant (Integer-to-Integer Rescaling)

### 4.1. Công thức chuẩn

Dùng xuyên suốt cho tất cả primitive:

```
M[cout] = scale_x * scale_w[cout] / scale_y   (float32, tính offline)

Decompose M thành fixed-point:
  M_int = round(M * 2^shift)   (chọn shift để M_int ∈ [2^15, 2^16-1])
  
Tại runtime (hardware):
  y_raw = (acc_int32 * M_int) >> shift   (arithmetic right shift)
  y_clamped = clamp(y_raw + zp_y, -128, 127)
```

### 4.2. Quy tắc cứng (không được vi phạm)

- **Chỉ một đường requant**: Không tự implement lại outside `quant_affine.py`
- **Không dùng float dequantize trong execution path**: Float chỉ cho reference test
- **Shift range**: shift ∈ [0, 31], M_int phải fit INT32
- **Rounding**: `round()` là round-half-to-even (banker's rounding) hoặc round-half-up (chọn nhất quán và khóa)

---

## 5. Activation Policy

### 5.1. SiLU (Sigmoid Linear Unit)

```
SiLU(x) = x × sigmoid(x) = x / (1 + exp(-x))

Phương án hardware: LUT 256 entry (precomputed)
  LUT_index = y_int8_pre_act + 128    (shift về [0,255] range)
  y_silu = SiLU_LUT[LUT_index]

LUT được tính offline từ float SiLU:
  for i in range(-128, 128):
    x_float = (i - zp_pre_act) * scale_pre_act
    y_float  = x_float * sigmoid(x_float)
    LUT[i+128] = clamp(round(y_float / scale_y + zp_y), -128, 127)

Ràng buộc: LUT phải nhất quán giữa Golden Python và RTL (same rounding policy).
```

### 5.2. Không có activation (identity)

Một số OS_1x1 không có activation (trong bottleneck internal). Flag `activation=None`.

---

## 6. Policy cho Tensor Operations

### 6.1. UPSAMPLE_NEAREST

```
scale_out = scale_in    ← KHÔNG ĐỔI
zp_out    = zp_in       ← KHÔNG ĐỔI
Dữ liệu INT8 chỉ copy, không có arithmetic
```

### 6.2. MOVE (Skip buffer copy)

```
scale_out = scale_in    ← KHÔNG ĐỔI
zp_out    = zp_in       ← KHÔNG ĐỔI
```

### 6.3. MAXPOOL_5x5

```
scale_out = scale_in    ← KHÔNG ĐỔI
zp_out    = zp_in       ← KHÔNG ĐỔI
Max comparison là INT8 comparison: đúng về mặt số học vì scale/zp chung.
```

### 6.4. CONCAT – Common-Domain Requant

**Đây là phần quan trọng nhất, rủi ro số 1 của neck.**

```
Inputs: A_int8 (scale_A, zp_A), B_int8 (scale_B, zp_B)
Output: Y_int8 (scale_Y, zp_Y)

scale_Y, zp_Y: được xác định bởi PTQ calibration của layer QConcat.
               (KHÔNG tự chọn; lấy từ calibrated model)

Bước 1: Requant A nếu cần
  if |scale_A - scale_Y| > epsilon OR zp_A ≠ zp_Y:
    A_float = (A_int8.float() - zp_A) * scale_A
    A_aligned = clamp(round(A_float / scale_Y) + zp_Y, -128, 127).int8()
  else:
    A_aligned = A_int8  (no requant, pass-through)

Bước 2: Requant B nếu cần (tương tự)

Bước 3: Concat theo channel
  Y_int8 = numpy.concatenate([A_aligned, B_aligned], axis=channel)
  Y có quant params (scale_Y, zp_Y)
```

**Áp dụng tại 4 QConcat layers:**

| Layer | Input A | Input B | Ghi chú |
|---|---|---|---|
| L12 | F11 (từ upsample path) | F6 (backbone skip) | Risk cao – path dài |
| L15 | F14 (từ upsample path) | F4 (backbone skip SKIP-A) | Risk cao nhất – skip 11 layer |
| L18 | F17 (từ PAN down path) | F13 (FPN mid skip) | Risk trung bình |
| L21 | F20 (từ SCDown path) | F8 (deep backbone SKIP-C) | Risk cao – skip 13 layer |

**Lý do domain mismatch nguy hiểm**: F4, F6, F8 và F13 đi qua nhiều conv → scale drift xa; trong khi nhánh upsample/downsample đi qua ít conv hơn. Khi concat mà không có alignment → outlier artifact, object detection sai.

### 6.5. EWISE_ADD – Common-Domain Add

```
Inputs: A_int8 (scale_A, zp_A), B_int8 (scale_B, zp_B)
Output: Y_int8 (scale_Y, zp_Y)   ← calibrated

Bước 1: Align cả A và B về scale_Y, zp_Y (như CONCAT)
Bước 2: Add với intermediate INT16 để tránh overflow:
  sum_int16 = int16(A_aligned - zp_Y) + int16(B_aligned - zp_Y)
Bước 3: Requant về output:
  Y_pre = round(sum_int16 * scale_Y / scale_Y_out) + zp_Y_out
  Y = clamp(Y_pre, -128, 127)

Saturation rule: values > 127 → 127; values < -128 → -128 (hard clamp)
```

---

## 7. PSUM Mode và Last-Pass Policy

```
if NOT last_pass:
  output_namespace = PSUM_BUFFER  (INT32, accumulate phase)
  PPU không kích hoạt
  Không write ra GLB_OUTPUT (INT8)

if last_pass:
  output_namespace = ACT_BUFFER  (INT8, sau PPU)
  PPU kích hoạt:
    1. bias_add  : PSUM + B_int32[cout]
    2. requant   : (PSUM * M_int) >> shift + zp_out
    3. activation: SiLU_LUT (nếu có)
    4. clamp     : [-128, 127]
  Write ra GLB_OUTPUT

last_pass = last_cin AND last_kernel AND last_reduce
  last_cin    : đây là Cin chunk cuối (channel reduction hoàn tất)
  last_kernel : đây là kernel position cuối (spatial reduction hoàn tất)
  last_reduce : đây là reduce operation cuối trong tile
```

---

## 8. Quy Tắc Absolute (Không Được Cưỡng Lại)

```
Rule Q1: ZP của weight = 0 (symmetric quantization)
         → zp_w[cout] = 0 cho tất cả P0,P1,P2,P8

Rule Q2: Float dequantize chỉ cho reference / debug
         → Execution path chỉ dùng integer arithmetic

Rule Q3: Một implementation requant duy nhất
         → Tất cả file Python và RTL đều import/instantiate từ cùng module

Rule Q4: scale_Y trong CONCAT/ADD lấy từ PTQ calibration
         → Không tự chọn scale; không dùng max(scale_A, scale_B) trong production

Rule Q5: CONCAT/ADD PHẢI kiểm tra domain trước khi kết hợp
         → Dù scale khác 1e-6 cũng phải align

Rule Q6: Rounding policy nhất quán: round-half-up (away from zero)
         → Áp dụng giống nhau trong Golden Python và RTL
```

---

## 9. Ví Dụ Số Cụ Thể

### Layer 0: Conv(s=2), scale_x=1/255=0.003921, zp_x=0

```
X_int8 giá trị pixel=128 → x_float=0.502 (≈128/255)

weight W_int8[0,0,1,1]=64, scale_w[0]=0.001, zp_w=0
→ w_float = 64 * 0.001 = 0.064

acc_raw = Σ x_int8 * w_int8 (ví dụ: 1 entry = 128*64 = 8192)
zp_correction = 0 * Σ W_int8 = 0 (vì zp_x = 0)
B_int32[0] = 150 (giả định)

acc = 8192 + 150 = 8342  (cộng dồn tất cả kernel positions)

scale_y[0] = 0.025 (calibrated)
M[0] = 0.003921 * 0.001 / 0.025 = 0.0001568
M_int = round(0.0001568 * 2^23) = round(1314) = 1314, shift=23

y_raw = (8342 * 1314) >> 23 = 10961388 >> 23 = 1 (nếu acc nhỏ thế)
→ Thực tế acc lớn hơn nhiều sau toàn kernel sum

y_int8 = clamp(y_raw + zp_y, -128, 127)
```

### CONCAT tại L12: domain alignment

```
F11 (Upsample output): scale_A=0.05, zp_A=0
F6  (Backbone skip):   scale_B=0.09, zp_B=2
scale_Y (calibrated L12 QConcat output) = 0.07, zp_Y = 0

Requant F11 (scale_A=0.05 ≠ scale_Y=0.07):
  x_float = (F11_int8 - 0) * 0.05
  A_aligned = clamp(round(x_float / 0.07) + 0, -128, 127)
  → Ví dụ: F11_int8=100 → x_float=5.0 → A_aligned=round(5.0/0.07)=round(71.4)=71

Requant F6 (scale_B=0.09 ≠ scale_Y=0.07):
  x_float = (F6_int8 - 2) * 0.09
  B_aligned = clamp(round(x_float / 0.07) + 0, -128, 127)
  → Ví dụ: F6_int8=50 → x_float=(50-2)*0.09=4.32 → B_aligned=round(4.32/0.07)=round(61.7)=62

CONCAT([A_aligned[256ch], B_aligned[128ch]]) → Y[384ch], scale_Y=0.07, zp_Y=0
```

---

## 10. Sign-off Checklist

```
☐ Activation quantization: INT8 per-tensor confirmed
☐ Weight quantization: INT8 per-output-channel, zp_w=0 confirmed
☐ Bias quantization: INT32 per-channel, B=round(b_fused/(s_x * s_w)) confirmed
☐ Requant formula khóa: (acc*M_int)>>shift + zp_y, rounding=round-half-up
☐ SiLU LUT: 256 entries, offline precomputed, nhất quán Python ↔ RTL
☐ UPSAMPLE/MAXPOOL/MOVE: scale/zp pass-through confirmed (không đổi)
☐ CONCAT Policy: common-domain từ PTQ calibration, requant trước concat
☐ EWISE_ADD Policy: common-domain + INT16 intermediate để tránh overflow
☐ PSUM/ACT: PPU chỉ kích hoạt tại last_pass confirmed
☐ Tất cả ví dụ số được verify bằng tay hoặc Python script
```

*Mọi thay đổi quant policy sau file này phải được lượng hóa impact và sign-off lại.*
