# P12, P13, P14 – Post-Processing Unit (PPU) Primitives
## Source file: `PHASE1/python_golden/quant/quant_affine.py`

---

## Tổng quan

PPU là tầng xử lý **sau** MAC accumulation, chuyển đổi từ INT32/INT64 về INT8. Mọi primitive convolution đều phải đi qua PPU.

| Primitive | Tên | Hàm | Dùng trong |
|:----------|:----|:----|:-----------|
| **P12** | REQUANT (PPU) | `post_process_int32_to_int8` | Sau mỗi convolution |
| **P13** | SiLU_LUT | `apply_silu_float` / `apply_silu_lut` | Conv có SiLU activation |
| **P14** | ReLU/ReLU6 | `apply_relu` / clip logic | Conv có ReLU activation |

---

## Hàm tiện ích: `quantize_affine` và `dequantize_affine`

```python
def quantize_affine(x_float, scale, zp, dtype="int8"):
    """
    Float → INT8
    x_int = clamp(round(x_float / scale) + zp, min, max)
    """
    x_scaled = _round(np.asarray(x_float, dtype=np.float64) / scale)
    x_int = np.clip(x_scaled + zp, INT8_MIN, INT8_MAX).astype(np.int8)
    return x_int

def dequantize_affine(x_int, scale, zp):
    """
    INT8 → float64
    x_float = (x_int - zp) * scale
    """
    return (x_int.astype(np.float64) - zp) * float(scale)
```

### Hàm rounding nội bộ:

```python
def _round(x):
    """Rounding theo ROUNDING_MODE trong config.py"""
    if ROUNDING_MODE == "half_up":
        return np.floor(np.asarray(x, dtype=np.float64) + 0.5).astype(np.int64)
    else:  # "half_even" – banker's rounding (mặc định)
        return np.round(np.asarray(x, dtype=np.float64)).astype(np.int64)
```

**Lưu ý:** `ROUNDING_MODE = "half_even"` (mặc định). Đây là **Banker's Rounding** – giống PyTorch, khác với `floor` của hardware naive.

---

## Hàm helper: `make_requant_params`

```python
def make_requant_params(scale_in, scale_w_per_ch, scale_out):
    """
    Tính per-channel (M_int, shift) cho PPU.
    M[cout] = scale_in * scale_w[cout] / scale_out
    → decompose thành (M_int, shift): M ≈ M_int / 2^shift
    """
    M = scale_in * np.asarray(scale_w_per_ch, dtype=np.float64) / scale_out

    M_int_list, shift_list = [], []
    for m in M.flat:
        # Chọn shift sao cho M_int ≈ 2^30 (precision tốt nhất)
        log2_M = math.log2(m)
        shift = int(math.floor(30 - log2_M))
        shift = max(0, min(shift, 31))
        M_int = int(round(m * (2**shift)))
        M_int = min(max(M_int, 0), INT32_MAX)
        M_int_list.append(M_int)
        shift_list.append(shift)

    return (np.array(M_int_list, dtype=np.int64),
            np.array(shift_list, dtype=np.int32))
```

---

## P12 – `post_process_int32_to_int8` (REQUANT / PPU)

### Mục đích
Chuyển INT32/INT64 accumulator → INT8 bằng fixed-point multiply và shift. Đây là **PPU logic** của hardware.

### Signature

```python
def post_process_int32_to_int8(
    acc_int32: np.ndarray,  # [N,Cout,H,W] hoặc [N,Cout,L] hoặc [Cout] – INT64
    M_int: np.ndarray,      # [Cout] int64 – per-channel multipliers
    shift: np.ndarray,      # [Cout] int32 – per-channel shifts
    zp_out: int,            # output zero-point
) -> np.ndarray:            # same shape as acc_int32, dtype=int8
```

### Thuật toán (hardware-faithful):

```python
# Per-channel:
for c in range(Cout):
    sh = int(shift_arr[c])
    offset = (1 << (sh - 1)) if sh > 0 else 0    # rounding offset!

    # 4D case [N, Cout, H, W]:
    y_raw[:, c, :, :] = (acc[:, c, :, :] * M_int_arr[c] + offset) >> sh

# Final saturation
y_int32 = y_raw.astype(np.int32) + zp_out
y_int8 = np.clip(y_int32, INT8_MIN, INT8_MAX).astype(np.int8)
```

### Rounding offset quan trọng:
```
offset = (1 << (shift-1)) = 2^(shift-1)
```
Thêm `offset` trước `>> shift` → biến phép **floor** thành **round-to-nearest**.  
Không có offset: `(x >> shift)` = floor → sai ~1 LSB ở 30% pixels.

### Hỗ trợ nhiều shape:
- `ndim == 1`: `[Cout]` – 1D case
- `ndim == 2`: `[N, Cout]` – batch 1D
- `ndim == 3`: `[N, Cout, L]` – dùng cho GEMM attention
- `ndim == 4`: `[N, Cout, H, W]` – convolution output

---

## P13 – SiLU Activation

### Phương thức 1: `apply_silu_float` (Golden path – accuracy first)

```python
def apply_silu_float(y_int8, scale_y, zp_y):
    """
    Dequant → SiLU float → requant.
    SiLU(x) = x × sigmoid(x) = x / (1 + exp(-x))
    """
    y_float = dequantize_affine(y_int8, scale_y, zp_y)
    silu_float = y_float * (1.0 / (1.0 + np.exp(-y_float)))
    return quantize_affine(silu_float, scale_y, zp_y, dtype="int8")
```

**Dùng trong:** `_conv_requant_act` khi `activation == ACT_SILU`

### Phương thức 2: `apply_silu_lut` (Hardware-faithful)

```python
def build_silu_lut(scale_y, zp_y):
    """
    Xây 256-entry LUT. Index i → INT8 value (i - 128).
    LUT[i] = quantize(SiLU(dequantize(i-128, scale_y, zp_y)), scale_y, zp_y)
    """
    lut = np.zeros(256, dtype=np.int8)
    for i in range(256):        # i: 0..255
        int8_val = i - 128      # -128..127
        float_val = (int8_val - zp_y) * scale_y
        silu_val = float_val * (1.0 / (1.0 + math.exp(-float_val)))
        lut[i] = quantize_affine(np.array([silu_val]), scale_y, zp_y, dtype="int8")[0]
    return lut

def apply_silu_lut(y_int8, lut):
    """Tra LUT: idx = value + 128"""
    idx = (y_int8.astype(np.int32) + 128).clip(0, 255).astype(np.uint8)
    return lut[idx].astype(np.int8)
```

---

## P14 – ReLU / ReLU6 Activation

### `apply_relu`

```python
def apply_relu(Y_int8, zp_y):
    """
    ReLU trong INT8 domain:
    Y = max(Y, zp_y)
    (vì zp_y = quantize(0.0) → giá trị đại diện cho float 0)
    """
    return np.maximum(Y_int8, zp_y).astype(np.int8)
```

### ReLU trong `_conv_requant_act`:

```python
elif activation == ACT_RELU:
    float_zero = quantize_affine(np.array([0.0]), scale_y, zp_y, "int8")[0]
    y_out = np.clip(y_pre_act, float_zero, INT8_MAX).astype(np.int8)

elif activation == ACT_RELU6:
    float_zero = quantize_affine(np.array([0.0]), scale_y, zp_y, "int8")[0]
    float_six  = quantize_affine(np.array([6.0]), scale_y, zp_y, "int8")[0]
    y_out = np.clip(y_pre_act, float_zero, float_six).astype(np.int8)
```

**Lưu ý:** Trong Conv blocks của qYOLOv10n, activation mặc định thường là **ReLU** (không phải SiLU). SiLU chỉ dùng ở một số layers đặc biệt.

---

## Signed vs Unsigned Domain: `-128 shift`

### Vấn đề cốt lõi:
- PyTorch: `uint8` (0..255), zero_point ∈ [0,255]
- Hardware: `int8` (-128..127), zero_point ∈ [-128,127]

### Giải pháp (áp dụng toàn bộ codebase):

```python
# Khi extract params từ PyTorch model:
zp_signed = int(pytorch_module.zero_point) - 128

# Khi convert tensor:
X_signed = X_uint8.astype(np.int16) - 128  # uint8 → int8 equivalent
```

### Proof of invariance:
$$(X_s - ZP_s) = (X_u - 128) - (ZP_u - 128) = X_u - ZP_u$$

→ Toán học float hoàn toàn bất biến dưới phép shift này.

---

## Fixed-Point Decomposition: `_fixed_point_decompose_scalar`

```python
def _fixed_point_decompose_scalar(M):
    """
    M → (M_int, shift) sao cho M ≈ M_int / 2^shift
    Target: M_int ≈ 2^30 (maximum precision trong INT32)
    """
    log2_M = math.log2(M)
    shift = int(math.floor(30 - log2_M))  # shift để M_int ≈ 2^30
    shift = max(0, min(shift, 31))         # clamp [0, 31]
    M_int = int(round(M * (2**shift)))
    M_int = min(max(M_int, 0), INT32_MAX) # clamp INT32
    return M_int, shift
```

**Ví dụ:** M = 0.25
- log2(0.25) = -2
- shift = floor(30 - (-2)) = 32 → clamp = 31
- M_int = round(0.25 × 2^31) = round(536870912) = 536870912

---

## Luồng PPU hoàn chỉnh (P12 → P13/P14):

```
acc_int64 [N, Cout, H, W]   (sau MAC + bias correction)
    │
    ├─[P12] Per-channel: (acc × M_int[c] + 2^(shift-1)) >> shift[c] + zp_y
    ├─      Clamp [-128, 127] → y_pre_act (int8)
    │
    ├─[P14] ReLU: max(y_pre_act, quantize(0.0))         [nếu activation=relu]
    │             clip(y_pre_act, q(0), q(6.0))         [nếu relu6]
    │
    └─[P13] SiLU: dequant → x×sigmoid(x) → requant     [nếu activation=silu]
    │
Output INT8 [N, Cout, H, W]
```

---

## Import chain

```
primitive_conv.py
primitive_dw.py
primitive_psa.py
  └── quant_affine.py:
        make_requant_params()
        post_process_int32_to_int8()   ← P12
        apply_silu_float()             ← P13 (golden path)
        apply_silu_lut()               ← P13 (HW path)
        build_silu_lut()               ← P13 LUT builder
        apply_relu()                   ← P14
        quantize_affine()
        dequantize_affine()
          └── config.py: INT8_MIN, INT8_MAX, ROUNDING_MODE
```
