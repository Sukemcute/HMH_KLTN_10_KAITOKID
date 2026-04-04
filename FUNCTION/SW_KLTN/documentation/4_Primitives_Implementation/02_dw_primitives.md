# P2 & P8 – Depthwise Convolution Primitives
## Source file: `PHASE1/python_golden/primitives/primitive_dw.py`

---

## Tổng quan

| Primitive | Tên | Kernel | Dùng trong |
|:----------|:----|:-------|:-----------|
| **P2** | `dw_3x3` | DW 3×3 | SCDown cv2 (stride=2), QPSA PE, QC2fCIB |
| **P8** | `dw_7x7_multipass` | DW 7×7 | QC2fCIB bên trong QCIB |
| *(util)* | `dw_conv_int` | DW bất kỳ | QCIB (generic fallback) |

**Điểm khác biệt cơ bản so với P0/P1:**
- `groups = Cin` → **mỗi channel xử lý độc lập**, không cross-channel reduce
- Weight shape: `[C, kH, kW]` (không có `Cin` dim vì groups=C)
- **Per-channel** scale_w, bias, M_int, shift (một bộ tham số riêng cho từng channel)

---

## Hàm lõi nội bộ: `_dw_conv_channel`

```python
def _dw_conv_channel(x_ch, w_ch, stride, pad, zp_x):
    """
    Depthwise conv cho 1 channel.
    x_ch : [H, W] int64 – single channel
    w_ch : [kH, kW] int64 – single channel kernel
    Returns: [Hout, Wout] int64 – raw MAC (trước zp correction)
    """
    if pad > 0:
        x_pad = np.pad(x_ch, ((pad,pad),(pad,pad)),
                       mode='constant', constant_values=int(zp_x))
    else:
        x_pad = x_ch

    Hout = (x_pad.shape[0] - kH) // stride + 1
    Wout = (x_pad.shape[1] - kW) // stride + 1
    acc = np.zeros((Hout, Wout), dtype=np.int64)

    for kh in range(kH):
        for kw in range(kW):
            x_slice = x_pad[kh:kh+Hout*stride:stride, kw:kw+Wout*stride:stride]
            acc += x_slice * int(w_ch[kh, kw])
    return acc
```

---

## P2 – `dw_3x3` (DW_3x3)

### Mục đích
Depthwise 3×3 conv với per-channel requant. Quan trọng trong:
- **SCDown**: stride=2 → downsampling không gian
- **QPSA positional encoding**: stride=1 → encode vị trí vào V tensor
- **QC2fCIB**: stride=1 → spatial processing bên trong QCIB

### Signature

```python
def dw_3x3(
    X_int8: np.ndarray,        # [N, C, H, W] int8
    W_int8_per_ch: np.ndarray, # [C, 3, 3] int8  ← weight shape khác P0!
    B_int32_per_ch: np.ndarray,# [C] int32
    scale_x: float,
    zp_x: int,
    scale_w_per_ch: np.ndarray,# [C] float – per-channel weight scale
    scale_y: float,
    zp_y: int,
    stride: int = 1,           # 1 hoặc 2
    activation: str = "none",  # "none" | "silu"
    dump: bool = False,
) -> tuple:                    # (Y_int8, scale_y, zp_y)
```

### Thuật toán per-channel (khác hoàn toàn P0):

```python
pad = 1  # kernel=3, same padding – luôn cố định

for c in range(C):
    # Zero-point correction per channel
    partial_sum_w_c = int(W_i64[c].sum())
    
    for n in range(N):
        raw_mac = _dw_conv_channel(X_i64[n,c], W_i64[c], stride, pad, zp_x)
        acc_all[n,c] = raw_mac - zp_x * partial_sum_w_c + int(B_int32_per_ch[c])

# Per-channel requant:
# M[c] = scale_x * scale_w[c] / scale_y
M_int_arr, shift_arr = make_requant_params(scale_x, scale_w, scale_y)
Y_int8 = post_process_int32_to_int8(acc_all, M_int_arr, shift_arr, zp_y)
```

### Điểm đặc biệt:
- `padding = 1` cố định (không nhận tham số padding)
- Mỗi channel `c` có `M_int[c]`, `shift[c]` riêng
- Weight: `W_int8_per_ch[c, kh, kw]` – không có Cin dimension

---

## P8 – `dw_7x7_multipass` (DW_7x7_MULTIPASS)

### Mục đích
Depthwise 7×7 conv cho **QC2fCIB QCIB block** (Layer 22). Kernel 7×7 chia làm **3 pass** để phù hợp line buffer phần cứng nhỏ:
- Pass 1: rows 0–2 (3 rows)
- Pass 2: rows 3–5 (3 rows)
- Pass 3: row 6 (1 row) → requant ở đây

### Signature

```python
def dw_7x7_multipass(
    X_int8: np.ndarray,        # [N, C, H, W] int8
    W_int8_per_ch: np.ndarray, # [C, 7, 7] int8
    B_int32_per_ch: np.ndarray,# [C] int32
    scale_x: float,
    zp_x: int,
    scale_w_per_ch: np.ndarray,# [C] float
    scale_y: float,
    zp_y: int,
    stride: int = 1,
    split: tuple = None,       # default (3,3,1) từ config.DW7x7_SPLIT
    activation: str = "none",
    dump: bool = False,        # True → trả thêm psum_traces dict
) -> tuple:                    # (Y_int8, scale_y, zp_y)
```

### Cơ chế multipass:

```python
split = (3, 3, 1)   # rows per pass: [0:3], [3:6], [6:7]
pad = 3             # kernel=7, same padding

PSUM = np.zeros((N, C, Hout, Wout), dtype=np.int64)  # INT64 accumulator

# Pass 1, 2: tích lũy PSUM, KHÔNG requant
for pass_idx in [0, 1]:
    for c in range(C):
        w_rows = W_i64[c, row_start:row_end, :]  # [n_rows, 7]
        for n in range(N):
            partial = _dw_conv_channel_partial(X_i64[n,c], w_rows, ...)
            PSUM[n, c] += partial

# Pass 3 (last_pass): + bias → requant → INT8
partial_sum_w_full = W_i64.sum(axis=(1,2))       # [C]
zp_correction = (zp_x * partial_sum_w_full).reshape(1,C,1,1)
PSUM_final = PSUM - zp_correction + bias

Y_int8 = post_process_int32_to_int8(PSUM_final, M_int_arr, shift_arr, zp_y)
```

### Hardware invariance: `multipass == monolithic`
Kết quả phải giống hệt `dw_conv_int` với kernel 7×7 (không chia pass). Đây là yêu cầu thiết kế quan trọng.

---

## Utility: `dw_conv_int` (Generic DW)

```python
def dw_conv_int(
    X_int8: np.ndarray,
    W_int8_per_ch: np.ndarray, # [C, kH, kW] int8 – bất kỳ kích thước
    B_int32_per_ch: np.ndarray,# [C] int32
    scale_x: float,
    zp_x: int,
    scale_w_per_ch: np.ndarray,# [C] float
    scale_y: float,
    zp_y: int,
    stride: int = 1,
    padding: int = 0,          # ← phải truyền tường minh!
    activation: str = "none",
) -> tuple:                    # (Y_int8, scale_y, zp_y)
```

**Dùng trong `block_qc2f_cib.py`:**
```python
# DW 3×3 với pad tường minh
y0, s0, z0 = dw_conv_int(X_int8, padding=1, **cv_params_list[0])
# DW 7×7 với pad tường minh
y2, s2, z2 = dw_conv_int(y1,     padding=3, **cv_params_list[2])
```

**Lưu ý quan trọng:** Nếu không truyền `padding=` thì mặc định `padding=0` → spatial shrink! Đây là bug đã được fix bằng cách truyền tường minh.

---

## Luồng dữ liệu P2 (SCDown)

```
X_int8 [N, C, H, W]  (scale_x, zp_x)
    │
    ├─ Pad mỗi channel với zp_x (pad=1)
    ├─ DW MAC per channel: Σ_{kh,kw} x[c,h,w] × w[c,kh,kw]  (int64)
    ├─ ZP correction per channel: -zp_x × Σ_w W[c] + B[c]
    ├─ Per-channel PPU: (acc×M_int[c] + offset) >> shift[c] + zp_y → clip
    └─ (Activation nếu cần)
    │
Y_int8 [N, C, Hout, Wout]  (scale_y, zp_y)
   Hout = (H+2*1-3)//stride+1
```

---

## Luồng dữ liệu P8 (QCIB trong QC2fCIB)

```
X_int8 [N, C, H, W]  (scale_x, zp_x)
    │
    ├─ PSUM = 0 (int64)
    ├─ Pass 1: PSUM += Σ_{rows 0-2} x×w
    ├─ Pass 2: PSUM += Σ_{rows 3-5} x×w
    ├─ Pass 3: PSUM += Σ_{row 6}   x×w
    │          PSUM_final = PSUM - zp_x×ΣW + B
    │          Y = PPU(PSUM_final)
    └─ (Activation)
    │
Y_int8 [N, C, H, W]  (scale_y, zp_y)  ← cùng H,W vì pad=3, stride=1
```

---

## Import chain

```
block_scdown.py → dw_3x3
block_qpsa.py   → dw_3x3
block_qc2f_cib.py → dw_conv_int (generic), dw_3x3
    └── primitive_dw.py
          └── quant_affine.py: make_requant_params, post_process_int32_to_int8,
                                apply_silu_float, apply_relu
```
