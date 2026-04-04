# P0 & P1 – Convolution Primitives
## Source file: `PHASE1/python_golden/primitives/primitive_conv.py`

---

## Tổng quan

Hai primitives P0 và P1 xử lý **toàn bộ phép tính convolution** trong mô hình. Chúng được dùng bởi **mọi block** (Conv, QC2f, SCDown, SPPF, QPSA, QC2fCIB).

| Primitive | Tên | Kernel | Dùng trong |
|:----------|:----|:-------|:-----------|
| **P0** | `rs_dense_3x3` | 3×3 | Conv (L0,1,3,17), QBottleneck bên trong QC2f |
| **P1** | `os_1x1` | 1×1 | cv1/cv2 của QC2f, SPPF, SCDown, QPSA, QC2fCIB |

Cả hai đều gọi chung hàm lõi `_conv_requant_act()`.

---

## Hàm lõi nội bộ: `_conv2d_int`

```python
def _conv2d_int(X_pad, W, stride):
    """
    2D cross-correlation trên input đã PAD sẵn.
    X_pad : [N, Cin, H_pad, W_pad] int64
    W     : [Cout, Cin, kH, kW]   int64
    stride: int
    Returns: [N, Cout, Hout, Wout] int64
    """
    N, Cin, H_pad, W_pad = X_pad.shape
    Cout, _, kH, kW = W.shape
    Hout = (H_pad - kH) // stride + 1
    Wout = (W_pad - kW) // stride + 1
    Y = np.zeros((N, Cout, Hout, Wout), dtype=np.int64)

    for kh in range(kH):
        for kw in range(kW):
            X_slice = X_pad[:, :, kh:kh + Hout*stride:stride,
                                  kw:kw + Wout*stride:stride]
            W_kh_kw = W[:, :, kh, kw].astype(np.int64)
            Y += np.tensordot(W_kh_kw, X_slice.astype(np.int64),
                              axes=([1], [1])).transpose(1, 0, 2, 3)
    return Y
```

**Lưu ý:** Dùng `int64` để tránh overflow ở layer sâu (256+ channels).

---

## Hàm lõi: `_conv_requant_act`

Đây là **engine chính** – thực hiện đầy đủ 5 bước pipeline conv INT8:

```python
def _conv_requant_act(
    X_int8,     # [N, Cin, H, W] int8 – input
    W_int8,     # [Cout, Cin, kH, kW] int8 – weight
    B_int32,    # [Cout] int32 – bias (đã fuse BN)
    scale_x,    # float – scale input
    zp_x,       # int   – zero-point input  (signed, đã -128)
    scale_w,    # [Cout] float – per-channel weight scale
    zp_w,       # int   – weight zero-point (PHẢI = 0, symmetric)
    scale_y,    # float – scale output
    zp_y,       # int   – zero-point output (signed)
    stride,     # int
    kernel,     # int (3 hoặc 1)
    padding,    # int
    activation, # ACT_SILU | ACT_RELU | ACT_RELU6 | ACT_NONE
) -> np.ndarray:
```

### Pipeline 5 bước (chính xác theo hardware spec):

```
Bước 1: Pad input với zp_x (không phải 0!)
        X_pad = pad(X_int8, constant=zp_x, pad=padding)

Bước 2: Raw MAC (int64 accumulation)
        acc_raw = Σ_{cin,kh,kw} X_pad[n,cin,...] × W[cout,cin,kh,kw]

Bước 3: Zero-point correction + bias
        partial_sum_w[cout] = Σ_{cin,kh,kw} W[cout,cin,kh,kw]
        acc = acc_raw - zp_x × partial_sum_w + B_int32

Bước 4: Per-channel requant (PPU)
        M[cout] = scale_x × scale_w[cout] / scale_y
        y_raw = (acc × M_int + (1 << (shift-1))) >> shift
        y_pre_act = clamp(y_raw + zp_y, -128, 127)

Bước 5: Activation
        SILU  → apply_silu_float(y_pre_act, scale_y, zp_y)
        ReLU  → clip(y_pre_act, quantize(0.0), 127)
        ReLU6 → clip(y_pre_act, quantize(0.0), quantize(6.0))
        None  → pass-through
```

### Code thực hiện Bước 3 + 4:

```python
# Bước 3
partial_sum_w = W_i64.sum(axis=(1, 2, 3))          # [Cout]
zp_correction = (zp_x * partial_sum_w).reshape(1, Cout, 1, 1)
bias = B_int32.astype(np.int64).reshape(1, Cout, 1, 1)
acc = acc_raw - zp_correction + bias                # [N, Cout, Hout, Wout] int64

# Bước 4
M_int_arr, shift_arr = make_requant_params(scale_x, scale_w, scale_y)
y_pre_act = post_process_int32_to_int8(acc, M_int_arr, shift_arr, zp_y)
```

---

## P0 – `rs_dense_3x3` (RS_DENSE_3x3)

### Mục đích
Regular Strided Dense Conv 3×3. Dùng cho Layer 0,1,3,17 (stride=2) và bên trong QBottleneck của QC2f (stride=1).

### Signature đầy đủ

```python
def rs_dense_3x3(
    X_int8: np.ndarray,   # [N, Cin, H, W] int8
    W_int8: np.ndarray,   # [Cout, Cin, 3, 3] int8
    B_int32: np.ndarray,  # [Cout] int32
    scale_x: float,       # input scale
    zp_x: int,            # input zero-point (signed: pytorch_zp - 128)
    scale_w: np.ndarray,  # [Cout] per-channel weight scale
    zp_w: int,            # weight zero-point (MUST = 0)
    scale_y: float,       # output scale
    zp_y: int,            # output zero-point (signed)
    stride: int = 1,      # 1 hoặc 2
    padding: str = "same",# "same" → pad=1, hoặc int
    activation: str = "silu", # "silu"|"relu"|"relu6"|"none"
    dump: bool = False,
) -> tuple:               # (Y_int8, scale_y, zp_y)
```

### Cách gọi từ `block_conv.py`:

```python
# Khi kernel 3×3
rs_dense_3x3(
    X_int8, W_int8, B_int32,
    scale_x, zp_x, scale_w, zp_w, scale_y, zp_y,
    stride=stride, padding=padding, activation=activation
)
```

### Verify script pattern (từ verify_layer_0.py):

```python
# Extract params từ model
m = model.model.model[0]  # Layer 0
is_3x3 = (m.conv.kernel_size[0] == 3)
my_out, _, _ = block_conv(
    in_signed,
    padding="same",
    **get_conv_params(m, in_scale, in_zp, is_3x3=is_3x3)
)
```

---

## P1 – `os_1x1` (OS_1x1)

### Mục đích
Output-Stationary 1×1 Pointwise Conv. Kernel=1, stride=1, padding=0. Dùng cho **tất cả phép channel projection** trong mọi block.

### Signature đầy đủ

```python
def os_1x1(
    X_int8: np.ndarray,   # [N, Cin, H, W] int8
    W_int8: np.ndarray,   # [Cout, Cin, 1, 1] int8
    B_int32: np.ndarray,  # [Cout] int32
    scale_x: float,
    zp_x: int,
    scale_w: np.ndarray,  # [Cout] per-channel weight scale
    zp_w: int,            # MUST = 0
    scale_y: float,
    zp_y: int,
    activation: str = "none",  # "silu"|"relu"|"relu6"|"none"
    dump: bool = False,
) -> tuple:               # (Y_int8, scale_y, zp_y)
```

### Khác biệt với P0:
- `padding = 0` cố định (không có padding)
- `stride = 1` cố định
- `activation` mặc định = `"none"` (projection layers không activate)

### Cách gọi từ các blocks:

```python
# Trong block_qc2f.py
y_cv1, s_cv1, z_cv1 = os_1x1(X_int8, **cv1_params)

# Trong block_sppf.py
y_cv1, s_cv1, z_cv1 = os_1x1(X_int8, **cv1_params)
y_out, s_out, z_out = os_1x1(y_cat, **cv2_params)

# Trong block_qpsa.py (GEMM_ATTN)
Q_int8, sq, zq = os_1x1(X_int8, W_Q, B_Q,
                          scale_x, zp_x,
                          sp["scale_wQ"], sp.get("zp_wQ", 0),
                          sp["scale_Q"], sp["zp_Q"])
```

---

## Bảng tham số cần trích xuất từ model

Hàm `get_conv_params(m, in_scale, in_zp, is_3x3)` trong verify scripts trích xuất:

| Tham số | Nguồn từ model | Chú ý |
|:--------|:---------------|:------|
| `W_int8` | `m.conv.weight().int_repr().numpy()` | int8 |
| `B_int32` | `m.conv.bias().numpy()` | int32 |
| `scale_x` | `in_scale` (từ layer trước) | float |
| `zp_x` | `in_zp` (từ layer trước, đã -128) | signed int |
| `scale_w` | `m.conv.weight().q_per_channel_scales().numpy()` | per-channel |
| `zp_w` | `0` | symmetric weight |
| `scale_y` | `float(m.conv.scale)` | float |
| `zp_y` | `int(m.conv.zero_point) - 128` | signed int |
| `stride` | `m.conv.stride[0]` | int |
| `activation` | `"relu"` nếu `isinstance(m.act, nn.ReLU)` | string |

---

## Luồng dữ liệu tổng thể

```
Input INT8 [N, Cin, H, W]   (scale_x, zp_x)
    │
    ├─ Pad với zp_x
    ├─ MAC INT64: Σ X×W
    ├─ Correction: -zp_x×ΣW + B
    ├─ PPU requant: (Acc×M_int + offset) >> shift + zp_y → clip INT8
    └─ Activation (ReLU/SiLU/None)
    │
Output INT8 [N, Cout, Hout, Wout]   (scale_y, zp_y)
```

---

## Import chain

```
block_conv.py
  └── primitive_conv.py: rs_dense_3x3, os_1x1
        └── quant_affine.py: make_requant_params, post_process_int32_to_int8,
                              apply_silu_float, quantize_affine
              └── config.py: INT8_MIN, INT8_MAX, ACT_SILU, ACT_RELU, ...
```
