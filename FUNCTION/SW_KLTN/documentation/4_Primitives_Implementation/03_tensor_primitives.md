# P3–P7 – Tensor Manipulation Primitives
## Source files:
- `PHASE1/python_golden/primitives/primitive_pool.py` → P3
- `PHASE1/python_golden/primitives/primitive_tensor.py` → P4, P5, P6, P7
- `PHASE1/python_golden/quant/quant_domain_align.py` → logic cho P5, P7

---

## P3 – `maxpool_5x5` (MAXPOOL_5x5)

### Source: `primitive_pool.py`

### Mục đích
Max Pooling 5×5, stride=1, padding=2. Dùng 3 lần liên tiếp trong **SPPF** block. Là phép so sánh số nguyên → **KHÔNG thay đổi scale/zp**.

### Signature

```python
def maxpool_5x5(
    X_int8: np.ndarray,  # [N, C, H, W] int8
    scale_x: float,      # pass-through (KHÔNG thay đổi)
    zp_x: int,           # pass-through (KHÔNG thay đổi)
    padding: int = 2,    # default=2 để giữ H,W không đổi
) -> tuple:              # (Y_int8, scale_x, zp_x)  ← scale/zp giữ nguyên!
```

### Implementation

```python
kernel = 5
stride = 1
pad_val = np.iinfo(np.int8).min  # -128 = INT8_MIN

# Pad với -128 (giá trị min → không ảnh hưởng max comparison)
X_pad = np.pad(X_int8.astype(np.int16),
               ((0,0),(0,0),(padding,padding),(padding,padding)),
               mode='constant', constant_values=int(pad_val))

Y = np.full((N,C,Hout,Wout), fill_value=pad_val, dtype=np.int16)

for kh in range(kernel):    # 0..4
    for kw in range(kernel): # 0..4
        x_slice = X_pad[:,:, kh:kh+Hout:1, kw:kw+Wout:1]
        Y = np.maximum(Y, x_slice)

return Y.astype(np.int8), float(scale_x), int(zp_x)
```

### Tại sao pad với -128?
Vì phép max pooling trong không gian signed: padding = -128 đảm bảo bất kỳ pixel thật nào (≥ -128) sẽ thắng trong so sánh. Nếu dùng 0 làm pad, kết quả sẽ sai với negative activations.

### Dùng trong SPPF:

```python
# block_sppf.py
y_cv1, s_cv1, z_cv1 = os_1x1(X_int8, **cv1_params)  # [N,C/2,H,W]

y_pooled = [y_cv1]
curr_y = y_cv1
for i in range(3):
    curr_y, _, _ = maxpool_5x5(curr_y, s_cv1, z_cv1)  # 3 lần liên tiếp
    y_pooled.append(curr_y)
# y_pooled = [cv1, pool1, pool2, pool3]  ← cùng scale/zp!
```

---

## P4 – `move` (MOVE)

### Mục đích
Copy tensor, giữ nguyên quant metadata. Dùng cho **skip connections** – lưu output của layer vào buffer trong khi các layer sau tiếp tục xử lý.

### Signature & Implementation

```python
def move(
    X_int8: np.ndarray,  # [N, C, H, W] int8
    scale: float,
    zp: int,
) -> tuple:              # (X_int8.copy(), scale, zp)
    return X_int8.copy(), float(scale), int(zp)
```

**Dùng trong:**
- SPPF: lưu cv1 output để concat sau 3 pool
- QPSA: lưu branch `a` trong khi branch `b` qua attention
- QC2fCIB: lưu y[0] (identity branch) trong khi y[1] qua QCIB

---

## P5 – `concat` (CONCAT)

### Source: `primitive_tensor.py` + `quant_domain_align.py`

### Mục đích
Concatenate INT8 tensors với **Domain Alignment** (requantize về common scale trước khi join). Dùng trong: QC2f, SPPF, Concat layers (12, 15, 18, 21), QPSA, QC2fCIB.

### Signature

```python
def concat(
    tensors: list,           # list of np.ndarray int8 – cùng H, W
    scales: list,            # list of float – scale per tensor
    zps: list,               # list of int   – zp per tensor
    axis: int = 1,           # thường = 1 (channel dim)
    strategy: str = "max",   # "max"|"min"|"offline"
    scale_out: float = None, # nếu None → tự tính
    zp_out: int = None,      # nếu None → dùng zp của tensor được chọn
) -> tuple:                  # (Y_int8, common_scale, common_zp)
```

### Cơ chế Domain Alignment (từ `quant_domain_align.py`):

```python
def align_and_concat(tensors_int8, scales, zps, axis=1,
                     strategy="max", scale_out=None, zp_out=None):
    # Bước 1: Xác định common domain
    if scale_out is not None and zp_out is not None:
        common_scale, common_zp = scale_out, zp_out      # offline (từ model)
    else:
        common_scale, common_zp = compute_common_scale(scales, zps, strategy)

    # Bước 2: Requant từng tensor về common domain
    aligned = []
    for t, s, z in zip(tensors_int8, scales, zps):
        aligned.append(requant_to_common(t, s, z, common_scale, common_zp))

    # Bước 3: Join theo channel
    Y_int8 = np.concatenate(aligned, axis=axis).astype(np.int8)
    return Y_int8, common_scale, common_zp
```

### `requant_to_common` – logic tái lượng tử hóa:

```python
def requant_to_common(x_int8, scale_src, zp_src, scale_dst, zp_dst):
    # Identity path: cùng domain → copy thôi
    if abs(scale_src - scale_dst) < 1e-12 and zp_src == zp_dst:
        return x_int8.copy()

    # Golden path: dequant → float → quantize
    x_float = (x_int8.astype(np.float64) - zp_src) * scale_src
    return quantize_affine(x_float, scale_dst, zp_dst, dtype="int8")
```

### Strategy selection:

| Strategy | Common scale chọn | Dùng khi nào |
|:---------|:------------------|:-------------|
| `"offline"` | `scale_out` từ model observer | Preferred cho HW – đã biết trước |
| `"max"` | max(scales) | Giữ precision nhánh lớn nhất |
| `"min"` | min(scales) | Tránh clamp nhánh nhỏ nhất |

### Ví dụ Concat Layer 12 (từ verify script):

```python
# Layer 12: concat [Layer11_out (256 ch)] + [Layer6_out (128 ch)] → 384 ch
concat_map = {12: [11, 6], 15: [14, 4], 18: [17, 13], 21: [20, 10]}
in_signed = [to_signed_int8(getattr(res, f"Layer{idx}")().int_repr().numpy())
             for idx in concat_map[12]]     # [Layer11, Layer6]
in_scale  = [float(getattr(res, f"Layer{idx}")().q_scale()) for idx in [11, 6]]
in_zp     = [int(getattr(res, f"Layer{idx}")().q_zero_point())-128 for idx in [11, 6]]

# c_p chứa scale_out, zp_out từ model QConcat
my_out, _, _ = block_concat(in_signed, in_scale, in_zp, c_p)
```

---

## P6 – `upsample_nearest` (UPSAMPLE_NEAREST)

### Mục đích
Nearest-neighbor upsampling ×2. Dùng trong Layer 11, 14. **Không thay đổi scale/zp** – chỉ address remapping.

### Signature & Implementation

```python
def upsample_nearest(
    X_int8: np.ndarray,   # [N, C, H, W] int8
    scale_x: float,       # pass-through
    zp_x: int,            # pass-through
    scale_factor: int = 2,
) -> tuple:               # (Y_int8, scale_x, zp_x)

    sf = scale_factor
    N, C, H, W = X_int8.shape
    Y = np.zeros((N, C, H*sf, W*sf), dtype=np.int8)

    for di in range(sf):
        for dj in range(sf):
            Y[:, :, di::sf, dj::sf] = X_int8  # nearest-neighbor fill

    return Y, float(scale_x), int(zp_x)
```

### Kết quả:
- Mỗi pixel `[n,c,i,j]` được copy ra `sf×sf = 4` vị trí: `[di::2, dj::2]`
- H×W → 2H×2W, channels giữ nguyên
- Match rate Layer 11, 14: **100.00%** (0 LSB diff)

---

## P7 – `ewise_add` (EWISE_ADD)

### Source: `primitive_tensor.py` + `quant_domain_align.py`

### Mục đích
Element-wise addition của 2 INT8 tensors với domain alignment. Dùng cho residual connections trong **QPSA** và **QC2fCIB**.

### Signature

```python
def ewise_add(
    A_int8: np.ndarray,    # [N, C, H, W] int8
    scale_A: float,
    zp_A: int,
    B_int8: np.ndarray,    # [N, C, H, W] int8 – cùng shape với A
    scale_B: float,
    zp_B: int,
    scale_out: float = None,  # None → auto-select
    zp_out: int = None,
    strategy: str = "max",
) -> tuple:                # (Y_int8, scale_out, zp_out)
```

### Cơ chế (Golden Path – từ `quant_domain_align.py`):

```python
def align_and_add(A_int8, scale_A, zp_A, B_int8, scale_B, zp_B,
                  scale_out=None, zp_out=0, strategy="max"):
    # 1. Xác định target domain
    if scale_out is not None:
        target_scale, target_zp = scale_out, zp_out
    else:
        target_scale, target_zp = compute_common_scale(
            [scale_A, scale_B], [zp_A, zp_B], strategy=strategy)

    # 2. Golden Math: Add in float domain → re-quantize
    #    (match PyTorch QFunctional.add behavior)
    A_float = (A_int8.astype(np.float64) - zp_A) * scale_A
    B_float = (B_int8.astype(np.float64) - zp_B) * scale_B
    sum_float = A_float + B_float

    Y_int8 = quantize_affine(sum_float, target_scale, target_zp, dtype="int8")
    return Y_int8, float(target_scale), int(target_zp)
```

### Toán học đúng:
$$Y_{float} = (A_{int8} - ZP_A) \times S_A + (B_{int8} - ZP_B) \times S_B$$
$$Y_{int8} = \text{clamp}\left(\text{round}\left(\frac{Y_{float}}{S_{out}}\right) + ZP_{out},\; -128,\; 127\right)$$

### Dùng trong QPSA (block_qpsa.py):

```python
# Shortcut 1: b_attn = b + attention(b)
b_attn, s_b_attn, z_b_attn = ewise_add(
    b, s_cv1, z_cv1,
    y_attn, s_attn, z_attn,
    **attn_add_params   # scale_out, zp_out từ model
)

# Shortcut 2: b_final = b_attn + ffn(b_attn)
b_final, s_b_final, z_b_final = ewise_add(
    b_attn, s_b_attn, z_b_attn,
    y_f2, s_f2, z_f2,
    **ffn_add_params
)
```

### Dùng trong QC2fCIB (block_qcib):

```python
# Residual shortcut bên trong QCIB
if shortcut:
    return ewise_add(X_int8, scale_x, zp_x, y4, s4, z4, **add_params)
```

---

## Tóm tắt I/O của 5 primitives

| Primitive | Input shape | Output shape | scale/zp thay đổi? |
|:----------|:------------|:-------------|:-------------------|
| P3 maxpool_5x5 | [N,C,H,W] | [N,C,H,W] | **Không** (pass-through) |
| P4 move | [N,C,H,W] | [N,C,H,W] (copy) | **Không** |
| P5 concat | [N,Ci,H,W]×n | [N,ΣCi,H,W] | **Có** → common domain |
| P6 upsample | [N,C,H,W] | [N,C,2H,2W] | **Không** (pass-through) |
| P7 ewise_add | [N,C,H,W]×2 | [N,C,H,W] | **Có** → target domain |

---

## Import chain

```
block_concat.py   → primitive_tensor.concat
block_upsample.py → primitive_tensor.upsample_nearest
block_sppf.py     → primitive_tensor.concat + primitive_pool.maxpool_5x5
block_qpsa.py     → primitive_tensor.ewise_add, concat
block_qc2f_cib.py → primitive_tensor.ewise_add, concat
block_qc2f.py     → primitive_tensor.concat, ewise_add

primitive_tensor.py
  └── quant_domain_align.py: align_and_concat, align_and_add
        └── quant_affine.py: quantize_affine, dequantize_affine
```
