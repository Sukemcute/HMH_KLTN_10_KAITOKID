# Golden Block: QC2fCIB
## Source: `PHASE1/python_golden/blocks/block_qc2f_cib.py`
## Layer: 22

---

## Cấu trúc block trong model

```
Layer 22: QC2fCIB [1, 384, 20, 20] → [1, 256, 20, 20]
          (QC2f với Conditional Identity Block)
```

---

## Primitives được sử dụng

```
P1  (os_1x1)      – cv1, PW 1×1 ×2 bên trong QCIB, cv2
P2  (dw_3x3)      – thông qua dw_conv_int với pad=1 cho DW 3×3
P8  (dw_7x7)      – thông qua dw_conv_int với pad=3 cho DW 7×7
P5  (concat)      – ghép y[0]+y[1]+QCIB_out
P7  (ewise_add)   – residual shortcut bên trong QCIB
P12 (PPU)         – tích hợp trong mọi ops
```

---

## Code đầy đủ: `block_qc2f_cib.py`

### Hàm phụ: `block_qcib`

```python
from primitives.primitive_conv import os_1x1
from primitives.primitive_dw import dw_conv_int   # generic DW, bất kỳ kernel
from primitives.primitive_tensor import ewise_add, concat

def block_qcib(
    X_int8: np.ndarray,         # [N, C, H, W] int8
    scale_x: float,
    zp_x: int,
    cv_params_list: list,       # List 5 dicts cho 5 convolutions
                                # [DW3x3, PW1x1, DW7x7, PW1x1, DW3x3]
    add_params: dict,           # {"scale_out", "zp_out"} cho shortcut
    shortcut: bool = True,
) -> tuple:                     # (Y_int8, scale_y, zp_y)
    """
    QCIB sequence: DW3x3 → PW1x1 → DW7x7 → PW1x1 → DW3x3 [+ shortcut]

    QUAN TRỌNG: padding phải tường minh!
    - DW 3×3 (conv 0, 4): padding=1
    - DW 7×7 (conv 2):    padding=3
    Nếu không có padding → spatial shrink → skip connection sẽ shape mismatch!
    """
    # 0: DW 3×3 – P2 (via generic dw_conv_int)
    y0, s0, z0 = dw_conv_int(X_int8, padding=1, **cv_params_list[0])

    # 1: PW 1×1 – P1
    y1, s1, z1 = os_1x1(y0, **cv_params_list[1])

    # 2: DW 7×7 – P8 (via generic dw_conv_int với padding=3)
    y2, s2, z2 = dw_conv_int(y1, padding=3, **cv_params_list[2])

    # 3: PW 1×1 – P1
    y3, s3, z3 = os_1x1(y2, **cv_params_list[3])

    # 4: DW 3×3 – P2
    y4, s4, z4 = dw_conv_int(y3, padding=1, **cv_params_list[4])

    if shortcut:
        # Residual: X + y4 – P7
        return ewise_add(X_int8, scale_x, zp_x, y4, s4, z4, **add_params)
    else:
        return y4, s4, z4
```

### Hàm chính: `block_qc2f_cib`

```python
def block_qc2f_cib(
    X_int8: np.ndarray,               # [N, Cin, H, W] int8
    cv1_params: dict,                 # OS_1x1 cho expansion
    cv2_params: dict,                 # OS_1x1 cho compression
    qcib_params_list: list,           # list of [5 dict] per QCIB
    qcib_add_params_list: list,       # list of {"scale_out","zp_out"} per QCIB
    concat_params: dict,              # {"scale_out","zp_out"} cho concat
) -> tuple:                           # (Y_int8, scale_y, zp_y)
    """
    Mapping cho QC2fCIB block:
    1. cv1 (1×1): expansion
    2. split: y[0] (identity) + y[1] (→ QCIB chain)
    3. QCIB chain: stacked CIB modules
    4. concat: y[0] + y[1] + QCIB_outputs
    5. cv2 (1×1): compression
    """
    # 1. Expansion – P1
    y_cv1, s_cv1, z_cv1 = os_1x1(X_int8, **cv1_params)

    # 2. Split
    mid = y_cv1.shape[1] // 2
    y = [y_cv1[:, :mid, :, :],   # y[0]: identity branch
         y_cv1[:, mid:, :, :]]   # y[1]: input to QCIB

    curr_scale = s_cv1
    curr_zp    = z_cv1

    # 3. Stacked QCIBs
    for i, qcib_p in enumerate(qcib_params_list):
        y_next, s_next, z_next = block_qcib(
            y[-1], curr_scale, curr_zp,
            qcib_p, qcib_add_params_list[i]
        )
        y.append(y_next)
        curr_scale = s_next
        curr_zp    = z_next

    # 4. Collect scales/zps cho tất cả tensors cần concat
    # y[0], y[1] có s_cv1, z_cv1
    # y[2] (= QCIB output) có curr_scale, curr_zp
    scales = [s_cv1, s_cv1]
    zps    = [z_cv1, z_cv1]
    if len(qcib_params_list) == 1:
        scales.append(curr_scale)
        zps.append(curr_zp)

    # 5. Concat – P5
    y_cat, s_cat, z_cat = concat(y, scales=scales, zps=zps, **concat_params)

    # 6. Compression – P1
    return os_1x1(y_cat, **cv2_params)
```

---

## Luồng dữ liệu chi tiết

```
X_int8 [1, 384, 20, 20]
    │
    ├─[P1] cv1: OS_1x1 → [1, 512, 20, 20]   (expand 384→512)
    │
    ├─ Split:
    │      y[0] = [:, :256, :, :]   [1,256,20,20] identity
    │      y[1] = [:, 256:, :, :]   [1,256,20,20] → QCIB
    │
    │  === QCIB Module ===
    ├─[P2]  DW 3×3 pad=1: y[1] → [1,256,20,20]
    ├─[P1]  PW 1×1:        →      [1,256,20,20]
    ├─[P8]  DW 7×7 pad=3:  →      [1,256,20,20]   (3-pass multipass)
    ├─[P1]  PW 1×1:        →      [1,256,20,20]
    ├─[P2]  DW 3×3 pad=1:  →      [1,256,20,20]
    └─[P7]  Add: y[1] + ^  →      [1,256,20,20]   QCIB_out
    │  === End QCIB ===
    │
    ├─[P5] Concat: [y[0], y[1], QCIB_out] → [1, 768, 20, 20]
    │      (3 tensors × 256 channels, domain align)
    │
    └─[P1] cv2: OS_1x1 → [1, 256, 20, 20]   (compress 768→256)
    │
Output [1, 256, 20, 20]
```

---

## Cấu trúc `cv_params_list` (5 entries cho QCIB)

```python
# Layer 22: QCIB params cho m[0] (chỉ 1 QCIB)
qcib_params_list = [
    [
        # conv 0: DW 3×3
        {
            "W_int8_per_ch":  m.m[0].cv1[0].conv.weight().int_repr().numpy().squeeze(1),
            "B_int32_per_ch": m.m[0].cv1[0].conv.bias().numpy(),
            "scale_x":        s_cv1,
            "zp_x":           z_cv1,
            "scale_w_per_ch": m.m[0].cv1[0].conv.weight().q_per_channel_scales().numpy(),
            "scale_y":        float(m.m[0].cv1[0].conv.scale),
            "zp_y":           int(m.m[0].cv1[0].conv.zero_point) - 128,
            "stride":         1,
        },
        # conv 1: PW 1×1
        {
            "W_int8":  m.m[0].cv1[1].conv.weight().int_repr().numpy(),
            "B_int32": m.m[0].cv1[1].conv.bias().numpy(),
            "scale_x": ...,  "zp_x": ...,
            "scale_w": ...,  "zp_w": 0,
            "scale_y": ...,  "zp_y": ...,
        },
        # conv 2: DW 7×7  ← truyền vào dw_conv_int với padding=3
        { ... scale_w_per_ch là [C] float ... },
        # conv 3: PW 1×1
        { ... },
        # conv 4: DW 3×3
        { ... },
    ]
]
```

---

## Bug đã fix: Padding tường minh

```python
# ❌ SAI – mặc định padding=0 → spatial shrink 20×20 → 14×14 → ... → 0×0
y0, s0, z0 = dw_conv_int(X_int8, **cv_params_list[0])

# ✅ ĐÚNG – padding tường minh
y0, s0, z0 = dw_conv_int(X_int8, padding=1, **cv_params_list[0])  # DW 3×3
y2, s2, z2 = dw_conv_int(y1,     padding=3, **cv_params_list[2])  # DW 7×7
```

---

## Kết quả verify Layer 22

| Metric | Kết quả |
|:-------|:--------|
| Input shape | [1, 384, 20, 20] |
| Output shape | [1, 256, 20, 20] |
| Mean Match | **99.96%** |
| Max Diff | **1 LSB** |
| Status | PASS ✓ |

---

## Import chain

```
block_qc2f_cib.py
  ├── primitive_conv.os_1x1             (P1)
  ├── primitive_dw.dw_conv_int          (P2/P8 generic)
  ├── primitive_tensor.ewise_add        (P7)
  └── primitive_tensor.concat           (P5)
        └── quant_domain_align.*
              └── quant_affine.*
```
