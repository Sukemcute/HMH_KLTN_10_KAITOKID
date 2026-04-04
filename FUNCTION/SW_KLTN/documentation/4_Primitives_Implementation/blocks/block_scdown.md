# Golden Block: SCDown
## Source: `PHASE1/python_golden/blocks/block_scdown.py`
## Layers: 5, 7, 20

---

## Cấu trúc block trong model

```
Layer 5:  SCDown [1,64,80,80]   → [1,128,40,40]   cv1: 64→128, cv2: stride=2
Layer 7:  SCDown [1,128,40,40]  → [1,256,20,20]   cv1: 128→256, cv2: stride=2
Layer 20: SCDown [1,128,40,40]  → [1,128,20,20]   cv1: 128→128, cv2: stride=2
```

---

## Primitives được sử dụng

```
P1 (os_1x1)  – cv1: Pointwise channel expansion
P2 (dw_3x3)  – cv2: Depthwise stride=2 spatial downsampling
P12 (PPU)    – tích hợp bên trong P1 và P2
```

---

## Code đầy đủ: `block_scdown.py`

```python
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primitives.primitive_conv import os_1x1
from primitives.primitive_dw import dw_3x3

def block_scdown(
    X_int8: np.ndarray,   # [N, Cin, H, W] int8
    cv1_params: dict,     # params cho os_1x1
    cv2_params: dict,     # params cho dw_3x3
    dump: bool = False,
) -> tuple:               # (Y_int8, scale_y, zp_y)
    """
    Mapping cho SCDown (Spatial-Channel Downsampling):
    1. cv1: OS_1x1  – channel expansion
    2. cv2: DW_3x3  – stride=2 spatial downsampling
    """
    # Stage 1: Channel Expansion – P1
    y1, s1, z1 = os_1x1(X_int8, **cv1_params)
    # y1: [N, Cout_cv1, H, W]

    # Stage 2: Spatial Downsampling – P2
    y2, s2, z2 = dw_3x3(y1, **cv2_params)
    # y2: [N, Cout_cv1, H//2, W//2]  (stride=2 trong cv2_params)

    return y2, s2, z2
```

---

## Luồng dữ liệu

```
X_int8 [N, Cin, H, W]   (scale_x, zp_x)
    │
    ├─[P1] cv1: OS_1x1 – channel expansion (e.g. 64→128)
    │      output: [N, 128, H, W]   (s_cv1, z_cv1)
    │      padding=0, stride=1 (1×1 conv)
    │
    └─[P2] cv2: DW_3x3 – spatial ×0.5 (stride=2)
           ├─ Pad mỗi channel với z_cv1 (pad=1)
           ├─ DW MAC per channel (int64)
           ├─ ZP correction + bias per channel
           └─ Per-channel PPU requant
           output: [N, 128, H//2, W//2]   (scale_y, zp_y)
    │
Y_int8 [N, Cout, H/2, W/2]   (scale_y, zp_y)
```

---

## Cấu trúc params dict

### `cv1_params` (OS_1x1):

```python
cv1_params = {
    "W_int8":  m.cv1.conv.weight().int_repr().numpy(),  # [Cout, Cin, 1, 1]
    "B_int32": m.cv1.conv.bias().numpy(),               # [Cout]
    "scale_x": in_scale,
    "zp_x":    in_zp,
    "scale_w": m.cv1.conv.weight().q_per_channel_scales().numpy(),  # [Cout]
    "zp_w":    0,
    "scale_y": float(m.cv1.conv.scale),
    "zp_y":    int(m.cv1.conv.zero_point) - 128,
    "activation": "relu",  # cv1 có ReLU
}
```

### `cv2_params` (DW_3x3):

```python
cv2_params = {
    "W_int8_per_ch":  m.cv2.conv.weight().int_repr().numpy().squeeze(1),
    #                   PyTorch DW: [C, 1, 3, 3] → squeeze → [C, 3, 3]
    "B_int32_per_ch": m.cv2.conv.bias().numpy(),      # [C]
    "scale_x":        float(m.cv1.conv.scale),         # = s_cv1
    "zp_x":           int(m.cv1.conv.zero_point) - 128,
    "scale_w_per_ch": m.cv2.conv.weight().q_per_channel_scales().numpy(),
    "scale_y":        float(m.cv2.conv.scale),
    "zp_y":           int(m.cv2.conv.zero_point) - 128,
    "stride":         2,   # stride=2 cho downsampling
    "activation":     "none",  # cv2 không có activation
}
```

**Quan trọng:** DW weight từ PyTorch có shape `[C, 1, 3, 3]` (groups=C) → phải `squeeze(1)` → `[C, 3, 3]` trước khi truyền vào `dw_3x3`.

---

## Đặc điểm kỹ thuật quan trọng

### 1. Asymmetric padding cho DW_3x3:
`dw_3x3` luôn dùng `pad=1` (kernel=3, same padding). Padding value = `zp_x` (output của cv1), không phải 0. Điều này đảm bảo các pixel pad = "true float zero" trong quantized space.

### 2. Per-channel requant trong DW:
Mỗi channel `c` của DW có thể có `scale_w[c]` khác nhau → `M_int[c]` và `shift[c]` riêng. Đây là điểm khác biệt quan trọng so với dense conv (per-channel shared).

### 3. Hardware streaming:
Trong hardware, output của P1 (cv1) có thể stream trực tiếp vào line buffer của P2 (cv2) → giảm external memory access.

---

## Kết quả verify

| Layer | Shape In | Shape Out | Mean Match | Max Diff |
|:------|:---------|:----------|:-----------|:---------|
| L5 | [1,64,80,80] | [1,128,40,40] | **99.90%** | 1 LSB |
| L7 | [1,128,40,40] | [1,256,20,20] | **99.92%** | 1 LSB |
| L20 | [1,128,40,40] | [1,128,20,20] | **99.99%** | 1 LSB |

SCDown đạt match rate rất cao (>99.9%) do chỉ có 2 operations đơn giản.

---

## Import chain

```
block_scdown.py
  ├── primitive_conv.os_1x1  (P1)
  └── primitive_dw.dw_3x3    (P2)
        ├── quant_affine.make_requant_params
        └── quant_affine.post_process_int32_to_int8  (P12)
```
