# Golden Block: SPPF
## Source: `PHASE1/python_golden/blocks/block_sppf.py`
## Layer: 9

---

## Cấu trúc block trong model

```
Layer 9: SPPF [1, 256, 20, 20] → [1, 256, 20, 20]
         (Spatial Pyramid Pooling - Fast)
```

---

## Primitives được sử dụng

```
P1  (os_1x1)     – cv1: initial expansion (256→128)
P3  (maxpool_5x5)– m: 3 lần liên tiếp, stride=1, pad=2
P5  (concat)     – join 4 tensors [cv1, p1, p2, p3]
P1  (os_1x1)     – cv2: final compression/fusion (512→256)
P12 (PPU)        – tích hợp bên trong P1
```

---

## Code đầy đủ: `block_sppf.py`

```python
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primitives.primitive_conv import os_1x1
from primitives.primitive_pool import maxpool_5x5
from primitives.primitive_tensor import concat

def block_sppf(
    X_int8: np.ndarray,          # [N, C, H, W] int8
    cv1_params: dict,            # params cho os_1x1 (expansion)
    cv2_params: dict,            # params cho os_1x1 (compression)
    pool_params: dict = None,    # (không dùng, pool tự dùng s_cv1, z_cv1)
    concat_params: dict = None,  # {"scales":[], "zps":[], "scale_out", "zp_out"}
    dump: bool = False,
) -> tuple:                      # (Y_int8, scale_y, zp_y)
    """
    Mapping cho SPPF block:
    1. cv1 (1x1): Expansion C→C/2
    2. m[0]: MaxPool 5x5, stride=1
    3. m[1]: MaxPool 5x5 (trên kết quả m[0])
    4. m[2]: MaxPool 5x5 (trên kết quả m[1])
    5. Concat: [cv1, m0, m1, m2]  ← 4 tensors
    6. cv2 (1x1): Compression/Fusion 2C→C
    """
    # Stage 1: Initial 1x1 Conv – P1
    y_cv1, s_cv1, z_cv1 = os_1x1(X_int8, **cv1_params)
    # y_cv1: [N, C/2, H, W]

    # Stage 2: Sequential Max Pooling – P3 (×3)
    # Tương đương: y = [cv1_out]; y.extend(m(y[-1]) for _ in range(3))
    y_pooled = [y_cv1]
    curr_y = y_cv1
    for i in range(3):
        curr_y, _, _ = maxpool_5x5(curr_y, s_cv1, z_cv1)
        # scale/zp KHÔNG đổi qua maxpool!
        y_pooled.append(curr_y)
    # y_pooled = [y_cv1, pool1, pool2, pool3] – tất cả cùng scale!

    # Stage 3: Concatenate – P5
    if concat_params is None:
        # Nếu không cung cấp, dùng scale/zp của cv1 (tất cả input giống nhau)
        concat_params = {
            "scales":    [s_cv1] * 4,
            "zps":       [z_cv1] * 4,
            "scale_out": s_cv1,
            "zp_out":    z_cv1
        }
    y_cat, s_cat, z_cat = concat(y_pooled, **concat_params)
    # y_cat: [N, 2C, H, W]  (4 × C/2 channels)

    # Stage 4: Final 1x1 Conv – P1
    y_out, s_out, z_out = os_1x1(y_cat, **cv2_params)
    # y_out: [N, C, H, W]  (2C → C)

    return y_out, s_out, z_out
```

---

## Luồng dữ liệu

```
X_int8 [N, 256, H, W]   (scale_x, zp_x)
    │
    ├─[P1] cv1: OS_1x1 → [N, 128, H, W]   (s_cv1, z_cv1)
    │          (Expansion 256→128 với ReLU)
    │
    ├─[P3] MaxPool 5×5 → pool1 [N, 128, H, W]   (s_cv1, z_cv1) ← KHÔNG ĐỔI!
    ├─[P3] MaxPool 5×5 → pool2 [N, 128, H, W]   (s_cv1, z_cv1) ← KHÔNG ĐỔI!
    ├─[P3] MaxPool 5×5 → pool3 [N, 128, H, W]   (s_cv1, z_cv1) ← KHÔNG ĐỔI!
    │
    ├─[P5] CONCAT: [cv1, pool1, pool2, pool3]
    │      → [N, 512, H, W]   (s_cat, z_cat)
    │      (Vì cùng scale → concat không cần align, pass-through!)
    │
    └─[P1] cv2: OS_1x1 → [N, 256, H, W]   (scale_y, zp_y)
               (Compression 512→256 với ReLU)
    │
Y_int8 [N, 256, H, W]   (scale_y, zp_y)
```

---

## Tính chất đặc biệt: Quantization Invariance của MaxPool

MaxPool là phép so sánh số nguyên → **KHÔNG đổi scale/zp**:

$$Scale_{pool1} = Scale_{pool2} = Scale_{pool3} = Scale_{cv1}$$

→ 4 tensors đưa vào CONCAT có **cùng scale** → không cần domain alignment.  
→ `concat_params["scales"] = [s_cv1, s_cv1, s_cv1, s_cv1]` và `scale_out = s_cv1`.

---

## Cấu trúc params dict

### `cv1_params` (OS_1x1 – expansion):

```python
cv1_params = {
    "W_int8":  m.cv1.conv.weight().int_repr().numpy(),  # [128, 256, 1, 1]
    "B_int32": m.cv1.conv.bias().numpy(),
    "scale_x": in_scale,
    "zp_x":    in_zp,
    "scale_w": m.cv1.conv.weight().q_per_channel_scales().numpy(),
    "zp_w":    0,
    "scale_y": float(m.cv1.conv.scale),
    "zp_y":    int(m.cv1.conv.zero_point) - 128,
    "activation": "relu",
}
```

### `concat_params`:

```python
s_cv1 = float(m.cv1.conv.scale)
z_cv1 = int(m.cv1.conv.zero_point) - 128

concat_params = {
    "scales":    [s_cv1, s_cv1, s_cv1, s_cv1],  # tất cả cùng scale!
    "zps":       [z_cv1, z_cv1, z_cv1, z_cv1],
    "scale_out": s_cv1,    # output scale giống input (no align needed)
    "zp_out":    z_cv1,
}
```

### `cv2_params` (OS_1x1 – compression):

```python
cv2_params = {
    "W_int8":  m.cv2.conv.weight().int_repr().numpy(),  # [256, 512, 1, 1]
    "B_int32": m.cv2.conv.bias().numpy(),
    "scale_x": s_cv1,   # output của concat = s_cv1
    "zp_x":    z_cv1,
    "scale_w": m.cv2.conv.weight().q_per_channel_scales().numpy(),
    "zp_w":    0,
    "scale_y": float(m.cv2.conv.scale),
    "zp_y":    int(m.cv2.conv.zero_point) - 128,
    "activation": "relu",
}
```

---

## Kết quả verify Layer 9

| Metric | Kết quả |
|:-------|:--------|
| Input shape | [1, 256, 20, 20] |
| Output shape | [1, 256, 20, 20] |
| Mean Match | **99.94%** |
| Max Diff | **1 LSB** |
| Status | PASS ✓ |

---

## Import chain

```
block_sppf.py
  ├── primitive_conv.os_1x1       (P1)
  ├── primitive_pool.maxpool_5x5  (P3)
  └── primitive_tensor.concat     (P5)
        ├── quant_affine.*
        └── quant_domain_align.align_and_concat
```
