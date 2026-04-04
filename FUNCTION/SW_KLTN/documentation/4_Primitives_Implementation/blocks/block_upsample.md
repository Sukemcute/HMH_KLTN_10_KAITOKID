# Golden Block: Upsample
## Source: `PHASE1/python_golden/blocks/block_upsample.py`
## Layers: 11, 14

---

## Cấu trúc block trong model

```
Layer 11: Upsample [1, 256, 20, 20] → [1, 256, 40, 40]  scale_factor=2
Layer 14: Upsample [1, 128, 40, 40] → [1, 128, 80, 80]  scale_factor=2
```

---

## Primitives được sử dụng

```
P6 (upsample_nearest) – nearest-neighbor ×2, pass-through scale/zp
```

---

## Code đầy đủ: `block_upsample.py`

```python
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primitives.primitive_tensor import upsample_nearest

def block_upsample(
    X_int8: np.ndarray,    # [N, C, H, W] int8
    scale_x: float,        # pass-through (KHÔNG thay đổi)
    zp_x: int,             # pass-through (KHÔNG thay đổi)
    scale_factor: int = 2,
) -> tuple:                # (Y_int8, scale_x, zp_x)
    """
    Mapping cho Upsample block (nearest-neighbor ×2).
    Scale/ZP được pass-through – không đổi.
    """
    return upsample_nearest(X_int8, scale_x, zp_x, scale_factor=scale_factor)
```

---

## Cơ chế nearest-neighbor

```python
# upsample_nearest implementation:
sf = scale_factor  # = 2
N, C, H, W = X_int8.shape
Y = np.zeros((N, C, H*sf, W*sf), dtype=np.int8)

for di in range(sf):      # di = 0, 1
    for dj in range(sf):  # dj = 0, 1
        Y[:, :, di::sf, dj::sf] = X_int8
# Kết quả: mỗi pixel được copy ra 2×2=4 vị trí
```

**Ví dụ với sf=2:**
```
Input pixel X[n,c,i,j] → Output pixels:
  Y[n,c, 2i,   2j  ]
  Y[n,c, 2i,   2j+1]
  Y[n,c, 2i+1, 2j  ]
  Y[n,c, 2i+1, 2j+1]
```

---

## Luồng dữ liệu

```
X_int8 [N, C, H, W]   (scale_x, zp_x)
    │
    └─[P6] Nearest-neighbor copy: mỗi pixel → 2×2 block
    │
Y_int8 [N, C, H*2, W*2]   (scale_x, zp_x)   ← scale/zp KHÔNG ĐỔI!
```

---

## Cách gọi từ verify scripts

```python
# Layer 11: Upsample [1,256,20,20] → [1,256,40,40]
m = model.model.model[11]
ref_in = getattr(res, "Layer10")()   # output của Layer 10

in_signed = to_signed_int8(ref_in.int_repr().numpy())
in_scale  = float(ref_in.q_scale())
in_zp     = int(ref_in.q_zero_point()) - 128

my_out, _, _ = block_upsample(in_signed, in_scale, in_zp, scale_factor=2)
```

---

## Kết quả verify

| Layer | Shape In | Shape Out | Mean Match | Max Diff |
|:------|:---------|:----------|:-----------|:---------|
| L11 | [1,256,20,20] | [1,256,40,40] | **100.00%** | 0 LSB |
| L14 | [1,128,40,40] | [1,128,80,80] | **100.00%** | 0 LSB |

**100% bit-exact** vì chỉ là address remapping, không có computation.

---

## Import chain

```
block_upsample.py
  └── primitive_tensor.upsample_nearest  (P6)
        (không cần quant_affine)
```
