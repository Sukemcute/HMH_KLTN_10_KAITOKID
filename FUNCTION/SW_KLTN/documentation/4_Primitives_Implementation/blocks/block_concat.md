# Golden Block: Concat (QConcat)
## Source: `PHASE1/python_golden/blocks/block_concat.py`
## Layers: 12, 15, 18, 21

---

## Cấu trúc block trong model

```
Layer 12: QConcat [1,256,40,40] + [1,128,40,40] → [1,384,40,40]
          (from Layer 11 + Layer 6)
Layer 15: QConcat [1,128,80,80] + [1,64,80,80]  → [1,192,80,80]
          (from Layer 14 + Layer 4)
Layer 18: QConcat [1,64,40,40]  + [1,128,40,40] → [1,192,40,40]
          (from Layer 17 + Layer 13)
Layer 21: QConcat [1,128,20,20] + [1,256,20,20] → [1,384,20,20]
          (from Layer 20 + Layer 10)
```

Skip connection map (cố định trong tất cả verify scripts):
```python
concat_map = {12: [11, 6], 15: [14, 4], 18: [17, 13], 21: [20, 10]}
```

---

## Primitives được sử dụng

```
P5 (concat) – với domain alignment (requant về common scale)
P12 (PPU)   – thông qua requant_to_common bên trong align_and_concat
```

---

## Code đầy đủ: `block_concat.py`

```python
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primitives.primitive_tensor import concat

def block_concat(
    tensors: list,         # list of np.ndarray int8 – [Tensor_A, Tensor_B, ...]
    scales: list,          # [scale_A, scale_B, ...] – scale per tensor
    zps: list,             # [zp_A, zp_B, ...] – zp per tensor
    concat_params: dict,   # {"scale_out": float, "zp_out": int,
                           #  "strategy": "offline"}
) -> tuple:                # (Y_int8, scale_out, zp_out)
    """
    Mapping cho QConcat block (channel-wise concatenation với domain align).
    """
    return concat(
        tensors,
        scales=scales,
        zps=zps,
        **concat_params
    )
```

---

## Pipeline bên trong `concat` → `align_and_concat`

```python
# primitive_tensor.concat gọi:
def align_and_concat(tensors_int8, scales, zps, axis=1,
                     strategy="offline", scale_out=None, zp_out=None):

    # Bước 1: Xác định common domain
    common_scale, common_zp = scale_out, zp_out  # "offline" → từ model

    # Bước 2: Requant từng tensor về common domain
    aligned = []
    for t, s, z in zip(tensors_int8, scales, zps):
        # requant_to_common: dequant → float → quantize
        x_float = (t.astype(np.float64) - z) * s
        t_aligned = quantize_affine(x_float, common_scale, common_zp, "int8")
        aligned.append(t_aligned)

    # Bước 3: Concatenate
    Y_int8 = np.concatenate(aligned, axis=1).astype(np.int8)
    return Y_int8, common_scale, common_zp
```

---

## Luồng dữ liệu

```
Tensor_A [N, Ca, H, W]  (scale_A, zp_A)    – từ layer gần nhất
Tensor_B [N, Cb, H, W]  (scale_B, zp_B)    – từ skip connection
    │
    ├─[P5] Requant A: float_A = (A - zp_A) × scale_A
    │      Reint A:   A_aligned = round(float_A / scale_out) + zp_out → clip INT8
    │
    ├─[P5] Requant B: tương tự
    │
    └─ np.concatenate([A_aligned, B_aligned], axis=1)
    │
Y_int8 [N, Ca+Cb, H, W]  (scale_out, zp_out)
```

---

## Cách gọi từ verify scripts

```python
# Layer 12: Concat [Layer11 + Layer6] → [1, 384, 40, 40]
concat_map = {12: [11, 6], 15: [14, 4], 18: [17, 13], 21: [20, 10]}

# Lấy tensors input từ model results
in_signed = [
    to_signed_int8(getattr(res, f"Layer{idx}")().int_repr().numpy())
    for idx in concat_map[12]    # [Layer11, Layer6]
]
in_scale = [
    float(getattr(res, f"Layer{idx}")().q_scale())
    for idx in concat_map[12]
]
in_zp = [
    int(getattr(res, f"Layer{idx}")().q_zero_point()) - 128
    for idx in concat_map[12]
]

# QConcat params từ model (model.model.model[12])
m = model.model.model[12]
c_p = {
    "scale_out": float(m.scale),    # model's observer output scale
    "zp_out":    int(m.zero_point) - 128,
    "strategy":  "offline",
}

my_out, _, _ = block_concat(in_signed, in_scale, in_zp, c_p)
```

---

## Tại sao cần Domain Alignment?

Hai tensors từ các nhánh khác nhau có scale/zp khác nhau:
```
Layer 11 (Upsample output):  scale=0.1471, zp=-20
Layer 6  (QC2f output):      scale=0.0784, zp=-8
```
→ Nếu concat trực tiếp (không align), kết quả sẽ **sai về mặt numeric** vì channel 0-255 và channel 256-383 dùng scale khác nhau → model không thể xử lý đúng.

→ Phải requant tất cả về `scale_out = 0.1471` (hoặc scale_out từ model) trước khi join.

---

## Kết quả verify

| Layer | Tensor A | Tensor B | Shape Out | Mean Match | Max Diff |
|:------|:---------|:---------|:----------|:-----------|:---------|
| L12 | [1,256,40,40] | [1,128,40,40] | [1,384,40,40] | **100.00%** | 0 LSB |
| L15 | [1,128,80,80] | [1,64,80,80] | [1,192,80,80] | **100.00%** | 0 LSB |
| L18 | [1,64,40,40] | [1,128,40,40] | [1,192,40,40] | **100.00%** | 0 LSB |
| L21 | [1,128,20,20] | [1,256,20,20] | [1,384,20,20] | **100.00%** | 0 LSB |

**100% bit-exact** – Golden Path (float intermediate) khớp hoàn toàn với PyTorch `FloatFunctional.cat`.

---

## Import chain

```
block_concat.py
  └── primitive_tensor.concat              (P5)
        └── quant_domain_align.align_and_concat
              └── quant_affine.{quantize_affine, dequantize_affine}
```
