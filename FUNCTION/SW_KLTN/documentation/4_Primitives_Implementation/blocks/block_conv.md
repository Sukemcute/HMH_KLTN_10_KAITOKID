# Golden Block: Conv

## Source: `PHASE1/python_golden/blocks/block_conv.py`

## Layers: 0, 1, 3, 17

---

## Cấu trúc block trong model

```
Layer 0:  Conv [1,3,640,640]   → [1,16,320,320]  kernel=3, stride=2
Layer 1:  Conv [1,16,320,320]  → [1,32,160,160]  kernel=3, stride=2
Layer 3:  Conv [1,32,160,160]  → [1,64,80,80]    kernel=3, stride=2
Layer 17: Conv [1,64,80,80]    → [1,64,40,40]    kernel=3, stride=2
```

---

## Primitives được sử dụng

```
P0 (rs_dense_3x3)  – nếu kernel = 3
P1 (os_1x1)        – nếu kernel = 1
P12 (REQUANT/PPU)  – tích hợp bên trong P0/P1
P14 (ReLU)         – tích hợp bên trong P0/P1
```

---

## Code đầy đủ: `block_conv.py`

```python
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primitives.primitive_conv import rs_dense_3x3, os_1x1

def block_conv(
    X_int8: np.ndarray,    # [N, Cin, H, W] int8 – input
    W_int8: np.ndarray,    # [Cout, Cin, k, k] int8 – weight
    B_int32: np.ndarray,   # [Cout] int32 – bias (BN fused)
    scale_x: float,        # input scale
    zp_x: int,             # input zero-point (signed: pytorch_zp - 128)
    scale_w: np.ndarray,   # [Cout] float – per-channel weight scale
    zp_w: int,             # weight zero-point (MUST = 0)
    scale_y: float,        # output scale
    zp_y: int,             # output zero-point (signed)
    stride: int = 1,       # 1 hoặc 2
    padding: str = "same", # "same" hoặc int
    activation: str = "relu",  # "relu"|"silu"|"relu6"|"none"
    dump: bool = False,
) -> tuple:                # (Y_int8, scale_y, zp_y)
    """
    Mapping cho YOLO Conv block (Conv + BN_fused + Activation).
    Tự động chọn primitive dựa trên kernel size.
    """
    k = W_int8.shape[2]   # kernel size từ weight shape

    if k == 3:
        return rs_dense_3x3(
            X_int8, W_int8, B_int32,
            scale_x, zp_x, scale_w, zp_w, scale_y, zp_y,
            stride=stride, padding=padding, activation=activation, dump=dump
        )
    elif k == 1:
        return os_1x1(
            X_int8, W_int8, B_int32,
            scale_x, zp_x, scale_w, zp_w, scale_y, zp_y,
            activation=activation, dump=dump
        )
    else:
        raise ValueError(f"Unsupported kernel size {k}")
```

---

## Luồng dữ liệu

```
X_int8 [N, Cin, H, W]   (scale_x, zp_x)
    │
    ├─ detect kernel size từ W_int8.shape[2]
    │
    ├─ kernel=3: rs_dense_3x3
    │     ├─ Pad với zp_x (pad=1 cho same)
    │     ├─ MAC INT64: Σ X_pad × W
    │     ├─ Correction: -zp_x×ΣW + B
    │     ├─ PPU: (Acc×M_int + offset)>>shift + zp_y → clip INT8
    │     └─ ReLU: max(y, quantize(0.0))
    │
    └─ kernel=1: os_1x1
          ├─ MAC INT64 (no padding)
          ├─ Correction + bias
          ├─ PPU requant
          └─ Activation
    │
Y_int8 [N, Cout, Hout, Wout]   (scale_y, zp_y)
  Hout = (H+2*pad-k)//stride+1
```

---

## Cách extract params từ model (từ verify scripts)

```python
def get_conv_params(m, in_scale, in_zp, is_3x3=True):
    """
    m: model.model.model[layer_idx]
    in_scale: scale của tensor input (từ layer trước)
    in_zp: zero-point đã -128 của tensor input
    """
    return {
        "W_int8":  m.conv.weight().int_repr().numpy(),
        "B_int32": m.conv.bias().numpy(),
        "scale_x": in_scale,
        "zp_x":    in_zp,
        "scale_w": m.conv.weight().q_per_channel_scales().numpy(),
        "zp_w":    0,   # symmetric weight
        "scale_y": float(m.conv.scale),
        "zp_y":    int(m.conv.zero_point) - 128,   # PyTorch → signed
        "stride":  m.conv.stride[0],
        "activation": "relu" if isinstance(m.act, nn.ReLU) else
                       "silu" if isinstance(m.act, nn.SiLU) else "none",
    }
```

---

## Ví dụ verify Layer 0 đầy đủ

```python
# Layer 0: Conv [1,3,640,640] → [1,16,320,320]
m = model.model.model[0]
node = getattr(res, "Layer0")
ref_out_signed = to_signed_int8(node().int_repr().numpy())

# Input: quantized image
ref_in = model.model.quant(input_tensor)   # Layer 0 dùng quantized input
in_signed = to_signed_int8(ref_in.int_repr().numpy())
in_scale = float(ref_in.q_scale())
in_zp = int(ref_in.q_zero_point()) - 128  # -128 shift

# Run Golden Block
is_3x3 = (m.conv.kernel_size[0] == 3)     # True
my_out, _, _ = block_conv(
    in_signed, padding="same",
    **get_conv_params(m, in_scale, in_zp, is_3x3=is_3x3)
)

# So sánh với PyTorch reference
diff = np.abs(my_out.astype(np.int16) - ref_out_signed.astype(np.int16))
match_pct = (diff == 0).sum() / diff.size * 100
# Expected: Layer0 ≈ 99.99% match, max_diff = 1 LSB
```

---

## Kết quả verify (100 samples, 640×640)

| Layer | Shape Out      | Mean Match | Max Diff |
| :---- | :------------- | :--------- | :------- |
| L0    | [1,16,320,320] | **99.99%** | 1 LSB    |
| L1    | [1,32,160,160] | **99.96%** | 1 LSB    |
| L3    | [1,64,80,80]   | **99.98%** | 1 LSB    |
| L17   | [1,64,40,40]   | **99.99%** | 1 LSB    |

1 LSB diff do hardware rounding tại biên ±127/±128.

---

## Import chain

```
block_conv.py
  ├── primitive_conv.rs_dense_3x3   (P0)
  └── primitive_conv.os_1x1         (P1)
        ├── quant_affine.make_requant_params
        ├── quant_affine.post_process_int32_to_int8  (P12)
        ├── quant_affine.apply_silu_float            (P13)
        └── quant_affine.quantize_affine (cho ReLU clip) (P14)
```
