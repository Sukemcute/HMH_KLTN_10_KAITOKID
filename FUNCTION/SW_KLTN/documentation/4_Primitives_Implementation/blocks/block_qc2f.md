# Golden Block: QC2f
## Source: `PHASE1/python_golden/blocks/block_qc2f.py`
## Layers: 2, 4, 6, 8, 13, 16, 19

---

## Cấu trúc block trong model

```
Layer 2:  QC2f [1,32,160,160]  → [1,32,160,160]   n=1 bottleneck
Layer 4:  QC2f [1,64,80,80]    → [1,64,80,80]     n=2 bottlenecks
Layer 6:  QC2f [1,128,40,40]   → [1,128,40,40]    n=2 bottlenecks
Layer 8:  QC2f [1,256,20,20]   → [1,256,20,20]    n=2 bottlenecks
Layer 13: QC2f [1,384,40,40]   → [1,128,40,40]    n=2 bottlenecks
Layer 16: QC2f [1,192,80,80]   → [1,64,80,80]     n=2 bottlenecks
Layer 19: QC2f [1,192,40,40]   → [1,128,40,40]    n=2 bottlenecks
```

---

## Primitives được sử dụng

```
P1  (os_1x1)     – cv1 (expand) và cv2 (compress)
P0  (rs_dense_3x3) – conv1 và conv2 bên trong mỗi QBottleneck
P7  (ewise_add)  – shortcut bên trong QBottleneck (nếu shortcut=True)
P5  (concat)     – ghép y[0], y[1], y_m[0], y_m[1], ...
P12 (PPU)        – tích hợp bên trong P0/P1
```

---

## Code đầy đủ: `block_qc2f.py`

### Hàm phụ: `block_qbottleneck`

```python
def block_qbottleneck(
    X_int8: np.ndarray,      # [N, C, H, W] int8
    cv1_params: dict,        # params cho rs_dense_3x3 (conv1)
    cv2_params: dict,        # params cho rs_dense_3x3 (conv2)
    shortcut: bool = False,  # True = có residual add
    add_params: dict = None, # nếu shortcut=True: {"scale_out", "zp_out"}
    dump: bool = False,
) -> tuple:                  # (Y_int8, scale_y, zp_y)
    """
    QBottleneck: conv1(3x3) → conv2(3x3) [+ shortcut add]
    """
    # 1. First Conv (3x3) – P0
    y1, s1, z1 = rs_dense_3x3(X_int8, **cv1_params)

    # 2. Second Conv (3x3) – P0
    y2, s2, z2 = rs_dense_3x3(y1, **cv2_params)

    if shortcut:
        # Shortcut: X + cv2_output – P7
        # A = X_int8 (input), B = y2 (after 2 convs)
        out, sout, zout = ewise_add(
            A_int8 = X_int8,
            scale_A = cv1_params["scale_x"],  # scale của X input
            zp_A    = cv1_params["zp_x"],
            B_int8  = y2,
            scale_B = s2,
            zp_B    = z2,
            scale_out = add_params["scale_out"],
            zp_out    = add_params["zp_out"],
            strategy  = "offline"
        )
        return out, sout, zout

    return y2, s2, z2
```

### Hàm chính: `block_qc2f`

```python
def block_qc2f(
    X_int8: np.ndarray,          # [N, Cin, H, W] int8 – input
    cv1_params: dict,            # params cho os_1x1 (expand 1x1)
    cv2_params: dict,            # params cho os_1x1 (compress 1x1)
    bottleneck_list_params: list,# list of dict, mỗi dict chứa:
                                 #   {"cv1_params": {...},
                                 #    "cv2_params": {...},
                                 #    "shortcut": bool,
                                 #    "add_params": {...} hoặc None}
    concat_params: dict,         # {"scales": [...], "zps": [...],
                                 #  "scale_out": float, "zp_out": int,
                                 #  "strategy": "offline"}
    dump: bool = False,
) -> tuple:                      # (Y_int8, scale_y, zp_y)
    """
    Mapping cho QC2f block (Quantized Cross-Stage Partial with 2 convs).

    Pipeline:
    1. cv1 (1x1) → expand channels [Cin → 2*c_]
    2. Split thành y[0] và y[1] (channel dim)
    3. y[1] → n QBottlenecks → y_m[0], y_m[1], ...
    4. CONCAT: [y[0], y[1], y_m[0], ...] → domain align
    5. cv2 (1x1) → compress [sum_channels → Cout]
    """
    # 1. Initial 1x1 Conv – P1
    y_cv1, s_cv1, z_cv1 = os_1x1(X_int8, **cv1_params)
    # y_cv1: [N, 2*c_, H, W]

    # 2. Split thành 2 halves
    mid = y_cv1.shape[1] // 2
    y_split = [y_cv1[:, :mid, :, :],   # y[0]: identity branch
               y_cv1[:, mid:, :, :]]   # y[1]: processed branch

    # 3. Chạy qua N bottlenecks
    y_m = []
    current_input = y_split[1]
    for b_params in bottleneck_list_params:
        out, s_out, z_out = block_qbottleneck(
            current_input,
            cv1_params=b_params["cv1_params"],
            cv2_params=b_params["cv2_params"],
            shortcut=b_params["shortcut"],
            add_params=b_params.get("add_params")
        )
        y_m.append(out)
        current_input = out  # output của bn này = input của bn tiếp theo

    # 4. Concat tất cả – P5
    # Tensors: [y[0], y[1], m[0], m[1], ...]
    to_concat = y_split + y_m
    y_cat, s_cat, z_cat = concat(
        to_concat,
        scales=concat_params["scales"],
        zps=concat_params["zps"],
        strategy=concat_params.get("strategy", "offline"),
        scale_out=concat_params.get("scale_out"),
        zp_out=concat_params.get("zp_out")
    )

    # 5. Final 1x1 Conv – P1
    y_out, s_out, z_out = os_1x1(y_cat, **cv2_params)

    return y_out, s_out, z_out
```

---

## Luồng dữ liệu chi tiết

```
X_int8 [N, Cin, H, W]   (scale_x, zp_x)
    │
    ├─[P1] cv1: OS_1x1 → [N, 2*c_, H, W]   (s_cv1, z_cv1)
    │
    ├─ Split channel:
    │      y[0] = [:, :c_, :, :]    → [N, c_, H, W]  identity branch
    │      y[1] = [:, c_:, :, :]    → [N, c_, H, W]  input to bottleneck 0
    │
    ├─ QBottleneck 0:
    │   ├─[P0] conv1: RS_DENSE_3x3(y[1]) → [N, c_, H, W]
    │   ├─[P0] conv2: RS_DENSE_3x3(^)    → [N, c_, H, W]
    │   └─[P7] ewise_add(y[1], ^)        → [N, c_, H, W]  (nếu shortcut)
    │       = y_m[0]
    │
    ├─ QBottleneck 1 (nếu n=2):
    │   ├─[P0] conv1: RS_DENSE_3x3(y_m[0])
    │   ├─[P0] conv2: RS_DENSE_3x3(^)
    │   └─[P7] ewise_add(y_m[0], ^)
    │       = y_m[1]
    │
    ├─[P5] CONCAT: [y[0], y[1], y_m[0], y_m[1]] + Domain Align
    │      → [N, Cin + c_*n_bn, H, W]   (s_cat, z_cat)
    │
    └─[P1] cv2: OS_1x1 → [N, Cout, H, W]   (scale_y, zp_y)
    │
Output [N, Cout, H, W]   (scale_y, zp_y)
```

---

## Cấu trúc `bottleneck_list_params`

```python
# Ví dụ cho QC2f với 2 bottlenecks, shortcut=True
bottleneck_list_params = [
    {
        "cv1_params": {
            # RS_DENSE_3x3 params cho conv1 của bottleneck 0
            "W_int8":  m.m[0].cv1.conv.weight().int_repr().numpy(),
            "B_int32": m.m[0].cv1.conv.bias().numpy(),
            "scale_x": s_cv1,            # scale output của cv1
            "zp_x":    z_cv1,
            "scale_w": m.m[0].cv1.conv.weight().q_per_channel_scales().numpy(),
            "zp_w":    0,
            "scale_y": float(m.m[0].cv1.conv.scale),
            "zp_y":    int(m.m[0].cv1.conv.zero_point) - 128,
            "stride":  1,
            "activation": "relu",
        },
        "cv2_params": {
            # RS_DENSE_3x3 params cho conv2 của bottleneck 0
            ...
        },
        "shortcut": True,
        "add_params": {
            "scale_out": float(m.m[0].add.scale),
            "zp_out":    int(m.m[0].add.zero_point) - 128,
        },
    },
    {
        # bottleneck 1...
    }
]
```

---

## Cấu trúc `concat_params`

```python
# concat_params phải chứa scale của TẤT CẢ tensors được concat
# Thứ tự: [y[0], y[1], y_m[0], y_m[1], ...]
concat_params = {
    "scales":    [s_cv1, s_cv1, s_bn0, s_bn1],  # s_cv1 cho y[0],y[1]
    "zps":       [z_cv1, z_cv1, z_bn0, z_bn1],
    "scale_out": float(m.fl.scale),    # từ model's FloatFunctional
    "zp_out":    int(m.fl.zero_point) - 128,
    "strategy":  "offline",
}
```

---

## Kết quả verify

| Layer | n_bottleneck | Shape Out | Mean Match | Max Diff |
|:------|:-------------|:----------|:-----------|:---------|
| L2 | 1 | [1,32,160,160] | **99.09%** | 3 LSB |
| L4 | 2 | [1,64,80,80] | **96.52%** | 3 LSB |
| L6 | 2 | [1,128,40,40] | **94.46%** | 3 LSB |
| L8 | 2 | [1,256,20,20] | **99.20%** | 2 LSB |
| L13 | 2 | [1,128,40,40] | **98.53%** | 2 LSB |
| L16 | 2 | [1,64,80,80] | **99.02%** | 9 LSB |
| L19 | 2 | [1,128,40,40] | **98.94%** | 3 LSB |

Sai số tích lũy theo số bottlenecks và depth của network.

---

## Import chain

```
block_qc2f.py
  ├── primitive_conv.rs_dense_3x3   (P0)
  ├── primitive_conv.os_1x1         (P1)
  └── primitive_tensor.concat       (P5)
       ├── primitive_tensor.ewise_add  (P7) – qua block_qbottleneck
       └── quant_domain_align.align_and_concat
             └── quant_affine.*
```
