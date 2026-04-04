# Golden Block: QPSA
## Source: `PHASE1/python_golden/blocks/block_qpsa.py`
## Layer: 10

---

## Cấu trúc block trong model

```
Layer 10: QPSA [1, 256, 20, 20] → [1, 256, 20, 20]
          (Quantized Position-Sensitive Attention)
```

---

## Primitives được sử dụng

```
P1  (os_1x1)              – cv1, QKV proj, output proj, FFN ×2, cv2
P2  (dw_3x3)              – Positional Encoding trong QAttention
P5  (concat)              – ghép branch a + branch b_final
P7  (ewise_add)           – shortcut add ×2 bên trong attention path
P10 (_int8_matmul)        – Q×K^T và V×Attn^T
P11 (_softmax_int8_approx)– normalize attention scores
P12 (PPU)                 – tích hợp trong tất cả các ops
```

---

## Code đầy đủ: `block_qpsa.py`

### Hàm phụ: `block_qattention`

```python
from primitives.primitive_conv import os_1x1
from primitives.primitive_dw import dw_3x3
from primitives.primitive_tensor import ewise_add, concat
from primitives.primitive_psa import _int8_matmul, _softmax_int8_approx
from quant.quant_affine import post_process_int32_to_int8, make_requant_params
from quant.quant_affine import quantize_affine, dequantize_affine

def block_qattention(
    X_int8: np.ndarray,       # [B, C, H, W] int8 – branch b input
    scale_x: float,
    zp_x: int,
    qkv_params: dict,         # OS_1x1 params cho joint QKV projection
    proj_params: dict,        # OS_1x1 params cho output projection
    pe_params: dict,          # DW_3x3 params cho positional encoding
    sm_params: dict,          # {"scale_out", "zp_out"} softmax output
    matmul1_params: dict,     # {"key_dim", "head_dim", "head_scale",
                              #  "scale_out", "zp_out"}
    matmul2_params: dict,     # {"head_dim", "scale_out", "zp_out"}
    add_pe_params: dict,      # {"scale_out", "zp_out"} cho add(x_attn, PE)
    num_heads: int,
    dump: bool = False,
) -> tuple:

    B, C, H, W = X_int8.shape
    N = H * W   # sequence length = 400 cho 20×20

    # ── Bước 1: Joint QKV Projection – P1 ───────────────────────────────
    split_sizes = qkv_params.pop("split_sizes", [32, 32, 64])
    y_qkv, s_qkv, z_qkv = os_1x1(X_int8, **qkv_params)
    # y_qkv: [B, 2*kd+hd, H, W]

    # ── Bước 2: Reshape + Split → Q, K, V ───────────────────────────────
    kd = matmul1_params["key_dim"]    # e.g. 32
    hd = matmul2_params["head_dim"]   # e.g. 64
    y_reshaped = y_qkv.reshape(B, num_heads, kd*2+hd, N)
    q = y_reshaped[:, :, :kd,      :]   # [B, heads, kd, N]
    k = y_reshaped[:, :, kd:kd*2,  :]   # [B, heads, kd, N]
    v = y_reshaped[:, :, kd*2:,    :]   # [B, heads, hd, N]

    # ── Bước 3: Matmul 1: q^T × k → Attention scores – P10 ─────────────
    q_t = q.transpose(0, 1, 3, 2)       # [B, heads, N, kd]
    amul_raw = _int8_matmul(q_t, k, zp_A=z_qkv, zp_B=z_qkv)
    # amul_raw: [B, heads, N, N] int64

    # Requant về float domain, scale = s_qkv * s_qkv
    amul_float = amul_raw.astype(np.float64) * (s_qkv * s_qkv)
    amul_i8 = quantize_affine(amul_float,
                               matmul1_params["scale_out"],
                               matmul1_params["zp_out"], dtype="int8")

    # Scale bởi head_scale = 1/sqrt(kd)
    amul_f = dequantize_affine(amul_i8,
                                matmul1_params["scale_out"],
                                matmul1_params["zp_out"])
    attn_float = amul_f * matmul1_params["head_scale"]
    attn_score = quantize_affine(attn_float,
                                  matmul1_params["scale_out"],
                                  matmul1_params["zp_out"], dtype="int8")

    # ── Bước 4: Softmax – P11 ────────────────────────────────────────────
    attn_soft = _softmax_int8_approx(
        attn_score,
        matmul1_params["scale_out"], matmul1_params["zp_out"],
        sm_params["scale_out"], sm_params["zp_out"]
    )   # [B, heads, N, N] int8

    # ── Bước 5: Matmul 2: v × attn_soft^T – P10 ─────────────────────────
    attn_soft_t = attn_soft.transpose(0, 1, 3, 2)  # [B, heads, N, N]
    x_raw = _int8_matmul(v, attn_soft_t,
                          zp_A=z_qkv, zp_B=sm_params["zp_out"])
    # x_raw: [B, heads, hd, N] int64

    # Requant
    x_f = x_raw.astype(np.float64) * (s_qkv * sm_params["scale_out"])
    x_attn_i8 = quantize_affine(x_f,
                                  matmul2_params["scale_out"],
                                  matmul2_params["zp_out"], dtype="int8")

    # ── Bước 6: Reshape về spatial ───────────────────────────────────────
    x_spatial = x_attn_i8.reshape(B, C, H, W)  # [B, C, H, W]

    # ── Bước 7: Positional Encoding – P2 ─────────────────────────────────
    # PE được apply lên V (không phải lên output attention!)
    v_spatial = v.reshape(B, C, H, W)
    y_pe, s_pe, z_pe = dw_3x3(v_spatial, **pe_params)

    # ── Bước 8: Add PE – P7 ──────────────────────────────────────────────
    x_fused, s_fused, z_fused = ewise_add(
        x_spatial, matmul2_params["scale_out"], matmul2_params["zp_out"],
        y_pe, s_pe, z_pe,
        scale_out=add_pe_params["scale_out"],
        zp_out=add_pe_params["zp_out"]
    )

    # ── Bước 9: Output Projection – P1 ───────────────────────────────────
    y_out, s_out, z_out = os_1x1(x_fused, **proj_params)

    return y_out, s_out, z_out
```

### Hàm chính: `block_qpsa`

```python
def block_qpsa(
    X_int8: np.ndarray,        # [N, 256, 20, 20] int8
    cv1_params: dict,          # OS_1x1 cho initial split conv
    attn_params: dict,         # tất cả params cho block_qattention
    ffn_params: list,          # [ffn1_params, ffn2_params] cho 2 OS_1x1
    ffn_add_params: dict,      # {"scale_out", "zp_out"} cho shortcut 2
    attn_add_params: dict,     # {"scale_out", "zp_out"} cho shortcut 1
    cv2_params: dict,          # OS_1x1 cho final projection
    concat_params: dict,       # cho concat(a, b_final)
    dump: bool = False,
) -> tuple:                    # (Y_int8, scale_y, zp_y)

    # 1. Initial Conv + Split – P1
    y_cv1, s_cv1, z_cv1 = os_1x1(X_int8, **cv1_params)
    mid = y_cv1.shape[1] // 2
    a = y_cv1[:, :mid, :, :]   # [1, 128, 20, 20] – identity branch
    b = y_cv1[:, mid:, :, :]   # [1, 128, 20, 20] – attention branch

    # 2. Attention path
    y_attn, s_attn, z_attn = block_qattention(b, s_cv1, z_cv1, **attn_params)

    # 3. Shortcut 1: b = b + attention(b) – P7
    b_attn, s_b_attn, z_b_attn = ewise_add(
        b, s_cv1, z_cv1,
        y_attn, s_attn, z_attn,
        **attn_add_params
    )

    # 4. FFN path – 2× P1
    y_f1, s_f1, z_f1 = os_1x1(b_attn, **ffn_params[0])
    y_f2, s_f2, z_f2 = os_1x1(y_f1, **ffn_params[1])

    # 5. Shortcut 2: b = b_attn + FFN(b_attn) – P7
    b_final, s_b_final, z_b_final = ewise_add(
        b_attn, s_b_attn, z_b_attn,
        y_f2, s_f2, z_f2,
        **ffn_add_params
    )

    # 6. Concat: [a, b_final] – P5
    concat_params.pop("scales", None)   # scales được tính tại runtime
    concat_params.pop("zps", None)
    y_cat, s_cat, z_cat = concat(
        [a, b_final],
        scales=[s_cv1, s_b_final],
        zps=[z_cv1, z_b_final],
        **concat_params
    )

    # 7. Final Projection – P1
    y_out, s_out, z_out = os_1x1(y_cat, **cv2_params)
    return y_out, s_out, z_out
```

---

## Luồng dữ liệu tổng thể

```
X [1,256,20,20]
    │
    ├─[P1] cv1 → [1,256,20,20] → split:
    │      a=[1,128,20,20] (identity)
    │      b=[1,128,20,20] (→ attention)
    │
    │  === Attention Branch (b) ===
    ├─[P1] QKV Proj: b → [1, 128, 20, 20] (joint q+k+v)
    │       reshape → q[1,1,32,400], k[1,1,32,400], v[1,1,64,400]
    │
    ├─[P10] Matmul1: q_t×k → [1,1,400,400] → requant INT8
    │        scale × head_scale (1/√32)
    ├─[P11] Softmax → [1,1,400,400] int8
    ├─[P10] Matmul2: v×soft_t → [1,1,64,400] → requant INT8
    │        reshape → [1,128,20,20]
    │
    ├─[P2]  PE: DW_3x3(v_spatial) → [1,128,20,20]
    ├─[P7]  Add: x_attn + PE → [1,128,20,20]
    ├─[P1]  Output proj → [1,128,20,20]
    │  === End Attention ===
    │
    ├─[P7] Shortcut1: b + attn_out → b_attn [1,128,20,20]
    ├─[P1] FFN1: OS_1x1 b_attn → [1,256,20,20]
    ├─[P1] FFN2: OS_1x1 → [1,128,20,20]
    ├─[P7] Shortcut2: b_attn + FFN → b_final [1,128,20,20]
    │
    ├─[P5] Concat: [a, b_final] → [1,256,20,20]
    └─[P1] cv2 → [1,256,20,20]
    │
Output [1,256,20,20]
```

---

## Kết quả verify Layer 10

| Metric | Kết quả |
|:-------|:--------|
| Input/Output shape | [1, 256, 20, 20] |
| Mean Match | **83.52%** |
| Max Diff | **23 LSB** |
| Lý do match thấp hơn | ~14 operations ghép, error tích lũy qua matmul + softmax |

---

## Import chain

```
block_qpsa.py
  ├── primitive_conv.os_1x1               (P1)
  ├── primitive_dw.dw_3x3                 (P2)
  ├── primitive_tensor.ewise_add          (P7)
  ├── primitive_tensor.concat             (P5)
  ├── primitive_psa._int8_matmul          (P10)
  ├── primitive_psa._softmax_int8_approx  (P11)
  └── quant_affine.{quantize_affine, dequantize_affine,
                    post_process_int32_to_int8, make_requant_params}
```
