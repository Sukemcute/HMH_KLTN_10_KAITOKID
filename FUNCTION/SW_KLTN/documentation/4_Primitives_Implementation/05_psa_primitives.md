# P9, P10, P11 – PSA / Attention Primitives

## Source file: `PHASE1/python_golden/primitives/primitive_psa.py`

---

## Tổng quan

Ba primitives này chỉ được dùng trong **Layer 10 (QPSA)** – block phức tạp nhất của model.

| Primitive | Tên             | Hàm                    | Vai trò                                |
| :-------- | :-------------- | :--------------------- | :------------------------------------- |
| **P10**   | INT8_MATMUL     | `_int8_matmul`         | Q×K^T và Attn×V                        |
| **P11**   | SOFTMAX_APPROX  | `_softmax_int8_approx` | Normalize attention scores             |
| **P9**    | GEMM_ATTN_BASIC | `gemm_attn_basic`      | Orchestrate toàn bộ attention pipeline |

**Lưu ý:** Trong codebase thực, QPSA dùng `block_qpsa.py` với `block_qattention()`, không gọi trực tiếp `gemm_attn_basic()`. Nhưng `_int8_matmul` và `_softmax_int8_approx` được gọi từ cả hai.

---

## P10 – `_int8_matmul` (INT8_MATMUL)

### Mục đích

INT8 × INT8 matrix multiplication với INT64 accumulation và zero-point subtraction trước khi nhân.

### Signature

```python
def _int8_matmul(
    A_int8: np.ndarray,  # [..., M, K] int8
    B_int8: np.ndarray,  # [..., K, L] int8
    zp_A: int = 0,       # zero-point của A (PHẢI subtract trước khi nhân!)
    zp_B: int = 0,       # zero-point của B
) -> np.ndarray:         # [..., M, L] int64
```

### Implementation:

```python
def _int8_matmul(A_int8, B_int8, zp_A=0, zp_B=0):
    A = A_int8.astype(np.int64) - zp_A   # subtract ZP trước khi nhân
    B = B_int8.astype(np.int64) - zp_B
    return np.matmul(A, B)               # kết quả int64
```

### Tại sao PHẢI subtract ZP?

Quantized matmul: `A_true = A_int8 - ZP_A`, `B_true = B_int8 - ZP_B`  
Nếu không subtract → attention scores sai hoàn toàn (random values).

Dùng trong `block_qattention.py`:

```python
# Matmul 1: Q^T × K → attention scores
amul_raw = _int8_matmul(q_t, k, zp_A=z_qkv, zp_B=z_qkv)

# Matmul 2: V × Attn^T → context
x_raw = _int8_matmul(v, attn_soft_t, zp_A=z_qkv, zp_B=sm_params["zp_out"])
```

---

## P11 – `_softmax_int8_approx` (SOFTMAX_APPROX)

### Mục đích

Softmax trên INT8 attention scores. Dùng "golden path" (float) để đảm bảo accuracy cho Python reference model. Hardware sẽ dùng LUT/piecewise.

### Signature

```python
def _softmax_int8_approx(
    attn_int8: np.ndarray,  # [N, HW, HW] int8 – attention scores
    scale_attn: float,      # scale của attn_int8
    zp_attn: int,           # zp của attn_int8
    scale_out: float,       # output scale
    zp_out: int,            # output zp
) -> np.ndarray:            # [N, HW, HW] int8
```

### Implementation (golden path):

```python
def _softmax_int8_approx(attn_int8, scale_attn, zp_attn, scale_out, zp_out):
    # 1. Dequant về float
    attn_float = dequantize_affine(attn_int8, scale_attn, zp_attn)

    # 2. Numerically stable softmax (trừ max trên axis cuối)
    attn_max = attn_float.max(axis=-1, keepdims=True)
    attn_exp = np.exp(attn_float - attn_max)
    attn_sum = attn_exp.sum(axis=-1, keepdims=True)
    soft_float = attn_exp / attn_sum

    # 3. Requant về INT8
    return quantize_affine(soft_float, scale_out, zp_out, dtype="int8")
```

---

## P9 – `gemm_attn_basic` (GEMM_ATTN_BASIC)

### Mục đích

Orchestrate toàn bộ pipeline self-attention cơ bản. Đây là phiên bản đơn giản hóa (so với `block_qattention` đầy đủ hơn trong `block_qpsa.py`).

### Signature

```python
def gemm_attn_basic(
    X_int8: np.ndarray,    # [N, C, H, W] int8 – input (Layer10: [1,256,20,20])
    scale_x: float,
    zp_x: int,
    # Projection weights
    W_Q, W_K, W_V, W_out,  # [Hq/Hk/Hv/C, C, 1, 1] int8
    B_Q, B_K, B_V, B_out,  # [Hq/Hk/Hv/C] int32
    # Quant params dict
    scale_params: dict,     # xem bảng keys bên dưới
) -> tuple:                 # (Y_int8, scale_out, zp_out)
```

### `scale_params` dict – các keys bắt buộc:

| Key                       | Ý nghĩa                         |
| :------------------------ | :------------------------------ |
| `scale_Q`, `zp_Q`         | Output của Q projection         |
| `scale_K`, `zp_K`         | Output của K projection         |
| `scale_V`, `zp_V`         | Output của V projection         |
| `scale_wQ`, `zp_wQ`       | Weight scale/zp cho Q (zp=0)    |
| `scale_wK`, `zp_wK`       | Weight scale/zp cho K           |
| `scale_wV`, `zp_wV`       | Weight scale/zp cho V           |
| `scale_wOut`, `zp_wOut`   | Weight scale/zp cho output proj |
| `scale_Attn`, `zp_Attn`   | Requant output của Q×K^T        |
| `scale_Soft`, `zp_Soft`   | Softmax output                  |
| `scale_AttnV`, `zp_AttnV` | Requant output của Attn×V       |
| `scale_out`, `zp_out`     | Final output projection         |

### Pipeline 7 bước (từ `gemm_attn_basic`):

```python
N, C, H, W = X_int8.shape
HW = H * W   # = 400 cho 20×20

# ── Bước 1: Q, K, V Projections (P1 OS_1x1) ──────────────────────────
Q_int8, sq, zq = os_1x1(X_int8, W_Q, B_Q,
                          scale_x, zp_x, sp["scale_wQ"], 0,
                          sp["scale_Q"], sp["zp_Q"])   # [N, Hq, H, W]
K_int8, sk, zk = os_1x1(X_int8, W_K, B_K, ...)       # [N, Hk, H, W]
V_int8, sv, zv = os_1x1(X_int8, W_V, B_V, ...)       # [N, Hv, H, W]

# ── Bước 2: Reshape → sequence ────────────────────────────────────────
Q_seq = Q_int8.reshape(N, Hq, HW).transpose(0,2,1)   # [N, HW, Hq]
K_seq = K_int8.reshape(N, Hk, HW).transpose(0,2,1)   # [N, HW, Hk]
V_seq = V_int8.reshape(N, Hv, HW).transpose(0,2,1)   # [N, HW, Hv]

# ── Bước 3: Attention Matrix Q × K^T ─────────────────────────────────
K_t = K_seq.transpose(0,2,1)                          # [N, Hk, HW]
Attn_raw = _int8_matmul(Q_seq, K_t)                  # [N, HW, HW] int64

# Scale: 1/sqrt(Hq) fused vào requant multiplier
sqrt_Hq = math.sqrt(float(Hq))
scale_Attn_eff = sp["scale_Q"] * sp["scale_K"] / (sp["scale_Attn"] * sqrt_Hq)
scale_Attn_w = np.array([scale_Attn_eff] * HW, dtype=np.float64)

# Requant Attn → INT8 (treat HW as "Cout")
Attn_perm = Attn_raw.transpose(0,2,1)                # [N, HW, HW] trục HW trước
M_attn, shift_attn = make_requant_params(1.0, scale_Attn_w, 1.0)
Attn_i8_perm = post_process_int32_to_int8(Attn_perm, M_attn, shift_attn, sp["zp_Attn"])
Attn_i8 = Attn_i8_perm.transpose(0,2,1)              # [N, HW, HW]

# ── Bước 4: Softmax ────────────────────────────────────────────────────
Attn_soft = _softmax_int8_approx(
    Attn_i8, sp["scale_Attn"], sp["zp_Attn"],
    sp["scale_Soft"], sp["zp_Soft"])                  # [N, HW, HW]

# ── Bước 5: Attn_soft × V ─────────────────────────────────────────────
Out_raw = _int8_matmul(Attn_soft, V_seq)             # [N, HW, Hv] int64

# Requant Out → INT8
scale_AttnV_w = np.array(
    [sp["scale_Soft"] * sp["scale_V"] / sp["scale_AttnV"]] * Hv,
    dtype=np.float64)
Out_perm = Out_raw.transpose(0,2,1)                   # [N, Hv, HW]
M_out, shift_out = make_requant_params(1.0, scale_AttnV_w, 1.0)
Out_i8_perm = post_process_int32_to_int8(Out_perm, M_out, shift_out, sp["zp_AttnV"])
Out_i8 = Out_i8_perm.transpose(0,2,1)                # [N, HW, Hv]

# ── Bước 6: Reshape back → spatial ────────────────────────────────────
Out_spatial = Out_i8.transpose(0,2,1).reshape(N, Hv, H, W)  # [N, Hv, H, W]

# ── Bước 7: Output Projection ─────────────────────────────────────────
Y_int8, scale_out, zp_out = os_1x1(
    Out_spatial, W_out, B_out,
    sp["scale_AttnV"], sp["zp_AttnV"],
    sp["scale_wOut"], sp.get("zp_wOut",0),
    sp["scale_out"], sp["zp_out"])                    # [N, C, H, W]
```

---

## Phiên bản đầy đủ hơn: `block_qattention` trong `block_qpsa.py`

`block_qattention` là phiên bản production được dùng thực tế. Nó **khác** với `gemm_attn_basic` ở một số điểm:

| Khía cạnh           | `gemm_attn_basic`    | `block_qattention`           |
| :------------------ | :------------------- | :--------------------------- |
| QKV projection      | 3 lần `os_1x1` riêng | 1 lần `os_1x1` → split       |
| Reshape             | 2D [HW, H]           | Multi-head [B, heads, kd, N] |
| Matmul              | `Q × K^T` trực tiếp  | Per-head `q_t × k`           |
| Matmul scale        | `scale_Q * scale_K`  | `s_qkv * s_qkv`              |
| PE (Positional Enc) | Không có             | Có: `dw_3x3(v_spatial)`      |
| PE Add              | Không có             | `ewise_add(x_attn, y_pe)`    |

### Pipeline `block_qattention` (phiên bản đầy đủ):

```python
# 1. QKV Joint projection → split
y_qkv, s_qkv, z_qkv = os_1x1(X_int8, **qkv_params)
# split_sizes = [32, 32, 64] (kd, kd, hd)
y_reshaped = y_qkv.reshape(B, num_heads, kd*2+hd, N)
q = y_reshaped[:,:,:kd,:]       # [B, heads, kd, N]
k = y_reshaped[:,:,kd:kd*2,:]
v = y_reshaped[:,:,kd*2:,:]

# 2. Matmul 1: q^T × k → attention score
q_t = q.transpose(0,1,3,2)      # [B, heads, N, kd]
amul_raw = _int8_matmul(q_t, k, zp_A=z_qkv, zp_B=z_qkv)

# 3. Requant + scale by head_scale
amul_float = amul_raw.astype(np.float64) * (s_qkv * s_qkv)
amul_i8 = quantize_affine(amul_float, matmul1_params["scale_out"], ...)
amul_f = dequantize_affine(amul_i8, ...)
attn_score = quantize_affine(amul_f * matmul1_params["head_scale"], ...)

# 4. Softmax
attn_soft = _softmax_int8_approx(attn_score, ...)

# 5. Matmul 2: v × attn_soft^T → context
attn_soft_t = attn_soft.transpose(0,1,3,2)
x_raw = _int8_matmul(v, attn_soft_t, zp_A=z_qkv, zp_B=sm_params["zp_out"])
x_attn_i8 = quantize_affine(x_raw * (s_qkv * sm_params["scale_out"]), ...)

# 6. Reshape + Positional Encoding
x_spatial = x_attn_i8.reshape(B, C, H, W)
v_spatial = v.reshape(B, C, H, W)
y_pe, s_pe, z_pe = dw_3x3(v_spatial, **pe_params)   # P2 DW_3x3

# 7. Add PE: x = x_attn + PE(v)
x_fused, s_fused, z_fused = ewise_add(
    x_spatial, matmul2_params["scale_out"], matmul2_params["zp_out"],
    y_pe, s_pe, z_pe,
    scale_out=add_pe_params["scale_out"], zp_out=add_pe_params["zp_out"])

# 8. Output projection
y_out, s_out, z_out = os_1x1(x_fused, **proj_params)
```

---

## Luồng dữ liệu đầy đủ QPSA (Layer 10)

```
X_int8 [1, 256, 20, 20]
    │
    ├─[P1]   cv1: OS_1x1 → [1, 256, 20, 20]
    ├─       Split → a [1,128,20,20] + b [1,128,20,20]
    │
    │  Branch b (Attention):
    ├─[P1]   QKV: OS_1x1 → [1, 128, 20, 20]
    ├─       Reshape multi-head: q[B,1,32,400], k[B,1,32,400], v[B,1,64,400]
    ├─[P10]  Matmul 1: q_t×k → [B,1,400,400] int64 → requant INT8
    ├─       Scale × head_scale
    ├─[P11]  Softmax → [B,1,400,400] int8
    ├─[P10]  Matmul 2: v×soft_t → [B,1,64,400] int64 → requant INT8
    ├─       Reshape → [1, 64, 20, 20] (spatial)
    ├─[P2]   PE: DW_3x3(v_spatial) → [1, 64, 20, 20]
    ├─[P7]   Add: x_attn + PE(v) → [1, 64, 20, 20]
    ├─[P1]   Proj: OS_1x1 → [1, 128, 20, 20]
    │
    ├─[P7]   Shortcut 1: b + attn_proj → [1, 128, 20, 20]
    ├─[P1]   FFN expand: OS_1x1 → [1, 256, 20, 20]
    ├─[P1]   FFN compress: OS_1x1 → [1, 128, 20, 20]
    ├─[P7]   Shortcut 2: b_attn + FFN → [1, 128, 20, 20]
    │
    ├─[P5]   Concat: [a, b_final] → [1, 256, 20, 20]
    └─[P1]   cv2: OS_1x1 → [1, 256, 20, 20]
    │
Output [1, 256, 20, 20]  ← cùng shape với input
```

---

## Kết quả verify Layer 10

Match rate: **83.52%** (max diff: 23 LSB). Thấp hơn các layer khác do:

- Matmul INT64 rất nhạy cảm với rounding errors ở từng bước
- Softmax float accumulation errors
- Nhiều bước requant ghép liên tiếp (~14 operations)

---

## Import chain

```
block_qpsa.py
  ├── primitive_conv.py: os_1x1
  ├── primitive_dw.py: dw_3x3
  ├── primitive_tensor.py: ewise_add, concat
  ├── primitive_psa.py: _int8_matmul, _softmax_int8_approx
  └── quant_affine.py: post_process_int32_to_int8, make_requant_params,
                        quantize_affine, dequantize_affine
```
