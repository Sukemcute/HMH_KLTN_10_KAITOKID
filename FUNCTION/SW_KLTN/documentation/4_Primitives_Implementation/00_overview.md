# Phase 1 – Primitives Implementation Overview
## Mục tiêu

Mỗi **block cấp cao** (Conv, QC2f, SCDown, SPPF, QPSA, QC2fCIB, Upsample, Concat) trong mô hình qYOLOv10n được phân rã thành một chuỗi **primitive operations** (P0–P14). Mục tiêu của tài liệu này là:

1. Xác định chính xác primitive nào tạo nên block nào.
2. Cung cấp code Python **bit-accurate** cho từng primitive.
3. Cung cấp **Golden Block function** cho từng block – nhận `input → qua primitives → trả output` để so sánh với RTL phần cứng.

---

## Danh sách 15 Primitives (P0–P14)

| ID | Tên | Loại | File | Hàm Python |
|:---|:----|:-----|:-----|:-----------|
| **P0** | RS_DENSE_3x3 | Conv 3×3 dense | `primitive_conv.py` | `rs_dense_3x3()` |
| **P1** | OS_1x1 | Conv 1×1 pointwise | `primitive_conv.py` | `os_1x1()` |
| **P2** | DW_3x3 | Depthwise 3×3 | `primitive_dw.py` | `dw_3x3()` |
| **P3** | MAXPOOL_5x5 | Max Pooling 5×5 s=1 | `primitive_pool.py` | `maxpool_5x5()` |
| **P4** | MOVE | Buffer copy | `primitive_tensor.py` | `move()` |
| **P5** | CONCAT | Channel concat + requant | `primitive_tensor.py` | `concat()` |
| **P6** | UPSAMPLE_NEAREST | Upsample ×2 | `primitive_tensor.py` | `upsample_nearest()` |
| **P7** | EWISE_ADD | Element-wise add | `primitive_tensor.py` | `ewise_add()` |
| **P8** | DW_7x7_MULTIPASS | Depthwise 7×7 | `primitive_dw.py` | `dw_7x7_multipass()` |
| **P9** | GEMM_ATTN_BASIC | Self-Attention top-level | `primitive_psa.py` | `gemm_attn_basic()` |
| **P10** | INT8_MATMUL | INT8 Matrix Multiply | `primitive_psa.py` | `_int8_matmul()` |
| **P11** | SOFTMAX_APPROX | Softmax (LUT approx) | `primitive_psa.py` | `_softmax_int8_approx()` |
| **P12** | REQUANT (PPU) | Post-Processing Unit | `quant_affine.py` | `post_process_int32_to_int8()` |
| **P13** | SiLU_LUT | SiLU activation | `quant_affine.py` | `apply_silu_lut()` |
| **P14** | ReLU/ReLU6 | Clip activation | *(integrated in conv)* | `_conv_requant_act()` |

---

## Phân rã Block → Primitives

### Block: **Conv** (Layer 0, 1, 3, 17)
```
Input (int8)
  │
  ├─[P12] Padding với in_zp
  ├─[P0]  RS_DENSE_3x3  (nếu kernel 3×3)   hoặc
  │       OS_1x1 [P1]   (nếu kernel 1×1)
  ├─[P12] PPU Requant: (Acc × M_int + round) >> shift + zp_out
  └─[P14] ReLU clip [-128, 127]
Output (int8)
```
**Primitives sử dụng:** P0 hoặc P1, P12, P14

---

### Block: **QC2f** (Layer 2, 4, 6, 8, 13, 16, 19)
```
Input (int8)
  │
  ├─[P1]  cv1: OS_1x1 → expand channels (2×c_)
  ├─ Split thành y[0] và y[1] (địa chỉ, không tính toán)
  │
  ├─ Với mỗi QBottleneck:
  │   ├─[P0]  conv1: RS_DENSE_3x3
  │   └─[P0]  conv2: RS_DENSE_3x3
  │
  ├─[P5]  CONCAT: join y[0] + y[1] + all bottleneck outputs (Domain Align)
  └─[P1]  cv2: OS_1x1 → compress to output channels
Output (int8)
```
**Primitives sử dụng:** P1 (×2+), P0 (×2 per bottleneck), P5, P12, P14

---

### Block: **SCDown** (Layer 5, 7, 20)
```
Input (int8)
  │
  ├─[P1]  cv1: OS_1x1 → channel expansion (e.g. 64→128)
  └─[P2]  cv2: DW_3x3 stride=2 → spatial ×0.5 downsampling
Output (int8)
```
**Primitives sử dụng:** P1, P2, P12

---

### Block: **SPPF** (Layer 9)
```
Input (int8)
  │
  ├─[P1]  cv1: OS_1x1 → initial expansion
  ├─[P4]  MOVE: lưu output cv1 vào buffer (cần cho concat cuối)
  │
  ├─[P3]  MaxPool 5×5 s=1 → P1_pool
  ├─[P3]  MaxPool 5×5 s=1 → P2_pool
  ├─[P3]  MaxPool 5×5 s=1 → P3_pool
  │
  ├─[P5]  CONCAT: [cv1_out, P1_pool, P2_pool, P3_pool]
  │        (Scale không đổi → không cần align)
  └─[P1]  cv2: OS_1x1 → fusion + compression
Output (int8)
```
**Primitives sử dụng:** P1 (×2), P3 (×3), P4, P5, P12

---

### Block: **QPSA** (Layer 10)
```
Input (int8)
  │
  ├─[P1]  cv1: OS_1x1 → split thành branch a và branch b
  ├─[P4]  MOVE: lưu branch a
  │
  │  Branch b (Attention):
  ├─[P1]  QKV Projection: 1×1 → [Q, K, V]
  ├─[P10] INT8_MATMUL: Q×K^T → Attention Scores
  ├─[P11] SOFTMAX_APPROX: normalize scores
  ├─[P10] INT8_MATMUL: Scores×V → Context
  ├─[P2]  Positional Encoding: DW 3×3
  ├─[P1]  Final Projection: 1×1
  │
  │  FFN sub-block:
  ├─[P7]  EWISE_ADD: residual shortcut add
  ├─[P1]  FFN expand: 1×1
  ├─[P1]  FFN compress: 1×1
  ├─[P7]  EWISE_ADD: second residual add
  │
  ├─[P5]  CONCAT: branch a + branch b
  └─[P1]  cv2: OS_1x1 → final projection
Output (int8)
```
**Primitives sử dụng:** P1 (×6+), P2, P4, P5, P7 (×2), P10 (×2), P11, P12

---

### Block: **QC2fCIB** (Layer 22)
```
Input (int8)
  │
  ├─[P1]  cv1: OS_1x1 → expand
  ├─ Split: identity branch (y[0]) + processed branch (y[1])
  ├─[P4]  MOVE: lưu y[0]
  │
  │  Trong mỗi QCIB module:
  ├─[P2]  DW 3×3 (pad=1)
  ├─[P1]  PW 1×1
  ├─[P8]  DW 7×7 MULTIPASS (pad=3)
  ├─[P1]  PW 1×1
  ├─[P2]  DW 3×3 (pad=1)
  ├─[P7]  EWISE_ADD: residual shortcut
  │
  ├─[P5]  CONCAT: y[0] + QCIB_output (Domain Align)
  └─[P1]  cv2: OS_1x1 → compress
Output (int8)
```
**Primitives sử dụng:** P1 (×3+), P2 (×2), P4, P5, P7, P8, P12

---

### Block: **Upsample** (Layer 11, 14)
```
Input (int8)
  │
  └─[P6]  UPSAMPLE_NEAREST ×2 (metadata scale/zp không đổi)
Output (int8)  [H×2, W×2, C giữ nguyên]
```
**Primitives sử dụng:** P6

---

### Block: **Concat** / QConcat (Layer 12, 15, 18, 21)
```
Inputs: [Tensor_A (int8, scale_A, zp_A), Tensor_B (int8, scale_B, zp_B)]
  │
  ├─[P5]  Requant Tensor_A → target domain (scale_out, zp_out)
  ├─[P5]  Requant Tensor_B → target domain
  └─      Memory join theo channel dim
Output (int8, scale_out, zp_out)
```
**Primitives sử dụng:** P5, P12

---

## Bảng tổng hợp Layer → Primitives (qYOLOv10n, 640×640)

| Layer | Block | Shape Out | Primitives |
|:------|:------|:----------|:-----------|
| 0 | Conv | [1,16,320,320] | P0, P12, P14 |
| 1 | Conv | [1,32,160,160] | P0, P12, P14 |
| 2 | QC2f | [1,32,160,160] | P1, P0×2, P5, P12, P14 |
| 3 | Conv | [1,64,80,80] | P0, P12, P14 |
| 4 | QC2f | [1,64,80,80] | P1, P0×4, P5, P12, P14 |
| 5 | SCDown | [1,128,40,40] | P1, P2, P12 |
| 6 | QC2f | [1,128,40,40] | P1, P0×4, P5, P12, P14 |
| 7 | SCDown | [1,256,20,20] | P1, P2, P12 |
| 8 | QC2f | [1,256,20,20] | P1, P0×4, P5, P12, P14 |
| 9 | SPPF | [1,256,20,20] | P1×2, P3×3, P4, P5, P12 |
| 10 | QPSA | [1,256,20,20] | P1×6, P2, P4, P5, P7×2, P10×2, P11, P12 |
| 11 | Upsample | [1,256,40,40] | P6 |
| 12 | Concat | [1,384,40,40] | P5, P12 |
| 13 | QC2f | [1,128,40,40] | P1, P0×4, P5, P12, P14 |
| 14 | Upsample | [1,128,80,80] | P6 |
| 15 | Concat | [1,192,80,80] | P5, P12 |
| 16 | QC2f | [1,64,80,80] | P1, P0×4, P5, P12, P14 |
| 17 | Conv | [1,64,40,40] | P0, P12, P14 |
| 18 | Concat | [1,192,40,40] | P5, P12 |
| 19 | QC2f | [1,128,40,40] | P1, P0×4, P5, P12, P14 |
| 20 | SCDown | [1,128,20,20] | P1, P2, P12 |
| 21 | Concat | [1,384,20,20] | P5, P12 |
| 22 | QC2fCIB | [1,256,20,20] | P1×3, P2×2, P4, P5, P7, P8, P12 |

---

## Files trong thư mục này

| File | Nội dung |
|:-----|:---------|
| `01_conv_primitives.md` | P0 RS_DENSE_3x3, P1 OS_1x1 – code chính xác |
| `02_dw_primitives.md` | P2 DW_3x3, P8 DW_7x7_MULTIPASS – code chính xác |
| `03_tensor_primitives.md` | P3 MAXPOOL, P4 MOVE, P5 CONCAT, P6 UPSAMPLE, P7 EWISE_ADD |
| `04_ppu_primitives.md` | P12 REQUANT, P13 SiLU_LUT, P14 ReLU/ReLU6 |
| `05_psa_primitives.md` | P9 GEMM_ATTN, P10 INT8_MATMUL, P11 SOFTMAX_APPROX |
| `blocks/block_conv.md` | Golden block function cho Conv |
| `blocks/block_qc2f.md` | Golden block function cho QC2f |
| `blocks/block_scdown.md` | Golden block function cho SCDown |
| `blocks/block_sppf.md` | Golden block function cho SPPF |
| `blocks/block_qpsa.md` | Golden block function cho QPSA |
| `blocks/block_qc2f_cib.md` | Golden block function cho QC2fCIB |
| `blocks/block_upsample.md` | Golden block function cho Upsample |
| `blocks/block_concat.md` | Golden block function cho Concat |
