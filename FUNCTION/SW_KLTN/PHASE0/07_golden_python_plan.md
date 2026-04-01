# 07 – Golden Python Plan (Freeze Spec)
## qYOLOv10n INT8 – Kế Hoạch Xây Dựng Oracle Software

> **Trạng thái**: FREEZE | **Version**: v1.0 | **Ngày**: 2026-03-16

---

## 1. Mục Đích

Định nghĩa cấu trúc file, API và tiêu chí test cho Golden Python – oracle phần mềm INT8 cho layer 0–22, trả về P3/P4/P5 và metadata quantization.

---

## 2. Cấu Trúc Thư Mục

```
python_golden/
├── config.py                  ← Constants kiến trúc
├── types.py                   ← Enums, dataclasses
├── quant/
│   ├── quant_affine.py        ← Quantize/dequantize/requant
│   └── quant_domain_align.py  ← Common-domain cho CONCAT/ADD
├── primitives/
│   ├── primitive_conv.py      ← RS_DENSE_3x3, OS_1x1
│   ├── primitive_dw.py        ← DW_3x3, DW_7x7_MULTIPASS
│   ├── primitive_pool.py      ← MAXPOOL_5x5
│   ├── primitive_tensor.py    ← MOVE, CONCAT, UPSAMPLE, EWISE_ADD
│   └── primitive_psa.py       ← GEMM_ATTN_BASIC
├── layout/
│   ├── banking_model.py       ← bank_input, bank_output
│   ├── row_slot_model.py      ← Q_in, row_slot
│   ├── lane_packing.py        ← pack16/unpack16
│   ├── address_model.py       ← logical → physical address
│   └── psum_act_model.py      ← PSUM/ACT namespace semantics
├── blocks/
│   ├── block_qc2f.py          ← QC2f block-level model
│   ├── block_scdown.py        ← SCDown block
│   ├── block_sppf.py          ← SPPF block
│   ├── block_qpsa.py          ← QPSA block
│   └── block_qc2fcib.py       ← QC2fCIB block
├── model/
│   ├── layer_specs.py         ← Layer 0–22 table
│   └── model_forward_runner.py ← Entry point
└── tests/
    ├── test_primitives.py
    ├── test_quant.py
    ├── test_layout.py
    ├── test_blocks.py
    └── test_model_forward.py
```

---

## 3. File Specs

### 3.1. config.py

```python
# config.py – Hằng số kiến trúc
INPUT_BANKS   = 3       # 3 input GLB banks
OUTPUT_BANKS  = 4       # 4 output banks
LANES         = 16      # 16 PE lanes
PSUM_BITS     = 32      # PSUM accumulator width (INT32)
ACT_BITS      = 8       # Activation INT8
WEIGHT_BITS   = 8       # Weight INT8
MAX_KERNEL    = 7       # Kernel size tối đa (DW_7x7)
DW7x7_SPLIT   = (3,3,1) # DW_7x7_MULTIPASS split: rows per pass

# Primitive IDs
P0_RS_DENSE_3x3    = 0
P1_OS_1x1          = 1
P2_DW_3x3          = 2
P3_MAXPOOL_5x5     = 3
P4_MOVE            = 4
P5_CONCAT          = 5
P6_UPSAMPLE_NEAREST = 6
P7_EWISE_ADD       = 7
P8_DW_7x7_MULTIPASS = 8
P9_GEMM_ATTN_BASIC = 9

# Activation modes
ACT_NONE  = 0
ACT_SILU  = 1
ACT_RELU  = 2

INT8_MIN = -128
INT8_MAX =  127
```

---

### 3.2. types.py

```python
# types.py – Dataclasses và Enums

@dataclass
class QuantParams:
    scale: float
    zp: int
    dtype: str = "int8"  # "int8" hoặc "int32"

@dataclass
class TensorMeta:
    tensor: np.ndarray    # INT8 data
    scale: float
    zp: int

@dataclass
class LayerSpec:
    idx: int
    block_type: str       # "Conv", "QC2f", ...
    primitive_seq: list   # list of primitive IDs
    in_shape: tuple
    out_shape: tuple
    stride: int = 1
    kernel: int = 3
    sources: list = field(default_factory=lambda: [-1])
    hold_output: bool = False
    hold_until: int = -1
    output_name: str = None  # "P3", "P4", "P5" hoặc None

@dataclass
class TileFlags:
    first_tile: bool = False
    edge_tile_h: bool = False
    edge_tile_w: bool = False
    hold_skip: bool = False
    need_swizzle: bool = False
    psum_carry_in: bool = False

@dataclass
class LastFlags:
    last_cin: bool = False
    last_kernel: bool = False
    last_reduce: bool = False

    @property
    def last_pass(self):
        return self.last_cin and self.last_kernel and self.last_reduce

class Primitive(Enum):
    RS_DENSE_3x3     = 0
    OS_1x1           = 1
    DW_3x3           = 2
    MAXPOOL_5x5      = 3
    MOVE             = 4
    CONCAT           = 5
    UPSAMPLE_NEAREST = 6
    EWISE_ADD        = 7
    DW_7x7_MULTIPASS = 8
    GEMM_ATTN_BASIC  = 9
```

---

### 3.3. quant/quant_affine.py

**API bắt buộc**:

```python
def quantize_affine(x_float: np.ndarray, scale: float, zp: int,
                    dtype=np.int8) -> np.ndarray:
    """Float → INT8. Rounding: round-half-up."""
    x_int = np.floor(x_float / scale + 0.5).astype(np.int32) + zp
    return np.clip(x_int, -128, 127).astype(dtype)

def dequantize_affine(x_int: np.ndarray, scale: float, zp: int) -> np.ndarray:
    """INT8 → float32."""
    return (x_int.astype(np.float32) - zp) * scale

def make_requant_params(scale_in: float, scale_w: np.ndarray,
                        scale_out: float) -> tuple:
    """
    Tính M = scale_in * scale_w[cout] / scale_out
    Decompose thành (M_int_array, shift) dùng cho fixed-point requant.
    Returns: (M_int: np.ndarray[Cout], shift: int)
    """
    M = scale_in * scale_w / scale_out
    shift = max(0, int(np.floor(np.log2(1.0 / M.max()))) + 15)
    M_int = np.round(M * (2 ** shift)).astype(np.int32)
    return M_int, shift

def post_process_int32_to_int8(acc: np.ndarray,  # [Cout, H, W] INT32
                                 M_int: np.ndarray, shift: int,
                                 zp_out: int) -> np.ndarray:
    """
    Requant INT32 → INT8.
    y = clamp(round(M_int * acc >> shift) + zp_out, -128, 127)
    """
    y_raw = np.right_shift(acc * M_int[:, None, None], shift) + zp_out
    return np.clip(y_raw, -128, 127).astype(np.int8)

def silu_lut(y_pre: np.ndarray, scale_pre: float, zp_pre: int,
             scale_post: float, zp_post: int) -> np.ndarray:
    """
    Áp dụng SiLU thông qua precomputed LUT.
    LUT[i] đã được tính offline: SiLU_LUT[i]=quantize(SiLU(dequant(i)))
    """
    # Precompute (offline, stored as constant)
    lut = _build_silu_lut(scale_pre, zp_pre, scale_post, zp_post)
    idx = (y_pre.astype(np.int32) + 128)  # shift to [0,255]
    return lut[idx].astype(np.int8)
```

---

### 3.4. quant/quant_domain_align.py

**API bắt buộc**:

```python
def compute_common_domain(scales: list, zps: list,
                           target_scale: float, target_zp: int):
    """
    Trả về common domain params từ calibration.
    target_scale, target_zp: calibrated output của QConcat layer (từ PTQ).
    """
    return target_scale, target_zp

def requant_to_domain(x_int8: np.ndarray,
                       scale_src: float, zp_src: int,
                       scale_dst: float, zp_dst: int) -> np.ndarray:
    """
    Requant tensor từ (scale_src, zp_src) về (scale_dst, zp_dst).
    Chỉ dùng integer arithmetic (float path chỉ cho reference).
    """
    x_float = dequantize_affine(x_int8, scale_src, zp_src)
    return quantize_affine(x_float, scale_dst, zp_dst)

def align_and_concat(tensors: list, scales: list, zps: list,
                      target_scale: float, target_zp: int,
                      axis: int = 1) -> TensorMeta:
    """
    Align tất cả tensors về common domain rồi concat.
    axis=1: channel dimension.
    """
    aligned = []
    for t, s, z in zip(tensors, scales, zps):
        if abs(s - target_scale) < 1e-7 and z == target_zp:
            aligned.append(t)  # no requant needed
        else:
            aligned.append(requant_to_domain(t, s, z, target_scale, target_zp))
    return TensorMeta(
        tensor=np.concatenate(aligned, axis=axis),
        scale=target_scale, zp=target_zp
    )

def align_and_add(A: np.ndarray, scale_A: float, zp_A: int,
                   B: np.ndarray, scale_B: float, zp_B: int,
                   target_scale: float, target_zp: int,
                   out_scale: float, out_zp: int) -> TensorMeta:
    """
    Align A,B về target_scale, add, requant về (out_scale, out_zp).
    Intermediate: INT16 để tránh overflow.
    """
    A_al = requant_to_domain(A, scale_A, zp_A, target_scale, target_zp)
    B_al = requant_to_domain(B, scale_B, zp_B, target_scale, target_zp)
    # INT16 intermediate
    sum_i16 = A_al.astype(np.int16) + B_al.astype(np.int16)
    # Requant về output
    result = quantize_affine(
        dequantize_affine(sum_i16, target_scale, target_zp),
        out_scale, out_zp
    )
    return TensorMeta(tensor=result, scale=out_scale, zp=out_zp)
```

---

### 3.5. model/layer_specs.py

```python
# layer_specs.py – Nguồn chân lý duy nhất cho layer sequence

LAYER_SPECS = [
  LayerSpec(idx=0,  block_type="Conv",     in_shape=(1,3,640,640),
            out_shape=(1,16,320,320),   stride=2, kernel=3,
            sources=[-1], hold_output=False),
  LayerSpec(idx=1,  block_type="Conv",     in_shape=(1,16,320,320),
            out_shape=(1,32,160,160),   stride=2, kernel=3,
            sources=[-1], hold_output=False),
  LayerSpec(idx=2,  block_type="QC2f",     in_shape=(1,32,160,160),
            out_shape=(1,32,160,160),   sources=[-1], hold_output=False),
  LayerSpec(idx=3,  block_type="Conv",     in_shape=(1,32,160,160),
            out_shape=(1,64,80,80),     stride=2, kernel=3,
            sources=[-1], hold_output=False),
  LayerSpec(idx=4,  block_type="QC2f",     in_shape=(1,64,80,80),
            out_shape=(1,64,80,80),     sources=[-1],
            hold_output=True, hold_until=15),   # SKIP-A
  LayerSpec(idx=5,  block_type="SCDown",   in_shape=(1,64,80,80),
            out_shape=(1,128,40,40),    sources=[-1], hold_output=False),
  LayerSpec(idx=6,  block_type="QC2f",     in_shape=(1,128,40,40),
            out_shape=(1,128,40,40),    sources=[-1],
            hold_output=True, hold_until=12),   # SKIP-B
  LayerSpec(idx=7,  block_type="SCDown",   in_shape=(1,128,40,40),
            out_shape=(1,256,20,20),    sources=[-1], hold_output=False),
  LayerSpec(idx=8,  block_type="QC2f",     in_shape=(1,256,20,20),
            out_shape=(1,256,20,20),    sources=[-1],
            hold_output=True, hold_until=21),   # SKIP-C
  LayerSpec(idx=9,  block_type="SPPF",     in_shape=(1,256,20,20),
            out_shape=(1,256,20,20),    sources=[-1], hold_output=False),
  LayerSpec(idx=10, block_type="QPSA",     in_shape=(1,256,20,20),
            out_shape=(1,256,20,20),    sources=[-1], hold_output=False),
  LayerSpec(idx=11, block_type="Upsample", in_shape=(1,256,20,20),
            out_shape=(1,256,40,40),    sources=[-1], hold_output=False),
  LayerSpec(idx=12, block_type="QConcat",  in_shape=(1,384,40,40),
            out_shape=(1,384,40,40),    sources=[11, 6], hold_output=False),
  LayerSpec(idx=13, block_type="QC2f",     in_shape=(1,384,40,40),
            out_shape=(1,128,40,40),    sources=[-1],
            hold_output=True, hold_until=18),   # SKIP-D
  LayerSpec(idx=14, block_type="Upsample", in_shape=(1,128,40,40),
            out_shape=(1,128,80,80),    sources=[-1], hold_output=False),
  LayerSpec(idx=15, block_type="QConcat",  in_shape=(1,192,80,80),
            out_shape=(1,192,80,80),    sources=[14, 4], hold_output=False),
  LayerSpec(idx=16, block_type="QC2f",     in_shape=(1,192,80,80),
            out_shape=(1,64,80,80),     sources=[-1],
            hold_output=False, output_name="P3"),
  LayerSpec(idx=17, block_type="Conv",     in_shape=(1,64,80,80),
            out_shape=(1,64,40,40),     stride=2, kernel=3,
            sources=[-1], hold_output=False),
  LayerSpec(idx=18, block_type="QConcat",  in_shape=(1,192,40,40),
            out_shape=(1,192,40,40),    sources=[17, 13], hold_output=False),
  LayerSpec(idx=19, block_type="QC2f",     in_shape=(1,192,40,40),
            out_shape=(1,128,40,40),    sources=[-1],
            hold_output=False, output_name="P4"),
  LayerSpec(idx=20, block_type="SCDown",   in_shape=(1,128,40,40),
            out_shape=(1,128,20,20),    sources=[-1], hold_output=False),
  LayerSpec(idx=21, block_type="QConcat",  in_shape=(1,384,20,20),
            out_shape=(1,384,20,20),    sources=[20, 8], hold_output=False),
  LayerSpec(idx=22, block_type="QC2fCIB",  in_shape=(1,384,20,20),
            out_shape=(1,256,20,20),    sources=[-1],
            hold_output=False, output_name="P5"),
]
```

---

## 4. Test Criteria Đầy Đủ

### test_quant.py

```
Test A1: quantize_affine round-trip
  x_float = random uniform [0, 1]
  x_q = quantize(x_float, scale=1/255, zp=0)
  x_back = dequantize(x_q, scale=1/255, zp=0)
  assert np.abs(x_float - x_back).max() < 1/255  (≤ 1 LSB)

Test A2: clamp behavior
  x = np.array([200.0, -200.0])
  q = quantize(x, scale=1.0, zp=0)
  assert q[0] == 127 and q[1] == -128

Test A3: requant params
  M_int, shift = make_requant_params(scale_in=0.004, scale_w=0.001*ones(16), scale_out=0.025)
  M_actual = M_int / 2**shift
  assert np.allclose(M_actual, 0.004*0.001/0.025, rtol=1e-4)

Test B1: concat same domain
  A = random_int8([1,64,20,20]), B = random_int8([1,32,20,20])
  scale=0.05, zp=0 cho cả hai
  Y = align_and_concat([A,B], [0.05,0.05], [0,0], 0.05, 0, axis=1)
  assert Y.tensor.shape == (1,96,20,20)
  assert np.array_equal(Y.tensor[:,:64], A)  # no requant → identical

Test B2: concat domain mismatch
  A_int8 with scale_A=0.1, zp_A=0
  B_int8 with scale_B=0.05, zp_B=2
  target_scale=0.07, target_zp=0
  Y = align_and_concat([A,B], [0.1,0.05], [0,2], 0.07, 0, axis=1)
  # Verify by dequant and check float values close enough

Test B3: add saturation
  A_int8 = np.full((1,4,4,4), 100, dtype=np.int8)
  B_int8 = np.full((1,4,4,4), 100, dtype=np.int8)
  Y = align_and_add(A, 0.1, 0, B, 0.1, 0, 0.1, 0, 0.1, 0)
  # 100+100=200 > 127 → should clamp to 127
  assert np.all(Y.tensor <= 127)
```

### test_primitives.py

```
Test Conv s1/s2, Test DW s1/s2, Test DW7x7 multipass==monolithic
Test MAXPOOL shape+value, Test UPSAMPLE shape+content+metadata
Test CONCAT same/diff domain, Test EWISE_ADD basic/saturation
Test GEMM_ATTN shape+deterministic

Key invariant to test:
  dw_7x7(X, W) == dw_7x7_multipass(X, W)  ← BIT EXACT
```

### test_layout.py

```
Test banking: h%3 cycle
Test row_slot: Conv3x3 s1 Q_in=2; s2 Q_in=3; DW7x7 Q_in=4
Test pack16/unpack16 round-trip: unpack(pack(x))==x for W=40,80,160,320
Test address no-overlap: all (h,x,cin) → unique (bank,offset)
Test psum_act: last_pass=False → None; last_pass=True → INT8
```

### test_blocks.py

```
Test QC2f: shape, dtype, intermediate dumps
Test SCDown: shape, two-branch concat
Test SPPF: pool chain correct, 4-way concat
Test QPSA: shape preserved, attention output non-zero
Test QC2fCIB: DW7x7_multipass == monolithic (most important)
```

### test_model_forward.py

```
Test 1: P3 shape = [1,64,80,80],  dtype=int8
Test 2: P4 shape = [1,128,40,40], dtype=int8
Test 3: P5 shape = [1,256,20,20], dtype=int8
Test 4: scale/zp valid for P3,P4,P5
Test 5: 23 stage_outputs exist
Test 6: Barrier logic (L12 has sources [11,6] both computed)
Test 7: Accuracy vs PyTorch float (RMSE < 2% threshold)
Test 8: Determinism (same input → same P3/P4/P5)
```

---

## 5. Dump Format (Trace Output)

```python
# Stage output format
stage_outputs[layer_idx] = TensorMeta(
    tensor = np.ndarray,  # INT8 output
    scale  = float,
    zp     = int
)

# P3/P4/P5 format
result = {
    "P3": TensorMeta(...),
    "P4": TensorMeta(...),
    "P5": TensorMeta(...),
    "stage_outputs": dict[int, TensorMeta],
    "layout_traces": dict,  # optional: banking,slot,address per layer
    "psum_traces": dict,    # optional: PSUM per DW7x7 pass
}

# Save oracle
np.save("oracle_P3.npy", result["P3"].tensor)
np.save("oracle_P4.npy", result["P4"].tensor)
np.save("oracle_P5.npy", result["P5"].tensor)
```

---

## 6. Sign-off Checklist

```
STRUCTURE:
☐ python_golden/ folder tạo đúng structure
☐ config.py: tất cả constants từ spec
☐ types.py: LayerSpec.hold_until và sources được implement
☐ layer_specs.py: 23 entries, hold_output đúng 4 layers (L4,L6,L8,L13)

QUANT:
☐ test_quant.py: 100% PASS
☐ quant_domain_align.py: gọi từ CONCAT và EWISE_ADD (không duplicate code)

PRIMITIVES:
☐ test_primitives.py: 100% PASS
☐ DW_7x7_MULTIPASS == monolithic: BIT EXACT

LAYOUT:
☐ test_layout.py: 100% PASS
☐ pack16/unpack16 round-trip: verified tất cả W sizes

BLOCKS:
☐ test_blocks.py: 100% PASS
☐ QC2fCIB dump psum_traces sau từng pass

MODEL FORWARD:
☐ test_model_forward.py: 100% PASS
☐ P3/P4/P5 oracle files saved: oracle_P3.npy, oracle_P4.npy, oracle_P5.npy
☐ stage_outputs 0–22 dumped và validated
```

*Golden Python phải hoàn thành và pass 100% trước khi bắt đầu viết RTL leaf module đầu tiên.*
