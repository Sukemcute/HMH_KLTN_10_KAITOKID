# Flow Hiện Thực Tăng Tốc qYOLOv10n INT8 trên Phần Cứng
## (Không Lỗi – Đúng Từng Bước)

> **Triết lý cốt lõi**: Không bao giờ đi xuống tầng thấp hơn khi tầng trên chưa verified.  
> **Thứ tự không thể đảo ngược**: Quant Math → Primitive → Block → Layer Runner → RTL

---

## TỔNG QUAN 5 GIAI ĐOẠN

```
┌────────────────────────────────────────────────────────────────────────────┐
│  GIAI ĐOẠN 0   │  GIAI ĐOẠN 1    │  GIAI ĐOẠN 2  │  GIAI ĐOẠN 3  │ GĐ 4 │
│  Spec Freeze   │  Golden Python  │  Block Oracle  │  Model Runner │  RTL │
│  (Thiết kế)    │  (Quant+Prim)   │  (Block+Mem)   │  (L0–L22)     │      │
│                │                 │                │               │      │
│  ← Đầu vào ──►│ ← Xây dựng ──► │ ← Tích hợp ──►│ ← End-to-end►│      │
│  cho mọi giai  │   từng viên     │   thành khối   │   P3/P4/P5    │      │
│  đoạn sau      │   gạch          │   lớn          │               │      │
└────────────────────────────────────────────────────────────────────────────┘

      ↑ Nếu test FAIL tại giai đoạn nào, SỬA ngay tại giai đoạn đó,
        KHÔNG đi xuống giai đoạn sau khi chưa pass.
```

---

## GIAI ĐOẠN 0: SPEC FREEZE (Thiết Kế & Khóa Tài Liệu)

> Mục tiêu: Chốt mọi quyết định thiết kế TRƯỚC khi viết code.

### 0.1. Khóa Ranh Giới Hệ Thống

```
CPU side:
  [Image] → LetterBox/Resize → Normalize(÷255) → Quantize(scale_in,zp_in) → X_int8

Accelerator side:
  [X_int8, scale_in, zp_in] → Layer 0–22 → [P3_int8, P4_int8, P5_int8]
  + quant metadata: (scale_3,zp_3), (scale_4,zp_4), (scale_5,zp_5)

CPU side (post):
  [P3/P4/P5_int8] → Dequant → Qv10Detect → NMS → Bbox results
```

**Checklist Phase 0 phải hoàn thành:**

- [ ] **01_primitive_matrix.md** – Primitive ID, input/output rank, stride, padding, quant rule, PPU flag
- [ ] **02_layer_mapping.md** – L0–L22 → primitive sequence + 4 skip dependency (L4→L15, L6→L12, L8→L21, L13→L18)  
- [ ] **03_quant_policy.md** – activation INT8 per-tensor, weight INT8 per-channel, CONCAT/ADD dùng common-domain
- [ ] **04_layout_addressing.md** – bank_input=h%3, bank_output=out_row%4, Q_in formula, lane=x%16
- [ ] **05_descriptor_spec.md** – NET/LAYER/TILE/ROUTER/POST descriptor formats + flag semantics
- [ ] **06_execution_semantics.md** – last_pass=last_cin∧last_kernel∧last_reduce, PSUM/ACT, HOLD_SKIP, Barrier
- [ ] **07_golden_python_plan.md** – Cấu trúc file + test criteria
- [ ] **08_rtl_mapping_plan.md** – Primitive → RTL module mapping

> ⛔ **STOP**: Không bắt đầu code khi 8 file chưa đồng thuận giữa các bên (SW/HW/Verification).

---

## GIAI ĐOẠN 1: GOLDEN PYTHON – QUANT & PRIMITIVES

> Mục tiêu: Xây dựng oracle số học chính xác cho từng phép tính.

### 1.1. Bước 1A – Core Quantization (PHẢI VIẾT ĐẦU TIÊN)

```
File: quant_affine.py
─────────────────────────────────────────────────────────────
Implement:
  quantize_affine(x_float, scale, zp, dtype=int8)
    → x_int = clamp(round(x/scale) + zp, min_val, max_val)

  dequantize_affine(x_int, scale, zp)
    → x_float = (x_int - zp) * scale

  make_requant_params(scale_in, scale_w_per_ch, scale_out)
    → M[cout] = scale_in * scale_w[cout] / scale_out
    → (M_int, shift) = fixed_point_decompose(M)
    → Constraint: M_int fits INT32, shift ∈ [0,31]

  post_process_int32_to_int8(acc_int32, M_int, shift, zp_out)
    → y_raw = (acc_int32 * M_int) >> shift
    → y_int8 = clamp(y_raw + zp_out, -128, 127)

Test bắt buộc (test_quant.py – Step A):
  ✓ round-trip: quantize(dequantize(x)) ≈ x (error < 1 LSB)
  ✓ clamp đúng biên [-128, 127] 
  ✓ M decompose: (M_int >> shift) ≈ M với sai số < 1e-5
  ✓ overflow safety: acc_int32 = INT32_MAX không gây crash
```

```
File: quant_domain_align.py    ← RỦI RO SỐ 1 của toàn dự án
─────────────────────────────────────────────────────────────
Implement:
  compute_common_scale(scale_list)
    → Chiến lược 1: common = max(scale_list)  ← giữ precision
    → Chiến lược 2: common = scale được định offline (preferred for HW)
    → Trả về common_scale, common_zp

  requant_to_common(x_int8, scale_src, zp_src, scale_dst, zp_dst)
    → x_float = dequantize(x_int8, scale_src, zp_src)
    → x_out   = quantize(x_float, scale_dst, zp_dst)
    → Chỉ dùng integer arithmetic (không float path trong production)

  align_and_concat(tensors_int8, scales, zps, axis=channel)
    → Bước 1: xác định common domain
    → Bước 2: requant tất cả tensor về common domain
    → Bước 3: numpy.concatenate theo axis

  align_and_add(A_int8, scale_A, zp_A, B_int8, scale_B, zp_B)
    → common domain alignment
    → add với saturation clamp
    → requant về output scale

Test bắt buộc (test_quant.py – Step B):
  ✓ concat domain_mismatch: scale_A=0.1, scale_B=0.05 → output đúng
  ✓ add saturation: giá trị cộng vượt INT8 range → clamp đúng
  ✓ add domain_equal: không requant thêm (identity path)
  ✓ concat 4 nhánh (SPPF case): X1,P1,P2,P3 → ghép 4 nhánh đúng
```

> 🔑 **Rule quan trọng**: Sau bước này, `quant_domain_align.py` IDs là nguồn chân lý duy nhất.  
> Mọi primitive khác PHẢI gọi qua đây, **không tự viết lại logic align**.

---

### 1.2. Bước 1B – Primitive Convolution

```
File: primitive_conv.py
─────────────────────────────────────────────────────────────
Implement:
  rs_dense_3x3(X_int8, W_int8, B_int32, 
               scale_x, zp_x, scale_w, zp_w, scale_y, zp_y,
               stride=1, padding='same', activation='silu')

  Thuật toán chính xác:
    1. Zero-fold correction (precompute offline):
       partial_sum_w[cout] = Σ_{cin,kh,kw} W_int8[cout,cin,kh,kw]
       zp_correction[cout] = zp_x * partial_sum_w[cout]
       
    2. Raw MAC:
       acc_raw[cout,h,w] = Σ_{cin,kh,kw} x_int8[cin,h_in,w_in] * w_int8[cout,cin,kh,kw]
       
    3. Zero-point subtract + bias:
       acc[cout,h,w] = acc_raw - zp_correction[cout] + B_int32[cout]
       
    4. Requant:
       M[cout] = scale_x * scale_w[cout] / scale_y
       y_raw = round(acc * M) + zp_y
       
    5. Activation SiLU LUT (256-entry, precomputed):
       y_silu = silu_lut[clamp(y_raw, 0, 255)]  ← index by unsigned
       
    6. Clamp: y_int8 = clamp(y_silu, -128, 127)

  os_1x1(X_int8, W_int8, B_int32, ..., activation=None)
    → Gọi rs_dense_3x3 với kernel=1, stride=1, padding=0

Test bắt buộc (test_primitives.py – Conv):
  ✓ conv3x3 stride=1: output shape đúng, giá trị so với scipy.signal.correlate
  ✓ conv3x3 stride=2: shape = ceil(H/2) × ceil(W/2)
  ✓ padding='same': output H=input H khi stride=1
  ✓ multi-channel: Cin=64, Cout=128, batch random tensor
  ✓ zp_correction: test với zp_x ≠ 0 cho ra kết quả đúng
  ✓ os_1x1: Cin≠Cout projection đúng
  ✓ random regression: so với torch.nn.functional.conv2d(float) sau quantize
```

---

### 1.3. Bước 1C – Primitive Depthwise

```
File: primitive_dw.py
─────────────────────────────────────────────────────────────
Implement:
  dw_3x3(X_int8, W_int8_per_ch, B_int32_per_ch,
          scale_x, zp_x, scale_w_per_ch, scale_y, zp_y,
          stride=1)

  QUAN TRỌNG: Per-channel requant
    for c in range(C):
      M[c] = scale_x * scale_w_per_ch[c] / scale_y
      acc[c] = Σ_{kh,kw} x_int8[c, h_in, w_in] * W_int8_per_ch[c, kh, kw]
               - zp_x * Σ_{kh,kw} W_int8_per_ch[c, kh, kw]
               + B_int32[c]
      y[c] = clamp(round(acc[c] * M[c]) + zp_y, -128, 127)
    
    last_pass luôn = True (không accumulate cross-channel)

  dw_7x7_multipass(X_int8, W_int8_per_ch, B_int32_per_ch, ...,
                   split='3-3-1')
  
  Thuật toán multi-pass:
    PSUM = zeros_int32(shape=output_shape)
    
    Pass 1: kernel_rows = W[c, 0:3, :]   → PSUM += Σ_{kh=0,1,2} x*w
    Pass 2: kernel_rows = W[c, 3:6, :]   → PSUM += Σ_{kh=3,4,5} x*w
    Pass 3: kernel_rows = W[c, 6:7, :]   → PSUM += Σ_{kh=6}     x*w  (last_pass)
            → PSUM += B_int32[c]
            → y = clamp(round(PSUM * M[c]) + zp_y, -128, 127)
    
    Return y_int8, psum_per_pass=[PSUM_after_p1, PSUM_after_p2]  ← trace

Test bắt buộc (test_primitives.py – DW):
  ✓ dw3x3 stride=1: kết quả == groups=C conv với torch
  ✓ dw3x3 stride=2: shape đúng
  ✓ per-channel bias: từng channel có bias riêng
  ✓ dw7x7 multipass: OUTPUT phải == monolithic dw7x7 result  ← BẮT BUỘC
  ✓ dw7x7 psum trace: PSUM_after_p2 nhỏ hơn PSUM_after_p3
```

---

### 1.4. Bước 1D – Primitive Pool & Tensor

```
File: primitive_pool.py
─────────────────────────────────────────────────────────────
  maxpool_5x5(X_int8, padding=2)
    → kernel=5, stride=1, padding=2
    → max(x_int8) comparison only: NO requant, scale/zp unchanged
    → return Y_int8, scale_in, zp_in  (pass-through metadata)

File: primitive_tensor.py
─────────────────────────────────────────────────────────────
  upsample_nearest(X_int8, scale_factor=2)
    → Y[c, 2h,   2w  ] = X[c, h, w]
    → Y[c, 2h,   2w+1] = X[c, h, w]
    → Y[c, 2h+1, 2w  ] = X[c, h, w]
    → Y[c, 2h+1, 2w+1] = X[c, h, w]
    → scale_out = scale_in, zp_out = zp_in  ← KHÔNG đổi
    → return Y_int8, scale_in, zp_in

  concat(tensors, scales, zps, axis=1)
    → Gọi align_and_concat() từ quant_domain_align.py
    → KHÔNG tự implement lại align logic

  ewise_add(A_int8, scale_A, zp_A, B_int8, scale_B, zp_B,
            scale_out, zp_out)
    → Gọi align_and_add() từ quant_domain_align.py

  move(X_int8, scale, zp)
    → return X_int8.copy(), scale, zp  (copy tensor, giữ metadata)

Test bắt buộc (test_primitives.py – Tensor):
  ✓ upsample: shape 20×20 → 40×40, content replicated đúng 4 lần
  ✓ upsample: scale và zp KHÔNG đổi
  ✓ concat same domain: không có requant, output = numpy.concatenate
  ✓ concat diff domain (scale_A=0.1, scale_B=0.05): align trước concat
  ✓ maxpool: max value preserved, scale/zp unchanged
  ✓ maxpool repeated 3x (SPPF): shape không đổi [256,20,20]
```

---

### 1.5. Bước 1E – Primitive PSA

```
File: primitive_psa.py
─────────────────────────────────────────────────────────────
  gemm_attn_basic(X_int8, scale_x, zp_x,
                  W_Q, W_K, W_V, W_out,     # weight tensors INT8
                  B_Q, B_K, B_V, B_out,     # bias INT32
                  scale_params, ...):
  
  Thuật toán (ĐÚNG CHỨC NĂNG là ưu tiên số 1):
    Step 1: Q = os_1x1(X, W_Q, ...)    → INT8 [HW, Hq]
            K = os_1x1(X, W_K, ...)    → INT8 [HW, Hk]
            V = os_1x1(X, W_V, ...)    → INT8 [HW, Hv]
    
    Step 2: Attn_raw = Q × K^T          → INT32 [HW, HW]
            Scale: M_attn = scale_Q * scale_K / scale_Attn
            Attn_int8 = requant(Attn_raw, M_attn) → INT8 [HW, HW]
    
    Step 3: Attn_scale = Attn_int8 / sqrt(Hq)  (fixed-point approx)
            softmax_int8 = softmax_approx_lut(Attn_scale)
    
    Step 4: Out = softmax_int8 × V      → INT32 [HW, Hv]
            Out_int8 = requant(Out, M_out) → INT8 [HW, Hv]
    
    Step 5: reshape → [C, H, W]
            output_proj = os_1x1(Out_int8, W_out, ...)

Test bắt buộc (test_primitives.py – PSA):
  ✓ shape: [1,256,20,20] → [1,256,20,20]
  ✓ deterministic: cùng input → cùng output (không có random)
  ✓ small tensor: [1,8,4,4] test case với known values
```

---

## GIAI ĐOẠN 2: BLOCK ORACLE + LAYOUT MODEL

> Mục tiêu: Tổng hợp primitives thành block-level model và xác minh layout.

### 2.1. Bước 2A – Layout Models (Phải làm SONG SONG với Block Oracle)

```
File: banking_model.py
─────────────────────────────────────────────────────────────
  bank_input(h):  return h % 3
  bank_output(out_row): return out_row % 4
  
  get_resident_rows(H, stride, kernel=3):
    "Trả về danh sách row nào cùng bank ở từng step"
    for h_out in range(ceil(H/stride)):
      yield h_out, [h_out*stride + k for k in range(kernel)]

File: row_slot_model.py
─────────────────────────────────────────────────────────────
  compute_Q_in(K_eff, stride):
    return ceil((K_eff + 3*stride) / 3)
  
  row_slot(h, Q_in):
    return (h // 3) % Q_in

File: lane_packing.py
─────────────────────────────────────────────────────────────
  pack16(data_hwc):   → packed[W//16, H, C, 16]
  unpack16(packed):   → data[H, W, C]
  
  Invariant: unpack16(pack16(x)) == x  ← test ngay

File: address_model.py
─────────────────────────────────────────────────────────────
  compute_input_addr(h, x, cin, H, W, Cin):
    bank     = bank_input(h)
    Q_in     = compute_Q_in(K_eff=3, stride=stride)
    slot     = row_slot(h, Q_in)
    Wblk     = x // 16
    lane     = x % 16
    Wblk_total = ceil(W / 16)
    offset   = slot*(Wblk_total*Cin*16) + Wblk*(Cin*16) + cin*16 + lane
    return (bank, offset)

File: psum_act_model.py
─────────────────────────────────────────────────────────────
  class TileState:
    psum_buf: INT32 accumulator
    
    def accumulate(self, mac_result, last_cin, last_kernel, last_reduce):
      self.psum_buf += mac_result
      last_pass = last_cin and last_kernel and last_reduce
      if last_pass:
        return self.ppu_process(self.psum_buf)  # → INT8
      else:
        return None  # vẫn đang accumulate

Test bắt buộc (test_layout.py):
  ✓ bank_input: h=0→0, h=1→1, h=2→2, h=3→0, h=4→1 (cyclic)
  ✓ bank_output: out_row=0→0, 1→1, 2→2, 3→3, 4→0 (cyclic)
  ✓ Q_in conv3x3 stride=1: Q_in = 2
  ✓ Q_in conv3x3 stride=2: Q_in = 3
  ✓ Q_in dw7x7 stride=1: Q_in = ceil((7+3)/3) = 4
  ✓ pack16/unpack16 round-trip: unpack(pack(x)) == x
  ✓ address no-overlap: tất cả input pixel map đến unique (bank, offset)
  ✓ psum_act: last_pass=False → None; last_pass=True → INT8 output
```

---

### 2.2. Bước 2B – Block Models

```
Pattern chung cho mọi block:
  - Nhận tensors INT8 + quant metadata làm input
  - Gọi primitives theo đúng sequence đã freeze trong layer_specs
  - Dump intermediate tensors (cho debug RTL về sau)
  - Trả về tensor INT8 + quant metadata

File: block_qc2f.py
─────────────────────────────────────────────────────────────
  def block_qc2f(X_int8, scales, zps, weights, n_bottleneck=1, dump=False):
  """
  Primitive sequence: OS_1x1 → (RS_DENSE_3x3 × n) → CONCAT → OS_1x1
  """
    # Step 1: cv1 (OS_1x1 expansion/split)
    X1 = os_1x1(X_int8, W_cv1, B_cv1, ...)
    
    intermediates = [X1]  # nhánh split đầu tiên
    
    # Step 2: n bottleneck
    y = X1
    for i in range(n_bottleneck):
      y_tmp = rs_dense_3x3(y, W_bn1[i], B_bn1[i], ...)  # cv1 bottleneck
      y     = rs_dense_3x3(y_tmp, W_bn2[i], B_bn2[i], ...) # cv2 bottleneck
      intermediates.append(y)
    
    # Step 3: CONCAT tất cả nhánh
    Y_cat = concat(intermediates, scales_cat, zps_cat, axis=channel)
    
    # Step 4: cv2 (OS_1x1 merge)
    Y_out = os_1x1(Y_cat, W_cv2, B_cv2, ...)
    
    if dump:
      return Y_out, {"X1": X1, "intermediates": intermediates, "Y_cat": Y_cat}
    return Y_out

  Test: shape, dtype, intermediate shapes, cross-check với float forward

File: block_scdown.py
─────────────────────────────────────────────────────────────
  def block_scdown(X_int8, ..., stride=2):
  """
  Primitive sequence: OS_1x1 (per branch) → DW_3x3(s2) → CONCAT
  """
    # Branch A (Cout/2 channels)
    A1 = os_1x1(X_int8, W_A, B_A, out_channels=Cout//2, ...)
    A2 = dw_3x3(A1, W_DW_A, B_DW_A, stride=2, ...)
    
    # Branch B (Cout/2 channels)
    B1 = os_1x1(X_int8, W_B, B_B, out_channels=Cout//2, ...)
    B2 = dw_3x3(B1, W_DW_B, B_DW_B, stride=2, ...)
    
    # CONCAT theo channel
    Y = concat([A2, B2], [scale_A2, scale_B2], [zp_A2, zp_B2], axis=channel)
    return Y

File: block_sppf.py
─────────────────────────────────────────────────────────────
  def block_sppf(X_int8, ..., k=5):
  """
  OS_1x1 → MAXPOOL×3 → CONCAT(4 nhánh) → OS_1x1
  """
    X1 = os_1x1(X_int8, W_cv1, ...)
    P1 = maxpool_5x5(X1)
    P2 = maxpool_5x5(P1)
    P3 = maxpool_5x5(P2)
    
    # Tất cả 4 nhánh cùng scale/zp (từ cùng qconfig) → concat đơn giản
    Y_cat = concat([X1, P1, P2, P3], same_scale_list, same_zp_list, axis=channel)
    Y_out = os_1x1(Y_cat, W_cv2, ...)
    return Y_out

  QUAN TRỌNG: verify P1, P2, P3 có scale/zp giữ nguyên từ X1

File: block_qpsa.py
─────────────────────────────────────────────────────────────
  def block_qpsa(X_int8, ...):
  """
  OS_1x1(split) → GEMM_ATTN_BASIC → CONCAT → OS_1x1(merge)
  """
    X_attn, X_pass = split_channels(X_int8, C//2, C//2)
    Y_attn = gemm_attn_basic(X_attn, ...)
    Y_merged = concat([Y_attn, X_pass], ...)
    Y_out = os_1x1(Y_merged, W_proj, ...)
    return Y_out

File: block_qc2fcib.py
─────────────────────────────────────────────────────────────
  def block_qc2fcib(X_int8, ..., use_large_kernel=True):
  """
  OS_1x1 → CIB(DW_7x7_MULTIPASS + OS_1x1) → (RS_3x3) → CONCAT → OS_1x1
  """
    X1 = os_1x1(X_int8, W_cv1, ...)
    
    # CIB path
    Y_dw, psum_traces = dw_7x7_multipass(X1, W_dw7x7, B_dw7x7, ...)
    Y_cib = os_1x1(Y_dw, W_cib, ...)
    
    # CONCAT (CIB output + skip từ X1)
    Y_cat = concat([Y_cib, X1], ...)
    
    Y_out = os_1x1(Y_cat, W_cv2, ...)
    return Y_out, psum_traces  # trace để verify RTL

Test bắt buộc (test_blocks.py):
  ✓ QC2f: shape [1,32,160,160]→[1,32,160,160], int8 dtype
  ✓ SCDown: shape [1,64,80,80]→[1,128,40,40]
  ✓ SCDown: hai nhánh có concat đúng channel count
  ✓ SPPF: 3× pool shapes cùng [1,128,20,20], concat → [1,512,20,20]
  ✓ QPSA: shape preserved [1,256,20,20]→[1,256,20,20]
  ✓ QC2fCIB: DW7x7_multipass == monolithic DW7x7 (so sánh psum trace)
  ✓ Mọi block: compare với float forward của PyTorch model (≤2 INT8 LSB error)
```

---

## GIAI ĐOẠN 3: LAYER RUNNER & END-TO-END ORACLE

> Mục tiêu: Chạy đúng chuỗi L0→L22 và sinh ra P3/P4/P5.

### 3.1. Bước 3A – Layer Specs Table

```
File: layer_specs.py
─────────────────────────────────────────────────────────────
LAYER_SPECS = [
  LayerSpec(
    idx=0, block_type="Conv",
    primitive_seq=["RS_DENSE_3x3"],
    in_shape=[1,3,640,640], out_shape=[1,16,320,320],
    stride=2, kernel=3,
    sources=[-1],        # chỉ dùng output của layer trước
    hold_output=False,   # không cần giữ lại
  ),
  LayerSpec(idx=4, ..., hold_output=True,  hold_until=15),  # F4 giữ đến L15
  LayerSpec(idx=6, ..., hold_output=True,  hold_until=12),  # F6 giữ đến L12
  LayerSpec(idx=8, ..., hold_output=True,  hold_until=21),  # F8 giữ đến L21
  LayerSpec(idx=12, block_type="QConcat",
    sources=[11, 6],    # L11 (upsample) + L6 (skip)
    is_output_P=None),
  LayerSpec(idx=13, ..., hold_output=True, hold_until=18),  # F13 giữ đến L18
  LayerSpec(idx=16, ..., output_name="P3"),  # P3 output
  LayerSpec(idx=19, ..., output_name="P4"),  # P4 output
  LayerSpec(idx=22, ..., output_name="P5"),  # P5 output
  ...
]

Đây là nguồn chân lý duy nhất.
model_forward_runner.py KHÔNG được hardcode bất cứ gì ngoài LayerSpec.
```

---

### 3.2. Bước 3B – Model Forward Runner

```
File: model_forward_runner.py
─────────────────────────────────────────────────────────────
def model_forward(X_int8, scale_in, zp_in, weights_dict, layer_specs):
  """
  Chạy toàn bộ layer 0–22, trả về P3/P4/P5 và stage_outputs.
  """
  stage_outputs = {}   # key = layer_idx, value = (tensor_int8, scale, zp)
  hold_buffer = {}     # key = layer_idx, value = tensor đang được giữ
  
  current = (X_int8, scale_in, zp_in)
  
  for spec in layer_specs:
    # 1. Lấy inputs theo sources[-1 hoặc list]
    if spec.sources == [-1]:
      inputs = [current]
    else:
      inputs = []
      for src_idx in spec.sources:
        if src_idx == -1:
          inputs.append(current)
        else:
          inputs.append(stage_outputs[src_idx])  # từ buffer
    
    # 2. Barrier check: đảm bảo tất cả dependencies đã hoàn thành
    for src_idx in spec.sources:
      if src_idx != -1:
        assert src_idx in stage_outputs, \
          f"BARRIER FAIL: Layer {spec.idx} needs L{src_idx} but not ready"
    
    # 3. Gọi block tương ứng
    if spec.block_type == "Conv":
      out = rs_dense_3x3(inputs[0][0], weights_dict[spec.idx], ...)
    elif spec.block_type == "QC2f":
      out = block_qc2f(inputs[0][0], weights_dict[spec.idx], ...)
    elif spec.block_type == "SCDown":
      out = block_scdown(inputs[0][0], weights_dict[spec.idx], ...)
    elif spec.block_type == "SPPF":
      out = block_sppf(inputs[0][0], weights_dict[spec.idx], ...)
    elif spec.block_type == "QPSA":
      out = block_qpsa(inputs[0][0], weights_dict[spec.idx], ...)
    elif spec.block_type == "Upsample":
      out = upsample_nearest(inputs[0][0], inputs[0][1], inputs[0][2])
    elif spec.block_type == "QConcat":
      out = concat([inp[0] for inp in inputs],
                   [inp[1] for inp in inputs],
                   [inp[2] for inp in inputs])
    elif spec.block_type == "QC2fCIB":
      out = block_qc2fcib(inputs[0][0], weights_dict[spec.idx], ...)
    
    # 4. Lưu layer output
    stage_outputs[spec.idx] = out
    current = out
    
    # 5. Nếu cần giữ (hold_output), đánh dấu
    if spec.hold_output:
      hold_buffer[spec.idx] = out
    
    # 6. Giải phóng buffer khi không còn cần
    for idx in list(hold_buffer.keys()):
      spec_held = get_spec(idx)
      if spec.idx >= spec_held.hold_until:
        del hold_buffer[idx]  # giải phóng
  
  P3 = stage_outputs[16]
  P4 = stage_outputs[19]
  P5 = stage_outputs[22]
  
  return {
    "P3": P3, "P4": P4, "P5": P5,
    "stage_outputs": stage_outputs
  }
```

---

### 3.3. Bước 3C – End-to-End Test

```
File: test_model_forward.py
─────────────────────────────────────────────────────────────
Test Suite (PHẢI PASS TRƯỚC KHI ĐI XUỐNG RTL):

Test 1 – Shape correctness:
  result = model_forward(X_random_int8, ...)
  assert result["P3"][0].shape == (1, 64, 80, 80)
  assert result["P4"][0].shape == (1, 128, 40, 40)
  assert result["P5"][0].shape == (1, 256, 20, 20)
  assert result["P3"][0].dtype == np.int8
  assert result["P4"][0].dtype == np.int8
  assert result["P5"][0].dtype == np.int8

Test 2 – Quant metadata valid:
  for name, (tensor, scale, zp) in [("P3",P3), ("P4",P4), ("P5",P5)]:
    assert scale > 0
    assert isinstance(zp, int)
    assert np.all(tensor >= -128) and np.all(tensor <= 127)

Test 3 – Stage output count:
  assert len(result["stage_outputs"]) == 23  # L0 to L22

Test 4 – Skip dependency resolved:
  # Đảm bảo barrier logic đúng
  assert 12 in result["stage_outputs"]  # QConcat L12 đã chạy
  assert 15 in result["stage_outputs"]  # QConcat L15 đã chạy
  assert 18 in result["stage_outputs"]  # QConcat L18 đã chạy
  assert 21 in result["stage_outputs"]  # QConcat L21 đã chạy

Test 5 – Accuracy vs float reference:
  # Chạy qYOLOv10n PyTorch model ở float precision với cùng mock input
  P3_float_ref = pytorch_model_float(X_test_float)["P3"]
  P3_our_dequant = dequantize(result["P3"][0], result["P3"][1], result["P3"][2])
  
  # Sai số cho phép: ≤ 2% RMSE difference (typical for INT8 PTQ)
  rmse = np.sqrt(np.mean((P3_float_ref - P3_our_dequant)**2))
  assert rmse < threshold, f"P3 RMSE too large: {rmse}"

Test 6 – Determinism:
  result_1 = model_forward(X_fixed, ...)
  result_2 = model_forward(X_fixed, ...)
  assert np.array_equal(result_1["P3"][0], result_2["P3"][0])
```

---

## GIAI ĐOẠN 4: RTL IMPLEMENTATION

> Bắt đầu RTL chỉ khi Giai đoạn 3 PASS hoàn toàn.

### 4.1. Thứ tự phát triển RTL (theo dependency)

```
Level 0 – Package (Không có circuit, chỉ là type definitions):
  accel_pkg.sv     ← primitive IDs, constants (LANES=16, BANKS=3/4)
  desc_pkg.sv      ← descriptor structs (NET/LAYER/TILE/ROUTER/POST)
  csr_pkg.sv       ← CSR register map

Level 1 – Memory Leaf Modules:
  glb_input_bank.sv  ← 3-bank circular buffer (bank = h%3)
  glb_output_bank.sv ← 4-bank output buffer (bank = out_row%4)
  glb_weight_bank.sv ← weight SRAM interface
  psum_buffer.sv     ← INT32 accumulator buffer

Level 2 – Address Generation:
  addr_gen_input.sv  ← bank + row_slot + Wblk + lane → physical addr
  addr_gen_weight.sv ← weight address generation per primitive
  addr_gen_output.sv ← output address with bank rotation
  row_slot_manager.sv ← Q_in computation, slot rotation

Level 3 – Compute Primitives:
  window_gen.sv     ← 3×3/1×1/5×5/7×7 window extraction
  pe_lane_mac.sv    ← INT8×INT8→INT32 MAC, 16 lanes parallel
  column_reduce.sv  ← Horizontal accumulation across Cin chunks
  pool_compare.sv   ← 25-input INT8 max tree (MAXPOOL_5x5)
  ppu_lite.sv       ← bias_add + fixed-point requant + SiLU_LUT + clamp

Level 4 – Cluster Level:
  pe_cluster.sv     ← window_gen + pe_lane_mac + column_reduce (Dense mode)
  pe_cluster_dw.sv  ← Depthwise mode (per-channel, no cross-channel reduce)
  pool_engine.sv    ← window_gen_5x5 + pool_compare
  gemm_attn_engine.sv ← Matrix GEMM for attention (optional, or use pe_cluster)

Level 5 – Data Movement:
  router_cluster.sv   ← GLB input→PE routing, broadcast control
  swizzle_engine.sv   ← Tensor transpose/reshape
  tensor_post_engine.sv ← UPSAMPLE_NEAREST (address remapping DMA)
  concat_engine.sv    ← Channel concatenation with optional mini-requant

Level 6 – Control:
  tile_fsm.sv         ← Tile loop control (h, w, cin chunk, cout chunk)
  desc_fetch_engine.sv ← Fetch/parse descriptor stack from DDR/SRAM
  barrier_manager.sv  ← Producer/Consumer sync cho skip connections
  subcluster_wrapper.sv ← Glue wrapper cho block-level composition

Level 7 – Top Level:
  accel_top.sv        ← DMA + memory controller + subcluster_wrapper + CSR
```

---

### 4.2. Chiến lược Verification RTL (Golden Python làm oracle)

```
Cho mỗi RTL module, verify theo quy trình:

STEP A – Unit Test với hand-crafted test vector:
  Python: compute expected_output = primitive_func(test_input)
  RTL:    apply test_input → simulate → capture output
  Compare: assert output_rtl == expected_output (bit-exact)

STEP B – Golden Reference Test:
  Python golden: stage_outputs = model_forward(X_rand)
  RTL simulation: run same X_rand through RTL
  Compare: stage_outputs_rtl vs stage_outputs_golden (layer by layer)

STEP C – Regression với 100 random inputs:
  for i in range(100):
    X = random_int8_input()
    P3_golden, P4_golden, P5_golden = golden_model(X)
    P3_rtl, P4_rtl, P5_rtl = rtl_simulation(X)
    assert bit_exact_equal(P3_rtl, P3_golden)
    assert bit_exact_equal(P4_rtl, P4_golden)
    assert bit_exact_equal(P5_rtl, P5_golden)
```

---

### 4.3. RTL Checklist theo từng Primitive

```
RS_DENSE_3x3 / OS_1x1:
  ✓ window_gen tạo đúng 9 pixels cho mỗi output position
  ✓ pe_lane_mac: 16 lanes MAC cùng lúc, output INT32 không overflow
  ✓ column_reduce: tích lũy đúng qua Cin chunks (last_cin flag)
  ✓ psum_buf: hold state đúng across Cin passes
  ✓ ppu_lite: bias_add → (M_int * psum) >> shift → clamp → INT8
  ✓ SiLU LUT: 256 entries pre-loaded, index = y_int8 + offset
  ✓ padding: zeros correction ở biên ảnh (edge_tile flag)
  ✓ PSUM_MODE: khi NOT last_pass, output không ra GLB_OUTPUT
  ✓ ACT_MODE: khi last_pass, PPU kích hoạt và write INT8 ra

DW_3x3 / DW_7x7_MULTIPASS:
  ✓ pe_cluster_dw mode: mỗi lane chỉ MAC 1 channel (groups=C)
  ✓ per-channel scale: ppu_lite nhận scale_w[c] riêng cho từng channel
  ✓ dw7x7 pass control: last_kernel flag chỉ set ở pass cuối (row 6)
  ✓ PSUM carry-over: PSUM accumulated đúng qua 3 passes

MAXPOOL_5x5:
  ✓ window_gen_5x5: 25 pixel window
  ✓ max_tree: binary max tree, 5 levels deep, INT8 unsigned compare
  ✓ NO PPU path: output scale/zp = input scale/zp

UPSAMPLE_NEAREST:
  ✓ tensor_post_engine: phát 4 write addresses cho mỗi read address
  ✓ scale/zp metadata pass-through (không có compute)

CONCAT:
  ✓ router_cluster: chuyển channel A trước, rồi channel B sau
  ✓ mini-requant path: nếu LAYER_DESC chỉ định alignment cần thiết
  ✓ barrier: CONCAT block đợi cả 2 producer done (barrier_manager)

GEMM_ATTN_BASIC:
  ✓ K^T transpose: đọc K theo column-first thay vì row-first
  ✓ Attn_scale: fixed-point division by sqrt(Hq)
  ✓ softmax_lut: piecewise linear hay table lookup
```

---

## TỔNG HỢP: ĐỒ THỊ PHỤ THUỘC & THỜI GIAN

```
                    THỜI GIAN (tương đối)
Phase 0 Spec:       ████ (1x)
                         │
Phase 1A Quant:     ████ (1x)    ← CRITICAL PATH
                         │
Phase 1B Conv:      ████ (1x)
Phase 1C DW:        ██   (0.5x)
Phase 1D Pool:      █    (0.25x)
Phase 1E PSA:       ████ (1x)
                         │
Phase 2A Layout:    ████ (1x)    ← PARALLEL với Phase 2B
Phase 2B Blocks:    ██████ (1.5x)
                         │
Phase 3A Specs:     █    (0.25x)
Phase 3B Runner:    ████ (1x)
Phase 3C E2E Test:  ████ (1x)    ← GATE trước RTL
                         │
Phase 4 RTL:        ████████████████████████████ (6x+)

Total trước RTL ≈ 10x unit → sau đó RTL sẽ nhanh và ít lỗi hơn nhiều
```

---

## QUY TẮC VÀNG – KHÔNG ĐƯỢC VI PHẠM

### Rule 1: Không đi xuống khi test fail
```
Nếu test_quant.py FAIL          → FIX quant_affine.py trước khi code primitive
Nếu test_primitives.py FAIL     → FIX primitive trước khi code block
Nếu test_blocks.py FAIL         → FIX block trước khi chạy model_forward
Nếu test_model_forward.py FAIL  → FIX runner trước khi viết RTL
Nếu RTL unit test FAIL          → FIX RTL module trước khi integration
```

### Rule 2: Một nguồn chân lý duy nhất
```
quant_domain_align.py   = nguồn chân lý cho CONCAT/ADD requant
layer_specs.py          = nguồn chân lý cho layer sequence & dependencies
quant_affine.py         = nguồn chân lý cho phép tính quantize/requant
```

### Rule 3: Không tự implement lại shared logic
```
Mọi block PHẢI gọi quant_domain_align.py  (không tự viết concat align)
Mọi layer PHẢI đọc từ layer_specs.py       (không hardcode sequence)
Mọi RTL module PHẢI có golden test        (từ Python oracle)
```

### Rule 4: Trace dump là bắt buộc từ đầu
```
Mỗi block: dump intermediate tensors (flag dump=True)
DW_7x7: dump PSUM sau từng pass
CONCAT: dump scale_A, scale_B, scale_common trước và sau align
model_forward: dump toàn bộ 23 stage_outputs
```

### Rule 5: Quantization metadata KHÔNG được mất
```
Mọi function nhận INT8 tensor PHẢI nhận (scale, zp) cùng lúc
Mọi function trả INT8 tensor PHẢI trả (tensor, scale, zp) cùng lúc
Không bao giờ trả tensor đơn lẻ mà không kèm quant metadata
```

---

## CHECKLIST TỔNG THỂ TRƯỚC KHI ĐI XUỐNG RTL

```
PHASE 0 – Spec:
  ☐ 8 file spec được review và sign-off
  ☐ Layer dependency (4 QConcat skip) xác nhận đúng

PHASE 1 – Primitives:
  ☐ test_quant.py: 100% PASS (quant + common-domain align)
  ☐ test_primitives.py: PASS conv3x3/1x1/dw3x3/dw7x7/pool/upsample/concat/add/psa
  ☐ DW7x7_multipass == monolithic: VERIFIED

PHASE 2 – Blocks & Layout:
  ☐ test_layout.py: PASS bank/row_slot/pack16/address/psum_act
  ☐ test_blocks.py: PASS QC2f/SCDown/SPPF/QPSA/QC2fCIB

PHASE 3 – End-to-End:
  ☐ test_model_forward.py: PASS shape/dtype/metadata/barrier/accuracy
  ☐ P3[1,64,80,80], P4[1,128,40,40], P5[1,256,20,20]: VERIFIED
  ☐ Stage outputs 0–22 được dump và lưu làm oracle

PHASE 4 – RTL Gateway:
  ☐ accel_pkg.sv + desc_pkg.sv: định nghĩa xong trước viết bất kỳ RTL nào
  ☐ Mỗi RTL leaf module có test bench với golden vector từ Python
```

---

*Flow này đảm bảo: nếu RTL sai → lỗi bị phát hiện tại tầng Golden Python, không phải tại synthesis hay silicon.*  
*Nguyên tắc: Càng fix sớm → càng rẻ và nhanh.*
