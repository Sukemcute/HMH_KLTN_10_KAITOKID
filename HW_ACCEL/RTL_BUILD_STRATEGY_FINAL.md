# CHIẾN LƯỢC XÂY DỰNG RTL PHẦN CỨNG — BẢN CUỐI
## YOLOv10n INT8 Accelerator trên Xilinx Virtex-7 VC707

> **Ngày**: 2026-03-26
> **Target FPGA**: Xilinx VC707 Board → **XC7VX485T** (KHÔNG PHẢI XC7VX690T)
> **Mục tiêu**: Primitive PASS → Layer PASS → Inference PASS (bit-exact với Golden Python)
> **Tiêu chí thành công**: Output P3/P4/P5 từ RTL khớp Golden Python, mAP50 ≈ 0.93

---

## 0. PHÂN TÍCH THAY ĐỔI SO VỚI CÁC PHASE ĐÃ ĐỀ XUẤT

### 0.1. Những gì KHÔNG cần thay đổi (đã verified trong documentation)

| Kết quả verification | Ý nghĩa cho RTL |
|---|---|
| 14 primitive PASS (100% verified) | Semantic tính toán đã đúng → RTL chỉ cần implement đúng cùng công thức |
| 23 layer PASS (L0-L22, 100 samples/layer) | Pipeline giữa primitive đã đúng → RTL FSM follow cùng sequence |
| mAP50 = 0.9302 trên 7,902 ảnh | Functional parity đã chứng minh → RTL target cùng golden vectors |
| Upsample/QConcat = 100% bit-exact | Các primitive data-movement trivial → RTL đơn giản |
| Conv/SCDown/SPPF > 99.9% match | Compute core ổn định → RTL focus chính xác MAC + PPU |

### 0.2. Những gì CẦN THAY ĐỔI (critical discoveries từ documentation)

| Discovery | Tài liệu nguồn | Impact lên RTL |
|---|---|---|
| **Rounding: Half-up (NOT floor)** | `detailed_primitive_modification_report.md` | PPU requant PHẢI dùng `(acc*M + (1<<(shift-1))) >> shift`. Nếu dùng floor → sai ~30% |
| **Signed domain (int8, NOT quint8)** | `dtype_alignment_strategy.md` | Toàn bộ RTL PHẢI dùng **signed int8 [-128,127]**. ZP offset: `zp_hw = zp_pytorch - 128` |
| **INT64 accumulator cho 640x640** | `detailed_primitive_modification_report.md` | Accumulator 32-bit overflow ở deep layers (Cin=256, 640x640). Cần **INT40 hoặc INT64** |
| **Activation = ReLU (KHÔNG PHẢI SiLU)** | `mapping_conv_block.md`, verify code | Model QAT dùng **ReLU**, không phải SiLU. `(act): ReLU(inplace=True)` |
| **ZP-corrected matmul (QPSA)** | `mapping_qpsa_block.md` | Layer 10 attention: PHẢI trừ ZP trước multiply. Thiếu → sai hoàn toàn |
| **VC707 = XC7VX485T** | User requirement | V2 full (3,360 DSPs) KHÔNG VỪA XC7VX485T (2,800 DSPs). Cần **V2-lite** |
| **QPSA chỉ 83.6% match** | `exhaustive_verification_640_report.md` | 14 sequential primitives gây tích lũy rounding noise. RTL chấp nhận ≤25 LSB |

### 0.3. Điều chỉnh kiến trúc cho VC707 (XC7VX485T)

```
XC7VX485T Resource Budget:
┌───────────────────┬──────────┬────────────────────────────────────────────┐
│ Resource          │ Available│ Phương án V2-lite                          │
├───────────────────┼──────────┼────────────────────────────────────────────┤
│ DSP48E1           │ 2,800    │ 2,304 (82%) = 12 sub × 12 PE × 16 DSP    │
│ BRAM36K           │ 1,030    │ ~520 (50%) = 12 sub × 36 + 88 misc       │
│ LUT6              │ 303,600  │ ~210K (69%) = control + DSP correction    │
│ FF                │ 607,200  │ ~240K (40%) = pipeline registers          │
└───────────────────┴──────────┴────────────────────────────────────────────┘

V2-lite config cho VC707:
  LANES         = 32          (giữ nguyên, critical cho spatial mapping)
  PE_ROWS       = 3           (giữ nguyên, 3 kernel rows)
  PE_COLS       = 4           (giữ nguyên)
  SUPER_CLUSTERS = 4          (giữ nguyên)
  SUBS_PER_SC   = 3           (giảm từ 4 → 3)
  ACTIVE_PER_SC = 2           (Dual-RUNNING giữ nguyên)
  Total subs    = 12          (giảm từ 16)
  MACs/cycle    = 2,304       (đủ cho ~80-100 FPS)
  DSP usage     = 12 × 192 = 2,304 DSPs (82% of 2,800)
```

---

## 1. CHIẾN LƯỢC XÂY DỰNG RTL — 6 PHASES

### Tổng quan: Bottom-Up V-Model

```
Phase 1: Compute Atoms         ← dsp_pair_int8, pe_unit (BIT-EXACT test)
Phase 2: Primitive Engines     ← conv3x3, conv1x1, dwconv3x3, maxpool5x5, dwconv7x7
Phase 3: PPU + Memory + Router ← bias+requant+act+clamp, GLB banks, addr_gen
Phase 4: Layer Composition     ← QC2f, SCDown, SPPF, QConcat, Upsample
Phase 5: System Integration    ← subcluster, tile_fsm, DMA, scheduler
Phase 6: End-to-End Inference  ← L0-L22 chạy xuyên suốt, verify P3/P4/P5
```

---

## PHASE 1: COMPUTE ATOMS (Tuần 1-2)

### Mục tiêu: Chứng minh MAC computation bit-exact

### 1A. `dsp_pair_int8` — Nền tảng (PHẢI ĐÚNG 100%)

**File**: Reuse `PHASE_3/01_compute_leaf/rtl/dsp_pair_int8.sv`

**Test criteria** (từ documentation verification):
```
Requirement: signed int8 × signed int8 → signed int32
             Với unsigned-offset DSP48E1 packing (2 MACs/DSP)

Test 1: Exhaustive corner cases
  (-128)×(-128) = 16384   ✓
  127×127 = 16129          ✓
  (-128)×127 = -16256      ✓
  0×anything = 0           ✓
  1×1 = 1                  ✓

Test 2: 9-cycle accumulation (conv3x3 pattern)
  Σ_{i=0..8} x_a[i] × w[i]  ==  golden_sum  ✓

Test 3: Random stress 10,000 products
  100% match required        ✓

CRITICAL FIX từ documentation:
  Correction formula: signed_prod = raw_u - 128*(x_u + w_u) + 16384
  Khi x=0, w=0: correction = 16384 - 128*256 + 16384 = 0 ✓ (zero-safe)
```

**Pass criteria**: 0 errors trên 65,536 exhaustive + 10,000 random

### 1B. `pe_unit` — 32-lane PE

**File**: Reuse `PHASE_3/01_compute_leaf/rtl/pe_unit.sv`

**Test criteria**:
```
Mode PE_RS3: Weight per-lane (dense conv)
  - 32 lanes × 9 cycles (3×3 kernel) → verify each lane independently
  - Compare vs golden: Σ_{kw=0..2, cin=0..C} x[lane][cin]*w[kw][cin]

Mode PE_OS1: Weight broadcast (1×1 conv)
  - w_sel[l] = w_in[0] for all lanes
  - Verify: all 32 lanes get same weight

Mode PE_DW3: Per-channel weight (depthwise)
  - Each lane = 1 channel
  - Verify per-channel accumulation

Pipeline timing: 4-cycle latency
  - Feed 9 values → wait 4 → read psum
  - Verify en gating: psum holds when en=0
```

### 1C. `column_reduce` — 3-row summation

```
Verify: col_psum[c][l] = pe_psum[0][c][l] + pe_psum[1][c][l] + pe_psum[2][c][l]
Latency: 1 cycle
Test: Random 3×4×32 INT32 values, exact match
```

### 1D. `comparator_tree` — MaxPool 5×5

```
Verify: max of 25 signed INT8 inputs per lane
Pipeline: 5 stages (25→13→7→4→2→1)
Test: Exhaustive boundaries (-128, 127), random stress
```

### 1E. `silu_lut` — SiLU activation

```
CRITICAL: LUT values PHẢI khớp Golden Python
Precompute: silu_lut[i] = clamp(round(SiLU((i-128)/scale)*scale), -128, 127)

Verify: 256 entries × golden comparison
Test: 32 parallel reads per cycle
```

**Phase 1 Sign-off**: Mọi compute atom PHẢI 100% bit-exact. Không có tolerance.

---

## PHASE 2: PRIMITIVE ENGINES (Tuần 3-5)

### Mục tiêu: Mỗi primitive tính đúng 1 tile output

**Thứ tự xây dựng** (theo tần suất sử dụng trong YOLOv10n):

### 2A. RS_DENSE_3x3 (P0) — Conv 3×3

**Sử dụng tại**: L0, L1, L3, L17, và bên trong mọi QC2f block

**Kiến trúc engine**:
```
Input SRAM → [3 Row Buffers] → 3× pe_unit → Column Reduce → PPU → Output SRAM
                                  (kh=0,1,2)    (sum 3 rows)

FSM: IDLE → LOAD_ROWS → LOAD_WGT → COMPUTE(3 kw cycles) → NEXT_CIN → DRAIN → PPU → WRITE
```

**PPU pipeline** (CRITICAL — từ documentation fixes):
```python
# Stage 1: Bias add (INT32 + INT32)
biased = acc + bias[cout]

# Stage 2: Requant với HALF-UP rounding (KHÔNG PHẢI floor!)
mult = int64(biased) * int64(m_int[cout])
rounded = mult + (1 << (shift[cout] - 1))    # ← FIX quan trọng nhất
shifted = rounded >> shift[cout]

# Stage 3: Clamp to 16-bit → Activation
if act_mode == SILU:
    act_val = silu_lut[clamp(shifted + 128, 0, 255)]  # Index = shifted + 128
elif act_mode == RELU:
    act_val = max(0, clamp(shifted, -128, 127))
else:
    act_val = clamp(shifted, -128, 127)

# Stage 4: ZP output + final clamp
output = clamp(act_val + zp_out, -128, 127)
```

**Golden test vectors** (lấy từ L0 verification):
```
Input:  X_int8[1, 3, 8, 64] (signed, padded)
Weight: W_int8[16, 3, 3, 3] (signed, ZP=0)
Bias:   B_int32[16]
m_int:  M_int32[16]
shift:  S_uint6[16]
Output: Y_int8[1, 16, 4, 32] (stride=2)

Expected accuracy: >99.9% bit-exact (từ L0 verification = 99.99%)
```

### 2B. OS_1x1 (P1) — Conv 1×1

**Sử dụng tại**: QC2f cv1/cv2, SCDown cv1, SPPF cv1/cv2

**Khác biệt vs 3×3**:
- Chỉ 1 pe_unit (không cần 3 rows, không cần column_reduce)
- PE_OS1 mode: broadcast weight
- 1 cycle/cin (không có kw loop)
- Không cần row buffer (đọc trực tiếp SRAM)

**Expected accuracy**: >99.9% (QC2f cv1/cv2 verified)

### 2C. DW_3x3 (P2) — Depthwise 3×3

**Sử dụng tại**: SCDown (L5, L7, L20)

**Khác biệt vs dense conv**:
- Cout = Cin (per-channel)
- Không accumulate across channels
- Per-channel bias, m_int, shift
- Chỉ 3 kw cycles per channel (không có cin loop)

**Expected accuracy**: >99.9% (SCDown verified = 99.94%)

### 2D. MAXPOOL_5x5 (P3) — Max Pooling

**Sử dụng tại**: SPPF L9 (×3 cascade)

**Đặc điểm**:
- Không MAC, chỉ comparison → dùng comparator_tree
- 5 row buffers thay vì 3
- Padding = 2, Stride = 1
- KHÔNG qua PPU (output scale/zp = input scale/zp)
- Cascadable: output của lần 1 = input lần 2

**Expected accuracy**: 99.93% (SPPF verified)

### 2E. DW_7x7_MULTIPASS (P8) — Depthwise 7×7

**Sử dụng tại**: QC2fCIB L22

**3-pass strategy**:
```
Pass 1: kh=0,1,2 → PE accumulates 7 kw → store PSUM_INT32
Pass 2: kh=3,4,5 → PE accumulates 7 kw → add to PSUM → store
Pass 3: kh=6     → PE accumulates 7 kw → add to PSUM → PPU → INT8
```

**Expected accuracy**: 99.96% (QC2fCIB verified)

### Phase 2 Test Strategy

Mỗi primitive engine test với **3 cấp độ**:

| Cấp | Test | Nguồn golden | Pass criteria |
|-----|------|--------------|---------------|
| Unit | Minimal tile (Cin=1, Cout=1, 32×32) | Manual calculation | 100% exact |
| Integration | Realistic tile (matching L0/L5/L9 params) | `verify_layer_X.py` output | ≥99% match |
| Stress | Random (Cin=64, Cout=64, 80×80) | Golden Python runtime | ≥97% match |

---

## PHASE 3: PPU + MEMORY + DATA MOVEMENT (Tuần 5-7)

### 3A. PPU Pipeline (4-stage registered)

**CRITICAL REQUIREMENTS** (từ documentation fixes):

```systemverilog
// Stage 2: Requant — MUST use half-up rounding
automatic logic signed [63:0] mult;
automatic logic signed [63:0] rounded;

mult = 64'(biased) * 64'(m_int);        // INT64 multiply!
if (shift > 0)
    rounded = mult + (64'sd1 <<< (shift - 1));  // Half-up rounding
else
    rounded = mult;
shifted = 32'(rounded >>> shift);

// KHÔNG ĐƯỢC dùng: shifted = 32'(mult >>> shift);  ← Đây là floor, SAI!
```

**Accumulator width decision**:
```
Worst case: Cin=384, kernel=3×3, input=127, weight=127
Max acc = 384 × 9 × 127 × 127 = 55,741,536
→ Fits INT32 (max 2,147,483,647) ✓

Nhưng tại 640×640 với chaining layers:
Max biased = acc + bias (bias can be large)
→ Dùng INT40 (5 bytes) cho safety, hoặc INT64 cho PPU multiply

Quyết định: Accumulator = INT32, PPU multiply = INT64
```

### 3B. GLB Memory Banks (on-chip SRAM)

```
Input banks:  3 banks (h mod 3), mỗi bank = 32 subbanks (1 per lane)
Weight banks: 3 banks (per kernel row)
Output banks: 4 banks (out_row mod 4), dual namespace PSUM/ACT

Per subcluster BRAM estimate (V2-lite):
  Input:  3 × ~16KB = 48KB → 11 BRAM36K
  Weight: 3 × ~12KB = 36KB → 8 BRAM36K
  Output: 4 × ~12KB = 48KB → 11 BRAM36K
  Misc:   ~4KB → 1 BRAM36K
  Total:  ~31 BRAM36K per sub

12 subclusters: 372 BRAM36K (36% of 1,030) ✓
```

### 3C. Address Generators

```
addr_gen_input:  (h, w, cin) → (bank_id, sram_addr)
  bank_id = h mod 3
  ⚠️ Padding: out-of-bounds → output cfg_zp_x (NOT zero!)

addr_gen_weight: mode-dependent
  RS3: bank_id = kh, addr = f(cout, cin, kw)
  OS1: bank_id = cin_slice, addr = f(cout, cin)
  DW:  bank_id = kh, addr = f(ch, kw)

addr_gen_output: (h_out, w_out, cout) → (bank_id, addr)
  bank_id = pe_col (0-3)
```

### 3D. Data Movement (Router + Swizzle)

```
router_cluster:
  RIN: 3 input banks → 3 PE rows (select per rin_src[])
  RWT: 3 weight banks → 3 PE rows (select per rwt_src[])
  RPS: 4 PE columns → 4 output banks (psum write)
  Bypass: MOVE (passthrough), CONCAT (channel offset), UPSAMPLE (2× dup)

swizzle_engine:
  Transform output_bank(ACT) → input_bank for next layer
  Modes: identity, upsample_nearest(2×), concat(channel offset)
```

---

## PHASE 4: LAYER COMPOSITION (Tuần 7-10)

### Mục tiêu: Mỗi YOLOv10n block type tính đúng

### 4A. Layer Conv (L0, L1, L3, L17)

```
Primitive sequence: RS_DENSE_3x3(stride=2) + PPU(SiLU)

Test: L0 golden vectors
  Input:  X_int8[1, 3, 640, 640] (từ Quantized_model.ipynb)
  Output: Y_int8[1, 16, 320, 320]
  Compare vs verify_layer_0.py output → expect 99.99% match
```

### 4B. QC2f Block (L2, L4, L6, L8, L13, L16, L19)

```
Primitive sequence (6 steps):
  Step 1: cv1 = OS_1x1(Cin→C_internal) + PPU(SiLU)
  Step 2: Split (C_internal → C_half + C_half)
  Step 3: bottleneck_cv1 = RS_DENSE_3x3(C_half→C_half) + PPU(SiLU)
  Step 4: bottleneck_cv2 = RS_DENSE_3x3(C_half→C_half) + PPU(SiLU)
  Step 5: CONCAT(bottleneck_output, skip_half)
          ⚠️ Domain alignment nếu scale khác nhau!
  Step 6: cv2 = OS_1x1(C_internal→Cout) + PPU(SiLU)

CRITICAL từ documentation:
  - CONCAT cần common-domain requant nếu scale_A ≠ scale_B
  - Split là logical (channel indexing), không cần compute
  - Intermediate tensors phải fit trong GLB

Test: L2 golden vectors → expect ~99% match (verified = 99.09%)
```

### 4C. SCDown Block (L5, L7, L20)

```
Primitive sequence:
  Step 1: cv1 = OS_1x1(Cin→Cout) + PPU(SiLU)
  Step 2: cv2 = DW_3x3(Cout, stride=2) + PPU(SiLU)

Test: L5 golden vectors → expect >99.9% (verified = 99.90%)
```

### 4D. SPPF Block (L9)

```
Primitive sequence (6 steps):
  Step 1: cv1 = OS_1x1(256→128) + PPU(SiLU)
  Step 2: pool1 = MAXPOOL_5x5(128)
  Step 3: pool2 = MAXPOOL_5x5(128) (cascaded from pool1)
  Step 4: pool3 = MAXPOOL_5x5(128) (cascaded from pool2)
  Step 5: CONCAT(cv1_out, pool1, pool2, pool3) → [512]
          ⚠️ Tất cả 4 tensors cùng scale/zp → concat trivial
  Step 6: cv2 = OS_1x1(512→256) + PPU(SiLU)

CRITICAL: pool1/pool2/pool3 outputs PHẢI buffer đồng thời (3×128×20×20 = 153.6KB)

Test: L9 golden vectors → expect 99.93%
```

### 4E. QConcat (L12, L15, L18, L21)

```
Primitive: CONCAT với domain alignment

CRITICAL từ detailed_concat_logic_report.md:
  if scale_A ≠ scale_B:
    // Requant nhánh nhỏ hơn về common scale
    A_aligned = round((A_int8 - zp_A) × (scale_A/scale_common)) + zp_common
  CONCAT(A_aligned, B) theo channel dimension

⚠️ Skip connections:
  L12: F11(upsample) + F6(backbone) → barrier_L12
  L15: F14(upsample) + F4(backbone) → barrier_L15
  L18: F17(PAN) + F13(FPN)         → barrier_L18
  L21: F20(SCDown) + F8(backbone)  → barrier_L21

Test: L12 golden vectors → expect 100% (verified = 100.00%)
```

### 4F. Upsample (L11, L14)

```
Primitive: UPSAMPLE_NEAREST(2×)
  y[h,w,c] = x[h//2, w//2, c]
  scale_out = scale_in, zp_out = zp_in (unchanged)

Implementation: Address remapping in swizzle_engine
  dst[c][2h  ][2w  ] = src[c][h][w]
  dst[c][2h  ][2w+1] = src[c][h][w]
  dst[c][2h+1][2w  ] = src[c][h][w]
  dst[c][2h+1][2w+1] = src[c][h][w]

Test: L11 golden vectors → expect 100% (verified = 100.00%)
```

### 4G. QC2fCIB (L22)

```
Primitive sequence:
  Step 1: cv1 = OS_1x1(384→256) + PPU(SiLU)
  Step 2: DW_7x7_MULTIPASS(256, pad=3) + PPU(SiLU)   ← 3 passes
  Step 3: OS_1x1 compression
  Step 4: CONCAT + cv2 = OS_1x1

Test: L22 golden vectors → expect 99.96%
```

### 4H. QPSA (L10) — Optional / Deferred

```
⚠️ Layer 10 là phức tạp nhất (14 sequential primitives)
⚠️ Accuracy chỉ 83.6% trong software (do rounding noise)
⚠️ Cần INT8 matmul, softmax LUT, multi-head reshape

KHUYẾN NGHỊ: Defer QPSA đến sau khi L0-L22 (trừ L10) chạy đúng
  - Giai đoạn đầu: bypass L10 bằng identity hoặc software fallback
  - Giai đoạn sau: implement GEMM_ATTN_BASIC trên PE cluster
```

### Phase 4 Test Strategy

```
Per-block test:
  1. Load golden input + weight + bias/quant từ verify_layer_X.py
  2. Run RTL engine
  3. Compare output vs golden → report match % và max LSB diff

Pass criteria (từ documentation baseline):
  Conv:     ≥99.9% match, ≤1 LSB
  QC2f:     ≥97% match, ≤4 LSB
  SCDown:   ≥99.9% match, ≤2 LSB
  SPPF:     ≥99.9% match, ≤1 LSB
  QConcat:  100% match, 0 LSB
  Upsample: 100% match, 0 LSB
  QC2fCIB:  ≥99.9% match, ≤1 LSB
  QPSA:     ≥83% match, ≤25 LSB (nếu implement)
```

---

## PHASE 5: SYSTEM INTEGRATION (Tuần 10-14)

### 5A. Subcluster Wrapper

```
Tích hợp trong 1 subcluster:
  tile_fsm → shadow_reg_file → GLB banks → addr_gen → router →
  window_gen → pe_cluster → ppu → swizzle_engine

FSM states:
  IDLE → LOAD_CFG → PREFILL_WT → PREFILL_IN → WAIT_READY →
  RUN_COMPUTE → ACCUMULATE → POST_PROCESS → SWIZZLE_STORE → DONE

Test: Run 1 layer (L0) trên 1 subcluster → verify output
```

### 5B. SuperCluster (4 subclusters + arbiter + DMA)

```
V2-lite cho VC707:
  3 subclusters per SC (thay vì 4)
  Dual-RUNNING: 2 compute + 1 fill/drain
  1 tensor_dma shared

Local arbiter manages role rotation:
  Sub-0: RUNNING(tile_A)  Sub-1: RUNNING(tile_B)  Sub-2: FILLING(tile_C)
  → rotate khi sub finish
```

### 5C. System Top (4 SC + controller)

```
accel_top.sv:
  AXI-Lite slave (MMIO) → controller_system
  AXI4 master (DDR) → 4× supercluster_wrapper
  controller_system: CSR + desc_fetch + barrier_manager + global_scheduler

Barrier points (4):
  barrier_L12: wait(L11_done AND F6_ready)
  barrier_L15: wait(L14_done AND F4_ready)
  barrier_L18: wait(L17_done AND F13_ready)
  barrier_L21: wait(L20_done AND F8_ready)
```

---

## PHASE 6: END-TO-END INFERENCE (Tuần 14-16)

### 6A. Full L0-L22 Inference

```
Test flow:
  1. CPU loads descriptors (NET_DESC + 23 LAYER_DESC + N TILE_DESC) vào DDR
  2. CPU loads input X_int8[1,3,640,640] vào DDR
  3. CPU loads weights + bias + quant params vào DDR
  4. CPU writes CSR_START → accelerator begins
  5. Accelerator processes L0→L1→...→L22 (tự động qua descriptor chain)
  6. Accelerator writes P3/P4/P5 vào DDR, asserts IRQ
  7. CPU reads P3/P4/P5 → compare vs Golden Python

Pass criteria:
  P3[1,64,80,80]:   ≥97% bit-exact vs golden
  P4[1,128,40,40]:  ≥97% bit-exact vs golden
  P5[1,256,20,20]:  ≥97% bit-exact vs golden

  Nếu feed P3/P4/P5 vào Qv10Detect head → mAP50 ≈ 0.93 trên validation set
```

### 6B. Multi-image Validation

```
Test: 100 images từ ALPR dataset
  Per-image: compare P3/P4/P5 vs golden
  Summary: average match %, max LSB diff distribution

Final sign-off: mAP50 ≥ 0.92 (< 1% degradation from golden 0.9302)
```

---

## 2. BẢNG TỔNG HỢP: FILE RTL CẦN XÂY DỰNG

### Reuse từ PHASE_3 (đã có, đã test):

| Module | File | Lines | Status |
|--------|------|-------|--------|
| dsp_pair_int8 | PHASE_3/01_compute_leaf/rtl/dsp_pair_int8.sv | 104 | Reuse |
| pe_unit | PHASE_3/01_compute_leaf/rtl/pe_unit.sv | 74 | Reuse |
| column_reduce | PHASE_3/01_compute_leaf/rtl/column_reduce.sv | 49 | Reuse |
| comparator_tree | PHASE_3/01_compute_leaf/rtl/comparator_tree.sv | 114 | Reuse |
| silu_lut | PHASE_3/01_compute_leaf/rtl/silu_lut.sv | 31 | Reuse |
| ppu | PHASE_3/02_ppu/rtl/ppu.sv | 211 | Reuse (verify rounding!) |

### Mới xây dựng trong HW_ACCEL:

| Module | File | Purpose | Phase |
|--------|------|---------|-------|
| yolo_accel_pkg | packages/yolo_accel_pkg.sv | Package definitions | 1 |
| conv3x3_engine | rtl/conv3x3_engine.sv | P0: RS_DENSE_3x3 | 2 |
| conv1x1_engine | rtl/conv1x1_engine.sv | P1: OS_1x1 | 2 |
| dwconv3x3_engine | rtl/dwconv3x3_engine.sv | P2: DW_3x3 | 2 |
| maxpool5x5_engine | rtl/maxpool5x5_engine.sv | P3: MAXPOOL_5x5 | 2 |
| dwconv7x7_engine | rtl/dwconv7x7_engine.sv | P8: DW_7x7_MULTIPASS | 2 |

### Cần xây dựng thêm (Phase 3-5):

| Module | Purpose | Phase |
|--------|---------|-------|
| glb_input_bank | Input activation SRAM | 3 |
| glb_weight_bank | Weight SRAM + FIFO | 3 |
| glb_output_bank | Dual namespace PSUM/ACT | 3 |
| addr_gen_input | Input address generator | 3 |
| addr_gen_weight | Weight address generator | 3 |
| addr_gen_output | Output address generator | 3 |
| window_gen | Sliding window (K=1,3,5,7) | 3 |
| router_cluster | Data routing hub | 3 |
| swizzle_engine | Layout transform | 3 |
| metadata_ram | Slot validity manager | 3 |
| shadow_reg_file | Config snapshot | 4 |
| tile_fsm | Tile execution FSM | 4 |
| pe_cluster | 3×4 PE array | 4 |
| subcluster_wrapper | Full tile datapath | 5 |
| barrier_manager | Skip dependency sync | 5 |
| local_arbiter | Dual-RUNNING scheduler | 5 |
| tensor_dma | AXI4 DMA controller | 5 |
| supercluster_wrapper | 3 sub + arbiter + DMA | 5 |
| desc_fetch_engine | Descriptor parser | 5 |
| global_scheduler | Tile dispatcher | 5 |
| controller_system | CSR + fetch + barrier | 5 |
| accel_top | System top-level | 5 |

---

## 3. QUY TẮC VÀNG (GOLDEN RULES)

Từ toàn bộ documentation verification, đúc kết 10 quy tắc bất di bất dịch:

```
RULE 1: Signed INT8 [-128, 127] everywhere. ZP_hw = ZP_pytorch - 128
RULE 2: Half-up rounding: (acc*M + (1<<(sh-1))) >> sh. NEVER floor.
RULE 3: INT64 cho PPU multiply. INT32 cho accumulator.
RULE 4: Model QAT dùng ReLU (y = max(0,x)), KHÔNG dùng SiLU. Giữ SiLU LUT cho tính tổng quát.
RULE 5: Padding fill = zero_point_x (KHÔNG PHẢI 0)
RULE 6: Per-channel weight: scale_w[cout], bias[cout], m_int[cout], shift[cout]
RULE 7: CONCAT cần domain alignment nếu scale khác nhau
RULE 8: DW_7x7 multipass: PSUM namespace pass 1,2; PPU chỉ pass 3 (last)
RULE 9: Barrier sync cho 4 skip connections: L12, L15, L18, L21
RULE 10: Output = clamp(act_val + zp_out, -128, 127). ZP_out thêm SAU activation.
```

---

## 4. TIMELINE DỰ KIẾN

```
Tuần 1-2:   Phase 1 — Compute atoms (verify 100% exact)
Tuần 3-5:   Phase 2 — 5 primitive engines + testbenches
Tuần 5-7:   Phase 3 — PPU + Memory + Router
Tuần 7-10:  Phase 4 — Layer composition (L0-L22 blocks)
Tuần 10-14: Phase 5 — System integration (subcluster → accel_top)
Tuần 14-16: Phase 6 — End-to-end inference verification

Milestone 1 (Tuần 5):  Mọi primitive PASS golden test
Milestone 2 (Tuần 10): Mọi layer block PASS golden test
Milestone 3 (Tuần 16): Full inference L0-L22 PASS, mAP50 ≈ 0.93
```

---

## 5. KẾT LUẬN

### So sánh phase cũ vs mới:

| Khía cạnh | Phase cũ (PHASE_3) | Phase mới (điều chỉnh) |
|---|---|---|
| Rounding | Không specify | **Half-up bắt buộc** |
| Data type | Implicit unsigned | **Signed int8, ZP-128 offset** |
| Accumulator | INT32 | **INT32 acc + INT64 PPU multiply** |
| Target FPGA | XC7VX690T | **XC7VX485T (VC707)** |
| Architecture | V2 full (16 sub) | **V2-lite (12 sub, 2304 MACs)** |
| QPSA | Must implement | **Defer, optional** |
| Verification | Per-module TB | **Golden vectors từ documentation verify** |
| Test dataset | None | **7,902 images, mAP validation** |

### Kết luận:

1. **Documentation verification đã chứng minh** rằng Golden Python implementation chính xác. RTL chỉ cần replicate cùng arithmetic.

2. **4 critical fixes** (rounding, signed domain, INT64, ZP correction) PHẢI được apply từ đầu. Nếu thiếu bất kỳ fix nào → accuracy drop 30-90%.

3. **VC707 (XC7VX485T) vừa đủ** cho V2-lite (12 subclusters, 2304 MACs, ~80-100 FPS).

4. **Thứ tự build bottom-up** từ compute atom → primitive → layer → system đảm bảo mỗi level verified trước khi lên level tiếp.

5. **QPSA (Layer 10)** nên defer vì phức tạp nhất và tolerance cao nhất (83.6%). 22 layers còn lại đủ để chứng minh accelerator hoạt động.

---
---

# ════════════════════════════════════════════════════════════════
# PHẦN 6: CHECKLIST CHI TIẾT — RTL IMPLEMENT → TEST PASS
# ════════════════════════════════════════════════════════════════

> **Mục tiêu**: Theo checklist này từ trên xuống dưới → RTL tính đúng primitive → đúng layer → đúng inference.
> Mỗi mục có: (1) Công thức toán CHÍNH XÁC, (2) Mô tả RTL cần implement, (3) Tiêu chí PASS lấy từ kết quả verification đã chứng minh.

---

## CHECKLIST A: QUY TẮC NỀN TẢNG (Apply TRƯỚC KHI viết bất kỳ RTL nào)

### A1. Domain: Signed INT8 [-128, 127]

```
☐ Toàn bộ activation, weight, output đều là signed int8
☐ Khi import từ PyTorch (quint8 [0,255]):
    X_hw  = X_pytorch - 128
    ZP_hw = ZP_pytorch - 128
☐ Weight ZP luôn = 0 (symmetric weight quantization)

Chứng minh toán học (từ dtype_alignment_strategy.md):
  (X_s - ZP_s) = (X_u - 128) - (ZP_u - 128) = X_u - ZP_u  ✓
  → Kết quả tính toán KHÔNG đổi khi shift cả data và ZP
```

### A2. Rounding: Half-Up (KHÔNG PHẢI Floor)

```
☐ Mọi phép right-shift trong PPU:
    y = (acc * M_int + (1 << (shift - 1))) >> shift     ← ĐÚNG (half-up)
    y = (acc * M_int) >> shift                            ← SAI (floor)

☐ Impact nếu dùng floor: ~30% pixels sai 1 LSB
☐ Rounding mode chung: Banker's Rounding (Half-Even) cho quantize_affine
```

### A3. Accumulator: INT32 cho MAC, INT64 cho PPU multiply

```
☐ MAC accumulator: signed 32-bit (đủ cho Cin ≤ 384, kernel ≤ 7×7)
    Max: 384 × 49 × 127 × 127 = 303,464,448 < 2^31 ✓
☐ PPU multiply: signed 64-bit
    mult = int64(biased) * int64(M_int)
    rounded = mult + (1 << (shift - 1))
    result = int32(rounded >> shift)
```

### A4. Padding: Fill = Zero-Point (KHÔNG PHẢI 0)

```
☐ Khi input pixel nằm ngoài biên (out-of-bounds):
    pad_value = cfg_zp_x (zero-point của input)
    KHÔNG PHẢI pad_value = 0
☐ Lý do: trong INT8 quantized, "true zero" = ZP, không phải 0
```

### A5. Activation: Model QAT dùng ReLU

```
☐ Từ mapping_conv_block.md: (act): ReLU(inplace=True)
☐ Verify code: "relu" if "ReLU" in str(type(qconv_module.act)) else "none"
☐ RTL implementation:
    if (act_mode == RELU)
        y = (x > 0) ? x : 0;    // Trivial: 1 comparator
    else  // act_mode == NONE (Identity)
        y = x;
☐ SiLU LUT KHÔNG dùng cho model hiện tại (giữ cho tính tổng quát)
```

---

## CHECKLIST B: PRIMITIVE RTL — Công thức & Pass Criteria

### B1. P0 — RS_DENSE_3x3 (Conv 3×3)

**Dùng tại**: L0, L1, L3, L17, bên trong QC2f (bottleneck cv1/cv2)

```
☐ CÔNG THỨC TOÁN (từ primitive_modification_report):

  Cho mỗi output pixel (h_out, w_out, cout):

  Step 1 — MAC accumulation (INT32):
    acc = 0
    for cin in [0, Cin):
      for kh in [0, 3):
        for kw in [0, 3):
          h_in = h_out * stride + kh - pad
          w_in = w_out * stride + kw - pad
          x_val = (h_in, w_in trong bounds) ? X[h_in][w_in][cin] : ZP_x
          acc += int32(x_val) * int32(W[cout][cin][kh][kw])

  Step 2 — Zero-point correction + Bias:
    acc_corrected = acc - ZP_x * sum(W[cout, :, :, :])
    acc_biased = acc_corrected + B_int32[cout]

  Step 3 — Requant (INT64 multiply, half-up rounding):
    mult = int64(acc_biased) * int64(M_int[cout])
    rounded = mult + (1 << (shift[cout] - 1))
    y_raw = int32(rounded >> shift[cout])

  Step 4 — Activation:
    if ReLU: y_act = max(0, clamp(y_raw, -128, 127))
    if None: y_act = clamp(y_raw, -128, 127)

  Step 5 — Output ZP + final clamp:
    Y[h_out][w_out][cout] = clamp(y_act + ZP_out, -128, 127)

☐ RTL CẦN:
  - 3 pe_unit instances (1 per kernel row kh)
  - Mỗi pe_unit: 32 lanes (LANES DSP pairs)
  - Column reduce: sum 3 PE outputs
  - PPU inline: bias → requant(INT64) → ReLU → clamp+ZP
  - Row buffers cho sliding window

☐ PASS CRITERIA (từ verification):
  - L0 (Cin=3, Cout=16, stride=2):  ≥99.99% match, ≤1 LSB
  - L1 (Cin=16, Cout=32, stride=2): ≥99.96% match, ≤1 LSB
  - L3 (Cin=32, Cout=64, stride=2): ≥99.98% match, ≤1 LSB
  - L17 (Cin=64, Cout=64, stride=2): ≥99.99% match, ≤1 LSB
```

### B2. P1 — OS_1x1 (Conv 1×1)

**Dùng tại**: QC2f cv1/cv2, SCDown cv1, SPPF cv1/cv2

```
☐ CÔNG THỨC TOÁN:

  Cho mỗi output pixel (h, w, cout):

  Step 1 — MAC (không có kh/kw loop, kernel=1×1):
    acc = 0
    for cin in [0, Cin):
      acc += int32(X[h][w][cin]) * int32(W[cout][cin])

  Step 2-5: Giống P0 (correction, bias, requant, activation, ZP)

☐ RTL CẦN:
  - 1 pe_unit (PE_OS1 mode: broadcast weight to all lanes)
  - Không cần row buffer (đọc trực tiếp)
  - 1 cycle per cin (vs 3 kw cycles trong P0)
  - PPU inline

☐ PASS CRITERIA:
  - OS_1x1 verified qua QC2f cv1/cv2: ≥99% match
```

### B3. P2 — DW_3x3 (Depthwise Conv 3×3)

**Dùng tại**: SCDown cv2 (L5, L7, L20)

```
☐ CÔNG THỨC TOÁN:

  Depthwise: Cout = Cin, mỗi channel độc lập

  Cho mỗi output pixel (h_out, w_out, c):

  Step 1 — MAC per-channel:
    acc = 0
    for kh in [0, 3):
      for kw in [0, 3):
        h_in = h_out * stride + kh - pad
        w_in = w_out * stride + kw - pad
        x_val = (bounds) ? X[h_in][w_in][c] : ZP_x
        acc += int32(x_val) * int32(W[c][kh][kw])    // W per-channel!

  Step 2 — Correction + Bias (per-channel):
    acc_corrected = acc - ZP_x * sum(W[c, :, :])
    acc_biased = acc_corrected + B_int32[c]

  Step 3-5: Requant per-channel: M_int[c], shift[c], ZP_out

☐ RTL CẦN:
  - 3 pe_unit (PE_DW3 mode)
  - Per-channel weight, bias, M_int, shift
  - Chỉ 3 kw cycles (không có cin accumulation)

☐ PASS CRITERIA (SCDown verified):
  - L5: ≥99.90% match, ≤1 LSB
  - L7: ≥99.92% match, ≤1 LSB
  - L20: ≥99.99% match, ≤1 LSB
```

### B4. P3 — MAXPOOL_5x5

**Dùng tại**: SPPF L9 (×3 cascade)

```
☐ CÔNG THỨC TOÁN:

  Cho mỗi output pixel (h, w, c):
    Y[h][w][c] = max over 5×5 window:
      max_{kh=0..4, kw=0..4} X[h+kh-2][w+kw-2][c]

  Out-of-bounds padding = -128 (INT8_MIN, identity for max)
  scale_out = scale_in, ZP_out = ZP_in (unchanged!)
  KHÔNG qua PPU

☐ RTL CẦN:
  - comparator_tree (25 inputs → 1 max, 5-stage pipeline)
  - 5 row buffers (thay vì 3)
  - Stride=1, Pad=2

☐ PASS CRITERIA (SPPF verified):
  - L9 SPPF: ≥99.93% match, ≤1 LSB
  - Cascade: P1=MP(X), P2=MP(P1), P3=MP(P2) phải đúng cả 3 lần
```

### B5. P5 — CONCAT (Channel Concatenation)

**Dùng tại**: QConcat L12, L15, L18, L21; QC2f internal

```
☐ CÔNG THỨC TOÁN:

  Case 1: scale_A == scale_B (trivial):
    Y = [A, B] along channel dim

  Case 2: scale_A ≠ scale_B (domain alignment):
    Cho mỗi input tensor cần align:
      X_float = (X_int8 - ZP_src) * scale_src
      X_aligned = clamp(round(X_float / scale_target) + ZP_target, -128, 127)
    Y = [A_aligned, B_aligned] along channel dim

☐ RTL CẦN:
  - Router channel interleaving
  - Optional mini-PPU cho domain alignment requant
  - Barrier sync cho skip connections (L12, L15, L18, L21)

☐ PASS CRITERIA (QConcat verified):
  - L12, L15, L18, L21: 100.00% match, 0 LSB
```

### B6. P6 — UPSAMPLE_NEAREST (2×)

**Dùng tại**: L11, L14

```
☐ CÔNG THỨC TOÁN:

  Y[c][2h  ][2w  ] = X[c][h][w]
  Y[c][2h  ][2w+1] = X[c][h][w]
  Y[c][2h+1][2w  ] = X[c][h][w]
  Y[c][2h+1][2w+1] = X[c][h][w]

  scale_out = scale_in, ZP_out = ZP_in (unchanged!)
  KHÔNG qua PPU, KHÔNG cần compute

☐ RTL CẦN:
  - Address remapping trong swizzle_engine
  - Hoặc DMA pattern: mỗi source byte → 4 destination bytes

☐ PASS CRITERIA:
  - L11, L14: 100.00% match, 0 LSB
```

### B7. P7 — EWISE_ADD (Element-wise Addition)

**Dùng tại**: QC2fCIB residual (L22), QC2f bottleneck shortcut

```
☐ CÔNG THỨC TOÁN (từ detailed_add_logic_report.md):

  Golden Path (float domain):
    A_float = (A_int8 - ZP_A) * scale_A
    B_float = (B_int8 - ZP_B) * scale_B
    sum_float = A_float + B_float
    Y_int8 = clamp(round(sum_float / scale_out) + ZP_out, -128, 127)

  ⚠️ KHÔNG ĐƯỢC dùng: Y = A + B - ZP (thiếu domain alignment → sai 255 LSB)

☐ RTL CẦN:
  - Dequant cả hai nhánh (hoặc dùng integer rescale)
  - Add trong domain chung
  - Requant output

☐ PASS CRITERIA:
  - Verified: 100.00% match, 0 LSB
```

### B8. P8 — DW_7x7_MULTIPASS

**Dùng tại**: QC2fCIB L22 internal

```
☐ CÔNG THỨC TOÁN:

  Depthwise 7×7, chia 3 passes (fit 3 PE rows):

  Pass 1 (kh = 0,1,2): 3 PEs accumulate 7 kw values
    psum_pass1[c] = Σ_{kh=0..2, kw=0..6} X[h+kh][w+kw-3][c] * W[c][kh][kw]

  Pass 2 (kh = 3,4,5): 3 PEs accumulate 7 kw values
    psum_pass2[c] = psum_pass1[c] + Σ_{kh=3..5, kw=0..6} ...

  Pass 3 (kh = 6): 1 PE accumulates 7 kw values (PE[1],PE[2] idle)
    psum_pass3[c] = psum_pass2[c] + Σ_{kw=0..6} X[h+6][w+kw-3][c] * W[c][6][kw]
    → PPU: bias[c] → requant(M_int[c], shift[c]) → ReLU → ZP → output

  Pass 1,2: Store PSUM (INT32) → KHÔNG qua PPU
  Pass 3 (last): PPU fires → output INT8

☐ RTL CẦN:
  - 3 pe_unit instances
  - PSUM buffer: [num_wblk][LANES] INT32 (giữ giữa passes)
  - Weight: [C][7][7] = 49 weights per channel
  - 7 kw cycles per pass (vs 3 cho P0)

☐ PASS CRITERIA (QC2fCIB verified):
  - L22 overall: ≥99.96% match, ≤1 LSB
```

### B9. P4 — MOVE (Buffer Copy)

```
☐ CÔNG THỨC: Y = X (identity copy)
☐ RTL: DMA copy hoặc address remap
☐ PASS: 100% exact (trivial)
```

---

## CHECKLIST C: BLOCK/LAYER COMPOSITION — Ghép Primitive → Layer

### C1. Conv Block (L0, L1, L3, L17)

```
☐ Primitive sequence:
    1 primitive call: P0 (RS_DENSE_3x3) với stride=2, activation=ReLU

☐ Parameter extraction:
    W_int8 = conv.weight()                    # [Cout, Cin, 3, 3] signed int8
    B_int32 = conv.bias()                     # [Cout] signed int32
    scale_y = conv.scale                      # float32
    zp_y = conv.zero_point - 128              # ← SHIFT!
    M_int, shift = decompose(scale_x * scale_w / scale_y)
    activation = "relu"

☐ PASS CRITERIA:
  | Layer | Cin | Cout | Stride | Verified Match | Max LSB |
  |-------|-----|------|--------|----------------|---------|
  | L0    | 3   | 16   | 2      | 99.99%         | 1       |
  | L1    | 16  | 32   | 2      | 99.96%         | 1       |
  | L3    | 32  | 64   | 2      | 99.98%         | 1       |
  | L17   | 64  | 64   | 2      | 99.99%         | 1       |
```

### C2. QC2f Block (L2, L4, L6, L8, L13, L16, L19)

```
☐ Primitive sequence (6 steps — từ mapping_qc2f_block.md):

  Step 1: cv1 = P1(OS_1x1, Cin → 2×c_)        + ReLU
  Step 2: Split y[0], y[1] (channel indexing, no compute)
  Step 3: bottleneck.cv1 = P0(RS_DENSE_3x3, c_ → c_) + ReLU
  Step 4: bottleneck.cv2 = P0(RS_DENSE_3x3, c_ → c_) + ReLU
  Step 5: CONCAT(bottleneck_out, y[0])           + Domain Alignment
  Step 6: cv2 = P1(OS_1x1, 2×c_ → Cout)        + ReLU

  ⚠️ Nếu có n>1 bottleneck: repeat Step 3-4, output chain vào nhau
  ⚠️ Step 5 CONCAT: scale khác nhau giữa bottleneck và split → PHẢI domain align

☐ Shortcut (trong bottleneck):
  Nếu m.add == True:
    output = P7(EWISE_ADD, bottleneck_out, shortcut_input)
    scale_out, zp_out từ m.fl_func.scale / m.fl_func.zero_point

☐ PASS CRITERIA:
  | Layer | Cin  | Cout | Verified Match | Max LSB |
  |-------|------|------|----------------|---------|
  | L2    | 32   | 32   | 99.09%         | 3       |
  | L4    | 64   | 64   | 96.52%         | 3       |
  | L6    | 128  | 128  | 94.46%         | 3       |
  | L8    | 256  | 256  | 99.20%         | 2       |
  | L13   | 384  | 128  | 98.53%         | 2       |
  | L16   | 192  | 64   | 99.02%         | 9       |
  | L19   | 192  | 128  | 98.94%         | 3       |
```

### C3. SCDown Block (L5, L7, L20)

```
☐ Primitive sequence (2 steps — từ mapping_scdown_block.md):

  Step 1: cv1 = P1(OS_1x1, Cin → Cout)         + ReLU
  Step 2: cv2 = P2(DW_3x3, Cout, stride=2)     + ReLU

  ⚠️ P2 PHẢI dùng cv1 output ZP cho padding (không phải original input ZP)

☐ PASS CRITERIA:
  | Layer | Cin→Cout   | Verified Match | Max LSB |
  |-------|------------|----------------|---------|
  | L5    | 64→128     | 99.90%         | 1       |
  | L7    | 128→256    | 99.92%         | 1       |
  | L20   | 128→128    | 99.99%         | 1       |
```

### C4. SPPF Block (L9)

```
☐ Primitive sequence (6 steps — từ mapping_sppf_block.md):

  Step 1: cv1 = P1(OS_1x1, 256 → 128)          + ReLU
  Step 2: pool1 = P3(MAXPOOL_5x5, 128)          (no PPU)
  Step 3: pool2 = P3(MAXPOOL_5x5, 128)          input = pool1 output
  Step 4: pool3 = P3(MAXPOOL_5x5, 128)          input = pool2 output
  Step 5: CONCAT(cv1_out, pool1, pool2, pool3)   → [512]
          Scale/ZP giống nhau → concat trivial (no domain align)
  Step 6: cv2 = P1(OS_1x1, 512 → 256)          + ReLU

  ⚠️ 4 tensors (cv1, pool1, pool2, pool3) phải buffer đồng thời

☐ PASS CRITERIA:
  - L9: 99.94% match, ≤1 LSB
```

### C5. QConcat (L12, L15, L18, L21)

```
☐ Primitive: P5(CONCAT) với domain alignment

  L12: CONCAT(F11_upsample[256,40,40], F6_backbone[128,40,40]) → [384,40,40]
  L15: CONCAT(F14_upsample[128,80,80], F4_backbone[64,80,80])  → [192,80,80]
  L18: CONCAT(F17_conv[64,40,40], F13_fpn[128,40,40])          → [192,40,40]
  L21: CONCAT(F20_scdown[128,20,20], F8_backbone[256,20,20])   → [384,20,20]

  ⚠️ SKIP dependencies — feature map phải HOLD trong GLB:
    F4 hold L4 → L15 (11 layers, 409.6 KB)
    F6 hold L6 → L12 (6 layers, 204.8 KB)
    F8 hold L8 → L21 (13 layers, 102.4 KB)
    F13 hold L13 → L18 (5 layers, 204.8 KB)

  ⚠️ BARRIER sync: L12/L15/L18/L21 chỉ start khi BOTH inputs ready

☐ PASS CRITERIA:
  - L12, L15, L18, L21: 100.00% match, 0 LSB
```

### C6. Upsample (L11, L14)

```
☐ Primitive: P6(UPSAMPLE_NEAREST, factor=2)
☐ L11: [256,20,20] → [256,40,40]
☐ L14: [128,40,40] → [128,80,80]
☐ PASS: 100.00% match, 0 LSB
```

### C7. QC2fCIB Block (L22)

```
☐ Primitive sequence (từ mapping_qc2fcib_block.md):

  Step 1: cv1 = P1(OS_1x1, 384 → 256)          + ReLU
  Step 2: Split → identity branch + process branch

  QCIB internal (process branch):
    Step 3a: P2(DW_3x3, channels)               + ReLU
    Step 3b: P1(OS_1x1, channels)               + ReLU
    Step 3c: DW_7x7(channels, pad=3)            + ReLU    ← P8 MULTIPASS
    Step 3d: P1(OS_1x1, channels)               + ReLU
    Step 3e: P2(DW_3x3, channels)               + ReLU
    Step 3f: P7(EWISE_ADD, process + shortcut)   ← Residual

  Step 4: CONCAT(identity, processed)
  Step 5: cv2 = P1(OS_1x1, 256 → Cout)         + ReLU

  ⚠️ DW 7x7 cần padding=3, 3-pass multipass
  ⚠️ Residual add cần domain alignment

☐ PASS CRITERIA:
  - L22: 99.96% match, ≤1 LSB
```

### C8. QPSA Block (L10) — Optional/Deferred

```
☐ Primitive sequence (14 sequential ops — từ mapping_qpsa_block.md):

  Step 1: cv1 = P1(OS_1x1) → split branch_a, branch_b
  Step 2: QKV projection = 3× P1(OS_1x1)
  Step 3: Multi-head split (reshape, no compute)
  Step 4: Attention = P10(INT8_MATMUL: Q × K^T)
          ⚠️ ZP correction: Q_true = Q - ZP_Q, K_true = K - ZP_K
  Step 5: Scale by 1/√d
  Step 6: P11(SOFTMAX_APPROX via LUT)
  Step 7: P10(INT8_MATMUL: Score × V)
  Step 8: P2(DW_3x3) positional encoding
  Step 9: P1(OS_1x1) output projection
  Step 10: P7(EWISE_ADD) residual
  Step 11: FFN = P1 + P1
  Step 12: P7(EWISE_ADD) residual

☐ PASS CRITERIA:
  - L10: ~83.52% match, ≤23 LSB (rounding noise qua 14 stages)
  - Acceptable vì error không propagate sang layers sau
```

---

## CHECKLIST D: FULL INFERENCE L0-L22

### D1. Layer Execution Order

```
☐ Sequential execution (respect dependencies):

  L0  → L1  → L2  → L3  → L4* → L5  → L6* → L7  → L8* → L9  → L10
  → L11 → L12(F11+F6*) → L13* → L14 → L15(F14+F4*) → L16
  → L17 → L18(F17+F13*) → L19 → L20 → L21(F20+F8*) → L22

  * = output cần HOLD cho skip connection
```

### D2. Output Targets

```
☐ P3 = L16 output: INT8 [1, 64, 80, 80]    + scale_P3, zp_P3
☐ P4 = L19 output: INT8 [1, 128, 40, 40]   + scale_P4, zp_P4
☐ P5 = L22 output: INT8 [1, 256, 20, 20]   + scale_P5, zp_P5
```

### D3. Full Model Pass Criteria

```
☐ Per-layer match (RTL vs Golden Python):

  | Layer | Type     | Target Match | Target Max LSB |
  |-------|----------|-------------|----------------|
  | 0     | Conv     | ≥99.9%      | ≤1             |
  | 1     | Conv     | ≥99.9%      | ≤1             |
  | 2     | QC2f     | ≥99.0%      | ≤3             |
  | 3     | Conv     | ≥99.9%      | ≤1             |
  | 4     | QC2f     | ≥96.0%      | ≤3             |
  | 5     | SCDown   | ≥99.9%      | ≤1             |
  | 6     | QC2f     | ≥94.0%      | ≤3             |
  | 7     | SCDown   | ≥99.9%      | ≤1             |
  | 8     | QC2f     | ≥99.0%      | ≤2             |
  | 9     | SPPF     | ≥99.9%      | ≤1             |
  | 10    | QPSA     | ≥83.0%      | ≤25            |
  | 11    | Upsample | 100.0%      | 0              |
  | 12    | QConcat  | 100.0%      | 0              |
  | 13    | QC2f     | ≥98.0%      | ≤2             |
  | 14    | Upsample | 100.0%      | 0              |
  | 15    | QConcat  | 100.0%      | 0              |
  | 16    | QC2f     | ≥99.0%      | ≤9             |
  | 17    | Conv     | ≥99.9%      | ≤1             |
  | 18    | QConcat  | 100.0%      | 0              |
  | 19    | QC2f     | ≥98.0%      | ≤3             |
  | 20    | SCDown   | ≥99.9%      | ≤1             |
  | 21    | QConcat  | 100.0%      | 0              |
  | 22    | QC2fCIB  | ≥99.9%      | ≤1             |

☐ End-to-end functional: mAP50 ≥ 0.92 trên validation dataset
   (Golden Python đạt 0.9302 → RTL chấp nhận ≤1% degradation)
```

### D4. Skip Connection Buffer Checklist

```
☐ F4 (L4 → L15): HOLD 409.6 KB qua 11 layers
☐ F6 (L6 → L12): HOLD 204.8 KB qua 6 layers
☐ F8 (L8 → L21): HOLD 102.4 KB qua 13 layers (longest!)
☐ F13 (L13 → L18): HOLD 204.8 KB qua 5 layers
☐ Total: ~921.6 KB simultaneous GLB reservation

☐ Barrier sync:
  barrier_L12: wait(L11_done AND F6_ready) → release L12
  barrier_L15: wait(L14_done AND F4_ready) → release L15
  barrier_L18: wait(L17_done AND F13_ready) → release L18
  barrier_L21: wait(L20_done AND F8_ready) → release L21
```

---

## CHECKLIST E: TEST FLOW — Từ Golden Python → RTL Verify

### E1. Tạo Golden Test Vectors

```
☐ Chạy verify_layer_X.py cho layer X:
    - Export input tensor (signed int8, domain-shifted)
    - Export weights, bias, M_int, shift, ZP
    - Export expected output tensor
    - Format: .npy hoặc .hex cho RTL testbench

☐ Mỗi primitive test ít nhất 3 bộ vectors:
    - Minimal (Cin=1, small spatial)
    - Realistic (match actual layer params)
    - Stress (random, large, edge cases)
```

### E2. RTL Testbench Structure

```
☐ Per-primitive TB:
    1. Load golden input/weight/params vào behavioral SRAM
    2. Configure engine (cfg_*)
    3. Assert start, wait done
    4. Read output from SRAM
    5. Compare bit-by-bit vs golden expected
    6. Report: match%, max_diff, PASS/FAIL

☐ Per-layer TB:
    1. Chain primitive engines theo block sequence
    2. Intermediate tensors passed via shared SRAM
    3. Compare final output vs golden layer output

☐ Full inference TB:
    1. Load L0 input + all weights/params
    2. Run L0 → L22 sequentially
    3. Extract P3, P4, P5 outputs
    4. Compare vs golden oracle_P3.npy, oracle_P4.npy, oracle_P5.npy
```

### E3. Sign-off Criteria

```
☐ Level 1 — Primitive PASS:
    Tất cả 9 primitive (P0-P8) pass golden test
    ≥3 test vectors mỗi primitive

☐ Level 2 — Block/Layer PASS:
    Tất cả 8 block types pass golden test
    Match % ≥ documented baseline (từ Checklist D3)

☐ Level 3 — Inference PASS:
    Full L0-L22 chạy xuyên suốt
    P3/P4/P5 output khớp golden
    mAP50 ≥ 0.92

☐ Level 4 — Dataset validation (optional):
    100+ images qua RTL simulation
    mAP50 ≈ 0.93 (matching Golden Python)
```

---

## TÓM TẮT: CON ĐƯỜNG TỪ RTL → INFERENCE ĐÚNG

```
┌─────────────────────────────────────────────────────────────────┐
│ A. Apply 5 quy tắc nền tảng (signed, rounding, INT64, pad, ReLU) │
├─────────────────────────────────────────────────────────────────┤
│ B. Implement 9 primitive engines (P0-P8)                        │
│    → Test mỗi primitive vs golden → ALL PASS                   │
├─────────────────────────────────────────────────────────────────┤
│ C. Ghép primitive thành 8 block types                           │
│    (Conv, QC2f, SCDown, SPPF, QConcat, Upsample, QC2fCIB, QPSA)│
│    → Test mỗi block vs golden layer output → ALL PASS          │
├─────────────────────────────────────────────────────────────────┤
│ D. Chain 23 layers L0-L22 với skip connections + barriers       │
│    → Test full inference vs golden P3/P4/P5 → PASS             │
├─────────────────────────────────────────────────────────────────┤
│ E. Validate trên dataset → mAP50 ≈ 0.93                        │
│    → INFERENCE ĐÚNG ĐẮN NHƯ PHẦN MỀM ✓                        │
└─────────────────────────────────────────────────────────────────┘
```

---
---

# ════════════════════════════════════════════════════════════════
# PHẦN 7: MASTER CHECKLIST — TOÀN BỘ DỰ ÁN TỪ ĐẦU ĐẾN CUỐI
# ════════════════════════════════════════════════════════════════

> Checklist này liệt kê **MỌI THỨ** đã có và cần làm — từ nghiên cứu đến RTL chạy inference đúng.
> ✅ = Đã hoàn thành | 🔧 = Cần sửa/cải tiến | ⬜ = Chưa làm

---

## STAGE 0: NGHIÊN CỨU & ĐẶC TẢ

```
✅ 0.1  Phân tích model YOLOv10n forward flow (MODEL_FORWARD_FLOW.md)
✅ 0.2  Trace INT8 dtype/shape cho 23 layers (MODEL_LAYERS_INT8_FLOW.md)
✅ 0.3  Phân tích chi tiết 8 block types (MODEL_BLOCKS_INT8_DETAIL.md)
✅ 0.4  Xác định dependency graph & skip connections (MODEL_LAYER_DEPENDENCIES.md)
✅ 0.5  Định nghĩa 10 primitive P0-P9 (PHASE0/01_primitive_matrix.md)
✅ 0.6  Mapping Layer → Primitive cho L0-L22 (PHASE0/02_layer_mapping.md)
✅ 0.7  Quantization policy freeze (PHASE0/03_quant_policy.md)
✅ 0.8  Memory layout & addressing (PHASE0/04_layout_addressing.md)
✅ 0.9  Descriptor specification (PHASE0/05_descriptor_spec.md)
✅ 0.10 Execution semantics (PHASE0/06_execution_semantics.md)
✅ 0.11 Golden Python plan (PHASE0/07_golden_python_plan.md)
✅ 0.12 RTL mapping plan (PHASE0/08_rtl_mapping_plan.md)
✅ 0.13 HW Mapping Research — Primitive→RTL (HW_MAPPING_RESEARCH.md)
✅ 0.14 Architecture V2 >100FPS design (HW_ARCHITECTURE_V2_100FPS.md)
✅ 0.15 Implementation flow 5-phase (HW_ACCELERATION_IMPL_FLOW.md)
✅ 0.16 RTL Module Spec cho 35+ modules (RTL_MODULE_SPEC.md)
✅ 0.17 Build Strategy bottom-up (PHASE_3/BUILD_STRATEGY.md)
✅ 0.18 Tổng hợp toàn bộ nghiên cứu (TONG_HOP_NGHIEN_CUU.md — 11,561 dòng)
```

**Kết luận Stage 0**: Đặc tả HOÀN CHỈNH — đủ thông tin để implement RTL.

---

## STAGE 1: GOLDEN PYTHON & SOFTWARE VERIFICATION

### 1A. Golden Python Primitives (python_golden_originial/)

```
✅ 1A.1  config.py — Hardware constants (LANES, BANKS, PSUM_BITS)
✅ 1A.2  accel_types.py — Data types và structures
✅ 1A.3  quant/quant_affine.py — quantize, dequantize, requant, SiLU LUT, rounding
✅ 1A.4  quant/quant_domain_align.py — Domain alignment cho CONCAT/ADD
✅ 1A.5  primitives/primitive_conv.py — P0 (RS_DENSE_3x3), P1 (OS_1x1)
✅ 1A.6  primitives/primitive_dw.py — P2 (DW_3x3), P8 (DW_7x7)
✅ 1A.7  primitives/primitive_pool.py — P3 (MAXPOOL_5x5)
✅ 1A.8  primitives/primitive_tensor.py — P4 (MOVE), P5 (CONCAT), P6 (UPSAMPLE), P7 (EWISE_ADD)
✅ 1A.9  primitives/primitive_psa.py — P9 (GEMM_ATTN), P10 (INT8_MATMUL), P11 (SOFTMAX)
✅ 1A.10 tests/test_primitives.py + test_quant.py — Unit tests
```

### 1B. Critical Fixes Đã Apply (documentation/2_Mapping_Strategies/)

```
✅ 1B.1  Half-up rounding fix: (acc*M + (1<<(sh-1))) >> sh
         (detailed_primitive_modification_report.md)
✅ 1B.2  Signed domain shift: ZP_hw = ZP_pytorch - 128
         (dtype_alignment_strategy.md)
✅ 1B.3  INT64 accumulator cho PPU multiply
         (detailed_primitive_modification_report.md)
✅ 1B.4  ZP-corrected matmul cho QPSA
         (mapping_qpsa_block.md)
✅ 1B.5  Padding-aware depthwise (pad=1 cho 3x3, pad=3 cho 7x7)
         (detailed_primitive_modification_report.md)
✅ 1B.6  Activation = ReLU (KHÔNG phải SiLU) trong model QAT
         (mapping_conv_block.md, verify code)
```

### 1C. Block Mapping Strategies (documentation/2_Mapping_Strategies/)

```
✅ 1C.1  mapping_conv_block.md — Conv → 1 primitive call
✅ 1C.2  mapping_qc2f_block.md — QC2f → 6-step pipeline
✅ 1C.3  mapping_scdown_block.md — SCDown → 2-step (OS_1x1 + DW_3x3)
✅ 1C.4  mapping_sppf_block.md — SPPF → 6-step (OS + MP×3 + CAT + OS)
✅ 1C.5  mapping_qc2fcib_block.md — QC2fCIB → multi-step with DW_7x7
✅ 1C.6  mapping_qpsa_block.md — QPSA → 14-step attention
✅ 1C.7  detailed_add_logic_report.md — EWISE_ADD domain fix
✅ 1C.8  detailed_concat_logic_report.md — CONCAT domain alignment
```

### 1D. Per-Layer Verification PASS (documentation/3_Layer_Reports/)

```
✅ 1D.0  L0  Conv     — 99.99% match, ≤1 LSB  (verify_layer_0.py)
✅ 1D.1  L1  Conv     — 99.96% match, ≤1 LSB  (verify_layer_1.py)
✅ 1D.2  L2  QC2f     — 99.09% match, ≤3 LSB  (verify_layer_2.py)
✅ 1D.3  L3  Conv     — 99.98% match, ≤1 LSB  (verify_layer_3.py)
✅ 1D.4  L4  QC2f     — 96.52% match, ≤3 LSB  (verify_layer_4.py)
✅ 1D.5  L5  SCDown   — 99.90% match, ≤1 LSB  (verify_layer_5.py)
✅ 1D.6  L6  QC2f     — 94.46% match, ≤3 LSB  (verify_layer_6.py)
✅ 1D.7  L7  SCDown   — 99.92% match, ≤1 LSB  (verify_layer_7.py)
✅ 1D.8  L8  QC2f     — 99.20% match, ≤2 LSB  (verify_layer_8.py)
✅ 1D.9  L9  SPPF     — 99.94% match, ≤1 LSB  (verify_layer_9.py)
✅ 1D.10 L10 QPSA     — 83.52% match, ≤23 LSB (verify_layer_10.py)
✅ 1D.11 L11 Upsample — 100.0% match, 0 LSB   (verify_layer_11.py)
✅ 1D.12 L12 QConcat  — 100.0% match, 0 LSB   (verify_layer_12.py)
✅ 1D.13 L13 QC2f     — 98.53% match, ≤2 LSB  (verify_layer_13.py)
✅ 1D.14 L14 Upsample — 100.0% match, 0 LSB   (verify_layer_14.py)
✅ 1D.15 L15 QConcat  — 100.0% match, 0 LSB   (verify_layer_15.py)
✅ 1D.16 L16 QC2f     — 99.02% match, ≤9 LSB  (verify_layer_16.py)
✅ 1D.17 L17 Conv     — 99.99% match, ≤1 LSB  (verify_layer_17.py)
✅ 1D.18 L18 QConcat  — 100.0% match, 0 LSB   (verify_layer_18.py)
✅ 1D.19 L19 QC2f     — 98.94% match, ≤3 LSB  (verify_layer_19.py)
✅ 1D.20 L20 SCDown   — 99.99% match, ≤1 LSB  (verify_layer_20.py)
✅ 1D.21 L21 QConcat  — 100.0% match, 0 LSB   (verify_layer_21.py)
✅ 1D.22 L22 QC2fCIB  — 99.96% match, ≤1 LSB  (verify_layer_22.py)
```

### 1E. End-to-End & Dataset Validation

```
✅ 1E.1  exhaustive_verify_model_flow.py — 100 samples, all 23 layers PASS
✅ 1E.2  verify_mapped_features.py — Cumulative error propagation verified
✅ 1E.3  test_mapped_model.py — Single inference functional equivalence
✅ 1E.4  val_mapped_model.py — mAP50 = 0.9302 trên 7,902 images
✅ 1E.5  dataset_validation_report.md — Final sign-off PASS
✅ 1E.6  full_model_flow_640_report.md — 640×640 spatial integrity verified
✅ 1E.7  exhaustive_verification_640_report.md — Master verification signed off
```

**Kết luận Stage 1**: Software golden reference HOÀN CHỈNH, 100% verified. Sẵn sàng cho RTL.

---

## STAGE 2: RTL COMPUTE ATOMS (Đã có từ PHASE_3, cần verify lại)

### 2A. Existing PHASE_3 Leaf Modules

```
✅ 2A.1  accel_pkg.sv (101 dòng) — Package definitions
✅ 2A.2  desc_pkg.sv (77 dòng) — Descriptor types
✅ 2A.3  csr_pkg.sv (20 dòng) — CSR definitions
✅ 2A.4  dsp_pair_int8.sv (104 dòng) — 2-MAC DSP48E1 primitive
✅ 2A.5  pe_unit.sv (74 dòng) — 32-lane PE (16 DSP pairs)
✅ 2A.6  column_reduce.sv (49 dòng) — 3-row cross-reduction
✅ 2A.7  comparator_tree.sv (114 dòng) — MaxPool 5×5 (25→1)
✅ 2A.8  silu_lut.sv (31 dòng) — 256-entry SiLU LUT
✅ 2A.9  ppu.sv (211 dòng) — Bias + Requant + Activation + Clamp

  Testbenches (có sẵn):
✅ 2A.10 tb_dsp_pair_int8.sv (407 dòng) — Exhaustive + random tests
✅ 2A.11 tb_pe_unit.sv (226 dòng)
✅ 2A.12 tb_column_reduce.sv (131 dòng)
✅ 2A.13 tb_comparator_tree.sv (185 dòng)
✅ 2A.14 tb_silu_lut.sv (146 dòng)
✅ 2A.15 tb_ppu.sv (236 dòng)
```

### 2B. Verify Compute Atoms Với Golden Vectors

```
🔧 2B.1  Chạy tb_dsp_pair_int8 → xác nhận 100% bit-exact
         ⚠️ Cần verify: correction formula signed→unsigned→signed
         Test: (-128)×(-128)=16384, 127×127=16129, 0×anything=0

🔧 2B.2  Chạy tb_pe_unit → xác nhận 32-lane MAC đúng
         ⚠️ Verify PE_RS3 mode (per-lane weight) vs PE_OS1 (broadcast)
         ⚠️ Verify pipeline timing: 4-cycle latency, en gating

🔧 2B.3  Chạy tb_column_reduce → xác nhận sum 3 rows exact

🔧 2B.4  Chạy tb_comparator_tree → xác nhận max of 25 signed INT8

🔧 2B.5  Chạy tb_ppu → ⚠️ CRITICAL: verify half-up rounding
         Test: (acc*M + (1<<(sh-1))) >> sh  PHẢI match golden
         Test: ReLU activation (max(0,x)), KHÔNG phải SiLU

🔧 2B.6  Export golden vectors từ python_golden cho từng atom
         Script: verify_layer_0.py → extract intermediate MAC/PPU values
         Format: .hex hoặc .mem cho $readmemh trong testbench
```

**Action cần làm**: Chạy tất cả TB đã có, fix nếu fail, đặc biệt PPU rounding.

---

## STAGE 3: RTL PRIMITIVE ENGINES (HW_ACCEL — Đã có khung, cần hoàn thiện)

### 3A. Existing HW_ACCEL Engine Files

```
✅ 3A.1  yolo_accel_pkg.sv (168 dòng) — Package import accel_pkg types
✅ 3A.2  conv3x3_engine.sv (518 dòng) — P0 RS_DENSE_3x3 engine
✅ 3A.3  conv1x1_engine.sv (433 dòng) — P1 OS_1x1 engine
✅ 3A.4  dwconv3x3_engine.sv (546 dòng) — P2 DW_3x3 engine
✅ 3A.5  maxpool5x5_engine.sv (357 dòng) — P3 MAXPOOL_5x5 engine
✅ 3A.6  dwconv7x7_engine.sv (636 dòng) — P8 DW_7x7_MULTIPASS engine

  Golden Testbenches (có sẵn):
✅ 3A.7  tb_conv3x3_golden.sv (868 dòng) — 4 test scenarios
✅ 3A.8  tb_conv1x1_golden.sv (577 dòng) — 4 test scenarios
✅ 3A.9  tb_dwconv3x3_golden.sv (443 dòng) — 3 test scenarios
✅ 3A.10 tb_maxpool5x5_golden.sv (419 dòng) — 3 tests incl. SPPF cascade

  Simulation script:
✅ 3A.11 compile_all.do (123 dòng) — Vivado xvlog/xelab/xsim
```

### 3B. Compile & Simulate Primitive Engines

```
⬜ 3B.1  Chạy compile_all.do → fix compilation errors
         Potential issues: cross-package type compatibility, syntax

⬜ 3B.2  Simulate tb_conv3x3_golden → verify P0 tính đúng
         Expected: Test 1 (minimal) PASS exact
                   Test 2 (L0-style) ≥99.9% match
                   Test 3 (large Cin) PASS
                   Test 4 (random stress) ≥97% match

⬜ 3B.3  Simulate tb_conv1x1_golden → verify P1 tính đúng
         Expected: ≥99% match cho QC2f cv1/cv2 style

⬜ 3B.4  Simulate tb_dwconv3x3_golden → verify P2 tính đúng
         Expected: ≥99.9% match cho SCDown style

⬜ 3B.5  Simulate tb_maxpool5x5_golden → verify P3 tính đúng
         Expected: 100% match (pure comparison, no arithmetic)
         ⚠️ Test cascade: P1=MP(X), P2=MP(P1), P3=MP(P2)

⬜ 3B.6  Viết tb_dwconv7x7_golden.sv → verify P8 multipass
         Test: 3-pass accumulation, PSUM hold between passes
```

### 3C. Primitive Engines Còn Thiếu

```
⬜ 3C.1  concat_engine.sv — P5 CONCAT với domain alignment
         Input: 2 tensors + scale/zp mỗi nhánh + output scale/zp
         Logic: requant_to_common → interleave channels
         Test: L12 golden vectors → expect 100% match

⬜ 3C.2  upsample_engine.sv — P6 UPSAMPLE_NEAREST 2×
         Logic: address remap (mỗi pixel → 4 pixels)
         Test: L11 golden vectors → expect 100% match

⬜ 3C.3  ewise_add_engine.sv — P7 EWISE_ADD
         Logic: dequant A + dequant B → requant output
         ⚠️ PHẢI domain align trước khi add
         Test: QC2fCIB residual → expect 100% match
```

**Action cần làm**: Compile → simulate → fix → iterate cho đến ALL PASS.

---

## STAGE 4: RTL MEMORY & DATA MOVEMENT (PHASE_3 modules, cần verify)

### 4A. Existing PHASE_3 Memory Modules

```
✅ 4A.1  glb_input_bank.sv (47 dòng) + tb (142 dòng)
✅ 4A.2  glb_weight_bank.sv (72 dòng) + tb (180 dòng)
✅ 4A.3  glb_output_bank.sv (53 dòng) + tb (210 dòng)
✅ 4A.4  metadata_ram.sv (68 dòng) + tb (235 dòng)
✅ 4A.5  addr_gen_input.sv (75 dòng) + tb (198 dòng)
✅ 4A.6  addr_gen_weight.sv (75 dòng) + tb (95 dòng)
✅ 4A.7  addr_gen_output.sv (46 dòng) + tb (88 dòng)
```

### 4B. Existing Data Movement Modules

```
✅ 4B.1  window_gen.sv (56 dòng) + tb (196 dòng)
✅ 4B.2  router_cluster.sv (81 dòng) + tb (98 dòng)
✅ 4B.3  swizzle_engine.sv (179 dòng) + tb (81 dòng)
```

### 4C. Verify Memory + Data Movement

```
⬜ 4C.1  Chạy tất cả memory TB → verify read/write, banking, address
⬜ 4C.2  Chạy window_gen TB → verify K=1,3,5,7 tap generation
⬜ 4C.3  Chạy router_cluster TB → verify RIN/RWT/RPS routing
⬜ 4C.4  Chạy swizzle_engine TB → verify upsample 2× + concat

⬜ 4C.5  Integration test: addr_gen + glb_bank
         Feed realistic address pattern → verify no bank conflict
         Test: L0 input banking (h mod 3), L0 output banking (out_row mod 4)

⬜ 4C.6  Integration test: window_gen + pe_unit
         Feed K=3 rows → verify 3×3 sliding window taps correct
```

---

## STAGE 5: SUBCLUSTER — 1 PHẦN CỨNG DÙNG CHUNG, CONFIG QUA DESCRIPTOR

> **NGUYÊN TẮC CỐT LÕI**: KHÔNG tạo module riêng cho mỗi layer/block.
> 1 subcluster = 1 compute engine CỐ ĐỊNH.
> tile_fsm + shadow_reg_file đọc descriptor → config datapath → chạy BẤT KỲ primitive nào.

### 5A. Kiến trúc 1 Subcluster (phần cứng CỐ ĐỊNH, config bằng descriptor)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    1 SUBCLUSTER (phần cứng CỐ ĐỊNH)                 │
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────┐    ┌───┐ │
│  │ GLB_IN   │───►│Window_Gen│───►│PE_Cluster│───►│ PPU │───►│OUT│ │
│  │ (3 bank) │    │(K=1,3,5,7)   │(3×4×32)  │    │     │    │   │ │
│  └──────────┘    └──────────┘    └──────────┘    └─────┘    └───┘ │
│       ↑               ↑              ↑              ↑          ↑   │
│  ┌──────────┐    ┌──────────┐   ┌────────┐   ┌──────────┐         │
│  │ GLB_WT   │    │  Router  │   │Comp_Tree│  │ Swizzle  │         │
│  │ (3 bank) │    │ Cluster  │   │(MaxPool)│  │ Engine   │         │
│  └──────────┘    └──────────┘   └────────┘   └──────────┘         │
│       ↑               ↑              ↑              ↑              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              tile_fsm + shadow_reg_file                      │   │
│  │         (đọc descriptor → config MỌI THỨ ở trên)            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       ↑                                                            │
│  ┌──────────┐                                                      │
│  │Descriptor│  ← từ desc_fetch_engine (LAYER_DESC + TILE_DESC)     │
│  └──────────┘                                                      │
└─────────────────────────────────────────────────────────────────────┘

Descriptor chỉ thay đổi config, KHÔNG thay đổi phần cứng:
  L0:  mode=PE_RS3, K=3, stride=2, Cin=3,  Cout=16,  act=ReLU
  L2:  mode=PE_OS1, K=1, stride=1, Cin=32, Cout=64,  act=ReLU  (QC2f cv1)
       mode=PE_RS3, K=3, stride=1, Cin=16, Cout=16,  act=ReLU  (QC2f bottleneck)
       ... (cùng phần cứng, chỉ đổi descriptor)
  L5:  mode=PE_DW3, K=3, stride=2, C=128, act=ReLU   (SCDown cv2)
  L9:  mode=PE_MP5, K=5, no PPU                        (SPPF maxpool)
  L11: mode=PE_PASS, swizzle=UPSAMPLE_2X              (Upsample)
  L22: mode=PE_DW7, K=7, 3-pass, C=128, act=ReLU     (QC2fCIB DW7×7)
```

### 5B. Build Subcluster Compute Datapath

```
⬜ 5B.1  Wire up TRONG subcluster_wrapper.sv (đã có 777 dòng, behavioral):
         Thay behavioral conv bằng actual RTL modules:
           glb_input_bank ×3
           glb_weight_bank ×3
           glb_output_bank ×4
           addr_gen_input + addr_gen_weight + addr_gen_output
           window_gen (configurable K=1,3,5,7)
           pe_cluster (3×4 PE array + column_reduce + comparator_tree)
           ppu (bias + requant + ReLU + clamp)
           router_cluster (RIN/RWT/RPS)
           swizzle_engine (upsample, concat, move)
           metadata_ram

⬜ 5B.2  tile_fsm đọc descriptor → set:
           pe_mode    ← PE_RS3 / PE_OS1 / PE_DW3 / PE_DW7 / PE_MP5 / PE_PASS
           window_K   ← 1 / 3 / 5 / 7
           stride     ← 1 / 2
           act_mode   ← ReLU / None
           cin, cout, spatial dims
           num_cin_pass, num_k_pass (for multipass)
           swizzle_mode ← NORMAL / UPSAMPLE / CONCAT

⬜ 5B.3  shadow_reg_file latch descriptor fields → stable config cho datapath
```

### 5C. Verify Subcluster Chạy Từng Primitive Mode

```
⬜ 5C.1  Test mode PE_RS3 (Conv 3×3):
         Load L0 descriptor + weights + input vào GLB
         tile_fsm sequence: PREFILL_WT → PREFILL_IN → RUN_COMPUTE → PPU → DONE
         Compare output vs golden L0 → ≥99.9%
         ★ CÙNG phần cứng, chỉ config mode=PE_RS3

⬜ 5C.2  Test mode PE_OS1 (Conv 1×1):
         Load QC2f-cv1 descriptor → CÚ ĐỐI phần cứng
         tile_fsm sequence tương tự, chỉ khác K=1, broadcast weight
         Compare vs golden → ≥99%

⬜ 5C.3  Test mode PE_DW3 (DW Conv 3×3):
         Load SCDown-cv2 descriptor → CÙNG phần cứng
         Per-channel weight, bias, requant
         Compare vs golden → ≥99.9%

⬜ 5C.4  Test mode PE_MP5 (MaxPool 5×5):
         Load SPPF-pool descriptor → CÙNG phần cứng
         comparator_tree active, PE bypassed, no PPU
         Compare vs golden → ≥99.9%

⬜ 5C.5  Test mode PE_PASS + UPSAMPLE:
         Load L11 descriptor → swizzle_engine active
         Address remap 2×, compute bypassed
         Compare vs golden → 100%

⬜ 5C.6  Test mode PE_PASS + CONCAT:
         Load L12 descriptor → router concat mode
         Domain alignment nếu scale khác
         Compare vs golden → 100%

⬜ 5C.7  Test mode PE_DW7 multipass:
         Load L22-DW7 descriptor → CÙNG phần cứng
         3 passes: kh=0-2 → kh=3-5 → kh=6
         PSUM namespace hold giữa passes
         Compare vs golden → ≥99.9%
```

---

## STAGE 6: VERIFY LAYER SEQUENCES (Chuỗi descriptors trên CÙNG phần cứng)

> **Ý nghĩa**: 1 block (VD: QC2f) = CHUỖI 6 descriptors gửi cho CÙNG 1 subcluster.
> tile_fsm xử lý tuần tự: desc_1 → done → desc_2 → done → ... → desc_6 → block done.
> KHÔNG CẦN tạo module riêng cho QC2f, SCDown, SPPF, v.v.

### 6A. Verify Conv Block Sequence (L0, L1, L3, L17)

```
⬜ 6A.1  Gửi 1 descriptor (PE_RS3, stride=2, act=ReLU) cho subcluster
         L0: [3,640,640] → [16,320,320]
         → Subcluster chạy xong → compare output vs golden → ≥99.99%

⬜ 6A.2  Gửi 1 descriptor khác (Cin=16, Cout=32) cho CÙNG subcluster
         L1: [16,320,320] → [32,160,160]
         → CÙNG phần cứng, chỉ config khác → ≥99.96%
```

### 6B. Verify QC2f Sequence (L2, L4, L6, L8, L13, L16, L19)

```
⬜ 6B.1  Gửi CHUỖI 6 descriptors cho CÙNG subcluster:
         Desc 1: PE_OS1, Cin=32, Cout=64 (cv1)
         Desc 2: PE_RS3, Cin=16, Cout=16 (bottleneck cv1)
         Desc 3: PE_RS3, Cin=16, Cout=16 (bottleneck cv2)
         Desc 4: PE_PASS + CONCAT (merge channels)
         Desc 5: PE_OS1, Cin=64, Cout=32 (cv2)

         Intermediate results chain: output desc_1 = input desc_2, etc.
         GLB output of one step = GLB input of next step (swizzle_engine)
         Final output vs golden L2 → ≥99%

⬜ 6B.2  Test L4, L6, L8 (backbone QC2f — cùng pattern, khác Cin/Cout)
⬜ 6B.3  Test L13, L16, L19 (neck QC2f — Cin lớn hơn do concat input)
```

### 6C. Verify SCDown Sequence (L5, L7, L20)

```
⬜ 6C.1  Gửi 2 descriptors tuần tự cho CÙNG subcluster:
         Desc 1: PE_OS1, Cin=64, Cout=128 (cv1, channel expand)
         Desc 2: PE_DW3, C=128, stride=2 (cv2, spatial downsample)
         Final output vs golden L5 → ≥99.9%
```

### 6D. Verify SPPF Sequence (L9)

```
⬜ 6D.1  Gửi CHUỖI 6 descriptors:
         Desc 1: PE_OS1, Cin=256, Cout=128 (cv1)
         Desc 2: PE_MP5, C=128 (pool1) — output lưu GLB slot A
         Desc 3: PE_MP5, C=128 (pool2) — input=pool1, output lưu slot B
         Desc 4: PE_MP5, C=128 (pool3) — input=pool2, output lưu slot C
         Desc 5: PE_PASS + CONCAT 4-way (cv1+pool1+pool2+pool3 → 512ch)
         Desc 6: PE_OS1, Cin=512, Cout=256 (cv2)

         ⚠️ Cần metadata_ram quản lý 4 GLB slots đồng thời
         Final output vs golden L9 → ≥99.9%
```

### 6E. Verify QConcat Sequence (L12, L15, L18, L21)

```
⬜ 6E.1  L12: Gửi 1 descriptor PE_PASS + CONCAT:
         Input A = L11 output (đã trong GLB)
         Input B = F6 skip (đã HOLD trong GLB từ L6)
         ⚠️ barrier_manager: wait(L11_done AND F6_ready)
         Output vs golden L12 → 100%

⬜ 6E.2  L15, L18, L21 tương tự (khác source skip tensor)
```

### 6F. Verify Upsample (L11, L14)

```
⬜ 6F.1  L11: Gửi 1 descriptor PE_PASS + UPSAMPLE_2X
         swizzle_engine address remap: mỗi pixel → 4 pixels
         Output vs golden L11 → 100%
```

### 6G. Verify QC2fCIB Sequence (L22)

```
⬜ 6G.1  Gửi CHUỖI ~10 descriptors:
         Desc 1: PE_OS1 (cv1, 384→256)
         Desc 2-3: PE_DW3 + PE_OS1 (QCIB internal)
         Desc 4: PE_DW7, 3-pass multipass (DW 7×7)
         Desc 5-6: PE_OS1 + PE_DW3 (QCIB internal)
         Desc 7: EWISE_ADD (residual shortcut)
         Desc 8: CONCAT (identity + processed)
         Desc 9: PE_OS1 (cv2)

         ⚠️ DW_7x7: tile_fsm tự loop 3 passes (num_k_pass=3)
         PSUM namespace hold giữa passes, PPU chỉ ở pass cuối
         Final output vs golden L22 → ≥99.9%
```

---

## STAGE 7: RTL SYSTEM INTEGRATION (PHASE_3 system modules)

### 7A. Existing PHASE_3 Control & System Modules

```
✅ 7A.1  tile_fsm.sv (276 dòng) + tb (154 dòng)
✅ 7A.2  shadow_reg_file.sv (81 dòng) + tb (82 dòng)
✅ 7A.3  barrier_manager.sv (33 dòng) + tb (123 dòng)
✅ 7A.4  local_arbiter.sv (148 dòng) + tb (102 dòng)
✅ 7A.5  desc_fetch_engine.sv (255 dòng) + tb (91 dòng)
✅ 7A.6  global_scheduler.sv (174 dòng) + tb (90 dòng)
✅ 7A.7  tensor_dma.sv (195 dòng) + tb (143 dòng)
✅ 7A.8  pe_cluster.sv (106 dòng) + tb (101 dòng)
✅ 7A.9  subcluster_wrapper.sv (777 dòng)
✅ 7A.10 supercluster_wrapper.sv (450 dòng)
✅ 7A.11 controller_system.sv (254 dòng)
✅ 7A.12 accel_top.sv (434 dòng) + tb (156 dòng)
✅ 7A.13 tb_accel_e2e.sv (114 dòng) — E2E testbench stub
```

### 7B. Subcluster Integration

```
⬜ 7B.1  Verify tile_fsm trạng thái chuyển đúng cho mỗi primitive type
         FSM: IDLE→LOAD_CFG→PREFILL_WT→PREFILL_IN→RUN→ACCUM→PPU→SWIZZLE→DONE

⬜ 7B.2  Thay behavioral conv trong subcluster_wrapper bằng
         actual pe_cluster + window_gen + ppu pipeline
         ⚠️ subcluster_wrapper.sv (777 dòng) hiện dùng behavioral code

⬜ 7B.3  Test subcluster chạy 1 tile L0 → verify output

⬜ 7B.4  Test subcluster chạy 1 tile L2 (QC2f) → verify primitive chaining

⬜ 7B.5  Test subcluster chạy multi-tile L0 → verify tiling correct
```

### 7C. SuperCluster & System Top

```
⬜ 7C.1  Verify local_arbiter: Dual-RUNNING rotation
         2 sub compute + 1 fill + 1 drain → rotate

⬜ 7C.2  Verify tensor_dma: AXI4 burst read/write DDR ↔ GLB

⬜ 7C.3  Verify barrier_manager: 4 barriers (L12, L15, L18, L21)
         Signal/wait protocol correct

⬜ 7C.4  Verify global_scheduler: sc_mask routing to 4 SCs

⬜ 7C.5  Verify desc_fetch_engine: Parse NET→LAYER→TILE descriptors

⬜ 7C.6  Adjust for V2-lite (VC707): SUBS_PER_SC = 3 (thay vì 4)
         Modify supercluster_wrapper.sv
```

---

## STAGE 8: RTL FULL INFERENCE L0-L22

### 8A. Descriptor Generation

```
⬜ 8A.1  Viết Python script: generate_descriptors.py
         Input: model weights + quant params
         Output: binary descriptor blob (NET_DESC + 23 LAYER_DESC + N TILE_DESC)

⬜ 8A.2  Viết Python script: generate_weight_blob.py
         Pack all weights + bias + M_int + shift vào 1 binary DDR image

⬜ 8A.3  Generate .hex files cho simulation DDR model
```

### 8B. End-to-End Simulation

```
⬜ 8B.1  Populate tb_accel_e2e.sv:
         - Load descriptors + weights + input image vào DDR model
         - Start accelerator via AXI-Lite CSR write
         - Wait IRQ (inference complete)
         - Read P3, P4, P5 từ DDR model

⬜ 8B.2  Test 1 image: compare P3/P4/P5 vs golden
         Target: ≥97% bit-exact per output tensor

⬜ 8B.3  Test 10 images: verify consistency

⬜ 8B.4  Test skip connections:
         F4(L4→L15), F6(L6→L12), F8(L8→L21), F13(L13→L18)
         Verify barrier sync timing correct
```

### 8C. Final Sign-Off

```
⬜ 8C.1  Per-layer match table: RTL output vs golden cho mỗi L0-L22
         (target: match documented baseline từ Stage 1D)

⬜ 8C.2  P3/P4/P5 output: ≥97% bit-exact vs golden oracle

⬜ 8C.3  Feed RTL P3/P4/P5 vào CPU Qv10Detect head
         → mAP50 ≥ 0.92 (< 1% degradation)

⬜ 8C.4  (Optional) Synthesize cho VC707 XC7VX485T
         → Verify resource usage ≤ budget
         → Verify timing closure @ 200 MHz
```

---

## STAGE 9: FPGA DEPLOYMENT (VC707 Board)

```
⬜ 9.1   Synthesize accel_top trên Vivado cho XC7VX485T
⬜ 9.2   Implement (Place & Route) → timing report
⬜ 9.3   Generate bitstream
⬜ 9.4   Program VC707 board
⬜ 9.5   CPU driver: load descriptors + weights qua AXI-Lite
⬜ 9.6   Run inference on real hardware
⬜ 9.7   Compare output vs golden → FINAL VALIDATION
```

---

## TỔNG KẾT INVENTORY

```
╔══════════════════════════════════════════════════════════════════════╗
║                    TRẠNG THÁI TỔNG THỂ DỰ ÁN                       ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  STAGE 0: Nghiên cứu & Đặc tả        ████████████████████ 100% ✅  ║
║  STAGE 1: Golden Python & SW Verify   ████████████████████ 100% ✅  ║
║  STAGE 2: RTL Compute Atoms           ██████████████░░░░░░  70% 🔧  ║
║           (có code, cần chạy verify)                                 ║
║  STAGE 3: RTL Primitive Engines       ██████████░░░░░░░░░░  50% 🔧  ║
║           (có 5 engines + 4 TB, cần compile & test)                  ║
║  STAGE 4: RTL Memory & Data Movement  ██████████████░░░░░░  70% 🔧  ║
║           (có modules + TB, cần verify)                              ║
║  STAGE 5: Subcluster (1 HW chung,     ░░░░░░░░░░░░░░░░░░░░   0% ⬜  ║
║           config qua descriptor)                                     ║
║  STAGE 6: Layer Sequences (chuỗi      ░░░░░░░░░░░░░░░░░░░░   0% ⬜  ║
║           descriptors trên cùng HW)                                  ║
║  STAGE 7: System Integration          ██████░░░░░░░░░░░░░░  30% 🔧  ║
║           (có system modules, cần adapt cho V2-lite + wire up)       ║
║  STAGE 8: Full Inference E2E          ░░░░░░░░░░░░░░░░░░░░   0% ⬜  ║
║  STAGE 9: FPGA Deployment             ░░░░░░░░░░░░░░░░░░░░   0% ⬜  ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  TỔNG FILE ĐÃ CÓ:                                                   ║
║    RTL (.sv):        64 files, ~13,700 dòng (PHASE_3 + HW_ACCEL)    ║
║    Python golden:    ~40 files (primitives + tests + verify)         ║
║    Documentation:    ~45 files markdown (specs + reports + mapping)  ║
║    Verify scripts:   23 files (verify_layer_0..22.py)               ║
║                                                                      ║
║  TIẾP THEO (ưu tiên):                                               ║
║    → Stage 2B: Chạy verify compute atoms (TB đã có)                 ║
║    → Stage 3B: Compile & simulate primitive engines                  ║
║    → Stage 3C: Viết 3 engines còn thiếu (concat, upsample, add)     ║
║    → Stage 5:  Ghép atoms thành integrated primitives                ║
║    → Stage 6:  Ghép primitives thành layer blocks                    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

### NEXT ACTIONS (Thứ tự ưu tiên):

```
NGAY BÂY GIỜ:
  1. ⬜ Chạy compile_all.do → fix tất cả compilation errors
  2. ⬜ Chạy từng TB compute atom → confirm 100% PASS
  3. ⬜ Chạy từng TB primitive engine → confirm golden match

SAU KHI PRIMITIVES PASS:
  4. ⬜ Wire up subcluster_wrapper.sv — thay behavioral bằng actual RTL
  5. ⬜ Test subcluster chạy từng primitive mode (Stage 5C)
  6. ⬜ Test với real layer golden vectors

SAU KHI SUBCLUSTER PASS TẤT CẢ MODES:
  7. ⬜ Test chuỗi descriptors cho block sequences (Stage 6)
       (QC2f = 6 desc, SCDown = 2 desc, SPPF = 6 desc, v.v.)
  8. ⬜ Adapt system modules cho V2-lite VC707
  9. ⬜ E2E inference L0-L22 test

CUỐI CÙNG:
  10. ⬜ FPGA synthesis + board test
```
