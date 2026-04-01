# PHASE 4 — Software-Hardware Co-Simulation
## YOLOv10n INT8 Accelerator: Golden Data Export + RTL Verification + CPU Postprocessing

> **Mục tiêu**: Kiểm chứng rằng PHASE 3 RTL (IP accelerator) cho kết quả **bit-exact**
> với model quantized Python, TRƯỚC KHI nhúng vào board Virtex-7.

---

## 1. TỔNG QUAN FLOW

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 4: CO-SIMULATION FLOW                          │
│                                                                             │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐    │
│  │  01_EXPORT        │     │  03_RTL_COSIM     │     │  04_POSTPROCESS  │    │
│  │  (Python)         │     │  (Vivado xsim)    │     │  (Python)        │    │
│  │                   │     │                   │     │                  │    │
│  │  Load PTQ model   │     │  tb reads hex     │     │  Read P3/P4/P5   │    │
│  │  Preprocess img   │     │  files from       │     │  hex from RTL    │    │
│  │  QuantStub→INT8   │     │  02_golden_data/  │     │  → Dequantize    │    │
│  │  Hook every layer │────▶│  Feed to accel_top│────▶│  → NMS           │    │
│  │  Export weights   │     │  Compare output   │     │  → Scale boxes   │    │
│  │  Export golden    │     │  with golden hex  │     │  → Draw bbox     │    │
│  │  P3/P4/P5        │     │  → BIT-EXACT?     │     │  → Save image    │    │
│  └──────────────────┘     └──────────────────┘     └──────────────────┘    │
│           │                         │                        │              │
│           ▼                         ▼                        ▼              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     02_GOLDEN_DATA (hex files)                       │   │
│  │                                                                      │   │
│  │  input_act.hex        — INT8 input activation (1,3,640,640)          │   │
│  │  weights_L{i}.hex     — INT8 weights per layer                       │   │
│  │  bias_L{i}.hex        — INT32 bias per layer                         │   │
│  │  quant_params.json    — scale, zero_point per layer                  │   │
│  │  silu_lut.hex         — 256-entry SiLU lookup table                  │   │
│  │  golden_P3.hex        — P3 [1,128,80,80]  (quint8)                   │   │
│  │  golden_P4.hex        — P4 [1,256,40,40]                             │   │
│  │  golden_P5.hex        — P5 [1,512,20,20]  (quint8, khớp golden_outputs)│   │
│  │  desc_net.hex / desc_layers.hex / desc_tiles.hex — bảng mô tả cho DDR   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. CHẠY TỪNG BƯỚC

### Bước 0 (khuyến nghị): Verify phần mềm — block phức tạp + P3/P4/P5

Sau khi export, chạy:

```bat
run_phase4_verify.bat
```

Hoặc đọc `03_rtl_cosim/COMPLEX_AND_FULL_MODEL.md` (Upsample, QConcat, QC2f/SCDown/…, full head).

### Bước 1: Export Golden Data (Python)
```bash
cd PHASE_4/01_export
python export_golden_data.py --image ../../img1.jpg --output ../02_golden_data/
```

Hoặc chạy một lần: `PHASE_4\run_phase4.bat` (gồm 1a export chính, 1b layer-by-layer, **1c `generate_descriptors.py`**).

**Kết quả**:
- `input_act.hex` — INT8 activation sau QuantStub, format `$readmemh`
- `weights_L{i}.hex` — INT8 weights cho từng layer
- `bias_L{i}.hex` — INT32 bias
- `quant_params.json` — {layer_id: {scale, zero_point, weight_scale, weight_zp}}
- `silu_lut.hex` — 256-entry precomputed SiLU LUT
- `golden_P3.hex`, `golden_P4.hex`, `golden_P5.hex` — Golden outputs (tensor đầu vào **Detect**; kích thước C/H/W lấy từ model — xem `quant_params.json` → `golden_outputs`)
- `golden_outputs.json` — snapshot kích thước / scale / zp cho P3/P4/P5
- `desc_net.hex`, `desc_layers.hex`, `desc_tiles.hex` — sau **`generate_descriptors.py`** (bắt buộc cho `tb_golden_check` / `desc_fetch_engine`)

### Bước 1b: Bước 4A — Kiểm tra hệ thống **chỉ phần mềm** (khuyến nghị trước RTL)

Xác nhận đủ file golden, byte count khớp `golden_outputs`, descriptor có mặt:

```bat
cd PHASE_4
python 03_rtl_cosim/verify_system_readiness.py
REM Siết descriptor (fail nếu thiếu desc_*.hex):
python 03_rtl_cosim/verify_system_readiness.py --strict-descriptors
```

Chi tiết: `03_rtl_cosim/SYSTEM_VERIFY_STEP4.md` (4A = phần mềm, 4B = Vivado + `compare_golden_vs_rtl`).

### Bước 2: RTL Co-Simulation (Vivado) — Bước 4B
```tcl
# Trong Vivado Tcl console:
source PHASE_4/03_rtl_cosim/run_cosim.tcl
```

**Testbench `tb_golden_check.sv`** (đường dẫn file **absolute** trong TB để XSIM ổn định):
1. Load `input_act.hex`, weights mẫu, **`desc_net/layers/tiles.hex`** vào DDR model (địa chỉ khớp `generate_descriptors.py` `memory_map`)
2. Load golden P3/P4/P5 vào vùng DDR riêng (chỉ để tham chiếu / debug)
3. Ghi CSR (`NET_DESC_BASE`, layer range, `START`)
4. Chờ IRQ / timeout — **cần RTL báo done** (`controller_system` + fetch/scheduler); xem `03_rtl_cosim/COSIM_DEEP_DIVE_AND_FIXES.md`
5. So sánh vùng **OUTPUT** (P3→P5 nối tiếp) với `golden_P3/P4/P5.hex`

**Hạn chế hiện tại**: TB mới nạp **`weights_L0_conv.hex`** làm ví dụ; full 23 layer cần nạp đủ weight/bias theo `weight_arena_base` + descriptor hoặc mở rộng TB.

### Bước 3: CPU Postprocessing (Python)

```bash
cd PHASE_4/04_postprocess
python cpu_postprocess.py --hex_dir ../02_golden_data --image ../../img1.jpg --output output_rtl.jpg
# Bbox minh họa (PyTorch): thêm --boxes-from-model
```

Xem `INFERENCE_P3P4P5_AND_BOXES.md` (P3/P4/P5 là feature map trước head).

---

## 3. FORMAT HEX FILES

### 3.1 Activation / Weight hex format
```
# Mỗi dòng = 32 bytes = 256 bits (1 AXI beat)
# Giá trị INT8 signed → 2's complement hex (00-FF)
# Byte order: lane[0] ở LSB, lane[31] ở MSB
# Ví dụ: 32 giá trị INT8 = 64 hex chars
FF01027F80...  (lane[0]=0xFF=-1, lane[1]=0x01=1, ...)
```

### 3.2 Bias hex format
```
# Mỗi dòng = 32 x INT32 = 128 bytes = 1024 bits
# 4 hex chars per INT32 value, little-endian
```

### 3.3 Quant params JSON
```json
{
  "input": {"scale": 0.003921, "zero_point": 0},
  "layers": [
    {"id": 0, "name": "conv0", "act_scale": 0.3527, "act_zp": 62,
     "weight_scale": 0.0234, "weight_zp": 0,
     "bias_scale": 0.0000918, "output_scale": 0.3527, "output_zp": 62},
    ...
  ]
}
```

---

## 4. VERIFICATION CHECKLIST

```
☐ export_golden_data.py chạy thành công, tạo đủ hex files
☐ Hex files load được trong Vivado ($readmemh không lỗi)
☐ Layer 0 (Conv3x3 s=2): RTL output == golden_L0.hex
☐ Layer 1–8 (backbone): progressive layer-by-layer match
☐ verify_system_readiness.py PASS (4A)
☐ P3 [1,128,80,80]: RTL dump == golden_P3.hex (bit-exact)
☐ P4 [1,256,40,40]: RTL == golden_P4.hex (bit-exact)
☐ P5 [1,512,20,20]: RTL == golden_P5.hex (bit-exact)
☐ cpu_postprocess.py: bounding boxes match output_quant.jpg
☐ NMS results: same objects detected, same confidence scores
```

---

## 5. MAPPING: Python Layer → RTL Module → Tile Descriptor

| Python Layer | Type | RTL pe_mode | Cin→Cout | H×W |
|---|---|---|---|---|
| L0 | Conv(s2)+SiLU | PE_RS3 | 3→16 | 640→320 |
| L1 | Conv(s2)+SiLU | PE_RS3 | 16→32 | 320→160 |
| L2 | C2f | PE_OS1+PE_RS3+CAT+PE_OS1 | 32→32 | 160×160 |
| L3 | Conv(s2)+SiLU | PE_RS3 | 32→64 | 160→80 |
| L4 | C2f | PE_OS1+PE_RS3+CAT+PE_OS1 | 64→64 | 80×80 ★ |
| L5 | SCDown | PE_OS1+PE_DW3+... | 64→128 | 80→40 |
| L6 | C2f | ... | 128→128 | 40×40 ★ |
| L7 | SCDown | ... | 128→256 | 40→20 |
| L8 | C2f | ... | 256→256 | 20×20 ★ |
| L9 | SPPF | PE_OS1+PE_MP5+CAT+PE_OS1 | 256→256 | 20×20 |
| ... | ... | ... | ... | ... |

★ = skip connection output (barrier dependency)

### 5.1 P3 / P4 / P5 (golden hex) — **phải khớp `quant_params.json`**

Export tìm **ba tensor 4D** đầu vào khối Detect (thứ tự từ model). Với YOLOv10n đang dùng trong repo:

| Tên | Shape (NCHW) | Ghi chú |
|-----|----------------|--------|
| **P3** | **[1, 128, 80, 80]** | quint8, ~819200 B |
| **P4** | **[1, 256, 40, 40]** | ~409600 B |
| **P5** | **[1, 512, 20, 20]** | ~204800 B |

*(Trước đây tài liệu ghi nhầm 64/128/256 channels — **đã bỏ**; mọi script/TB dùng `golden_outputs` / `tb_golden_check` localparam.)*

**Layer index** tương ứng P3/P4/P5 **không cố định** trong guide — phụ thuộc trace (`export_common.find_detect_feature_inputs`). Khi đổi model, chạy lại export và kiểm tra `golden_outputs` trong `quant_params.json`.

---

## 6. CẤU TRÚC THƯ MỤC

```
PHASE_4/
├── PHASE_4_GUIDE.md                    ← Tài liệu này
├── run_phase4.bat                      ← Master script chạy tất cả
│
├── 01_export/                          ← Python scripts xuất golden data
│   ├── export_golden_data.py           ← Xuất input/weight/P3/P4/P5 hex
│   ├── export_layer_by_layer.py        ← Xuất TỪNG layer input/output hex
│   ├── generate_descriptors.py         ← Tạo NET/LAYER/TILE descriptors hex
│   └── compare_golden_vs_rtl.py        ← So sánh golden vs dump RTL (sau sim)
│
├── 02_golden_data/                     ← Thư mục chứa hex files (auto-generated)
│   ├── input_act.hex                   ← INT8 input activation
│   ├── weights_L0_conv.hex, ...        ← INT8 weights (một file / submodule quant)
│   ├── quant_params.json              ← Scale/ZP metadata
│   ├── silu_lut.hex                   ← SiLU lookup table
│   ├── golden_P3.hex                  ← P3 [1,128,80,80] (khớp golden_outputs)
│   ├── golden_P4.hex                  ← P4 [1,256,40,40]
│   ├── golden_P5.hex                  ← P5 [1,512,20,20]
│   ├── desc_net.hex                   ← NET descriptor
│   ├── desc_layers.hex                ← LAYER descriptors
│   ├── desc_tiles.hex                 ← TILE descriptors
│   └── layer_by_layer/               ← Per-layer I/O hex files
│       ├── act_L0_input.hex
│       ├── act_L0_output.hex
│       ├── weight_L0_conv.hex
│       ├── ...
│       └── layer_summary.json
│
├── 03_rtl_cosim/                      ← RTL testbenches + Vivado scripts
│   ├── tb_single_layer.sv            ← Layer 0 isolated verification
│   ├── tb_golden_check.sv            ← Full system co-simulation
│   └── run_cosim.tcl                 ← Vivado compile + elaborate script
│
└── 04_postprocess/                    ← CPU postprocessing
    ├── cpu_postprocess.py            ← Đọc P3/P4/P5 → NMS → bounding boxes
    └── verify_detection.py           ← So sánh detections Python vs RTL
```

---

## 7. CHIẾN LƯỢC DEBUG LAYER-BY-LAYER

Khi RTL output ≠ golden, debug theo thứ tự:

```
1. Layer 0 (Conv3x3 s2, Cin=3→Cout=16):
   - Input: act_L0_input.hex (từ QuantStub)
   - Weight: weight_L0_conv.hex
   - Expected: act_L0_output.hex
   → Nếu FAIL: kiểm tra pe_unit, dsp_pair, ppu
   → Nếu PASS: tiếp Layer 1

2. Layer 1 (Conv3x3 s2, 16→32):
   - Input: act_L1_input.hex (= act_L0_output)
   - Weight: weight_L1_conv.hex
   - Expected: act_L1_output.hex
   → Tương tự...

3. Layer 2 (C2f: bottleneck):
   - Phức tạp hơn: gồm 1x1 split + 3x3 conv + 1x1 merge
   - Kiểm tra từng sub-layer

...tiếp cho đến Layer 22
```

Tool so sánh:
```bash
python compare_golden_vs_rtl.py \
    --golden layer_by_layer/act_L0_output.hex \
    --rtl    rtl_sim_output/act_L0.hex \
    --name "Layer 0" --shape 1,16,320,320
```

---

*Mỗi bước PHẢI pass trước khi tiến sang bước tiếp theo.*
*Nếu RTL output ≠ golden → debug layer-by-layer cho đến khi bit-exact.*

---

## 8. ĐỒNG BỘ Ý TƯỞNG vs HIỆN TRẠNG (tóm tắt kiểm tra)

| Ý tưởng PHASE 4 | Trạng thái | Ghi chú |
|-----------------|------------|---------|
| Export Python → hex + JSON | **Đúng** | `export_golden_data.py`, `export_layer_by_layer.py`, `generate_descriptors.py` |
| 4A kiểm tra mặt bằng dữ liệu (không Vivado) | **Đúng** | `verify_system_readiness.py`; thêm `--strict-descriptors` nếu bắt buộc có `desc_*.hex` |
| 4B cosim `accel_top` so P3/P4/P5 | **Đúng hướng** | Cần RTL đủ (CSR đọc đúng, done/IRQ, DMA/SC ghi output) + TB nạp đủ weight/descriptor — xem `COSIM_DEEP_DIVE_AND_FIXES.md` |
| Postprocess CPU từ hex | **Đúng hướng** | `cpu_postprocess.py` lấy shape từ `golden_outputs`; decode NMS đơn giản có thể bỏ qua nếu C khác nhau giữa scale — dùng `--boxes-from-model` khi cần |
| So sánh golden vs RTL | **Đúng** | `01_export/compare_golden_vs_rtl.py` (sau khi có file dump từ sim) |

**Điểm cần tiếp tục hoàn thiện (không làm sai ý tưởng, nhưng chưa đủ cho bit-exact SoC):** nạp full weight arena trong TB hoặc trong mô hình DDR; tích hợp `tensor_dma` nếu output đi qua DMA; assert `inference_complete` đúng ngữ nghĩa “compute + ghi xong” thay vì chỉ “fetch descriptor xong”.
