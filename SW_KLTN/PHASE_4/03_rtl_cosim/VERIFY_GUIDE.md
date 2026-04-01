# Layer-by-Layer Verification Guide

## Tổng quan

Verification flow gồm 2 tầng:

| Tầng | Mục đích | Tool |
|------|----------|------|
| **Tầng 1: Python verify** | Xác nhận golden data bit-exact bằng pure integer math | `verify_conv_layer.py` |
| **Tầng 2: RTL PPU verify** | Xác nhận RTL PPU module (requant + SiLU + zp) đúng | `tb_ppu_golden.sv` |

## Thứ tự thực hiện

```
Conv layers:   L00 → L01 → L03 → L17
Complex blocks: QC2f → SCDown → QConcat → Upsample
Full system:    accel_top end-to-end
```

---

## Bước 1: Python Verification (đã pass)

Từ thư mục `PHASE_4/`:

```bash
python 03_rtl_cosim/verify_conv_layer.py --layer 0 --rows 2
python 03_rtl_cosim/verify_conv_layer.py --layer 1 --rows 2
python 03_rtl_cosim/verify_conv_layer.py --layer 3 --rows 2
python 03_rtl_cosim/verify_conv_layer.py --layer 17 --rows 2
```

Kết quả đã xác nhận: L00/L01/L03 = 100%, L17 = 99.98% (1 pixel off-by-1).

## Bước 2: Export RTL Parameters (đã chạy)

```bash
python 03_rtl_cosim/export_rtl_params.py --all
```

Tạo ra `rtl_test_L{00,01,03,17}/` chứa:
- `m_int.hex` — requant multiplier per channel
- `shift.hex` — requant shift per channel
- `silu_lut.hex` — SiLU LUT 256 entries (signed int8)
- `bias.hex` — bias per channel (int32)
- `zp_out.hex` — output zero point
- `psum_posN.hex` — pre-computed INT32 psum (4 positions)
- `golden_posN.hex` — expected uint8 output

## Bước 3: RTL PPU Simulation (Vivado)

### 3.1 Mở Vivado Tcl console

```
cd E:/KLTN_HMH_FINAL/SW_KLTN
```

### 3.2 Compile & run

```tcl
source PHASE_4/03_rtl_cosim/run_ppu_golden.tcl

# Chạy từng layer
run_ppu_test 0
run_ppu_test 1
run_ppu_test 3
run_ppu_test 17
```

### 3.3 Kết quả mong đợi

```
================================================================
  [PASS] Layer 0 PPU: ALL 64 values bit-exact
================================================================
```

Nếu FAIL: kiểm tra:
1. m_int/shift có đúng không (so sánh hex vs Python)
2. SiLU LUT signed/unsigned mapping
3. Pipeline timing (act_valid latency)

## Bước 4: Complex blocks + full model (Python)

Xem chi tiết: **`03_rtl_cosim/COMPLEX_AND_FULL_MODEL.md`**

```bat
cd PHASE_4
run_phase4_verify.bat
```

Hoặc lẻ:

```bash
python 03_rtl_cosim/verify_complex_blocks.py --upsample --qconcat   # 100% numpy
python 03_rtl_cosim/verify_complex_blocks.py --torch-blocks         # cần dill + PyTorch
python 03_rtl_cosim/verify_complex_blocks.py --all
python 03_rtl_cosim/verify_full_model_outputs.py --golden-dir 02_golden_data
```

- **Upsample / QConcat**: đã verify bit-exact với golden hex (numpy).
- **QC2f / SCDown / SPPF / QPSA**: so PyTorch submodule vs hex (`--torch-blocks`).
- **P3/P4/P5**: `verify_full_model_outputs.py` (cùng trace như export).

## Bước 5: Full System (RTL)

Sau khi `accel_top` thật sự chạy end-to-end:

- Chạy `tb_golden_check.sv` — đã chỉnh **P3/P4/P5 shapes** (128/256/512 ch) và đường dẫn golden tuyệt đối.
- Kiểm tra **layout DDR** của IP có khớp `OUTPUT_BASE`, `OUTPUT_BASE+P3_BYTES`, … hay không; nếu khác, sửa offset trong TB.

Các block Upsample/QConcat trên FPGA thường là **data movement** (DMA + layout); verify chức năng bằng Python trước, RTL map bộ nhớ sau.

---

## File Layout

```
PHASE_4/03_rtl_cosim/
├── verify_conv_layer.py      ← Python bit-exact verify
├── export_rtl_params.py      ← Export hex for Verilog
├── tb_ppu_golden.sv          ← PPU RTL testbench
├── run_ppu_golden.tcl        ← Vivado compile+run script
├── VERIFY_GUIDE.md           ← Tài liệu này
├── rtl_test_L00/             ← Hex data cho Layer 0
│   ├── m_int.hex
│   ├── shift.hex
│   ├── silu_lut.hex
│   ├── bias.hex
│   ├── zp_out.hex
│   ├── psum_pos{0-3}.hex
│   └── golden_pos{0-3}.hex
├── rtl_test_L01/             ← Hex data cho Layer 1
├── rtl_test_L03/             ← Hex data cho Layer 3
└── rtl_test_L17/             ← Hex data cho Layer 17
```

## Quan trọng: Padding Convention

RTL accelerator PHẢI pad bằng `zero_point` (không phải 0):
- Khi pixel nằm ngoài input tensor, giá trị = `inp_zp`
- `act_val = inp_zp - inp_zp = 0` → MAC contribution = 0
- Đây là convention của PyTorch quantized conv2d
- Nếu RTL pad bằng 0 thay vì zp: sẽ sai cho layers có `zp != 0` (L01, L03, L17)
