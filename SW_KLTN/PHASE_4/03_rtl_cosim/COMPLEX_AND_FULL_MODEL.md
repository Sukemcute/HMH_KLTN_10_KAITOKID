# Block phức tạp + kiểm thử full model (PHASE 4)

## 1. Python — block phức tạp

### Upsample + QConcat (không cần load PyTorch model)

```bat
cd PHASE_4
python 03_rtl_cosim\verify_complex_blocks.py --upsample --qconcat
```

- **Upsample**: nearest-neighbor 2×, so khớp `act_L11_output.hex`, `act_L14_output.hex`.
- **QConcat**: dequant từng nhánh → concat theo kênh → requant theo `output.scale/zero_point` (khớp L12, L15, L18, L21).

### SCDown, QC2f, SPPF, QPSA (cần `load_quant_model`)

```bat
python 03_rtl_cosim\verify_complex_blocks.py --torch-blocks
```

Chạy submodule `model.model.model[layer_idx]` trên tensor quantized dựng lại từ hex, so `int_repr` với `act_Lxx_output.hex`.

**Yêu cầu:** môi trường giống lúc export (thường cần `pip install dill` nếu Ultralytics báo thiếu).

### Tất cả

```bat
python 03_rtl_cosim\verify_complex_blocks.py --all
```

---

## 2. Python — full model (P3 / P4 / P5)

So khớp tensor đầu vào detect head với `golden_P3.hex`, `golden_P4.hex`, `golden_P5.hex` (cùng logic `find_detect_feature_inputs` như `export_golden_data.py`).

```bat
python 03_rtl_cosim\verify_full_model_outputs.py --golden-dir 02_golden_data
```

Hoặc chỉ định ảnh:

```bat
python 03_rtl_cosim\verify_full_model_outputs.py --image E:/path/to/img1.jpg
```

---

## 3. Batch một lần

```bat
PHASE_4\run_phase4_verify.bat
```

---

## 4. RTL — full model (`tb_golden_check.sv`)

Đã cập nhật:

- Kích thước **P3 [1,128,80,80]**, **P4 [1,256,40,40]**, **P5 [1,512,20,20]** (số dòng hex 32 byte/dòng).
- Offset bộ nhớ giả định: P4 ngay sau P3, P5 ngay sau P4 (`OUTPUT_BASE`, `+P3_BYTES`, `+P3_BYTES+P4_BYTES`).
- Đường dẫn golden dùng **absolute path** `E:/KLTN_HMH_FINAL/...` — nếu project ở ổ khác, sửa `G_*` trong `tb_golden_check.sv`.

**Lưu ý:** `tb_golden_check` chỉ meaningful khi `accel_top` + scheduler + DMA thật sự ghi đúng vùng DDR và hoàn thành inference. Trước đó, dùng các bước Python ở trên + `tb_ppu_golden.sv`.

---

## 5. RTL tiếp theo (đề xuất)

| Block | Gợi ý RTL |
|--------|-----------|
| Upsample | DMA / tensor engine: nhân chỉ số pixel (nearest 2×), không qua MAC |
| QConcat | Sắp xếp vùng nhớ + (nếu cần) requant trong PPU hoặc CPU |
| Residual / QC2f | `ppu` với `ewise_en=1`, cùng domain quant hoặc requant trước add |
| Full model | `tb_golden_check` sau khi map đúng địa chỉ ghi P3/P4/P5 của IP |
