# Inference ảnh → P3/P4/P5 → CPU vẽ bbox & xác nhận phần cứng

## Mục tiêu

1. **Phần mềm (golden):** ảnh → model quantized → P3/P4/P5 lưu hex → CPU đọc hex → (decode/NMS) → vẽ box.  
2. **Phần cứng:** IP ghi P3/P4/P5 ra DDR (hoặc dump hex sau sim) → **cùng bước CPU** → so với golden để confirm.

**Confirm IP đúng:** so **tensor P3/P4/P5 bit-exact (hoặc gần đúng)** với `golden_*.hex` trước; bbox chỉ tin cậy khi decode khớp head thật của YOLOv10.

---

## Bước A — Export golden (một lần / đổi ảnh)

```bat
cd E:\KLTN_HMH_FINAL\SW_KLTN\PHASE_4\01_export
python export_golden_data.py --image ..\..\img1.jpg --output ..\02_golden_data
```

Có: `02_golden_data/golden_P3.hex`, `golden_P4.hex`, `golden_P5.hex`, `quant_params.json` (có `golden_outputs`), `letterbox_info.json`.

**Xác nhận P3/P4/P5 khớp model:**

```bat
cd E:\KLTN_HMH_FINAL\SW_KLTN\PHASE_4
python 03_rtl_cosim\verify_full_model_outputs.py --golden-dir 02_golden_data
```

---

## Bước B — CPU postprocess + vẽ bbox (từ hex golden)

```bat
cd E:\KLTN_HMH_FINAL\SW_KLTN\PHASE_4\04_postprocess
python cpu_postprocess.py --hex_dir ..\02_golden_data --image ..\..\img1.jpg --output out_golden.jpg
```

- Đọc shape/scale/zp từ `quant_params.json` → `golden_outputs`.
- **P3/P4/P5 golden hiện là feature map (128 / 256 / 512 kênh)** — không phải tensor dạng `4+num_classes` trên mỗi anchor, nên **không decode NMS được** từ hex theo kiểu YOLO cũ. Script sẽ ghi ảnh + dòng chữ hướng dẫn (không crash).
- **Chỉ để vẽ bbox minh họa** (cùng weights, không chứng minh IP):

```bat
python cpu_postprocess.py --hex_dir ..\02_golden_data --image ..\..\img1.jpg --output out_viz.jpg --boxes-from-model
```

- Xác nhận IP: **so khớp tensor** (`verify_full_model_outputs.py`, `compare_golden_vs_rtl.py`).

---

## Bước C — Sau khi có P3/P4/P5 từ RTL / FPGA

1. Dump từ sim hoặc đọc DDR thành 3 file hex **cùng format** với export (32 byte/dòng, thứ tự NCHW như `golden_*.hex`).  
2. Tạo thư mục, ví dụ `02_golden_data\rtl_dump\`, copy vào:
   - `golden_P3.hex`, `golden_P4.hex`, `golden_P5.hex` (ghi đè tên hoặc giữ tên + sửa `golden_outputs` trong JSON — cách đơn giản nhất là **giữ tên file** như golden).  
   - Copy kèm **`quant_params.json`** và **`letterbox_info.json`** từ `02_golden_data` (scale/zp/shape không đổi).  
3. Chạy lại postprocess trỏ `--hex_dir` vào thư mục dump:

```bat
python cpu_postprocess.py --hex_dir ..\02_golden_data\rtl_dump --image ..\..\img1.jpg --output out_fpga.jpg
```

4. **So sánh tensor với golden (bắt buộc để “confirm phần cứng”):**

```bat
cd E:\KLTN_HMH_FINAL\SW_KLTN\PHASE_4\01_export
python ..\03_rtl_cosim\compare_golden_vs_rtl.py
```

*(Nếu script compare cần đường dẫn — chỉnh trong file hoặc thêm tham số; hoặc so bằng tool tự viết: đếm byte khác nhau giữa hai file hex.)*

---

## Bước D — So detection với PyTorch (tùy chọn)

```bat
cd E:\KLTN_HMH_FINAL\SW_KLTN\PHASE_4\04_postprocess
python verify_detection.py ...
```

Cần ảnh output từ hai nguồn; xem docstring trong `verify_detection.py`.

---

## Tóm tắt

| Việc | Lệnh / file |
|------|-------------|
| Export P3/P4/P5 + params | `export_golden_data.py` |
| Confirm model ≡ hex | `verify_full_model_outputs.py` |
| Vẽ bbox từ hex | `cpu_postprocess.py` |
| Confirm RTL ≡ golden | Dump hex + `compare_golden_vs_rtl` (hoặc diff file) |
| Full RTL SoC | `tb_golden_check.sv` khi `accel_top` chạy xong |
