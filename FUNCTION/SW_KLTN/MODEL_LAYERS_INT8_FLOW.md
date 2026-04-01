## Nghiên cứu chi tiết MODEL FORWARD (INT8) theo từng layer – qYOLOv10n PTQ

File này dựa trên:
- Bảng trace thực tế:

```text
Layer  Module     Input dtype  Output dtype  Tính toán  Output shape
0      Conv       quint8       quint8        INT        (1, 16, 320, 320)
1      Conv       quint8       quint8        INT        (1, 32, 160, 160)
2      QC2f       quint8       quint8        INT        (1, 32, 160, 160)
3      Conv       quint8       quint8        INT        (1, 64, 80, 80)
4      QC2f       quint8       quint8        INT        (1, 64, 80, 80)
5      SCDown     quint8       quint8        INT        (1, 128, 40, 40)
6      QC2f       quint8       quint8        INT        (1, 128, 40, 40)
7      SCDown     quint8       quint8        INT        (1, 256, 20, 20)
8      QC2f       quint8       quint8        INT        (1, 256, 20, 20)
9      SPPF       quint8       quint8        INT        (1, 256, 20, 20)
10     QPSA       quint8       quint8        INT        (1, 256, 20, 20)
11     Upsample   quint8       quint8        INT        (1, 256, 40, 40)
12     QConcat    quint8       quint8        INT        (1, 384, 40, 40)
13     QC2f       quint8       quint8        INT        (1, 128, 40, 40)
14     Upsample   quint8       quint8        INT        (1, 128, 80, 80)
15     QConcat    quint8       quint8        INT        (1, 192, 80, 80)
16     QC2f       quint8       quint8        INT        (1, 64, 80, 80)
17     Conv       quint8       quint8        INT        (1, 64, 40, 40)
18     QConcat    quint8       quint8        INT        (1, 192, 40, 40)
19     QC2f       quint8       quint8        INT        (1, 128, 40, 40)
20     SCDown     quint8       quint8        INT        (1, 128, 20, 20)
21     QConcat    quint8       quint8        INT        (1, 384, 20, 20)
22     QC2fCIB    quint8       quint8        INT        (1, 256, 20, 20)
23     Qv10Detect quint8       float32       Float      (1, 300, 6)
```

- Và mô tả tổng thể trong `MODEL_FORWARD_FLOW.md`.

Giả thiết ở đây: **bộ tăng tốc phần cứng** thực hiện các layer **0–22** (backbone + neck), trả về 3 feature maps INT8 (P3, P4, P5). Layer **23 – Qv10Detect** và phần POSTPROCESS chạy trên CPU.

Trong phần dưới, mỗi layer gồm:
- **Thông tin chung**: index, kích thước, kiểu dữ liệu.
- **File / class định nghĩa**.
- **Luồng dữ liệu (input → output)**: INT8 / float, concat, skip connection.
- **Cách quantize / dequantize / requantize** liên quan.

> Lưu ý: mọi “INT/INT8” dưới đây đều ám chỉ tensor kiểu `quint8` (quantized INT8) theo PyTorch.

---

## 0. Bối cảnh trước layer 0 – input cho model forward

Sau khi CPU đã làm **PREPROCESS + QUANTIZE**:

- Tensor vào model forward (vào layer 0):
  - `X_int8`: `INT8 [1, 3, 640, 640]`
  - Metadata: `scale_in`, `zero_point_in`
- Trong code gốc, bước này tương đương:

```python
# nn/tasks.py::_predict_once_quantized
x = self.quant(x_float)  # QuantStub: float32 -> INT8
```

Trong thiết kế phần cứng theo đề tài, **bước quantize này đã được dời lên CPU**, nên **layer 0 nhận trực tiếp INT8**.

---

## 1. Layer 0 – `Conv` (downsample 2×, tăng kênh lên 16)

- **Trace**:  
  - Input dtype: `quint8`, shape ≈ `[1, 3, 640, 640]` (sau quant)  
  - Output dtype: `quint8`, shape: **`[1, 16, 320, 320]`**
- **File / class**:
  - `ultralytics/nn/modules/conv.py::class Conv`
  - Sử dụng `forward_fuse` trong inference (Conv + BN đã fuse).

### Kiến trúc layer

```python
class Conv(nn.Module):
    default_act = nn.SiLU()  # đã patch để hỗ trợ quantized

    def __init__(self, c1, c2, k=1, s=1, p=None, ...):
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), bias=False)
        self.bn   = nn.BatchNorm2d(c2)
        self.act  = self.default_act

    def forward_fuse(self, x):
        out = self.conv(x)
        # PATCH: xử lý khi out là INT8 (quint8/qint8)
        if out.dtype in (torch.quint8, torch.qint8):
            scale, zp = out.q_scale(), out.q_zero_point()
            qdtype = out.dtype
            out = self.act(out.dequantize())               # INT8 -> float32 -> SiLU
            out = torch.quantize_per_tensor(out, scale, zp, qdtype)  # float32 -> INT8
            return out
        return self.act(out)
```

### Dữ liệu đi qua

- **Input**: `X_int8 [1, 3, 640, 640]`
- **Tính toán**:
  - Conv2D với trọng số đã quantize (INT8) và BN fuse.
  - Kết quả `out_conv` là INT8 với scale/zp mới (`scale_0, zp_0`).
  - Do SiLU không có kernel QuantizedCPU, code dùng:
    - Dequant: `out_f = dequant(out_conv)` (float32).
    - SiLU trên float32: `out_act = SiLU(out_f)`.
    - Requant: `out_q = quantize(out_act, scale_0, zp_0)`.
- **Output**: `out_q: INT8 [1, 16, 320, 320]`.

**Trên phần cứng**, bạn có thể:
- Hoặc hỗ trợ SiLU INT8 native (không cần dequant/requant).
- Hoặc làm giống PyTorch: kernel dequant → SiLU float/s16 → quant.

---

## 2. Layer 1 – `Conv` (downsample 2×, tăng kênh lên 32)

- **Trace**:  
  - Input: `INT8 [1, 16, 320, 320]`  
  - Output: `INT8 [1, 32, 160, 160]`
- **File / class**: giống layer 0 (`Conv`).

### Dữ liệu đi qua

- **Input**: feature map INT8 `F1_in`.
- **Tính toán**:
  - Conv stride = 2 → giảm kích thước H,W xuống một nửa.
  - BN fuse + activation SiLU với patch INT8 tương tự layer 0.
- **Output**: `F1_out: INT8 [1, 32, 160, 160]`.

---

## 3. Layer 2 – `QC2f` (quantized C2f block, giữ kích thước, tăng representational capacity)

- **Trace**:  
  - Input: `INT8 [1, 32, 160, 160]`  
  - Output: `INT8 [1, 32, 160, 160]`
- **File / class**:
  - `ultralytics/nn/modules/block.py::class QC2f`

### Kiến trúc khối

`C2f` là block dạng “CSP/Res-like”: chia channels, apply các Conv nội bộ, rồi concat và Conv cuối. `QC2f` là phiên bản có chèn QuantStub/DeQuantStub ở các vị trí phù hợp để hỗ trợ quantization.

Ở mức cao, bạn có thể hình dung:

```text
Input INT8 (32 ch) ──► [split channels] ──► Conv/Conv/... (INT8)
                      │                      (các nhánh song song)   
                      └──────── concat ─────► Conv cuối (INT8) ─► Output INT8 (32 ch)
```

- Tất cả conv bên trong đã được thiết kế để tương thích với quantization (dùng `QConv` hoặc Conv + QuantStub/DeQuantStub).
- Skip connection chủ yếu là **concat** channel, không đổi dữ liệu (vẫn INT8).

### Dữ liệu đi qua

- **Input**: `F2_in: INT8 [1, 32, 160, 160]`.
- **Tính toán**:
  - Các conv nội bộ: INT8 conv.
  - Activation (SiLU/ReLU) như Conv.
  - Concat kênh, cuối cùng là 1 Conv kết hợp.
- **Output**: `F2_out: INT8 [1, 32, 160, 160]`.

---

## 4. Layer 3 – `Conv` (downsample 2×, tăng kênh lên 64)

- **Trace**:  
  - Input: `INT8 [1, 32, 160, 160]`  
  - Output: `INT8 [1, 64, 80, 80]`
- **File / class**: `Conv` như layer 0, 1.

### Dữ liệu đi qua

- Conv stride 2 → giảm H,W một nửa, tăng channel.
- Output dùng làm “P3” feature thô cho các block C2f tiếp theo.

---

## 5. Layer 4 – `QC2f` (block C2f trên feature 80×80, 64 kênh)

- **Trace**:  
  - Input: `INT8 [1, 64, 80, 80]`  
  - Output: `INT8 [1, 64, 80, 80]`
- **File / class**: `QC2f` (block.py).

Giống mô tả layer 2, nhưng kích thước lớn hơn, nhiều kênh hơn. Đây là một phần quan trọng của backbone – trích xuất đặc trưng ở tỉ lệ 1/8.

---

## 6. Layer 5 – `SCDown` (stride conv downsample, tăng kênh lên 128)

- **Trace**:  
  - Input: `INT8 [1, 64, 80, 80]`  
  - Output: `INT8 [1, 128, 40, 40]`
- **File / class**:
  - `ultralytics/nn/modules/block.py::class SCDown`

### Dữ liệu đi qua

`SCDown` thường là khối downsample tối ưu hơn conv stride 2 thuần túy, có thể gồm 2 nhánh:
- Nhánh conv stride 2.
- Nhánh pool stride 2 rồi conv.
- Kết hợp (concat/add) hai nhánh.

### Dữ liệu đi qua

- **Input**: `F5_in: INT8 [1, 64, 80, 80]`.
- **Tính toán**:
  - Mỗi nhánh là các conv/pool INT8.
  - Kết hợp (add/concat) trên INT8 (giả sử cùng scale/zp hoặc đã rescale nội bộ).
- **Output**: `F5_out: INT8 [1, 128, 40, 40]`.

---

## 7. Layer 6 – `QC2f` (block C2f ở tỉ lệ 1/16, 128 kênh)

- **Trace**:  
  - Input: `INT8 [1, 128, 40, 40]`  
  - Output: `INT8 [1, 128, 40, 40]`
- **File / class**: `QC2f` (block.py).

### Vai trò

- Tăng độ biểu diễn tại tỉ lệ 1/16 (40×40), nơi có nhiều object vừa và lớn.
- Vẫn giữ kích thước H,W, chỉ biến đổi feature trong không gian kênh.

### Dữ liệu đi qua

- **Input**: `F6_in: INT8 [1, 128, 40, 40]`.
- **Tính toán**: nhiều conv INT8 nội bộ + concat + conv cuối.
- **Output**: `F6_out: INT8 [1, 128, 40, 40]`.

---

## 8. Layer 7 – `SCDown` (downsample 2×, tăng kênh lên 256)

- **Trace**:  
  - Input: `INT8 [1, 128, 40, 40]`  
  - Output: `INT8 [1, 256, 20, 20]`
- **File / class**: `SCDown` (block.py).

### Dữ liệu đi qua

- Tương tự layer 5 nhưng ở mức kênh lớn hơn:
  - Conv/pool stride 2 trên INT8.
  - Kết hợp nhánh trên INT8.
- Đây là tỉ lệ 1/32 (20×20), feature rất nén, nhắm tới object lớn.

---

## 9. Layer 8 – `QC2f` (C2f trên tỉ lệ 1/32, 256 kênh)

- **Trace**:  
  - Input: `INT8 [1, 256, 20, 20]`  
  - Output: `INT8 [1, 256, 20, 20]`
- **File / class**: `QC2f`.

### Vai trò

- Củng cố đặc trưng ở tầng sâu nhất (1/32), sử dụng các block C2f CSP-like nhưng tương thích quantization.

### Dữ liệu đi qua

- **Input**: `F8_in: INT8 [1, 256, 20, 20]`.
- **Tính toán**: conv INT8 + concat như các QC2f khác.
- **Output**: `F8_out: INT8 [1, 256, 20, 20]`.

---

## 10. Layer 9 – `SPPF` (Spatial Pyramid Pooling Fast, INT8)

- **Trace**:  
  - Input: `INT8 [1, 256, 20, 20]`  
  - Output: `INT8 [1, 256, 20, 20]`
- **File / class**:
  - `ultralytics/nn/modules/block.py::class SPPF`

### Kiến trúc

- Một conv giảm kênh (hoặc giữ nguyên), sau đó là một chuỗi max-pooling với các kernel/lớp lặp lại, concat kết quả rồi conv lại.

### Dữ liệu đi qua

- **Input**: `F9_in: INT8 [1, 256, 20, 20]`.
- **Tính toán**:
  - MaxPool INT8 (kernel 5, v.v.) tạo multi-scale context.
  - Concat các kết quả pool.
  - Conv INT8 cuối để trộn.
- **Output**: `F9_out: INT8 [1, 256, 20, 20]`.

Lưu ý: max-pooling trên INT8 chỉ là so sánh số nguyên; không gây vấn đề về scale/zp, vì chúng dùng chung scale/zp của feature.

---

## 11. Layer 10 – `QPSA` (quantized PSA block)

- **Trace**:  
  - Input: `INT8 [1, 256, 20, 20]`  
  - Output: `INT8 [1, 256, 20, 20]`
- **File / class**:
  - `ultralytics/nn/modules/block.py::class QPSA`

### Vai trò

- `PSA` thường là một block attention (Pixel/Position-Sensitive Attention). `QPSA` là biến thể hỗ trợ quantization.

### Dữ liệu đi qua

- **Input**: `F10_in: INT8 [1, 256, 20, 20]`.
- **Tính toán**:
  - Một số conv/FC nội bộ trên INT8.
  - Tính attention weights, áp dụng trên feature map (nhân/scale).
  - Có thể có dequant/requant ở các bước attention khó lượng tử trực tiếp.
- **Output**: `F10_out: INT8 [1, 256, 20, 20]`.

---

## 12. Layer 11 – `Upsample` (lên mẫu 2×, 256 kênh)

- **Trace**:  
  - Input: `INT8 [1, 256, 20, 20]`  
  - Output: `INT8 [1, 256, 40, 40]`
- **File / class**:
  - `ultralytics/nn/modules/block.py` hoặc `nn/functional.interpolate` được wrap

### Dữ liệu đi qua

- **Input**: `F11_in: INT8 [1, 256, 20, 20]`.
- **Tính toán**:
  - `Upsample` (nearest) chỉ copy/replicate giá trị INT8 → **không đổi scale/zp**.
  - H,W nhân đôi: 20 → 40.
- **Output**: `F11_out: INT8 [1, 256, 40, 40]`.

---

## 13. Layer 12 – `QConcat` (concat features 256×40×40 và skip từ 128×40×40)

- **Trace**:  
  - Input dtype: `quint8` (cả 2 nhánh)  
  - Output: `INT8 [1, 384, 40, 40]`
- **File / class**:
  - `ultralytics/nn/modules/conv.py::class QConcat`

### Kiến trúc & scale/zp

`QConcat` nhận **hai tensor INT8** cùng H,W nhưng kênh khác nhau (ví dụ):
- Nhánh 1 (từ Upsample layer 11): `INT8 [1, 256, 40, 40]`, scale/zp = `s11, zp11`.
- Nhánh 2 (skip connection từ layer 6/5): `INT8 [1, 128, 40, 40]`, scale/zp = `s_skip, zp_skip`.

Để concat hợp lệ, có 2 cách:
1. **Chuẩn (đẹp)**: đảm bảo hai nhánh đã dùng **cùng scale/zp** (kiến trúc quantization chung). Khi đó:
   - Concat chỉ là ghép channel:  
     `[C=256] || [C=128] → [C=384]`, scale/zp giữ nguyên.
2. **Nếu scale khác** (trường hợp tổng quát):
   - Rescale một nhánh về scale của nhánh kia trước khi concat (trên INT8).
   - Ultralytics trong thiết kế Q* cố gắng **dùng cùng qconfig** để tránh rescale phức tạp.

### Dữ liệu đi qua

- **Input**:
  - `F11_out: INT8 [1, 256, 40, 40]`
  - `F_skip:  INT8 [1, 128, 40, 40]` (từ backbone, cùng scale trong thiết kế chuẩn)
- **Tính toán**: concat theo chiều kênh.
- **Output**: `F12_out: INT8 [1, 384, 40, 40]`.

---

## 14. Layer 13 – `QC2f` (C2f trên feature 384×40×40, output 128×40×40)

- **Trace**:  
  - Input: `INT8 [1, 384, 40, 40]`  
  - Output: `INT8 [1, 128, 40, 40]`
- **File / class**: `QC2f`.

### Dữ liệu đi qua

- **Input**: `F13_in: INT8 [1, 384, 40, 40]`.
- **Tính toán**:
  - Một chuỗi conv nội bộ (thường giảm/đổi kênh) + concat + conv cuối.
  - Tất cả trên INT8 (có thể patch tương tự cho activation như Conv).
- **Output**: `F13_out: INT8 [1, 128, 40, 40]`.

Đây là một trong các feature map quan trọng cho detection (tầng trung bình).

---

## 15. Layer 14 – `Upsample` (lên mẫu feature 128×40×40 → 128×80×80)

- **Trace**:  
  - Input: `INT8 [1, 128, 40, 40]`  
  - Output: `INT8 [1, 128, 80, 80]`
- **File / class**: Upsample (nearest) trên INT8.

### Dữ liệu đi qua

- Không thay đổi scale/zp, chỉ nhân đôi H,W.
- Output được dùng để concat với skip feature ở tỉ lệ 1/8.

---

## 16. Layer 15 – `QConcat` (concat với skip ở 64×80×80)

- **Trace**:  
  - Input: 2 nhánh INT8: `[1, 128, 80, 80]` và `[1, 64, 80, 80]`  
  - Output: `INT8 [1, 192, 80, 80]`
- **File / class**: `QConcat`.

### Dữ liệu đi qua

- Giống layer 12, concat theo channel:
  - Nhánh upsample từ tầng trung (`128 ch`).
  - Nhánh skip từ backbone ở tỉ lệ 1/8 (`64 ch`).
- Kết quả: `F15_out: INT8 [1, 192, 80, 80]`.

---

## 17. Layer 16 – `QC2f` (C2f trên 192×80×80, output 64×80×80)

- **Trace**:  
  - Input: `INT8 [1, 192, 80, 80]`  
  - Output: `INT8 [1, 64, 80, 80]`
- **File / class**: `QC2f`.

### Vai trò & dữ liệu

- Đây là tầng đặc trưng có độ phân giải cao nhất dùng cho head (detection các object nhỏ).
- Tương tự QC2f các tầng trước, nhưng kích thước H,W lớn nhất (80×80).

---

## 18. Layer 17 – `Conv` (giảm kích thước 80→40, 64→64 ch)

- **Trace**:  
  - Input: `INT8 [1, 64, 80, 80]`  
  - Output: `INT8 [1, 64, 40, 40]`
- **File / class**: `Conv`.

### Dữ liệu đi qua

- Conv stride 2 trên INT8:
  - Giảm H,W, giữ số kênh.
- Feature này sau đó được concat với feature trung gian (layer 13) để tạo “P4 head feature”.

---

## 19. Layer 18 – `QConcat` (ghép feature 64×40×40 và 128×40×40)

- **Trace**:  
  - Input: INT8 `[1, 64, 40, 40]` + `[1, 128, 40, 40]`  
  - Output: INT8 `[1, 192, 40, 40]`
- **File / class**: `QConcat`.

### Dữ liệu đi qua

- Concat hai nhánh theo channel:
  - Nhánh từ downsample output của high-res.
  - Nhánh từ backbone trung gian.
- Kết quả: `F18_out: INT8 [1, 192, 40, 40]`.

---

## 20. Layer 19 – `QC2f` (C2f trên 192×40×40, output 128×40×40)

- **Trace**:  
  - Input: `INT8 [1, 192, 40, 40]`  
  - Output: `INT8 [1, 128, 40, 40]`
- **File / class**: `QC2f`.

### Vai trò

- Đưa về số kênh chuẩn (128) cho tỉ lệ 1/16, đây là một trong các feature P4 dùng cho detection.

---

## 21. Layer 20 – `SCDown` (downsample 40→20, 128 ch)

- **Trace**:  
  - Input: `INT8 [1, 128, 40, 40]`  
  - Output: `INT8 [1, 128, 20, 20]`
- **File / class**: `SCDown`.

### Dữ liệu đi qua

- Downsample nhóm feature trung bình để ghép với feature sâu (256 ch) ở 1/32.

---

## 22. Layer 21 – `QConcat` (ghép 128×20×20 và 256×20×20)

- **Trace**:  
  - Input: `[1, 128, 20, 20]` + `[1, 256, 20, 20]`  
  - Output: `[1, 384, 20, 20]`
- **File / class**: `QConcat`.

### Dữ liệu đi qua

- Concat 2 feature map INT8:
  - Từ nhánh medium (downsample từ 40×40).
  - Từ nhánh deep (256 ch).
- Kết quả: `F21_out: INT8 [1, 384, 20, 20]`.

---

## 23. Layer 22 – `QC2fCIB` (C2f với “CIB” head, output 256×20×20)

- **Trace**:  
  - Input: `INT8 [1, 384, 20, 20]`  
  - Output: `INT8 [1, 256, 20, 20]`
- **File / class**:
  - `ultralytics/nn/modules/block.py::class QC2fCIB`

### Vai trò

- Đây là tầng cuối của **neck**; output 256×20×20 là một trong các feature chính cho Qv10Detect ở tỉ lệ 1/32.

### Dữ liệu đi qua

- Nhiều conv nội bộ trên INT8, thiết kế chuyên cho YOLOv10 CIB (Classifier-In-Box) head.

---

## 24. Layer 23 – `Qv10Detect` (head, chuyển INT8 → float32 predictions)

> Trong kiến trúc “bộ tăng tốc backbone+neck”, **layer 23 nằm trên CPU**, không thuộc bộ tăng tốc. Tuy nhiên, để hiểu luồng dữ liệu, ta vẫn mô tả ngắn.

- **Trace**:  
  - Input: `INT8 [1, 256, 20, 20]` (và thêm các feature P3, P4 từ các layer trước)  
  - Output: `float32 [1, 300, 6]`
- **File / class**: `ultralytics/nn/modules/head.py::class Qv10Detect`

### Dữ liệu đi qua

- Qv10Detect thường nhận 3 feature maps đa tỉ lệ (P3,P4,P5):
  - P3: `float32 [1, 64, 80, 80]`
  - P4: `float32 [1, 128, 40, 40]`
  - P5: `float32 [1, 256, 20, 20]`
- Trong PTQ, trước khi vào head:
  - Dequant 3 feature maps: INT8 → float32.
- Head:
  - Một số conv 1×1 và 3×3 trên float32.
  - Tách ra nhánh **regression** (bbox) và **classification** (logit class).
  - Gộp / decode thành tensor `(N, NumAnchors, 4+Nc)`.

Kết quả này là **đầu vào của POSTPROCESS (NMS)**.

---

## 25. Tóm tắt cách quantize / dequantize / requantize & concat/skip

### 25.1. Quantize & DeQuantize & ReQuantize

1. **Quantize (trước model forward)** – trên CPU:
   - `X_float [1,3,640,640]` → `X_int8 [1,3,640,640]` + `(scale_in, zp_in)`.
2. **Bên trong các Conv/QConv/Block**:
   - Input/Output phần lớn là INT8, với **một scale/zp cho mỗi tensor hoặc mỗi kênh** (tùy qconfig).
   - Ở các điểm activation **SiLU**:
     - Dequant tạm thời: `INT8 → float32`.
     - Tính SiLU trên float32.
     - ReQuant: `float32 → INT8` với scale/zp thích hợp.
3. **DeQuantize cuối (trước head/NMS)**:
   - Các feature maps head (P3,P4,P5) được dequant sang float32 nếu head chạy trên CPU.

### 25.2. Concat & Skip connection và scale/zero-point

- Trong các `QConcat`/skip connection:
  - **Thiết kế chuẩn** cố gắng dùng **cùng qconfig, cùng scale/zp** trên các nhánh được concat.
  - Khi đó, concat chỉ là nối kênh, không cần rescale.
  - Nếu scale khác nhau (edge case):
    - Cần một bước rescale trên INT8 để đưa về cùng scale trước khi concat.
    - Tuy nhiên, trong code Q* của Ultralytics (QC2f, QConcat, QSCDown...), cấu hình được chuẩn hoá để tránh trường hợp này càng nhiều càng tốt.

### 25.3. View tổng quan cho phần cứng

- **Backbone + Neck (Layer 0–22)**:
  - Hầu hết conv/block chạy trên **INT8**:
    - Input/Output: INT8 tensors với scale/zp từng tensor.
  - Activation khó lượng tử (SiLU): `INT8 → float → INT8` nội bộ.
  - Concat/skip: hoạt động trên INT8, giả sử scale/zp đã đồng nhất.
- **Head + NMS (Layer 23+) trên CPU**:
  - Head: dequant feature maps về float32, tính bbox + logits.
  - NMS + scale bbox + output: float32.

Điều này cho phép bạn thiết kế một **bộ tăng tốc INT8** cho toàn bộ backbone + neck, chỉ cần đảm bảo:
- Hỗ trợ conv/pool/add/concat INT8 với scale/zp nhất quán.
- Có giải pháp cho activation (SiLU) – INT8 native hoặc dequant/requant cục bộ.
- Xuất 3 feature maps cuối (P3,P4,P5) dưới dạng INT8 + scale/zp, cho CPU tiếp tục head + POSTPROCESS.


