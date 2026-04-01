## Mô tả chi tiết flow inference và model forward (qYOLOv10n PTQ)

### 1. Phân chia CPU (phần mềm) vs. Bộ tăng tốc (phần cứng)

#### 1.1. CPU – INPUT → SETUP → PREPROCESS → QUANTIZE

- **INPUT**
  - Ảnh gốc (ví dụ `img1.jpg`), đọc bằng OpenCV/PIL → `numpy (H, W, 3)` BGR.

- **SETUP SOURCE**  
  - **File / hàm**:
    - `ultralytics/engine/predictor.py::BasePredictor.setup_source`
    - `ultralytics/data/build.py::load_inference_source`
  - **Nhiệm vụ**:
    - Xác định kiểu nguồn (`image`, `video`, `webcam`, ...).
    - Chọn kích thước inference: `imgsz` (thường 640) và `stride` (thường 32).
    - Tạo iterator `dataset` (trả ra các batch `paths, im0s, s`).

- **PREPROCESS**  
  - **File / hàm**:
    - `ultralytics/engine/predictor.py::BasePredictor.pre_transform`
    - `ultralytics/engine/predictor.py::BasePredictor.preprocess`
    - `ultralytics/data/augment.py::LetterBox`
  - **Luồng xử lý** trên CPU:
    1. `LetterBox(imgsz, stride)`:
       - Resize ảnh giữ tỉ lệ.
       - Thêm padding (màu xám/đen) để kích thước khớp với (H,W) bội số stride.
    2. Chuyển kênh màu:
       - BGR → RGB (nếu cần).
    3. Đổi layout:
       - HWC `(H, W, 3)` → CHW `(3, H, W)`.
    4. Chuyển sang tensor:
       - `numpy` → `torch.Tensor.float32`.
       - Chuẩn hóa: chia 255 → giá trị trong `[0.0, 1.0]`.
    5. Thêm batch dimension:
       - Shape cuối cùng: **`X_float: float32 [N, 3, H, W]`** (thường `[1, 3, 640, 640]`).

- **QUANTIZE TRÊN CPU (chuẩn bị INPUT INT8 cho bộ tăng tốc)**  
  - Logic tương đương `QuantStub` trong `_predict_once_quantized` (file `nn/tasks.py`), nhưng ta dời ra CPU.
  - **Đầu vào**:  
    - `X_float: float32 [1, 3, 640, 640]`, trong `[0,1]`.
  - **Quantize**:
    - Chọn (hoặc đã học) `scale_in` và `zero_point_in`.
    - Ánh xạ:
      \[
      X_\text{int8} = \text{round}\left(\frac{X_\text{float}}{\text{scale\_in}}\right) + \text{zero\_point\_in}
      \]
    - Kết quả:
      - **`X_int8: INT8 [1, 3, 640, 640]`** (thực tế dùng `quint8`).  
      - Metadata kèm theo: `scale_in` (float32), `zero_point_in` (int).

> **Kết luận 1:** CPU chịu trách nhiệm: **đọc ảnh → resize/pad → chuẩn hóa float32 → quantize về INT8**.  
> Bộ tăng tốc **không cần xử lý trực tiếp ảnh gốc**, chỉ nhận `X_int8` và metadata.

---

#### 1.2. Bộ tăng tốc – MODEL FORWARD (Backbone + Neck)

Ta giả sử bộ tăng tốc thực hiện trọn vẹn **Backbone + Neck** của qYOLOv10n PTQ, còn **Qv10Detect head + POSTPROCESS** để lại cho CPU.

- **INPUT chính thức của bộ tăng tốc**:
  - `X_int8: INT8 [1, 3, 640, 640]`
  - `scale_in: float32`
  - `zero_point_in: int32`

- **Trong mã nguồn PyTorch**, đoạn bạn đang thay thế tương đương với:

```python
# nn/tasks.py::_predict_once_quantized (đã lược giản)
def _predict_once_quantized(self, x, ...):
    x = self.quant(x)  # bước này bạn đã đưa ra CPU
    y = []
    for m in self.model:   # self.model = nn.Sequential(*layers)
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x nếu j==-1 else y[j] for j in m.f]
        x = m(x)           # Conv, QConv, C2f, QSPPF, SCDown, ...
        y.append(x nếu m.i in self.save else None)
    return x
```

- **Triển khai trong bộ tăng tốc (mức khối):**

  1. **Nhận `X_int8` + `scale_in`, `zero_point_in`** từ CPU.
  2. **Backbone + Neck** – chạy chuỗi các layer:
     - Conv/QConv + BN đã fuse + activation (SiLU/ReLU) trên INT8.
     - Các block như `C2f`, `QC2f`, `SCDown`, `QSPPF`, `QC2fCIB`…
     - Các phép cộng skip connection, concat feature maps.
  3. **Kết quả cuối Backbone/Neck** (YOLOv10-style):
     - 3 feature maps đa tỉ lệ:
       - `P3_int8: INT8 [1, C3, H3, W3]`
       - `P4_int8: INT8 [1, C4, H4, W4]`
       - `P5_int8: INT8 [1, C5, H5, W5]`
     - Mỗi tensor đi kèm `scale_3, zp_3`, `scale_4, zp_4`, `scale_5, zp_5` tương ứng.

> **Kết luận 2:** Bộ tăng tốc nhận **một feature map INT8 đầu vào duy nhất** (`X_int8`), và trả về **ba feature map INT8 cuối** (`P3_int8, P4_int8, P5_int8`) – chính là input cho head Qv10Detect ở phía CPU.

---

#### 1.3. CPU – Qv10Detect head + POSTPROCESS

Sau khi nhận `P3_int8, P4_int8, P5_int8` từ bộ tăng tốc:

1. **Dequantize về float32** (nếu head không được quantize trên CPU):

```python
P3_float = dequant(P3_int8, scale_3, zp_3)  # float32 [1, C3, H3, W3]
P4_float = dequant(P4_int8, scale_4, zp_4)
P5_float = dequant(P5_int8, scale_5, zp_5)
```

2. **Qv10Detect head** (chạy trên CPU, float32):
   - **File / hàm**: `ultralytics/nn/modules/head.py::Qv10Detect.forward`
   - Input: 3 feature maps `P3_float, P4_float, P5_float`.
   - Output:
     - `preds_raw: float32 [N, NumAnchors, 4 + Nc]`
     - 4 = thông tin bbox (định dạng nội bộ).
     - `Nc` = số lớp (class).

3. **POSTPROCESS**:
   - **File / hàm**:
     - `ultralytics/models/yolo/detect/predict.py::DetectionPredictor.postprocess`
     - `ultralytics/utils/nms.py::non_max_suppression`
   - Các bước:
     - **NMS**: loại bỏ box trùng, giữ box có conf tốt hơn.
     - Scale bbox từ tọa độ trong ảnh resize (letterbox) → tọa độ ảnh gốc.
     - Tạo `Results`:
       - `boxes.xyxy`, `boxes.conf`, `boxes.cls`, `names`, `orig_shape`.
     - Vẽ box lên ảnh, lưu file (`output_quant.jpg`).

> **Kết luận 3:** CPU sau bộ tăng tốc đảm nhiệm các khối **Qv10Detect head → NMS → tạo output cuối**.

---

### 2. Tổng hợp giao diện bộ tăng tốc (Model Forward)

Để mapping xuống phần cứng, có thể chốt lại giao diện như sau.

#### 2.1. Input bộ tăng tốc

- **Dữ liệu chính**:
  - `X_int8: INT8 [1, 3, 640, 640]`  
    (INT8 có thể dùng `quint8` chuẩn PyTorch).
- **Metadata quantization**:
  - `scale_in: float32` – hệ số scale của input.
  - `zero_point_in: int32` – zero-point của input.

> Đây là kết quả sau khi **CPU đã hoàn thành toàn bộ PREPROCESS + QUANTIZE**.

#### 2.2. Output bộ tăng tốc

- **Feature maps đầu ra (đa tỉ lệ)**:
  - `P3_int8: INT8 [1, C3, H3, W3]`
  - `P4_int8: INT8 [1, C4, H4, W4]`
  - `P5_int8: INT8 [1, C5, H5, W5]`
- **Metadata cho từng feature map**:
  - `scale_3, zp_3` cho `P3_int8`
  - `scale_4, zp_4` cho `P4_int8`
  - `scale_5, zp_5` cho `P5_int8`

Các tensor này sẽ là **input trực tiếp** cho Qv10Detect head phía CPU (sau dequantize).

#### 2.3. Ranh giới trách nhiệm

- **CPU (Phần mềm)**:
  - Tiền xử lý ảnh (I/O, resize, normalize).
  - Quantize / Dequantize.
  - Qv10Detect head (nếu không đưa xuống HW).
  - NMS, scale bbox, hiển thị/lưu kết quả.

- **Bộ tăng tốc (Phần cứng)**:
  - Nhận INT8 feature map đầu vào `X_int8`.
  - Thực hiện toàn bộ backbone + neck của qYOLOv10n PTQ trên INT8.
  - Trả về 3 feature maps INT8 cuối (`P3_int8, P4_int8, P5_int8`) + metadata quantization.

> Với cách phân tách này, việc mapping từng **layer / block** xuống phần cứng trở nên rõ ràng: mỗi `m(x)` trong `_predict_once_quantized` tương ứng với một khối xử lý trên HW (Conv, C2f, SCDown, SPPF, ...), trong khi CPU chỉ làm glue logic và các bước sau cùng.

