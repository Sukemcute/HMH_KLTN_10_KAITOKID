# TỔNG HỢP TOÀN BỘ TÀI LIỆU NGHIÊN CỨU
# Đề tài: Thiết kế Bộ Tăng Tốc Phần Cứng YOLOv10n INT8 trên FPGA

> **Tác giả**: HMH  
> **Phạm vi**: Backbone + Neck (Layer 0–22) của qYOLOv10n PTQ  
> **Target**: Xilinx Virtex-7 XC7VX690T, >100 FPS @ 200-250 MHz  

---

## MỤC LỤC TỔNG HỢP

### PHẦN I — PHÂN TÍCH MÔ HÌNH (Model Analysis)
- [1.1. Luồng Inference & Model Forward](#phần-i1---luồng-inference--model-forward)
- [1.2. Chi Tiết Từng Layer INT8 (L0–L22)](#phần-i2---chi-tiết-từng-layer-int8)
- [1.3. Các Khối Tính Toán INT8 Chi Tiết](#phần-i3---các-khối-tính-toán-int8-chi-tiết)
- [1.4. Đồ Thị Phụ Thuộc Giữa Các Layer](#phần-i4---đồ-thị-phụ-thuộc-giữa-các-layer)

### PHẦN II — NGHIÊN CỨU MAPPING PHẦN CỨNG
- [2.1. Mapping Khối Tính Toán lên Phần Cứng qua Primitive Set](#phần-ii1---mapping-phần-cứng)

### PHẦN III — KIẾN TRÚC PHẦN CỨNG V2 (>100 FPS)
- [3.1. Kiến Trúc V2: Scale-Up Analysis & Redesign](#phần-iii1---kiến-trúc-v2)

### PHẦN IV — QUY TRÌNH HIỆN THỰC
- [4.1. Quy Trình Hiện Thực Tăng Tốc Phần Cứng](#phần-iv1---quy-trình-hiện-thực)

### PHẦN V — ĐẶC TẢ MODULE RTL
- [5.1. Đặc Tả Toàn Bộ Module RTL](#phần-v1---đặc-tả-module-rtl)

### PHẦN VI — ĐẶC TẢ PHASE 0 (Spec Freeze)
- [6.1. Primitive Matrix](#phần-vi1---primitive-matrix)
- [6.2. Layer Mapping](#phần-vi2---layer-mapping)
- [6.3. Quantization Policy](#phần-vi3---quantization-policy)
- [6.4. Layout & Addressing](#phần-vi4---layout--addressing)
- [6.5. Descriptor Spec](#phần-vi5---descriptor-spec)
- [6.6. Execution Semantics](#phần-vi6---execution-semantics)
- [6.7. Golden Python Plan](#phần-vi7---golden-python-plan)
- [6.8. RTL Mapping Plan](#phần-vi8---rtl-mapping-plan)

### PHẦN VII — CHIẾN LƯỢC XÂY DỰNG & NGHIÊN CỨU BỔ SUNG
- [7.1. Build Strategy (Phase 3)](#phần-vii1---build-strategy)
- [7.2. Kiến Trúc & Primitive RTL (Phase 5)](#phần-vii2---kiến-trúc--primitive-rtl)
- [7.3. RTL Hierarchy Top-Down (Phase 6)](#phần-vii3---rtl-hierarchy)
- [7.4. Primitive to PE Mode Mapping (Phase 6)](#phần-vii4---primitive-to-pe-mode)
- [7.5. Subcluster Internal Datapath (Phase 6)](#phần-vii5---subcluster-datapath)
- [7.6. Inference P3/P4/P5 & Layer Map (Phase 6)](#phần-vii6---inference-layer-map)
- [7.7. Research: Compute Blocks Primitive to Layer (Phase 6)](#phần-vii7---compute-blocks-research)

---
---


# ════════════════════════════════════════════════════════════════
# PHẦN I — PHÂN TÍCH MÔ HÌNH (Model Analysis)
# ════════════════════════════════════════════════════════════════

---

<a id='phần-i1---luồng-inference--model-forward'></a>

# PHẦN I.1 — Luồng Inference & Model Forward
> Nguồn: `SW_KLTN/MODEL_FORWARD_FLOW.md`

---

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



---
---

<a id='phần-i2---chi-tiết-từng-layer-int8'></a>

# PHẦN I.2 — Chi Tiết Từng Layer INT8 (L0–L22)
> Nguồn: `SW_KLTN/MODEL_LAYERS_INT8_FLOW.md`

---

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




---
---

<a id='phần-i3---các-khối-tính-toán-int8-chi-tiết'></a>

# PHẦN I.3 — Các Khối Tính Toán INT8 Chi Tiết
> Nguồn: `SW_KLTN/MODEL_BLOCKS_INT8_DETAIL.md`

---

## Các khối tính toán INT8 chính trong qYOLOv10n PTQ

Tài liệu này tóm tắt và đi sâu vào các **khối layer** mà bộ tăng tốc của bạn cần hỗ trợ, dựa trên `MODEL_LAYERS_INT8_FLOW.md`:

- `Conv`
- `QC2f`
- `SCDown`
- `SPPF`
- `QPSA`
- `Upsample`
- `QConcat`
- `QC2fCIB`

Mỗi mục gồm:
- **Input / Output (INT8)**: dạng tensor, kích thước, vai trò trong mạng.
- **Cấu trúc tính toán** (mức block).
- **Quantize / DeQuant / ReQuant & concat/skip**.
- **File + class / hàm định nghĩa** trong Ultralytics.

---

## 1. Khối `Conv` (Conv2d + BN fuse + activation)

### 1.1. Input / Output (theo flow INT8)

Ví dụ từ `MODEL_LAYERS_INT8_FLOW.md`:

- Layer 0:
  - Input: `quint8 [1, 3, 640, 640]` → Output: `quint8 [1, 16, 320, 320]`
- Layer 1:
  - Input: `quint8 [1, 16, 320, 320]` → Output: `quint8 [1, 32, 160, 160]`
- Layer 3:
  - Input: `quint8 [1, 32, 160, 160]` → Output: `quint8 [1, 64, 80, 80]`
- Layer 17:
  - Input: `quint8 [1, 64, 80, 80]` → Output: `quint8 [1, 64, 40, 40]`

**Tổng quát:**

- Input: `X_int8 ∈ INT8 [N, C_in, H_in, W_in]` (per-tensor scale `s_in`, zp `z_in`).
- Output: `Y_int8 ∈ INT8 [N, C_out, H_out, W_out]` với scale `s_out`, zp `z_out`.

### 1.2. Cấu trúc tính toán

**File / class / hàm:**

- `ultralytics/nn/modules/conv.py`
  - `class Conv(nn.Module)`
  - `Conv.forward_fuse(self, x)` – dùng trong inference, sau khi BN đã fuse.

**Cấu trúc logic:**

```python
class Conv(nn.Module):
    default_act = nn.SiLU()  # đã được patch để chơi với quant

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn   = nn.BatchNorm2d(c2)
        self.act  = default_act or nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        out = self.conv(x)
        # PATCH cho quantized:
        if out.dtype in (torch.quint8, torch.qint8):
            scale, zp = out.q_scale(), out.q_zero_point()
            qdtype = out.dtype
            out = self.act(out.dequantize())  # INT8 -> float32 -> SiLU
            out = torch.quantize_per_tensor(out, scale, zp, qdtype)  # float32 -> INT8
            return out
        return self.act(out)
```

**View phần cứng:**

1. **Conv2d INT8**:
   - MAC: (int8×int8 → int32), chạy trên engine conv.
   - Kết quả accumulator int32.
2. **Requant thành INT8**:

   - Dùng scale, zp đã được thiết lập sau PTQ: int32 → int8.
3. **Activation (SiLU)**:
   - Cách 1 (như PyTorch patch): INT8 → dequant (float32) → SiLU → INT8.
   - Cách 2 (tối ưu HW): triển khai LUT hoặc hàm xấp xỉ SiLU trực tiếp trên INT8.

### 1.3. Pseudo-code logic (cho tài liệu luận văn)

```python
class Conv(nn.Module):
    default_act = nn.SiLU()  # đã được patch để chơi với quant

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        self.conv = nn.Conv2d(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k,
            stride=s,
            padding=autopad(k, p, d),
            groups=g,
            dilation=d,
            bias=False,
        )
        self.bn   = nn.BatchNorm2d(c2)
        self.act  = self.default_act if act else nn.Identity()

    def forward(self, x):
        # training / float path: Conv + BN + activation
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        # inference path sau khi BN được fuse vào Conv
        out = self.conv(x)
        # PATCH cho quantized:
        if out.dtype in (torch.quint8, torch.qint8):
            scale, zp = out.q_scale(), out.q_zero_point()
            qdtype = out.dtype
            out_f  = out.dequantize()          # INT8 -> float32
            out_f  = self.act(out_f)           # SiLU trên float32
            out_q  = torch.quantize_per_tensor(out_f, scale, zp, qdtype)
            return out_q                       # float32 -> INT8
        # float32 / float16 bình thường
        return self.act(out)
```

---

## 2. Khối `QC2f` – quantized C2f block (CSP-like)

### 2.1. Input / Output

Ví dụ:

- Layer 2: `INT8 [1, 32, 160, 160]` → `INT8 [1, 32, 160, 160]`
- Layer 4: `INT8 [1, 64, 80, 80]` → `INT8 [1, 64, 80, 80]`
- Layer 6: `INT8 [1, 128, 40, 40]` → `INT8 [1, 128, 40, 40]`
- Các layer 8, 13, 16, 19 tương tự.

**Tổng quát:**

- Input: `X_int8 [N, C_in, H, W]`.
- Output: `Y_int8 [N, C_out, H, W]` (thường `C_out = C_in`).

### 2.2. Cấu trúc C2f

**File / class:**

- `ultralytics/nn/modules/block.py`
  - `class C2f`
  - `class QC2f` – biến thể có thêm logic quantization.

**Ý tưởng kiến trúc (C2f):**

```text
X (C) ── Conv ──► X1
  │
  ├─────────────────────┐
  │                     ▼
  │               [n lần Bottleneck/Conv nhỏ]
  │                     │
  └──────────── concat ─┘
              │
              └─► Conv cuối ─► Y (C_out)
```

Trong `QC2f`, các Conv được thay bằng `QConv` hoặc được chèn thêm QuantStub/DeQuantStub trong giai đoạn chuẩn bị PTQ/QAT, để đảm bảo luồng dtype đúng.

### 2.3. View INT8 và concat/skip

- Các Conv bên trong chạy INT8 (giống Conv).
- Các đường skip/concat trong block:
  - Ghép các feature map INT8 theo chiều channel.
  - Trong thiết kế chuẩn, các nhánh này dùng **cùng scale/zp** (cùng qconfig), nên concat không cần rescale.

**View phần cứng:**

`QC2f` có thể coi là **một “super-block” gồm nhiều Conv INT8 + concat**. Bạn có thể:

- Hoặc ánh xạ thành nhiều kernel conv liên tiếp.
- Hoặc “gộp” nó thành macro trong IP để giảm overhead cấu hình.

### 2.4. Pseudo-code logic

```python
class QC2f(nn.Module):
    def __init__(self, c1, c2, n=1, ...):
        # conv mở rộng/điều chỉnh kênh đầu vào
        self.cv1 = Conv(c1, c2, k=1, s=1)
        # conv cuối để gom các nhánh sau concat
        # (n+1 nhánh: 1 shortcut + n nhánh bottleneck)
        self.cv2 = Conv(c2 * (n + 1), c2, k=1, s=1)
        # n block nhỏ bên trong (dạng Bottleneck)
        self.m   = nn.ModuleList(Bottleneck(c2, c2, ...) for _ in range(n))

    def forward(self, x):
        # Nhánh chính qua Conv 1x1
        y = self.cv1(x)          # (N,c2,H,W)
        outs = [y]
        # Lặp n lần block nhỏ
        for block in self.m:
            y = block(y)         # conv/bottleneck INT8 bên trong
            outs.append(y)
        # Concat tất cả nhánh theo chiều channel
        y_cat = torch.cat(outs, dim=1)  # (N,c2*(n+1),H,W)
        # Conv cuối để đưa về số kênh c2
        return self.cv2(y_cat)         # (N,c2,H,W)
```

---

## 3. Khối `SCDown` – downsample thông minh (stride conv + skip)

### 3.1. Input / Output

Ví dụ:

- Layer 5: `INT8 [1, 64, 80, 80]` → `INT8 [1, 128, 40, 40]`
- Layer 7: `INT8 [1, 128, 40, 40]` → `INT8 [1, 256, 20, 20]`
- Layer 20: `INT8 [1, 128, 40, 40]` → `INT8 [1, 128, 20, 20]`

**Tổng quát:**

- Input: `X_int8 [N, C_in, H, W]`.
- Output: `Y_int8 [N, C_out, H/2, W/2]`.

### 3.2. Cấu trúc logic

**File / class:**

- `ultralytics/nn/modules/block.py::class SCDown`

View phổ biến (tùy biến thể nhưng ý tưởng tương tự):

```text
           ┌─ Conv(stride=2) ──────────────┐
X_int8 ────┤                               ├─► concat/add ─► Y_int8
           └─ MaxPool(stride=2) → Conv ───┘
```

- Hai nhánh đều xử lý trên INT8.
- Kết hợp nhánh bằng concat channel hoặc add.

### 3.3. View phần cứng

Trong IP:

- Có thể xếp **2 pipeline song song**:
  - Pipeline 1: Conv stride 2.
  - Pipeline 2: Pool stride 2 + Conv.
- Sau đó một khối concat/add để trộn hai output INT8.

Do các nhánh này xuất phát từ cùng input / backbone, assumption hợp lý là **cùng scale/zp**, concat/add không cần rescale phức tạp.

### 3.4. Pseudo-code logic

```python
class SCDown(nn.Module):
    def __init__(self, c1, c2, ...):
        # Nhánh 1: Conv stride=2
        self.conv1 = Conv(c1, c2 // 2, k=3, s=2)
        # Nhánh 2: MaxPool stride=2 + Conv 1x1
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = Conv(c1, c2 // 2, k=1, s=1)

    def forward(self, x):
        # Nhánh 1
        y1 = self.conv1(x)          # (N,c2/2,H/2,W/2)
        # Nhánh 2
        y2 = self.pool(x)
        y2 = self.conv2(y2)         # (N,c2/2,H/2,W/2)
        # Kết hợp 2 nhánh theo channel
        return torch.cat([y1, y2], dim=1)  # (N,c2,H/2,W/2)
```

---

## 4. Khối `SPPF` – Spatial Pyramid Pooling Fast

### 4.1. Input / Output

- Layer 9: `INT8 [1, 256, 20, 20]` → `INT8 [1, 256, 20, 20]`.

**Tổng quát:**

- Input: `X_int8 [N, C, H, W]`.
- Output: `Y_int8 [N, C, H, W]` (cùng kích thước không gian).

### 4.2. Cấu trúc

**File / class:**

- `ultralytics/nn/modules/block.py::class SPPF`

Kiến trúc điển hình:

```text
X ── Conv ──► X1
X1 ── MaxPool(k=5) ─► P1
P1 ── MaxPool(k=5) ─► P2
P2 ── MaxPool(k=5) ─► P3
Concat [X1, P1, P2, P3] theo channel ─► Conv ─► Y
```

### 4.3. View INT8

- Conv: INT8 → INT8 (theo mô hình Conv ở trên).
- MaxPool trên INT8:
  - Thao tác so sánh số nguyên → không thay đổi scale/zp.
- Concat: các tensor từ conv + pool share scale/zp chung (do cùng qconfig).

**View phần cứng:**

- Một dãy **MaxPool** + concat + Conv INT8.
- Không cần dequant nếu bạn không thay đổi activation phức tạp.

### 4.4. Pseudo-code logic

```python
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        # Conv đầu (thường giảm/chuẩn hóa kênh)
        self.cv1 = Conv(c1, c2, k=1, s=1)
        # Conv cuối sau khi concat 4 nhánh
        self.cv2 = Conv(c2 * 4, c2, k=1, s=1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x  = self.cv1(x)     # (N,c2,H,W)
        y1 = self.pool(x)    # (N,c2,H,W)
        y2 = self.pool(y1)   # (N,c2,H,W)
        y3 = self.pool(y2)   # (N,c2,H,W)
        # concat 4 feature: x, y1, y2, y3
        y_cat = torch.cat([x, y1, y2, y3], dim=1)  # (N,4c2,H,W)
        return self.cv2(y_cat)                     # (N,c2,H,W)
```

---

## 5. Khối `QPSA` – Quantized PSA (Attention)

### 5.1. Input / Output

- Layer 10: `INT8 [1, 256, 20, 20]` → `INT8 [1, 256, 20, 20]`.

### 5.2. Cấu trúc / ý nghĩa

**File / class:**

- `ultralytics/nn/modules/block.py::class QPSA`

Ý tưởng:

- Tính **attention map** cho từng vị trí/kênh:
  - Dùng vài conv/FC nhỏ để tạo trọng số attention.
  - Áp chúng lên feature (nhân/scale).

Trên INT8:

- Các phần linear (conv/FC) vẫn có thể chạy INT8.
- Các bước như **softmax**, **chuẩn hóa** có thể cần float:
  - INT8 → dequant → tính attention → (có thể) re-quant nếu muốn giữ output INT8.

### 5.3. View phần cứng

Đây là block phức tạp hơn, bạn có hai phương án:

1. **Đưa hoàn toàn sang CPU** (nếu IP không đủ phức tạp).
2. **Tối giản trong IP**, chỉ hỗ trợ:
   - Một vài conv INT8 + scale theo bảng hệ số đã clamp (fixed-point).
   - Hạn chế dùng softmax thực sự, thay bằng hàm gần đúng (LUT).

Trong PTQ hiện tại, output QPSA vẫn là `INT8 [1,256,20,20]`, nên ở biên block bạn vẫn giữ interface INT8 với scale/zp.

### 5.4. Pseudo-code logic (đơn giản hoá)

```python
class QPSA(nn.Module):
    def __init__(self, c, r=4, ...):
        # Các conv để sinh đặc trưng attention
        self.conv_theta = Conv(c, c // r, k=1, s=1)
        self.conv_phi   = Conv(c, c // r, k=1, s=1)
        self.conv_g     = Conv(c, c // r, k=1, s=1)
        self.conv_out   = Conv(c // r, c, k=1, s=1)

    def forward(self, x):
        # x: (N,C,H,W), thường INT8
        # Đơn giản hoá: dequant toàn block để tính attention
        x_f = x.dequantize()              # float32
        theta = self.conv_theta(x_f)      # (N,Cr,H,W)
        phi   = self.conv_phi(x_f)        # (N,Cr,H,W)
        g     = self.conv_g(x_f)          # (N,Cr,H,W)

        # tính attention (minh họa, không đúng code gốc 100%)
        attn = torch.softmax(theta * phi, dim=1)
        y    = attn * g
        y    = self.conv_out(y)           # (N,C,H,W)

        # Re-quant về INT8 để hoà vào pipeline INT8
        y_q = torch.quantize_per_tensor(y, x.q_scale(), x.q_zero_point(), torch.quint8)
        return y_q
```

---

## 6. Khối `Upsample` – lên mẫu nearest INT8

### 6.1. Input / Output

- Layer 11: `INT8 [1, 256, 20, 20]` → `INT8 [1, 256, 40, 40]`
- Layer 14: `INT8 [1, 128, 40, 40]` → `INT8 [1, 128, 80, 80]`

**Tổng quát:**

- Input: `X_int8 [N, C, H, W]`.
- Output: `Y_int8 [N, C, rH, rW]` với `r=2` (upsample 2×).

### 6.2. Cấu trúc

**File / hàm:**

- Dùng `torch.nn.Upsample` hoặc `F.interpolate` (nearest) được gọi trong các block YOLO.

Nearest neighbor trên INT8 đơn giản là:

```text
Y[n, c, i, j] = X[n, c, floor(i/r), floor(j/r)]
```

Không nhân/biến đổi số liệu → **scale và zero-point giữ nguyên**.

**View phần cứng:**

- Khối “upscale” chỉ cần **bộ lập chỉ số đọc lại** từ SRAM, không cần ALU phức tạp.

### 6.3. Pseudo-code logic

```python
class Upsample(nn.Module):
    def __init__(self, scale_factor=2):
        self.scale_factor = scale_factor

    def forward(self, x):
        # nearest neighbor upsample
        return F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
```

---

## 7. Khối `QConcat` – ghép channel các feature map INT8

### 7.1. Input / Output

Ví dụ:

- Layer 12: `[1,256,40,40]` + `[1,128,40,40]` → `[1,384,40,40]`
- Layer 15: `[1,128,80,80]` + `[1,64,80,80]` → `[1,192,80,80]`
- Layer 18, 21: tương tự.

**Tổng quát:**

- Inputs:
  - `A_int8: [N, C1, H, W]` với `(s_A, zp_A)`
  - `B_int8: [N, C2, H, W]` với `(s_B, zp_B)`
- Output:
  - `Y_int8: [N, C1+C2, H, W]` với `(s_Y, zp_Y)`

### 7.2. Scale / zero-point khi concat

Lý tưởng:

- `s_A = s_B = s_Y`, `zp_A = zp_B = zp_Y` → concat chỉ là ghép mảng.

Trong thực tế Ultralytics PTQ:

- Các nhánh concat được sinh ra từ cùng backbone/task và cùng qconfig → scale/zp thường **đã đồng bộ**.

Nếu không đồng bộ (trường hợp tổng quát):

- Cần rescale một nhánh trước khi concat:
  - `B_real = s_B*(B_int8 - zp_B)`.
  - Chọn scale mới `s_Y, zp_Y`.
  - `B_int8' = quantize(B_real, s_Y, zp_Y)`.
  - Concat `A_int8` và `B_int8'` cùng scale/zp.

**File / class:**

- `ultralytics/nn/modules/conv.py::class QConcat`

**View phần cứng:**

- Nếu bạn áp dụng **chiến lược “cùng scale cho mọi nhánh concat”**, QConcat là khối **rất rẻ**: chỉ là copy/gép dữ liệu theo chiều channel.

### 7.3. Pseudo-code logic

```python
class QConcat(nn.Module):
    def __init__(self, dim=1):
        self.dim = dim  # thường = 1 (channel)

    def forward(self, xs):
        # xs: list các tensor [N,C_i,H,W] cùng H,W và (lý tưởng) cùng scale/zp
        return torch.cat(xs, dim=self.dim)
```

---

## 8. Khối `QC2fCIB` – C2f dành cho head CIB

### 8.1. Input / Output

- Layer 22: `INT8 [1, 384, 20, 20]` → `INT8 [1, 256, 20, 20]`

**Tổng quát:**

- Input: `X_int8 [N, C_in, H, W]` (C_in ~ 384).
- Output: `Y_int8 [N, C_out, H, W]` (C_out ~ 256).

### 8.2. Cấu trúc

**File / class:**

- `ultralytics/nn/modules/block.py::class QC2fCIB`

Ý tưởng:

- Giống `QC2f` (nhiều conv nhỏ + concat + conv cuối), nhưng:
  - Tối ưu cho CIB head của YOLOv10.
  - Điều chỉnh C_out và cách phân chia kênh hợp lý cho head.

**View phần cứng:**

- Một macro-block gồm **vài conv INT8 + concat + conv cuối**, giống `QC2f` nhưng với tham số kênh/nhánh được design cho tầng cuối neck.

### 8.3. Pseudo-code logic

```python
class QC2fCIB(nn.Module):
    def __init__(self, c1, c2, n=1, ...):
        # Conv đầu đưa kênh về c2
        self.cv1 = Conv(c1, c2, k=1, s=1)
        # n block nhỏ bên trong
        self.m   = nn.ModuleList(Bottleneck(c2, c2, ...) for _ in range(n))
        # Conv cuối sau concat (n+1 nhánh)
        self.cv2 = Conv(c2 * (n + 1), c2, k=1, s=1)

    def forward(self, x):
        y = self.cv1(x)        # (N,c2,H,W)
        outs = [y]
        for block in self.m:
            y = block(y)       # conv/bottleneck INT8
            outs.append(y)
        y_cat = torch.cat(outs, dim=1)  # (N,c2*(n+1),H,W)
        return self.cv2(y_cat)          # (N,c2,H,W)
```

---

## 9. Tổng kết cho thiết kế bộ tăng tốc

Những khối mà phần cứng của bạn cần hỗ trợ (ở mức primitive hoặc macro):

1. **Conv INT8 + BN fuse + activation**  
   - Cơ sở cho hầu hết các block khác.
2. **QC2f / QC2fCIB**  
   - Nhiều Conv INT8 + concat/skip nội bộ.
3. **SCDown**  
   - 2 nhánh downsample INT8 + concat/add.
4. **SPPF**  
   - Conv + MaxPool INT8 đa tỉ lệ + concat + Conv.
5. **QPSA** (attention)  
   - Có thể chạy HW hoặc CPU tuỳ độ phức tạp.
6. **Upsample (nearest)**  
   - Copy/replicate INT8 – khối truy xuất bộ nhớ.
7. **QConcat**  
   - Ghép channel trên INT8 với giả thiết cùng scale/zp.

Tất cả các khối trên **giữ interface INT8 + scale/zero-point** giữa các block. CPU chịu trách nhiệm:
- PREPROCESS + QUANTIZE đầu vào.
- (Tùy thiết kế) dequant feature cuối để chạy Qv10Detect head + POSTPROCESS (NMS).  
Do vậy, tài liệu này có thể được dùng trực tiếp làm **spec khối tính toán** cho bộ tăng tốc phần cứng của bạn.



---
---

<a id='phần-i4---đồ-thị-phụ-thuộc-giữa-các-layer'></a>

# PHẦN I.4 — Đồ Thị Phụ Thuộc Giữa Các Layer
> Nguồn: `SW_KLTN/MODEL_LAYER_DEPENDENCIES.md`

---

## Phụ thuộc giữa các layer trong qYOLOv10n PTQ

Tài liệu này tổng hợp từ:

- `MODEL_FORWARD_FLOW.md` – flow tổng thể.
- `MODEL_LAYERS_INT8_FLOW.md` – dtype & shape từng layer.
- `MODEL_BLOCKS_INT8_DETAIL.md` – cấu trúc logic các block.

Mục tiêu:

- Chỉ rõ **layer nào chỉ dùng output của layer liền trước**,
- Và **layer nào còn dùng thêm kết quả trung gian (skip / concat)** từ các layer cũ hơn.

Ký hiệu:

- `Lk` = layer số k trong `MODEL_LAYERS_INT8_FLOW.md`.
- `in(Lk)` = nguồn dữ liệu đầu vào cho layer k.
- `prev` = chỉ lấy output của layer liền trước.
- `[a, b, ...]` = lấy thêm output từ các layer `La, Lb, ...`.

> Ghi chú: Con số chính xác `a, b` có thể thay đổi nếu parse trực tiếp từ `parse_model`, nhưng **mô hình phụ thuộc** (chỉ prev vs prev + skip) là chính xác cho qYOLOv10n.

---

## 1. Các layer **chỉ lấy output của layer liền trước**

Nhóm này tương đối “dễ” cho phần cứng: **input = output của layer trước đó**, không cần truy xuất thêm feature cũ.

### 1.1. Backbone “xuống” – conv + C2f + downsample

- `L0: Conv`
  - `in(L0) = prev` (đầu vào model sau QuantStub / quantize trên CPU).
- `L1: Conv`
  - `in(L1) = L0`.
- `L2: QC2f`
  - `in(L2) = L1`.
- `L3: Conv`
  - `in(L3) = L2`.
- `L4: QC2f`
  - `in(L4) = L3`.
- `L5: SCDown`
  - `in(L5) = L4`.
- `L6: QC2f`
  - `in(L6) = L5`.
- `L7: SCDown`
  - `in(L7) = L6`.
- `L8: QC2f`
  - `in(L8) = L7`.

### 1.2. Đoạn cuối backbone + neck “giữa”

- `L9: SPPF`
  - `in(L9) = L8`.
- `L10: QPSA`
  - `in(L10) = L9`.
- `L11: Upsample`
  - `in(L11) = L10`.

### 1.3. Neck sau các concat đầu tiên

- `L13: QC2f`
  - `in(L13) = L12` (sau QConcat đã ghép feature).
- `L14: Upsample`
  - `in(L14) = L13`.
- `L16: QC2f`
  - `in(L16) = L15`.
- `L17: Conv`
  - `in(L17) = L16`.
- `L19: QC2f`
  - `in(L19) = L18`.
- `L20: SCDown`
  - `in(L20) = L19`.
- `L22: QC2fCIB`
  - `in(L22) = L21`.

> **Kết luận nhóm 1:**  
> Các layer [0,1,2,3,4,5,6,7,8,9,10,11,13,14,16,17,19,20,22] **chỉ phụ thuộc vào output trực tiếp của layer ngay trước đó**.  
> Đối với bộ tăng tốc, chúng là các block “tuyến tính”: không cần đọc thêm feature map cũ ngoài `L(k-1)`.

---

## 2. Các layer **lấy thêm kết quả trung gian** (skip / concat)

Các layer sau **không chỉ dùng output liền trước**, mà còn lấy thêm feature từ các layer cũ hơn (skip connection), hoặc từ nhiều nhánh khác nhau (FPN/PAN).

### 2.1. `QConcat` trong FPN/PAN

Đây là những nơi **chắc chắn có skip**:

- `L12: QConcat`
  - Vai trò: ghép feature **từ nhánh upsample (40×40)** với **skip từ backbone (40×40)**.
  - `in(L12) = [L11, L6]` (mô hình điển hình):
    - Nhánh 1: output của `L11` (Upsample từ deep feature 20×20 → 40×40).
    - Nhánh 2: output của `L6` (backbone ở tỉ lệ 1/16, 40×40).

- `L15: QConcat`
  - Vai trò: ghép feature **upsampled từ tầng trung (40×40 → 80×80)** với **skip từ backbone (80×80)**.
  - `in(L15) = [L14, L4]`:
    - Nhánh 1: output `L14` (Upsample từ 40×40 → 80×80).
    - Nhánh 2: output `L4` (backbone 1/8, 80×80).

- `L18: QConcat`
  - Vai trò: ghép feature giảm tỉ lệ từ high-res với feature trung bình.
  - `in(L18) = [L17, L13]`:
    - Nhánh 1: `L17` (Conv stride 2 từ 80×80 → 40×40).
    - Nhánh 2: `L13` (QC2f ở 40×40).

- `L21: QConcat`
  - Vai trò: ghép feature từ đường “đi xuống” với feature deep nhất.
  - `in(L21) = [L20, L8]`:
    - Nhánh 1: `L20` (SCDown 40×40 → 20×20).
    - Nhánh 2: `L8` (QC2f deep 20×20).

**Nhận xét quan trọng:**

- `QConcat` **luôn nhận từ nhiều layer** (ít nhất hai), vì thế:
  - Bộ tăng tốc phải **lưu lại** các feature map `L6`, `L4`, `L13`, `L8` nếu sau này cần concat.
  - Không thể đơn giản xử lý streaming “bỏ qua” các output đó.


---
---

# ════════════════════════════════════════════════════════════════
# PHẦN II — NGHIÊN CỨU MAPPING PHẦN CỨNG
# ════════════════════════════════════════════════════════════════

---

<a id='phần-ii1---mapping-phần-cứng'></a>

# PHẦN II.1 — Mapping Khối Tính Toán lên Phần Cứng qua Primitive Set
> Nguồn: `SW_KLTN/HW_MAPPING_RESEARCH.md`

---

# Nghiên cứu Mapping Khối Tính Toán YOLOv10n lên Phần Cứng qua Primitive Set

> **Phạm vi**: Layer 0–22 (Backbone + Neck) của qYOLOv10n PTQ  
> **Mục tiêu**: Xây dựng bản đồ chi tiết từ model block → hardware primitive → RTL module  
> **Phiên bản**: Phase 0 Freeze Spec

---

## 1. Tổng quan kiến trúc phân cấp

```
┌─────────────────────────────────────────────────────────────────┐
│                    qYOLOv10n Model Block                        │
│  (Conv, QC2f, SCDown, SPPF, QPSA, Upsample, QConcat, QC2fCIB)  │
└──────────────────────────┬──────────────────────────────────────┘
                           │  decompose
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Primitive Set                              │
│  RS_DENSE_3x3 | OS_1x1 | DW_3x3 | MAXPOOL_5x5 | MOVE          │
│  CONCAT | UPSAMPLE_NEAREST | EWISE_ADD | DW_7x7_MULTIPASS      │
│  GEMM_ATTN_BASIC                                                │
└──────────────────────────┬──────────────────────────────────────┘
                           │  map to RTL
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RTL Hardware                               │
│  PE Cluster | GLB Banks | Router | PPU | Swizzle Engine        │
│  Descriptor Stack | Tile FSM | Barrier Manager                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Primitive Set chính thức & Đặc tả

### 2.1. Bảng Primitive Matrix

| Primitive ID | Tên chuẩn | Input Shape | Output Shape | Stride | Padding | Quant Domain | PPU | Dùng cho |
|---|---|---|---|---|---|---|---|---|
| P0 | RS_DENSE_3x3 | [H,W,Cin] | [Hout,Wout,Cout] | 1,2 | same | scale_in×scale_w→scale_out | Yes | Conv layers 0,1,3,17 |
| P1 | OS_1x1 | [H,W,Cin] | [H,W,Cout] | 1 | 0 | scale_in×scale_w→scale_out | Yes | Projection trong C2f, SPPF |
| P2 | DW_3x3 | [H,W,C] | [Hout,Wout,C] | 1,2 | same | per-channel weight | Yes | SCDown nhánh DW |
| P3 | MAXPOOL_5x5 | [H,W,C] | [H,W,C] | 1 | 2 | giữ nguyên (INT8 compare) | No | SPPF (×3 lặp) |
| P4 | MOVE | [H,W,C] | [H,W,C] | - | - | giữ nguyên | No | Skip lưu buffer |
| P5 | CONCAT | [H,W,C1],[H,W,C2] | [H,W,C1+C2] | - | - | common-domain requant | No | FPN/PAN neck |
| P6 | UPSAMPLE_NEAREST | [H,W,C] | [2H,2W,C] | - | - | giữ nguyên scale/zp | No | Neck upsample |
| P7 | EWISE_ADD | [H,W,C],[H,W,C] | [H,W,C] | - | - | common-domain requant | Yes | Residual (dự phòng) |
| P8 | DW_7x7_MULTIPASS | [H,W,C] | [H,W,C] | 1 | 3 | per-channel, bias ở pass cuối | Yes | QC2fCIB large kernel |
| P9 | GEMM_ATTN_BASIC | [N,HW,C] | [N,HW,C] | - | - | INT8 GEMM→requant | Yes | QPSA attention |

### 2.2. Quantization Rule cho từng Primitive

```
┌──────────────────┬──────────────────────────────────────────────────────────────┐
│ Primitive        │ Quantization Rule                                            │
├──────────────────┼──────────────────────────────────────────────────────────────┤
│ RS_DENSE_3x3     │ acc[i] = Σ (x-zp_x)(w-zp_w)·scale_x·scale_w + bias        │
│                  │ y_int8 = clamp(round(acc/scale_y + zp_y), -128, 127)        │
├──────────────────┼──────────────────────────────────────────────────────────────┤
│ OS_1x1           │ Tương tự RS_DENSE_3x3 với kernel 1×1                        │
├──────────────────┼──────────────────────────────────────────────────────────────┤
│ DW_3x3           │ Per-channel weight: scale_w[cout] riêng từng channel        │
│                  │ bias[cout] = fused_bn với scale_bias=scale_x·scale_w[cout]  │
├──────────────────┼──────────────────────────────────────────────────────────────┤
│ MAXPOOL_5x5      │ max(x_int8) → không đổi scale/zp, chỉ so sánh số nguyên   │
├──────────────────┼──────────────────────────────────────────────────────────────┤
│ CONCAT           │ Nếu scale A ≠ scale B: requant nhánh có scale nhỏ hơn       │
│                  │ về common scale trước khi concat                             │
├──────────────────┼──────────────────────────────────────────────────────────────┤
│ UPSAMPLE_NEAREST │ y[i,j] = x[i//2, j//2] → giữ nguyên scale/zp              │
├──────────────────┼──────────────────────────────────────────────────────────────┤
│ EWISE_ADD        │ align cả hai nhánh về common_scale trước khi add            │
│                  │ common_scale = max(scale_A, scale_B)                        │
├──────────────────┼──────────────────────────────────────────────────────────────┤
│ DW_7x7_MULTIPASS │ Pass 1,2: acc lưu PSUM; Pass cuối: +bias → requant → INT8 │
├──────────────────┼──────────────────────────────────────────────────────────────┤
│ GEMM_ATTN_BASIC  │ Q,K,V projection (OS_1x1) → QK^T/sqrt(d) → softmax(INT8-  │
│                  │ approx) → ×V → output projection                           │
└──────────────────┴──────────────────────────────────────────────────────────────┘
```

---

## 3. Layer-to-Primitive Mapping chi tiết (Layer 0–22)

### 3.1. Backbone – Đường xuống (Layer 0–10)

#### Layer 0 – `Conv` [3,640,640] → [16,320,320]

```
Block: Conv (k=3, s=2, BN_fuse, SiLU)
─────────────────────────────────────────────
Primitive:  RS_DENSE_3x3(stride=2)
  Input:    X_int8[1,3,640,640]
  Weight:   W_int8[3,3,3,16], B_int32[16]
  Output:   Y_int8[1,16,320,320]
  scale_in: 0.00392 (= 1/255), zp_in: 0
  
Execution steps:
  1. MAC: acc[cout] = Σ_{kh,kw,cin} (x-zp_x)(w-zp_w)   →  int32 psum
  2. Bias: acc += B_int32[cout]
  3. Requant: y_raw = round(acc * M) >> shift             →  int32 clip
     where M = scale_in * scale_w[cout] / scale_out
  4. Activation: SiLU approximation (LUT) hoặc dequant→SiLU→requant
  5. Clamp: y_int8 = clamp(y_raw + zp_out, -128, 127)
  
Hardware path: GLB_IN → window_gen_3x3 → PE_MAC × LANES=16 → PSUM_buf → PPU → GLB_OUT
Tile size: tile_h × tile_w × Cin_chunk → Cout_chunk per tile
Memory: GLB_bank_input (h mod 3), bank_output (out_row mod 4)
```

#### Layer 1 – `Conv` [16,320,320] → [32,160,160]

```
Primitive:  RS_DENSE_3x3(stride=2)
  Input:    F0_out: INT8[1,16,320,320]
  Weight:   W_int8[3,3,16,32], B_int32[32]
  Output:   INT8[1,32,160,160]

Notes: Giống layer 0, chỉ khác Cin=16, Cout=32, H/W=320→160
```

#### Layer 2 – `QC2f` [32,160,160] → [32,160,160]

```
Block: QC2f (n=1 bottleneck)
─────────────────────────────────────────────────────────────
Primitive sequence (OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1):

  Step 1: cv1 = OS_1x1
    Input:  F1_out [1,32,160,160]
    Weight: W[1,1,32,32], B[32]
    Output: X1_int8 [1,32,160,160]   ← nhánh split

  Step 2: bottleneck_cv1 = RS_DENSE_3x3 (internal)
    Input:  X1_int8 [1,32,160,160]  (low-half channels = 16ch actual)
    Weight: W[3,3,16,16]
    Output: Y_tmp [1,16,160,160]

  Step 3: bottleneck_cv2 = RS_DENSE_3x3 (internal)
    Input:  Y_tmp [1,16,160,160]
    Output: Y_branch [1,16,160,160]

  Step 4: CONCAT (Y_branch, X1_nhánh_giữ)
    Inputs: [1,16,160,160] + [1,16,160,160] = [1,32,160,160]
    → common-domain requant nếu scale khác nhau

  Step 5: cv2 = OS_1x1
    Input:  [1,32,160,160]  (sau concat)
    → OS_1x1 gom về C_out=32

  Output: F2_out [1,32,160,160]

Intermediate buffers cần lưu:
  - X1_int8 (nhánh skip nội bộ C2f)
  - Y_tmp  (kết quả giữa chừng trong bottleneck)
```

#### Layer 3 – `Conv` [32,160,160] → [64,80,80]

```
Primitive: RS_DENSE_3x3(stride=2), Cin=32, Cout=64
```

#### Layer 4 – `QC2f` [64,80,80] → [64,80,80]

```
Tương tự Layer 2, scale tăng lên Cin=Cout=64
Primitive sequence: OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1
Output: F4_out [1,64,80,80]   ⚠️ CẦN LƯU - skip đến L15
```

#### Layer 5 – `SCDown` [64,80,80] → [128,40,40]

```
Block: SCDown (Spatial Channel Downsample)
─────────────────────────────────────────────────────────────
Primitive sequence (OS_1x1 + DW_3x3(stride=2)):

  Phương án hardware-friendly (theo spec freeze):
  
  Step 1: OS_1x1  ← channel adjustment
    Input:  F4_out [1,64,80,80]
    Output: tmp [1,128,80,80]  (hoặc [1,64,80,80] rồi split)

  Step 2: DW_3x3(stride=2)  ← spatial downsample, per-channel
    Input:  tmp [1,...,80,80]
    Output: F5_out [1,128,40,40]

Notes về SCDown trong YOLOv10:
  - Thực tế có 2 nhánh: nhánh conv3x3 s2 + nhánh DW_3x3 s2 sau OS_1x1
  - 2 nhánh xử lý C/2 kênh mỗi nhánh, sau đó CONCAT
  
Primitive sequence đầy đủ:
  Branch A: OS_1x1(Cin→Cout/2) → DW_3x3(s2)
  Branch B: OS_1x1(Cin→Cout/2) → DW_3x3(s2)
  CONCAT(A, B) → F5_out [1,128,40,40]
```

#### Layer 6 – `QC2f` [128,40,40] → [128,40,40]

```
Primitive sequence: OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1
Output: F6_out [1,128,40,40]   ⚠️ CẦN LƯU - skip đến L12
```

#### Layer 7 – `SCDown` [128,40,40] → [256,20,20]

```
Tương tự Layer 5, scale lên Cin=128, Cout=256
```

#### Layer 8 – `QC2f` [256,20,20] → [256,20,20]

```
Primitive sequence: OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1
Output: F8_out [1,256,20,20]   ⚠️ CẦN LƯU - skip đến L21
```

#### Layer 9 – `SPPF` [256,20,20] → [256,20,20]

```
Block: SPPF (Spatial Pyramid Pooling Fast)
─────────────────────────────────────────────────────────────
Primitive sequence:
  Step 1: cv1 = OS_1x1
    Input:  F8_out [1,256,20,20]
    Output: X1 [1,128,20,20]   (giảm kênh xuống c2=128)

  Step 2: MAXPOOL_5x5 × 3 (lặp)
    P1 = MAXPOOL_5x5(X1)  → [1,128,20,20]
    P2 = MAXPOOL_5x5(P1)  → [1,128,20,20]
    P3 = MAXPOOL_5x5(P2)  → [1,128,20,20]

  Step 3: CONCAT(X1, P1, P2, P3)
    [1,128], [1,128], [1,128], [1,128] → [1,512,20,20]
    ⚠️ Tất cả 4 nhánh từ cùng qconfig → scale/zp tương tự → concat đơn giản

  Step 4: cv2 = OS_1x1
    Input:  [1,512,20,20]
    Output: F9_out [1,256,20,20]

Lưu ý hardware:
  - MAXPOOL_5x5 chỉ so sánh INT8, không cần PPU hay requant
  - X1, P1, P2 phải được buffer đồng thời trước khi CONCAT
```

#### Layer 10 – `QPSA` [256,20,20] → [256,20,20]

```
Block: QPSA (Quantized Pixel/Position Sensitive Attention)
─────────────────────────────────────────────────────────────
Primitive sequence (OS_1x1 + GEMM_ATTN_BASIC + OS_1x1):

  Step 1: split = OS_1x1
    Input:  F9_out [1,256,20,20]
    → split thành nhánh attn và nhánh pass-through
    QAttn_in [1,128,20,20], Pass [1,128,20,20]

  Step 2: GEMM_ATTN_BASIC
    Input:  QAttn_in [1,128,20,20]
    → reshape: [1, 400, 128]  (HW=20×20=400, C=128)
    → Q proj OS_1x1 → [1,400,64]
    → K proj OS_1x1 → [1,400,64]
    → V proj OS_1x1 → [1,400,128]
    → Attn = Q×K^T / sqrt(64)  →  [1,400,400]  (INT8 GEMM)
    → Attn_soft = softmax_approx(Attn)  →  requant → INT8
    → Out = Attn_soft × V →  [1,400,128]
    → reshape back: [1,128,20,20]

  Step 3: CONCAT / EWISE_ADD (merge với Pass)
    [1,128,20,20] + [1,128,20,20] → concat → [1,256,20,20]

  Step 4: output proj = OS_1x1
    [1,256,20,20] → [1,256,20,20]

  Output: F10_out [1,256,20,20]

⚠️ GEMM_ATTN_BASIC là primitive phức tạp nhất:
   - GEMM thực hiện bằng PE MAC với output accumulate INT32
   - softmax approximation: dùng INT8 LUT hoặc log-sum-exp phân đoạn
   - Tại stage 20×20, tensor nhỏ: throughput không phải ưu tiên số 1
```

---

### 3.2. Neck – FPN (Layer 11–16)

#### Layer 11 – `Upsample` [256,20,20] → [256,40,40]

```
Primitive: UPSAMPLE_NEAREST(scale=2)
  Input:   F10_out [1,256,20,20]
  Output:  F11_out [1,256,40,40]
  
Rule: y[h,w,c] = x[h//2, w//2, c]
  → scale_out = scale_in, zp_out = zp_in  (không thay đổi)
  
Hardware path: swizzle_engine / tensor_post_engine
  - Phát địa chỉ source theo pattern ×2 repetition
  - Không cần PE MAC hay PPU
  - Chỉ là router/DMA với address remapping
```

#### Layer 12 – `QConcat` [256,40,40]+[128,40,40] → [384,40,40]

```
Primitive: CONCAT
  Input A: F11_out [1,256,40,40]  (scale_A, zp_A) từ upsample
  Input B: F6_out  [1,128,40,40]  (scale_B, zp_B) từ backbone  ⚠️ SKIP DEPENDENCY
  Output:  F12_out [1,384,40,40]  (scale_Y, zp_Y)

Common-domain alignment:
  if scale_A ≠ scale_B:
    Chọn scale_Y = scale_B (thường scale backbone là reference)
    Requant A → scale_Y: A' = round((A - zp_A) × (scale_A/scale_Y)) + zp_Y
    Concat(A', B) theo chiều channel
  else:
    Concat trực tiếp

Hardware path:
  - Đọc F11_out từ GLB bank A
  - Đọc F6_out (đã được MOVE/HOLD từ bước trước) từ GLB bank B
  - Router chuyển luân phiên: 256 channel A, rồi 128 channel B → concatenated output
  
⚠️ F6_out phải được lưu trong GLB (HOLD_SKIP) từ thời điểm compute layer 6
   cho đến khi layer 12 sẵn sàng consume nó.
```

#### Layer 13 – `QC2f` [384,40,40] → [128,40,40]

```
Primitive sequence: OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1
  Input Cin=384, Output Cout=128
  Đây là tầng FPN feature trung bình (P4 vùng)
Output: F13_out [1,128,40,40]   ⚠️ CẦN LƯU - skip đến L18
```

#### Layer 14 – `Upsample` [128,40,40] → [128,80,80]

```
Primitive: UPSAMPLE_NEAREST(scale=2)
  Input:  F13_out [1,128,40,40]
  Output: F14_out [1,128,80,80]
  (scale, zp giữ nguyên)
```

#### Layer 15 – `QConcat` [128,80,80]+[64,80,80] → [192,80,80]

```
Primitive: CONCAT
  Input A: F14_out [1,128,80,80] từ upsample
  Input B: F4_out  [1,64,80,80]  từ backbone  ⚠️ SKIP DEPENDENCY
  Output:  F15_out [1,192,80,80]

⚠️ F4_out phải được lưu HOLD_SKIP từ layer 4 → đây là skip dài nhất trong flow
```

#### Layer 16 – `QC2f` [192,80,80] → [64,80,80]

```
Primitive sequence: OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1
  Cin=192, Cout=64, H=W=80  ← spatial size lớn nhất
  → Đây là P3 feature (80×80) → OUTPUT P3
Output: F16_out = P3_int8 [1,64,80,80]  ✅ OUTPUT P3
```

---

### 3.3. Neck – PAN (Layer 17–22)

#### Layer 17 – `Conv` [64,80,80] → [64,40,40]

```
Primitive: RS_DENSE_3x3(stride=2)
  Cin=64, Cout=64, H=80→40
  Đây là nhánh downsample để kết nối P3→P4 trong PAN
```

#### Layer 18 – `QConcat` [64,40,40]+[128,40,40] → [192,40,40]

```
Primitive: CONCAT
  Input A: F17_out [1,64,40,40]  từ conv downsample
  Input B: F13_out [1,128,40,40] từ tầng trung FPN  ⚠️ SKIP DEPENDENCY
  Output:  F18_out [1,192,40,40]

⚠️ F13_out phải được lưu HOLD_SKIP từ layer 13 → consume tại layer 18
```

#### Layer 19 – `QC2f` [192,40,40] → [128,40,40]

```
Primitive sequence: OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1
  Cin=192, Cout=128
Output: F19_out = P4_int8 [1,128,40,40]  ✅ OUTPUT P4
```

#### Layer 20 – `SCDown` [128,40,40] → [128,20,20]

```
Primitive sequence: OS_1x1 + DW_3x3(stride=2)
  Cin=Cout=128, downsample 40→20
```

#### Layer 21 – `QConcat` [128,20,20]+[256,20,20] → [384,20,20]

```
Primitive: CONCAT
  Input A: F20_out [1,128,20,20] từ SCDown
  Input B: F8_out  [1,256,20,20] từ deep backbone  ⚠️ SKIP DEPENDENCY
  Output:  F21_out [1,384,20,20]

⚠️ F8_out phải được lưu HOLD_SKIP từ layer 8 → đây là skip dài nhất toàn flow
   Giữ đến tận layer 21 → cần allocation GLB space xuyên suốt
```

#### Layer 22 – `QC2fCIB` [384,20,20] → [256,20,20]

```
Block: QC2fCIB (C2f with CIB - large kernel)
─────────────────────────────────────────────────────────────
Primitive sequence:
  OS_1x1 + DW_7x7_MULTIPASS + OS_1x1 + (RS_DENSE_3x3 | DW_3x3) + CONCAT + OS_1x1

  Step 1: cv1 = OS_1x1
    Input:  [1,384,20,20] → [1,256,20,20]

  Step 2: CIB bottleneck:
    a. DW_7x7_MULTIPASS
       Kernel 7×7 chia làm 3 pass: [3 rows] + [3 rows] + [1 row]
       Pass 1,2: lưu PSUM namespace
       Pass 3 (last_pass): PSUM + bias → requant →INT8
       
    b. OS_1x1 compression
    
    c. (Optional) RS_DENSE_3x3 thêm

  Step 3: CONCAT (nhánh CIB + skip) theo chiều channel

  Step 4: cv2 = OS_1x1
    Output: P5_int8 [1,256,20,20]  ✅ OUTPUT P5

Output: F22_out = P5_int8 [1,256,20,20]

⚠️ DW_7x7_MULTIPASS cần trace pass-by-pass cho RTL verification
```

---

## 4. Dependency Graph và Buffer Management

### 4.1. Sơ đồ phụ thuộc Skip Connection

```
L0 → L1 → L2 → L3 → L4 ─────────────────────────────────────────────► L15 (concat)
                      │                                                   ▲
                      └─► L5 → L6 ──────────────────────────────────► L12 (concat)
                                 │                                       ▲
                                 └─► L7 → L8 ────────────────────────► L21 (concat)
                                           │                             ▲
                                           └─► L9 → L10 → L11 ─────────┘
                                                              │
                                                              └─► L12 (nhánh upsample)
                                                                    │
                                                                    ▼
                                                              L13 ─────────────────► L18 (concat)
                                                               │                      ▲
                                                               └─► L14 → L15          │
                                                                           │           │
                                                                           ▼           │
P3 = L16 ◄───────────────────────────────────────────────────── L16          L17 ────┘
P4 = L19 ◄── L18 ← L13(skip) + L17
P5 = L22 ◄── L22 ← L21 ← L20 + L8(skip)
```

### 4.2. Bảng HOLD_SKIP Buffer Requirements

| Skip tensor | Sinh ra tại | Tiêu thụ tại | Khoảng cách | Kích thước buffer |
|---|---|---|---|---|
| F4_out | L4 | L15 | 11 layer | INT8 [1,64,80,80] = 409,600 bytes |
| F6_out | L6 | L12 | 6 layer | INT8 [1,128,40,40] = 204,800 bytes |
| F8_out | L8 | L21 | 13 layer | INT8 [1,256,20,20] = 102,400 bytes |
| F13_out | L13 | L18 | 5 layer | INT8 [1,128,40,40] = 204,800 bytes |

**Tổng GLB buffer cần dự trữ cho SKIP**: ~921,600 bytes (~900 KB)

### 4.3. Barrier Management

```
barrier_L12: wait (L11_done AND L6_hold_ready) → release L12_start
barrier_L15: wait (L14_done AND L4_hold_ready) → release L15_start
barrier_L18: wait (L17_done AND L13_hold_ready) → release L18_start
barrier_L21: wait (L20_done AND L8_hold_ready) → release L21_start
```

---

## 5. Hardware Datapath theo Primitive

### 5.1. RS_DENSE_3x3 / OS_1x1 – Compute Path

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Compute Pipeline                                 │
│                                                                      │
│  GLB_INPUT ──► window_gen_3x3/1x1                                   │
│                    │                                                 │
│                    ▼                                                 │
│  GLB_WEIGHT ──► PE_LANE_MAC × 16 (LANES=16)                        │
│                    │ INT8×INT8 → INT32 MAC                          │
│                    ▼                                                 │
│               column_reduce (Cin tích lũy, horizontal)              │
│                    │                                                 │
│                    ▼                                                 │
│               PSUM_BUF → last_pass?                                 │
│                   No: lưu PSUM namespace                            │
│                   Yes: → PPU                                        │
│                         ├─ bias_add (INT32)                         │
│                         ├─ requant (scale_mul × shift)              │
│                         ├─ activation (SiLU LUT / approx)          │
│                         └─ clamp → INT8                             │
│                              │                                      │
│                              ▼                                      │
│               GLB_OUTPUT (bank_output = out_row mod 4)             │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.2. DW_3x3 – Depthwise Path

```
Khác biệt so với RS_DENSE_3x3:
- groups = Cin → mỗi channel xử lý độc lập
- weight shape: [3,3,1,Cout] với Cout = Cin
- Mỗi PE_LANE chỉ MAC trong 1 channel → không cần column_reduce
- Per-channel bias và scale_w[c] riêng từng channel
- last_pass luôn = true (không accumulate cross-channel)
```

### 5.3. MAXPOOL_5x5 – Pool Path

```
window_gen_5x5 ──► max_compare_tree (INT8 max, 25 inputs)
                        │
                        ▼ (không qua PPU)
                   GLB_OUTPUT (cùng scale/zp với input)

- Kernel 5×5, padding=2, stride=1
- Lặp 3 lần trong SPPF → output của lần trước là input lần sau
- GLB cần buffer P1, P2 song song với X1 để CONCAT sau
```

### 5.4. UPSAMPLE_NEAREST – Tensor Post Path

```
Không qua PE. Thực hiện qua tensor_post_engine hoặc swizzle_engine.

Address remapping:
  src_addr(c, h, w) → dst_addr(c, h//2, w//2)  ← đọc cùng src addr 4 lần (2×2 block)

Hoặc DMA với stride pattern:
  dst[c][2h  ][2w  ] = src[c][h][w]
  dst[c][2h  ][2w+1] = src[c][h][w]
  dst[c][2h+1][2w  ] = src[c][h][w]
  dst[c][2h+1][2w+1] = src[c][h][w]
```

### 5.5. CONCAT – Router Path

```
Input A: [H,W,C_A] với (scale_A, zp_A)
Input B: [H,W,C_B] với (scale_B, zp_B)

Case 1: scale_A = scale_B (ideal)
  → router_cluster chuyển A_channels, rồi B_channels → output interleaved by channel

Case 2: scale_A ≠ scale_B (cần requant)
  → nhánh có scale khác phải qua PPU mini-requant trước khi vào concat
  → common_scale = được chọn offline (thường là scale của backbone branch)
  → requant_params được kiểm tra tại LAYER_DESC
```

### 5.6. DW_7x7_MULTIPASS – Multi-pass Pipeline

```
Kernel 7×7 chia làm passes: [rows 0-2] + [rows 3-5] + [row 6]
(đảm bảo tương đương kết quả monolithic 7×7)

Pass 1 (rows 0-2):
  window_gen partial (3 kernel rows) + DW_MAC → PSUM_namespace (INT32)
  last_kernel = 0, last_pass = 0

Pass 2 (rows 3-5):
  window_gen partial (3 kernel rows) + DW_MAC → accumulate PSUM_namespace
  last_kernel = 0, last_pass = 0

Pass 3 (row 6, last_kernel=1, last_pass=1):
  window_gen partial (1 kernel row) + DW_MAC → ACC to PSUM
  → PPU: bias_add + requant + clamp → INT8 → ACT_namespace

Debug trace: xuất PSUM sau mỗi pass để verify golden vs RTL
```

### 5.7. GEMM_ATTN_BASIC – Attention Path

```
QPSA tại 20×20 → HW_seq_len = 400 (nhỏ, phù hợp hardware)

Q = OS_1x1(X_attn)   → [1,Hq,20*20]
K = OS_1x1(X_attn)   → [1,Hk,20*20]
V = OS_1x1(X_attn)   → [1,Hv,20*20]

Attn = Q × K^T  → [1,400,400]  INT8 GEMM
  - Dùng PE MAC với K^T được transpose trong GLB
  - Scale: M_attn = scale_Q × scale_K / scale_Attn

Attn_soft = softmax_approx(Attn / sqrt(Hq))
  - INT8 softmax approx: dùng lookup table hoặc piecewise linear
  - Output requant về INT8

Out = Attn_soft × V → [1,400,Hv]
  - INT8 GEMM
  - reshape → [1,Hv,20,20]
  
Output proj = OS_1x1 → F10_out [1,256,20,20]
```

---

## 6. Layout/Addressing Rules cho từng Primitive

### 6.1. Banking Model

```
bank_input  = h mod 3           → 3 input banks: bank0, bank1, bank2
bank_output = out_row mod 4     → 4 output banks

Ý nghĩa: Conv3x3 stride1 cần row h-1, h, h+1 → 3 banks xoay vòng
Conv3x3 stride2 cần row h, h+1, h+1 → vẫn 3 banks
```

### 6.2. Row Slot Model

```
Q_in    = ceil((K_eff + 3*stride) / 3)
row_slot = floor(h / 3) mod Q_in

Ví dụ Conv3x3 stride=1:
  K_eff = 3, stride = 1
  Q_in = ceil((3 + 3*1) / 3) = ceil(6/3) = 2
  row_slot ∈ {0, 1} → 2 slot positions per bank

Ví dụ Conv3x3 stride=2:
  K_eff = 3, stride = 2
  Q_in = ceil((3 + 3*2) / 3) = ceil(9/3) = 3
  row_slot ∈ {0, 1, 2}
```

### 6.3. Lane Packing

```
LANES = 16
lane  = x mod 16          → column trong warp
Wblk  = floor(x / 16)     → horizontal block index

pack16(data[W, C]):
  → packed[W//16, C, 16]   (16 values per lane per channel)

unpack16(packed[W//16, C, 16]):
  → data[W, C]

Áp dụng: khi load input từ GLB vào PE lane array
  - 16 lanes × Cin channels × 1 spatial point per cycle
```

### 6.4. Address Mapping

```
Physical address (logical):
  addr = bank_base
       + row_slot * (Wblk_total * Cin * LANES)
       + Wblk * (Cin * LANES)
       + cin_idx * LANES
       + lane_id

Với:
  bank_base   = GLB_BANK[bank_input]  phần bắt đầu cho bank số đó
  Wblk_total  = ceil(W / LANES)
  lane_id     = x mod LANES
  Wblk        = x // LANES
```

### 6.5. PSUM_MODE vs ACT_MODE

```
if NOT last_pass:
    output → PSUM_namespace (INT32, địa chỉ trong GLB output riêng)
    Không qua PPU bias/requant
    
if last_pass:
    PSUM_namespace + bias_add → requant → activation → clamp
    output → ACT_namespace (INT8, địa chỉ normal GLB output)
    
Điều kiện last_pass:
    last_pass = last_cin AND last_kernel AND last_reduce
```

---

## 7. Descriptor Mapping

### 7.1. Mỗi Primitive cần các Descriptor

| Descriptor | Nội dung bắt buộc | Ví dụ cho RS_DENSE_3x3 |
|---|---|---|
| NET_DESC | version, num_layers, weight_base, act_base | v1, 23 layers, 0x0000, 0x40000 |
| LAYER_DESC | primitive_id, in/out shape, kernel, stride | P0, [1,3,640,640],[1,16,320,320], 3, 2 |
| TILE_DESC | tile bounds, cin_chunk, cout_chunk, flags | h=0..31, Wblk=0..39, cin=0..2, cout=0..15 |
| ROUTER_PROFILE | route source→dest, broadcast mask | GLB_IN→PE, no broadcast |
| POST_PROFILE | bias_en, scale_mul, scale_shift, zp_out, clamp | en=1, M=..., shift=..., zp=0, [-128,127] |

### 7.2. Flags Semantics

```
flags:
  first_tile  : tile đầu tiên của layer → reset PSUM accumulator
  edge_tile   : tile chạm biên ảnh → padding zeros
  hold_skip   : tensor output cần giữ lại trong GLB cho skip đến sau
  need_swizzle: output cần qua swizzle engine (transpose, upsample, concat)

last_flags:
  last_cin    : đây là Cin chunk cuối → reduce channel dimension
  last_kernel : đây là kernel position cuối → reduce kernel dimension
  last_reduce : đây là tile cuối trong reduction → trigger PPU
```

---

## 8. Block → RTL Module Mapping

### 8.1. Bảng Mapping

| Block | Primitives | RTL Modules chính |
|---|---|---|
| Conv (L0,1,3,17) | RS_DENSE_3x3(s2) | window_gen, pe_lane_mac, column_reduce, ppu_lite |
| QC2f (L2,4,6,8,13,16,19) | OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1 | pe_cluster, glb_banks, router_cluster, ppu_lite |
| SCDown (L5,7,20) | OS_1x1 + DW_3x3(s2) | pe_cluster (dw mode), glb_banks, ppu_lite |
| SPPF (L9) | OS_1x1 + MAXPOOL_5x5×3 + CONCAT + OS_1x1 | pool_engine, router_cluster, glb_banks |
| QPSA (L10) | OS_1x1 + GEMM_ATTN_BASIC + OS_1x1 | gemm_attn_engine, pe_cluster, ppu_lite |
| Upsample (L11,14) | UPSAMPLE_NEAREST | swizzle_engine / tensor_post_engine |
| QConcat (L12,15,18,21) | CONCAT | router_cluster, (mini PPU cho requant nếu cần) |
| QC2fCIB (L22) | OS_1x1 + DW_7x7_MULTIPASS + OS_1x1 + CONCAT | pe_cluster (dw multipass), ppu_lite |

### 8.2. RTL Module Roles

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Module               │ Chức năng                                        │
├─────────────────────────────────────────────────────────────────────────┤
│ window_gen           │ Sinh cửa sổ 3×3/1×1/5×5 từ GLB input           │
│ pe_lane_mac          │ 16-lane INT8×INT8→INT32 MAC                     │
│ column_reduce        │ Sum across kernel positions (Cin tích lũy)      │
│ pe_cluster           │ Bao gồm window_gen + pe_lane_mac + column_reduce │
│ ppu_lite             │ bias_add + requant (scale×shift) + clamp + act  │
│ pool_engine          │ max_compare_tree cho MAXPOOL                    │
│ gemm_attn_engine     │ Matrix multiplication cho attention             │
│ router_cluster       │ Routing dữ liệu giữa GLB banks và PE          │
│ swizzle_engine       │ Tensor reshape/upsample/transpose               │
│ tensor_post_engine   │ UPSAMPLE_NEAREST, MOVE operations               │
│ glb_*_bank           │ Global Line Buffer banks (input/output/weight)  │
│ addr_gen_input       │ Tạo địa chỉ load input theo banking model       │
│ addr_gen_weight      │ Tạo địa chỉ load weight                        │
│ addr_gen_output      │ Tạo địa chỉ write output                       │
│ row_slot_manager     │ Quản lý row_slot vòng xoay trong GLB          │
│ tile_fsm             │ Điều khiển tiling loop                         │
│ desc_fetch_engine    │ Fetch và parse descriptor stack                │
│ barrier_manager      │ Đồng bộ producer→consumer cho skip/concat     │
│ subcluster_wrapper   │ Wrapper ghép các module thành block-level      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Golden Python → RTL Verification Flow

### 9.1. Thứ tự xây dựng Golden Python

```
Phase 1: Core primitives
  config.py + types.py
  quant_affine.py (quantize, dequantize, requant, clamp)
  quant_domain_align.py (common-domain cho concat/add)
  
Phase 2: Primitives
  primitive_conv.py   (RS_DENSE_3x3, OS_1x1)
  primitive_dw.py     (DW_3x3, DW_7x7_MULTIPASS)
  primitive_pool.py   (MAXPOOL_5x5)
  primitive_tensor.py (MOVE, CONCAT, UPSAMPLE_NEAREST, EWISE_ADD)
  primitive_psa.py    (GEMM_ATTN_BASIC)
  
Phase 3: Layout models
  banking_model.py    (bank_input, bank_output)
  row_slot_model.py   (Q_in, row_slot)
  lane_packing.py     (pack16/unpack16)
  address_model.py    (logical → physical)
  psum_act_model.py   (PSUM/ACT namespace semantics)
  
Phase 4: Block models
  block_qc2f.py
  block_scdown.py
  block_sppf.py
  block_qpsa.py
  block_qc2fcib.py
  
Phase 5: Model runner
  layer_specs.py      (bảng layer 0–22)
  model_forward_runner.py (chạy end-to-end)
  
Phase 6: Tests
  test_primitives.py, test_quant.py, test_layout.py
  test_blocks.py, test_model_forward.py
```

### 9.2. Sign-off Criteria

```
☐ test_primitives.py PASS:
    conv3x3 s1/s2, conv1x1, dw3x3, dw7x7 multipass, pool, upsample, concat, add

☐ test_quant.py PASS:
    quantize/dequantize round-trip, requant accuracy
    common-domain concat với domain mismatch
    common-domain add với saturation

☐ test_layout.py PASS:
    bank_input: h=0→bank0, h=1→bank1, h=2→bank2, h=3→bank0
    row_slot: Conv3x3 s1: Q_in=2; Conv3x3 s2: Q_in=3
    pack16/unpack16 round-trip
    address mapping không overlap

☐ test_blocks.py PASS:
    QC2f: shape [1,32,160,160]→[1,32,160,160], int8 output
    SCDown: shape [1,64,80,80]→[1,128,40,40]
    SPPF: shape [1,256,20,20]→[1,256,20,20], 3× pool đúng
    QPSA: shape [1,256,20,20]→[1,256,20,20], attention correct
    QC2fCIB: DW7x7 multipass == monolithic 7x7

☐ test_model_forward.py PASS:
    Layer 0–22 end-to-end: P3[1,64,80,80], P4[1,128,40,40], P5[1,256,20,20]
    Stage outputs cho tất cả 23 stages
    Quant metadata đúng (scale, zp cho P3/P4/P5)
    Dump traces: layout/address traces, PSUM/ACT traces, DW7x7 pass traces
```

---

## 10. Critical Path và Risk Analysis

### 10.1. Rủi ro cao nhất

| Rủi ro | Mô tả | Giải pháp |
|---|---|---|
| **Concat domain mismatch** | scale_A ≠ scale_B tại L12,15,18,21 | common-domain requant, kiểm tra qua test_quant.py |
| **DW_7x7 multipass accuracy** | Pass split phải cho kết quả bằng monolithic | Test bắt buộc: multipass == monolithic 7×7 |
| **GLB skip buffer pressure** | F4, F6, F8 phải sống lâu trong GLB | Tính toán tổng 900KB dự trữ, verify không overlap |
| **QPSA softmax approximation** | INT8 softmax thiếu chính xác | Kiểm tra accuracy drop so với float reference |
| **SiLU INT8 approximation** | SiLU không natural cho INT8 | LUT 8-bit với 256 entries hoặc piecewise linear |
| **PSUM/ACT namespace** | last_pass sai → output sai stage | Verify flag logic qua psum_act_model.py trước RTL |

### 10.2. Thứ tự ưu tiên implement

```
1. quant_affine.py         ← nền tảng cho mọi thứ
2. quant_domain_align.py   ← risk số 1 của neck
3. primitive_conv.py       ← RS_DENSE_3x3 chiếm ~70% compute
4. primitive_tensor.py     ← CONCAT, UPSAMPLE critical path
5. primitive_dw.py         ← DW_7x7_MULTIPASS risk
6. block_qc2f.py           ← most repeated block (7 lần)
7. model_forward_runner.py ← integration test
```

---

## 11. Ví dụ Tính Toán Chi tiết: RS_DENSE_3x3

### 11.1. Math Reference (Layer 0)

```python
# Input đã quantize từ CPU
X_int8 = tensor(shape=[1,3,640,640], dtype=int8)  # int8 thực tế [-128,127]
scale_x = 0.003921568627  # ≈ 1/255
zp_x = 0  # quant8: y = round(x/scale + zp), zp=0 cho unsigned

# Weight (per-output-channel quantize)
W_int8 = tensor(shape=[16,3,3,3], dtype=int8)  # [Cout, Cin, kH, kW]
scale_w = tensor(shape=[16], dtype=float32)  # per-channel
zp_w = 0  # weight INT8 thường zp=0 (symmetric)

# Bias (đã fuse BN offline)
B_int32 = tensor(shape=[16], dtype=int32)
# B_int32[cout] = BN_fused_bias / (scale_x * scale_w[cout])

# Output scale (học từ PTQ calibration)
scale_y = tensor(scalar, dtype=float32)
zp_y = 0

# Tính toán (floor model):
for cout in range(16):
    for h_out in range(320):
        for w_out in range(320):
            acc = 0  # int32
            for cin in range(3):
                for kh in range(3):
                    for kw in range(3):
                        h_in = 2*h_out + kh - 1  # stride=2, padding=1
                        w_in = 2*w_out + kw - 1
                        if 0 <= h_in < 640 and 0 <= w_in < 640:
                            acc += int32(X_int8[0,cin,h_in,w_in] - zp_x) * \
                                   int32(W_int8[cout,cin,kh,kw] - zp_w)
                        # else: padding (value 0 - zp_x)*w → cộng zp correction
            
            acc += B_int32[cout]  # bias add
            
            # Requant: M = scale_x * scale_w[cout] / scale_y
            M = scale_x * scale_w[cout] / scale_y
            # Hardware: M được biểu diễn bằng (M_int, shift) fixed-point
            # M_int ≈ round(M * 2^shift), chọn shift để M_int fit INT16/INT32
            
            y_raw = round(M * acc) + zp_y
            Y_int8[0,cout,h_out,w_out] = clamp(y_raw, -128, 127)
```

### 11.2. Zero-Point Correction (Optimization)

```python
# Với zp_w = 0 (symmetric weight):
# acc = Σ (x-zp_x)(w-0) = Σ x*w - zp_x * Σ w

# Tách: acc = Σ x*w - zp_x * partial_sum_w

# partial_sum_w[cout] = Σ_{cin,kh,kw} W_int8[cout,cin,kh,kw]
#   → precomputed offline, stored as INT32 constant

# Optimized hardware:
acc_raw = mac(X_int8, W_int8)  # INT8*INT8 → INT32, 16 lanes parallel
acc = acc_raw - zp_x * partial_sum_w[cout] + B_int32[cout]
```

---

## 12. Kết luận

Bảng mapping hoàn chỉnh Layer 0–22 → Primitive → RTL được tóm tắt:

| Layer | Block | Primitive decomposition | RTL chính |
|---|---|---|---|
| 0 | Conv(s2) | RS_DENSE_3x3(s=2) | pe_cluster, ppu_lite |
| 1 | Conv(s2) | RS_DENSE_3x3(s=2) | pe_cluster, ppu_lite |
| 2 | QC2f | OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1 | pe_cluster, router_cluster |
| 3 | Conv(s2) | RS_DENSE_3x3(s=2) | pe_cluster, ppu_lite |
| 4 | QC2f | OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1 | pe_cluster, router_cluster |
| 5 | SCDown | OS_1x1 + DW_3x3(s=2) | pe_cluster(dw), ppu_lite |
| 6 | QC2f | OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1 | pe_cluster, router_cluster |
| 7 | SCDown | OS_1x1 + DW_3x3(s=2) | pe_cluster(dw), ppu_lite |
| 8 | QC2f | OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1 | pe_cluster, router_cluster |
| 9 | SPPF | OS_1x1 + MAXPOOL_5x5×3 + CONCAT + OS_1x1 | pool_engine, router_cluster |
| 10 | QPSA | OS_1x1 + GEMM_ATTN_BASIC + OS_1x1 | gemm_attn_engine |
| 11 | Upsample | UPSAMPLE_NEAREST | tensor_post_engine |
| 12 | QConcat | CONCAT (skip: L6) | router_cluster, barrier_manager |
| 13 | QC2f | OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1 | pe_cluster, router_cluster |
| 14 | Upsample | UPSAMPLE_NEAREST | tensor_post_engine |
| 15 | QConcat | CONCAT (skip: L4) | router_cluster, barrier_manager |
| 16 | QC2f | OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1 | pe_cluster, router_cluster → **P3** |
| 17 | Conv(s2) | RS_DENSE_3x3(s=2) | pe_cluster, ppu_lite |
| 18 | QConcat | CONCAT (skip: L13) | router_cluster, barrier_manager |
| 19 | QC2f | OS_1x1 + RS_DENSE_3x3 + CONCAT + OS_1x1 | pe_cluster, router_cluster → **P4** |
| 20 | SCDown | OS_1x1 + DW_3x3(s=2) | pe_cluster(dw), ppu_lite |
| 21 | QConcat | CONCAT (skip: L8) | router_cluster, barrier_manager |
| 22 | QC2fCIB | OS_1x1 + DW_7x7_MULTIPASS + OS_1x1 + CONCAT | pe_cluster(dw_mp), ppu_lite → **P5** |

**Outputs cuối:**
- `P3_int8 = F16_out [1,64,80,80]`
- `P4_int8 = F19_out [1,128,40,40]`
- `P5_int8 = F22_out [1,256,20,20]`

---

*Tài liệu này là kết quả tổng hợp từ MODEL_BLOCKS_INT8_DETAIL.md, MODEL_FORWARD_FLOW.md, MODEL_LAYER_DEPENDENCIES.md và MODEL_LAYERS_INT8_FLOW.md.*


---
---

# ════════════════════════════════════════════════════════════════
# PHẦN III — KIẾN TRÚC PHẦN CỨNG V2 (>100 FPS)
# ════════════════════════════════════════════════════════════════

---

<a id='phần-iii1---kiến-trúc-v2'></a>

# PHẦN III.1 — Kiến Trúc V2: Scale-Up Analysis & Architecture Redesign
> Nguồn: `SW_KLTN/HW_ARCHITECTURE_V2_100FPS.md`

---

# Kiến Trúc V2: YOLOv10n INT8 Accelerator – Target >100 FPS
## Scale-Up Analysis & Architecture Redesign

> **Phiên bản**: V2 – Nâng cấp từ V1 để đạt >100 FPS trên Virtex 7
> **FPGA target**: XC7VX690T (primary), XC7VX485T (secondary)
> **Nguyên tắc bất di bất dịch**: CHẠY ĐÚNG INFERENCE NHƯ PHẦN MỀM

---

## 1. PHÂN TÍCH YÊU CẦU: TỪ 15 FPS LÊN >100 FPS

### 1.1. Hiệu năng V1 hiện tại

```
V1 Architecture:
  16 subclusters (4 SC × 4 Sub)
  Per sub: 3 rows × 4 cols × 16 lanes = 192 INT8 MACs
  Active: 4 subclusters (1 RUNNING per SC)
  Peak: 4 × 192 = 768 MACs/cycle

  @ 200 MHz, 45% utilization:
  Effective: 768 × 200M × 0.45 = 69.1 GOPS
  HW compute: 3.0 GMAC / 69.1 GOPS = 43.4 ms → 23 FPS (HW only)
  End-to-end: ~65-80 ms → 12-15 FPS
```

### 1.2. Target >100 FPS – Yêu cầu tính toán

```
YOLOv10n backbone+neck (L0-L22): ~3.0 GMACs
  (6.7 GFLOPs total model, head ~10% → backbone+neck ≈ 3.0 GMACs)

Để đạt 100 FPS trên HW:
  T_hw < 10 ms
  Required throughput: 3.0 GMAC / 10 ms = 300 GOPS

V1 cung cấp: 69.1 GOPS
→ CẦN TĂNG ≈ 4.3× THROUGHPUT
```

### 1.3. Phân rã bottleneck V1

```
V1 bottleneck analysis:
┌──────────────────────────────────────────────────────────────────────────────┐
│ Factor               │ V1 value   │ Lý do mất hiệu suất                    │
├──────────────────────┼────────────┼─────────────────────────────────────────┤
│ Total instantiated   │ 3,072 MACs │ 16 sub × 192 MAC → phần cứng đủ lớn   │
│ Active (computing)   │ 768 MACs   │ Chỉ 25% PEs đang compute (1/4 subs)   │ ← BOTTLENECK #1
│ Spatial utilization  │ ~77%       │ LANES=16 vs feature map widths          │
│ Temporal utilization │ ~58%       │ Fill/drain/barrier stalls               │ ← BOTTLENECK #2
│ Overall utilization  │ ~45%       │ 77% × 58% = 44.7%                      │
│ Effective MACs       │ 345 /cycle │ 768 × 0.45                              │
└──────────────────────┴────────────┴─────────────────────────────────────────┘

Insight: 75% phần cứng IDLE vì 4-phase scheduler chỉ cho 1 sub RUN per SC.
         Nếu tăng active subs lên 2 per SC → 2× compute ngay lập tức.
```

---

## 2. CHIẾN LƯỢC SCALE-UP: 4 TRỤC CẢI TIẾN

### 2.1. Bốn trục scaling khả thi

```
┌───────────────────────┬──────────────┬──────────────┬──────────┬─────────────┐
│ Trục                  │ V1           │ V2 (target)  │ Hệ số    │ Trả giá     │
├───────────────────────┼──────────────┼──────────────┼──────────┼─────────────┤
│ ① LANES (data width)  │ 16           │ 32           │ 2.0×     │ +DSP, +BRAM │
│ ② Active subs/SC      │ 1            │ 2            │ 2.0×     │ +scheduler  │
│ ③ Clock frequency     │ 200 MHz      │ 200-250 MHz  │ 1.0-1.25×│ timing      │
│ ④ Utilization         │ 45%          │ 50-55%       │ 1.1-1.22×│ SW compiler │
└───────────────────────┴──────────────┴──────────────┴──────────┴─────────────┘

Tổng hệ số tối đa: 2.0 × 2.0 × 1.25 × 1.22 = 6.1× → dư sức cho 4.3×
```

### 2.2. Tại sao chọn ① LANES=32 + ② Dual-RUNNING?

**LANES 16→32 (trục ①):**
- Mỗi PE xử lý 32 spatial positions/cycle thay vì 16
- EXT_PORT_WIDTH tăng 128b→256b (2 AXI beats)
- Không thay đổi logic control (cùng #rows, #cols)
- Internal data paths: 256-bit thay vì 128-bit

**Dual-RUNNING scheduler (trục ②):**
- 4-phase scheduler gốc: RUNNING + FILLING + DRAINING + HOLD_SKIP
- V2: **RUNNING×2 + FILLING + DRAINING/HOLD**
- 2 subclusters compute từ local GLB_BANK đồng thời → không tranh chấp external port
- External port chỉ phục vụ FILLING (read) và DRAINING (write)

**Tại sao KHÔNG tăng PE_COLS (4→8)?**
- PE_COLS=8 → cần 8 bank_output, 8 PPU lanes, phức tạp hóa routing
- Lợi ích H-parallelism nhỏ (H=20 vẫn chỉ cần ceil(20/4)=5 waves)
- Giữ PE_COLS=4 → giữ nguyên bank_output/PPU/router design từ V1

**Tại sao KHÔNG dùng LANES=64?**
- W=20 (feature maps của P5): spatial util = 20/64 = 31.25% → waste 69%!
- LANES=32 tại W=20: util = 20/32 = 62.5% → chấp nhận được
- Bus 512b (64×8) gây routing congestion nặng trên Virtex 7

---

## 3. KIẾN TRÚC V2 – THÔNG SỐ CHỐT

### 3.1. So sánh V1 vs V2

```
┌───────────────────────────┬────────────┬────────────┬───────────────────────┐
│ Parameter                 │ V1         │ V2         │ Ghi chú               │
├───────────────────────────┼────────────┼────────────┼───────────────────────┤
│ LANES                     │ 16         │ 32         │ 2× spatial parallel   │
│ PE_ROWS                   │ 3          │ 3          │ Giữ nguyên (K=3)      │
│ PE_COLS                   │ 4          │ 4          │ Giữ nguyên            │
│ MACs per subcluster       │ 192        │ 384        │ 2×                    │
│ SUPER_CLUSTERS            │ 4          │ 4          │ Giữ nguyên            │
│ SUBCLUSTERS_PER_SC        │ 4          │ 4          │ Giữ nguyên            │
│ Total subclusters         │ 16         │ 16         │ Giữ nguyên            │
│ Active subs per SC        │ 1          │ 2          │ Dual-RUNNING          │
│ Total active MACs         │ 768        │ 3,072      │ 4×                    │
│ EXT_PORT_WIDTH            │ 128b       │ 256b       │ 2× (for wider lanes)  │
│ PSUM_WIDTH                │ 32         │ 32         │ Giữ nguyên            │
│ ACT_WIDTH                 │ 8          │ 8          │ Giữ nguyên            │
│ SiLU_LUT_SIZE             │ 256        │ 256        │ Giữ nguyên            │
│ Clock target              │ 200 MHz    │ 200-250MHz │ ↑ nếu timing cho phép │
│ Target utilization        │ 45%        │ 50%        │ ↑ nhờ dual-pipeline   │
└───────────────────────────┴────────────┴────────────┴───────────────────────┘
```

### 3.2. Package định nghĩa V2

```systemverilog
package accel_pkg_v2;
  parameter LANES         = 32;        // ← thay đổi chính
  parameter PE_ROWS       = 3;
  parameter PE_COLS       = 4;
  parameter INPUT_BANKS   = 3;
  parameter OUTPUT_BANKS  = 4;
  parameter WEIGHT_BANKS  = 3;
  parameter PSUM_WIDTH    = 32;
  parameter ACT_WIDTH     = 8;
  parameter EXT_PORT_WIDTH = 256;      // ← thay đổi chính
  parameter SUPER_CLUSTERS = 4;
  parameter SUBS_PER_SC   = 4;
  parameter ACTIVE_PER_SC = 2;         // ← thay đổi chính
  parameter MACS_PER_SUB  = PE_ROWS * PE_COLS * LANES;  // = 384
  parameter TOTAL_ACTIVE_MACS = SUPER_CLUSTERS * ACTIVE_PER_SC * MACS_PER_SUB; // = 3,072
endpackage
```

---

## 4. DSP48E1 RESOURCE STRATEGY: 2 MACs PER DSP

### 4.1. Thách thức: DSP48E1 không hỗ trợ INT8 SIMD natively

```
DSP48E1 (Virtex 7): 25×18 → 43-bit multiplier
DSP48E2 (UltraScale+): native INT8 SIMD (2 MACs/DSP)
→ Trên Virtex 7, cần kỹ thuật unsigned-offset packing thủ công
```

### 4.2. Kỹ thuật Unsigned-Offset Packing (2 INT8 MACs / 1 DSP48E1)

```
Bước 1: Chuyển signed INT8 → unsigned INT8
  a_u = a_signed + 128    ∈ [0, 255]
  b_u = b_signed + 128    ∈ [0, 255]

Bước 2: Pack 2 activation values vào A port (25-bit)
  A[24:0] = {a2_u[7:0], 9'b0, a1_u[7:0]}
  ─── bit 24..17: a2_unsigned ───── bit 16..8: guard zeros ─── bit 7..0: a1_unsigned ───

Bước 3: Weight vào B port (18-bit)
  B[17:0] = {10'b0, w_u[7:0]}

Bước 4: DSP multiply
  P[42:0] = A × B = {a2_u × w_u} << 17  |  {a1_u × w_u}

  Max(a1_u × w_u) = 255 × 255 = 65,025 < 2^16 = 65,536
  → product 1 fits in P[15:0], product 2 in P[32:17]
  → NO CARRY OVERLAP (guard bit P[16] always 0) ✓

Bước 5: Extract & correct
  p1_u = P[15:0]   = a1_u × w_u
  p2_u = P[32:17]  = a2_u × w_u

  a1 × w = p1_u - 128×(a1_u + w_u) + 16384
  a2 × w = p2_u - 128×(a2_u + w_u) + 16384

  Correction: 2 add/sub operations in LUTs (~25 LUTs per pair)
```

### 4.3. Ưu/nhược điểm

```
✓ 2× MAC throughput per DSP slice
✓ Works on Virtex 7 DSP48E1 (không cần UltraScale)
✓ Correction logic nhỏ (~25 LUTs per DSP)
✗ Latency +1 cycle cho correction pipeline stage
✗ Cần precompute 128×(a_u + w_u) — trivially pipelined
```

### 4.4. Phương án dự phòng: Hybrid DSP + LUT MACs

```
Nếu DSP packing quá phức tạp cho implementation:

  LANES [0:15]:   192 DSPs per sub (1 MAC/DSP, giống V1)
  LANES [16:31]:  LUT-based MACs

  Per LUT-MAC: ~80 LUTs + 40 FFs (8×8 multiply + 32-bit accumulate)
  Per sub extra lanes: 12 PEs × 16 lanes × 80 LUTs = 15,360 LUTs
  Total 16 subs: 245,760 LUTs (57% of VX690T)

  DSP usage: 16 × 192 = 3,072 DSPs (85%)
  LUT usage: 246K + 100K control = 346K (80%)
  → Fits XC7VX690T ✓ (tight on LUTs)
```

---

## 5. DUAL-RUNNING SCHEDULER: CHI TIẾT

### 5.1. So sánh V1 vs V2 scheduler

```
V1 (4-phase, 1 RUNNING):                V2 (4-phase, 2 RUNNING):
┌─────┬─────────┬─────────┬──────┐     ┌─────┬─────────┬─────────┬──────┐
│Sub-0│RUNNING  │FILLING  │DRAIN │     │Sub-0│RUNNING  │DRAIN    │FILL  │
│Sub-1│FILLING  │DRAIN    │HOLD  │     │Sub-1│RUNNING  │FILL     │DRAIN │
│Sub-2│DRAIN    │HOLD     │RUN   │     │Sub-2│FILLING  │RUNNING  │RUN   │
│Sub-3│HOLD     │RUNNING  │FILL  │     │Sub-3│DRAIN    │RUNNING  │HOLD  │
│     │  T=0    │  T=1    │ T=2  │     │     │  T=0    │  T=1    │ T=2  │
└─────┴─────────┴─────────┴──────┘     └─────┴─────────┴─────────┴──────┘
Active: 1 sub                           Active: 2 subs
```

### 5.2. Rotation protocol

```
Timeline (per SuperCluster):

Phase 0:  Sub-0 = RUNNING(tile_A)     Sub-2 = FILLING(tile_C)
          Sub-1 = RUNNING(tile_B)     Sub-3 = DRAINING(tile_prev) or HOLD

     ──── Sub-0 finishes tile_A ────

Phase 1:  Sub-0 → DRAINING(tile_A)    Sub-3 → FILLING(tile_D)
          Sub-1 = RUNNING(tile_B)     Sub-2 → RUNNING(tile_C)

     ──── Sub-1 finishes tile_B ────

Phase 2:  Sub-1 → DRAINING(tile_B)    Sub-0 → FILLING(tile_E)
          Sub-2 = RUNNING(tile_C)     Sub-3 → RUNNING(tile_D)

     ... vòng lặp ...

Key insight: 2 subs compute đồng thời từ local GLB_BANK riêng.
             External port phục vụ 1 FILL + 1 DRAIN time-multiplexed.
```

### 5.3. External port bandwidth check

```
Per tile (typical L6: QC2f, Cin=128, Cout=128, H=40, W=40):
  Input data:  6 rows × 64ch_tile × ceil(40/32) × 32B = 24,576 B
  Weight data: 9 × 64 × 64 = 36,864 B (per Cin×Cout tile)
  Output data: 4 rows × 64ch × ceil(40/32) × 32B = 16,384 B

  Fill:  (24,576 + 36,864) B / (32 B/cycle × 200 MHz) = 9.6 µs
  Drain: 16,384 B / (32 B/cycle × 200 MHz) = 2.6 µs
  Fill + Drain = 12.2 µs

  Compute: 9×64×64×40×2_Wblk = 2,949,120 MACs / 384 MACs/cycle = 7,680 cycles = 38.4 µs

  Compute >> Fill+Drain → dual-RUNNING works perfectly ✓

Per tile (small, L10: QPSA GEMM, H=20, W=20):
  Fill: ~15 µs (more weight-heavy)
  Compute: ~25 µs

  Still compute > fill → works ✓
```

### 5.4. HOLD_SKIP management trong V2

```
V1: 1 sub dedicated HOLD_SKIP (có thể giữ skip tensor trong GLB)
V2: HOLD_SKIP role xoay vòng, không dedicated → cần linh hoạt hơn

Strategy cho V2:
  Option A (preferred): GLB_BANK đủ lớn → giữ skip data trong local GLB
    - Sub chuyển từ HOLD→RUNNING: trước khi compute, check GLB có skip data
    - Skip data chỉ overwrite vùng bank_output, bank_input preserved
    - Feasible vì skip tensors (F4/F6/F8/F13) nhỏ hơn GLB capacity

  Option B (fallback): Spill skip data to DDR3
    - Khi sub cần rotate out of HOLD: DMA write skip to DDR3 skip arena
    - Khi barrier release: DMA read skip back
    - Extra DDR3 traffic: 921 KB × 2 = 1.84 MB
    - At 12.8 GB/s: 0.14 ms overhead → negligible

Skip tensor sizes (unchanged from V1):
  F4_out [1,64,80,80]   = 409,600 B
  F6_out [1,128,40,40]  = 204,800 B
  F8_out [1,256,20,20]  = 102,400 B
  F13_out [1,128,40,40] = 204,800 B
  Total: 921,600 B ≈ 900 KB

GLB_BANK per sub (V2, LANES=32): ~200 KB
→ F4 (400KB) KHÔNG vừa 1 sub → phải split hoặc dùng Option B cho F4
→ F6/F8/F13 vừa trong 1 sub GLB → Option A works
```

---

## 6. TÀI NGUYÊN FPGA – ƯỚC TÍNH CHI TIẾT

### 6.1. XC7VX690T Resource Budget

```
┌───────────────────┬──────────┬──────────────┬─────────────────────────────────┐
│ Resource          │ Available│ V2 Usage     │ Chi tiết                        │
├───────────────────┼──────────┼──────────────┼─────────────────────────────────┤
│ DSP48E1           │ 3,600    │ 3,072 (85%)  │ 16 sub × 192 DSP (2 MAC/DSP)   │
│ BRAM36K           │ 1,470    │ ~850 (58%)   │ 16 × 48 GLB + 82 misc          │
│ LUT6              │ 433,200  │ ~280K (65%)  │ 92K DSP-correct + 188K control  │
│ FF                │ 866,400  │ ~320K (37%)  │ Pipeline regs + FSM             │
└───────────────────┴──────────┴──────────────┴─────────────────────────────────┘
```

### 6.2. DSP allocation breakdown

```
Per subcluster (384 MACs, using 2-MAC packing):
  PE_CLUSTER: 12 PEs × 16 lanes_per_DSP = 192 DSPs
  (Mỗi DSP xử lý 2 lanes: lane[2i] và lane[2i+1])

Total: 16 subclusters × 192 DSPs = 3,072 DSPs
PPU requant: ~16 DSPs per sub = 256 DSPs (shared timing)
Controller misc: ~32 DSPs
Grand total: ~3,360 DSPs (93% of 3,600) ✓
```

### 6.3. BRAM allocation breakdown

```
Per subcluster GLB_BANK (LANES=32):
  bank_input[3]:
    Each bank: R_need × Cin_tile × ceil(W_max/32) × 32 bytes
    Worst case (L16: H=80, Cin=192, W=80):
      6 × 48 × 3 × 32 = 27,648 B ≈ 28 KB per bank
      3 banks = 84 KB → ceil(84K / 4.5K) = 19 BRAM36K

  bank_weight[3]:
    Each bank: kernel_data per reduction lane
    Worst case: ~16 KB per bank → 3 × 16K = 48 KB
    11 BRAM36K

  bank_output[4]:
    Each bank: Cout_tile × ceil(W/32) × 32 × 4B (INT32 PSUM) or 1B (INT8 ACT)
    Worst case (PSUM mode): 4 × 32 × 3 × 32 × 4B = 49,152 B ≈ 48 KB
    11 BRAM36K

  psum_buffer: 12 PEs × 32 lanes × 4B = 1,536 B → shared with bank_output

  metadata_RAM + SiLU_LUT: ~4 KB → 1 BRAM36K

  Per sub total: ~42 BRAM36K

Total: 16 subs × 42 = 672 BRAM36K
Desc_RAM + DMA buffers + perf counters: ~80 BRAM36K
Grand total: ~752 BRAM36K (51% of 1,470) ✓
```

### 6.4. XC7VX485T Feasibility (secondary target)

```
┌───────────────────┬──────────┬────────────┬────────────────────────────────┐
│ Resource          │ Available│ V2 Usage   │ Fit?                           │
├───────────────────┼──────────┼────────────┼────────────────────────────────┤
│ DSP48E1           │ 2,800    │ 3,360      │ ✗ EXCEEDS by 560              │
│ BRAM36K           │ 1,030    │ ~752       │ ✓ 73%                         │
│ LUT6              │ 303,600  │ ~280K      │ ✗ 92% (very tight)            │
│ FF                │ 866,400  │ ~320K      │ ✓ 37%                         │
└───────────────────┴──────────┴────────────┴────────────────────────────────┘

→ XC7VX485T KHÔNG ĐỦ cho V2 full config
→ Giải pháp: V2-lite với LANES=24 hoặc 12 subclusters
   (xem mục 6.5)
```

### 6.5. V2-lite cho XC7VX485T

```
V2-lite: LANES=24, 12 subclusters (4 SC × 3 Sub), 2 active per SC

Per sub: 3 × 4 × 24 = 288 MACs
DSPs: 12 × 144 = 1,728 (62% of 2,800) ✓
Active: 8 × 288 = 2,304 MACs
BRAMs: 12 × 36 = 432 + 80 = 512 (50%) ✓
LUTs: ~210K (69%) ✓

Performance @ 250 MHz, 50%:
  2,304 × 250M × 0.50 = 288 GOPS → T_hw = 10.4 ms → 96 FPS ✗ (gần!)

@ 250 MHz, 55%: 317 GOPS → 9.5 ms → 105 FPS ✓ (chỉ đủ nếu tối ưu)
```

---

## 7. TÍNH TOÁN FPS CHI TIẾT PER-LAYER

### 7.1. MAC counts per resolution tier

```
┌────────────────────┬────────────┬───────┬────────────────────────────────────┐
│ Resolution tier    │ Layers     │ MACs  │ Chú thích                          │
├────────────────────┼────────────┼───────┼────────────────────────────────────┤
│ 640→320 (s2)       │ L0         │  44 M │ Conv3×3, Cin=3→16                  │
│ 320→160 (s2)       │ L1         │ 118 M │ Conv3×3, Cin=16→32                 │
│ 160×160            │ L2         │ 170 M │ QC2f(32→32): OS1+RS3×2+CAT+OS1    │
│ 160→80 (s2)        │ L3         │ 118 M │ Conv3×3, Cin=32→64                 │
│ 80×80              │ L4         │ 340 M │ QC2f(64→64)                        │
│ 80→40 (SCDown)     │ L5         │  60 M │ OS1×2 + DW3×2                      │
│ 40×40              │ L6         │ 320 M │ QC2f(128→128)                      │
│ 40→20 (SCDown)     │ L7         │ 120 M │ OS1×2 + DW3×2                      │
│ 20×20              │ L8         │ 550 M │ QC2f(256→256)                      │
│ 20×20              │ L9         │ 200 M │ SPPF: OS1+MP5×3+CAT+OS1           │
│ 20×20              │ L10        │ 360 M │ QPSA: GEMM_ATTN + OS1             │
│ 40×40 (upsample)   │ L11        │   0   │ address remap only                 │
│ 40×40 (concat)     │ L12        │   0   │ data movement only                 │
│ 40×40              │ L13        │ 220 M │ QC2f(384→128)                      │
│ 80×80 (upsample)   │ L14        │   0   │ address remap only                 │
│ 80×80 (concat)     │ L15        │   0   │ data movement only                 │
│ 80×80              │ L16        │ 180 M │ QC2f(192→64)                       │
│ 80→40 (s2)         │ L17        │  37 M │ Conv3×3, Cin=64→64                 │
│ 40×40 (concat)     │ L18        │   0   │ data movement only                 │
│ 40×40              │ L19        │ 110 M │ QC2f(192→128)                      │
│ 40→20 (SCDown)     │ L20        │  30 M │ OS1 + DW3                          │
│ 20×20 (concat)     │ L21        │   0   │ data movement only                 │
│ 20×20              │ L22        │ 100 M │ QC2fCIB: OS1+DW7×3pass+CAT+OS1    │
├────────────────────┼────────────┼───────┼────────────────────────────────────┤
│ TỔNG               │ L0-L22     │3,077M │ ≈ 3.08 GMACs (khớp 6.7 GFLOPs)   │
└────────────────────┴────────────┴───────┴────────────────────────────────────┘
```

### 7.2. Spatial utilization per tier (LANES=32, PE_COLS=4)

```
Spatial utilization = (valid_W_positions / padded_W_positions)
  padded_W = ceil(W / LANES) × LANES

┌──────────┬─────┬───────────────────┬───────────────┬───────────────────┐
│ W_out    │ Wblk│ Padded W          │ W spatial util│ Layers            │
├──────────┼─────┼───────────────────┼───────────────┼───────────────────┤
│ 320      │ 10  │ 320               │ 100.0%        │ L0                │
│ 160      │  5  │ 160               │ 100.0%        │ L1, L2            │
│ 80       │  3  │ 96                │  83.3%        │ L3,L4,L15,L16,L17 │
│ 40       │  2  │ 64                │  62.5%        │ L5,L6,L12,L13,    │
│          │     │                   │               │ L18,L19            │
│ 20       │  1  │ 32                │  62.5%        │ L7-L10,L20-L22    │
└──────────┴─────┴───────────────────┴───────────────┴───────────────────┘
```

### 7.3. Per-tier compute time (V2 @ 200 MHz)

```
Active MACs = 3,072 per cycle

T_tier = MACs_tier / (3,072 × spatial_util × 200M)

┌──────────────┬───────┬────────┬────────────────────────────┐
│ Tier         │ MACs  │ S_util │ T_compute (no overhead)    │
├──────────────┼───────┼────────┼────────────────────────────┤
│ 320 (L0)     │  44 M │ 100%   │  44M/(3072×1.0×200M)=0.07ms│
│ 160 (L1-L2)  │ 288 M │ 100%   │ 288M/(3072×1.0×200M)=0.47ms│
│ 80 (L3-L4,   │ 675 M │ 83.3%  │ 675M/(3072×0.833×200M)=1.32ms│
│    L15-L17)  │       │        │                             │
│ 40 (L5-L6,   │ 740 M │ 62.5%  │ 740M/(3072×0.625×200M)=1.93ms│
│    L12-L13,  │       │        │                             │
│    L18-L19)  │       │        │                             │
│ 20 (L7-L10,  │1,330 M│ 62.5%  │1330M/(3072×0.625×200M)=3.46ms│
│    L20-L22)  │       │        │                             │
├──────────────┼───────┼────────┼────────────────────────────┤
│ TỔNG         │3,077M │  —     │ 7.25 ms (pure compute)     │
└──────────────┴───────┴────────┴────────────────────────────┘
```

### 7.4. Temporal overhead estimation

```
Temporal overhead sources:
  1. Fill/drain pipeline gap:     10% (2-phase overlap, small gaps at tile boundaries)
  2. Descriptor fetch:             3% (pipelined with compute, occasional stalls)
  3. Barrier stalls (4 points):    5% (L12/L15/L18/L21 wait for skip tensors)
  4. DW_7x7 pass 3 inefficiency:  2% (1/3 PE rows used in final pass, L22 only)
  5. Tile boundary waste:          5% (edge tiles with partial valid data)
  6. QPSA softmax float path:     2% (L10 only, small tensor 20×20)
  ─────────────────────────────────────────────────────────────
  Total temporal overhead:        ~27%
  Temporal utilization:           ~73%

Hoặc equivalently:
  T_hw = T_pure_compute / temporal_util = 7.25 / 0.73 = 9.93 ms
```

### 7.5. FPS Summary (V2 trên XC7VX690T)

```
┌────────────────────┬──────────┬────────────┬───────────┬────────────┐
│ Scenario           │ Clock    │ Util total │ Eff GOPS  │ FPS (HW)   │
├────────────────────┼──────────┼────────────┼───────────┼────────────┤
│ Conservative       │ 200 MHz  │ 45%        │ 276.5     │ ~90        │
│ ★ Realistic V2     │ 200 MHz  │ 50%        │ 307.2     │ ~101       │
│ Optimized compiler │ 200 MHz  │ 55%        │ 337.9     │ ~111       │
│ Higher clock       │ 250 MHz  │ 50%        │ 384.0     │ ~125       │
│ ★ Target sweet spot│ 220 MHz  │ 52%        │ 351.5     │ ~115       │
└────────────────────┴──────────┴────────────┴───────────┴────────────┘

★ Kết luận: Với V2 @ 200-220 MHz, đạt 100-115 FPS trên accelerator.
```

---

## 8. END-TO-END PIPELINE ANALYSIS

### 8.1. Pipeline 3 tầng

```
Frame N:   ┌─────────────────┐
           │ CPU Preprocess   │
           │ (host PC, C++)  │──→ X_int8 to DDR3
           └─────────────────┘
                    ↓ pipeline overlap
Frame N-1: ┌──────────────────────────┐
           │ Accelerator L0-L22       │
           │ (DDR3 ← DMA → accel_top) │──→ P3/P4/P5 to DDR3
           └──────────────────────────┘
                    ↓ pipeline overlap
Frame N-2: ┌─────────────────────────┐
           │ CPU Postprocess          │
           │ dequant + Qv10Detect     │
           │ + decode bbox + draw     │
           └─────────────────────────┘

Throughput = 1 / max(T_preprocess, T_hw, T_postprocess)
Latency = T_preprocess + T_hw + T_postprocess (3 frames)
```

### 8.2. Timing breakdown per stage

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Stage            │ On host PC       │ On MicroBlaze     │ On ARM (Zynq)   │
│                  │ (x86, C++, AVX2) │ (200 MHz, no FPU) │ (667 MHz, NEON) │
├──────────────────┼──────────────────┼───────────────────┼─────────────────┤
│ LetterBox/Resize │ 1.0 ms           │ 8 ms              │ 3 ms            │
│ Normalize (÷255) │ 0.5 ms           │ 5 ms              │ 1.5 ms          │
│ quantize_affine  │ 0.8 ms           │ 7 ms              │ 2 ms            │
│ DMA write input  │ 0.1 ms (PCIe)    │ 0.1 ms (local)    │ 0.1 ms          │
│ ─────────────────┼──────────────────┼───────────────────┼─────────────────┤
│ T_preprocess     │ ≈ 2.5 ms         │ ≈ 20 ms           │ ≈ 6.5 ms        │
├──────────────────┼──────────────────┼───────────────────┼─────────────────┤
│ HW Accel L0-L22  │ 9.9 ms           │ 9.9 ms            │ 9.9 ms          │
├──────────────────┼──────────────────┼───────────────────┼─────────────────┤
│ DMA read P3/P4/P5│ 0.05 ms          │ 0.05 ms           │ 0.05 ms         │
│ dequantize       │ 0.3 ms           │ 3 ms              │ 1 ms            │
│ Qv10Detect head  │ 2.0 ms           │ 12 ms             │ 4 ms            │
│ decode bbox      │ 0.5 ms           │ 3 ms              │ 1 ms            │
│ draw bbox        │ 0.5 ms           │ 2 ms              │ 1 ms            │
│ ─────────────────┼──────────────────┼───────────────────┼─────────────────┤
│ T_postprocess    │ ≈ 3.5 ms         │ ≈ 20 ms           │ ≈ 7 ms          │
├──────────────────┼──────────────────┼───────────────────┼─────────────────┤
│ THROUGHPUT       │ 1/max(2.5,9.9,   │ 1/max(20,9.9,     │ 1/max(6.5,9.9,  │
│ (pipelined)      │  3.5) = 101 FPS  │  20) = 50 FPS     │  7) = 101 FPS   │
│ LATENCY          │ 15.9 ms          │ 49.9 ms           │ 23.4 ms         │
└──────────────────┴──────────────────┴───────────────────┴─────────────────┘
```

### 8.3. Kết luận end-to-end

```
★ Host PC (x86 + FPGA qua PCIe): 101 FPS throughput ✓   (bottleneck = HW accelerator)
★ Zynq ARM + PL:                  101 FPS throughput ✓   (bottleneck = HW accelerator)
✗ MicroBlaze:                      50 FPS throughput ✗   (bottleneck = CPU pre/postprocess)

Để đạt >100 FPS end-to-end, cần:
  ① HW accelerator V2 (đã thiết kế) 
  ② CPU đủ mạnh: x86 host hoặc ARM Cortex-A9 (Zynq)
  ③ Nếu chỉ có MicroBlaze: cần thêm HW preprocessing module (xem mục 8.4)
```

### 8.4. HW Preprocessing Module (optional, cho pure-FPGA)

```
Nếu không có host PC / ARM mạnh, thêm module HW preprocessing:

module hw_preprocess (
  input  [7:0] pixel_bgr [0:2],    // raw BGR uint8
  output [7:0] x_int8              // quantized INT8
);
  // Normalize + quantize: q = round(pixel / (255 * scale)) + zp
  // Với scale ≈ 0.00392, zp = 0:
  //   q = round(pixel / 0.9996) ≈ pixel (almost identity for this scale!)
  // Hardware: fixed-point multiply by 256/255 ≈ 1.00392
  //   q = (pixel * 256 + 128) >> 8    (biased rounding)
  // → 1 multiplier + 1 adder per pixel → trivially pipelined
endmodule

Throughput: 1 pixel/cycle @ 200 MHz = 200 Mpixels/s
Input size: 3 × 640 × 640 = 1,228,800 pixels
Time: 1.23M / 200M = 6.1 ms → still a bottleneck at 6.1 ms

Fix: 4 parallel pixel lanes → 0.8M cycles → 4.1 ms → 243 FPS preprocessing → not bottleneck

Resources: 4 × (1 DSP + ~20 LUTs) = 4 DSPs + 80 LUTs → negligible
```

---

## 9. MODULES THAY ĐỔI SO VỚI V1

### 9.1. Danh sách modules cần sửa đổi

```
┌──────────────────────────┬────────────────────────────────────────────────────┐
│ Module                   │ Thay đổi                                          │
├──────────────────────────┼────────────────────────────────────────────────────┤
│ accel_pkg.sv             │ LANES=32, EXT_PORT=256, ACTIVE_PER_SC=2           │
│                          │                                                    │
│ pe_lane_mac.sv           │ 32 lanes thay vì 16; mỗi PE instantiates 16 DSPs │
│                          │ (2 MAC/DSP) thay vì 16 DSPs (1 MAC/DSP)          │
│                          │ Thêm correction logic (~25 LUTs per DSP)          │
│                          │                                                    │
│ window_gen.sv            │ Shift register 32-wide thay vì 16-wide            │
│                          │ K1/K3/K5/K7 tap generator output 32 elements      │
│                          │                                                    │
│ column_reduce.sv         │ Sum 3 rows × 32 lanes thay vì 3 × 16             │
│                          │ Output: 4 columns × 32 INT32 values               │
│                          │                                                    │
│ ppu_lite.sv              │ 32 requant lanes parallel thay vì 16              │
│                          │ SiLU LUT: 32 read ports (multi-port ROM hoặc      │
│                          │ time-multiplex 2 cycles × 16 ports)               │
│                          │                                                    │
│ glb_input_bank.sv        │ 32 subbanks per bank thay vì 16                   │
│                          │ addr_gen: Wblk_total = ceil(W/32)                  │
│                          │                                                    │
│ glb_output_bank.sv       │ 32-wide write port per bank                       │
│                          │                                                    │
│ glb_weight_bank.sv       │ 32-wide read port per lane bank                   │
│                          │                                                    │
│ router_cluster.sv        │ RIN: 32-element vectors thay vì 16                │
│                          │ RWT: 32-element weight broadcast                   │
│                          │ RPS: 32-wide psum path                             │
│                          │                                                    │
│ swizzle_engine.sv        │ 32-wide re-layout engine                           │
│                          │                                                    │
│ local_arbiter.sv         │ ★ THAY ĐỔI LỚN: dual-RUNNING scheduling          │
│                          │ Priority: RUN0 = RUN1 > FILL > DRAIN > HOLD       │
│                          │ External port arb: FILL và DRAIN time-multiplex    │
│                          │                                                    │
│ tensor_dma.sv            │ AXI4 burst width 256b thay vì 128b                │
│                          │ Hoặc: 2 × 128b bursts per cycle                   │
│                          │                                                    │
│ accel_top.sv             │ AXI port 256b master; updated parameters           │
└──────────────────────────┴────────────────────────────────────────────────────┘
```

### 9.2. Modules KHÔNG thay đổi (critical for correctness)

```
★ KHÔNG THAY ĐỔI LOGIC TÍNH TOÁN:
  - pe_lane_mac: phép MAC INT8×INT8→INT32 GIỐNG HỆT (chỉ thêm lanes)
  - ppu_lite: bias + requant + SiLU LUT GIỐNG HỆT (chỉ thêm lanes)
  - column_reduce: sum 3 rows GIỐNG HỆT (chỉ rộng hơn)
  - comparator_tree: MAXPOOL logic GIỐNG HỆT
  - barrier_manager: dependency logic GIỐNG HỆT
  - tile_fsm: FSM states GIỐNG HỆT

→ Correctness of compute ĐƯỢC BẢO TOÀN vì:
  1. Cùng phép toán (MAC, requant, SiLU, maxpool) — chỉ parallelize nhiều hơn
  2. Cùng data type (INT8 in, INT32 psum, INT8 out)
  3. Cùng quantization parameters (scale, zp, M_int, shift)
  4. Cùng rounding mode (half_up)
  5. Kết quả INDEPENDENT giữa các lanes (no cross-lane interaction in MAC)
```

---

## 10. ĐẢM BẢO TÍNH ĐÚNG ĐẮN INFERENCE

### 10.1. Tại sao V2 cho kết quả GIỐNG HỆT V1 và Golden Python?

```
CHỨNG MINH: Scale-up lanes KHÔNG ảnh hưởng kết quả

Xét 1 output element y[cout, h_out, x_out]:

  y[cout, h_out, x_out] = PPU(Σ_{kh,kw,cin} x[cin, h_in+kh, x_out+kw] × w[cout, cin, kh, kw])

Trong V1 (LANES=16):
  - x_out ∈ {0, 1, ..., 15} xử lý song song trong cycle T
  - x_out ∈ {16, 17, ..., 31} xử lý trong cycle T+1

Trong V2 (LANES=32):
  - x_out ∈ {0, 1, ..., 31} xử lý song song trong CÙNG cycle T

Kết quả y[cout, h_out, x_out] KHÔNG PHỤ THUỘC vào lane nào xử lý nó,
vì mỗi lane tính HOÀN TOÀN ĐỘC LẬP (no cross-lane data dependency).

QED: V2 output === V1 output === Golden Python output (bit-exact)
```

### 10.2. Rủi ro thay đổi và cách kiểm soát

```
┌─────────────────────────┬──────────────────────────────────────────────────┐
│ Rủi ro                  │ Kiểm soát                                       │
├─────────────────────────┼──────────────────────────────────────────────────┤
│ DSP packing correction  │ Unit test: 2-MAC DSP output vs reference MAC    │
│ (unsigned offset error) │ Test ALL corner cases: (-128×-128), (-128×127), │
│                         │ (0×0), (127×127)                                │
│                         │                                                  │
│ 32-wide bank addressing │ Verify addr_gen cho W=20 (1 Wblk, 12 padding)  │
│ (padding elements)      │ Padding must fill with zp_x, not 0             │
│                         │                                                  │
│ Dual-RUNNING race cond. │ 2 RUNNING subs access DIFFERENT GLB_BANKs      │
│                         │ No shared memory → no race condition             │
│                         │ External port: FILL/DRAIN arbitrated by         │
│                         │ local_arbiter → serialized → no conflict         │
│                         │                                                  │
│ HOLD_SKIP rotation      │ Skip data integrity check after rotation         │
│                         │ Compare with Golden Python skip tensor dump      │
│                         │                                                  │
│ 256b AXI burst align    │ Ensure burst addresses 32B-aligned               │
│                         │ ARLEN calculation correct for wider bus           │
└─────────────────────────┴──────────────────────────────────────────────────┘
```

### 10.3. Verification flow (V2-specific additions)

```
Phase 0 (NEW): DSP packing unit test
  - Testbench drives all 65536 (x, w) INT8 pairs through 2-MAC DSP
  - Compare vs behavioral: for all a,b in [-128..127]: assert DSP_out == a*b
  - Must pass 100% (no tolerance)

Phase 1: Memory correctness (giống V1, nhưng LANES=32)
  - bank_input: verify modulo-3 với 32 subbanks
  - Wblk_total = ceil(W/32) thay vì ceil(W/16)
  - Padding positions [W..32*Wblk-1] must contain zp_x

Phase 2: Primitive unit tests (giống V1)
  - Golden Python numpy dumps → hex → testbench
  - Check: V2 output == V1 output == Golden Python (bit-exact)

Phase 3: Dual-RUNNING integration test (NEW)
  - Run 2 tiles simultaneously on same SC
  - Verify both outputs correct (no interference)
  - Check external port arbitration doesn't corrupt data

Phase 4: End-to-end L0-L22 (giống V1)
  - Input: X_int8[1,3,640,640] from Golden Python
  - Output: P3[1,64,80,80], P4[1,128,40,40], P5[1,256,20,20]
  - Compare bit-exact with Golden Python
```

---

## 11. SƠ ĐỒ KIẾN TRÚC V2 (TOP-LEVEL)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              accel_top_v2                                     │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────┐                │
│  │               controller_system (unchanged logic)        │                │
│  │  CSR/MMIO + desc_fetch + barrier_mgr + DMA + sched       │                │
│  └────────────────────────┬────────────────────────────────┘                │
│             cluster_cmd[4]│ cluster_sts[4]                                   │
│  ┌──────────────────────────────────────────────────────┐                   │
│  │                 Inter-cluster Fabrics                  │                   │
│  │  IACT_fabric(N/S/E/W 256b) + WGT_fabric(E/W 256b)   │  ← widened       │
│  │  PSUM_fabric(N/S, 32×32b)                             │  ← widened       │
│  └──────────┬───────────────┬──────────────┬────────────┘                   │
│             │               │              │                                  │
│  ┌──────────▼──┐   ┌────────▼────┐  ┌─────▼──────┐  ┌───────────┐         │
│  │ SC(0,0)     │   │ SC(0,1)     │  │ SC(1,0)    │  │ SC(1,1)   │         │
│  │ 4 subcluster│   │ 4 subcluster│  │ 4 subcluster│  │ 4 subcluster│        │
│  │ 2 RUNNING   │   │ 2 RUNNING   │  │ 2 RUNNING  │  │ 2 RUNNING │  ← NEW  │
│  │ Port0 256b  │   │ Port1 256b  │  │ Port2 256b │  │ Port3 256b│  ← wider│
│  └─────────────┘   └─────────────┘  └────────────┘  └───────────┘         │
│                                                                               │
│  ┌──────────────────┐  ┌───────────────────┐  ┌──────────────────────────┐ │
│  │  desc_ram         │  │  tensor_arena DMA  │  │  perf_mon / IRQ          │ │
│  │  (NET/LAYER/TILE) │  │  (AXI4 256b master)│  │  counters + CSR readout  │ │
│  └──────────────────┘  └───────────────────┘  └──────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 11.1. Subcluster V2 internal

```
ext_beat_in (256b) ──────────────────────────────────┐
                                                       │
cfg (shadow_regs) ────────┐                           │
                           ▼                           ▼
                    ┌──────────────────────────────────────┐
                    │              GLB_BANK_V2              │
                    │  bank_input[3]: 32 subbanks each     │  ← wider
                    │  bank_weight[3]: 32-wide read port   │  ← wider
                    │  bank_output[4]: 32-wide write port  │  ← wider
                    │  swizzle_engine_v2 (32-wide)         │
                    └────────────┬─────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────────────┐
                    │       ROUTER_CLUSTER_V2             │
                    │  RIN: 3 × 32-element vectors       │  ← wider
                    │  RWT: 3 × 32-element weight ports  │  ← wider
                    │  RPS: 4 × 32-element psum paths    │  ← wider
                    └────────────┬───────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────────────┐
                    │       WINDOW_GEN_V2                 │
                    │  Shift register: 32-wide × K_max   │  ← wider
                    └────────────┬───────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────────────┐
                    │       PE_CLUSTER_V2 (3×4×32)       │
                    │  12 PEs × 16 DSP48E1 (2 MAC/DSP)  │  ← 2×lanes
                    │  + correction LUTs per DSP pair    │
                    │  column_reduce: 32-wide             │
                    └────────────┬───────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────────────┐
                    │       PPU_V2                        │
                    │  32-lane requant + SiLU_LUT         │  ← wider
                    │  (2 cycles × 16-port LUT ROM,       │
                    │   or 32-port dual-read ROM)         │
                    └────────────┬───────────────────────┘
                                 │
                                 ▼
                    bank_output[4] / swizzle / ext_beat_out (256b)
```

---

## 12. PE_LANE_MAC V2: DSP PACKING DETAIL

### 12.1. Single PE (32 lanes, using 16 DSP48E1)

```
┌───────────────────────────────────────────────────────────────────┐
│ PE[row][col] — 32 lanes                                          │
│                                                                   │
│  x_in[31:0] (INT8 activations, 32 elements)                     │
│  w_in[31:0] (INT8 weights, 32 elements)                         │
│  psum_buf[31:0] (INT32 accumulators, 32 elements)               │
│                                                                   │
│  ┌─────────────┐  ┌─────────────┐       ┌─────────────┐         │
│  │ DSP_PAIR_0  │  │ DSP_PAIR_1  │  ...  │ DSP_PAIR_15 │         │
│  │ lane[0,1]   │  │ lane[2,3]   │       │ lane[30,31] │         │
│  │ 1 DSP48E1   │  │ 1 DSP48E1   │       │ 1 DSP48E1   │         │
│  │ + correct   │  │ + correct   │       │ + correct   │         │
│  └──────┬──────┘  └──────┬──────┘       └──────┬──────┘         │
│         │                │                      │                 │
│    psum[0], psum[1]  psum[2], psum[3]   psum[30], psum[31]      │
│         │                │                      │                 │
│  ┌──────▼────────────────▼──────────────────────▼──────────────┐ │
│  │                   psum_acc[31:0]                              │ │
│  │          (32 × INT32 accumulate registers)                   │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  Total per PE: 16 DSPs + ~400 LUTs (correction) + 32×32 FFs     │
│  Total per sub (12 PEs): 192 DSPs + ~4,800 LUTs                 │
└───────────────────────────────────────────────────────────────────┘
```

### 12.2. DSP_PAIR module

```systemverilog
module dsp_pair_int8 (
  input  logic        clk, rst_n, en,
  input  logic [7:0]  x_a, x_b,      // 2 signed INT8 activations (2 lanes)
  input  logic [7:0]  w,              // 1 signed INT8 weight (shared)
  output logic [31:0] mac_a, mac_b    // 2 INT32 accumulate outputs
);
  logic [7:0]  x_a_u, x_b_u, w_u;
  logic [24:0] dsp_A;
  logic [17:0] dsp_B;
  logic [42:0] dsp_P;
  logic [15:0] prod_a_u, prod_b_u;
  logic [31:0] correction_a, correction_b;

  assign x_a_u = x_a + 8'd128;
  assign x_b_u = x_b + 8'd128;
  assign w_u   = w   + 8'd128;

  assign dsp_A = {x_b_u, 9'b0, x_a_u};
  assign dsp_B = {10'b0, w_u};

  DSP48E1 #(/* OPMODE for A*B, accumulate */) dsp_inst (
    .CLK(clk), .A({5'b0, dsp_A[24:0]}), .B(dsp_B),
    .P(dsp_P), .CEP(en), .RSTP(rst_n)
  );

  assign prod_a_u = dsp_P[15:0];
  assign prod_b_u = dsp_P[32:17];

  assign correction_a = 32'(prod_a_u) - 32'(128) * (32'(x_a_u) + 32'(w_u)) + 32'd16384;
  assign correction_b = 32'(prod_b_u) - 32'(128) * (32'(x_b_u) + 32'(w_u)) + 32'd16384;

  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) begin
      mac_a <= '0;
      mac_b <= '0;
    end else if (en) begin
      mac_a <= mac_a + correction_a;
      mac_b <= mac_b + correction_b;
    end
endmodule
```

---

## 13. DUAL-RUNNING LOCAL_ARBITER V2

### 13.1. FSM states per subcluster

```systemverilog
typedef enum logic [2:0] {
  SUB_IDLE    = 3'h0,
  SUB_RUNNING = 3'h1,    // PE computing from local GLB
  SUB_FILLING = 3'h2,    // DMA → GLB (activation + weight)
  SUB_DRAINING= 3'h3,    // GLB → DMA (output activation)
  SUB_HOLD    = 3'h4     // holding skip tensor for barrier
} sub_state_e;
```

### 13.2. Role assignment logic

```systemverilog
always_comb begin
  // Priority: ensure 2 RUNNING at all times when tiles available
  num_running = count(sub_state == SUB_RUNNING);
  num_filling = count(sub_state == SUB_FILLING);

  // When a sub finishes RUNNING:
  if (sub_done_pulse) begin
    finished_sub → SUB_DRAINING;
    if (filled_sub_ready)
      filled_sub → SUB_RUNNING;    // promote FILLED → RUNNING
    // Find idle sub → SUB_FILLING (prefetch next tile)
    if (idle_sub_available && tiles_remaining)
      idle_sub → SUB_FILLING;
  end

  // External port allocation (time-multiplex):
  ext_port_grant =
    (num_filling > 0) ? FILL_FIRST :    // FILL has priority
    (num_draining > 0) ? DRAIN :         // then DRAIN
    IDLE;

  // Within FILL phase: alternate read bursts (weight / activation)
  // Within DRAIN phase: write bursts to DDR
end
```

### 13.3. Bandwidth sharing

```
External port: 256b/cycle @ 200 MHz = 6.4 GB/s per SC

Time budget between tile completions:
  Average tile compute: ~20 µs (varies 5-50 µs)
  Fill needed: ~12 µs (input + weight)
  Drain needed: ~5 µs (output)
  Fill + Drain: ~17 µs < 20 µs → fits in 1 tile compute window ✓

Edge case (small tiles at 20×20):
  Compute: ~10 µs
  Fill + Drain: ~8 µs → tight but works
  → temporal utilization ~80% for these tiles
```

---

## 14. DDR3 MEMORY MAP V2

```
DDR3 Base: 0x0000_0000  (unchanged layout, wider burst access)

├── 0x0000_0000 – 0x000F_FFFF (1 MB):    Descriptors (NET/LAYER/TILE)
├── 0x0010_0000 – 0x002F_FFFF (2 MB):    Weight arena
├── 0x0030_0000 – 0x0043_FFFF (1.25 MB): Input tensor X_int8[1,3,640,640]
├── 0x0050_0000 – 0x006F_FFFF (2 MB):    Activation double-buffer (ping/pong)
├── 0x0080_0000 – 0x009F_FFFF:           P3 output [1,64,80,80] = 409,600 B
├── 0x00A0_0000 – 0x00AF_FFFF:           P4 output [1,128,40,40] = 204,800 B
├── 0x00B0_0000 – 0x00B2_7FFF:           P5 output [1,256,20,20] = 102,400 B
├── 0x00C0_0000 – 0x00CF_FFFF:           Skip spill arena (F4 if needed)
└── 0x00D0_0000+:                         Debug / checkpoint buffers

AXI4 bus: 256b master @ 200 MHz
  Burst: ARLEN=15, ARSIZE=5 (32 bytes), burst = 16×32 = 512 B
  DDR3 effective BW: ~12.8 GB/s (unchanged, DDR3-1600)
  Total data per inference: ~43 MB → 43M/12.8G = 3.4 ms
  → Memory NOT bottleneck ✓
```

---

## 15. TỔNG KẾT & QUYẾT ĐỊNH

### 15.1. V2 Performance Card

```
╔══════════════════════════════════════════════════════════════════╗
║  YOLOv10n INT8 Accelerator V2 — Performance Card                ║
╠══════════════════════════════════════════════════════════════════╣
║  FPGA Target:    XC7VX690T                                      ║
║  Clock:          200 MHz (conservative) / 250 MHz (aggressive)  ║
║  Active MACs:    3,072 INT8 MACs/cycle                          ║
║  Peak:           614.4 GOPS @ 200 MHz / 768.0 GOPS @ 250 MHz   ║
║  Effective:      307.2 GOPS @ 200 MHz, 50% util                ║
║  ────────────────────────────────────────────────────────────── ║
║  Workload:       3.08 GMACs (YOLOv10n backbone+neck, 640×640)  ║
║  HW compute:     ~10.0 ms @ 200 MHz / ~8.0 ms @ 250 MHz       ║
║  HW FPS:         ~100 FPS @ 200 MHz / ~125 FPS @ 250 MHz       ║
║  ────────────────────────────────────────────────────────────── ║
║  End-to-end:     ~100 FPS (with x86 host or Zynq ARM)          ║
║  ────────────────────────────────────────────────────────────── ║
║  DSP48E1:        3,360 / 3,600 (93%)                            ║
║  BRAM36K:        752 / 1,470 (51%)                              ║
║  LUT6:           280K / 433K (65%)                              ║
║  FF:             320K / 866K (37%)                              ║
║  ────────────────────────────────────────────────────────────── ║
║  Correctness:    Bit-exact with Golden Python Phase 1           ║
║                  (same arithmetic, just wider parallelism)      ║
╚══════════════════════════════════════════════════════════════════╝
```

### 15.2. So sánh V1 vs V2

```
┌──────────────────────┬──────────────┬──────────────┬────────────┐
│ Metric               │ V1           │ V2           │ Tỷ lệ      │
├──────────────────────┼──────────────┼──────────────┼────────────┤
│ Active MACs          │ 768          │ 3,072        │ 4.0×       │
│ Effective GOPS       │ 69.1         │ 307.2        │ 4.4×       │
│ HW compute time      │ 44 ms        │ 10 ms        │ 4.4×       │
│ HW FPS               │ 23           │ 101          │ 4.4×       │
│ End-to-end FPS       │ 12-15        │ 100+         │ 6.7-8.3×   │
│ DSP usage            │ 85%          │ 93%          │ ↑          │
│ BRAM usage           │ 51%          │ 51%          │ =          │
│ LUT usage            │ ~45%         │ ~65%         │ ↑          │
│ RTL complexity       │ baseline     │ +30%         │ moderate   │
│ Correctness          │ bit-exact    │ bit-exact    │ preserved  │
└──────────────────────┴──────────────┴──────────────┴────────────┘
```

### 15.3. Recommended implementation path

```
Step 1: Implement V1 architecture FIRST (simpler, prove correctness)
  - LANES=16, 1 RUNNING per SC
  - Full verification against Golden Python
  - Establish baseline FPS (~23 FPS HW)

Step 2: Upgrade to V2 (parametric changes)
  - Change LANES parameter: 16 → 32
  - Widen all data paths
  - Add DSP packing (2 MAC/DSP)
  - Verify: V2 output === V1 output (bit-exact regression test)

Step 3: Enable dual-RUNNING scheduler
  - Modify local_arbiter for 2 RUNNING
  - Test concurrency: no data corruption
  - Measure actual utilization, tune tile sizes

Step 4: Clock optimization
  - Timing analysis at 200 MHz
  - If slack permits: push to 220-250 MHz
  - Re-verify at higher clock

Target: V2 @ 200-250 MHz → 100-125 FPS HW → >100 FPS end-to-end ✓
```

---

*V2 architecture kế thừa toàn bộ primitive set, descriptor format, và verification methodology từ V1.
Sự khác biệt DUY NHẤT là data parallelism (wider lanes) và scheduling (dual-RUNNING).
Phép toán arithmetic KHÔNG THAY ĐỔI → kết quả inference GIỐNG HỆT phần mềm.*


---
---

# ════════════════════════════════════════════════════════════════
# PHẦN IV — QUY TRÌNH HIỆN THỰC
# ════════════════════════════════════════════════════════════════

<a id='phần-iv1---quy-trình-hiện-thực'></a>

# PHẦN IV.1 — Quy Trình Hiện Thực Tăng Tốc Phần Cứng
> Nguồn: `SW_KLTN/HW_ACCELERATION_IMPL_FLOW.md`

---

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


---
---

# ════════════════════════════════════════════════════════════════
# PHẦN V — ĐẶC TẢ MODULE RTL
# ════════════════════════════════════════════════════════════════

<a id='phần-v1---đặc-tả-module-rtl'></a>

# PHẦN V.1 — Đặc Tả Toàn Bộ Module RTL
> Nguồn: `SW_KLTN/RTL_MODULE_SPEC.md`

---

# RTL MODULE SPECIFICATION – YOLOv10n INT8 Accelerator V2
## Danh Sách Module, Interface, Logic Chi Tiết Cho Prompt Verilog

> **Mục tiêu**: Mô tả ĐẦY ĐỦ mọi module RTL cần xây dựng, đủ chi tiết để prompt sinh Verilog.
> **Kiến trúc**: V2 (LANES=32, dual-RUNNING, 3,072 active MACs)
> **Tham chiếu**: `HW_ARCHITECTURE_V2_100FPS.md`, `HW_MAPPING_RESEARCH.md`

---

# PHẦN A: TỔNG QUAN CÂY MODULE

```
accel_top
├── controller_system
│   ├── csr_mmio
│   ├── desc_fetch_engine
│   │   └── axi_read_master
│   ├── barrier_manager
│   └── global_scheduler
│
├── supercluster_wrapper [×4]
│   ├── local_arbiter
│   └── subcluster_wrapper [×4 per SC]
│       ├── tile_fsm
│       ├── shadow_reg_file
│       ├── glb_bank
│       │   ├── glb_input_bank [×3]
│       │   ├── glb_weight_bank [×3]
│       │   ├── glb_output_bank [×4]
│       │   ├── metadata_ram
│       │   ├── addr_gen_input
│       │   ├── addr_gen_weight
│       │   └── addr_gen_output
│       │
│       ├── router_cluster
│       │   ├── rin_router [×3]
│       │   ├── rwt_router [×3]
│       │   └── rps_router [×4]
│       │
│       ├── window_gen
│       ├── pe_cluster
│       │   ├── dsp_pair_int8 [×12 PEs × 16 DSP pairs]
│       │   ├── column_reduce
│       │   └── comparator_tree (MAXPOOL mode)
│       │
│       ├── ppu
│       │   ├── bias_add_unit
│       │   ├── requant_unit
│       │   ├── silu_lut
│       │   ├── clamp_unit
│       │   └── ewise_add_unit
│       │
│       └── swizzle_engine
│
├── tensor_dma
│   ├── axi_read_master
│   └── axi_write_master
│
└── perf_mon
```

**Tổng số module files cần viết: ~35 files**

---

# PHẦN B: LEVEL 0 – PACKAGE DEFINITIONS

---

## M00. `accel_pkg.sv` — Hằng số và kiểu dữ liệu chung

**Mục đích**: Định nghĩa tất cả tham số toàn cục, kiểu enum, struct dùng chung bởi mọi module.

```systemverilog
package accel_pkg;

  // ═══════════ Compute Array Parameters ═══════════
  parameter int LANES          = 32;    // spatial positions parallel per PE
  parameter int PE_ROWS        = 3;     // reduction rows (matches 3×3 kernel)
  parameter int PE_COLS        = 4;     // output rows parallel
  parameter int NUM_PES        = PE_ROWS * PE_COLS; // = 12
  parameter int MACS_PER_SUB   = PE_ROWS * PE_COLS * LANES; // = 384
  parameter int DSP_PAIRS      = LANES / 2; // = 16 DSP48E1 per PE

  // ═══════════ Memory Parameters ═══════════
  parameter int INPUT_BANKS    = 3;     // h mod 3 banking
  parameter int OUTPUT_BANKS   = 4;     // PE_COLS output streams
  parameter int WEIGHT_BANKS   = 3;     // 3 reduction lanes
  parameter int PSUM_WIDTH     = 32;    // INT32 accumulator
  parameter int ACT_WIDTH      = 8;     // INT8 activation
  parameter int WEIGHT_WIDTH   = 8;     // INT8 weight
  parameter int BIAS_WIDTH     = 32;    // INT32 bias

  // ═══════════ System Parameters ═══════════
  parameter int SUPER_CLUSTERS   = 4;
  parameter int SUBS_PER_SC      = 4;
  parameter int ACTIVE_PER_SC    = 2;   // dual-RUNNING
  parameter int EXT_PORT_WIDTH   = 256; // bits, external memory port
  parameter int EXT_PORT_BYTES   = EXT_PORT_WIDTH / 8; // = 32
  parameter int AXI_ADDR_WIDTH   = 40;
  parameter int AXI_DATA_WIDTH   = 256;

  // ═══════════ Descriptor Parameters ═══════════
  parameter int DESC_WIDTH       = 512; // 64 bytes = 512 bits per descriptor
  parameter int MAX_LAYERS       = 32;
  parameter int MAX_TILES        = 4096;
  parameter int BARRIER_BITS     = 32;

  // ═══════════ SiLU LUT ═══════════
  parameter int SILU_LUT_DEPTH   = 256;
  parameter int SILU_LUT_WIDTH   = 8;

  // ═══════════ Derived ═══════════
  parameter int WBLK_MAX         = 20;  // ceil(640/32)
  parameter int CIN_TILE_MAX     = 256;
  parameter int COUT_TILE_MAX    = 256;
  parameter int H_TILE_MAX       = 80;

  // ═══════════ PE Mode Enum ═══════════
  typedef enum logic [3:0] {
    PE_RS3     = 4'h0,  // RS_DENSE_3x3 (conv 3×3, stride 1 or 2)
    PE_OS1     = 4'h1,  // OS_1x1 (pointwise conv)
    PE_DW3     = 4'h2,  // DW_3x3 (depthwise 3×3)
    PE_DW7     = 4'h3,  // DW_7x7_MULTIPASS (3 passes)
    PE_MP5     = 4'h4,  // MAXPOOL_5x5 (comparator tree)
    PE_GEMM    = 4'h5,  // GEMM_ATTN (matrix multiply)
    PE_PASS    = 4'h6   // bypass (MOVE, CONCAT, UPSAMPLE)
  } pe_mode_e;

  // ═══════════ Activation Mode ═══════════
  typedef enum logic [1:0] {
    ACT_NONE   = 2'h0,
    ACT_SILU   = 2'h1,
    ACT_RELU   = 2'h2,
    ACT_CLAMP  = 2'h3
  } act_mode_e;

  // ═══════════ Subcluster Role ═══════════
  typedef enum logic [2:0] {
    ROLE_IDLE     = 3'h0,
    ROLE_RUNNING  = 3'h1,
    ROLE_FILLING  = 3'h2,
    ROLE_DRAINING = 3'h3,
    ROLE_HOLD     = 3'h4
  } sc_role_e;

  // ═══════════ Tile FSM States ═══════════
  typedef enum logic [3:0] {
    TILE_IDLE          = 4'h0,
    TILE_LOAD_CFG      = 4'h1,
    TILE_PREFILL_WT    = 4'h2,
    TILE_PREFILL_IN    = 4'h3,
    TILE_WAIT_READY    = 4'h4,
    TILE_RUN_COMPUTE   = 4'h5,
    TILE_ACCUMULATE    = 4'h6,
    TILE_POST_PROCESS  = 4'h7,
    TILE_SWIZZLE_STORE = 4'h8,
    TILE_DONE          = 4'h9
  } tile_state_e;

  // ═══════════ Quant Mode ═══════════
  typedef enum logic [1:0] {
    QMODE_PER_TENSOR   = 2'h0,
    QMODE_PER_CHANNEL  = 2'h1,
    QMODE_NONE         = 2'h2
  } quant_mode_e;

  // ═══════════ PSUM/ACT Namespace ═══════════
  typedef enum logic {
    NS_PSUM = 1'b0,  // INT32 partial sum (intermediate)
    NS_ACT  = 1'b1   // INT8 activation (final after PPU)
  } namespace_e;

endpackage
```

---

## M01. `desc_pkg.sv` — Descriptor Struct Definitions

**Mục đích**: Định nghĩa các struct cho NET_DESC, LAYER_DESC, TILE_DESC, POST_PROFILE, ROUTER_PROFILE.

```systemverilog
package desc_pkg;
  import accel_pkg::*;

  typedef struct packed {
    logic [15:0] magic;          // 0x594F
    logic [7:0]  version;
    logic [7:0]  num_layers;
    logic [63:0] layer_table_base;
    logic [63:0] weight_arena_base;
    logic [63:0] act0_arena_base;  // ping
    logic [63:0] act1_arena_base;  // pong
    logic [63:0] aux_arena_base;   // skip buffer
  } net_desc_t;                    // 48 bytes used / 64 bytes padded

  typedef struct packed {
    logic [3:0]  template_id;      // pe_mode_e
    logic [4:0]  layer_id;
    logic [8:0]  cin_total;
    logic [8:0]  cout_total;
    logic [9:0]  hin, win;
    logic [9:0]  hout, wout;
    logic [3:0]  kh, kw;
    logic [2:0]  sh, sw;           // stride
    logic [3:0]  pad_top, pad_bot, pad_left, pad_right;
    logic [7:0]  tile_cin, tile_cout;
    logic [5:0]  tile_w_blks;
    logic [11:0] num_tile_hw;
    logic [3:0]  r_need, r_new, q_in, q_out;
    logic [3:0]  num_cin_pass, num_k_pass;
    logic [7:0]  router_profile_id;
    logic [7:0]  post_profile_id;
    logic [4:0]  src_in_tid, src_w_tid, src_skip_tid, dst_tid;
    logic [63:0] tile_table_offset;
    logic [15:0] layer_flags;       // [0]keep_on_chip [1]need_barrier [2]hold_skip_after
  } layer_desc_t;                   // 64 bytes

  typedef struct packed {
    logic [15:0] tile_id;
    logic [4:0]  layer_id;
    logic [3:0]  sc_mask;           // which SC processes this tile
    logic [9:0]  h_out0, wblk0;
    logic [8:0]  cin0, cout0;
    logic [5:0]  valid_h, valid_w;
    logic [3:0]  halo_top, halo_bot, halo_left, halo_right;
    logic [31:0] src_in_off;
    logic [31:0] src_w_off;
    logic [31:0] src_skip_off;
    logic [31:0] dst_off;
    logic [9:0]  in_base_h, in_base_c;
    logic [9:0]  out_base_h, out_base_c;
    logic [3:0]  first_cin_pass, num_cin_pass;
    logic [3:0]  first_k_pass, num_k_pass;
    logic [15:0] tile_flags;
    // tile_flags bits:
    // [0] FIRST_TILE      → reset psum
    // [1] LAST_TILE       → signal layer done
    // [2] EDGE_TILE       → needs padding
    // [3] HAS_SKIP        → read skip tensor
    // [4] NEED_SWIZZLE    → output → swizzle → next bank_input
    // [5] NEED_SPILL      → output spill to DDR
    // [6] BARRIER_BEFORE  → wait barrier before run
    // [7] BARRIER_AFTER   → signal barrier when done
    // [10] HOLD_SKIP_ROLE → this sub holds skip data
  } tile_desc_t;

  typedef struct packed {
    logic        bias_en;
    quant_mode_e quant_mode;
    act_mode_e   act_mode;
    logic        ewise_en;
    logic [31:0] bias_scale_offset;    // offset in weight arena to bias/scale tables
    logic [7:0]  concat_ch_offset;
    logic [1:0]  upsample_factor;
  } post_profile_t;

  typedef struct packed {
    logic [2:0]  rin_src [3];          // source select per RIN channel
    logic [3:0]  rin_dst_mask [3];
    logic [2:0]  rwt_src [3];
    logic        rwt_h_multicast;
    logic [1:0]  rps_accum_mode;       // none/local/vertical/writeback
    logic        concat_offset_mode;
    logic        upsample_dup_mode;
  } router_profile_t;

endpackage
```

---

## M02. `csr_pkg.sv` — CSR Register Map

**Mục đích**: Định nghĩa địa chỉ và cấu trúc các Control/Status Registers.

```systemverilog
package csr_pkg;
  parameter int CSR_CTRL          = 12'h000;  // [0]start [1]soft_reset [2]irq_clr
  parameter int CSR_STATUS        = 12'h004;  // [0]busy [1]done [2]irq [3]error
  parameter int CSR_VERSION       = 12'h008;
  parameter int CSR_CAP0          = 12'h00C;  // num_sc, subcl/sc, pe_rows, pe_cols, lanes
  parameter int CSR_NET_DESC_LO   = 12'h010;
  parameter int CSR_NET_DESC_HI   = 12'h014;
  parameter int CSR_LAYER_START   = 12'h018;
  parameter int CSR_LAYER_END     = 12'h01C;
  parameter int CSR_IRQ_MASK      = 12'h020;
  parameter int CSR_PERF_CTRL     = 12'h030;
  parameter int CSR_PERF_CYCLE_LO = 12'h034;
  parameter int CSR_PERF_CYCLE_HI = 12'h038;
  parameter int CSR_PERF_TILE_DONE= 12'h03C;
  parameter int CSR_PERF_STALL    = 12'h040;
  parameter int CSR_BARRIER_STATUS= 12'h050;
  parameter int CSR_DBG_LAYER_ID  = 12'h060;
  parameter int CSR_DBG_TILE_ID   = 12'h064;
endpackage
```

---

# PHẦN C: LEVEL 1 – MEMORY LEAF MODULES

---

## M03. `glb_input_bank.sv` — Input Activation SRAM Bank

**Mục đích**: 1 trong 3 input banks. Lưu activation INT8. Banking rule: `bank_id = h mod 3`.
Mỗi bank chứa 32 subbanks (1 per lane), mỗi subbank là SRAM đơn cổng.

**Parameters**:
```
LANES           = 32
SUBBANK_DEPTH   = 2048    // max entries per subbank
DATA_WIDTH      = 8       // INT8
```

**Ports**:
```systemverilog
module glb_input_bank #(
  parameter int LANES         = 32,
  parameter int SUBBANK_DEPTH = 2048,
  parameter int ADDR_W        = $clog2(SUBBANK_DEPTH)
)(
  input  logic                  clk, rst_n,

  // Write port (from DMA / swizzle engine during FILLING)
  input  logic                  wr_en,
  input  logic [ADDR_W-1:0]    wr_addr,         // shared across all 32 subbanks
  input  logic [LANES*8-1:0]   wr_data,         // 32 × INT8 = 256 bits
  input  logic [LANES-1:0]     wr_lane_mask,    // per-lane write enable (for edge tiles)

  // Read port (to router → window_gen → PE during RUNNING)
  input  logic                  rd_en,
  input  logic [ADDR_W-1:0]    rd_addr,
  output logic [LANES*8-1:0]   rd_data          // 32 × INT8 = 256 bits
);
```

**Logic xử lý**:
1. **32 subbank instances**: Mỗi subbank là `SRAM[SUBBANK_DEPTH][8]`.
2. **Write**: Khi `wr_en=1`, ghi `wr_data[(lane+1)*8-1:lane*8]` vào `subbank[lane][wr_addr]` nếu `wr_lane_mask[lane]=1`.
3. **Read**: Khi `rd_en=1`, đọc `subbank[lane][rd_addr]` cho tất cả 32 lanes.
4. **Xilinx implementation**: Mỗi subbank map lên 1-2 BRAM36K (tuỳ SUBBANK_DEPTH).

---

## M04. `glb_weight_bank.sv` — Weight SRAM Bank

**Mục đích**: 1 trong 3 weight banks. Mỗi bank lưu weight cho 1 kernel row (reduction lane).
Có staging FIFO 8-entry để prefetch weight pass kế tiếp.

**Ports**:
```systemverilog
module glb_weight_bank #(
  parameter int LANES         = 32,
  parameter int BANK_DEPTH    = 1024,
  parameter int FIFO_DEPTH    = 8
)(
  input  logic                 clk, rst_n,

  // Write port (from DMA during FILLING)
  input  logic                 wr_en,
  input  logic [$clog2(BANK_DEPTH)-1:0] wr_addr,
  input  logic [LANES*8-1:0]  wr_data,          // 32 × INT8 weights

  // Read port (to PE via router)
  input  logic                 rd_en,
  input  logic [$clog2(BANK_DEPTH)-1:0] rd_addr,
  output logic [LANES*8-1:0]  rd_data,

  // Staging FIFO interface (prefetch for next pass)
  input  logic                 fifo_push,
  input  logic [LANES*8-1:0]  fifo_din,
  input  logic                 fifo_pop,
  output logic [LANES*8-1:0]  fifo_dout,
  output logic                 fifo_empty, fifo_full
);
```

**Logic**:
1. SRAM `[BANK_DEPTH][LANES*8]` dùng BRAM.
2. FIFO 8-entry dùng distributed RAM hoặc SRL16.
3. Trong mode RS_DENSE_3x3: bank[0] = kernel row 0, bank[1] = row 1, bank[2] = row 2.
4. Trong mode OS_1x1: bank[0..2] = 3 Cin slices parallel.
5. Trong mode DW_7x7 pass k: load 3 kernel rows per pass vào 3 banks.

---

## M05. `glb_output_bank.sv` — Output SRAM Bank (Dual-mode PSUM/ACT)

**Mục đích**: 1 trong 4 output banks. Dual-mode: lưu INT32 partial sums HOẶC INT8 activations.

**Ports**:
```systemverilog
module glb_output_bank #(
  parameter int LANES      = 32,
  parameter int BANK_DEPTH = 512
)(
  input  logic                  clk, rst_n,

  // Write port
  input  logic                  wr_en,
  input  logic [$clog2(BANK_DEPTH)-1:0] wr_addr,
  input  accel_pkg::namespace_e wr_ns,           // PSUM (32b) or ACT (8b)
  input  logic [LANES*32-1:0]  wr_data_psum,     // 32 × INT32 = 1024 bits (PSUM mode)
  input  logic [LANES*8-1:0]   wr_data_act,      // 32 × INT8  = 256 bits  (ACT mode)

  // Read port
  input  logic                  rd_en,
  input  logic [$clog2(BANK_DEPTH)-1:0] rd_addr,
  input  accel_pkg::namespace_e rd_ns,
  output logic [LANES*32-1:0]  rd_data_psum,
  output logic [LANES*8-1:0]   rd_data_act
);
```

**Logic**:
1. **PSUM mode**: SRAM `[BANK_DEPTH][LANES*32]` — 32 INT32 values = 128 bytes per entry. Sử dụng cho các pass trung gian (accumulate partial sums qua Cin pass / kernel pass).
2. **ACT mode**: SRAM `[BANK_DEPTH][LANES*8]` — 32 INT8 values = 32 bytes per entry. Sử dụng cho kết quả cuối cùng sau PPU.
3. Hai namespace dùng chung address space nhưng khác data width → implement bằng BRAM wide + byte-enable.

---

## M06. `metadata_ram.sv` — Slot Validity & Ring Pointers

**Mục đích**: Quản lý valid bits cho input/output slots, ring buffer pointers.

**Ports**:
```systemverilog
module metadata_ram #(
  parameter int NUM_SLOTS = 16,  // max slots in ring buffer
  parameter int META_BITS = 32   // per-slot metadata
)(
  input  logic                           clk, rst_n,
  input  logic                           clear_all,
  // Write
  input  logic                           set_valid,
  input  logic [$clog2(NUM_SLOTS)-1:0]  set_slot_id,
  input  logic [META_BITS-1:0]          set_meta,     // base_h, base_c
  // Read
  input  logic [$clog2(NUM_SLOTS)-1:0]  query_slot_id,
  output logic                           query_valid,
  output logic [META_BITS-1:0]          query_meta,
  // Ring management
  input  logic                           advance_ring,
  output logic [$clog2(NUM_SLOTS)-1:0]  ring_head,
  output logic [$clog2(NUM_SLOTS)-1:0]  ring_tail,
  output logic                           ring_full, ring_empty
);
```

**Logic**: Circular buffer management cho row-slot rotation. Khi `advance_ring`: `head = (head + 1) mod NUM_SLOTS`, invalidate oldest slot.

---

# PHẦN D: LEVEL 2 – ADDRESS GENERATORS

---

## M07. `addr_gen_input.sv` — Input Address Generator

**Mục đích**: Tính physical address từ logical (h, w, c) cho bank_input.

**Ports**:
```systemverilog
module addr_gen_input #(
  parameter int LANES     = 32,
  parameter int MAX_WIDTH = 640,
  parameter int MAX_CIN   = 256
)(
  input  logic                clk, rst_n,

  // Configuration (from shadow regs / tile descriptor)
  input  logic [9:0]         cfg_win,           // input width
  input  logic [8:0]         cfg_cin_tile,      // Cin tile size
  input  logic [3:0]         cfg_q_in,          // Q_in (circular slots)
  input  logic [3:0]         cfg_stride,
  input  logic [3:0]         cfg_pad_mode,
  input  logic signed [7:0]  cfg_zp_x,          // zero-point for padding

  // Request
  input  logic               req_valid,
  input  logic [9:0]         req_h,             // row in input
  input  logic [9:0]         req_w,             // column (start of lane block)
  input  logic [8:0]         req_c,             // channel

  // Output
  output logic               out_valid,
  output logic [1:0]         out_bank_id,        // h mod 3 → {0,1,2}
  output logic [15:0]        out_addr,           // physical address in subbank
  output logic               out_is_padding,     // true if (h,w) is in padding region
  output logic signed [7:0]  out_pad_value       // zp_x for padding positions
);
```

**Logic**:
```
bank_id    = h mod 3
row_slot   = (h / 3) mod q_in
wblk       = w / LANES
addr       = (row_slot × cfg_cin_tile + c) × wblk_total + wblk

is_padding = (h < pad_top) OR (h >= hin - pad_bot) OR (w < pad_left) OR (w >= win - pad_right)
pad_value  = cfg_zp_x   // CRITICAL: phải pad bằng zp_x, KHÔNG PHẢI 0
```

---

## M08. `addr_gen_weight.sv` — Weight Address Generator

**Mục đích**: Tính address cho weight data trong bank_weight.

**Ports**:
```systemverilog
module addr_gen_weight #(
  parameter int LANES = 32
)(
  input  logic              clk, rst_n,
  input  accel_pkg::pe_mode_e cfg_mode,
  input  logic [8:0]        cfg_cin_tile, cfg_cout_tile,

  input  logic              req_valid,
  input  logic [1:0]        req_kr,           // kernel row (0-2 for RS3/DW3, 0-6 for DW7)
  input  logic [8:0]        req_cin,
  input  logic [8:0]        req_cout,

  output logic              out_valid,
  output logic [1:0]        out_bank_id,      // kr mod 3 for reduction lane
  output logic [15:0]       out_addr
);
```

**Logic per mode**:
```
RS_DENSE_3x3:
  bank_id = kr (0,1,2 for 3 kernel rows)
  addr    = cout × cin_tile × kw_total + cin × kw_total + kw
  
OS_1x1:
  bank_id = cin_slice (0,1,2 for 3 Cin chunks)
  addr    = cout × cin_per_slice + cin_within_slice

DW_3x3:
  bank_id = kr (0,1,2)
  addr    = channel × kw + kw_idx     (groups=Cin, no cross-channel)

DW_7x7_MULTIPASS:
  pass 0: bank[0..2] = kernel rows 0-2
  pass 1: bank[0..2] = kernel rows 3-5
  pass 2: bank[0]    = kernel row 6, bank[1..2] unused
```

---

## M09. `addr_gen_output.sv` — Output Address Generator

**Mục đích**: Tính address cho write results vào bank_output.

**Ports**:
```systemverilog
module addr_gen_output #(
  parameter int LANES = 32
)(
  input  logic              clk, rst_n,
  input  logic [3:0]        cfg_stride_h,
  input  logic [3:0]        cfg_q_out,
  input  logic [8:0]        cfg_cout_tile,

  input  logic              req_valid,
  input  logic [9:0]        req_h_out,
  input  logic [9:0]        req_w_out,
  input  logic [8:0]        req_cout,

  output logic              out_valid,
  output logic [1:0]        out_bank_id,       // pe_col index (0-3)
  output logic [15:0]       out_addr
);
```

**Logic**:
```
obank  = pe_col                                    // output bank = PE column index
oslot  = (h_out / (PE_COLS × stride_h)) mod q_out
addr   = (oslot × cout_tile + cout) × wblk_out + wblk
```

---

# PHẦN E: LEVEL 3 – COMPUTE PRIMITIVES

---

## M10. `dsp_pair_int8.sv` — Dual INT8 MAC in 1 DSP48E1

**Mục đích**: Module cơ bản nhất — 2 phép INT8×INT8→INT32 MAC dùng 1 DSP48E1.

**Ports**:
```systemverilog
module dsp_pair_int8 (
  input  logic              clk, rst_n,
  input  logic              en,            // compute enable
  input  logic              clear,         // reset accumulator (FIRST_TILE)
  input  logic signed [7:0] x_a,           // activation lane 2i
  input  logic signed [7:0] x_b,           // activation lane 2i+1
  input  logic signed [7:0] w,             // shared weight
  output logic signed [31:0] psum_a,       // accumulated result lane 2i
  output logic signed [31:0] psum_b        // accumulated result lane 2i+1
);
```

**Logic chi tiết**:
```
Pipeline stage 1 (unsigned conversion):
  x_a_u = x_a + 128    // signed [-128,127] → unsigned [0,255]
  x_b_u = x_b + 128
  w_u   = w   + 128

Pipeline stage 2 (DSP48E1 multiply):
  dsp_A[24:0] = {x_b_u[7:0], 9'b0, x_a_u[7:0]}    // pack 2 activations
  dsp_B[17:0] = {10'b0, w_u[7:0]}
  dsp_P[42:0] = dsp_A × dsp_B                        // single DSP48E1 multiply

Pipeline stage 3 (extract & correct):
  prod_a_u = dsp_P[15:0]                              // a_u × w_u
  prod_b_u = dsp_P[32:17]                             // b_u × w_u
  
  // Reverse unsigned offset: (a+128)(w+128) = a×w + 128a + 128w + 16384
  signed_prod_a = prod_a_u - 128×(x_a_u + w_u) + 16384   // = x_a × w
  signed_prod_b = prod_b_u - 128×(x_b_u + w_u) + 16384   // = x_b × w

Pipeline stage 4 (accumulate):
  if (clear) psum_a <= signed_prod_a;  else psum_a <= psum_a + signed_prod_a;
  if (clear) psum_b <= signed_prod_b;  else psum_b <= psum_b + signed_prod_b;
```

**Overflow safety**: Max accumulation: `9 × 256 × 127 × 127 = 37,064,529 < 2^31` ✓

---

## M11. `pe_unit.sv` — Single Processing Element (32 lanes)

**Mục đích**: 1 PE = 16 DSP pairs = 32 lanes. Thực hiện 32 MAC operations per cycle.

**Ports**:
```systemverilog
module pe_unit #(
  parameter int LANES = 32
)(
  input  logic              clk, rst_n,
  input  logic              en,
  input  logic              clear_psum,       // from FIRST_TILE flag
  input  accel_pkg::pe_mode_e mode,

  // Activation input (32 INT8 values per cycle)
  input  logic signed [7:0] x_in [LANES],

  // Weight input (shared per DSP pair for RS3/DW/GEMM; broadcast in OS1)
  input  logic signed [7:0] w_in [LANES],

  // Partial sum output (32 INT32 values)
  output logic signed [31:0] psum_out [LANES],
  output logic               psum_valid
);
```

**Logic**:
```
Instantiate 16 dsp_pair_int8:
  for i in 0..15:
    dsp_pair[i].x_a = x_in[2*i]
    dsp_pair[i].x_b = x_in[2*i + 1]
    dsp_pair[i].w   = w_in[2*i]      // RS3/DW/GEMM: shared by lanes [2*i] and [2*i+1]
                                       // OS1: w broadcast to all lanes
    psum_out[2*i]   = dsp_pair[i].psum_a
    psum_out[2*i+1] = dsp_pair[i].psum_b

Mode-specific behavior:
  RS_DENSE_3x3: x_in = spatial window tap, w_in[2*i] shared across lanes [2*i] and [2*i+1]
  OS_1x1:       x_in = activation, w_in = same weight broadcast to all lanes
  DW_3x3:       x_in = spatial tap, w_in = per-channel weight (groups=C)
  MAXPOOL:      MAC disabled, use comparator_tree instead
  GEMM:         x_in = matrix row tile, w_in = matrix col tile (transposed)
```

---

## M12. `window_gen.sv` — Spatial Window Tap Generator

**Mục đích**: Tạo sliding window K=1/3/5/7 từ input vector stream. Mỗi cycle nhận 1 vector 32-wide, output K vectors (taps).

**Ports**:
```systemverilog
module window_gen #(
  parameter int LANES = 32,
  parameter int K_MAX = 7      // max kernel width
)(
  input  logic                clk, rst_n,
  input  logic [2:0]          cfg_kw,          // kernel width: 1,3,5,7
  input  logic                shift_in_valid,
  input  logic signed [7:0]  shift_in [LANES], // new input vector (1 row, 32 cols)

  output logic                taps_valid,
  output logic signed [7:0]  taps [K_MAX][LANES]  // K tap vectors, each 32-wide
);
```

**Logic**:
```
Internal: shift_reg[K_MAX][LANES] — shift register chain

Every cycle when shift_in_valid:
  shift_reg[K_MAX-1] = shift_reg[K_MAX-2]
  ...
  shift_reg[1] = shift_reg[0]
  shift_reg[0] = shift_in

Output taps selection based on cfg_kw:
  K1: taps[0] = shift_reg[0]
  K3: taps[0..2] = shift_reg[0..2]
  K5: taps[0..4] = shift_reg[0..4]
  K7: taps[0..6] = shift_reg[0..6]

taps_valid asserted when pipeline has accumulated enough rows (≥ cfg_kw entries).
```

---

## M13. `column_reduce.sv` — Cross-Row Partial Sum Reduction

**Mục đích**: Cộng kết quả từ 3 PE rows → 1 psum vector per PE column.

**Ports**:
```systemverilog
module column_reduce #(
  parameter int LANES   = 32,
  parameter int PE_ROWS = 3,
  parameter int PE_COLS = 4
)(
  input  logic               clk, rst_n,
  input  logic               en,
  input  accel_pkg::pe_mode_e mode,

  // From PE array: [PE_ROWS][PE_COLS] × LANES INT32 values
  input  logic signed [31:0] pe_psum [PE_ROWS][PE_COLS][LANES],

  // Reduced output: [PE_COLS] × LANES INT32 values
  output logic signed [31:0] col_psum [PE_COLS][LANES],
  output logic               col_valid
);
```

**Logic**:
```
For RS_DENSE_3x3, OS_1x1, GEMM:
  col_psum[col][lane] = Σ_{row=0..2} pe_psum[row][col][lane]
  // Sum 3 kernel rows → 1 psum per output position

For DW_3x3, DW_7x7:
  col_psum[col][lane] = Σ_{row=0..2} pe_psum[row][col][lane]
  // Same sum but per-CHANNEL (groups mode, no cross-channel)

For MAXPOOL_5x5:
  col_psum bypassed → use comparator_tree output instead
```

---

## M14. `comparator_tree.sv` — Max Comparator for MAXPOOL

**Mục đích**: Tìm max trong 25 input (5×5 window) per lane. Dùng cho MAXPOOL_5x5.

**Ports**:
```systemverilog
module comparator_tree #(
  parameter int LANES      = 32,
  parameter int NUM_INPUTS = 25    // 5×5 window
)(
  input  logic               clk, rst_n,
  input  logic               en,
  input  logic signed [7:0] data_in [NUM_INPUTS][LANES],  // 25 taps × 32 lanes
  output logic signed [7:0] max_out [LANES],               // 32 max values
  output logic               max_valid
);
```

**Logic**:
```
Tree reduction (pipelined 3 stages for 25→1):
  Stage 1: 25 inputs → 13 (compare pairs, 1 odd passes through)
  Stage 2: 13 → 7
  Stage 3: 7 → 4
  Stage 4: 4 → 2
  Stage 5: 2 → 1

Per lane: independent max-tree (no cross-lane interaction).
Latency: 5 cycles (pipelined).
Scale/zp: pass-through (maxpool preserves quantization domain).
```

---

## M15. `pe_cluster.sv` — Full PE Array (3×4×32)

**Mục đích**: Wrapper quanh 12 PE units + column_reduce + comparator_tree. Module tính toán chính của mỗi subcluster.

**Ports**:
```systemverilog
module pe_cluster #(
  parameter int LANES   = 32,
  parameter int PE_ROWS = 3,
  parameter int PE_COLS = 4
)(
  input  logic                clk, rst_n,
  input  logic                en,
  input  logic                clear_psum,
  input  accel_pkg::pe_mode_e mode,

  // Activation taps from window_gen (per PE row)
  input  logic signed [7:0]  act_taps [PE_ROWS][LANES],

  // Weight data from router (per PE row, broadcast to PE cols)
  input  logic signed [7:0]  wgt_data [PE_ROWS][LANES],

  // Psum input (for multi-pass accumulation from bank_output)
  input  logic signed [31:0] psum_in [PE_COLS][LANES],
  input  logic               psum_in_valid,

  // Output
  output logic signed [31:0] psum_out [PE_COLS][LANES],
  output logic               psum_out_valid,
  output logic               last_pass,         // from tile_fsm

  // MAXPOOL output (bypass psum path)
  output logic signed [7:0]  pool_out [LANES],
  output logic               pool_out_valid
);
```

**Logic interne**:
```
Instantiate PE_ROWS × PE_COLS = 12 pe_unit instances.

Weight routing per mode:
  RS3:  wgt_data[row] broadcast to all PE_COLS in that row
  OS1:  wgt_data[row] = Cin_slice weights, broadcast to PE_COLS
  DW3:  wgt_data[row] = per-channel weight, each col = different channel group
  GEMM: similar to OS1 but matrix tiles

Psum accumulation:
  if (psum_in_valid): load psum_in to PE accumulators (multi-pass)
  Column reduce: sum 3 rows → col_psum[PE_COLS][LANES]

  if (mode == PE_MP5):
    Gather 25 taps from window_gen → comparator_tree → pool_out
    psum path inactive
  else:
    psum_out = col_psum (after cross-row reduction)
```

---

# PHẦN F: LEVEL 3 – POST-PROCESSING UNIT

---

## M16. `ppu.sv` — Post-Processing Unit (Top Wrapper)

**Mục đích**: Xử lý hậu kỳ: bias + requant + activation + clamp + ewise_add. Pipeline 4-stage.

**Ports**:
```systemverilog
module ppu #(
  parameter int LANES = 32
)(
  input  logic                  clk, rst_n,
  input  logic                  en,

  // Configuration
  input  desc_pkg::post_profile_t cfg_post,
  input  accel_pkg::pe_mode_e     cfg_mode,

  // PSUM input (INT32 from PE cluster, per PE column, process 1 column at a time)
  input  logic signed [31:0]   psum_in [LANES],
  input  logic                  psum_valid,

  // Per-channel parameters (loaded from weight arena via DMA)
  input  logic signed [31:0]   bias_val [LANES],     // INT32 bias per output channel
  input  logic signed [31:0]   m_int [LANES],        // fixed-point multiplier M_int
  input  logic [5:0]           shift [LANES],         // right shift amount
  input  logic signed [7:0]    zp_out,               // output zero-point

  // SiLU LUT (preloaded)
  input  logic signed [7:0]    silu_lut [256],

  // Element-wise add input (for skip connection / residual)
  input  logic signed [7:0]    ewise_in [LANES],
  input  logic                  ewise_valid,

  // Output (INT8)
  output logic signed [7:0]    act_out [LANES],
  output logic                  act_valid
);
```

**Pipeline stages**:
```
Stage 1 — Bias Add:
  biased[lane] = psum_in[lane] + bias_val[lane]      // INT32 + INT32 → INT32

Stage 2 — Fixed-Point Requant:
  y_raw[lane] = (biased[lane] × m_int[lane]) >>> shift[lane]   // arithmetic right shift
  // Rounding: half_up → add (1 << (shift-1)) before shift

Stage 3 — Activation:
  switch (cfg_post.act_mode):
    ACT_SILU: idx = clamp(y_raw + 128, 0, 255); y_act = silu_lut[idx]
    ACT_RELU: y_act = (y_raw > 0) ? y_raw : 0
    ACT_NONE: y_act = y_raw
    ACT_CLAMP: y_act = y_raw (clamped in stage 4)

Stage 4 — Clamp + Ewise Add:
  if (cfg_post.ewise_en):
    y_add = y_act + ewise_in[lane]     // element-wise add (skip connection)
  else:
    y_add = y_act
  act_out[lane] = clamp(y_add + zp_out, -128, 127)
```

---

## M17. `silu_lut.sv` — SiLU Lookup Table ROM

**Mục đích**: 256-entry INT8 ROM preloaded với SiLU(x) values. Hỗ trợ 32 concurrent reads.

**Ports**:
```systemverilog
module silu_lut #(
  parameter int LANES = 32
)(
  input  logic                clk,
  // Load interface (preload from descriptor/DMA)
  input  logic                load_en,
  input  logic [7:0]          load_addr,
  input  logic signed [7:0]  load_data,
  // Lookup interface (32 parallel reads)
  input  logic [7:0]          idx [LANES],
  output logic signed [7:0]  out [LANES]
);
```

**Logic**:
```
ROM[256] of INT8, precomputed:
  For each q ∈ [-128..127]:
    x_float = (q - zp_out) × scale_out
    silu_float = x_float × sigmoid(x_float)
    ROM[q + 128] = clamp(round(silu_float / scale_out) + zp_out, -128, 127)

32 parallel reads: dùng 32-port distributed RAM
  hoặc: 2 cycles × 16-port BRAM (latency tradeoff)
  hoặc: replicate ROM 32 lần (LUT-based, ~8K LUTs)
```

---

# PHẦN G: LEVEL 5 – DATA MOVEMENT

---

## M18. `router_cluster.sv` — Data Routing Hub

**Mục đích**: Điều hướng dữ liệu giữa GLB banks ↔ PE cluster ↔ PPU ↔ swizzle.
3 sub-routers: RIN (activation), RWT (weight), RPS (psum/output).

**Ports**:
```systemverilog
module router_cluster #(
  parameter int LANES = 32
)(
  input  logic                     clk, rst_n,
  input  desc_pkg::router_profile_t cfg_profile,
  input  accel_pkg::pe_mode_e       cfg_mode,

  // ═══ RIN: Activation Router (3 channels → 3 PE rows) ═══
  input  logic [LANES*8-1:0]      bank_input_rd [3],     // from 3 input banks
  input  logic [LANES*8-1:0]      neighbor_in [4],        // N/S/E/W neighbors
  input  logic [LANES*8-1:0]      swizzle_in,             // from swizzle engine
  output logic signed [7:0]       pe_act [3][LANES],      // to 3 PE rows

  // ═══ RWT: Weight Router (3 channels → 3 PE rows) ═══
  input  logic [LANES*8-1:0]      bank_weight_rd [3],
  output logic signed [7:0]       pe_wgt [3][LANES],      // to 3 PE rows

  // ═══ RPS: Psum/Output Router (4 channels from PE cols) ═══
  input  logic signed [31:0]      pe_psum [4][LANES],     // from PE cluster
  output logic [LANES*32-1:0]     bank_output_wr [4],     // to 4 output banks
  output logic [LANES*8-1:0]      ppu_in,                  // to PPU
  output logic [LANES*8-1:0]      neighbor_out [4],        // to N/S/E/W neighbors

  // ═══ Bypass paths (MOVE, CONCAT, UPSAMPLE) ═══
  input  logic                     bypass_en,
  input  logic [LANES*8-1:0]      bypass_data,
  output logic [LANES*8-1:0]      bypass_out
);
```

**Logic per mode**:
```
RS_DENSE_3x3 / OS_1x1:
  RIN: bank_input[0..2] → pe_act[0..2] (1-to-1 per row)
  RWT: bank_weight[0..2] → pe_wgt[0..2] → broadcast to 4 PE cols
  RPS: pe_psum[0..3] → bank_output[0..3]

DW_3x3 / DW_7x7:
  RIN: same as RS3
  RWT: same but per-channel (no cross-channel broadcast)
  RPS: same as RS3

CONCAT:
  bypass_en = 1
  Route: bank_input (tensor A channels) → bypass → bank_output
         then bank_input (tensor B channels, offset) → bypass → bank_output
  No PE/PPU involvement

UPSAMPLE:
  bypass_en = 1
  Route: bank_input → swizzle_engine (address duplication) → bank_output

MOVE:
  bypass_en = 1
  Direct: bank_input → bypass → bank_output (or → external DMA)
```

---

## M19. `swizzle_engine.sv` — Tensor Layout Transform

**Mục đích**: Biến đổi layout giữa bank_output → bank_input cho layer kế tiếp.
Xử lý UPSAMPLE_NEAREST, CONCAT channel offset, re-layout.

**Ports**:
```systemverilog
module swizzle_engine #(
  parameter int LANES = 32
)(
  input  logic                clk, rst_n,
  input  logic                en,
  input  accel_pkg::pe_mode_e mode,

  // Configuration
  input  logic [1:0]          cfg_upsample_factor,  // 0=none, 1=2×, 2=4× (unused)
  input  logic [8:0]          cfg_concat_ch_offset,  // channel offset for B in concat
  input  logic [9:0]          cfg_src_h, cfg_src_w, cfg_src_c,
  input  logic [9:0]          cfg_dst_h, cfg_dst_w, cfg_dst_c,
  input  logic [3:0]          cfg_dst_q_in,         // Q_in for next layer's bank_input

  // Source: read from bank_output
  output logic                src_rd_en,
  output logic [15:0]         src_rd_addr,
  output logic [1:0]          src_rd_bank,
  input  logic [LANES*8-1:0]  src_rd_data,

  // Destination: write to bank_input (for next layer)
  output logic                dst_wr_en,
  output logic [15:0]         dst_wr_addr,
  output logic [1:0]          dst_wr_bank,    // h_dst mod 3
  output logic [LANES*8-1:0]  dst_wr_data,
  output logic [LANES-1:0]    dst_wr_mask,

  output logic                done
);
```

**Logic per mode**:
```
NORMAL (layer chaining):
  for each output element (h_out, w, c_out):
    ib_next   = h_out mod 3
    slot_next = (h_out / 3) mod cfg_dst_q_in
    wblk      = w / LANES
    addr      = (slot_next × c_tile_next + c_out) × wblk_total_next + wblk
    bank_input_next[ib_next][addr] = bank_output_current[element]

UPSAMPLE_NEAREST (scale=2):
  for each src element (h, w, c):
    write to 4 dst positions:
      (2h,   2w)   → bank_input[(2h) mod 3][addr(2h, 2w, c)]
      (2h,   2w+1) → bank_input[(2h) mod 3][addr(2h, 2w+1, c)]
      (2h+1, 2w)   → bank_input[(2h+1) mod 3][addr(2h+1, 2w, c)]
      (2h+1, 2w+1) → bank_input[(2h+1) mod 3][addr(2h+1, 2w+1, c)]
  → 4 writes per source read → throughput = 4 cycles/element

CONCAT:
  Tensor A: write channels [0..C_A-1] normally
  Tensor B: write channels [C_A..C_A+C_B-1] with cfg_concat_ch_offset = C_A
  Address: same (h,w) but channel offset applied
```

---

# PHẦN H: LEVEL 6 – CONTROL MODULES

---

## M20. `shadow_reg_file.sv` — Tile Descriptor Shadow Registers

**Mục đích**: Capture tile descriptor fields vào pipeline registers. Cung cấp cấu hình ổn định cho PE cluster suốt thời gian compute 1 tile.

**Ports**:
```systemverilog
module shadow_reg_file (
  input  logic              clk, rst_n,
  input  logic              load,           // pulse: capture from tile_desc
  input  desc_pkg::tile_desc_t  tile_desc,
  input  desc_pkg::layer_desc_t layer_desc,
  input  desc_pkg::post_profile_t post_profile,
  input  desc_pkg::router_profile_t router_profile,

  // Output: stable configuration signals
  output accel_pkg::pe_mode_e   o_mode,
  output logic [8:0]            o_cin_tile, o_cout_tile,
  output logic [9:0]            o_hin, o_win, o_hout, o_wout,
  output logic [3:0]            o_kh, o_kw, o_sh, o_sw,
  output logic [3:0]            o_pad_top, o_pad_bot, o_pad_left, o_pad_right,
  output logic [3:0]            o_r_need, o_q_in, o_q_out,
  output logic [3:0]            o_num_cin_pass, o_num_k_pass,
  output logic [15:0]           o_tile_flags,
  output logic signed [7:0]     o_zp_x,
  output desc_pkg::post_profile_t   o_post,
  output desc_pkg::router_profile_t o_router
);
```

---

## M21. `tile_fsm.sv` — Tile Execution FSM

**Mục đích**: Điều khiển toàn bộ quá trình thực thi 1 tile: load config → fill data → compute passes → post-process → store output.

**Ports**:
```systemverilog
module tile_fsm (
  input  logic              clk, rst_n,

  // Tile descriptor input (from global_scheduler)
  input  logic              tile_valid,
  input  desc_pkg::tile_desc_t  tile_desc,
  input  desc_pkg::layer_desc_t layer_desc,
  output logic              tile_accept,

  // GLB control
  output logic              glb_wr_en,        // enable writes to GLB during PREFILL
  output logic              glb_rd_en,        // enable reads from GLB during COMPUTE
  
  // PE cluster control
  output logic              pe_en,
  output logic              pe_clear_psum,
  output accel_pkg::pe_mode_e pe_mode,

  // PPU control
  output logic              ppu_en,
  output logic              ppu_last_pass,     // trigger PPU after final accumulation

  // Swizzle control
  output logic              swizzle_start,
  input  logic              swizzle_done,

  // DMA requests (for external data movement)
  output logic              dma_rd_req,        // request DMA read (fill)
  output logic [39:0]       dma_rd_addr,
  output logic [15:0]       dma_rd_len,
  input  logic              dma_rd_done,

  output logic              dma_wr_req,        // request DMA write (drain)
  output logic [39:0]       dma_wr_addr,
  output logic [15:0]       dma_wr_len,
  input  logic              dma_wr_done,

  // Barrier interface
  output logic              barrier_wait_req,  // request barrier check
  input  logic              barrier_grant,
  output logic              barrier_signal,    // signal barrier done

  // Status
  output accel_pkg::tile_state_e state,
  output logic              tile_done,
  output logic              layer_done         // when LAST_TILE flag
);
```

**FSM Logic chi tiết**:
```
TILE_IDLE:
  Wait for tile_valid → accept → goto TILE_LOAD_CFG

TILE_LOAD_CFG:
  shadow_reg_file.load = 1
  Configure pe_mode, addr_gen, router_profile, post_profile
  goto TILE_PREFILL_WT

TILE_PREFILL_WT:
  if (mode != PE_PASS):
    Issue dma_rd_req for weight data (src_w_off in weight_arena)
    Wait for weight data → write to glb_weight_bank[0..2]
  goto TILE_PREFILL_IN

TILE_PREFILL_IN:
  Issue dma_rd_req for input activation (src_in_off in act_arena)
  Write R_need resident rows to glb_input_bank[0..2]
  if (tile_flags.HAS_SKIP): also load skip tensor (src_skip_off)
  goto TILE_WAIT_READY

TILE_WAIT_READY:
  if (tile_flags.BARRIER_BEFORE): issue barrier_wait_req; wait barrier_grant
  Check: weight_loaded AND input_loaded AND router_configured
  goto TILE_RUN_COMPUTE

TILE_RUN_COMPUTE:
  pe_en = 1
  pe_clear_psum = tile_flags.FIRST_TILE
  
  Run compute loop:
    for each output row group (PE_COLS rows at a time):
      for each Wblk (0 to wblk_total-1):
        for each Cin pass (0 to num_cin_pass-1):
          for each K pass (0 to num_k_pass-1):
            addr_gen generates addresses
            window_gen produces taps
            PE computes MACs
            column_reduce produces psum
  goto TILE_ACCUMULATE

TILE_ACCUMULATE:
  if NOT (last_cin AND last_kernel):
    Write psum to bank_output (PSUM namespace)
    goto TILE_RUN_COMPUTE (next pass)
  else:
    ppu_last_pass = 1
    goto TILE_POST_PROCESS

TILE_POST_PROCESS:
  PPU processes: bias + requant + SiLU + clamp
  Write INT8 result to bank_output (ACT namespace)
  if (tile_flags.BARRIER_AFTER): barrier_signal = 1
  goto TILE_SWIZZLE_STORE

TILE_SWIZZLE_STORE:
  if (tile_flags.NEED_SWIZZLE):
    swizzle_start = 1; wait swizzle_done
  if (tile_flags.NEED_SPILL):
    dma_wr_req for output data → DDR
    wait dma_wr_done
  goto TILE_DONE

TILE_DONE:
  tile_done = 1
  if (tile_flags.LAST_TILE): layer_done = 1
  goto TILE_IDLE
```

---

## M22. `barrier_manager.sv` — Skip Dependency Barrier

**Mục đích**: Quản lý 4 skip dependencies cho YOLOv10n (L4→L15, L6→L12, L8→L21, L13→L18).

**Ports**:
```systemverilog
module barrier_manager #(
  parameter int NUM_BARRIERS = 32
)(
  input  logic              clk, rst_n,
  input  logic              clear_all,

  // Signal interface (producer signals completion)
  input  logic              signal_valid,
  input  logic [4:0]        signal_barrier_id,

  // Wait interface (consumer checks readiness)
  input  logic              wait_valid,
  input  logic [4:0]        wait_barrier_id,
  output logic              wait_grant,

  // Debug readback
  output logic [NUM_BARRIERS-1:0] scoreboard
);
```

**Logic**:
```
scoreboard[32]: 1 bit per barrier point

signal: scoreboard[signal_barrier_id] = 1
wait:   wait_grant = scoreboard[wait_barrier_id]
clear:  scoreboard = 0 (at start of new inference)

YOLOv10n barrier mapping:
  barrier[0]: L6_done  → enables L12
  barrier[1]: L4_done  → enables L15
  barrier[2]: L13_done → enables L18
  barrier[3]: L8_done  → enables L21
```

---

## M23. `local_arbiter.sv` — Dual-RUNNING 4-Phase Scheduler

**Mục đích**: Quản lý 4 subclusters per SuperCluster. Gán roles (2×RUNNING + 1×FILLING + 1×DRAINING/HOLD). Arbitrate external port access.

**Ports**:
```systemverilog
module local_arbiter #(
  parameter int NUM_SUBS = 4
)(
  input  logic              clk, rst_n,
  
  // Tile queue (from global_scheduler)
  input  logic              tile_available,
  input  desc_pkg::tile_desc_t next_tile,
  output logic              tile_consumed,

  // Per-subcluster status
  input  accel_pkg::tile_state_e sub_state [NUM_SUBS],
  input  logic              sub_tile_done [NUM_SUBS],

  // Role assignment output
  output accel_pkg::sc_role_e sub_role [NUM_SUBS],

  // External port arbitration
  input  logic              ext_port_ready,
  output logic [1:0]        ext_port_grant_sub,  // which sub gets ext port
  output logic              ext_port_is_read,     // read (fill) vs write (drain)

  // Tile dispatch to subclusters
  output logic              sub_tile_valid [NUM_SUBS],
  output desc_pkg::tile_desc_t sub_tile [NUM_SUBS]
);
```

**Logic**:
```
Role rotation (dual-RUNNING):
  Maintain role_reg[4] — each sub's current role

  When sub[i] finishes tile (sub_tile_done[i] = 1):
    sub[i].role → DRAINING (needs to output its result)
    Find sub in FILLING state with data ready → promote to RUNNING
    Find IDLE/DRAINED sub → assign FILLING (load next tile)

  Priority: RUNNING > RUNNING > FILLING > DRAINING/HOLD

External port arbitration:
  if (any sub in FILLING state AND ext_port_ready):
    Grant to FILLING sub (read operations)
  elif (any sub in DRAINING state AND ext_port_ready):
    Grant to DRAINING sub (write operations)
  
  Time-multiplex: alternate FILL/DRAIN bursts when both pending
```

---

## M24. `desc_fetch_engine.sv` — Descriptor Fetch from DDR

**Mục đích**: Đọc NET_DESC → LAYER_DESC → TILE_DESC từ DDR qua DMA. Parse và dispatch.

**Ports**:
```systemverilog
module desc_fetch_engine (
  input  logic               clk, rst_n,
  input  logic               start,           // from CSR.CTRL.start

  // AXI read master interface
  output logic [39:0]        axi_araddr,
  output logic               axi_arvalid,
  input  logic               axi_arready,
  input  logic [255:0]       axi_rdata,
  input  logic               axi_rvalid,
  output logic               axi_rready,

  // Configuration from CSR
  input  logic [63:0]        net_desc_base,
  input  logic [4:0]         layer_start, layer_end,

  // Output: parsed descriptors
  output desc_pkg::net_desc_t   net_desc,
  output logic                   net_desc_valid,
  output desc_pkg::layer_desc_t layer_desc,
  output logic                   layer_desc_valid,
  output desc_pkg::tile_desc_t  tile_desc,
  output logic                   tile_desc_valid,

  // Status
  output logic [4:0]         current_layer,
  output logic               all_layers_done
);
```

**FSM**:
```
IDLE → FETCH_NET → PARSE_NET → FETCH_LAYER[layer_id] → PARSE_LAYER →
FETCH_TILES → DISPATCH_TILES → NEXT_LAYER → DONE

FETCH_NET: DMA read 64B from net_desc_base → parse into net_desc_t
FETCH_LAYER: DMA read 64B from layer_table_base + layer_id × 64
FETCH_TILES: DMA read N × 64B tile descriptors from tile_table_offset
DISPATCH_TILES: push tile_desc_t to global_scheduler → local_arbiter → subcluster
NEXT_LAYER: increment layer_id; if layer_id > layer_end → DONE
```

---

## M25. `global_scheduler.sv` — Layer/Tile Dispatcher

**Mục đích**: Nhận layer_desc và tile_desc từ desc_fetch_engine, phân phối tới 4 SuperClusters theo sc_mask.

**Ports**:
```systemverilog
module global_scheduler (
  input  logic               clk, rst_n,

  // From desc_fetch_engine
  input  desc_pkg::layer_desc_t layer_desc,
  input  logic                   layer_valid,
  input  desc_pkg::tile_desc_t  tile_desc,
  input  logic                   tile_valid,

  // To 4 SuperClusters
  output desc_pkg::tile_desc_t  sc_tile [4],
  output logic                   sc_tile_valid [4],
  input  logic                   sc_tile_accept [4],

  // Barrier interface
  output logic                   barrier_signal,
  output logic [4:0]            barrier_id,
  input  logic                   barrier_grant,

  // Status
  output logic                   layer_complete,
  output logic                   inference_complete
);
```

**Logic**:
```
For each tile_desc received:
  sc_mask = tile_desc.sc_mask     // bits [3:0] indicate which SC processes this tile
  for sc in 0..3:
    if (sc_mask[sc]):
      push tile to sc_tile[sc] queue
      wait sc_tile_accept[sc]

Track: tiles_dispatched, tiles_completed per layer
When all tiles of current layer done: signal layer_complete
When all layers done: signal inference_complete → assert IRQ
```

---

## M26. `controller_system.sv` — Top Control Wrapper

**Mục đích**: Glue module: CSR + desc_fetch + barrier_manager + global_scheduler.

**Ports**:
```systemverilog
module controller_system (
  input  logic               clk, rst_n,

  // AXI-Lite MMIO (CPU control)
  input  logic [11:0]        mmio_addr,
  input  logic [31:0]        mmio_wdata,
  input  logic               mmio_we, mmio_re,
  output logic [31:0]        mmio_rdata,
  output logic               irq,

  // AXI4 DMA read (for descriptor fetch)
  output logic [39:0]        axi_araddr,
  output logic               axi_arvalid,
  input  logic               axi_arready,
  input  logic [255:0]       axi_rdata,
  input  logic               axi_rvalid,
  output logic               axi_rready,

  // To 4 SuperClusters
  output desc_pkg::tile_desc_t  sc_tile [4],
  output desc_pkg::layer_desc_t sc_layer_desc,
  output logic                   sc_tile_valid [4],
  input  logic                   sc_tile_accept [4],
  input  logic                   sc_tile_done [4],
  input  logic                   sc_layer_done [4],

  // Barrier net
  input  logic                   barrier_signal [4],
  input  logic [4:0]            barrier_signal_id [4],
  output logic [31:0]           barrier_scoreboard
);
```

---

# PHẦN I: LEVEL 7 – TOP WRAPPERS

---

## M27. `subcluster_wrapper.sv` — Full Compute Unit

**Mục đích**: Đơn vị tính toán hoàn chỉnh. Gồm: tile_fsm + shadow_regs + GLB + router + window_gen + PE_cluster + PPU + swizzle.

**Ports**:
```systemverilog
module subcluster_wrapper #(
  parameter int LANES = 32
)(
  input  logic               clk, rst_n,

  // Tile input (from local_arbiter)
  input  logic               tile_valid,
  input  desc_pkg::tile_desc_t  tile_desc,
  input  desc_pkg::layer_desc_t layer_desc,
  output logic               tile_accept,

  // External memory port (shared via arbiter)
  output logic               ext_rd_req,
  output logic [39:0]        ext_rd_addr,
  output logic [15:0]        ext_rd_len,
  input  logic               ext_rd_grant,
  input  logic [255:0]       ext_rd_data,
  input  logic               ext_rd_valid,

  output logic               ext_wr_req,
  output logic [39:0]        ext_wr_addr,
  input  logic               ext_wr_grant,
  output logic [255:0]       ext_wr_data,
  output logic               ext_wr_valid,

  // Barrier
  output logic               barrier_signal,
  output logic [4:0]         barrier_signal_id,
  input  logic               barrier_grant,

  // Status
  output accel_pkg::tile_state_e state,
  output logic               tile_done,
  output logic               layer_done
);
```

**Internal instantiation**:
```
shadow_reg_file      shadow_regs (.tile_desc, .layer_desc, ...)

glb_input_bank       glb_in[3]   (...)
glb_weight_bank      glb_wt[3]   (...)
glb_output_bank      glb_out[4]  (...)
metadata_ram         meta        (...)

addr_gen_input       agi         (...)
addr_gen_weight      agw         (...)
addr_gen_output      ago         (...)

router_cluster       router      (...)
window_gen           wgen        (...)
pe_cluster           pe          (...)
ppu                  ppu_inst    (...)
swizzle_engine       swiz        (...)

tile_fsm             fsm         (...)
```

---

## M28. `supercluster_wrapper.sv` — 4 Subclusters + Arbiter

**Ports**:
```systemverilog
module supercluster_wrapper #(
  parameter int NUM_SUBS = 4,
  parameter int LANES    = 32
)(
  input  logic               clk, rst_n,

  // From global_scheduler
  input  desc_pkg::tile_desc_t  tile_in,
  input  desc_pkg::layer_desc_t layer_desc,
  input  logic                   tile_valid,
  output logic                   tile_accept,

  // External DDR port (256b)
  output logic [39:0]        axi_araddr,  output logic axi_arvalid,
  input  logic               axi_arready,
  input  logic [255:0]       axi_rdata,   input  logic axi_rvalid,
  output logic               axi_rready,
  output logic [39:0]        axi_awaddr,  output logic axi_awvalid,
  input  logic               axi_awready,
  output logic [255:0]       axi_wdata,   output logic axi_wvalid,
  input  logic               axi_wready,

  // Barrier
  output logic               barrier_signal,
  output logic [4:0]         barrier_signal_id,
  input  logic               barrier_grant,

  // Status
  output logic               layer_done,
  output logic [15:0]        tiles_completed
);
```

**Internal**: 4 `subcluster_wrapper` + 1 `local_arbiter` + ext port mux.

---

## M29. `tensor_dma.sv` — AXI4 DMA Master

**Mục đích**: DMA engine cho load tensor/weight/descriptor từ DDR và store output.

**Ports**:
```systemverilog
module tensor_dma #(
  parameter int AXI_DATA_W = 256,
  parameter int AXI_ADDR_W = 40
)(
  input  logic                  clk, rst_n,

  // AXI4 Master Read
  output logic [AXI_ADDR_W-1:0] m_axi_araddr,
  output logic [7:0]            m_axi_arlen,
  output logic                  m_axi_arvalid,
  input  logic                  m_axi_arready,
  input  logic [AXI_DATA_W-1:0] m_axi_rdata,
  input  logic [1:0]            m_axi_rresp,
  input  logic                  m_axi_rlast,
  input  logic                  m_axi_rvalid,
  output logic                  m_axi_rready,

  // AXI4 Master Write
  output logic [AXI_ADDR_W-1:0] m_axi_awaddr,
  output logic [7:0]            m_axi_awlen,
  output logic                  m_axi_awvalid,
  input  logic                  m_axi_awready,
  output logic [AXI_DATA_W-1:0] m_axi_wdata,
  output logic                  m_axi_wlast,
  output logic                  m_axi_wvalid,
  input  logic                  m_axi_wready,
  input  logic [1:0]            m_axi_bresp,
  input  logic                  m_axi_bvalid,
  output logic                  m_axi_bready,

  // Internal request interface (from subclusters via arbiter)
  input  logic                  rd_req,
  input  logic [AXI_ADDR_W-1:0] rd_addr,
  input  logic [15:0]           rd_byte_len,
  output logic                  rd_data_valid,
  output logic [AXI_DATA_W-1:0] rd_data,
  output logic                  rd_done,

  input  logic                  wr_req,
  input  logic [AXI_ADDR_W-1:0] wr_addr,
  input  logic [15:0]           wr_byte_len,
  input  logic [AXI_DATA_W-1:0] wr_data,
  input  logic                  wr_data_valid,
  output logic                  wr_done
);
```

**Logic**:
```
Read path:
  Split rd_byte_len into AXI bursts (max ARLEN=15 → 16 beats × 32B = 512B per burst)
  Issue AXI AR, collect R data, push to rd_data output

Write path:
  Split wr_byte_len into AXI bursts
  Issue AXI AW, stream W data with WLAST on final beat
  Wait BRESP

Burst parameters:
  ARSIZE = 5 (32 bytes = 256 bits)
  ARBURST = INCR
  ARLEN = min(15, remaining_beats - 1)
```

---

## M30. `accel_top.sv` — Top-Level Module

**Ports**:
```systemverilog
module accel_top (
  input  logic        clk, rst_n,

  // AXI-Lite Slave (CPU MMIO)
  input  logic [11:0] s_axil_awaddr, input logic s_axil_awvalid,
  output logic        s_axil_awready,
  input  logic [31:0] s_axil_wdata,  input logic s_axil_wvalid,
  output logic        s_axil_wready,
  output logic [1:0]  s_axil_bresp,  output logic s_axil_bvalid,
  input  logic        s_axil_bready,
  input  logic [11:0] s_axil_araddr, input logic s_axil_arvalid,
  output logic        s_axil_arready,
  output logic [31:0] s_axil_rdata,  output logic s_axil_rvalid,
  input  logic        s_axil_rready,

  // AXI4 Master (DDR DMA — 256-bit)
  output logic [39:0]  m_axi_araddr,  output logic [7:0] m_axi_arlen,
  output logic         m_axi_arvalid, input  logic m_axi_arready,
  input  logic [255:0] m_axi_rdata,   input  logic m_axi_rvalid,
  input  logic         m_axi_rlast,   output logic m_axi_rready,
  output logic [39:0]  m_axi_awaddr,  output logic [7:0] m_axi_awlen,
  output logic         m_axi_awvalid, input  logic m_axi_awready,
  output logic [255:0] m_axi_wdata,   output logic m_axi_wvalid,
  output logic         m_axi_wlast,   input  logic m_axi_wready,
  input  logic [1:0]   m_axi_bresp,   input  logic m_axi_bvalid,
  output logic         m_axi_bready,

  // Interrupt
  output logic         irq
);
```

**Internal instantiation**:
```
controller_system       ctrl (...)
supercluster_wrapper    sc[4] (...)
tensor_dma              dma (...)
perf_mon                perf (...)

AXI interconnect:
  - ctrl.axi (desc fetch) + dma.axi (tensor load/store) → AXI arbiter → m_axi
  - 4 SC external ports → dma request mux → dma
```

---

# PHẦN J: MODULE → PRIMITIVE MAPPING

## Bảng: Primitive nào dùng module nào

```
┌─────────────────────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│ Module              │ RS3  │ OS1  │ DW3  │ DW7  │ MP5  │ GEMM │ MOVE │ CAT  │ UP   │ EADD │
├─────────────────────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ glb_input_bank      │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │
│ glb_weight_bank     │  ✓   │  ✓   │  ✓   │  ✓   │  -   │  ✓   │  -   │  -   │  -   │  -   │
│ glb_output_bank     │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │
│ addr_gen_input      │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │
│ addr_gen_weight     │  ✓   │  ✓   │  ✓   │  ✓   │  -   │  ✓   │  -   │  -   │  -   │  -   │
│ addr_gen_output     │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │
│ router_cluster      │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  -   │  ✓   │
│ window_gen          │  ✓K3 │  ✓K1 │  ✓K3 │  ✓K7 │  ✓K5 │  ✓K1 │  -   │  -   │  -   │  -   │
│ dsp_pair_int8       │  ✓   │  ✓   │  ✓   │  ✓   │  -   │  ✓   │  -   │  -   │  -   │  -   │
│ pe_unit             │  ✓   │  ✓   │  ✓   │  ✓   │  -   │  ✓   │  -   │  -   │  -   │  -   │
│ pe_cluster          │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  -   │  -   │  -   │  -   │
│ column_reduce       │  ✓   │  ✓   │  ✓   │  ✓   │  -   │  ✓   │  -   │  -   │  -   │  -   │
│ comparator_tree     │  -   │  -   │  -   │  -   │  ✓   │  -   │  -   │  -   │  -   │  -   │
│ ppu (bias+requant)  │  ✓   │  ✓   │  ✓   │  ✓   │  -   │  ✓   │  -   │  ?   │  -   │  ✓   │
│ silu_lut            │  ✓   │  ✓   │  -   │  -   │  -   │  -   │  -   │  -   │  -   │  -   │
│ swizzle_engine      │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  -   │  ✓   │  ✓   │  -   │
│ barrier_manager     │  -   │  -   │  -   │  -   │  -   │  -   │  -   │  ✓   │  -   │  -   │
│ tile_fsm            │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │  ✓   │
└─────────────────────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘

RS3 = RS_DENSE_3x3, OS1 = OS_1x1, DW3 = DW_3x3, DW7 = DW_7x7_MULTIPASS
MP5 = MAXPOOL_5x5, GEMM = GEMM_ATTN, CAT = CONCAT, UP = UPSAMPLE, EADD = EWISE_ADD
? = mini-requant khi scale_A ≠ scale_B
```

---

# PHẦN K: BLOCK → PRIMITIVE → LAYER DECOMPOSITION

## YOLOv10n L0–L22: Mỗi layer cần chạy primitive nào

```
┌───────┬───────────┬──────────────────────────────────────────────────────────────┬──────────────┐
│ Layer │ Block     │ Primitive Sequence (thứ tự thực thi)                         │ Output       │
├───────┼───────────┼──────────────────────────────────────────────────────────────┼──────────────┤
│  L0   │ Conv(s2)  │ RS_DENSE_3x3(3→16, s=2, SiLU)                              │[16,320,320]  │
│  L1   │ Conv(s2)  │ RS_DENSE_3x3(16→32, s=2, SiLU)                             │[32,160,160]  │
│  L2   │ QC2f      │ OS_1x1(32→32) → RS3(16→16) → RS3(16→16) → CAT → OS_1x1   │[32,160,160]  │
│  L3   │ Conv(s2)  │ RS_DENSE_3x3(32→64, s=2, SiLU)                             │[64,80,80]    │
│  L4   │ QC2f      │ OS_1x1(64→64) → RS3(32→32) → RS3(32→32) → CAT → OS_1x1   │[64,80,80] ★  │
│  L5   │ SCDown    │ OS_1x1(64→64) → DW_3x3(64, s=2) → OS_1x1(64→64) →        │[128,40,40]   │
│       │           │ DW_3x3(64, s=2) → CONCAT(64+64)                            │              │
│  L6   │ QC2f      │ OS_1x1 → RS3 → RS3 → CAT → OS_1x1 (Cin=Cout=128)          │[128,40,40] ★ │
│  L7   │ SCDown    │ OS_1x1 → DW_3x3(s=2) → OS_1x1 → DW_3x3(s=2) → CONCAT    │[256,20,20]   │
│  L8   │ QC2f      │ OS_1x1 → RS3 → RS3 → CAT → OS_1x1 (Cin=Cout=256)          │[256,20,20] ★ │
│  L9   │ SPPF      │ OS_1x1(256→128) → MP5 → MP5 → MP5 → CAT(128×4) →         │[256,20,20]   │
│       │           │ OS_1x1(512→256)                                             │              │
│  L10  │ QPSA      │ OS_1x1(split) → GEMM_ATTN(Q,K,V proj→QK^T→soft→×V) →     │[256,20,20]   │
│       │           │ CAT(attn+pass) → OS_1x1                                    │              │
│  L11  │ Upsample  │ UPSAMPLE_NEAREST(×2)                                        │[256,40,40]   │
│  L12  │ QConcat   │ CONCAT(L11 + L6★) [barrier wait L6]                         │[384,40,40]   │
│  L13  │ QC2f      │ OS_1x1 → RS3 → RS3 → CAT → OS_1x1 (384→128)               │[128,40,40] ★ │
│  L14  │ Upsample  │ UPSAMPLE_NEAREST(×2)                                        │[128,80,80]   │
│  L15  │ QConcat   │ CONCAT(L14 + L4★) [barrier wait L4]                         │[192,80,80]   │
│  L16  │ QC2f      │ OS_1x1 → RS3 → RS3 → CAT → OS_1x1 (192→64)                │[64,80,80] P3 │
│  L17  │ Conv(s2)  │ RS_DENSE_3x3(64→64, s=2, SiLU)                             │[64,40,40]    │
│  L18  │ QConcat   │ CONCAT(L17 + L13★) [barrier wait L13]                       │[192,40,40]   │
│  L19  │ QC2f      │ OS_1x1 → RS3 → RS3 → CAT → OS_1x1 (192→128)               │[128,40,40] P4│
│  L20  │ SCDown    │ OS_1x1 → DW_3x3(s=2)                                       │[128,20,20]   │
│  L21  │ QConcat   │ CONCAT(L20 + L8★) [barrier wait L8]                         │[384,20,20]   │
│  L22  │ QC2fCIB   │ OS_1x1(384→256) → DW_7x7×3pass → OS_1x1 → CAT →          │[256,20,20] P5│
│       │           │ OS_1x1(256)                                                 │              │
└───────┴───────────┴──────────────────────────────────────────────────────────────┴──────────────┘

★ = output phải lưu cho skip connection (HOLD_SKIP buffer)
P3/P4/P5 = final outputs gửi về CPU
```

---

# PHẦN L: THỨ TỰ XÂY DỰNG VÀ VERIFY

## Chiến thuật implementation (Bottom-Up)

```
PHASE 1 — Compute Leaf (2 tuần):
  ① dsp_pair_int8.sv       → unit test: all 65536 INT8 pairs, bit-exact
  ② pe_unit.sv              → test: 32-lane MAC accumulate
  ③ comparator_tree.sv      → test: max of 25 random values
  ④ column_reduce.sv        → test: sum 3 rows
  ⑤ silu_lut.sv             → test: preload + 32-parallel lookup

PHASE 2 — PPU (1 tuần):
  ⑥ ppu.sv                  → test: bias + requant + SiLU vs Golden Python
     (pipeline: bias_add → requant → silu → clamp)
     Verify: per-channel M_int/shift correctness

PHASE 3 — Memory (2 tuần):
  ⑦ glb_input_bank.sv       → test: modulo-3 banking, 32 subbanks
  ⑧ glb_weight_bank.sv      → test: 3 reduction lanes + FIFO
  ⑨ glb_output_bank.sv      → test: dual PSUM/ACT mode
  ⑩ addr_gen_input.sv       → test: no address overlap, padding=zp_x
  ⑪ addr_gen_weight.sv      → test: per-mode address patterns
  ⑫ addr_gen_output.sv      → test: output bank mapping

PHASE 4 — Data Movement (2 tuần):
  ⑬ window_gen.sv           → test: K1/K3/K5/K7 tap generation
  ⑭ router_cluster.sv       → test: per-mode routing patterns
  ⑮ swizzle_engine.sv       → test: upsample 20→40, concat offset

PHASE 5 — Integration (2 tuần):
  ⑯ pe_cluster.sv           → test: RS3/OS1/DW3/DW7/MP5/GEMM all modes
  ⑰ subcluster_wrapper.sv   → test: 1 tile RS_DENSE_3x3 end-to-end
  ⑱ shadow_reg_file.sv      → test: descriptor field capture

PHASE 6 — Control (2 tuần):
  ⑲ tile_fsm.sv             → test: FSM transitions, multi-pass accumulation
  ⑳ barrier_manager.sv      → test: 4 YOLOv10n barriers
  ㉑ local_arbiter.sv        → test: dual-RUNNING rotation
  ㉒ desc_fetch_engine.sv    → test: parse NET/LAYER/TILE descriptors
  ㉓ global_scheduler.sv     → test: tile dispatch to 4 SCs

PHASE 7 — System (2 tuần):
  ㉔ supercluster_wrapper.sv → test: 4 subclusters, role rotation
  ㉕ tensor_dma.sv           → test: AXI4 burst read/write
  ㉖ controller_system.sv    → test: CSR read/write, start→done flow
  ㉗ accel_top.sv            → test: full L0 inference, compare Golden Python

PHASE 8 — End-to-End (2 tuần):
  ㉘ Full L0→L22: input X_int8 → P3, P4, P5
     Compare bit-exact with Phase 1 Golden Python
     Verify: all 4 barrier points, skip tensors correct
     Performance: measure cycle count, compute utilization
```

---

# PHẦN M: VERIFICATION CHECKLIST

```
☐ dsp_pair_int8:    65536 pairs, max error = 0
☐ pe_cluster RS3:   Conv3×3 s=1 [3,16,16] 8×8 tile → match Golden Python
☐ pe_cluster RS3:   Conv3×3 s=2 [3,16,16] 16×16 tile → match
☐ pe_cluster OS1:   Conv1×1 [32,64] 8×8 → match
☐ pe_cluster DW3:   DW3×3 s=1 [64] 8×8 → match
☐ pe_cluster DW7:   DW7×7 3-pass [256] 20×20 → match monolithic
☐ pe_cluster MP5:   MaxPool5×5 [128] 20×20 → match
☐ pe_cluster GEMM:  MatMul [400,64]×[400,64]^T → match
☐ ppu:              bias+requant+SiLU per-channel → match Golden Python
☐ ppu:              ewise_add with saturation → match
☐ window_gen:       K7 taps from 32-wide stream → correct spatial values
☐ addr_gen_input:   padding positions output zp_x (NOT zero)
☐ addr_gen_input:   no address collision across (h,w,c) space
☐ swizzle:          upsample 20×20→40×40 content replicated correctly
☐ swizzle:          concat [128ch + 64ch] → [192ch] channel offset correct
☐ barrier:          L6→L12 dependency holds until L6 done
☐ barrier:          L4→L15, L13→L18, L8→L21 all correct
☐ tile_fsm:         multi-pass Cin accumulation: 3 passes → correct final psum
☐ tile_fsm:         DW7 3 passes: psum persists across passes
☐ local_arbiter:    dual-RUNNING: 2 subs compute simultaneously, no corruption
☐ end-to-end L0:    X_int8[1,3,640,640] → [1,16,320,320] bit-exact
☐ end-to-end L0-L4: sequential layers, data passing through GLB/swizzle
☐ end-to-end L0-L22: P3/P4/P5 bit-exact with Golden Python
```

---

*Tài liệu này cung cấp đầy đủ interface, logic, và thứ tự xây dựng cho toàn bộ RTL.
Mỗi module description đủ chi tiết để prompt sinh SystemVerilog code.
Tính đúng đắn được đảm bảo bởi: bottom-up verification, bit-exact comparison với Golden Python Phase 1.*


---
---

# ════════════════════════════════════════════════════════════════
# PHẦN VI — ĐẶC TẢ PHASE 0 (Spec Freeze)
# ════════════════════════════════════════════════════════════════

<a id='phần-vi1---primitive-matrix'></a>

# PHẦN VI.1 — Primitive Matrix
> Nguồn: `PHASE0/01_primitive_matrix.md`

---

# 01 – Primitive Matrix (Freeze Spec)
## qYOLOv10n INT8 Accelerator – Phase 0

> **Trạng thái**: FREEZE | **Version**: v1.0 | **Ngày**: 2026-03-16

---

## 1. Mục Đích

Chốt tập primitive chính thức của accelerator V1 để cover qYOLOv10n PTQ layer 0–22. Mọi RTL, Golden Python và test plan đều tham chiếu file này.

---

## 2. Bảng Primitive Chính Thức

| ID | Tên | Loại | Status | Layer dùng |
|---|---|---|---|---|
| P0 | RS_DENSE_3x3 | Conv 3×3 | **BẮT BUỘC** | L0,1,3,17; nội bộ QC2f |
| P1 | OS_1x1 | Conv 1×1 | **BẮT BUỘC** | Tất cả QC2f, SPPF, QPSA, SCDown |
| P2 | DW_3x3 | Depthwise 3×3 | **BẮT BUỘC** | SCDown L5,7,20; QC2fCIB |
| P3 | MAXPOOL_5x5 | Pooling | **BẮT BUỘC** | SPPF L9 (×3 lần) |
| P4 | MOVE | Tensor copy | **BẮT BUỘC** | Skip buffer HOLD |
| P5 | CONCAT | Tensor join | **BẮT BUỘC** | QConcat L12,15,18,21 |
| P6 | UPSAMPLE_NEAREST | Tensor upsample | **BẮT BUỘC** | L11, L14 |
| P7 | EWISE_ADD | Element-wise add | Nên giữ | Residual dự phòng |
| P8 | DW_7x7_MULTIPASS | Depthwise 7×7 | Nên giữ | QC2fCIB L22 |
| P9 | GEMM_ATTN_BASIC | Attention GEMM | Nên giữ | QPSA L10 |

---

## 3. Đặc Tả Chi Tiết

### P0 – RS_DENSE_3x3

| Trường | Giá trị |
|---|---|
| Input | [N, Cin, H, W] INT8 per-tensor (scale_x, zp_x) |
| Weight | [Cout, Cin, 3, 3] INT8 per-output-channel (scale_w[cout], zp_w=0) |
| Bias | [Cout] INT32, scale_bias[cout] = scale_x × scale_w[cout] |
| Output | [N, Cout, Hout, Wout] INT8 per-tensor (scale_y, zp_y) |
| Stride | {1, 2} |
| Padding | same: pad=1, Hout = ceil(H/stride) |
| PPU | YES: bias_add → requant → SiLU_LUT → clamp |
| PSUM | INT32 accumulator nội bộ |

**Golden Math**:
```
acc[cout,h,w] = Σ_{cin,kh,kw} X[cin,h*s+kh-1,w*s+kw-1] × W[cout,cin,kh,kw]
              - zp_x × Σ_{cin,kh,kw} W[cout,cin,kh,kw]   ← zp_correction (precomputed)
              + B[cout]

M[cout] = scale_x × scale_w[cout] / scale_y  → decompose: (M_int, shift)

y_raw = round(M_int × acc >> shift) + zp_y
y_int8 = SiLU_LUT[clamp(y_raw, -128, 127)]   (nếu có activation)
```

**Tests sign-off**: stride=1, stride=2, edge padding, Cin=128 Cout=256, zp_x≠0, random×100 vs PyTorch float conv (≤1 LSB)

---

### P1 – OS_1x1

| Trường | Giá trị |
|---|---|
| Input | [N, Cin, H, W] INT8 |
| Output | [N, Cout, H, W] INT8 (spatial giữ nguyên) |
| Stride | 1, Padding 0 |
| PPU | YES |

**Golden Math**: Như P0 với kh=kw=0 (kernel 1 element), không padding.

**Tests**: expand (Cin<Cout), compress (Cin>Cout), projection (Cin=Cout)

---

### P2 – DW_3x3

| Trường | Giá trị |
|---|---|
| Input | [N, C, H, W] INT8 |
| Output | [N, C, Hout, Wout] INT8 |
| Groups | C (mỗi channel độc lập) |
| Weight | [C, 1, 3, 3] INT8, per-channel: scale_w[c], zp_w[c]=0 |
| Bias | [C] INT32 per-channel |
| PPU | YES, per-channel requant |
| last_pass | LUÔN True (không cross-channel) |

**Golden Math**:
```
for c in C:
  acc[c,h,w] = Σ_{kh,kw} X[c,h*s+kh-1,w*s+kw-1] × W[c,0,kh,kw]
             - zp_x × Σ_{kh,kw} W[c,0,kh,kw] + B[c]
  y[c,h,w] = clamp(round(M[c] × acc) + zp_y, -128, 127)
  where M[c] = scale_x × scale_w[c] / scale_y
```

**Tests**: stride=1, stride=2, per-channel weight khác nhau giữa các channels

---

### P3 – MAXPOOL_5x5

| Trường | Giá trị |
|---|---|
| Kernel | 5×5, stride=1, padding=2 |
| Quant | **Pass-through**: scale_out=scale_in, zp_out=zp_in |
| PPU | KHÔNG – so sánh INT8 thuần |

**Golden Math**: `Y[c,h,w] = max(X[c, h+dh-2, w+dw-2] for dh,dw in 0..4)`

**Tests**: shape [1,128,20,20]→[1,128,20,20], lặp 3×, scale/zp unchanged

---

### P4 – MOVE

| Trường | Giá trị |
|---|---|
| Chức năng | Copy tensor, giữ nguyên (scale, zp) |
| PPU | KHÔNG |
| Use | HOLD_SKIP buffer, skip connection management |

---

### P5 – CONCAT

| Trường | Giá trị |
|---|---|
| Input | A[N,C_A,H,W](scale_A,zp_A) + B[N,C_B,H,W](scale_B,zp_B) |
| Output | [N,C_A+C_B,H,W](scale_Y,zp_Y) |
| Axis | Channel (dim=1) |
| Quant | Common-domain alignment nếu scale_A≠scale_B |

**Common-domain alignment**:
```
scale_Y, zp_Y = được chọn offline từ PTQ calibration

if scale_A ≠ scale_Y:
  A_float = (A_int8 - zp_A) × scale_A
  A_aligned = clamp(round(A_float/scale_Y) + zp_Y, -128, 127)

concat(A_aligned, B_aligned) theo chiều channel
```

**Tests**: same domain (no requant), diff domain (scale×0.5), 4-way concat (SPPF)

---

### P6 – UPSAMPLE_NEAREST

| Trường | Giá trị |
|---|---|
| Scale | ×2 spatial |
| Quant | Pass-through |
| Hardware | tensor_post_engine, address remapping |

**Golden Math**:
```
Y[n,c,2h,2w]=Y[n,c,2h,2w+1]=Y[n,c,2h+1,2w]=Y[n,c,2h+1,2w+1] = X[n,c,h,w]
```

---

### P7 – EWISE_ADD (Optional)

Common-domain alignment → INT16 add → requant → INT8.
Dự phòng residual. Chưa dùng trong L0–22 baseline.

---

### P8 – DW_7x7_MULTIPASS

| Trường | Giá trị |
|---|---|
| Pass split | 3-3-1 (rows 0-2, 3-5, 6) |
| PSUM | INT32, giữ qua 2 pass đầu |
| Bias+PPU | Chỉ tại last_pass (pass 3) |
| Invariant | **Output == monolithic DW_7x7** |

**Thuật toán**:
```
PSUM = 0
Pass 1 (kh=0,1,2): PSUM += Σ_{kh in [0,1,2], kw} X×W; last_pass=False
Pass 2 (kh=3,4,5): PSUM += Σ_{kh in [3,4,5], kw} X×W; last_pass=False
Pass 3 (kh=6):     PSUM += Σ_{kh=6, kw} X×W
                   PSUM -= zp_correction_full; PSUM += B_int32
                   Y = clamp(round(M×PSUM)+zp_y); last_pass=True
```

**Trace bắt buộc**: PSUM sau mỗi pass để RTL debug.

---

### P9 – GEMM_ATTN_BASIC

```
Input: [N,C,H,W] → reshape [N,HW,C] (HW=400 tại 20×20)
Q=OS_1x1(X), K=OS_1x1(X), V=OS_1x1(X)
Attn = Q×K^T → requant → softmax_approx → ×V → requant → reshape → OS_1x1
Output: [N,C,H,W]
```

---

## 4. Constraints Bất Biến

1. INT8 activation range: [-128, 127]
2. Weight zero-point: zp_w = 0 (symmetric)
3. PSUM width: INT32 (min 24-bit logic)
4. Requant path duy nhất: `y = clamp(round(M_int × acc >> shift) + zp_out)`
5. UPSAMPLE/MAXPOOL: scale_out = scale_in, zp_out = zp_in (không đổi)
6. scale > 0 tại mọi điểm

---

## 5. Sign-off Checklist

```
☐ P0 RS_DENSE_3x3 : test pass (stride 1/2, padding, multi-channel, random×100)
☐ P1 OS_1x1       : test pass (expand, compress, projection)
☐ P2 DW_3x3       : test pass (stride 1/2, per-channel)
☐ P3 MAXPOOL_5x5  : test pass (shape, value, 3×loop)
☐ P4 MOVE         : test pass (copy, metadata identical)
☐ P5 CONCAT       : test pass (same domain, diff domain, 4-way)
☐ P6 UPSAMPLE     : test pass (shape×2, content correct, metadata unchanged)
☐ P7 EWISE_ADD    : test pass (basic, saturation, domain mismatch)
☐ P8 DW_7x7       : test pass (BẮT BUỘC multipass == monolithic)
☐ P9 GEMM_ATTN    : test pass (shape, deterministic)
```

*Nguồn chân lý cho primitive set. Mọi thay đổi cần sign-off trước khi áp dụng.*


---
---

<a id='phần-vi2---layer-mapping'></a>

# PHẦN VI.2 — Layer Mapping
> Nguồn: `PHASE0/02_layer_mapping.md`

---

# 02 – Layer Mapping (Freeze Spec)
## qYOLOv10n INT8 PTQ – Layer 0–22 → Primitive Decomposition

> **Trạng thái**: FREEZE | **Version**: v1.0 | **Ngày**: 2026-03-16

---

## 1. Mục Đích

Chốt mapping từng layer (0–22) của qYOLOv10n PTQ sang chuỗi primitive tương ứng. Đây là cầu nối trực tiếp giữa model và hardware. `model_forward_runner.py` và RTL `desc_fetch_engine` phải tuân thủ file này.

---

## 2. Bảng Trace Thực Tế (Đo Từ PyTorch)

```
Layer  Module     Input shape          Output shape         Dtype
0      Conv       [1, 3,  640, 640]    [1, 16, 320, 320]    quint8→quint8
1      Conv       [1, 16, 320, 320]    [1, 32, 160, 160]    quint8→quint8
2      QC2f       [1, 32, 160, 160]    [1, 32, 160, 160]    quint8→quint8
3      Conv       [1, 32, 160, 160]    [1, 64,  80,  80]    quint8→quint8
4      QC2f       [1, 64,  80,  80]    [1, 64,  80,  80]    quint8→quint8
5      SCDown     [1, 64,  80,  80]    [1,128,  40,  40]    quint8→quint8
6      QC2f       [1,128,  40,  40]    [1,128,  40,  40]    quint8→quint8
7      SCDown     [1,128,  40,  40]    [1,256,  20,  20]    quint8→quint8
8      QC2f       [1,256,  20,  20]    [1,256,  20,  20]    quint8→quint8
9      SPPF       [1,256,  20,  20]    [1,256,  20,  20]    quint8→quint8
10     QPSA       [1,256,  20,  20]    [1,256,  20,  20]    quint8→quint8
11     Upsample   [1,256,  20,  20]    [1,256,  40,  40]    quint8→quint8
12     QConcat    [1,256+128,40, 40]   [1,384,  40,  40]    quint8→quint8
13     QC2f       [1,384,  40,  40]    [1,128,  40,  40]    quint8→quint8
14     Upsample   [1,128,  40,  40]    [1,128,  80,  80]    quint8→quint8
15     QConcat    [1,128+64, 80, 80]   [1,192,  80,  80]    quint8→quint8
16     QC2f       [1,192,  80,  80]    [1, 64,  80,  80]    quint8→quint8  ← P3
17     Conv       [1, 64,  80,  80]    [1, 64,  40,  40]    quint8→quint8
18     QConcat    [1, 64+128,40, 40]   [1,192,  40,  40]    quint8→quint8
19     QC2f       [1,192,  40,  40]    [1,128,  40,  40]    quint8→quint8  ← P4
20     SCDown     [1,128,  40,  40]    [1,128,  20,  20]    quint8→quint8
21     QConcat    [1,128+256,20, 20]   [1,384,  20,  20]    quint8→quint8
22     QC2fCIB    [1,384,  20,  20]    [1,256,  20,  20]    quint8→quint8  ← P5
```

---

## 3. Layer → Primitive Mapping Chi Tiết

### BACKBONE – Đường Xuống (L0–L10)

---

#### L0 – Conv [3,640,640] → [16,320,320]

```
Block type  : Conv (Conv2d + BN_fuse + SiLU)
Primitive   : RS_DENSE_3x3(stride=2)
Sources     : X_int8 từ CPU (input quantize)
Hold output : No

Primitive params:
  Cin=3, Cout=16, H=640, W=640, stride=2, padding=1
  W_int8[16,3,3,3], B_int32[16]
  scale_in=1/255, zp_in=0
  scale_out=s_L0, zp_out=zp_L0

Output: F0 [1,16,320,320]
```

---

#### L1 – Conv [16,320,320] → [32,160,160]

```
Primitive   : RS_DENSE_3x3(stride=2)
Sources     : F0
Hold output : No

Primitive params:
  Cin=16, Cout=32, H=320, W=320, stride=2

Output: F1 [1,32,160,160]
```

---

#### L2 – QC2f [32,160,160] → [32,160,160]

```
Block type  : QC2f (n=1 bottleneck)
Sources     : F1
Hold output : No

Primitive sequence:
  Step 1: OS_1x1(Cin=32, Cout=32)              → X1 [1,32,160,160]
  Step 2: RS_DENSE_3x3(Cin=16, Cout=16, s=1)   → Ytmp [1,16,160,160]    ← half channel
  Step 3: RS_DENSE_3x3(Cin=16, Cout=16, s=1)   → Ybranch [1,16,160,160]
  Step 4: CONCAT([Ybranch, X1_upper_half])      → Ycat [1,32,160,160]
  Step 5: OS_1x1(Cin=32, Cout=32)              → F2 [1,32,160,160]

Note: X1 được split: nửa dưới vào bottleneck, nửa trên giữ lại cho CONCAT
      (theo kiến trúc C2f: channels = [c1//2, c2//2, ...]

Output: F2 [1,32,160,160]
```

---

#### L3 – Conv [32,160,160] → [64,80,80]

```
Primitive   : RS_DENSE_3x3(stride=2)
Sources     : F2
Hold output : No

Primitive params: Cin=32, Cout=64, stride=2

Output: F3 [1,64,80,80]
```

---

#### L4 – QC2f [64,80,80] → [64,80,80]

```
Block type  : QC2f (n=1)
Sources     : F3
Hold output : YES → hold đến L15 (QConcat)

Primitive sequence: OS_1x1 + RS_DENSE_3x3 + RS_DENSE_3x3 + CONCAT + OS_1x1
  (Tương tự L2, với Cin=Cout=64)

Output: F4 [1,64,80,80]
⚠️  HOLD_SKIP = True, hold_until = L15
```

---

#### L5 – SCDown [64,80,80] → [128,40,40]

```
Block type  : SCDown
Sources     : F4
Hold output : No

Primitive sequence (2 nhánh song song):
  Branch A:
    OS_1x1(Cin=64, Cout=64)      → tmpA [1,64,80,80]
    DW_3x3(C=64, stride=2)       → A_out [1,64,40,40]

  Branch B:
    OS_1x1(Cin=64, Cout=64)      → tmpB [1,64,80,80]
    DW_3x3(C=64, stride=2)       → B_out [1,64,40,40]

  CONCAT(A_out, B_out)           → F5 [1,128,40,40]

Note: Cin_total=64, Cout_total=128
      Mỗi nhánh xử lý 64→64 channels, sau CONCAT ra 128

Output: F5 [1,128,40,40]
```

---

#### L6 – QC2f [128,40,40] → [128,40,40]

```
Block type  : QC2f (n=1)
Sources     : F5
Hold output : YES → hold đến L12 (QConcat)

Primitive sequence: OS_1x1 + RS_DENSE_3x3 + RS_DENSE_3x3 + CONCAT + OS_1x1
  (Cin=Cout=128)

Output: F6 [1,128,40,40]
⚠️  HOLD_SKIP = True, hold_until = L12
```

---

#### L7 – SCDown [128,40,40] → [256,20,20]

```
Block type  : SCDown
Sources     : F6
Hold output : No

Primitive sequence: (tương tự L5 với Cin=128, Cout=256)
  Branch A: OS_1x1(128→128) → DW_3x3(s2) → [1,128,20,20]
  Branch B: OS_1x1(128→128) → DW_3x3(s2) → [1,128,20,20]
  CONCAT → F7 [1,256,20,20]

Output: F7 [1,256,20,20]
```

---

#### L8 – QC2f [256,20,20] → [256,20,20]

```
Block type  : QC2f (n=1)
Sources     : F7
Hold output : YES → hold đến L21 (QConcat) ← SKIP DÀI NHẤT

Primitive sequence: OS_1x1 + RS_DENSE_3x3 + RS_DENSE_3x3 + CONCAT + OS_1x1
  (Cin=Cout=256)

Output: F8 [1,256,20,20]
⚠️  HOLD_SKIP = True, hold_until = L21  (giữ qua 13 layer!)
```

---

#### L9 – SPPF [256,20,20] → [256,20,20]

```
Block type  : SPPF
Sources     : F8
Hold output : No

Primitive sequence:
  Step 1: OS_1x1(Cin=256, Cout=128)     → X1 [1,128,20,20]  (giảm kênh)
  Step 2: MAXPOOL_5x5(X1)               → P1 [1,128,20,20]
  Step 3: MAXPOOL_5x5(P1)               → P2 [1,128,20,20]
  Step 4: MAXPOOL_5x5(P2)               → P3 [1,128,20,20]
  Step 5: CONCAT(X1, P1, P2, P3)        → Ycat [1,512,20,20]
          (4 nhánh cùng qconfig → scale/zp tương đồng → concat đơn giản)
  Step 6: OS_1x1(Cin=512, Cout=256)     → F9 [1,256,20,20]

Note: X1 phải được buffer đồng thời với P1,P2,P3 trước khi CONCAT

Output: F9 [1,256,20,20]
```

---

#### L10 – QPSA [256,20,20] → [256,20,20]

```
Block type  : QPSA (Quantized Position Sensitive Attention)
Sources     : F9
Hold output : No

Primitive sequence:
  Step 1: OS_1x1(Cin=256, Cout=256)     → X_split, phân thành:
          X_attn [1,128,20,20]  (nhánh attention)
          X_pass [1,128,20,20]  (nhánh pass-through)

  Step 2: GEMM_ATTN_BASIC(X_attn)
          reshape [1,128,20,20] → [1,400,128]
          Q = OS_1x1_proj(X) → [1,400,64]
          K = OS_1x1_proj(X) → [1,400,64]
          V = OS_1x1_proj(X) → [1,400,128]
          Attn = Q×K^T/sqrt(64) → softmax_approx → ×V
          Y_attn = reshape → [1,128,20,20]

  Step 3: CONCAT(Y_attn, X_pass)        → Ymerge [1,256,20,20]
  Step 4: OS_1x1(Cin=256, Cout=256)     → F10 [1,256,20,20]

Output: F10 [1,256,20,20]
```

---

### NECK – FPN (L11–L16)

---

#### L11 – Upsample [256,20,20] → [256,40,40]

```
Primitive   : UPSAMPLE_NEAREST(scale=2)
Sources     : F10
Hold output : No

Y[c,2h,2w]=Y[c,2h,2w+1]=Y[c,2h+1,2w]=Y[c,2h+1,2w+1] = X[c,h,w]
scale/zp giữ nguyên từ F10

Output: F11 [1,256,40,40]
```

---

#### L12 – QConcat [256,40,40]+[128,40,40] → [384,40,40]

```
Primitive   : CONCAT
Sources     : [F11, F6]  ← SKIP DEPENDENCY: F6 từ L6
Hold output : No

Input A: F11 [1,256,40,40] (scale_A=scale_F11, zp_A=zp_F11) ← từ upsample
Input B: F6  [1,128,40,40] (scale_B=scale_F6,  zp_B=zp_F6)  ← từ backbone skip

Common-domain alignment:
  scale_Y, zp_Y được chọn offline từ PTQ (scale của QConcat output layer 12)
  Nếu scale_A ≠ scale_Y → requant F11 về scale_Y
  Nếu scale_B ≠ scale_Y → requant F6 về scale_Y
  CONCAT(F11_aligned, F6_aligned) theo chiều channel

BARRIER: L12 phải đợi BOTH L11_done AND F6_hold_ready

Output: F12 [1,384,40,40]
```

---

#### L13 – QC2f [384,40,40] → [128,40,40]

```
Block type  : QC2f (n=1)
Sources     : F12
Hold output : YES → hold đến L18

Primitive sequence: OS_1x1(384→192) + RS_DENSE_3x3 + CONCAT + OS_1x1(→128)
  (Cin=384, Cout=128 – thu hẹp channels)

Output: F13 [1,128,40,40]
⚠️  HOLD_SKIP = True, hold_until = L18
```

---

#### L14 – Upsample [128,40,40] → [128,80,80]

```
Primitive   : UPSAMPLE_NEAREST(scale=2)
Sources     : F13
Hold output : No

Output: F14 [1,128,80,80] (scale/zp từ F13)
```

---

#### L15 – QConcat [128,80,80]+[64,80,80] → [192,80,80]

```
Primitive   : CONCAT
Sources     : [F14, F4]  ← SKIP DEPENDENCY: F4 từ L4 (xa nhất, 11 layer)
Hold output : No

Input A: F14 [1,128,80,80] ← từ upsample
Input B: F4  [1, 64,80,80] ← từ backbone L4 skip

BARRIER: L15 phải đợi BOTH L14_done AND F4_hold_ready

Output: F15 [1,192,80,80]
```

---

#### L16 – QC2f [192,80,80] → [64,80,80]

```
Block type  : QC2f (n=1)
Sources     : F15
Hold output : No  ← ĐÂY LÀ OUTPUT P3

Primitive sequence: OS_1x1(192→96) + RS_DENSE_3x3 + CONCAT + OS_1x1(→64)
  (Cin=192, Cout=64 – spatial 80×80 LỚN NHẤT trong toàn model)

Output: F16 = P3_int8 [1,64,80,80]   ✅ P3 OUTPUT
```

---

### NECK – PAN (L17–L22)

---

#### L17 – Conv [64,80,80] → [64,40,40]

```
Primitive   : RS_DENSE_3x3(stride=2)
Sources     : F16 (= P3)
Hold output : No

Primitive params: Cin=64, Cout=64, stride=2, H=80→40

Output: F17 [1,64,40,40]
```

---

#### L18 – QConcat [64,40,40]+[128,40,40] → [192,40,40]

```
Primitive   : CONCAT
Sources     : [F17, F13]  ← SKIP DEPENDENCY: F13 từ L13
Hold output : No

Input A: F17 [1, 64,40,40] ← từ conv downsample P3
Input B: F13 [1,128,40,40] ← từ FPN mid L13 skip

BARRIER: L18 phải đợi BOTH L17_done AND F13_hold_ready

Output: F18 [1,192,40,40]
```

---

#### L19 – QC2f [192,40,40] → [128,40,40]

```
Block type  : QC2f (n=1)
Sources     : F18
Hold output : No  ← ĐÂY LÀ OUTPUT P4

Primitive sequence: OS_1x1(192→128) + RS_DENSE_3x3 + CONCAT + OS_1x1(→128)

Output: F19 = P4_int8 [1,128,40,40]   ✅ P4 OUTPUT
```

---

#### L20 – SCDown [128,40,40] → [128,20,20]

```
Block type  : SCDown
Sources     : F19 (= P4)
Hold output : No

Primitive sequence: (tương tự L5, Cin=Cout=128)
  Branch A: OS_1x1(128→64) → DW_3x3(s2) → [1,64,20,20]
  Branch B: OS_1x1(128→64) → DW_3x3(s2) → [1,64,20,20]
  CONCAT → F20 [1,128,20,20]

Output: F20 [1,128,20,20]
```

---

#### L21 – QConcat [128,20,20]+[256,20,20] → [384,20,20]

```
Primitive   : CONCAT
Sources     : [F20, F8]  ← SKIP DEPENDENCY: F8 từ L8 (dài nhất, 13 layer)
Hold output : No

Input A: F20 [1,128,20,20] ← từ SCDown
Input B: F8  [1,256,20,20] ← từ deep backbone L8 skip

BARRIER: L21 phải đợi BOTH L20_done AND F8_hold_ready

Output: F21 [1,384,20,20]
```

---

#### L22 – QC2fCIB [384,20,20] → [256,20,20]

```
Block type  : QC2fCIB (C2f with CIB – large kernel)
Sources     : F21
Hold output : No  ← ĐÂY LÀ OUTPUT P5

Primitive sequence:
  Step 1: OS_1x1(Cin=384, Cout=256)       → X1 [1,256,20,20]

  Step 2: CIB bottleneck (nhánh branch):
    DW_7x7_MULTIPASS(C=128)               → Y_dw [1,128,20,20]
    OS_1x1(Cin=128, Cout=128)             → Y_cib [1,128,20,20]

  Step 3: CONCAT(Y_cib, X1_split_half)    → Ycat [1,256,20,20]

  Step 4: OS_1x1(Cin=256, Cout=256)       → F22 [1,256,20,20]

Note: DW_7x7_MULTIPASS với 3 pass (rows 0-2, 3-5, 6)
      Trace PSUM sau mỗi pass bắt buộc

Output: F22 = P5_int8 [1,256,20,20]   ✅ P5 OUTPUT
```

---

## 4. Bốn Skip Dependencies Bắt Buộc

```
┌────────────────────────────────────────────────────────────────────┐
│  Skip    │ Source │ Hold from │ Destination │ Hold to │ Dist │ Size│
├────────────────────────────────────────────────────────────────────┤
│ SKIP-A   │  F4    │    L4     │   L15       │   L15   │  11  │400K│
│ SKIP-B   │  F6    │    L6     │   L12       │   L12   │   6  │205K│
│ SKIP-C   │  F8    │    L8     │   L21       │   L21   │  13  │102K│  ← Dài nhất
│ SKIP-D   │  F13   │   L13     │   L18       │   L18   │   5  │205K│
└────────────────────────────────────────────────────────────────────┘
Total GLB skip buffer: ~912 KB cần dự trữ đồng thời
```

### Đồ thị phụ thuộc đầy đủ:

```
Input X_int8
    │
    L0→L1→L2→L3→L4 ──────────────────────────────────────── SKIP-A ──► L15
                    │                                                      ▲
                    └─► L5→L6 ──────────────────────── SKIP-B ──► L12    │
                               │                                   ▲      │
                               └─► L7→L8 ──── SKIP-C ──► L21     │      │
                                           │              ▲       │      │
                                           └─► L9→L10→L11─────────┘      │
                                                          │               │
                                                         L12              │
                                                          │               │
                                                         L13 ── SKIP-D ──► L18
                                                          │               ▲
                                                         L14              │
                                                          │               │
                                                         L15              │
                                                          │               │
                P3 ◄────────────────────────────────── L16        L17 ───┘
                                                                    │
                                                                   L18
                                                                    │
                P4 ◄─────────────────────────────────────────── L19
                                                                    │
                                                                   L20
                                                                    │
                                                                   L21
                                                                    │
                P5 ◄─────────────────────────────────────────── L22
```

---

## 5. Barrier Logic (4 điểm đồng bộ)

```
barrier_L12:
  precondition: (L11 == DONE) AND (F6_hold == READY)
  action: release L12_compute_start
  error_case: timeout nếu F6 không được hold đúng

barrier_L15:
  precondition: (L14 == DONE) AND (F4_hold == READY)
  action: release L15_compute_start
  error_case: timeout nếu F4 không được hold đúng

barrier_L18:
  precondition: (L17 == DONE) AND (F13_hold == READY)
  action: release L18_compute_start
  error_case: timeout nếu F13 không được hold đúng

barrier_L21:
  precondition: (L20 == DONE) AND (F8_hold == READY)
  action: release L21_compute_start
  error_case: timeout nếu F8 không được hold đúng
```

---

## 6. Output Mapping

| Output | Layer | Shape | Role |
|---|---|---|---|
| P3_int8 | L16 | [1, 64, 80, 80] | Small object feature (1/8 scale) |
| P4_int8 | L19 | [1, 128, 40, 40] | Medium object feature (1/16 scale) |
| P5_int8 | L22 | [1, 256, 20, 20] | Large object feature (1/32 scale) |

Quant metadata đi kèm: `(scale_P3, zp_P3)`, `(scale_P4, zp_P4)`, `(scale_P5, zp_P5)`

---

## 7. Sign-off Checklist

```
☐ Layer count: 23 layers (L0–L22) confirmed từ trace
☐ Shape của mỗi layer match với trace thực tế
☐ Hold_skip flags đúng cho: L4, L6, L8, L13
☐ 4 barrier conditions được xác nhận
☐ 3 output layers xác nhận: L16=P3, L19=P4, L22=P5
☐ DW_7x7_MULTIPASS tại L22 được annotate với pass split
☐ SPPF tại L9: OS_1x1 + MAXPOOL×3 + CONCAT×4 + OS_1x1 được confirm
☐ QPSA tại L10: OS_1x1 + GEMM_ATTN_BASIC + CONCAT + OS_1x1 được confirm
☐ Total skip buffer ~912KB được accepted trong GLB capacity planning
```

*File này là nguồn chân lý cho layer-to-primitive mapping. Không được hardcode sequence trong runner.*


---
---

<a id='phần-vi3---quantization-policy'></a>

# PHẦN VI.3 — Quantization Policy
> Nguồn: `PHASE0/03_quant_policy.md`

---

# 03 – Quantization Policy (Freeze Spec)
## qYOLOv10n INT8 PTQ – Quant Policy cho Model-Forward

> **Trạng thái**: FREEZE | **Version**: v1.0 | **Ngày**: 2026-03-16

---

## 1. Mục Đích

Chốt toàn bộ chính sách quantization: cách biểu diễn số, cách requant, cách xử lý CONCAT/ADD domain. Đây là file quan trọng nhất để tránh sai ở neck.

---

## 2. Quantization Scheme Cơ Bản

### 2.1. Affine Quantization Formula

```
Quantize:    x_int = clamp(round(x_float / scale) + zp, min_val, max_val)
Dequantize:  x_float = (x_int - zp) * scale

Trong đó:
  scale > 0   (float32)
  zp          (int32)
  min_val = -128, max_val = 127  (INT8 signed)
```

### 2.2. Bảng Policy Theo Đối Tượng

| Đối tượng | Scheme | Granularity | ZP |
|---|---|---|---|
| Activation input | INT8 | Per-tensor (1 scale, 1 zp cho toàn tensor) | Tự do |
| Activation output | INT8 | Per-tensor | Tự do |
| Weight conv/dw | INT8 | Per-output-channel (1 scale cho mỗi Cout) | **zp_w = 0** (symmetric) |
| Bias | INT32 | Per-output-channel | Không áp dụng |
| PSUM accumulator | INT32 | Nội bộ hardware | N/A |

---

## 3. Bias Fusion (BN Offline)

Batch Normalization được fuse vào Conv weight và bias trước khi deploy:

```
Fused weight: W_fused[cout, cin, kh, kw] = W_orig * (gamma / sqrt(var + eps))
Fused bias:   B_fused[cout] = beta - gamma * mean / sqrt(var + eps)

Sau khi quantize weight → W_int8, bias được scale thành:
B_int32[cout] = round(B_fused[cout] / (scale_x * scale_w[cout]))

Trong đó:
  scale_x     = scale của activation input
  scale_w[cout] = scale của weight channel cout
  B_int32 là INT32, không bị quantize cắt precision
```

---

## 4. Requant (Integer-to-Integer Rescaling)

### 4.1. Công thức chuẩn

Dùng xuyên suốt cho tất cả primitive:

```
M[cout] = scale_x * scale_w[cout] / scale_y   (float32, tính offline)

Decompose M thành fixed-point:
  M_int = round(M * 2^shift)   (chọn shift để M_int ∈ [2^15, 2^16-1])
  
Tại runtime (hardware):
  y_raw = (acc_int32 * M_int) >> shift   (arithmetic right shift)
  y_clamped = clamp(y_raw + zp_y, -128, 127)
```

### 4.2. Quy tắc cứng (không được vi phạm)

- **Chỉ một đường requant**: Không tự implement lại outside `quant_affine.py`
- **Không dùng float dequantize trong execution path**: Float chỉ cho reference test
- **Shift range**: shift ∈ [0, 31], M_int phải fit INT32
- **Rounding**: `round()` là round-half-to-even (banker's rounding) hoặc round-half-up (chọn nhất quán và khóa)

---

## 5. Activation Policy

### 5.1. SiLU (Sigmoid Linear Unit)

```
SiLU(x) = x × sigmoid(x) = x / (1 + exp(-x))

Phương án hardware: LUT 256 entry (precomputed)
  LUT_index = y_int8_pre_act + 128    (shift về [0,255] range)
  y_silu = SiLU_LUT[LUT_index]

LUT được tính offline từ float SiLU:
  for i in range(-128, 128):
    x_float = (i - zp_pre_act) * scale_pre_act
    y_float  = x_float * sigmoid(x_float)
    LUT[i+128] = clamp(round(y_float / scale_y + zp_y), -128, 127)

Ràng buộc: LUT phải nhất quán giữa Golden Python và RTL (same rounding policy).
```

### 5.2. Không có activation (identity)

Một số OS_1x1 không có activation (trong bottleneck internal). Flag `activation=None`.

---

## 6. Policy cho Tensor Operations

### 6.1. UPSAMPLE_NEAREST

```
scale_out = scale_in    ← KHÔNG ĐỔI
zp_out    = zp_in       ← KHÔNG ĐỔI
Dữ liệu INT8 chỉ copy, không có arithmetic
```

### 6.2. MOVE (Skip buffer copy)

```
scale_out = scale_in    ← KHÔNG ĐỔI
zp_out    = zp_in       ← KHÔNG ĐỔI
```

### 6.3. MAXPOOL_5x5

```
scale_out = scale_in    ← KHÔNG ĐỔI
zp_out    = zp_in       ← KHÔNG ĐỔI
Max comparison là INT8 comparison: đúng về mặt số học vì scale/zp chung.
```

### 6.4. CONCAT – Common-Domain Requant

**Đây là phần quan trọng nhất, rủi ro số 1 của neck.**

```
Inputs: A_int8 (scale_A, zp_A), B_int8 (scale_B, zp_B)
Output: Y_int8 (scale_Y, zp_Y)

scale_Y, zp_Y: được xác định bởi PTQ calibration của layer QConcat.
               (KHÔNG tự chọn; lấy từ calibrated model)

Bước 1: Requant A nếu cần
  if |scale_A - scale_Y| > epsilon OR zp_A ≠ zp_Y:
    A_float = (A_int8.float() - zp_A) * scale_A
    A_aligned = clamp(round(A_float / scale_Y) + zp_Y, -128, 127).int8()
  else:
    A_aligned = A_int8  (no requant, pass-through)

Bước 2: Requant B nếu cần (tương tự)

Bước 3: Concat theo channel
  Y_int8 = numpy.concatenate([A_aligned, B_aligned], axis=channel)
  Y có quant params (scale_Y, zp_Y)
```

**Áp dụng tại 4 QConcat layers:**

| Layer | Input A | Input B | Ghi chú |
|---|---|---|---|
| L12 | F11 (từ upsample path) | F6 (backbone skip) | Risk cao – path dài |
| L15 | F14 (từ upsample path) | F4 (backbone skip SKIP-A) | Risk cao nhất – skip 11 layer |
| L18 | F17 (từ PAN down path) | F13 (FPN mid skip) | Risk trung bình |
| L21 | F20 (từ SCDown path) | F8 (deep backbone SKIP-C) | Risk cao – skip 13 layer |

**Lý do domain mismatch nguy hiểm**: F4, F6, F8 và F13 đi qua nhiều conv → scale drift xa; trong khi nhánh upsample/downsample đi qua ít conv hơn. Khi concat mà không có alignment → outlier artifact, object detection sai.

### 6.5. EWISE_ADD – Common-Domain Add

```
Inputs: A_int8 (scale_A, zp_A), B_int8 (scale_B, zp_B)
Output: Y_int8 (scale_Y, zp_Y)   ← calibrated

Bước 1: Align cả A và B về scale_Y, zp_Y (như CONCAT)
Bước 2: Add với intermediate INT16 để tránh overflow:
  sum_int16 = int16(A_aligned - zp_Y) + int16(B_aligned - zp_Y)
Bước 3: Requant về output:
  Y_pre = round(sum_int16 * scale_Y / scale_Y_out) + zp_Y_out
  Y = clamp(Y_pre, -128, 127)

Saturation rule: values > 127 → 127; values < -128 → -128 (hard clamp)
```

---

## 7. PSUM Mode và Last-Pass Policy

```
if NOT last_pass:
  output_namespace = PSUM_BUFFER  (INT32, accumulate phase)
  PPU không kích hoạt
  Không write ra GLB_OUTPUT (INT8)

if last_pass:
  output_namespace = ACT_BUFFER  (INT8, sau PPU)
  PPU kích hoạt:
    1. bias_add  : PSUM + B_int32[cout]
    2. requant   : (PSUM * M_int) >> shift + zp_out
    3. activation: SiLU_LUT (nếu có)
    4. clamp     : [-128, 127]
  Write ra GLB_OUTPUT

last_pass = last_cin AND last_kernel AND last_reduce
  last_cin    : đây là Cin chunk cuối (channel reduction hoàn tất)
  last_kernel : đây là kernel position cuối (spatial reduction hoàn tất)
  last_reduce : đây là reduce operation cuối trong tile
```

---

## 8. Quy Tắc Absolute (Không Được Cưỡng Lại)

```
Rule Q1: ZP của weight = 0 (symmetric quantization)
         → zp_w[cout] = 0 cho tất cả P0,P1,P2,P8

Rule Q2: Float dequantize chỉ cho reference / debug
         → Execution path chỉ dùng integer arithmetic

Rule Q3: Một implementation requant duy nhất
         → Tất cả file Python và RTL đều import/instantiate từ cùng module

Rule Q4: scale_Y trong CONCAT/ADD lấy từ PTQ calibration
         → Không tự chọn scale; không dùng max(scale_A, scale_B) trong production

Rule Q5: CONCAT/ADD PHẢI kiểm tra domain trước khi kết hợp
         → Dù scale khác 1e-6 cũng phải align

Rule Q6: Rounding policy nhất quán: round-half-up (away from zero)
         → Áp dụng giống nhau trong Golden Python và RTL
```

---

## 9. Ví Dụ Số Cụ Thể

### Layer 0: Conv(s=2), scale_x=1/255=0.003921, zp_x=0

```
X_int8 giá trị pixel=128 → x_float=0.502 (≈128/255)

weight W_int8[0,0,1,1]=64, scale_w[0]=0.001, zp_w=0
→ w_float = 64 * 0.001 = 0.064

acc_raw = Σ x_int8 * w_int8 (ví dụ: 1 entry = 128*64 = 8192)
zp_correction = 0 * Σ W_int8 = 0 (vì zp_x = 0)
B_int32[0] = 150 (giả định)

acc = 8192 + 150 = 8342  (cộng dồn tất cả kernel positions)

scale_y[0] = 0.025 (calibrated)
M[0] = 0.003921 * 0.001 / 0.025 = 0.0001568
M_int = round(0.0001568 * 2^23) = round(1314) = 1314, shift=23

y_raw = (8342 * 1314) >> 23 = 10961388 >> 23 = 1 (nếu acc nhỏ thế)
→ Thực tế acc lớn hơn nhiều sau toàn kernel sum

y_int8 = clamp(y_raw + zp_y, -128, 127)
```

### CONCAT tại L12: domain alignment

```
F11 (Upsample output): scale_A=0.05, zp_A=0
F6  (Backbone skip):   scale_B=0.09, zp_B=2
scale_Y (calibrated L12 QConcat output) = 0.07, zp_Y = 0

Requant F11 (scale_A=0.05 ≠ scale_Y=0.07):
  x_float = (F11_int8 - 0) * 0.05
  A_aligned = clamp(round(x_float / 0.07) + 0, -128, 127)
  → Ví dụ: F11_int8=100 → x_float=5.0 → A_aligned=round(5.0/0.07)=round(71.4)=71

Requant F6 (scale_B=0.09 ≠ scale_Y=0.07):
  x_float = (F6_int8 - 2) * 0.09
  B_aligned = clamp(round(x_float / 0.07) + 0, -128, 127)
  → Ví dụ: F6_int8=50 → x_float=(50-2)*0.09=4.32 → B_aligned=round(4.32/0.07)=round(61.7)=62

CONCAT([A_aligned[256ch], B_aligned[128ch]]) → Y[384ch], scale_Y=0.07, zp_Y=0
```

---

## 10. Sign-off Checklist

```
☐ Activation quantization: INT8 per-tensor confirmed
☐ Weight quantization: INT8 per-output-channel, zp_w=0 confirmed
☐ Bias quantization: INT32 per-channel, B=round(b_fused/(s_x * s_w)) confirmed
☐ Requant formula khóa: (acc*M_int)>>shift + zp_y, rounding=round-half-up
☐ SiLU LUT: 256 entries, offline precomputed, nhất quán Python ↔ RTL
☐ UPSAMPLE/MAXPOOL/MOVE: scale/zp pass-through confirmed (không đổi)
☐ CONCAT Policy: common-domain từ PTQ calibration, requant trước concat
☐ EWISE_ADD Policy: common-domain + INT16 intermediate để tránh overflow
☐ PSUM/ACT: PPU chỉ kích hoạt tại last_pass confirmed
☐ Tất cả ví dụ số được verify bằng tay hoặc Python script
```

*Mọi thay đổi quant policy sau file này phải được lượng hóa impact và sign-off lại.*


---
---

<a id='phần-vi4---layout--addressing'></a>

# PHẦN VI.4 — Layout & Addressing
> Nguồn: `PHASE0/04_layout_addressing.md`

---

# 04 – Layout & Addressing (Freeze Spec)
## qYOLOv10n INT8 Accelerator – Memory Layout & Address Generation

> **Trạng thái**: FREEZE | **Version**: v1.0 | **Ngày**: 2026-03-16

---

## 1. Mục Đích

Chốt quy tắc banking, row-slot, lane packing và address generation. Software, Golden Python và RTL phải dùng cùng công thức này.

---

## 2. Tổng Quan Kiến Trúc Bộ Nhớ

```
┌──────────────────────────────────────────────────────────────────┐
│                    Global Line Buffer (GLB)                      │
│                                                                  │
│  INPUT  ┌──────┐  ┌──────┐  ┌──────┐                           │
│  BANKS  │Bank 0│  │Bank 1│  │Bank 2│   (3 banks, h mod 3)      │
│         └──────┘  └──────┘  └──────┘                           │
│                                                                  │
│  OUTPUT ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                │
│  BANKS  │Bank 0│  │Bank 1│  │Bank 2│  │Bank 3│ (4 banks)      │
│         └──────┘  └──────┘  └──────┘  └──────┘                │
│                                                                  │
│  WEIGHT ┌─────────────────────────────────────────┐            │
│  SRAM   │ Weights + Bias (packed, stationary)     │            │
│         └─────────────────────────────────────────┘            │
│                                                                  │
│  PSUM   ┌─────────────────────────────────────────┐            │
│  BUF    │ INT32 accumulator (non-last_pass)        │            │
│         └─────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Input Banking Model

### 3.1. Quy Tắc Phân Bank

```
bank_input(h) = h mod 3

Mapping:
  h=0  → Bank 0
  h=1  → Bank 1
  h=2  → Bank 2
  h=3  → Bank 0  (cyclic)
  h=4  → Bank 1
  ...
```

**Lý do 3 banks**: Conv3×3 cần 3 hàng liên tiếp (h-1, h, h+1). 3 banks xoay vòng cho phép đọc 3 hàng này từ 3 bank khác nhau đồng thời mà không xung đột (no bank conflict).

### 3.2. Sliding Window Với Banking

```
Conv3x3 stride=1, output row h_out:
  Cần input rows: h_in = h_out-1, h_out, h_out+1
  Bank(h_out-1) = (h_out-1) mod 3
  Bank(h_out)   = h_out mod 3
  Bank(h_out+1) = (h_out+1) mod 3
  → 3 giá trị khác nhau (no conflict)

Conv3x3 stride=2, output row h_out:
  Cần input rows: h_in = 2*h_out-1, 2*h_out, 2*h_out+1
  Bank(2*h_out-1) = (2*h_out-1) mod 3
  Bank(2*h_out)   = 2*h_out mod 3
  Bank(2*h_out+1) = (2*h_out+1) mod 3
  → 3 giá trị khác nhau (no conflict)
```

### 3.3. Ví Dụ Cụ Thể (Layer 0: H=640)

```
Output h_out=0: reads rows -1(pad), 0, 1
  Bank(-1) = padding (bank 2, value 0)
  Bank(0)  = 0 → Bank 0
  Bank(1)  = 1 → Bank 1

Output h_out=1: reads rows 0, 1, 2
  Bank(0)  → Bank 0
  Bank(1)  → Bank 1
  Bank(2)  → Bank 2

Output h_out=2: reads rows 1, 2, 3
  Bank(1)  → Bank 1
  Bank(2)  → Bank 2
  Bank(3)  → Bank 0   ← Bank 0 đã giải phóng h=0
```

---

## 4. Row Slot Model

### 4.1. Công Thức

```
Q_in     = ceil((K_eff + 3 × stride) / 3)
row_slot  = floor(h / 3) mod Q_in
```

**Ý nghĩa**: Mỗi bank có Q_in slot positions để lưu các hàng. Row h được lưu vào slot `floor(h/3) mod Q_in` trong bank `h mod 3`.

### 4.2. Tính Q_in Theo Primitive

| Primitive | K_eff | stride | Q_in | Số slot/bank |
|---|---|---|---|---|
| Conv3x3 stride=1 | 3 | 1 | ceil((3+3)/3)=2 | 2 |
| Conv3x3 stride=2 | 3 | 2 | ceil((3+6)/3)=3 | 3 |
| DW_3x3 stride=1 | 3 | 1 | 2 | 2 |
| DW_3x3 stride=2 | 3 | 2 | 3 | 3 |
| DW_7x7 stride=1 | 7 | 1 | ceil((7+3)/3)=4 | 4 |
| MaxPool5x5 s=1 | 5 | 1 | ceil((5+3)/3)=3 | 3 |
| Conv1x1 s=1 | 1 | 1 | ceil((1+3)/3)=2 | 2 |

### 4.3. Row Slot Ví Dụ (Conv3x3 stride=1, Q_in=2)

```
h=0: slot = floor(0/3) mod 2 = 0 mod 2 = 0  → Bank0, Slot0
h=1: slot = floor(1/3) mod 2 = 0 mod 2 = 0  → Bank1, Slot0
h=2: slot = floor(2/3) mod 2 = 0 mod 2 = 0  → Bank2, Slot0
h=3: slot = floor(3/3) mod 2 = 1 mod 2 = 1  → Bank0, Slot1
h=4: slot = floor(4/3) mod 2 = 1 mod 2 = 1  → Bank1, Slot1
h=5: slot = floor(5/3) mod 2 = 1 mod 2 = 1  → Bank2, Slot1
h=6: slot = floor(6/3) mod 2 = 2 mod 2 = 0  → Bank0, Slot0 (reuse!)
```

→ Slot0 của Bank0 được reuse (row 0→6→12→...), row cũ không cần nữa → tiết kiệm bộ nhớ.

---

## 5. Lane Packing

### 5.1. Lane Constants

```
LANES = 16
```

### 5.2. Spatial Decomposition

```
lane  (spatial x position in warp) = x mod 16
Wblk  (horizontal block index)      = floor(x / 16)
Wblk_total = ceil(W / LANES) = ceil(W / 16)
```

### 5.3. Pack/Unpack

```
pack16: data[H, W, Cin] → packed[H, Wblk_total, Cin, 16]
  packed[h, wblk, cin, lane] = data[h, wblk*16 + lane, cin]
  (nếu wblk*16+lane >= W: padding với zp_x)

unpack16: packed[H, Wblk_total, Cin, 16] → data[H, W, Cin]
  data[h, wblk*16+lane, cin] = packed[h, wblk, cin, lane]

Invariant: unpack16(pack16(x)) == x  (no data loss)
```

### 5.4. Ví Dụ (W=40, LANES=16)

```
Wblk_total = ceil(40/16) = 3
  Wblk=0: columns 0..15
  Wblk=1: columns 16..31
  Wblk=2: columns 32..39 (6 valid + 10 padding)
```

---

## 6. Address Generation

### 6.1. Input Address

```
physical_addr_input(h, x, cin) =
  bank       = bank_input(h)  = h mod 3
  slot       = row_slot(h, Q_in)
  lane       = x mod 16
  Wblk       = x // 16

  offset = slot × (Wblk_total × Cin × 16)
         + Wblk × (Cin × 16)
         + cin  × 16
         + lane

  return (bank, offset)
```

**Không overlap guarantee**: Mỗi pixel (h,x,cin) map đến unique (bank, offset) nếu slot và bank đúng.

### 6.2. Output Address

```
physical_addr_output(h_out, x_out, cout) =
  bank_out   = h_out mod 4   (4 output banks)
  lane_out   = x_out mod 16
  Wblk_out   = x_out // 16
  Wblk_out_total = ceil(W_out / 16)

  offset_out = (h_out // 4) × (Wblk_out_total × Cout × 16)
             + Wblk_out × (Cout × 16)
             + cout × 16
             + lane_out

  return (bank_out, offset_out)
```

### 6.3. Weight Address

```
Weight layout: [Cout, Cin, kH, kW] (KCRS format)

addr_weight(cout, cin, kh, kw) =
  = cout × (Cin × kH × kW) + cin × (kH × kW) + kh × kW + kw

Weight được load theo cout_chunk → packed tương ứng
```

### 6.4. Ví Dụ Đầy Đủ (Layer 0: 640×640→320×320, Cin=3, Cout=16)

```
Q_in = ceil((3 + 3*2)/3) = 3  (stride=2)
Wblk_total = ceil(640/16) = 40

Input pixel (h=4, x=32, cin=2):
  bank  = 4 mod 3 = 1        → Bank 1
  slot  = floor(4/3) mod 3 = 1 mod 3 = 1
  lane  = 32 mod 16 = 0
  Wblk  = 32 // 16 = 2
  offset = 1 × (40 × 3 × 16) + 2 × (3 × 16) + 2 × 16 + 0
         = 1920 + 96 + 32 + 0 = 2048

Output pixel (h_out=2, x_out=16, cout=5):
  bank_out = 2 mod 4 = 2    → Output Bank 2
  Wblk_out_total = ceil(320/16) = 20
  lane_out = 16 mod 16 = 0
  Wblk_out = 16 // 16 = 1
  offset_out = (2//4) × (20 × 16 × 16) + 1 × (16 × 16) + 5 × 16 + 0
             = 0 + 256 + 80 + 0 = 336
```

---

## 7. PSUM Buffer

### 7.1. PSUM Namespace

```
Khi NOT last_pass:
  output → PSUM_BUF[cout, h_out, w_out]  (INT32)
  Địa chỉ riêng, không chia sẻ với ACT output

Khi last_pass:
  PSUM_BUF → PPU (bias_add + requant + act + clamp) → ACT_BUF (INT8)
  ACT_BUF write vào GLB_OUTPUT với physical_addr_output()
```

### 7.2. PSUM Buffer Sizing

```
PSUM_BUF kích thước = Cout_chunk × tile_H × tile_W × 4 bytes (INT32)

Ví dụ tile Cout=16, tile_H=8, tile_W=16:
  PSUM_BUF = 16 × 8 × 16 × 4 = 8,192 bytes per tile
```

---

## 8. HOLD_SKIP Buffer

### 8.1. Nguyên Tắc

Khi một tensor cần được giữ lại cho skip connection sau:
1. Ghi output bình thường vào GLB_OUTPUT
2. Set `HOLD_SKIP = True` trong TILE_DESC
3. Không giải phóng GLB vùng nhớ đó cho đến khi consumer layer xong

### 8.2. Sizing Cụ Thể

| Skip buffer | Shape | Bytes | Held từ | Giải phóng tại |
|---|---|---|---|---|
| F4 (SKIP-A) | INT8[1,64,80,80] | 409,600 | L4 done | L15 done |
| F6 (SKIP-B) | INT8[1,128,40,40] | 204,800 | L6 done | L12 done |
| F8 (SKIP-C) | INT8[1,256,20,20] | 102,400 | L8 done | L21 done |
| F13 (SKIP-D) | INT8[1,128,40,40] | 204,800 | L13 done | L18 done |

**Tổng peak simultaneous**: F4 + F6 + F8 = 716,800 bytes ~700KB  
(F13 sinh sau khi F6 đã giải phóng, nên không cộng cùng lúc với F6)

**Peak thực tế**: Tại thời điểm L14→L15: F4(SKIP-A) + F8(SKIP-C) = ~512KB đang held

---

## 9. Ví Dụ Mapping Cho Từng Primitive

### Conv3x3 stride=1 (Layer nội bộ QC2f tại H=160):

```
Q_in = 2, Wblk_total = ceil(160/16) = 10, Cin=16

load_stage: h=0,1,2 → fill Bank0/Bank1/Bank2 slot0
compute_stage h_out=0:
  read Bank2 (h=−1, padded), Bank0 (h=0), Bank1 (h=1)
  PE MAC 16 lanes × 16 channels → INT32 psum

advance_stage h_out=1:
  row h=2 still valid in Bank2
  load h=3 → Bank0 slot1
  read Bank0 (h=−1 for h_out=1? No: h_in=0,1,2 for h_out=1)
  ...
```

### DW_3x3 stride=2 (SCDown at H=80):

```
Q_in = 3, Wblk_total = ceil(80/16) = 5, C=64 (per-channel independent)

For h_out=0: need h_in = -1(pad), 0, 1
  Bank_pad = padded zeros
  Bank(0) = Bank0, slot=0
  Bank(1) = Bank1, slot=0
  
For h_out=1: need h_in = 1, 2, 3
  Bank(1) = Bank1, slot0
  Bank(2) = Bank2, slot0
  Bank(3) = Bank0, slot1  (h=3 → bank=0, slot=floor(3/3)%3=1)

PE DW mode: lane=0..15 process columns 0..15
  Each lane processes 1 spatial column × C channels independently
  Per-channel weight: W[c, 0, kh, kw] for each channel c
```

### CONCAT (QConcat L12, H=40):

```
Input A: F11 [256, 40, 40] → stored in GLB region A
Input B: F6  [128, 40, 40] → stored in HOLD_SKIP region B

Router operation:
  For each h_row, w_col:
    If common-domain requant needed:
      Read A_val, compute A_aligned via mini-PPU
      Read B_val, compute B_aligned if needed
    Write [A_aligned_channels, B_aligned_channels] to output
    
Output interleaved by channel: [A_ch0..255, B_ch0..127] → Y[0..383]
```

---

## 10. Sign-off Checklist

```
BANKING:
☐ bank_input=h%3: verify h=0→0, h=1→1, h=2→2, h=3→0 (cycle)
☐ bank_output=h_out%4: verify h_out=0→0, 1→1, 2→2, 3→3, 4→0
☐ 3 banks đồng thời: verify no bank conflict tại h_out cho Conv3x3

ROW SLOT:
☐ Q_in Conv3x3 s1 = 2: verified
☐ Q_in Conv3x3 s2 = 3: verified
☐ Q_in DW7x7 s1  = 4: verified
☐ Q_in MaxPool5x5 s1 = 3: verified
☐ row reuse: h=0 và h=6 dùng cùng slot → no collision vì L[0] không cần khi compute L[6]

LANE:
☐ pack16/unpack16 round-trip: unpack(pack(x))==x cho mọi W
☐ Edge Wblk padding: columns >= W set về zp_x

ADDRESS:
☐ No-overlap: mọi (h,x,cin) → unique (bank, offset)
☐ Layer 0 example: pixpel (4,32,2) → (Bank1, 2048) verified
☐ Output address: h_out=2, x_out=16, cout=5 → (Bank2, 336) verified

SKIP BUFFER:
☐ F4 409,600 bytes allocated from L4_done đến L15_done
☐ F6 204,800 bytes allocated from L6_done đến L12_done
☐ F8 102,400 bytes allocated from L8_done đến L21_done
☐ F13 204,800 bytes allocated from L13_done đến L18_done
☐ Peak simultaneous: ~512KB có thể fit trong GLB
```

*Layout/addressing là foundation cho RTL addr_gen_input, addr_gen_output, row_slot_manager.*


---
---

<a id='phần-vi5---descriptor-spec'></a>

# PHẦN VI.5 — Descriptor Spec
> Nguồn: `PHASE0/05_descriptor_spec.md`

---

# 05 – Descriptor Spec (Freeze Spec)
## qYOLOv10n INT8 Accelerator – Descriptor Stack Format

> **Trạng thái**: FREEZE | **Version**: v1.0 | **Ngày**: 2026-03-16

---

## 1. Mục Đích

Chốt format descriptor stack mà software phát lệnh cho hardware. `desc_fetch_engine.sv` và compiler phải tuân thủ file này.

---

## 2. Tổng Quan Descriptor Hierarchy

```
NET_DESC (1 per inference)
  └── LAYER_DESC (1 per layer, L0–L22)
        └── TILE_DESC (N per layer, N = số tile)
              ├── ROUTER_PROFILE (1 per tile)
              └── POST_PROFILE   (1 per tile)
```

---

## 3. NET_DESC

**Mô tả**: Descriptor mức network, phát một lần trước khi bắt đầu inference.

```
struct NET_DESC {
  uint32_t  version;          // Phiên bản format = 1
  uint32_t  num_layers;       // Số layer = 23 (L0..L22)
  uint32_t  layer_table_base; // Địa chỉ SRAM của LAYER_DESC[0]
  uint32_t  weight_base;      // Địa chỉ đầu của vùng weight SRAM
  uint32_t  act_base;         // Địa chỉ đầu của activation buffer
  uint32_t  psum_base;        // Địa chỉ đầu của PSUM buffer
  uint32_t  skip_buf_base;    // Địa chỉ đầu của HOLD_SKIP buffer vùng
  uint32_t  reserved[1];      // Align to 32 bytes
};  // Total: 32 bytes
```

---

## 4. LAYER_DESC

**Mô tả**: Descriptor mức layer, một per layer.

```
struct LAYER_DESC {
  // Identity
  uint8_t   layer_idx;        // 0..22
  uint8_t   primitive_id;     // P0..P9 (xem primitive_matrix.md)
  uint8_t   block_type;       // CONV=0, QC2F=1, SCDOWN=2, SPPF=3,
                               // QPSA=4, UPSAMPLE=5, QCONCAT=6, QC2FCIB=7
  uint8_t   num_tiles;        // Số TILE_DESC cho layer này

  // Input tensor shape
  uint16_t  in_H, in_W;
  uint16_t  in_Cin;

  // Output tensor shape
  uint16_t  out_H, out_W;
  uint16_t  out_Cout;

  // Kernel params (dùng cho conv/dw primitives)
  uint8_t   kernel_h, kernel_w;
  uint8_t   stride_h, stride_w;
  uint8_t   pad_h, pad_w;
  uint8_t   groups;           // 1=dense conv; Cin=depthwise

  // Memory locations
  uint32_t  weight_offset;    // Offset từ NET_DESC.weight_base
  uint32_t  in_act_offset;    // Offset từ NET_DESC.act_base
  uint32_t  out_act_offset;   // Offset từ NET_DESC.act_base
  uint32_t  tile_desc_offset; // Offset từ layer_table_base đến TILE_DESC[0]

  // Skip/concat
  uint8_t   num_sources;      // 1=sequential; 2=concat (skip)
  uint8_t   source_layer[2];  // Layer index của từng source (255=none)
  uint8_t   hold_output;      // 1=giữ output trong HOLD_SKIP buffer
  uint8_t   hold_until_layer; // Giải phóng khi layer này done

  // Partition mode
  uint8_t   partition_mode;   // TILE_HW=0, TILE_COUT=1, TILE_CIN=2

  uint8_t   reserved[3];

};  // Total: 40 bytes, aligned
```

**Giá trị primitive_id**:
```
P0=0  RS_DENSE_3x3
P1=1  OS_1x1
P2=2  DW_3x3
P3=3  MAXPOOL_5x5
P4=4  MOVE
P5=5  CONCAT
P6=6  UPSAMPLE_NEAREST
P7=7  EWISE_ADD
P8=8  DW_7x7_MULTIPASS
P9=9  GEMM_ATTN_BASIC
```

---

## 5. TILE_DESC

**Mô tả**: Descriptor mức tile, định nghĩa một tile compute.

```
struct TILE_DESC {
  // Tile spatial bounds (output coordinates)
  uint16_t  h_out_start;      // First output row of tile
  uint16_t  h_out_end;        // Last output row (inclusive)
  uint16_t  w_blk_start;      // First Wblk index
  uint16_t  w_blk_end;        // Last Wblk index (inclusive)
  uint16_t  valid_h;          // Số valid output rows trong tile
  uint16_t  valid_w;          // Số valid columns (< LANES nếu edge)

  // Channel bounds
  uint16_t  cin_start;        // First input channel index
  uint16_t  cin_len;          // Số Cin channels xử lý
  uint16_t  cout_start;       // First output channel index
  uint16_t  cout_len;         // Số Cout channels xử lý (≤ LANES=16)

  // Control flags
  uint16_t  flags;            //  Bit field (xem bên dưới)
  uint16_t  last_flags;       //  Bit field (xem bên dưới)

  // Profile indices
  uint8_t   router_profile_id;
  uint8_t   post_profile_id;
  uint8_t   reserved[2];

};  // Total: 28 bytes
```

### 5.1. Flags Bit Field (flags)

```
Bit 0: first_tile
  1 = Đây là tile đầu tiên của layer
      → Reset PSUM accumulator về 0 trước khi MAC

Bit 1: edge_tile_h
  1 = Tile chạm biên trên hoặc dưới của ảnh
      → padding zeros cho các input rows ngoài biên
      → Điều chỉnh zp_correction cho padding pixels

Bit 2: edge_tile_w
  1 = Tile chạm biên trái hoặc phải
      → Xử lý Wblk cuối có valid_w < 16

Bit 3: hold_skip
  1 = Sau khi tile này done, output vùng [h_out_start..h_out_end] cần giữ
      → Không overwrite GLB vùng này cho đến khi consumer done

Bit 4: need_swizzle
  1 = Output cần qua swizzle_engine (reshape, upsample, concat-router)
      → Tensor-post path thay vì direct GLB write

Bit 5: psum_carry_in
  1 = PSUM từ tile trước cần được load vào accumulator trước khi MAC
      → Dùng khi một output position cần multiple Cin passes

Bits 6-15: reserved = 0
```

### 5.2. Last_Flags Bit Field (last_flags)

```
Bit 0: last_cin
  1 = Đây là Cin chunk cuối cho output position này
      → Sau tile này, Cin reduction hoàn tất
      
Bit 1: last_kernel
  1 = Đây là kernel position cuối (last kh, kw)
      → Sau tile này, spatial reduction hoàn tất

Bit 2: last_reduce
  1 = Đây là tile cuối trong reduction dimension
      → Trigger PPU path (bias_add + requant + act + clamp)
      
last_pass = last_cin AND last_kernel AND last_reduce
  → Khi last_pass=1: PPU kích hoạt, output ra ACT_namespace (INT8)
  → Khi last_pass=0: output vào PSUM_namespace (INT32)

Bit 3: last_pass_kernel (DW_7x7_MULTIPASS specific)
  1 = Đây là pass cuối của DW_7x7 kernel
      → Sau accumulate, trigger bias_add và requant

Bits 4-15: reserved = 0
```

---

## 6. ROUTER_PROFILE

**Mô tả**: Cấu hình routing từ source đến destination.

```
struct ROUTER_PROFILE {
  uint8_t   profile_id;       // Index của profile này
  uint8_t   source_select;    // SRC_GLB_IN=0, SRC_PSUM=1, SRC_SKIP_A=2,
                               // SRC_SKIP_B=3, SRC_EXTERNAL=4

  uint8_t   dest_select;      // DST_PE=0, DST_POOL=1, DST_SWIZZLE=2,
                               // DST_CONCAT=3, DST_GLB_OUT=4

  uint8_t   broadcast_mask;   // Bit mask: bit i=1 → broadcast to PE lane group i
                               // 0xFF = broadcast to all 16 lanes

  uint8_t   rps_destination;  // RPS = Router Path Select
                               // 0=pass-through, 1=interleave-A, 2=interleave-B
                               // 3=requant-path (mini PPU for CONCAT align)

  uint8_t   swizzle_mode;     // 0=none, 1=upsample_2x, 2=transpose,
                               // 3=concat_channel

  uint16_t  swizzle_param;    // Tham số cho swizzle (stride, offset, etc.)

};  // Total: 8 bytes
```

**Ví dụ Router Profiles**:

```
Profile 0 (Standard Conv):
  source_select = SRC_GLB_IN
  dest_select   = DST_PE
  broadcast_mask = 0xFF  (tất cả 16 lanes)
  rps_destination = 0 (pass-through)
  swizzle_mode = 0 (none)

Profile 1 (CONCAT path, domain align needed):
  source_select = SRC_SKIP_A  (F6, F4, F8, hoặc F13 từ hold buffer)
  dest_select   = DST_CONCAT
  broadcast_mask = 0x00
  rps_destination = 3 (requant-path, mini PPU thực hiện align)
  swizzle_mode = 3 (concat_channel)

Profile 2 (UPSAMPLE path):
  source_select = SRC_GLB_IN
  dest_select   = DST_SWIZZLE
  broadcast_mask = 0x00
  rps_destination = 0
  swizzle_mode = 1 (upsample_2x)

Profile 3 (MAXPOOL path):
  source_select = SRC_GLB_IN
  dest_select   = DST_POOL
  broadcast_mask = 0x00
  rps_destination = 0
  swizzle_mode = 0
```

---

## 7. POST_PROFILE

**Mô tả**: Cấu hình PPU (Post Processing Unit) – bias, requant, activation, clamp.

```
struct POST_PROFILE {
  uint8_t   profile_id;

  // Bias
  uint8_t   bias_en;          // 1=add bias, 0=skip bias
  uint32_t  bias_offset;      // Offset vào weight SRAM cho bias values

  // Requant
  uint8_t   requant_en;       // 1=apply requant
  int32_t   scale_mul;        // M_int (fixed-point multiplier), per-cout chunk
                               // (có thể là array, index theo cout_idx)
  uint8_t   scale_shift;      // Shift amount (0..31)

  // Zero point out
  int8_t    zp_out;           // Zero point của output tensor

  // Activation
  uint8_t   act_mode;         // 0=none, 1=SiLU_LUT, 2=ReLU, 3=ReLU6

  // Clamp
  int8_t    clamp_min;        // Default: -128
  int8_t    clamp_max;        // Default: 127

  // Common-domain requant (dùng cho CONCAT/ADD alignment)
  uint8_t   domain_align_en;  // 1=requant input trước khi concat/add
  float     domain_scale_in;  // scale_A hoặc scale_B (nguồn cần align)
  float     domain_scale_out; // scale_Y (common scale mục tiêu)
  int8_t    domain_zp_in;     // zp_A/zp_B
  int8_t    domain_zp_out;    // zp_Y
  uint8_t   reserved[2];

};  // Total: ~28 bytes
```

---

## 8. Ví Dụ Descriptor Stack Đầy Đủ (Layer 0)

### NET_DESC:

```
version          = 1
num_layers       = 23
layer_table_base = 0x0000_8000   (base SRAM addr)
weight_base      = 0x0001_0000
act_base         = 0x0010_0000   (activation ping-pong buffers)
psum_base        = 0x0020_0000
skip_buf_base    = 0x0030_0000
```

### LAYER_DESC[0] (Layer 0 – Conv stride=2):

```
layer_idx        = 0
primitive_id     = 0 (RS_DENSE_3x3)
block_type       = 0 (CONV)
num_tiles        = 320 (tất cả h_out=0..319, một tile mỗi row-block)

in_H=640, in_W=640, in_Cin=3
out_H=320, out_W=320, out_Cout=16

kernel_h=3, kernel_w=3
stride_h=2, stride_w=2
pad_h=1, pad_w=1
groups=1

weight_offset = 0  (ngay đầu weight region)
in_act_offset = 0  (X_int8, ngay đầu act region)
out_act_offset = 0x061_2000   (320*320*16=1,638,400 bytes offset)

num_sources      = 1
source_layer[0]  = 255 (no skip, lấy từ previous)
hold_output      = 0
hold_until_layer = 255

partition_mode   = TILE_HW
```

### TILE_DESC[0] (Layer 0, tile h_out=0, Wblk=0..19, Cin=0..2, Cout=0..15):

```
h_out_start = 0, h_out_end = 0
w_blk_start = 0, w_blk_end = 19   (20 Wblks × 16lanes = 320 columns)
valid_h = 1, valid_w = 16 (đủ 16, không edge)

cin_start  = 0, cin_len  = 3  (toàn bộ Cin=3)
cout_start = 0, cout_len = 16 (toàn bộ Cout=16, fit 1 lane group)

flags:
  first_tile   = 1  (tile đầu tiên)
  edge_tile_h  = 1  (h=0 là biên trên)
  edge_tile_w  = 0
  hold_skip    = 0
  need_swizzle = 0
  psum_carry_in = 0

last_flags:
  last_cin     = 1  (cin=0..2, toàn bộ Cin)
  last_kernel  = 1  (kernel 3×3 hoàn tất trong 1 compute)
  last_reduce  = 1
  → last_pass  = 1 → PPU kích hoạt

router_profile_id = 0  (Standard Conv)
post_profile_id   = 0  (Conv with SiLU)
```

### POST_PROFILE[0] (Conv Layer 0 với SiLU):

```
bias_en      = 1
bias_offset  = 0x0C00   (offset vào weight SRAM cho B_int32[16])
requant_en   = 1
scale_mul    = [M_int_0, M_int_1, ..., M_int_15]  (per-cout, 16 values)
scale_shift  = 23   (chung cho layer 0, hoặc per-cout array)
zp_out       = 0
act_mode     = 1    (SiLU_LUT)
clamp_min    = -128
clamp_max    = 127
domain_align_en = 0  (không phải CONCAT)
```

---

## 9. Descriptor Sequence cho QConcat (Layer 12)

### LAYER_DESC[12]:

```
layer_idx        = 12
primitive_id     = 5 (CONCAT)
block_type       = 6 (QCONCAT)
num_tiles        = ...

in_H=40, in_W=40
in_Cin = 384   (256 từ F11 + 128 từ F6)
out_H=40, out_W=40, out_Cout=384

kernel_h=0, kernel_w=0, stride=0 (N/A)

num_sources      = 2
source_layer[0]  = 11  (F11, upsample output)
source_layer[1]  = 6   (F6, backbone skip SKIP-B)
hold_output      = 0
hold_until_layer = 255
```

### POST_PROFILE cho QConcat L12 (nếu cần domain align):

```
bias_en      = 0
requant_en   = 0
act_mode     = 0 (none)
domain_align_en = 1
domain_scale_in  = scale_F11    (source scale để align về common)
domain_scale_out = scale_L12_out (common scale)
domain_zp_in     = zp_F11
domain_zp_out    = zp_L12_out
```

---

## 10. Sign-off Checklist

```
NET_DESC:
☐ version=1 hard-coded
☐ num_layers=23 confirmed
☐ Addresses non-overlapping: weight, act, psum, skip regions

LAYER_DESC:
☐ primitive_id lấy đúng từ primitive_matrix.md
☐ source_layer cho 4 QConcat đúng: L12=[11,6], L15=[14,4], L18=[17,13], L21=[20,8]
☐ hold_output=1 và hold_until_layer đúng cho L4, L6, L8, L13

TILE_DESC flags:
☐ first_tile: chỉ tile đầu tiên của layer mỗi layer có = 1
☐ edge_tile_h/w: tất cả biên ảnh có flag = 1
☐ hold_skip: tất cả tile của L4, L6, L8, L13 có hold_skip=1
☐ last_pass = last_cin AND last_kernel AND last_reduce: verified

POST_PROFILE:
☐ scale_mul và scale_shift offline computed và verified
☐ SiLU LUT pre-loaded trước inference
☐ domain_align_en=1 cho CONCAT layers khi scale mismatch
☐ clamp_min=-128, clamp_max=127 hard-coded

ROUTER_PROFILE:
☐ Mọi CONCAT tile có rps_destination=3 khi cần domain align
☐ UPSAMPLE tile có swizzle_mode=1
```

*Descriptor stack là giao diện giữa software compiler và hardware. Thay đổi format phải backward-compatible.*


---
---

<a id='phần-vi6---execution-semantics'></a>

# PHẦN VI.6 — Execution Semantics
> Nguồn: `PHASE0/06_execution_semantics.md`

---

# 06 – Execution Semantics (Freeze Spec)
## qYOLOv10n INT8 Accelerator – Tile/Layer Execution Semantics

> **Trạng thái**: FREEZE | **Version**: v1.0 | **Ngày**: 2026-03-16

---

## 1. Mục Đích

Chốt ý nghĩa chính xác của last_pass, PSUM/ACT namespace, HOLD_SKIP, barrier, và UPSAMPLE/CONCAT/MOVE path. RTL `tile_fsm.sv` và `barrier_manager.sv` phải tuân thủ file này.

---

## 2. Last Pass Semantics

### 2.1. Định Nghĩa

```
last_pass = last_cin AND last_kernel AND last_reduce

last_cin    : Cin chunk hiện tại là chunk cuối cùng để tích lũy PSUM
              → Reduction theo chiều channel hoàn tất
              
last_kernel : Kernel position hiện tại là vị trí cuối cùng trong tích chập
              → Reduction theo kernel hoàn tất (kH×kW done)
              
last_reduce : Đây là operation cuối trong toàn bộ reduction loop
              → Cả Cin và kernel position đều done
```

### 2.2. Flow Điều Khiển

```
if NOT last_pass:
  PSUM_accumulator += mac_result
  output_namespace = PSUM_BUFFER (INT32)
  PPU không kích hoạt
  Không write ra GLB_OUTPUT
  
if last_pass:
  PSUM_accumulator += mac_result    ← lần cộng cuối
  PPU kích hoạt:
    step 1: bias_add: PSUM += B_int32[cout]
    step 2: requant:  y_raw = (PSUM * M_int) >> shift
    step 3: offset:   y_off = y_raw + zp_out
    step 4: act:      y_act = activation_fn(y_off)   (SiLU LUT hoặc identity)
    step 5: clamp:    y_int8 = clamp(y_act, -128, 127)
  Write y_int8 → GLB_OUTPUT (ACT_namespace)
  Reset PSUM_accumulator = 0 cho output position tiếp theo

Lỗi nếu:
  last_pass=1 nhưng PPU chưa load bias/scale params → undefined behavior
  last_pass=0 nhưng PPU kích hoạt → wrong output
```

### 2.3. Ví Dụ: Conv3x3 Cin=64, Cout=16 với Cin_chunk=16

```
Tile 1 (cin=0..15,  last_cin=0): MAC → PSUM, last_pass=0 → accumulate
Tile 2 (cin=16..31, last_cin=0): MAC → PSUM, last_pass=0 → accumulate
Tile 3 (cin=32..47, last_cin=0): MAC → PSUM, last_pass=0 → accumulate
Tile 4 (cin=48..63, last_cin=1): MAC → PSUM, last_pass=1 → PPU → INT8

Nếu kernel cũng tích lũy (ít dùng cho Conv3x3 vì kernel nhỏ):
  last_kernel=1 chỉ sau khi toàn bộ 3×3=9 kernel positions đã xử lý
  Thông thường trong 1 tile: tất cả 9 kernel pos xử lý → last_kernel=1 luôn true
```

### 2.4. DW_7x7_MULTIPASS Last Pass

```
Pass 1 (rows 0-2): last_kernel=0, last_pass=0 → PSUM accumulate
Pass 2 (rows 3-5): last_kernel=0, last_pass=0 → PSUM accumulate
Pass 3 (row  6):   last_kernel=1, last_pass=1 → PPU kích hoạt

Chú ý: DW không có cross-channel Cin → last_cin luôn=1 (per-channel)
Vì last_pass = last_cin AND last_kernel AND last_reduce = 1 AND last_kernel AND 1
→ last_pass = last_kernel

Signal last_pass_kernel trong TILE_DESC bit 3 dùng để distinguish multipass cases.
```

---

## 3. PSUM / ACT Namespace

### 3.1. Phân Vùng Bộ Nhớ

```
┌──────────────────────────────────────────────────────────────┐
│                     GLB Memory Map                           │
│                                                              │
│  0x0000_0000 ┌─────────────────┐                           │
│              │  PSUM Buffer    │  (INT32, tối đa tile size) │
│  0x0002_0000 ├─────────────────┤                           │
│              │  ACT Buffer A   │  (INT8, ping-pong)        │
│  0x0010_0000 ├─────────────────┤                           │
│              │  ACT Buffer B   │  (INT8, ping-pong)        │
│  0x0020_0000 ├─────────────────┤                           │
│              │  HOLD_SKIP      │  (~900KB cho 4 skips)     │
│  0x0030_0000 ├─────────────────┤                           │
│              │  Weight SRAM    │                            │
│  0x0080_0000 └─────────────────┘                           │
└──────────────────────────────────────────────────────────────┘
```

### 3.2. Quy Tắc Truy Cập

```
PSUM_namespace (INT32):
  - Ghi: PE cluster → column_reduce → thẳng vào PSUM_Buffer
  - Đọc: PPU đọc từ PSUM_Buffer tại last_pass
  - Không expose ra ngoài accelerator
  - Giá trị INT32, ký hiệu: PSUM[cout, h_out, w_out]

ACT_namespace (INT8):
  - Ghi: PPU output → GLB_OUTPUT banks
  - Đọc: Layer tiếp theo đọc từ GLB_INPUT banks (ping-pong)
  - Có thể expose ra CPU interface (P3/P4/P5 outputs)
  - Giá trị INT8, ký hiệu: Y_int8[cout, h_out, w_out]

Ping-pong mechanism:
  Layer L viết vào ACT_Buffer_A
  Layer L+1 đọc từ ACT_Buffer_A và viết vào ACT_Buffer_B
  Layer L+2 đọc từ ACT_Buffer_B và viết vào ACT_Buffer_A
  (trừ HOLD_SKIP buffer – không ping-pong)
```

---

## 4. HOLD_SKIP Semantics

### 4.1. Định Nghĩa

HOLD_SKIP: Giữ lại tensor output trong bộ nhớ GLB sau khi layer compute xong, không giải phóng, cho đến khi consumer layer (QConcat) đọc xong.

### 4.2. State Machine

```
States:
  INACTIVE   : Không được sử dụng
  FILLING    : Layer producer đang compute, dần ghi vào HOLD_SKIP region
  READY      : Layer producer done, toàn bộ tensor đã ghi, sẵn sàng cho consumer
  CONSUMING  : Consumer (QConcat) đang đọc từ HOLD_SKIP region
  RELEASED   : Consumer done, vùng nhớ được giải phóng

Transitions:
  INACTIVE  → FILLING   : Khi first_tile của producer layer bắt đầu
  FILLING   → READY     : Khi last_tile của producer layer kết thúc
                          → Signal barrier: F{idx}_hold_ready = 1
  READY     → CONSUMING : Khi barrier release cho consumer layer
  CONSUMING → RELEASED  : Khi consumer layer last_tile done
                          → Giải phóng HOLD_SKIP region cho mục đích khác
```

### 4.3. Mapping Cụ Thể

```
HOLD_SKIP_A (F4, SKIP-A):
  Producer: L4 (QC2f)   → FILLING từ L4_tile0 đến L4_lastTile
  Ready signal: F4_hold_ready ← L4_done
  Consumer: L15 (QConcat source_layer[1])
  Region: skip_buf_base + 0, size=409,600 bytes

HOLD_SKIP_B (F6, SKIP-B):
  Producer: L6 (QC2f)   → FILLING trong L6 compute
  Ready signal: F6_hold_ready ← L6_done
  Consumer: L12 (QConcat source_layer[1])
  Region: skip_buf_base + 409,600, size=204,800 bytes

HOLD_SKIP_C (F8, SKIP-C):
  Producer: L8 (QC2f)   → FILLING trong L8 compute
  Ready signal: F8_hold_ready ← L8_done
  Consumer: L21 (QConcat source_layer[1])
  Region: skip_buf_base + 614,400, size=102,400 bytes

HOLD_SKIP_D (F13, SKIP-D):
  Producer: L13 (QC2f)  → FILLING trong L13 compute
  Ready signal: F13_hold_ready ← L13_done
  Consumer: L18 (QConcat source_layer[1])
  Region: skip_buf_base + 716,800, size=204,800 bytes
  Note: F6 đã được RELEASED tại L12_done trước khi F13 cần space
        → Vẫn dùng region của F6 (reuse) hoặc region riêng
```

---

## 5. Barrier Semantics

### 5.1. Định Nghĩa Barrier

Barrier = Synchronization point: một layer phải đợi tất cả producer dependencies hoàn tất trước khi bắt đầu compute.

### 5.2. Bốn Barrier trong Model (Critical Points)

```
BARRIER_L12:
  Wait condition: (L11_done == TRUE) AND (F6_hold_ready == TRUE)
  Action: release L12_first_tile_start
  Timeout: hardware timer (nếu quá lâu → error interrupt)
  
  Nếu L11 done nhưng F6 chưa READY: L12 phải stall
  Nếu F6 READY nhưng L11 chưa done: L12 phải stall
  Cả hai phải TRUE mới release

BARRIER_L15:
  Wait condition: (L14_done == TRUE) AND (F4_hold_ready == TRUE)
  Action: release L15_first_tile_start
  Note: F4 sinh từ L4, hold từ lúc L4 done, sẵn sàng rất lâu trước L15

BARRIER_L18:
  Wait condition: (L17_done == TRUE) AND (F13_hold_ready == TRUE)
  Action: release L18_first_tile_start

BARRIER_L21:
  Wait condition: (L20_done == TRUE) AND (F8_hold_ready == TRUE)
  Action: release L21_first_tile_start
  Note: F8 sinh SỚMEST nhất (L8), hold rất lâu, nên F8_hold_ready=1 từ rất sớm
```

### 5.3. Barrier Implementation

```
Hardware `barrier_manager.sv` duy trì:
  done_register[23]   : bit array, done_register[i]=1 khi layer i hoàn tất
  hold_ready[4]       : {F4_ready, F6_ready, F8_ready, F13_ready}

Logic:
  L12_start_en = done_register[11] AND hold_ready[F6]  ← BARRIER_L12
  L15_start_en = done_register[14] AND hold_ready[F4]  ← BARRIER_L15
  L18_start_en = done_register[17] AND hold_ready[F13] ← BARRIER_L18
  L21_start_en = done_register[20] AND hold_ready[F8]  ← BARRIER_L21

  done_register[i] ← set by tile_fsm khi last_tile của layer i complete
  hold_ready[Fx]   ← set bởi HOLD_SKIP state machine

desc_fetch_engine không fetch TILE_DESC[0] của QConcat layer
cho đến khi barrier release.
```

---

## 6. UPSAMPLE / CONCAT / MOVE Path

### 6.1. Các Primitive Không Qua PE

```
Primitives không cần PE compute:
  - UPSAMPLE_NEAREST (P6)
  - MOVE (P4)

Primitives cần routing nhưng không cần MAC:
  - CONCAT (P5) – chỉ routing, có thể cần mini-requant
  - MAXPOOL (P3) – cần max-compare tree, không phải MAC
```

### 6.2. UPSAMPLE_NEAREST Path

```
Thực hiện bởi tensor_post_engine (không qua PE cluster):

Input: F_in [C, H, W] được lưu trong GLB_OUTPUT của layer trước
Output: F_out [C, 2H, 2W] phải được ghi vào GLB_INPUT của layer tiếp theo

Mechanism: DMA với address pattern repeat:
  for h in 0..H-1:
    for w in 0..W-1:
      src_addr = compute_input_addr(h, w, ...)
      val = GLB.read(src_addr)
      
      dst_addr_00 = compute_output_addr(2h,   2w,   ...)
      dst_addr_01 = compute_output_addr(2h,   2w+1, ...)
      dst_addr_10 = compute_output_addr(2h+1, 2w,   ...)
      dst_addr_11 = compute_output_addr(2h+1, 2w+1, ...)
      
      GLB.write(dst_addr_00, val)  ← 4 writes cùng value
      GLB.write(dst_addr_01, val)
      GLB.write(dst_addr_10, val)
      GLB.write(dst_addr_11, val)

Scale/ZP: tensor_post_engine chuyển metadata unchanged.
No PPU involved.
```

### 6.3. CONCAT Path

```
Thực hiện bởi router_cluster + optional mini-PPU:

Case 1: Same domain (scale_A == scale_Y, zp_A == zp_Y):
  router_cluster routing:
    for h in 0..H-1:
      for w_blk in 0..Wblk_total-1:
        Read A_channels [Cin_A] từ GLB region A
        Write A_channels to output GLB at [cout=0..Cin_A-1]
        
        Read B_channels [Cin_B] từ HOLD_SKIP region B
        Write B_channels to output GLB at [cout=Cin_A..Cin_A+Cin_B-1]

Case 2: Domain mismatch (cần requant):
  router_cluster với mini-PPU:
    for each channel A:
      val_A = read from GLB region A
      val_A_aligned = mini_ppu_requant(val_A, scale_A, zp_A, scale_Y, zp_Y)
      write val_A_aligned to output GLB
    
    for each channel B:
      val_B = read from HOLD_SKIP region B
      val_B_aligned = mini_ppu_requant(val_B, scale_B, zp_B, scale_Y, zp_Y)
      write val_B_aligned to output GLB

mini_ppu_requant:
  y = clamp(round((val - zp_in) * (scale_in/scale_out)) + zp_out, -128, 127)
  Implemented với fixed-point multiply-shift
```

### 6.4. MOVE Path

```
Simple DMA copy:
  src_region → dst_region
  Metadata (scale, zp) pass-through từ LAYER_DESC
  Không qua PE, PPU, hay router routing logic
```

---

## 7. Tile Execution Order

### 7.1. General Tiling Policy

```
Outer loop: layer (L0 → L22)
  Middle loops: tiling dimensions (tùy partition_mode)
    partition_mode = TILE_HW:    outer=h_tile, inner=w_tile, innermost=cout_chunk
    partition_mode = TILE_COUT:  outer=cout_tile, inner=h_tile, innermost=cin_chunk
    partition_mode = TILE_CIN:   outer=h_tile, inner=cin_chunk, innermost=cout_chunk

  Innermost: PE execution per tile
```

### 7.2. SPPF Tile Order

```
L9 SPPF = OS_1x1 + MaxPool + MaxPool + MaxPool + CONCAT + OS_1x1

Execution:
  Sub-layer 9.1: OS_1x1 (Cin=256→128)  → output X1 [128,20,20]
  Sub-layer 9.2: MaxPool on X1          → P1 [128,20,20]  (hold X1, P1)
  Sub-layer 9.3: MaxPool on P1          → P2 [128,20,20]  (hold X1,P1,P2)
  Sub-layer 9.4: MaxPool on P2          → P3 [128,20,20]  (hold X1,P1,P2,P3)
  Sub-layer 9.5: CONCAT(X1,P1,P2,P3)   → Ycat [512,20,20] (release holds)
  Sub-layer 9.6: OS_1x1 (Cin=512→256)  → F9 [256,20,20]

Note: X1, P1, P2 cần được buffer đồng thời → total buffer cho SPPF tại 9.4:
  3 × 128 × 20 × 20 = 153,600 bytes ≈ 150KB internal SPPF buffer
```

### 7.3. QC2fCIB Tile Order (DW_7x7_MULTIPASS)

```
L22 QC2fCIB:
  Sub 22.1: OS_1x1(384→256)               → X1 [256,20,20]
  Sub 22.2: DW_7x7_MULTIPASS pass 1 (rows 0-2) → PSUM accumulate
  Sub 22.3: DW_7x7_MULTIPASS pass 2 (rows 3-5) → PSUM accumulate
  Sub 22.4: DW_7x7_MULTIPASS pass 3 (row 6)    → PPU → Y_dw [128,20,20]
  Sub 22.5: OS_1x1(128→128)               → Y_cib [128,20,20]
  Sub 22.6: CONCAT(Y_cib, X1_split)       → Ycat [256,20,20]
  Sub 22.7: OS_1x1(256→256)               → P5 [256,20,20]

Note: X1 một nửa channels (128) được giữ làm skip cho CONCAT trong Sub 22.6
```

---

## 8. Error và Exception Semantics

```
BARRIER TIMEOUT:
  Nếu barrier condition không được thỏa mãn trong N cycles:
  → Set error_interrupt
  → Halt execution
  → Report: which barrier, which dependency missing

PSUM OVERFLOW:
  PSUM INT32 cộng tất cả Cin×kH×kW terms
  INT32 có thể overflow nếu:
    Cin=256, kH=kW=3: 256×9=2304 terms × max(INT8×INT8)=127×127=16129
    max_psum = 2304 × 16129 = 37,161,216  ← fit INT32 vẫn OK
  Monitor: if psum > 2^31 - 1 → saturation flag (report, không crash)

HOLD_SKIP CONFLICT:
  Nếu producer cố ghi vào HOLD_SKIP region đang CONSUMING:
  → Error: producer layer chưa được cleared
  → Cần re-check layer HOLD_SKIP assignment không bị overlap

WRONG LAST_PASS:
  Nếu last_pass=1 nhưng không phải tile cuối → partial output to ACT
  Nếu last_pass=0 nhưng là tile cuối → PSUM never flushed
  → Cả hai: output sai nhưng không crash → phải catch trong test
```

---

## 9. Sign-off Checklist

```
LAST PASS:
☐ last_pass = last_cin AND last_kernel AND last_reduce: định nghĩa khóa
☐ PPU chỉ kích hoạt khi last_pass=1: verified trong psum_act_model.py
☐ DW_7x7: last_pass=1 chỉ tại pass 3 (row 6): verified
☐ PSUM reset sau mỗi last_pass: verified (không leak sang output tiếp)

HOLD_SKIP:
☐ 4 HOLD_SKIP regions định nghĩa đúng: F4, F6, F8, F13
☐ State machine INACTIVE→FILLING→READY→CONSUMING→RELEASED: correct
☐ Region sizes: F4=409600, F6=204800, F8=102400, F13=204800 bytes
☐ No overlap giữa 4 regions

BARRIER:
☐ 4 barriers: L12, L15, L18, L21 được implement trong barrier_manager
☐ Condition AND semantics: cả hai producer phải done
☐ Stall mechanism khi barrier chưa cleared

UPSAMPLE PATH:
☐ Không qua PE, không qua PPU
☐ scale/zp pass-through metadata
☐ 4 writes per source pixel (×2 repetition each dimension)

CONCAT PATH:
☐ router_cluster handle channel interleaving
☐ mini-PPU kích hoạt khi domain_align_en=1 trong POST_PROFILE
☐ Thứ tự channels: A_channels first, B_channels second

TILE ORDER:
☐ SPPF sub-layer order: OS_1x1 → Pool×3 → CONCAT → OS_1x1
☐ DW_7x7_MULTIPASS order: pass1 → pass2 → pass3(last_pass)
☐ first_tile flag reset PSUM: verified
```

*Execution semantics là "luật" mà tile_fsm.sv và barrier_manager.sv phải implement.*


---
---

<a id='phần-vi7---golden-python-plan'></a>

# PHẦN VI.7 — Golden Python Plan
> Nguồn: `PHASE0/07_golden_python_plan.md`

---

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


---
---

<a id='phần-vi8---rtl-mapping-plan'></a>

# PHẦN VI.8 — RTL Mapping Plan
> Nguồn: `PHASE0/08_rtl_mapping_plan.md`

---

# 08 – RTL Mapping Plan (Freeze Spec)
## qYOLOv10n INT8 Accelerator – Primitive/Layer → RTL Module Mapping

> **Trạng thái**: FREEZE | **Version**: v1.0 | **Ngày**: 2026-03-16

---

## 1. Mục Đích

Chốt đường đi từ primitive/layer xuống RTL module. Sau khi Golden Python pass, file này là bản đồ implement RTL theo thứ tự dependency.

---

## 2. RTL Module Hierarchy

```
accel_top.sv
  ├── desc_fetch_engine.sv        ← Fetch & parse NET/LAYER/TILE desc
  ├── barrier_manager.sv          ← Sync producer→consumer skip
  ├── tile_fsm.sv                 ← Tile loop control
  ├── subcluster_wrapper.sv       ← Block-level composition
  │     ├── pe_cluster.sv         ← Dense conv (RS_DENSE_3x3, OS_1x1)
  │     │     ├── window_gen.sv
  │     │     ├── pe_lane_mac.sv
  │     │     └── column_reduce.sv
  │     ├── pe_cluster_dw.sv      ← Depthwise conv (DW_3x3, DW_7x7)
  │     │     ├── window_gen.sv   (shared)
  │     │     └── pe_lane_mac_dw.sv
  │     ├── pool_engine.sv        ← MaxPool (MAXPOOL_5x5)
  │     │     ├── window_gen.sv   (shared)
  │     │     └── max_compare_tree.sv
  │     ├── router_cluster.sv     ← CONCAT, MOVE, routing
  │     ├── swizzle_engine.sv     ← UPSAMPLE_NEAREST, transpose
  │     └── tensor_post_engine.sv ← Address remapping DMA
  ├── ppu_lite.sv                 ← Bias + Requant + Act + Clamp
  ├── glb_input_bank.sv (×3)
  ├── glb_output_bank.sv (×4)
  ├── glb_weight_bank.sv
  ├── psum_buffer.sv
  ├── addr_gen_input.sv
  ├── addr_gen_weight.sv
  ├── addr_gen_output.sv
  └── row_slot_manager.sv
```

---

## 3. Primitive → RTL Module Mapping

| Primitive | RTL Module Chính | RTL Module Phụ |
|---|---|---|
| P0 RS_DENSE_3x3 | `pe_cluster.sv` | `window_gen`, `pe_lane_mac`, `column_reduce`, `ppu_lite` |
| P1 OS_1x1 | `pe_cluster.sv` (1×1 mode) | `pe_lane_mac`, `ppu_lite` |
| P2 DW_3x3 | `pe_cluster_dw.sv` | `window_gen`, `pe_lane_mac_dw`, `ppu_lite` |
| P3 MAXPOOL_5x5 | `pool_engine.sv` | `window_gen`, `max_compare_tree` |
| P4 MOVE | `tensor_post_engine.sv` | DMA controller |
| P5 CONCAT | `router_cluster.sv` | `ppu_lite` (mini, nếu domain align) |
| P6 UPSAMPLE_NEAREST | `swizzle_engine.sv` | `tensor_post_engine.sv` |
| P7 EWISE_ADD | `router_cluster.sv` | `ppu_lite` (add + requant) |
| P8 DW_7x7_MULTIPASS | `pe_cluster_dw.sv` (multipass mode) | `psum_buffer`, `ppu_lite` |
| P9 GEMM_ATTN_BASIC | `pe_cluster.sv` (GEMM mode) | `ppu_lite`, softmax LUT |

---

## 4. Đặc Tả Chi Tiết Từng RTL Module

### 4.1. window_gen.sv

```
Input:
  glb_input_banks[3]  ← 3-bank GLB input
  tile_params         ← h_out, w_blk, kernel_size, stride, padding

Output:
  window_data[K×K][Cin_chunk][LANES]  ← window data per cycle
  valid                               ← valid window data

Chức năng:
  - Sinh cửa sổ K×K pixel từ GLB theo bank model (h mod 3)
  - Xử lý edge padding (zeros) khi edge_tile flag set
  - Hỗ trợ kernel 1×1, 3×3, 5×5, 7×7 (partial cho multipass)
  - Output: LANES=16 columns đồng thời

Parameters: kernel_size ∈ {1,3,5}, stride ∈ {1,2}
Multipass param: kh_start, kh_end cho từng pass của DW_7x7
```

### 4.2. pe_lane_mac.sv (Dense mode)

```
Input:
  x_lane[LANES][Cin_chunk]   ← INT8 activation, 16 lanes × Cin
  w_col[Cin_chunk][Cout_chunk] ← INT8 weight, Cin × Cout
  psum_in[Cout_chunk][LANES]  ← INT32 partial sum carry-in

Output:
  psum_out[Cout_chunk][LANES]  ← INT32 updated PSUM

Operation:
  for lane in 0..LANES-1:
    for cout in 0..Cout_chunk-1:
      psum_out[cout][lane] = psum_in[cout][lane]
        + Σ_{cin} x_lane[lane][cin] × w_col[cin][cout]
  
  INT8 × INT8 → INT16 per multiply
  Σ Cin terms → INT32 accumulate
  
Clock budget: Cin_chunk term per cycle (systolic approach) OR
              All Cin in parallel (combinational MAC tree per lane)
```

### 4.3. pe_lane_mac_dw.sv (Depthwise mode)

```
Input:
  x_lane[LANES][1]      ← 1 channel per lane (depthwise)
  w_dw[1][1]            ← weight scalar per channel (broadcast)
  psum_dw_in[LANES]

Output:
  psum_dw_out[LANES]

Operation:
  for lane in 0..LANES-1:
    psum_dw_out[lane] = psum_dw_in[lane] + x_lane[lane][0] × w_dw[0][0]

  No cross-channel reduction.
  Per-channel weight loaded from glb_weight_bank with different offsets.
```

### 4.4. column_reduce.sv

```
Input:
  psum_partial[Cout][LANES]  ← partial sum từ current kernel position
  psum_acc[Cout][LANES]      ← accumulated PSUM từ previous Cin chunks

Output:
  psum_acc_new[Cout][LANES]  ← updated PSUM

Operation:
  psum_acc_new = psum_acc + psum_partial
  (Cin dimension reduce: cộng dồn qua các Cin chunks)

Triggered only when last_cin=0 or intermediate passes.
```

### 4.5. ppu_lite.sv

```
Input:
  psum[Cout][LANES]  ← INT32 sau last_pass
  bias[Cout]         ← INT32 bias values (từ POST_PROFILE)
  M_int[Cout]        ← INT32 fixed-point multiplier
  shift              ← Shift amount
  zp_out             ← Output zero point
  act_mode           ← 0=none, 1=SiLU_LUT
  clamp_min/max      ← -128/127

Output:
  y_int8[Cout][LANES]  ← INT8 final output

Pipeline stages:
  Stage 1: bias_add   → acc_biased = psum + bias[cout]  (INT32)
  Stage 2: multiply   → y_scaled = acc_biased × M_int[cout]  (INT64 intermediate)
  Stage 3: shift      → y_shifted = y_scaled >> shift  (INT32)
  Stage 4: offset     → y_off = y_shifted + zp_out  (INT32)
  Stage 5: act        → y_act = SiLU_LUT[y_off] if act_mode=1 else y_off
  Stage 6: clamp      → y_int8 = clamp(y_act, -128, 127)  (INT8)

SiLU LUT:
  - 256-entry ROM (or BRAM)
  - LUT[i] = SiLU value at index i (precomputed from software)
  - Address: y_off + 128 (shift to [0,255])
```

### 4.6. pool_engine.sv (MAXPOOL_5x5)

```
Input:
  window[5×5][LANES]  ← 25 INT8 values per lane từ window_gen_5x5

Output:
  max_val[LANES]  ← INT8 max per lane

Operation:
  max_val[lane] = max(window[0..24][lane])  ← 25-way INT8 max
  
Implementation: binary tree (5 levels), pipelined
  Level 0: 25 → 13 (one input passed through)
  Level 1: 13 → 7
  Level 2: 7  → 4
  Level 3: 4  → 2
  Level 4: 2  → 1 = max

No PPU, no requant. scale/zp metadata pass-through.
```

### 4.7. router_cluster.sv

```
Chức năng:
  1. Route GLB_INPUT → PE (standard conv path)
  2. Route HOLD_SKIP + GLB_INPUT → CONCAT output (QConcat path)
  3. Route GLB_OUTPUT → next layer GLB_INPUT (ping-pong)

For CONCAT mode:
  - Read A_channels từ GLB region A
  - Read B_channels từ HOLD_SKIP region
  - If domain_align_en: pass B through mini_ppu_requant
  - Write [A_channels, B_channels] interleaved to GLB_OUTPUT

mini_ppu_requant (inline in router):
  y = clamp(round((x-zp_in) * (scale_in/scale_out)) + zp_out, -128, 127)
  Implementation: fixed-point multiply-shift, same as ppu_lite but simpler
```

### 4.8. swizzle_engine.sv (UPSAMPLE_NEAREST)

```
Input:
  src_tensor[C][H][W]  ← INT8 từ GLB
  upsample_scale = 2

Output:
  dst_tensor[C][2H][2W]  ← Written to GLB_INPUT for next layer

Address pattern:
  for h in 0..H-1, w in 0..W-1:
    src_addr = addr_gen_input(h, w, ...)
    val = GLB.read(src_addr)
    GLB.write(addr_gen_output(2h,   2w,   ...), val)
    GLB.write(addr_gen_output(2h,   2w+1, ...), val)
    GLB.write(addr_gen_output(2h+1, 2w,   ...), val)
    GLB.write(addr_gen_output(2h+1, 2w+1, ...), val)

No arithmetic, no PPU. scale/zp unchanged.
```

### 4.9. addr_gen_input.sv

```
Input: h, x, cin, layer_params (stride, Q_in, Wblk_total, Cin)
Output: (bank_id[2], offset[32])

Logic:
  bank_id = h[1:0] % 3     (2-bit modulo 3)
  slot    = (h >> 2) % Q_in
  lane    = x[3:0]          (x mod 16)
  Wblk    = x >> 4          (x div 16)
  offset  = slot*(Wblk_total*Cin*16) + Wblk*(Cin*16) + cin*16 + lane
```

### 4.10. row_slot_manager.sv

```
Tham số:  K_eff, stride
Computed: Q_in = ceil((K_eff + 3*stride) / 3)

State:
  slot_reg[3]  ← current slot for each bank (3-entry array)
  
Operations:
  advance_slot(bank):       slot_reg[bank] = (slot_reg[bank] + 1) % Q_in
  get_slot(bank):           return slot_reg[bank]
  reset_all():              slot_reg = [0,0,0]
```

### 4.11. barrier_manager.sv

```
Registers:
  done_reg[23]      ← done_reg[i]=1 when layer i finishes
  hold_ready_reg[4] ← {F4_ready, F6_ready, F8_ready, F13_ready}

Outputs combinational:
  L12_start_en = done_reg[11] & hold_ready_reg[F6]
  L15_start_en = done_reg[14] & hold_ready_reg[F4]
  L18_start_en = done_reg[17] & hold_ready_reg[F13]
  L21_start_en = done_reg[20] & hold_ready_reg[F8]

Update rules:
  done_reg[i] ← set by tile_fsm when layer i last_tile completes
  hold_ready_reg[Fx] ← set when Fx producer layer last_tile completes
  hold_ready_reg[Fx] ← clear when Fx consumer layer starts

Timeout:
  Timeout counter per barrier: if stalled > N cycles → error_interrupt
```

### 4.12. tile_fsm.sv

```
State machine điều khiển tiling loop:

States:
  IDLE → FETCH_LAYER_DESC → FETCH_TILE_DESC → CHECK_BARRIER
       → EXECUTE_TILE → WAIT_TILE_DONE → NEXT_TILE → ...
       → LAYER_DONE → NEXT_LAYER → ... → INFERENCE_DONE

Key actions:
  FETCH_LAYER_DESC: desc_fetch_engine nhận LAYER_DESC[layer_idx]
  FETCH_TILE_DESC:  nhận TILE_DESC[tile_idx]
  CHECK_BARRIER:    nếu layer là QConcat → đợi barrier release
  EXECUTE_TILE:     dispatch tile sang pe_cluster/pool/router/swizzle
  LAYER_DONE:       set done_reg[layer_idx], update hold states
  NEXT_LAYER:       layer_idx++, tile_idx=0
```

---

## 5. Block → RTL Composition

| Block | RTL Composition |
|---|---|
| Conv (L0,1,3,17) | pe_cluster + addr_gen + ppu_lite |
| QC2f (L2,4...) | OS_1x1+RS_DENSE_3x3 sequences via pe_cluster, router concat nội bộ |
| SCDown (L5,7,20) | 2× pe_cluster_dw sequences + router_cluster CONCAT |
| SPPF (L9) | pe_cluster (OS_1x1) + pool_engine×3 + router_cluster (CONCAT) + pe_cluster |
| QPSA (L10) | pe_cluster (OS_1x1 Q/K/V proj) + pe_cluster (GEMM) + ppu_lite |
| Upsample (L11,14) | swizzle_engine |
| QConcat (L12,15,18,21) | router_cluster + barrier_manager + optional mini-PPU |
| QC2fCIB (L22) | pe_cluster + pe_cluster_dw (multipass) + ppu_lite + router_cluster |

---

## 6. RTL Implementation Order (Dependency)

```
Level 0 – Packages (no circuit):
  accel_pkg.sv      ← primitive IDs, constants, type definitions
  desc_pkg.sv       ← descriptor structs
  csr_pkg.sv        ← CSR register map

Level 1 – Memory primitives:
  glb_input_bank.sv
  glb_output_bank.sv
  glb_weight_bank.sv
  psum_buffer.sv

Level 2 – Address generation:
  addr_gen_input.sv
  addr_gen_weight.sv
  addr_gen_output.sv
  row_slot_manager.sv

Level 3 – Compute atoms:
  window_gen.sv       ← depends on addr_gen_input, glb_input_bank
  pe_lane_mac.sv      ← depends on nothing (pure combinational)
  pe_lane_mac_dw.sv
  column_reduce.sv
  max_compare_tree.sv

Level 4 – Post-processing:
  ppu_lite.sv         ← SiLU LUT, requant, clamp

Level 5 – Engines:
  pe_cluster.sv       ← window_gen + pe_lane_mac + column_reduce
  pe_cluster_dw.sv    ← window_gen + pe_lane_mac_dw (+ psum_buffer cho multipass)
  pool_engine.sv      ← window_gen + max_compare_tree
  tensor_post_engine.sv ← DMA + addr remapping

Level 6 – Data movement:
  router_cluster.sv   ← depends on glb banks, ppu_lite (mini)
  swizzle_engine.sv   ← depends on tensor_post_engine

Level 7 – Control:
  desc_fetch_engine.sv ← depends on desc_pkg, glb_weight_bank
  barrier_manager.sv   ← simple registers + combinational logic
  tile_fsm.sv          ← depends on all Level 6 engines

Level 8 – Top:
  subcluster_wrapper.sv ← composes Level 5+6 engines
  accel_top.sv          ← all modules + DMA + AXI interface
```

---

## 7. Verification Strategy

### RTL vs Golden Python

```
For each RTL module at Level 3+:
  1. Generate test vectors from Golden Python:
     python: input=test_vector, expected=golden_output
  2. Write SystemVerilog testbench:
     tb_pe_cluster.sv: drive same test_vector, capture output
  3. Simulate (VCS/Icarus): compare output vs expected
  4. Require: bit-exact match for all 100 random vectors

For top-level integration:
  tb_accel_top.sv: drive full model_forward
  Compare P3/P4/P5 output vs oracle_P3.npy, oracle_P4.npy, oracle_P5.npy
  Require: bit-exact match
```

### Golden Vector Format

```python
# Generate từ Python golden
test_vec = {
    "input": X_int8.tolist(),
    "scale_in": float(scale_in),
    "zp_in": int(zp_in),
    "weight": W_int8.tolist(),
    "bias": B_int32.tolist(),
    "expected_output": Y_int8.tolist(),
    "scale_out": float(scale_out),
    "zp_out": int(zp_out),
}
json.dump(test_vec, open("test_vec_layer0.json", "w"))

# Đọc từ SV testbench qua $readmemh hoặc DPI-C
```

---

## 8. RTL Packages Spec

### accel_pkg.sv (phải viết TRƯỚC tất cả RTL module)

```systemverilog
package accel_pkg;
  // Architecture constants
  parameter int LANES       = 16;
  parameter int INPUT_BANKS = 3;
  parameter int OUTPUT_BANKS = 4;
  parameter int PSUM_W      = 32;   // PSUM width bits
  parameter int ACT_W       = 8;    // Activation width bits
  parameter int WEIGHT_W    = 8;    // Weight width bits
  
  // Primitive IDs
  typedef enum logic [3:0] {
    P_RS_DENSE_3x3    = 4'd0,
    P_OS_1x1          = 4'd1,
    P_DW_3x3          = 4'd2,
    P_MAXPOOL_5x5     = 4'd3,
    P_MOVE            = 4'd4,
    P_CONCAT          = 4'd5,
    P_UPSAMPLE_NEAREST = 4'd6,
    P_EWISE_ADD       = 4'd7,
    P_DW_7x7_MULTIPASS = 4'd8,
    P_GEMM_ATTN_BASIC = 4'd9
  } primitive_id_t;

  // Activation mode
  typedef enum logic [1:0] {
    ACT_NONE = 2'd0,
    ACT_SILU = 2'd1,
    ACT_RELU = 2'd2
  } act_mode_t;

  // Partition mode
  typedef enum logic [1:0] {
    PART_HW   = 2'd0,
    PART_COUT = 2'd1,
    PART_CIN  = 2'd2
  } partition_mode_t;

  // Tile flags
  typedef struct packed {
    logic first_tile;
    logic edge_tile_h;
    logic edge_tile_w;
    logic hold_skip;
    logic need_swizzle;
    logic psum_carry_in;
    logic [9:0] reserved;
  } tile_flags_t;

  // Last flags
  typedef struct packed {
    logic last_cin;
    logic last_kernel;
    logic last_reduce;
    logic last_pass_kernel;
    logic [11:0] reserved;
  } last_flags_t;

endpackage
```

---

## 9. Sign-off Checklist

```
PACKAGES:
☐ accel_pkg.sv: primitive_id_t, act_mode_t, tile_flags_t, last_flags_t defined
☐ desc_pkg.sv: NET_DESC, LAYER_DESC, TILE_DESC, ROUTER_PROFILE, POST_PROFILE structs
☐ csr_pkg.sv: CSR map defined

RTL MODULES (theo Level):
Level 1:
☐ glb_input_bank.sv: 3-bank, h%3 addressing
☐ glb_output_bank.sv: 4-bank, out_row%4 addressing
☐ psum_buffer.sv: INT32 store/load per tile

Level 2:
☐ addr_gen_input.sv: bank+slot+Wblk+lane formula correct
☐ row_slot_manager.sv: Q_in compute correct

Level 3:
☐ window_gen.sv: 1×1/3×3/5×5/7×7 all tested
☐ pe_lane_mac.sv: INT8×INT8→INT32, 16 lanes parallel
☐ pe_lane_mac_dw.sv: per-channel independent

Level 4:
☐ ppu_lite.sv: bias→multiply→shift→act_lut→clamp pipeline correct
☐ SiLU LUT loaded correctly (256 entries)

Level 5+:
☐ pe_cluster.sv: golden vector match (conv3x3 s1, s2, conv1x1)
☐ pe_cluster_dw.sv: dw3x3 and dw7x7 multipass match
☐ pool_engine.sv: maxpool5x5 result match
☐ router_cluster.sv: concat output bit-exact with golden

Level 7:
☐ barrier_manager.sv: 4 barriers correct
☐ tile_fsm.sv: fsm steps correct

Integration:
☐ accel_top.sv: P3/P4/P5 bit-exact với oracle_P3/P4/P5.npy
```

*Sau khi 8 file Phase 0 được sign-off, bắt đầu viết accel_pkg.sv → Level 1 → ... theo thứ tự.*


---
---

# ════════════════════════════════════════════════════════════════
# PHẦN VII — CHIẾN LƯỢC XÂY DỰNG & NGHIÊN CỨU BỔ SUNG
# ════════════════════════════════════════════════════════════════

<a id='phần-vii1---build-strategy'></a>

# PHẦN VII.1 — Build Strategy (Phase 3)
> Nguồn: `PHASE_3/BUILD_STRATEGY.md`

---

# PHASE 3 — RTL Build & Verification Strategy
## YOLOv10n INT8 Accelerator V2 (LANES=32, Dual-RUNNING, 3,072 MACs)

> **Nguyên tắc**: Bottom-Up Build → Unit Test → Integration Test → System Test
> **Mỗi block PHẢI pass test trước khi tích hợp vào block cha**

---

## 1. TỔNG QUAN CHIẾN THUẬT

### 1.1. Dependency Graph (Build Order)

```
Level 0: PACKAGES (accel_pkg → desc_pkg → csr_pkg)
    │
    ├──────────────────────────────────────────────────────┐
    │                                                      │
Level 1: COMPUTE LEAF                              Level 1: MEMORY LEAF
    ├── dsp_pair_int8  ★FIRST                          ├── glb_input_bank
    ├── pe_unit (uses dsp_pair)                        ├── glb_weight_bank
    ├── column_reduce                                  ├── glb_output_bank
    ├── comparator_tree                                ├── metadata_ram
    └── silu_lut                                       ├── addr_gen_input
         │                                             ├── addr_gen_weight
         │                                             └── addr_gen_output
         │                                                  │
Level 2: PPU                    Level 2: DATA MOVEMENT      │
    └── ppu (uses silu_lut)         ├── window_gen          │
         │                          ├── router_cluster      │
         │                          └── swizzle_engine      │
         │                               │                  │
         └──────────────┬────────────────┘──────────────────┘
                        │
Level 3: INTEGRATION
    ├── pe_cluster (pe_unit × 12 + column_reduce + comparator_tree)
    ├── shadow_reg_file
    └── subcluster_wrapper (ALL above + tile_fsm)
                        │
Level 4: CONTROL
    ├── tile_fsm
    ├── barrier_manager
    ├── local_arbiter
    ├── desc_fetch_engine
    └── global_scheduler
                        │
Level 5: SYSTEM
    ├── supercluster_wrapper (4 × subcluster + arbiter)
    ├── tensor_dma
    ├── controller_system
    └── accel_top
                        │
Level 6: END-TO-END
    └── Full L0→L22 Inference Test
```

### 1.2. Nguyên tắc xây dựng mỗi Block

```
┌─────────────────────────────────────────────────────────────────────┐
│  BLOCK BUILD PROTOCOL (áp dụng cho MỌI module)                     │
│                                                                     │
│  Step 1: DEFINE I/O CONTRACT                                        │
│    - Input: data types, ranges, timing (valid/ready protocol)       │
│    - Output: data types, ranges, latency (cycles from input→output) │
│    - Config: parameters, modes supported                            │
│                                                                     │
│  Step 2: WRITE RTL                                                  │
│    - Implement logic theo spec                                      │
│    - Parameterize mọi thứ (LANES, DEPTH, WIDTH...)                 │
│    - Thêm assertions (SVA) cho critical invariants                 │
│                                                                     │
│  Step 3: WRITE TESTBENCH                                           │
│    - Self-checking testbench (tự so sánh output với golden)        │
│    - Edge cases: min/max values, boundary conditions               │
│    - Random stress test: N random vectors                          │
│    - Mode coverage: test ALL supported modes                       │
│                                                                     │
│  Step 4: SIMULATE & VERIFY                                         │
│    - 0 mismatches required → PASS                                  │
│    - Waveform check cho timing/pipeline behavior                   │
│    - Coverage check: tất cả modes/paths đã test                    │
│                                                                     │
│  Step 5: SIGN-OFF                                                  │
│    - Ghi nhận: module_name PASS/FAIL, date, notes                  │
│    - Nếu FAIL: fix → re-test → loop until PASS                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. STEP-BY-STEP CHI TIẾT

---

### STEP 0: Packages (Nền tảng)

**Files**: `accel_pkg.sv`, `desc_pkg.sv`, `csr_pkg.sv`

| Aspect       | Detail                                             |
|--------------|-----------------------------------------------------|
| **Mục đích** | Định nghĩa constants, types, structs dùng chung    |
| **Test**     | Compile-only (no simulation needed)                 |
| **Pass khi** | Tất cả modules import được, không lỗi compile       |

```
Compile command:
  vlog packages/accel_pkg.sv packages/desc_pkg.sv packages/csr_pkg.sv
  → 0 errors, 0 warnings = PASS
```

---

### STEP 1: Compute Leaf — Block đầu tiên và quan trọng nhất

#### 1A. `dsp_pair_int8` — ★ XÂY DỰNG ĐẦU TIÊN

**Lý do xây trước**: Đây là primitive nhỏ nhất, nền tảng của toàn bộ compute.
Nếu sai ở đây → toàn bộ hệ thống sai.

```
┌─────────────────────────────────────────────────────────┐
│ dsp_pair_int8 — I/O Contract                            │
│                                                         │
│ INPUT:                                                  │
│   clk, rst_n, en, clear                                 │
│   x_a  : signed [7:0]  — activation lane 2i             │
│   x_b  : signed [7:0]  — activation lane 2i+1           │
│   w    : signed [7:0]  — shared weight                  │
│                                                         │
│ PROCESSING:                                             │
│   Pipeline 4 stages:                                    │
│   S1: unsigned convert (x+128, w+128)                   │
│   S2: DSP48E1 multiply (pack 2 acts into A port)        │
│   S3: extract products + correction                     │
│   S4: accumulate (psum += product, or psum = product)   │
│                                                         │
│ OUTPUT:                                                 │
│   psum_a : signed [31:0] — accumulated MAC lane 2i      │
│   psum_b : signed [31:0] — accumulated MAC lane 2i+1    │
│   Latency: 4 cycles (pipeline)                          │
│                                                         │
│ CRITICAL INVARIANT:                                     │
│   psum_a == Σ(x_a[t] × w[t]) for all enabled cycles    │
│   psum_b == Σ(x_b[t] × w[t]) for all enabled cycles    │
│   Error tolerance: 0 (EXACT integer arithmetic)         │
└─────────────────────────────────────────────────────────┘

TEST PLAN:
  ① Exhaustive: all 256×256 = 65,536 (x,w) pairs → verify product
  ② Corner: (-128×-128), (-128×127), (127×127), (0×anything)
  ③ Accumulation: 9 MACs (typical conv3×3) → verify sum
  ④ Clear: mid-accumulation clear → psum resets correctly
  ⑤ Enable: en=0 → psum holds (no change)

PASS CRITERIA: 0 mismatches out of 65,536+ test vectors
```

#### 1B. `pe_unit` — Single PE (32 lanes)

```
┌─────────────────────────────────────────────────────────┐
│ pe_unit — I/O Contract                                  │
│                                                         │
│ INPUT:                                                  │
│   x_in[32]  : signed [7:0]  — 32 activations           │
│   w_in[32]  : signed [7:0]  — 32 weights               │
│   en, clear_psum, mode                                  │
│                                                         │
│ PROCESSING:                                             │
│   Instantiate 16 × dsp_pair_int8                        │
│   Lane 2i → dsp_pair[i].x_a                            │
│   Lane 2i+1 → dsp_pair[i].x_b                          │
│   Weight → dsp_pair[i].w (per-lane or broadcast)        │
│                                                         │
│ OUTPUT:                                                 │
│   psum_out[32] : signed [31:0]                          │
│   psum_valid                                            │
│   Latency: 4 cycles (inherited from dsp_pair)           │
│                                                         │
│ THỎA MÃN KHI NHÚNG VÀO PE_CLUSTER:                    │
│   psum_out[lane] == Σ(x_in[lane][t] × w_in[lane][t])   │
│   cho tất cả 32 lanes INDEPENDENT                       │
└─────────────────────────────────────────────────────────┘

TEST PLAN:
  ① 32-lane parallel MAC: random vectors × 10 cycles → verify all lanes
  ② Mode RS3: shared weight per DSP pair (lanes 2i/2i+1)
  ③ Mode OS1: broadcast weight (1×1 conv)
  ④ Clear/Enable control: verify independence
```

#### 1C. `column_reduce` — Cross-Row Reduction

```
┌─────────────────────────────────────────────────────────┐
│ column_reduce — I/O Contract                            │
│                                                         │
│ INPUT:                                                  │
│   pe_psum[3][4][32] : signed [31:0]                     │
│   (3 rows × 4 cols × 32 lanes)                         │
│                                                         │
│ PROCESSING:                                             │
│   For each (col, lane):                                 │
│     col_psum[col][lane] = Σ_{row=0}^{2} pe_psum[row]   │
│                                                         │
│ OUTPUT:                                                 │
│   col_psum[4][32] : signed [31:0]                       │
│   Latency: 1 cycle (combinational + register)           │
│                                                         │
│ THỎA MÃN:                                              │
│   col_psum[c][l] == pe_psum[0][c][l]                   │
│                    + pe_psum[1][c][l]                    │
│                    + pe_psum[2][c][l]                    │
└─────────────────────────────────────────────────────────┘

TEST: Random 3×4×32 INT32 values → verify sums. Edge: max INT32 overflow check.
```

#### 1D. `comparator_tree` — MAXPOOL

```
┌─────────────────────────────────────────────────────────┐
│ comparator_tree — I/O Contract                          │
│                                                         │
│ INPUT:  data_in[25][32] : signed [7:0] (5×5 × 32 lanes)│
│ OUTPUT: max_out[32]     : signed [7:0] (max per lane)   │
│ Latency: 5 cycles (pipelined tree)                      │
│                                                         │
│ THỎA MÃN:                                              │
│   max_out[l] == max(data_in[0..24][l])                  │
│   Per-lane INDEPENDENT (no cross-lane)                  │
└─────────────────────────────────────────────────────────┘

TEST:
  ① Known max at different positions (first, last, middle)
  ② All same values → output = that value
  ③ Signed edge: mix of -128 and 127
```

#### 1E. `silu_lut` — SiLU Lookup

```
┌─────────────────────────────────────────────────────────┐
│ silu_lut — I/O Contract                                 │
│                                                         │
│ INPUT (load): load_en, load_addr[8], load_data[8]       │
│ INPUT (read): idx[32] : [7:0] (32 parallel lookups)     │
│ OUTPUT:       out[32] : signed [7:0]                    │
│                                                         │
│ THỎA MÃN:                                              │
│   out[lane] == ROM[idx[lane]] after preload             │
│   32 simultaneous reads in 1 cycle                      │
└─────────────────────────────────────────────────────────┘

TEST:
  ① Preload 256 entries → read back each → verify
  ② 32 simultaneous reads with different indices
  ③ Boundary: idx=0 and idx=255
```

---

### STEP 2: PPU — Post-Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ ppu — I/O Contract                                                  │
│                                                                     │
│ INPUT:                                                              │
│   psum_in[32] : signed [31:0]  — raw accumulation from PE cluster  │
│   bias_val[32]: signed [31:0]  — per-channel bias                  │
│   m_int[32]   : signed [31:0]  — requant multiplier                │
│   shift[32]   : [5:0]          — requant shift amount              │
│   zp_out      : signed [7:0]   — output zero-point                 │
│   silu_lut[256]                 — preloaded LUT                    │
│   ewise_in[32]: signed [7:0]   — skip connection (optional)        │
│   cfg_post    : post_profile_t — mode config                       │
│                                                                     │
│ PROCESSING (4-stage pipeline):                                      │
│   Stage 1: biased  = psum_in + bias_val                            │
│   Stage 2: y_raw   = (biased × m_int) >>> shift (with rounding)    │
│   Stage 3: y_act   = SiLU(y_raw) or ReLU(y_raw) or y_raw          │
│   Stage 4: act_out = clamp(y_act + ewise + zp_out, -128, 127)     │
│                                                                     │
│ OUTPUT:                                                             │
│   act_out[32] : signed [7:0]  — final INT8 activation              │
│   Latency: 4 cycles                                                │
│                                                                     │
│ THỎA MÃN KHI NHÚNG VÀO SUBCLUSTER:                                │
│   act_out[ch] == clamp(SiLU(requant(psum[ch]+bias[ch])) + ewise, INT8)│
│   Must match Golden Python per-channel exactly                      │
│                                                                     │
│ CRITICAL:                                                           │
│   - Requant rounding: half_up (add 1<<(shift-1) before >>)         │
│   - SiLU index: clamp(y_raw + 128, 0, 255)                        │
│   - Final clamp: saturating arithmetic [-128, 127]                 │
└─────────────────────────────────────────────────────────────────────┘

TEST PLAN:
  ① Per-channel requant: known psum + known M/shift → verify INT8 output
  ② SiLU activation: verify LUT indexing matches Python precomputed
  ③ ReLU activation: negative→0, positive→pass-through
  ④ Element-wise add: with saturation at boundaries
  ⑤ Full pipeline: bias+requant+SiLU+ewise → compare Golden Python dump
```

---

### STEP 3: Memory Modules

#### Tất cả 7 modules cùng Level — Build song song

```
Module              | Input (Write)              | Output (Read)           | Key Invariant
--------------------|----------------------------|-------------------------|---------------------------
glb_input_bank      | 256b (32×INT8) + mask      | 256b (32×INT8)          | wr→rd same addr = same data
glb_weight_bank     | 256b + FIFO push           | 256b + FIFO pop         | FIFO FWFT ordering preserved
glb_output_bank     | 1024b PSUM or 256b ACT     | 1024b or 256b           | Namespace switch correct
metadata_ram        | set_valid + meta           | query_valid + meta      | Ring buffer no overflow
addr_gen_input      | (h,w,c) logical            | (bank_id, addr, pad?)   | h%3→bank, padding=zp_x
addr_gen_weight     | (kr, cin, cout) + mode     | (bank_id, addr)         | Per-mode address unique
addr_gen_output     | (h_out, w_out, cout)       | (bank_id, addr)         | bank = pe_col index
```

**CRITICAL TEST cho addr_gen_input**:
```
PHẢI kiểm tra:
  ① padding positions output zp_x (KHÔNG PHẢI 0!)
  ② Không có address collision: mọi (h,w,c) unique → unique physical addr
  ③ bank_id = h mod 3 (banking rule)
  ④ Stride support: stride=2 → skip alternate positions
```

---

### STEP 4: Data Movement

```
Module              | Input                      | Output                  | Key Invariant
--------------------|----------------------------|-------------------------|---------------------------
window_gen          | 32-wide INT8 stream        | K taps × 32-wide       | Shift register correct
router_cluster      | bank reads + profile       | PE act/wgt + bank write | Mode-dependent routing
swizzle_engine      | bank_output reads          | bank_input writes       | Upsample: 4 dst per src
```

**window_gen test quan trọng**:
```
Feed sequence: row0, row1, row2, row3...
K=3: output taps = [row_n, row_n-1, row_n-2]  (3 consecutive rows)
K=1: output taps = [row_n]                     (pass-through)
K=7: output taps = [row_n ... row_n-6]         (7 rows, DW7x7)

Verify: taps_valid asserted only after K rows accumulated
```

---

### STEP 5: Integration — PE Cluster

```
┌─────────────────────────────────────────────────────────────────────┐
│ pe_cluster — Integration I/O Contract                               │
│                                                                     │
│ INPUT:                                                              │
│   act_taps[3][32] : from window_gen → 3 PE rows                    │
│   wgt_data[3][32] : from router → 3 reduction lanes                │
│   psum_in[4][32]  : from bank_output (multi-pass accumulation)     │
│   mode            : PE_RS3/OS1/DW3/DW7/MP5/GEMM                   │
│                                                                     │
│ OUTPUT:                                                             │
│   psum_out[4][32] : 4 PE cols × 32 lanes → to bank_output/PPU     │
│   pool_out[32]    : MAXPOOL result (if mode=MP5)                   │
│                                                                     │
│ THỎA MÃN KHI NHÚNG VÀO SUBCLUSTER:                                │
│   RS3: psum_out[col][lane] = Σ over 3 kernel rows of (x × w)      │
│   OS1: psum_out[col][lane] = Σ over 3 cin_slices of (x × w)       │
│   DW3: psum_out[col][lane] = Σ over 3 kernel rows (per-channel)   │
│   MP5: pool_out[lane] = max(25 inputs per lane)                    │
└─────────────────────────────────────────────────────────────────────┘

TEST PLAN — Per Mode:
  RS3: Conv3×3 stride=1, Cin=8, Cout=4, H=8, W=32 → Golden Python
  OS1: Conv1×1, Cin=32, Cout=4, H=1, W=32 → Golden Python
  DW3: DWConv3×3, C=32, H=8, W=32 → Golden Python
  MP5: MaxPool5×5, C=32, H=8, W=32 → Golden Python
  Multi-pass: 2 Cin passes → psum accumulates correctly
```

---

### STEP 6: Control — FSM & Scheduling

```
Module              | Input                       | Output                   | Key Test
--------------------|-----------------------------|--------------------------|---------
tile_fsm            | tile_desc + signals          | control signals + state  | FSM transitions
barrier_manager     | signal + wait                | grant + scoreboard       | 4 YOLOv10n barriers
local_arbiter       | sub states + tile queue      | role assignments         | Dual-RUNNING rotation
desc_fetch_engine   | AXI read data                | parsed descriptors       | NET→LAYER→TILE parse
global_scheduler    | tile descs from fetch        | tile dispatch to 4 SCs   | sc_mask routing
```

**tile_fsm — FSM State Transition Test**:
```
IDLE → LOAD_CFG → PREFILL_WT → PREFILL_IN → WAIT_READY → 
RUN_COMPUTE → ACCUMULATE → (loop for multi-pass) →
POST_PROCESS → SWIZZLE_STORE → DONE → IDLE

Must test:
  ① Normal 1-pass tile (RS3, small)
  ② Multi-pass: 3 Cin passes → ACCUMULATE loops 3 times
  ③ DW7 3 K passes → ACCUMULATE loops 3 times
  ④ Barrier before: waits for barrier_grant
  ⑤ Barrier after: signals barrier_signal
  ⑥ NEED_SWIZZLE: waits for swizzle_done
  ⑦ NEED_SPILL: waits for dma_wr_done
```

---

### STEP 7: System Top

```
┌─────────────────────────────────────────────────────────────────────┐
│ accel_top — System I/O Contract                                     │
│                                                                     │
│ INPUT:                                                              │
│   AXI-Lite MMIO: CPU writes CSR (start, net_desc_base, etc.)       │
│   AXI4 Master: DDR3 responses (descriptors, weights, activations)  │
│                                                                     │
│ OUTPUT:                                                             │
│   AXI4 Master: DDR3 writes (output activations P3/P4/P5)           │
│   IRQ: inference complete                                           │
│                                                                     │
│ END-TO-END FLOW:                                                    │
│   1. CPU writes net_desc_base to CSR                                │
│   2. CPU writes start=1                                             │
│   3. Accelerator fetches NET_DESC → LAYER_DESCs → TILE_DESCs       │
│   4. Tiles dispatched to 4 SCs, processed L0→L22                   │
│   5. P3/P4/P5 written to DDR3                                      │
│   6. IRQ asserted → CPU reads P3/P4/P5                             │
│                                                                     │
│ PASS CRITERIA:                                                      │
│   P3[1,64,80,80]  bit-exact match Golden Python                    │
│   P4[1,128,40,40] bit-exact match Golden Python                    │
│   P5[1,256,20,20] bit-exact match Golden Python                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. FILE STRUCTURE

```
PHASE_3/
├── BUILD_STRATEGY.md          ← this file
├── packages/
│   ├── accel_pkg.sv           ← constants, types, enums
│   ├── desc_pkg.sv            ← descriptor structs
│   └── csr_pkg.sv             ← CSR register map
│
├── 01_compute_leaf/
│   ├── rtl/
│   │   ├── dsp_pair_int8.sv   ← 2-MAC DSP48E1 primitive
│   │   ├── pe_unit.sv         ← 32-lane PE (16 DSP pairs)
│   │   ├── column_reduce.sv   ← 3-row reduction
│   │   ├── comparator_tree.sv ← 25-input max (MAXPOOL)
│   │   └── silu_lut.sv        ← 256-entry LUT ROM
│   └── tb/
│       ├── tb_dsp_pair_int8.sv
│       ├── tb_pe_unit.sv
│       ├── tb_column_reduce.sv
│       ├── tb_comparator_tree.sv
│       └── tb_silu_lut.sv
│
├── 02_ppu/
│   ├── rtl/
│   │   └── ppu.sv
│   └── tb/
│       └── tb_ppu.sv
│
├── 03_memory/
│   ├── rtl/
│   │   ├── glb_input_bank.sv
│   │   ├── glb_weight_bank.sv
│   │   ├── glb_output_bank.sv
│   │   ├── metadata_ram.sv
│   │   ├── addr_gen_input.sv
│   │   ├── addr_gen_weight.sv
│   │   └── addr_gen_output.sv
│   └── tb/
│       ├── tb_glb_input_bank.sv
│       ├── tb_addr_gen_input.sv
│       └── ...
│
├── 04_data_movement/
│   ├── rtl/
│   │   ├── window_gen.sv
│   │   ├── router_cluster.sv
│   │   └── swizzle_engine.sv
│   └── tb/
│       └── ...
│
├── 05_integration/
│   ├── rtl/
│   │   ├── pe_cluster.sv
│   │   ├── shadow_reg_file.sv
│   │   └── subcluster_wrapper.sv
│   └── tb/
│       └── tb_pe_cluster.sv
│
├── 06_control/
│   ├── rtl/
│   │   ├── tile_fsm.sv
│   │   ├── barrier_manager.sv
│   │   ├── local_arbiter.sv
│   │   ├── desc_fetch_engine.sv
│   │   └── global_scheduler.sv
│   └── tb/
│       └── ...
│
├── 07_system/
│   ├── rtl/
│   │   ├── supercluster_wrapper.sv
│   │   ├── tensor_dma.sv
│   │   ├── controller_system.sv
│   │   └── accel_top.sv
│   └── tb/
│       └── ...
│
├── 08_e2e/
│   └── tb/
│       └── tb_accel_e2e.sv
│
└── sim_scripts/
    ├── compile_all.do         ← Vivado/ModelSim compile script
    └── run_step1.do           ← Run Step 1 tests
```

---

## 4. VERIFICATION CHECKLIST

```
STEP 0: Packages
  ☐ accel_pkg.sv compiles
  ☐ desc_pkg.sv compiles (imports accel_pkg)
  ☐ csr_pkg.sv compiles

STEP 1: Compute Leaf
  ☐ dsp_pair_int8:  65,536 pairs, 0 mismatch
  ☐ dsp_pair_int8:  accumulation 9 cycles, clear test
  ☐ pe_unit:        32-lane MAC, random 100 vectors
  ☐ column_reduce:  3×4×32 reduction, 1000 random
  ☐ comparator_tree: 25×32 max, position sweep
  ☐ silu_lut:       256-entry load + 32-port read

STEP 2: PPU
  ☐ ppu bias+requant: 100 vectors, half_up rounding
  ☐ ppu SiLU:       LUT index correctness
  ☐ ppu ReLU:       negative clamp
  ☐ ppu ewise_add:  saturation at ±128
  ☐ ppu full pipe:  match Golden Python (10 test cases)

STEP 3: Memory
  ☐ glb_input_bank:  write+read 32 subbanks, lane mask
  ☐ glb_weight_bank: SRAM + FIFO ordering
  ☐ glb_output_bank: PSUM↔ACT mode switch
  ☐ addr_gen_input:  padding=zp_x, no collision, bank=h%3
  ☐ addr_gen_weight: RS3/OS1/DW3/DW7 patterns
  ☐ addr_gen_output: bank=pe_col, slot rotation

STEP 4: Data Movement
  ☐ window_gen:      K1/K3/K5/K7 tap generation
  ☐ router_cluster:  RS3/OS1/DW3/bypass modes
  ☐ swizzle_engine:  upsample 2×, concat offset

STEP 5: Integration
  ☐ pe_cluster RS3:  Conv3×3 8×32 tile = Golden Python
  ☐ pe_cluster OS1:  Conv1×1 32-ch tile
  ☐ pe_cluster DW3:  DWConv3×3 32-ch tile
  ☐ pe_cluster MP5:  MaxPool5×5

STEP 6: Control
  ☐ tile_fsm:        FSM state transitions (all paths)
  ☐ tile_fsm:        multi-pass accumulation
  ☐ barrier_manager: 4 barriers signal/wait
  ☐ local_arbiter:   dual-RUNNING rotation
  ☐ desc_fetch:      parse NET/LAYER/TILE descriptors

STEP 7: System
  ☐ subcluster:      1 tile RS3 end-to-end
  ☐ supercluster:    4 subs, role rotation
  ☐ accel_top:       L0 inference bit-exact

STEP 8: End-to-End
  ☐ L0→L22:          P3/P4/P5 bit-exact Golden Python
  ☐ Barriers:        L4→L15, L6→L12, L8→L21, L13→L18
  ☐ Performance:     cycle count vs theoretical
```

---

## 5. SIMULATION TOOLS

```
Option A: Vivado Simulator (xvlog + xelab + xsim)
  Pro: Free, Xilinx-native, DSP48E1 simulation models
  Con: Slower than commercial

Option B: ModelSim/QuestaSim
  Pro: Fastest simulation, best debugging
  Con: License cost

Option C: Icarus Verilog (iverilog)
  Pro: Free, fast compile
  Con: Limited SystemVerilog support

RECOMMENDED: Vivado Simulator for RTL development
  → Has DSP48E1 behavioral model built-in
  → Directly targets XC7VX690T
```

---

*Mỗi STEP phải PASS 100% trước khi tiến sang STEP tiếp theo.*
*Nếu 1 module FAIL → fix → re-test → PASS → mới move on.*


---
---

<a id='phần-vii2---kiến-trúc--primitive-rtl'></a>

# PHẦN VII.2 — Kiến Trúc & Primitive RTL (Phase 5)
> Nguồn: `PHASE_5/HW_YOLOV10N_RTL_KIEN_TRUC_VA_PRIMITIVE.md`

---

# PHASE 5 — Kiến trúc RTL cho qYOLOv10n INT8 (L0–L22): primitive, đường dữ liệu, controller

**Vai trò tài liệu:** Đóng vai phân tích **thiết kế phần cứng / RTL**, tổng hợp từ các nguồn nghiên cứu trong repo (không thay thế bản gốc — xem mục 9).  
**Phạm vi bộ tăng tốc:** Backbone + Neck **layer 0–22** INT8; **Qv10Detect (L23) + NMS** để CPU/phần mềm theo `MODEL_FORWARD_FLOW.md`.

---

## 1. Mục tiêu thiết kế bạn đặt ra

Bạn muốn **một khối tính toán** có thể:

- Thực thi **các primitive** tương ứng các khối YOLOv10n đã PTQ INT8.
- Chạy **toàn bộ L0–L22** **hiệu quả** chủ yếu bằng **cấu hình đường đi dữ liệu** (descriptor / router / template), không Resynthesize từng layer thành IP riêng.

**Chốt kiến trúc:** Tách rõ ba lớp:

| Lớp | Nội dung |
|-----|----------|
| **A. Điều khiển & metadata** | MMIO, CSR, bảng net/layer/tile trong DDR, fetch, scheduler, barrier. |
| **B. Vận chuyển tensor** | Một (hoặc vài) master AXI, DMA burst, GLB/skip spill, địa chỉ từ `tile_desc`. |
| **C. Primitive compute** | Mảng PE cố định (conv / depthwise / pool / pass-through) + **PPU** (bias, requant INT8, SiLU LUT, e-wise) — **cấu hình bằng `template_id`, `router_profile`, `post_profile` + kích thước từ `layer_desc`**. |

Lớp **C** không “biết” tên layer YOLO; nó chỉ biết **template** (ví dụ RS3, OS1, DW3, DW7, MP5, GEMM, PASS) đã map sẵn từ compiler.

---

## 2. Primitive phần mềm → template phần cứng

Theo `MODEL_BLOCKS_INT8_DETAIL.md` và `MODEL_LAYERS_INT8_FLOW.md`, các **khối cần hỗ trợ** trên đường INT8:

| Primitive (model) | Ý nghĩa tính toán | Gợi ý map RTL (như `accel_pkg::pe_mode_e` + datapath) |
|--------------------|-------------------|--------------------------------------------------------|
| **Conv** (fused BN) | INT8 conv → INT32 MAC → requant + SiLU/ReLU | **OS1** (1×1) / **RS3** (3×3 standard) tùy kernel stride; **PPU** cho act + quant. |
| **QC2f** | Chuỗi Conv 1×1 + bottleneck + **concat** kênh + Conv cuối | **Nhiều tile/phase** hoặc một **macro template**: lặp RS3/OS1, **router** concat; không cần một “opcode QC2f” riêng nếu compiler bung thành subgraph. |
| **SCDown** | Conv s=2 ∥ (MaxPool s=2 → Conv) → **concat** kênh | Hai nhánh conv/pool; có thể **hai pass** hoặc template **DW3** + pool (tùy biến thể đã fuse). |
| **SPPF** | Conv → MaxPool×3 (k=5,s=1) → concat 4 nhánh → Conv | **MP5** + concat + OS1; pool trên INT8 là so sánh, không đổi scale nếu cùng qconfig. |
| **QPSA** | Attention (conv + có thể softmax) | **Khó nhất:** (1) bung **GEMM**/conv INT8 + LUT xấp xỉ; (2) hoặc **tách nhỏ** dequant cục bộ (tốn băng thông/logic); (3) hoặc **offload** một phần — cần khớp **đúng inference** như cảnh báo trong `HW_ARCHITECTURE_V2_100FPS.md`. |
| **Upsample** | Nearest ×2 | **PE_PASS** / **địa chỉ đọc lại** (không MAC); chỉ thay đổi `addr_gen` / swizzle. |
| **QConcat** | Ghép kênh từ 2 tensor | **Không MAC** — **router** + đồng bộ scale (thường cùng qconfig); nếu khác scale cần **requant** trước concat (compiler chèn). |
| **QC2fCIB** | C2f + CIB (dw lớn, multi-pass) | **DW7** / nhiều **pass** (`num_cin_pass`, `num_k_pass` trong descriptor) như đã có trong `tile_fsm` / spec V2. |

**Kết luận RTL:** IP **một** mảng PE + **PPU** + **router/swizzle** đủ nếu **compiler** sinh đúng **chuỗi tile + template + offset DDR** cho từng layer con của QC2f/SCDown/SPPF.

---

## 3. Vì sao chỉ “cấu hình đường đi” vẫn chạy được L0–L22?

1. **Mọi layer** trong trace INT8 đều là **biến đổi tensor** có thể phân rã thành: **đọc activation/weight** → **MAC/pool/concat logic** → **ghi activation** (và giữ **skip** trong GLB hoặc DDR).
2. **`MODEL_LAYER_DEPENDENCIES.md`** chỉ ra:
   - Phần lớn layer **chỉ** `in(Lk) = L(k-1)` → scheduler **tuần tự** đơn giản.
   - **L12, L15, L18, L21** là **QConcat** cần **L6, L4, L13, L8** (v.v.) → phần cứng **bắt buộc** có:
     - **Bộ nhớ on-chip** hoặc **DDR skip arena** giữ feature đến khi concat;
     - **Router** chọn **tensor nguồn** theo `src_in_tid` / bảng trong `layer_desc` (đã có hướng trong `desc_pkg`).
3. **Descriptor-driven:** Mỗi tile mang **`src_in_off`, `src_w_off`, `src_skip_off`, `dst_off`** — đúng với thiết kế `tile_fsm` + `tensor_dma`: **chỉ đổi cấu hình**, không đổi dây DMA.

---

## 4. Phân tách trách nhiệm: quantization & nơi đón dữ liệu

Theo `MODEL_FORWARD_FLOW.md`:

| Bước | Ai làm | Ghi chú RTL |
|------|--------|-------------|
| LetterBox, RGB, normalize float | **CPU** | Không thuộc accelerator. |
| **Quantize đầu vào** (scale_in, zp_in) | **CPU** (hoặc DMA nhỏ) | IP nhận **INT8** đã quant; có thể ghi vào DDR trước khi kick IP. |
| Requant giữa layer | **PPU trong IP** | `m_int`, `shift`, `zp_out`, bias — nên nạp từ **DDR blob** hoặc CSR (PHASE_5 đã chuẩn bị CSR PPU table). |
| SiLU sau conv | **PPU** (LUT) | Khớp `Conv.forward_fuse` patch INT8 → float → SiLU → quant; HW tối ưu = **LUT / cố định điểm**. |

**“Nơi đón dữ liệu” trên IP (tầng vận chuyển):**

1. **MMIO** — không đón tensor; chỉ **CSR** (pointer `net_desc`, layer range, start, irq mask).
2. **AXI master** — đón **descriptor** + **weight** + **activation** + **skip** từ DDR vào **GLB** (hoặc buffer behavioral tạm).
3. **Sau prefill** — dữ liệu “sẵn sàng trước PE” nằm ở **GLB input/weight** (kiến trúc mục tiêu trong `RTL_IP_DIRECTORY_TREE` / PHASE_3).

---

## 5. Controller cần làm gì để “chỉ cấu hình” là chạy L0–L22?

**Controller (như `controller_system` hiện tại):**

- Giữ **CSR**, kick **`desc_fetch_engine`**, điều phối **global_scheduler** tới 4 SC.
- **Không** tính conv; chỉ đảm bảo **tile descriptor** tới đúng SC đúng lúc.

**Compiler / offline tool (ngoài RTL nhưng bắt buộc cho “chỉ config”):**

- Sinh **`net_desc` / `layer_desc` / `tile_desc`** khớp `desc_pkg`.
- Gán **`template_id`** ↔ primitive (RS3, OS1, DW3, …) từng tile.
- Tính **`num_cin_pass` / `num_k_pass`** cho layer nặng (QC2fCIB, QC2f lớn).
- Đặt **`src_skip_off`** / barrier / `tile_flags` cho **QConcat** và **skip retention** (GLB vs spill DDR — tham chiếu Option A/B trong `HW_ARCHITECTURE_V2_100FPS.md`).

Không có compiler, “chỉ đổi config” trên chip **không đủ**.

---

## 6. Kiến trúc phần cứng đề xuất (tổng thể, hiệu quả, không quá phức tạp)

Rút từ `HW_ARCHITECTURE_V2_100FPS.md` + luồng PHASE_3:

```
                    ┌─────────────────────────────────────────┐
  CPU ──AXI-Lite──► │ CSR / Controller / Desc fetch / Sched   │
                    └──────────────┬──────────────────────────┘
                                   │ tile_desc, layer_desc
                    ┌──────────────▼──────────────────────────┐
  DDR ◄════════════►│ 4 × SuperCluster                       │
      AXI4 256b     │   each: FIFO tile + arb + tensor_dma   │
                    │   4 × subcluster: GLB + tile_fsm + …     │
                    └──────────────┬──────────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────────┐
                    │ PE array (template-driven) + PPU + swizzle│
                    └─────────────────────────────────────────┘
```

**Nguyên tắc hiệu quả (V2):**

- **LANES=32**, **EXT 256b**, **2 sub RUNNING / SC** khi compute > fill+drain — tăng throughput mà không nhân đôi số PE theo chiều COL.
- **Một master AXI** mux: descriptor + tensor (và sau này PPU table) — đơn giản hơn nhiều master.
- **Skip lớn (F4 80×80)** có thể **spill DDR**; skip nhỏ giữ **GLB** — trade-off BRAM đã phân tích trong tài liệu V2.

---

## 7. Thứ tự thực thi trên chip (trả lời “truyền xong mới tính?”)

Đúng với **từng tile**:

1. **Cấu hình** (MMIO + descriptor trong DDR).  
2. **Fetch descriptor** (AXI read).  
3. **DMA prefill** weight / input / (skip) vào buffer nội bộ.  
4. **RUN_COMPUTE** / multi-pass trên PE.  
5. **PPU** (requant + act + e-wise nếu có).  
6. **Swizzle + ghi DDR** (output tile).

Giữa các layer L0…L22, **scheduler** xếp tile theo dependency; **QConcat** buộc có **bước chờ** feature cũ đã lưu.

---

## 8. Bảng tham chiếu nhanh L0–L22 (shape & primitive)

Nguồn: `MODEL_LAYERS_INT8_FLOW.md` (đã rút gọn).

| L | Module | Output shape (1,…,H,W) | MAC-heavy? |
|---|--------|------------------------|------------|
| 0 | Conv | 16,320,320 | Có |
| 1 | Conv | 32,160,160 | Có |
| 2 | QC2f | 32,160,160 | Có |
| 3 | Conv | 64,80,80 | Có |
| 4 | QC2f | 64,80,80 | Có — **skip cho L15** |
| 5 | SCDown | 128,40,40 | Có |
| 6 | QC2f | 128,40,40 | Có — **skip cho L12** |
| 7 | SCDown | 256,20,20 | Có |
| 8 | QC2f | 256,20,20 | Có — **skip cho L21** |
| 9 | SPPF | 256,20,20 | Có |
| 10 | QPSA | 256,20,20 | Có (đặc biệt) |
| 11 | Upsample | 256,40,40 | Không |
| 12 | QConcat | 384,40,40 | Không |
| 13 | QC2f | 128,40,40 | Có — **skip cho L18** |
| 14 | Upsample | 128,80,80 | Không |
| 15 | QConcat | 192,80,80 | Không |
| 16 | QC2f | 64,80,80 | Có |
| 17 | Conv | 64,40,40 | Có |
| 18 | QConcat | 192,40,40 | Không |
| 19 | QC2f | 128,40,40 | Có |
| 20 | SCDown | 128,20,20 | Có |
| 21 | QConcat | 384,20,20 | Không |
| 22 | QC2fCIB | 256,20,20 | Có |

**Đầu ra accelerator (trước L23):** ba nhánh P3/P4/P5 INT8 + metadata — CPU chạy `Qv10Detect` + NMS.

---

## 9. Nguồn nghiên cứu gốc trong repo (giữ nguyên, tra cứu chi tiết)

| Tài liệu | Đường dẫn |
|----------|-----------|
| Khối INT8 chi tiết | `SW_KLTN/MODEL_BLOCKS_INT8_DETAIL.md` |
| CPU vs HW vs head | `SW_KLTN/MODEL_FORWARD_FLOW.md` |
| Skip / concat dependency | `SW_KLTN/MODEL_LAYER_DEPENDENCIES.md` |
| Trace từng layer | `SW_KLTN/MODEL_LAYERS_INT8_FLOW.md` |
| Kiến trúc V2, FPS, BRAM/DSP, scheduler | `SW_KLTN/HW_ARCHITECTURE_V2_100FPS.md` |

**RTL mẫu data path:** `SW_KLTN/PHASE_3/` + `PHASE_5/07_system/rtl/accel_top.sv`, `PHASE_5/data_path/spec_sv/p5_dp_*.sv`.

---

## 10. Rủi ro & việc cần làm tiếp (RTL)

1. **QPSA:** Xác nhận với trace PTQ thực tế; nếu có softmax float, cần **chuẩn fixed-point** hoặc **LUT** có chứng minh sai số chấp nhận được.  
2. **Đồng bộ quant ở QConcat:** Đảm bảo compiler không ghép hai tensor khác scale (hoặc chèn requant).  
3. **PPU parameters:** Nối **DDR/CSR** (PHASE_5) thay vì chỉ port top để SoC khép kín.  
4. **Đồng bộ “hoàn tất inference”:** `sched_inference_complete` nên gắn **DMA + tile xong** (đã ghi chú trong `RTL_IP_DIRECTORY_TREE.txt`).

---

## 11. >100 FPS: kiến trúc trong tài liệu có *thực sự* đạt được không? & đề xuất mảng PE / dataflow

### 11.1. Phân biệt “mục tiêu thiết kế” và “đã chứng minh trên silicon”

- File **`HW_YOLOV10N_RTL_KIEN_TRUC_VA_PRIMITIVE.md`** (và mục 6) mô tả **lớp kiến trúc** (descriptor-driven + 4 SC + PE template + PPU + DMA). **Bản thân nó không chứa chứng cứ đo FPS** — nó **tương thích** với đường triển khai RTL PHASE_3/5.
- Con số **~100–115 FPS (chỉ accelerator / L0–L22)** xuất phát từ **`HW_ARCHITECTURE_V2_100FPS.md`**, với **giả định**:
  - **3072 MAC hoạt động mỗi chu kỳ** (LANES=32, 12 PE/sub, **2 sub RUNNING / SC**, 4 SC),
  - **~3.08 GMAC** cho backbone+neck,
  - **clock 200–220 MHz**, **utilization tổng ~50–55%** (gồm spatial util theo W + overhead thời gian ~27% trong mục 7.4),
  - RTL **thực sự** đạt dual-RUNNING, fill/drain overlap như phân tích, và **QPSA** không phá pipeline.

**Kết luận thẳng:** Đó là **mục tiêu khả thi trên giấy** cho **XC7VX690T-class + host CPU đủ mạnh**; **không** tương đương “chắc chắn >100 FPS” cho đến khi:

1. RTL khớp V2 (PHASE_3 hiện có thể vẫn gần V1 về scheduler/active subs — cần đối chiếu `accel_pkg` / `local_arbiter` / occupancy thực tế).  
2. Có **cosim / FPGA measured** `T_hw` (wall-clock) thay cho spreadsheet.  
3. **End-to-end:** Throughput = `1 / max(T_preprocess, T_hw, T_post)`. Với **MicroBlaze**, tài liệu V2 ghi rõ bottleneck CPU → **~50 FPS** dù HW đủ nhanh (`HW_ARCHITECTURE_V2_100FPS.md` §8.2–8.3).

### 11.2. Khi nào coi là “hiệu quả” cho >100 FPS?

Cần **đồng thời**:

| Điều kiện | Ý nghĩa |
|-----------|---------|
| Đủ **OP/s** | ~**300 GOPS** hiệu dụng cho ~3 GMAC/frame @ 100 FPS (INT8 MAC ≈ OP trong mô hình đơn giản). |
| **Utilization** không bị vỡ | Dual-RUNNING + tile schedule sao cho **compute ≥ fill+drain** (đã phân tích per-tier trong V2 §5.3, 7.3). |
| **Bộ nhớ** | BRAM đủ GLB hoặc chấp nhận spill skip; không để DMA nghẽn lâu. |
| **Compiler** | Tile + barrier khớp **L12/L15/L18/L21**; giảm tile boundary waste. |

Thiếu một trong các mục trên → FPS **tụt** dù “đúng primitive”.

### 11.3. Đề xuất mảng PE & cấu hình dataflow (dễ map từ compiler, vẫn hiệu quả)

**Giữ lõi hiện tại (khuyến nghị làm baseline):**  
**Một lưới PE đồng nhất** (3×4 × LANES), **đa chế độ** qua `template_id` + **router + window_gen** — vì YOLOv10n là tập **conv / DW / pool / concat / upsample** lặp lại, không cần ASIC riêng từng layer.

**Bổ sung để “dễ config” và giảm rủi ro hiệu năng:**

1. **Router / `router_profile_t` giàu ngữ nghĩa hơn**  
   - Thay vì chỉ “mask”, mô tả rõ **cạnh dataflow**: *tensor A → bank slot i*, *tensor B → concat port*, *output → bank j / DDR*.  
   - Compiler sinh **DAG nhỏ** mỗi layer; RTL chỉ **thực thi edge** đã mã hóa → giảm lỗi cấu hình và dễ verify với `MODEL_LAYER_DEPENDENCIES.md`.

2. **Slot skip có tên (logical buffer id)** cho **L4, L6, L8, L13**  
   - Map cố định: ví dụ `SKIP_SLOT_P3 = 0` (L4), … — descriptor chỉ ghi **slot id + DDR fallback** nếu spill.  
   - QConcat (L12, L15, L18, L21) chỉ cần **chọn 2 slot** → cấu hình ổn định, debug nhanh.

3. **Chuẩn hóa “macro-tile” cho QC2f / QC2fCIB**  
   - Một layer phần mềm = **chuỗi micro-tile** cùng `layer_id` với **sub_index** hoặc **phase_id** trong `tile_flags` — hardware vẫn là primitive lặp lại, compiler gánh **số pass** (đã có `num_cin_pass` / `num_k_pass`).

4. **Không tăng PE_COLS lên 8** (đúng hướng V2)  
   - Lợi ích H song song giảm dần khi H nhỏ; chi phí router/PPU/bank **tăng mạnh**. Ưu tiên **LANES** và **2 active sub / SC**.

5. **Phương án thay thế (chỉ khi baseline không đủ FPS)** — trade-off phức tạp:
   - **Hai engine khác biệt:** ví dụ **wide MAC** cho tầng H,W lớn (L0–L4) + **deep engine** cho C lớn, H,W nhỏ — tăng peak GOPS nhưng **tăng gấp đôi công việc schedule và wiring**; chỉ xem xét sau khi đo được tier nào là bottleneck thực tế (bảng MAC/tier §7.1 V2).

6. **QPSA**  
   - Nếu giữ INT8 end-to-end: cần **một primitive GEMM/LUT** có **latency cố định** và **không chiếm master AXI** trong suốt layer; nếu không, mọi phân tích FPS ở L10 nên **tăng margin** hoặc **đo riêng** layer này.

### 11.4. Tóm tắt một câu

- Kiến trúc trong tài liệu PHASE 5 là **hướng đi đúng** và **khớp** với phân tích V2 **để hướng tới** >100 FPS trên HW; **không** nên hiểu là đã **đảm bảo** FPS nếu chưa có RTL V2 đầy đủ + đo thực tế + pipeline CPU phù hợp.  
- **Mảng PE:** nên giữ **đồng nhất + template + router/slot skip rõ ràng**; cải tiến hiệu quả chủ yếu đến từ **scheduler 2-sub-active**, **compiler tile**, và **bộ nhớ/DMA**, hơn là tăng số loại PE khác hẳn.

---

*Tài liệu này là bản tổng hợp thiết kế RTL; mọi số liệu FPS/BRAM/DSP lấy từ `HW_ARCHITECTURE_V2_100FPS.md` và có thể chỉnh khi đổi FPGA hoặc clock.*


---
---

<a id='phần-vii3---rtl-hierarchy'></a>

# PHẦN VII.3 — RTL Hierarchy Top-Down (Phase 6)
> Nguồn: `PHASE_6/PH6_00_RTL_HIERARCHY_TOP_DOWN.md`

---

# PH6-00 — Hệ thống phân cấp RTL: từ top xuống lá

*Tài liệu cho người thiết kế RTL: mỗi cấp trả lời **làm gì**, **nối với ai**, **con bên trong là gì**.*

---

## Cấp 0 — Biên SoC (chip / FPGA wrapper)

| Khối | File RTL (PHASE_3) | Chức năng |
|------|-------------------|-----------|
| **accel_top** | `07_system/rtl/accel_top.sv` | Top IP: **AXI-Lite slave** (MMIO), **một AXI4 master 256b** mux giữa *đọc descriptor* và *DMA tensor* (4 SC); có thể thêm bank PPU (PHASE_5). |
| *(SoC ngoài IP)* | — | CPU, DDR controller, interrupt controller, clock/reset — không nằm trong repo accelerator. |

**Luồng:** CPU ghi CSR → `controller_system` đọc DDR (descriptor) → scheduler đẩy tile → mỗi SC dùng `tensor_dma` đọc/ghi tensor → kết quả P3/P4/P5 nằm DDR theo `dst_off` trong `tile_desc`.

---

## Cấp 1 — Bốn nhánh song song (SuperCluster)

| Khối | File RTL | Chức năng |
|------|----------|-----------|
| **supercluster_wrapper** | `07_system/rtl/supercluster_wrapper.sv` | **Một SC** = FIFO hàng đợi tile từ `global_scheduler` + **local_arbiter** (chọn sub nào dùng cổng DMA) + **một tensor_dma** + **4 × subcluster_wrapper**. |
| **tensor_dma** | `07_system/rtl/tensor_dma.sv` | Engine burst AXI: `rd_req/addr/len` → `m_axi_ar*/*r*`; `wr_*` → `m_axi_aw*/*w*/*b*`. |
| **local_arbiter** | `06_control/rtl/local_arbiter.sv` | Phân cổng `ext_*` (logical) giữa 4 sub khi tranh DMA. |

**Kết nối:** `accel_top` nối `ctrl_sc_tile[gi]`, `ctrl_sc_layer`, `ctrl_sc_valid/accept` vào SC[gi]; mux `m_axi_*` giữa `u_ctrl` (chỉ đọc) và 4 SC.

---

## Cấp 2 — Điều khiển tập trung (không nằm trong SC)

| Khối | File RTL | Chức năng |
|------|----------|-----------|
| **controller_system** | `07_system/rtl/controller_system.sv` | **CSR** (MMIO decode ở `accel_top`), **desc_fetch_engine**, **global_scheduler**, **barrier_manager**, status/IRQ/perf. |
| **desc_fetch_engine** | `06_control/rtl/desc_fetch_engine.sv` | Đọc `net_desc` / `layer_desc` / `tile_desc` từ DDR (64B/block). |
| **global_scheduler** | `06_control/rtl/global_scheduler.sv` | Handshake tile với 4 SC (`sc_tile_valid` / `sc_tile_accept`). |
| **barrier_manager** | `06_control/rtl/barrier_manager.sv` | Scoreboard cho đồng bộ concat/skip (tín hiệu từ SC). |

---

## Cấp 3 — Một Subcluster (đơn vị tính toán + GLB + DMA handshake)

| Khối | File RTL | Chức năng |
|------|----------|-----------|
| **subcluster_wrapper** | `05_integration/rtl/subcluster_wrapper.sv` | **Ghép** tile_fsm, shadow_reg_file, (mục tiêu) GLB + addr_gen + router + window_gen + pe_cluster + ppu + swizzle; cổng **ext_*** tới `tensor_dma`. *Hiện trạng: có nhánh behavioral — cần nối dây đủ theo spec.* |
| **tile_fsm** | `06_control/rtl/tile_fsm.sv` | FSM pha: LOAD_CFG → PREFILL_WT/IN/SKIP → RUN_COMPUTE → POST_PROCESS → SWIZZLE_STORE; sinh `dma_rd_addr/len`, `dma_wr_*`. |
| **shadow_reg_file** | `05_integration/rtl/shadow_reg_file.sv` | Chốt `tile_desc` + `layer_desc` + profile → cấu hình ổn định cho cả tile. |

---

## Cấp 4 — Bộ nhớ on-chip & địa chỉ (theo HW_MAPPING §6)

| Khối | File RTL | Chức năng |
|------|----------|-----------|
| **glb_input_bank** | `03_memory/rtl/glb_input_bank.sv` | Line buffer / slot input (theo banking `h mod 3` trong nghiên cứu mapping). |
| **glb_weight_bank** | `03_memory/rtl/glb_weight_bank.sv` | Cache trọng số tile. |
| **glb_output_bank** | `03_memory/rtl/glb_output_bank.sv` | PSUM (INT32) hoặc ACT (INT8) tùy pass. |
| **metadata_ram** | `03_memory/rtl/metadata_ram.sv` | Valid/slot/ring cho bank. |
| **addr_gen_input** | `03_memory/rtl/addr_gen_input.sv` | Địa chỉ đọc activation (Q_in, row_slot, Wblk). |
| **addr_gen_weight** | `03_memory/rtl/addr_gen_weight.sv` | Địa chỉ đọc weight. |
| **addr_gen_output** | `03_memory/rtl/addr_gen_output.sv` | Địa chỉ ghi output. |

---

## Cấp 5 — Di chuyển & cửa sổ

| Khối | File RTL | Chức năng |
|------|----------|-----------|
| **router_cluster** | `04_data_movement/rtl/router_cluster.sv` | Định tuyến tensor giữa bank ↔ PE; **CONCAT**, broadcast (theo `router_profile_t`). |
| **window_gen** | `04_data_movement/rtl/window_gen.sv` | Sinh cửa sổ **3×3 / 1×1** (và mở rộng 5×5 cho pool nếu cùng engine địa chỉ). |
| **swizzle_engine** | `04_data_movement/rtl/swizzle_engine.sv` | **UPSAMPLE_NEAREST**, reorder kênh/HW trước khi ghi DDR. |

---

## Cấp 6 — Tính toán (MAC + giảm chiều + pool)

| Khối | File RTL | Chức năng |
|------|----------|-----------|
| **pe_cluster** | `05_integration/rtl/pe_cluster.sv` | **12 × pe_unit** + **column_reduce** + **comparator_tree** — thực thi **RS_DENSE_3x3**, **OS_1x1**, **DW_3x3**, một phần **DW_7x7 multipass** (theo mode). |
| **pe_unit** | `01_compute_leaf/rtl/pe_unit.sv` | Lõi MAC song song theo **LANES** (32 trong spec V2). |
| **dsp_pair_int8** | `01_compute_leaf/rtl/dsp_pair_int8.sv` | Cặp INT8×INT8→INT32 (packing DSP nếu có). |
| **column_reduce** | `01_compute_leaf/rtl/column_reduce.sv` | Cộng dồn theo trục giảm cho conv dày (không dùng cho DW thuần). |
| **comparator_tree** | `01_compute_leaf/rtl/comparator_tree.sv` | **MAXPOOL_5x5** (max trên cửa sổ) — vai trò “pool_engine” trong mapping. |

---

## Cấp 7 — Hậu xử lý lượng tử & kích hoạt

| Khối | File RTL | Chức năng |
|------|----------|-----------|
| **ppu** | `02_ppu/rtl/ppu.sv` | Bias INT32 → requant (M_int, shift) → **SiLU LUT** / ReLU / None → clamp INT8; hỗ trợ e-wise (skip). |
| **silu_lut** | `01_compute_leaf/rtl/silu_lut.sv` | ROM/LUT có thể chia sẻ với PPU. |

---

## Cấp 8 — Gói & CSR (không phải module tích hợp)

| Khối | File | Chức năng |
|------|------|-----------|
| **accel_pkg** | `packages/accel_pkg.sv` | `pe_mode_e`, `tile_state_e`, kích thước PE/LANES. |
| **desc_pkg** | `packages/desc_pkg.sv` | `net_desc_t`, `layer_desc_t`, `tile_desc_t`, profiles. |
| **csr_pkg** | `packages/csr_pkg.sv` | Địa chỉ MMIO. |

---

## Sơ đồ kết nối khối (một Subcluster — mục tiêu đích)

```
ext_* (DMA) ──┬──► GLB input/weight  ◄── addr_gen_*
              │
              ▼
         router_cluster ◄──► window_gen ──► pe_cluster ──► ppu ──► swizzle_engine ──► ext_* (write)
              ▲
              └── metadata_ram (slots)
tile_fsm / shadow_reg_file điều khiển pha và load profile
```

---

## Tóm tắt một dòng mỗi cấp

0. **accel_top** — biên bus.  
1. **supercluster_wrapper** — 4 sub + 1 DMA + arbiter tile.  
2. **controller_system** — CSR + fetch descriptor + schedule + barrier.  
3. **subcluster_wrapper** — orchestration một “slice” compute.  
4. **glb_* + addr_gen_*** — bộ nhớ & địa chỉ theo banking.  
5. **router + window + swizzle** — dataflow & reshape.  
6. **pe_cluster (+ comparator_tree)** — primitive MAC/pool.  
7. **ppu** — quant + activation.  
8. **packages** — kiểu dữ liệu chung.


---
---

<a id='phần-vii4---primitive-to-pe-mode'></a>

# PHẦN VII.4 — Primitive to PE Mode Mapping (Phase 6)
> Nguồn: `PHASE_6/PH6_01_PRIMITIVE_TO_PE_MODE_AND_RTL.md`

---

# PH6-01 — Primitive (HW_MAPPING) ↔ `pe_mode_e` ↔ file RTL PHASE_3

Nguồn primitive: `HW_MAPPING_RESEARCH.md` §2, §5, §8.  
Nguồn enum: `PHASE_3/packages/accel_pkg.sv` (`PE_RS3`, `PE_OS1`, `PE_DW3`, `PE_DW7`, `PE_MP5`, `PE_GEMM`, `PE_PASS`).

---

## Bảng ánh xạ chính

| ID | Primitive (mapping) | `pe_mode_e` (gợi ý) | Module RTL PHASE_3 chính | Ghi chú |
|----|---------------------|---------------------|---------------------------|---------|
| P0 | RS_DENSE_3x3 | `PE_RS3` | `window_gen`, `pe_cluster` (pe_unit, column_reduce), `ppu` | Stride/pad trong `layer_desc`; multipass `num_cin_pass` / `num_k_pass`. |
| P1 | OS_1x1 | `PE_OS1` | Cùng datapath; `window_gen` 1×1 | Projection, SPPF cv1/cv2. |
| P2 | DW_3x3 | `PE_DW3` | `pe_cluster` (bỏ/tắt column_reduce theo mode DW) | SCDown nhánh depthwise. |
| P3 | MAXPOOL_5x5 | `PE_MP5` | `comparator_tree` (+ địa chỉ từ `window_gen` 5×5 nếu hỗ trợ) | Không qua PPU requant (INT8 max). |
| P4 | MOVE | `PE_PASS` hoặc FSM-only | `tensor_dma` + GLB; không MAC | Lưu skip / copy buffer. |
| P5 | CONCAT | — (không phải mode PE) | `router_cluster` + có thể `ppu` (requant domain) | QConcat L12/L15/L18/L21. |
| P6 | UPSAMPLE_NEAREST | `PE_PASS` + addr | `swizzle_engine` | L11, L14. |
| P7 | EWISE_ADD | PPU `ewise_en` | `ppu` | Residual nếu dùng. |
| P8 | DW_7x7_MULTIPASS | `PE_DW7` | `pe_cluster` + `tile_fsm` last_pass | 3 pass kernel rows → PSUM rồi PPU ở pass cuối. |
| P9 | GEMM_ATTN_BASIC | `PE_GEMM` | `pe_cluster` (MAC) + **mở rộng điều khiển** | QPSA: GEMM 400×400 @ 20×20; softmax LUT — **rủi ro lớn**, xem PH6-05. |

---

## Luồng dữ liệu theo primitive (đối chiếu HW_MAPPING §5)

| Primitive | Đọc từ | Qua | Ghi vào |
|-----------|--------|-----|---------|
| P0/P1 | glb_input + glb_weight | window_gen → PE → (column_reduce) → PPU | glb_output / DDR |
| P2 | glb_input + glb_weight (per-ch) | window_gen → PE (DW) → PPU | glb_output |
| P3 | glb_input | window/comparator | glb_output (ACT INT8) |
| P5 | hai bank / DDR skip | router (interleave channel) | glb_output |
| P6 | glb hoặc DDR | swizzle (dup addr) | glb/DDR |
| P8 | glb | PE DW multipass, PSUM namespace | PPU last pass → INT8 |
| P9 | glb | nhiều vòng OS_1x1 + GEMM + LUT | PPU |

---

## Việc compiler phải ghi vào descriptor

- **`template_id`** trong `layer_desc` → chọn `pe_mode_e`.  
- **`post_profile_id`** → bias / act / quant_mode cho `ppu`.  
- **`router_profile_id`** → nguồn A/B cho CONCAT, broadcast.  
- **`tile_flags`** → `hold_skip`, `barrier_before/after`, `need_swizzle`, spill.

*Tài liệu chi tiết layer: `HW_MAPPING_RESEARCH.md` §3; output P3/P4/P5: PH6-03.*


---
---

<a id='phần-vii5---subcluster-datapath'></a>

# PHẦN VII.5 — Subcluster Internal Datapath (Phase 6)
> Nguồn: `PHASE_6/PH6_02_SUBCLUSTER_INTERNAL_DATAPATH.md`

---

# PH6-02 — Datapath nội bộ một Subcluster (theo mapping)

Mục tiêu: mô tả **chức năng từng khối nhỏ** và **thứ tự dữ liệu** để thực thi primitive P0–P9, khớp `HW_MAPPING_RESEARCH.md` §5–6.

---

## 1. tile_fsm — “bộ não pha”

**File:** `06_control/rtl/tile_fsm.sv`

| Pha | Hành động phần cứng (mapping) |
|-----|-------------------------------|
| PREFILL_WT | DMA đọc weight OIHW vào `glb_weight_bank`. |
| PREFILL_IN | DMA đọc activation (HWC) vào `glb_input_bank`. |
| PREFILL_SKIP | DMA đọc tensor skip (residual) nếu `tile_flags`. |
| RUN_COMPUTE | `glb_rd_en` + `pe_en`: MAC/pool theo `pe_mode`. |
| ACCUMULATE | Nhiều pass Cin/K — PSUM không qua PPU đến `last_pass`. |
| POST_PROCESS | `ppu_en`: bias + requant + SiLU + clamp. |
| SWIZZLE_STORE | `swizzle_engine` rồi DMA ghi `dst_off` (hoặc giữ GLB nếu HOLD_SKIP). |

---

## 2. shadow_reg_file — “ảnh chụp cấu hình”

**File:** `05_integration/rtl/shadow_reg_file.sv`

- Chốt `template_id` → `o_mode` (PE mode).  
- Chốt `q_in`, `q_out`, pad, stride, `num_*_pass` cho addr_gen và FSM.  
- Chốt `post_profile`, `router_profile` (router hiện có thể `'0` trong một số bản subcluster — cần nối đủ).

---

## 3. addr_gen_* + glb_* — banking (mapping §6)

| Quy tắc (nghiên cứu) | RTL |
|----------------------|-----|
| `bank_input = h mod 3` | `glb_input_bank` ×3 + logic trong addr_gen_input |
| `bank_output = out_row mod 4` | `glb_output_bank` ×4 |
| `row_slot`, `Q_in`, `Wblk` | `addr_gen_input.sv`, `addr_gen_output.sv` — trường `cfg_q_in`, `cfg_q_out` trong `layer_desc` |

**LANES:** `accel_pkg` — V2 dùng 32; mapping gốc viết 16 → đồng bộ với `pe_unit` packing.

---

## 4. window_gen

**File:** `04_data_movement/rtl/window_gen.sv`

- **P0/P1:** cửa sổ 3×3 hoặc 1×1 trên activation.  
- **P3:** mở rộng spec để 5×5 s=1 pad=2 (có thể tham số hóa kernel size trong layer_desc hoặc mode MP5).

---

## 5. pe_cluster

**File:** `05_integration/rtl/pe_cluster.sv`

- **pe_unit × 12:** mỗi unit × LANES MAC song song.  
- **column_reduce:** tích lũy theo Cin cho conv dày.  
- **comparator_tree:** max cho pool (P3).

**DW (P2/P8):** giảm hoặc bỏ column_reduce theo cấu hình mode; multipass P8 do FSM + `last_pass` điều khiển PPU.

---

## 6. ppu

**File:** `02_ppu/rtl/ppu.sv`

Thực hiện đúng chuỗi mapping §2.2: bias → `(acc * m_int) >>> shift` → SiLU LUT → clamp; hỗ trợ e-wise cho skip.

---

## 7. router_cluster

**File:** `04_data_movement/rtl/router_cluster.sv`

- **P5 CONCAT:** đọc lần lượt (hoặc song song) tensor A rồi B theo kênh; nếu scale khác → một nhánh qua PPU requant trước (mini-pipeline hoặc hai lần qua PPU).

---

## 8. swizzle_engine

**File:** `04_data_movement/rtl/swizzle_engine.sv`

- **P6:** nhân bản địa chỉ nguồn lên lưới 2H×2W.  
- Chuẩn bị layout ghi DDR (HWC).

---

## 9. Cổng ext_* ↔ tensor_dma

**File:** `05_integration/rtl/subcluster_wrapper.sv`

Handshake `ext_rd_*` / `ext_wr_*` là **ranh giới** giữa tile_fsm/DMA engine trong SC — đúng với mô hình “tensor post / move” (P4, P6) khi không dùng PE.

---

*Tích hợp đầy đủ: thay behavioral path trong `subcluster_wrapper` bằng nối dây các instance ở trên — xem `RTL_IP_DIRECTORY_TREE.txt` ghi chú.*


---
---

<a id='phần-vii6---inference-layer-map'></a>

# PHẦN VII.6 — Inference P3/P4/P5 & Layer Map (Phase 6)
> Nguồn: `PHASE_6/PH6_03_INFERENCE_P3_P4_P5_AND_LAYER_MAP.md`

---

# PH6-03 — Inference INT8: input → L0–L22 → P3, P4, P5

**Giả định:** Input đã là `X_int8 [1,3,640,640]` + metadata trên CPU (theo `MODEL_FORWARD_FLOW.md`). Accelerator thực hiện **L0–L22**; **L23 Qv10Detect** trên CPU.

---

## Đầu vào / đầu ra accelerator

| Tensor | Shape (batch=1) | Layer sinh ra |
|--------|-----------------|---------------|
| Input | `[3, 640, 640]` INT8 | Trước L0 (DDR) |
| **P3** | `[64, 80, 80]` INT8 | **L16** (sau QC2f) |
| **P4** | `[128, 40, 40]` INT8 | **L19** (sau QC2f) |
| **P5** | `[256, 20, 20]` INT8 | **L22** (sau QC2fCIB) |

---

## Bảng layer → primitive → ý nghĩa RTL

| L | Block | Primitive chính (rút gọn) | Ghi chú |
|---|-------|---------------------------|---------|
| 0 | Conv | P0 RS s2 | Đầu backbone |
| 1 | Conv | P0 RS s2 | |
| 2 | QC2f | P1+P0+P5+P1 | Bottleneck nội bộ |
| 3 | Conv | P0 RS s2 | |
| 4 | QC2f | P1+P0+P5+P1 | **Skip → L15** |
| 5 | SCDown | P1+P2+P5 | 2 nhánh → concat |
| 6 | QC2f | P1+P0+P5+P1 | **Skip → L12** |
| 7 | SCDown | P1+P2+P5 | |
| 8 | QC2f | P1+P0+P5+P1 | **Skip → L21** |
| 9 | SPPF | P1+P3×3+P5+P1 | Pool không PPU |
| 10 | QPSA | P1+P9+P1 | GEMM + softmax approx |
| 11 | Upsample | P6 | |
| 12 | QConcat | P5 | Cần F6 + F11 |
| 13 | QC2f | P1+P0+P5+P1 | **Skip → L18** |
| 14 | Upsample | P6 | |
| 15 | QConcat | P5 | Cần F4 + F14 |
| 16 | QC2f | P1+P0+P5+P1 | **= P3** |
| 17 | Conv | P0 RS s2 | PAN down |
| 18 | QConcat | P5 | F17 + F13 |
| 19 | QC2f | P1+P0+P5+P1 | **= P4** |
| 20 | SCDown | P1+P2 | |
| 21 | QConcat | P5 | F20 + F8 |
| 22 | QC2fCIB | P1+P8+… | **= P5** |

---

## Compiler / descriptor

Mỗi hàng trên được bung thành **một hoặc nhiều `tile_desc`** + **`layer_desc`** với `template_id`, `post_profile_id`, `router_profile_id`, offset DDR (`src_in_off`, `src_w_off`, `src_skip_off`, `dst_off`).

**Ba tensor ra:**

- Ghi DDR tại **3 vùng** (hoặc ping-pong `act0`/`act1`) với pointer trong `net_desc` / bảng output do toolchain định nghĩa.  
- CPU đọc P3/P4/P5 + `scale_3/4/5`, `zp_3/4/5` cho head.

---

*Chi tiết toán & buffer: `HW_MAPPING_RESEARCH.md` §3–4.*


---
---

<a id='phần-vii7---compute-blocks-research'></a>

# PHẦN VII.7 — Research: Compute Blocks Primitive to Layer (Phase 6)
> Nguồn: `PHASE_6/PH6_07_RESEARCH_COMPUTE_BLOCKS_PRIMITIVE_TO_LAYER.md`

---

# PH6-07 — Nghiên cứu chuyên sâu: xây khối tính toán từ Primitive → Layer (YOLOv10n INT8)

**Phạm vi:** Định hướng **nghiên cứu và triển khai** phần cứng kiểu PHASE_3, bám `HW_MAPPING_RESEARCH.md` và cấu trúc RTL thực tế trong `SW_KLTN/PHASE_3/`.  
**Giai đoạn ưu tiên:** Làm **đúng và kiểm thử đầy đủ** các **primitive** rồi mới **ghép layer**; **chưa bắt buộc** hoàn thiện toàn bộ **truyền nhận SoC** (AXI full graph, multi-tile scheduler production) ngay từ đầu — nhưng **không** được bỏ qua **khả năng nối dây** trong thiết kế khối (interface rõ ràng, testbench có thể thay “DMA giả” bằng port cố định).

**Tham chiếu chéo:** `PH6-01` (P→`pe_mode_e`), `PH6-02` (datapath subcluster), `PH6-04` (skip/barrier), `PH6-05` (gap + thứ tự verify), `PH6-06` (catalog file `.sv`).  
**Chuẩn vàng phần mềm:** `SW_KLTN/PHASE1/python_golden/` (ví dụ `primitives/primitive_conv.py` + `quant/quant_affine.py`).

---

## 1. Vấn đề cần giải quyết (problem statement)

Mạng qYOLOv10n PTQ được mô tả trong `HW_MAPPING_RESEARCH.md` như một **đồ thị các layer L0–L22**, trong khi phần cứng thực thi qua **một tập primitive P0–P9** và **một datapath cố định** (PE cluster, PPU, router, swizzle, GLB). Rủi ro chính:

1. **Semantic gap:** “Layer” trong PyTorch ≠ một lần kick PE; một layer có thể = **chuỗi primitive** + **buffer trung gian** + **requant giữa các nhánh**.  
2. **Numeric gap:** Sai một bit trong **làm tròn / thứ tự clamp / per-channel scale** → lệch dần qua nhiều layer.  
3. **Verification gap:** Chỉ chạy end-to-end full model sẽ **không định vị** được lỗi nằm ở primitive nào, descriptor nào, hay scheduler.

**Mục tiêu nghiên cứu của tài liệu này:** Đưa ra **khung phương pháp** để (a) xây và chứng minh **từng primitive** bit-true với golden INT8; (b) **tổ hợp có kiểm soát** thành **layer / block** như trong §3 mapping; (c) chuẩn bị sẵn **điểm nối** cho truyền nhận và descriptor sau này mà không đảo ngược kiến trúc khối.

---

## 2. Khái niệm cốt lõi: Primitive, Layer, Block

### 2.1 Primitive (đơn vị semantics toán học + quant)

Theo `HW_MAPPING_RESEARCH.md` §2.1, mỗi primitive có:

- **Input/output shape** chuẩn (tensor INT8, có thể có PSUM INT32 nội bộ trước requant).  
- **Stride / padding** (nếu có).  
- **Quant domain** (per-tensor, per-channel weight, giữ nguyên scale cho maxpool, align scale cho concat/ewise).  
- **Có đi qua PPU hay không** (bảng §2.1).

**Kiểm thử primitive** = một test có **golden duy nhất** cho **một** semantics đó, với vector nhỏ nhưng **đủ kích hoạt mọi nhánh quant** (ví dụ: có cả kênh dương/âm sau requant, có pad biên).

### 2.2 Layer (đơn vị trong graph YOLO)

Một **layer** trong §3 (ví dụ QC2f, SCDown, SPPF, QConcat) là **chuỗi có thứ tự** các primitive + **bộ nhớ trung gian** (tensor giữ giữa các bước).  
**Kiểm thử layer** = so sánh **tensor cuối của layer** với golden **sau khi** chạy **toàn bộ chuỗi** trong reference, **hoặc** so từng bước trung gian (checkpoint) để khoanh vùng lỗi.

### 2.3 Block (nhóm layer có pattern lặp)

Ví dụ: **QC2f** lặp lại nhiều resolution; **QConcat + QC2f** lặp ở neck. Nghiên cứu phần cứng: sau khi **một** QC2f nhỏ đã pass, các instance khác chủ yếu là **thay đổi shape/scale/weight**, không phải thay đổi RTL.

---

## 3. Nguồn sự thật (sources of truth) — thứ bậc ưu tiên

| Thứ tự | Nguồn | Dùng để |
|--------|--------|---------|
| 1 | `HW_MAPPING_RESEARCH.md` §2.2 | Định nghĩa **công thức quant** và **thứ tự bước** (MAC → bias → requant → act → clamp zp). |
| 2 | `PHASE1/python_golden` + `quant_affine` | **Golden có thể tái lập** (script, seed, export hex/bin). **Không** nhân bản công thức requant thủ công trong nhiều nơi — tránh drift. |
| 3 | `PHASE_3/packages` (`accel_pkg`, `desc_pkg`) | **Hợp đồng bit** giữa compiler và RTL (`pe_mode_e`, `post_profile_t`, v.v.). |
| 4 | RTL PHASE_3 (`ppu.sv`, `pe_unit.sv`, …) | **Implementation**; nếu lệch golden, hoặc sửa RTL hoặc sửa spec nếu golden sai (ghi rõ trong changelog). |

**Nguyên tắc bit-true:** Trước khi tranh luận “độ chính xác mô hình”, khóa **integer pipeline** giữa reference và RTL (cùng `M_int`, `shift`, `zp`, cùng rounding “half-up” như PPU nếu đó là spec).

---

## 4. Ánh xạ Primitive → khối RTL PHASE_3 (tóm tắt nghiên cứu)

Chi tiết bảng đầy đủ: `PH6_01_PRIMITIVE_TO_PE_MODE_AND_RTL.md`. Tóm tắt để định hướng **xây khối**:

| ID | Primitive | `pe_mode_e` / ghi chú | Khối RTL trọng tâm | PPU |
|----|-----------|------------------------|---------------------|-----|
| P0 | RS_DENSE_3x3 | `PE_RS3` | `window_gen`, `pe_cluster`, `column_reduce` | Có |
| P1 | OS_1x1 | `PE_OS1` | Cùng datapath, kernel 1×1 | Có |
| P2 | DW_3x3 | `PE_DW3` | PE DW, tắt/giảm reduce theo mode | Có |
| P3 | MAXPOOL_5x5 | `PE_MP5` | `comparator_tree` + window 5×5 | Không |
| P4 | MOVE | `PE_PASS` / DMA-FSM | `tensor_dma`, GLB, addr | Không |
| P5 | CONCAT | (profile router) | `router_cluster`, có thể + PPU requant | Theo nhánh |
| P6 | UPSAMPLE_NEAREST | `PE_PASS` + swizzle | `swizzle_engine` | Không |
| P7 | EWISE_ADD | PPU | `ppu` `ewise_en` | Có |
| P8 | DW_7x7_MULTIPASS | `PE_DW7` | PE + multipass + `NS_PSUM`/`NS_ACT` | Pass cuối |
| P9 | GEMM_ATTN_BASIC | `PE_GEMM` + điều khiển mở rộng | MAC + **thiếu module attn riêng** (xem PH6-05) | Có |

**Kết luận nghiên cứu:** P0–P8 là **đủ để cover backbone/neck phần lớn** nếu concat/router/swizzle ổn; **P9 là điểm rủi ro kiến trúc** — nên tách **kế hoạch verify** riêng (GEMM nhỏ → softmax approx → ghép QPSA).

---

## 5. Chiến lược xây dựng phần cứng theo pha (không cần full I/O ngay)

### Pha A — “Compute core đóng” (khuyến nghị bắt đầu từ đây)

**Mục tiêu:** Chứng minh **toán đúng** trên interface testbench, **không** phụ thuộc descriptor DDR hay SoC.

- **Đơn vị:** `pe_unit`, `dsp_pair_int8`, `column_reduce`, `comparator_tree`, `silu_lut`.  
- **Cách nối:** TB đứng trực tiếp trên port (array unpacked, valid) hoặc wrapper tối giản.  
- **Golden:** Một output channel / một cửa sổ pool / một lane — export từ `PHASE1/python_golden`.

**Tiêu chí xong pha:** Mọi mode PE liên quan P0–P3, P8 (từng phần) có **ít nhất một** test vector khớp từng phần tử INT8 (hoặc PSUM INT32 nếu test riêng stage).

### Pha B — “Datapath có bộ nhớ cục bộ”

**Mục tiêu:** Đúng **thứ tự byte, offset, halo**, giống một tile nhỏ.

- **Khối:** `window_gen`, `addr_gen_*`, `glb_*_bank` (hoặc model SRAM đơn giản tương đương).  
- **Golden:** Cùng tensor với pha A nhưng **đọc qua addr gen**; so memory dump sau khi fill.

**Tiêu chí xong pha:** Mismatch chỉ còn do **sai layout tensor** (NCHW vs block), không do MAC — phải **đóng băng layout** trong tài liệu compiler.

### Pha C — “PPU + router + swizzle”

**Mục tiêu:** Khóa **requant**, **SiLU LUT**, **concat cùng/khác scale**, **upsample**.

- **Khối:** `ppu.sv`, `router_cluster.sv`, `swizzle_engine.sv`.  
- **Golden:** `post_profile_t` / `router_profile_t` từ testbench (hardcode) trước khi có compiler.

**Tiêu chí xong pha:** Các rule §2.2 cho conv/DW/concat/upsample được **mô phỏng tối thiểu** trên shape nhỏ và khớp RTL.

### Pha D — “Subcluster shell” (`subcluster_wrapper`)

**Mục tiêu:** Một **tile** chạy hết đường ống như `PH6_02`.

- **Kiểm thử:** `tb_pe_cluster` mở rộng hoặc TB riêng gọi `tile_fsm` + shadow reg với **một** `layer_desc_t`/`tile_desc_t` cố định.  
- **Chưa cần:** `desc_fetch_engine` đọc DDR thật — có thể **inject struct** từ TB.

**Tiêu chí xong pha:** Ít nhất **một** primitive mỗi nhóm (conv, pool, DW, upsample, concat đơn giản) chạy qua **cùng shell** và khớp golden.

### Pha E — “Layer composition”

**Mục tiêu:** Chuỗi primitive như §3 (thu nhỏ).

- **Ví dụ thứ tự nghiên cứu:**  
  1. QC2f thu nhỏ: `OS_1x1 → RS_3x3 → CONCAT → OS_1x1` trên `H,W,C` nhỏ.  
  2. SPPF thu nhỏ: `OS_1x1 → MP5×3 → CONCAT 4 nhánh → OS_1x1`.  
  3. SCDown: hai nhánh `OS_1x1 → DW_3x3 s2` + CONCAT (đúng như §3.1 L5).  
- **Checkpoint:** Sau mỗi primitive trong chuỗi, có thể dump tensor trung gian và so với reference **cùng điểm dừng**.

**Tiêu chí xong pha:** Một layer phức tạp = **pass** khi **cả chuỗi** khớp; nếu fail, **đã có** checkpoint để biết bước nào.

### Pha F — Truyền nhận & descriptor đầy đủ (sau cùng trong roadmap này)

**Mục tiêu:** `controller_system`, `desc_fetch_engine`, CSR, DMA — đúng **binary layout** `net_desc_t`/`layer_desc_t`/`tile_desc_t`.

- Pha này **không thay thế** pha A–E; nó **đóng gói** đường đi đã chứng minh.

---

## 6. Kim tự tháp kiểm thử (test pyramid) — yêu cầu “đầy đủ”

### 6.1 Mức L0 — Unit RTL

- Mỗi file lá trong `01_compute_leaf`, `02_ppu` có TB riêng (repo đã có nhiều `tb_*.sv`).  
- **Pass criteria:** 100% mode được primitive sử dụng; coverage toggle cho `pe_mode_e` liên quan.

### 6.2 Mức L1 — Primitive integration

- Một test = **một** P0…P8 (P9 tách riêng).  
- **Vector:** nhỏ, cố định, lưu trong `PHASE1` hoặc `test_vectors/` (hex).  
- **Scoreboard:** so INT8 từng phần tử hoặc `max |a-b|` + số phần tử sai.

### 6.3 Mức L2 — Mini-layer / block

- Chuỗi primitive từ §3; shape giảm.  
- **Pass criteria:** output layer khớp reference; nếu có concat cross-scale, **bắt buộc** có case scale_A ≠ scale_B.

### 6.4 Mức L3 — Layer thật (subset L0–L22)

- Bắt đầu từ layer có ít phụ thuộc skip (ví dụ L0, L1), sau đó thêm **QConcat có skip** (L12, L15…) — cần **chiến lược buffer** `PH6_04`.

### 6.5 Mức L4 — E2E SoC

- `tb_accel_e2e` + DDR model; so P3/P4/P5 hoặc hash — chỉ khi L3 ổn.

---

## 7. Descriptor & truyền nhận: vai trò trong nghiên cứu

`desc_pkg.sv` **không** định nghĩa toán học primitive; nó định nghĩa **cách điều khiển** phần cứng. Do đó:

- **Giai đoạn primitive-first:** TB có thể **bỏ qua** `tile_table_offset` thật, inject struct trực tiếp.  
- **Giai đoạn layer:** Phải chứng minh **compiler** sinh `template_id` khớp `pe_mode_e`, `num_*_pass` khớp P8, `router_profile_id` khớp P5.  
- **Test riêng cho descriptor:** Đọc binary → `layer_desc_t` → so từng field với golden struct (không cần chạy MAC).

---

## 8. Rủi ro và giả định cần ghi nhận trong luận văn / báo cáo

1. **HW_MAPPING vs RTL:** Văn bản mapping có thể ghi `LANES=16` minh họa; RTL dùng `LANES=32` (`accel_pkg`) — **semantics không đổi**, **tiling đổi**; mọi golden phải **parameterize** theo package.  
2. **SCDown đầy đủ:** §3 có nhánh kép + CONCAT; verify layer L5/L7/L20 phải **bám đúng chuỗi** đã chọn, không chỉ một nhánh đơn giản hóa.  
3. **P9 QPSA:** GEMM 400×400 @ INT8 và softmax approx — **không** gộp vào tiêu chí “primitive cơ bản xong” cho đến khi có spec RTL softmax + test GEMM nhỏ.  
4. **Skip lớn (F4, F6, F8, F13):** Đúng layer = đúng **capacity GLB + thời gian giữ**; đây là kiểm thử **hệ thống bộ nhớ**, không chỉ MAC.

---

## 9. Tiêu chí hoàn thành (definition of done) theo cấp

| Cấp | Hoàn thành khi |
|-----|----------------|
| **Primitive** | Mỗi P0–P8 có ≥1 vector bit-true; tài liệu map P→RTL→TB; P9 có kế hoạch riêng. |
| **Layer** | ≥3 block đại diện (QC2f nhỏ, SPPF nhỏ, SCDown 2 nhánh) pass so với `PHASE1` hoặc script tương đương. |
| **Graph nhỏ** | Chuỗi L0→L3 hoặc tương đương không cần skip dài, pass end-to-end trên shell Pha D. |
| **Production path** | Descriptor + DMA + CSR pass trên `tb_accel_e2e` với cùng vector đã dùng ở Pha D. |

---

## 10. Kết luận nghiên cứu

- **Đúng primitive trước** là điều kiện **cần** để layer có ý nghĩa; **đúng layer** là điều kiện **cần** để graph YOLO có ý nghĩa.  
- **Chưa cần truyền nhận đầy đủ** ở giai đoạn đầu **nếu** interface khối tính toán được thiết kế để sau này **thay TB inject bằng descriptor/DMA** mà không đổi datapath.  
- **Đầy đủ kiểm thử** được hiểu là **kim tự tháp** L0–L4, không phải chỉ một testbench duy nhất.

---

*Tài liệu này là spec nghiên cứu / phương pháp; cập nhật khi RTL hoặc `HW_MAPPING_RESEARCH.md` thay đổi phiên bản.*


---
---

# ════════════════════════════════════════════════════════════════
# PHỤ LỤC — BẢNG TỔNG HỢP NGUỒN TÀI LIỆU
# ════════════════════════════════════════════════════════════════

| # | Phần | File nguồn | Số dòng | Nội dung chính |
|---|------|------------|---------|----------------|
| 1 | I.1 | MODEL_FORWARD_FLOW.md | ~175 | Luồng CPU→HW→CPU, phân chia trách nhiệm |
| 2 | I.2 | MODEL_LAYERS_INT8_FLOW.md | ~590 | Chi tiết L0-L22 INT8, quantize/dequant |
| 3 | I.3 | MODEL_BLOCKS_INT8_DETAIL.md | ~597 | 8 khối: Conv, QC2f, SCDown, SPPF, QPSA, Upsample, QConcat, QC2fCIB |
| 4 | I.4 | MODEL_LAYER_DEPENDENCIES.md | ~119 | Đồ thị phụ thuộc, 4 skip connections, ~900KB buffer |
| 5 | II.1 | HW_MAPPING_RESEARCH.md | ~989 | Primitive P0-P9, Layer→Primitive mapping, banking, addressing |
| 6 | III.1 | HW_ARCHITECTURE_V2_100FPS.md | ~1099 | V2: LANES=32, Dual-RUNNING, DSP packing, 3072 MACs, >100 FPS |
| 7 | IV.1 | HW_ACCELERATION_IMPL_FLOW.md | ~869 | 5 phases: Spec→Golden Python→Block Oracle→Layer Runner→RTL |
| 8 | V.1 | RTL_MODULE_SPEC.md | ~1903 | 35+ RTL modules: I/O, FSM, pipeline, test criteria |
| 9 | VI.1 | PHASE0/01_primitive_matrix.md | ~233 | P0-P9 freeze spec, golden math, PPU pipeline |
| 10 | VI.2 | PHASE0/02_layer_mapping.md | ~577 | L0-L22 decomposition, skip deps, barriers |
| 11 | VI.3 | PHASE0/03_quant_policy.md | ~303 | INT8 affine quant, per-channel, requant formula, SiLU LUT |
| 12 | VI.4 | PHASE0/04_layout_addressing.md | ~390 | Banking h mod 3, row slot, lane packing, HOLD_SKIP sizing |
| 13 | VI.5 | PHASE0/05_descriptor_spec.md | ~458 | NET/LAYER/TILE_DESC, ROUTER/POST_PROFILE structs |
| 14 | VI.6 | PHASE0/06_execution_semantics.md | ~442 | last_pass, PSUM/ACT namespace, barriers, non-PE paths |
| 15 | VI.7 | PHASE0/07_golden_python_plan.md | ~489 | Python oracle structure, test criteria, dump format |
| 16 | VI.8 | PHASE0/08_rtl_mapping_plan.md | ~528 | RTL hierarchy 8 levels, module specs, verification |
| 17 | VII.1 | PHASE_3/BUILD_STRATEGY.md | ~628 | Bottom-up build: dsp_pair→pe→PPU→memory→control→system |
| 18 | VII.2 | PHASE_5/HW_YOLOV10N_RTL...md | ~260 | Descriptor-driven template, >100 FPS feasibility |
| 19 | VII.3 | PHASE_6/PH6_00_RTL_HIERARCHY.md | ~131 | 8-level hierarchy top-down walkthrough |
| 20 | VII.4 | PHASE_6/PH6_01_PRIM_TO_PE.md | ~47 | P0-P9 → pe_mode_e mapping table |
| 21 | VII.5 | PHASE_6/PH6_02_SUBCLUSTER.md | ~100 | Subcluster datapath: tile_fsm→GLB→PE→PPU→swizzle |
| 22 | VII.6 | PHASE_6/PH6_03_INFERENCE.md | ~59 | P3/P4/P5 output mapping, compiler descriptors |
| 23 | VII.7 | PHASE_6/PH6_07_RESEARCH.md | ~207 | 6-phase build strategy, 5-level test pyramid |

**Tổng cộng: 23 tài liệu nghiên cứu, ~11,500+ dòng**

---

*Tài liệu này được tổng hợp tự động từ toàn bộ nguồn nghiên cứu của đề tài.*
