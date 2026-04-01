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
