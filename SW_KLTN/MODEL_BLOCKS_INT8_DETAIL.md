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

