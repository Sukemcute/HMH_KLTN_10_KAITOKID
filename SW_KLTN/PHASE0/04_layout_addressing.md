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
