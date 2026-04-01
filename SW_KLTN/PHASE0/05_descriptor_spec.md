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
