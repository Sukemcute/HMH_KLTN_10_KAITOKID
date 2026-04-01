# End-to-End YOLOv10n INT8 Inference Flow

## Từ Ảnh Input → Bounding Box Output trên Virtex-7 FPGA

> **Mục tiêu**: Mô tả TOÀN BỘ quy trình inference 1 ảnh qua 3 giai đoạn,
> tính toán chi tiết thời gian từng bước, và hướng dẫn nhúng lên board Virtex-7.

---

# MỤC LỤC

```
1. TỔNG QUAN HỆ THỐNG
2. GIAI ĐOẠN 1: PREPROCESSING + QUANTIZATION (CPU)
3. GIAI ĐOẠN 2: HW ACCELERATOR (IP trên FPGA)
4. GIAI ĐOẠN 3: POSTPROCESSING + DETECTION (CPU)
5. TÍNH TOÁN THỜI GIAN CHI TIẾT
6. KIẾN TRÚC NHÚNG TRÊN VIRTEX-7
7. GIAO TIẾP CPU ↔ DDR3 ↔ IP: FLOW CHI TIẾT
8. C CODE CHO MICROBLAZE
9. PIPELINE 3 TẦNG & FPS PHÂN TÍCH
10. CÁC KỊCH BẢN TRIỂN KHAI
```

---

# 1. TỔNG QUAN HỆ THỐNG

## 1.1. End-to-End Data Flow

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                     YOLOv10n INT8 — End-to-End Flow                      ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║   ┌──────────┐     ┌──────────────┐     ┌──────────────┐                ║
║   │  Camera   │────→│  GIAI ĐOẠN 1 │────→│  GIAI ĐOẠN 2 │               ║
║   │  /Image   │     │  CPU Preproc  │     │  HW Accel IP │               ║
║   │  640×640  │     │  + Quantize   │     │  L0 → L22    │               ║
║   └──────────┘     │              │     │              │                ║
║                     │ Output:      │     │ Output:      │                ║
║                     │ X_int8       │     │ P3,P4,P5     │                ║
║                     │ [1,3,640,640]│     │ (INT8)       │                ║
║                     └──────┬───────┘     └──────┬───────┘               ║
║                            │                     │                       ║
║                            ▼                     ▼                       ║
║                     ┌─────────────────────────────────┐                 ║
║                     │          DDR3 SDRAM              │                 ║
║                     │  (shared memory giữa CPU và IP) │                 ║
║                     └─────────────────┬───────────────┘                 ║
║                                       │                                  ║
║                                       ▼                                  ║
║                     ┌──────────────────────────────┐                    ║
║                     │       GIAI ĐOẠN 3             │                    ║
║                     │  CPU Postprocessing           │                    ║
║                     │  Dequant → Qv10Detect         │                    ║
║                     │  → Decode BBox → NMS → Draw   │                    ║
║                     │                                │                    ║
║                     │  Output: ảnh + bounding boxes  │                    ║
║                     └──────────────────────────────┘                    ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

## 1.2. Ba giai đoạn tóm tắt

| Giai đoạn | Thực hiện trên         | Input            | Output                   | Mô tả                          |
| --------- | ---------------------- | ---------------- | ------------------------ | ------------------------------ |
| **GĐ1**   | CPU (MicroBlaze / x86) | Ảnh BGR 640×640  | X_int8 [1,3,640,640]     | Resize + Normalize + Quantize  |
| **GĐ2**   | FPGA IP (accel_top)    | X_int8 từ DDR3   | P3/P4/P5 INT8 trong DDR3 | Backbone + Neck (23 layers)    |
| **GĐ3**   | CPU                    | P3/P4/P5 từ DDR3 | Bounding boxes + labels  | Dequant + DetHead + NMS + Draw |

---

# 2. GIAI ĐOẠN 1: PREPROCESSING + QUANTIZATION (CPU)

## 2.1. Pipeline chi tiết

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GIAI ĐOẠN 1: CPU PREPROCESSING                       │
│                                                                         │
│  ① Đọc ảnh gốc (JPEG/PNG/Camera frame)                                 │
│     Input:  image_raw[H_orig × W_orig × 3] (BGR, uint8)               │
│     Size:   ví dụ 1920×1080×3 = 6,220,800 bytes                       │
│                              │                                          │
│                              ▼                                          │
│  ② LetterBox Resize → 640×640                                         │
│     - Giữ tỷ lệ (aspect ratio)                                        │
│     - Padding thêm border (grey = 114)                                  │
│     - Bilinear interpolation                                           │
│     Output: image_resized[640 × 640 × 3] (BGR, uint8)                 │
│     Size:   640×640×3 = 1,228,800 bytes                               │
│                              │                                          │
│                              ▼                                          │
│  ③ Normalize (÷255) + Channel Reorder (BGR→RGB)                       │
│     pixel_float = pixel_uint8 / 255.0                                  │
│     Output: image_norm[3 × 640 × 640] (float32, range [0.0, 1.0])     │
│     Size:   3×640×640×4 = 4,915,200 bytes (float32)                   │
│                              │                                          │
│                              ▼                                          │
│  ④ Quantize Affine (float32 → INT8)                                   │
│     q = clamp(round(x_float / scale) + zero_point, -128, 127)         │
│                                                                         │
│     YOLOv10n input quantization params:                                │
│       scale   = 0.003921568627 ≈ 1/255                                │
│       zp      = 0                                                       │
│                                                                         │
│     ★ SIMPLIFICATION:                                                  │
│       q = clamp(round(pixel_uint8 / (255 × 0.003921568627)), -128, 127)│
│         = clamp(round(pixel_uint8 / 0.9999...), -128, 127)            │
│         ≈ pixel_uint8  (gần như identity mapping!)                     │
│                                                                         │
│     ★ FAST PATH (avoid float entirely):                                │
│       q_int8 = (pixel_uint8 * 256 + 128) >> 8                         │
│              ≈ pixel_uint8 (fixed-point approximation)                 │
│                                                                         │
│     Output: X_int8[1, 3, 640, 640] (signed INT8, NCHW layout)         │
│     Size:   1×3×640×640 = 1,228,800 bytes                             │
│                              │                                          │
│                              ▼                                          │
│  ⑤ DMA Write → DDR3                                                   │
│     Ghi X_int8 vào DDR3 tại địa chỉ 0x0030_0000                       │
│     Transfer size: 1,228,800 bytes                                     │
│     @ 32 bytes/beat × 200 MHz = 6.4 GB/s                              │
│     Time: 1.23 MB / 6.4 GB/s = 0.19 ms                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2.2. Tính toán thời gian GĐ1

### Trên MicroBlaze (200 MHz, no FPU, no cache)

```
┌───────────────────────────┬────────────────────────────────────────────┐
│ Sub-step                  │ Thời gian & Phân tích                      │
├───────────────────────────┼────────────────────────────────────────────┤
│ ② LetterBox Resize       │ 8.0 ms                                     │
│   640×640×3 = 1.23M pixels│ Bilinear interp: ~6.5 cycles/pixel        │
│   = 1.23M × 6.5 / 200M   │ = 0.04 ms/1K pixels                       │
│                           │ MicroBlaze: no FPU → fixed-point interp   │
│                           │ ≈ 1.23M × 6.5 = 8.0M cycles = 40 ms!     │
│                           │ ★ Optimized: integer bilinear = 8 ms      │
├───────────────────────────┼────────────────────────────────────────────┤
│ ③ Normalize ÷255         │ 5.0 ms                                     │
│   1.23M × 4 cycles/pixel │ MicroBlaze: integer div ~20 cycles         │
│                           │ ★ Fast: multiply by (256/255) as shift    │
│                           │ ★ Even faster: skip normalize, fold into  │
│                           │   quantize step                            │
├───────────────────────────┼────────────────────────────────────────────┤
│ ④ Quantize Affine        │ 7.0 ms                                     │
│   1.23M × round + clamp  │ Per pixel: multiply + add + clamp          │
│                           │ ★ With FAST PATH: just copy uint8 as int8 │
│                           │   → reduces to 2 ms (memcpy + sign adjust)│
├───────────────────────────┼────────────────────────────────────────────┤
│ ⑤ DMA Write to DDR3      │ 0.19 ms                                    │
│   1.23 MB / 6.4 GB/s     │ AXI4 burst write, 256-bit bus             │
├───────────────────────────┼────────────────────────────────────────────┤
│ ★ TOTAL (MicroBlaze)     │ ≈ 20 ms (worst case)                      │
│ ★ TOTAL (optimized)      │ ≈ 10 ms (with fast-path quant)            │
│ ★ TOTAL (x86 host, AVX2) │ ≈ 2.5 ms                                  │
└───────────────────────────┴────────────────────────────────────────────┘
```

### Trên Host PC (x86, C++, AVX2 SIMD)

```
② Resize:    1.0 ms (OpenCV, optimized SIMD)
③ Normalize: 0.5 ms (vectorized ÷255)
④ Quantize:  0.8 ms (SIMD round + clamp)
⑤ DMA:       0.1 ms (PCIe DMA write)
─────────────────────────────
TOTAL:       2.5 ms
```

## 2.3. Data Layout sau GĐ1

```
X_int8 trong DDR3:

Địa chỉ DDR3: 0x0030_0000 → 0x0043_BFFF (1,228,800 bytes)

Layout: NCHW (batch=1)
  Channel 0 (R): bytes [0 .. 409,599]        = 640×640 INT8
  Channel 1 (G): bytes [409,600 .. 819,199]   = 640×640 INT8
  Channel 2 (B): bytes [819,200 .. 1,228,799] = 640×640 INT8

Mỗi pixel: signed INT8, range [-128, 127]
  (thực tế với scale=1/255: range [0, 255] → [0, 127] do uint8→int8)
```

---

# 3. GIAI ĐOẠN 2: HW ACCELERATOR (IP trên FPGA)

## 3.1. Tổng quan hoạt động

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GIAI ĐOẠN 2: HW ACCELERATOR                          │
│                                                                         │
│  CPU đã ghi X_int8 vào DDR3 @ 0x0030_0000                             │
│  CPU đã ghi Descriptors vào DDR3 @ 0x0000_0000                        │
│  CPU đã ghi Weights vào DDR3 @ 0x0010_0000                            │
│                                                                         │
│  ① CPU viết CSR register → khởi động IP                               │
│     CSR[0x010] = net_desc_base_lo = 0x0000_0000                       │
│     CSR[0x014] = net_desc_base_hi = 0x0000_0000                       │
│     CSR[0x018] = layer_start = 0                                       │
│     CSR[0x01C] = layer_end = 22                                        │
│     CSR[0x000] = CTRL.start = 1                                        │
│                              │                                          │
│                              ▼                                          │
│  ② IP tự động thực hiện:                                               │
│     a. desc_fetch_engine đọc NET_DESC từ DDR3                          │
│     b. Parse LAYER_DESC[0..22] lần lượt                                │
│     c. Parse TILE_DESC cho mỗi layer                                   │
│     d. global_scheduler phân phối tiles → 4 SuperClusters              │
│     e. Mỗi SC:                                                         │
│        - local_arbiter gán roles (2×RUNNING + FILLING + DRAINING)      │
│        - subcluster_wrapper thực thi tile:                             │
│          LOAD_CFG → PREFILL_WT → PREFILL_IN → COMPUTE →               │
│          POST_PROCESS → SWIZZLE_STORE → DONE                          │
│     f. Barrier management cho skip connections:                        │
│        L6→L12, L4→L15, L13→L18, L8→L21                               │
│                              │                                          │
│                              ▼                                          │
│  ③ IP ghi kết quả vào DDR3:                                           │
│     P3 [1,64,80,80]   = 409,600 B → DDR3 @ 0x0080_0000              │
│     P4 [1,128,40,40]  = 204,800 B → DDR3 @ 0x00A0_0000              │
│     P5 [1,256,20,20]  = 102,400 B → DDR3 @ 0x00B0_0000              │
│                              │                                          │
│                              ▼                                          │
│  ④ IP assert IRQ → CPU biết inference hoàn thành                      │
│     CSR[0x004].done = 1, CSR[0x004].irq = 1                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 3.2. Tính toán thời gian GĐ2 — Per Layer

```
╔══════════════════════════════════════════════════════════════════════════╗
║  HW Accelerator V2: 3,072 active INT8 MACs/cycle @ 200 MHz             ║
║  Peak throughput: 614.4 GOPS                                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ┌────────┬───────────────────┬────────┬─────────┬────────────────────┐ ║
║  │ Layer  │ Operation         │ MACs   │ S_util  │ T_compute (ms)     │ ║
║  ├────────┼───────────────────┼────────┼─────────┼────────────────────┤ ║
║  │  L0    │ Conv3×3 s=2       │  44 M  │ 100.0%  │ 0.072              │ ║
║  │  L1    │ Conv3×3 s=2       │ 118 M  │ 100.0%  │ 0.192              │ ║
║  │  L2    │ QC2f (32→32)      │ 170 M  │ 100.0%  │ 0.277              │ ║
║  │  L3    │ Conv3×3 s=2       │ 118 M  │  83.3%  │ 0.231              │ ║
║  │  L4    │ QC2f (64→64)      │ 340 M  │  83.3%  │ 0.665              │ ║
║  │  L5    │ SCDown            │  60 M  │  62.5%  │ 0.156              │ ║
║  │  L6    │ QC2f (128→128)    │ 320 M  │  62.5%  │ 0.833              │ ║
║  │  L7    │ SCDown            │ 120 M  │  62.5%  │ 0.312              │ ║
║  │  L8    │ QC2f (256→256)    │ 550 M  │  62.5%  │ 1.432              │ ║
║  │  L9    │ SPPF              │ 200 M  │  62.5%  │ 0.521              │ ║
║  │  L10   │ QPSA (Attention)  │ 360 M  │  62.5%  │ 0.937              │ ║
║  │  L11   │ Upsample ×2       │   0    │   —     │ 0.050 (DMA only)   │ ║
║  │  L12   │ Concat            │   0    │   —     │ 0.030 (DMA only)   │ ║
║  │  L13   │ QC2f (384→128)    │ 220 M  │  62.5%  │ 0.573              │ ║
║  │  L14   │ Upsample ×2       │   0    │   —     │ 0.050 (DMA only)   │ ║
║  │  L15   │ Concat            │   0    │   —     │ 0.030 (DMA only)   │ ║
║  │  L16   │ QC2f (192→64)     │ 180 M  │  83.3%  │ 0.352              │ ║
║  │  L17   │ Conv3×3 s=2       │  37 M  │  83.3%  │ 0.072              │ ║
║  │  L18   │ Concat            │   0    │   —     │ 0.030 (DMA only)   │ ║
║  │  L19   │ QC2f (192→128)    │ 110 M  │  62.5%  │ 0.286              │ ║
║  │  L20   │ SCDown            │  30 M  │  62.5%  │ 0.078              │ ║
║  │  L21   │ Concat            │   0    │   —     │ 0.030 (DMA only)   │ ║
║  │  L22   │ QC2fCIB (DW7×7)   │ 100 M  │  62.5%  │ 0.260              │ ║
║  ├────────┼───────────────────┼────────┼─────────┼────────────────────┤ ║
║  │ TOTAL  │ Pure compute      │3,077 M │   —     │ 7.25 ms            │ ║
║  │        │ + temporal overhead│  ×1.37 │         │                    │ ║
║  │        │ = Actual HW time  │        │         │ 9.93 ms            │ ║
║  └────────┴───────────────────┴────────┴─────────┴────────────────────┘ ║
║                                                                          ║
║  T_compute formula: MACs / (3072 × spatial_util × 200MHz)               ║
║                                                                          ║
║  Temporal overhead breakdown:                                            ║
║    Fill/drain gaps:     10%                                              ║
║    Descriptor fetch:     3%                                              ║
║    Barrier stalls:       5%                                              ║
║    DW7×7 inefficiency:   2%                                              ║
║    Tile boundary waste:  5%                                              ║
║    QPSA softmax:         2%                                              ║
║    ─────────────────────────                                             ║
║    Total overhead:      27% → temporal_util = 73%                        ║
║                                                                          ║
║  ★ T_hw = 7.25 / 0.73 = 9.93 ms ≈ 10 ms                               ║
╚══════════════════════════════════════════════════════════════════════════╝
```

## 3.3. Data sizes chuyển qua DDR3 trong GĐ2

```
Read từ DDR3 (IP đọc):
  Descriptors:    ~64 KB  (NET + 23 LAYER + ~2000 TILE × 64B)
  Weights:        ~1.8 MB (all layers, INT8)
  Input X_int8:   1.23 MB
  Skip tensors:   ~0.9 MB (F4, F6, F8, F13 — đọc lại khi cần)
  ──────────────────────
  Total read:     ~4.0 MB

Write vào DDR3 (IP ghi):
  Intermediate:   ~2.0 MB (activation ping/pong buffers)
  P3 output:      400 KB
  P4 output:      200 KB
  P5 output:      100 KB
  Skip spill:     ~0.4 MB (F4 nếu cần)
  ──────────────────────
  Total write:    ~3.1 MB

Grand total DDR3 traffic: ~7.1 MB per inference
@ 12.8 GB/s DDR3 BW: 7.1 MB / 12.8 GB/s = 0.55 ms (lẫn vào compute time)
→ DDR3 KHÔNG phải bottleneck ✓
```

---

# 4. GIAI ĐOẠN 3: POSTPROCESSING + DETECTION (CPU)

## 4.1. Pipeline chi tiết

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GIAI ĐOẠN 3: CPU POSTPROCESSING                       │
│                                                                         │
│  IP đã ghi P3/P4/P5 vào DDR3. CPU nhận IRQ hoặc poll CSR.done.        │
│                                                                         │
│  ① DMA Read P3/P4/P5 từ DDR3                                          │
│     P3: [1,64,80,80]  = 409,600 B (INT8) @ 0x0080_0000              │
│     P4: [1,128,40,40] = 204,800 B (INT8) @ 0x00A0_0000              │
│     P5: [1,256,20,20] = 102,400 B (INT8) @ 0x00B0_0000              │
│     Total: 716,800 bytes                                               │
│     Time: 0.72 MB / 6.4 GB/s = 0.11 ms                                │
│                              │                                          │
│                              ▼                                          │
│  ② Dequantize (INT8 → float32)                                        │
│     x_float = (x_int8 - zero_point) × scale                           │
│     Per-channel scale/zp (từ calibration)                              │
│     Total elements: (64×80×80) + (128×40×40) + (256×20×20)            │
│                   = 409,600 + 204,800 + 102,400 = 716,800 elements    │
│                              │                                          │
│                              ▼                                          │
│  ③ Qv10Detect Head (Detection Head)                                   │
│     YOLOv10n detection head per scale level:                           │
│                                                                         │
│     P3 (80×80, small objects):                                          │
│       cls_pred: Conv1×1(64→num_classes=80)                             │
│       reg_pred: Conv1×1(64→4×reg_max=64)                              │
│       → 80×80 × (80 + 64) = 921,600 values                           │
│                                                                         │
│     P4 (40×40, medium objects):                                         │
│       cls_pred + reg_pred                                              │
│       → 40×40 × 144 = 230,400 values                                  │
│                                                                         │
│     P5 (20×20, large objects):                                          │
│       cls_pred + reg_pred                                              │
│       → 20×20 × 144 = 57,600 values                                   │
│                                                                         │
│     ★ NOTE: Detection head thường chạy trên CPU vì:                   │
│       - Nhỏ (~0.35 GFLOPs = 5% total model)                           │
│       - Có nhiều irregular operations (DFL, sigmoid)                    │
│       - Không đáng tạo HW riêng                                       │
│                              │                                          │
│                              ▼                                          │
│  ④ Decode Bounding Boxes                                               │
│     Raw predictions → (x_center, y_center, width, height, class, conf) │
│     DFL (Distribution Focal Loss): softmax + weighted sum              │
│     Adjust for stride and anchor offsets                               │
│     Total anchors: 80×80 + 40×40 + 20×20 = 8,400                     │
│                              │                                          │
│                              ▼                                          │
│  ⑤ Confidence Threshold Filter                                        │
│     Loại bỏ boxes có confidence < threshold (typ. 0.25)               │
│     Typical: 8,400 → ~50-200 candidate boxes                          │
│                              │                                          │
│                              ▼                                          │
│  ⑥ NMS (Non-Maximum Suppression)                                      │
│     ★ YOLOv10n: ONE-TO-ONE matching → NMS-free!                       │
│     Nhưng vẫn cần threshold filtering                                  │
│     Typical output: 5-50 final detections                              │
│                              │                                          │
│                              ▼                                          │
│  ⑦ Draw Bounding Boxes + Labels                                       │
│     Vẽ rectangles + text trên ảnh gốc                                  │
│     Output: ảnh với bounding boxes                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 4.2. Tính toán thời gian GĐ3

```
┌───────────────────────────┬──────────┬──────────┬──────────┐
│ Sub-step                  │ x86+AVX2 │MicroBlaze│ ARM Zynq │
├───────────────────────────┼──────────┼──────────┼──────────┤
│ ① DMA Read P3/P4/P5      │ 0.05 ms  │ 0.05 ms  │ 0.05 ms  │
│   716 KB / 6.4 GB/s      │ (PCIe)   │ (local)  │          │
├───────────────────────────┼──────────┼──────────┼──────────┤
│ ② Dequantize             │ 0.3 ms   │ 3.0 ms   │ 1.0 ms   │
│   716K × (sub + mul)     │ SIMD     │ soft FP  │ NEON FP  │
│   Per element: ~2 FLOPs  │          │          │          │
│   Total: 1.43 MFLOPs     │          │          │          │
├───────────────────────────┼──────────┼──────────┼──────────┤
│ ③ Qv10Detect Head        │ 2.0 ms   │ 12.0 ms  │ 4.0 ms   │
│   Conv1×1 + DFL + sigmoid│          │          │          │
│   ~350 MFLOPs            │          │          │          │
│   ★ Heaviest CPU step!   │          │          │          │
├───────────────────────────┼──────────┼──────────┼──────────┤
│ ④ Decode BBox            │ 0.5 ms   │ 3.0 ms   │ 1.0 ms   │
│   8,400 anchors decode   │          │          │          │
├───────────────────────────┼──────────┼──────────┼──────────┤
│ ⑤+⑥ Filter + NMS        │ 0.2 ms   │ 1.0 ms   │ 0.5 ms   │
│   YOLOv10n: NMS-free     │          │          │          │
├───────────────────────────┼──────────┼──────────┼──────────┤
│ ⑦ Draw BBox              │ 0.5 ms   │ 2.0 ms   │ 1.0 ms   │
├───────────────────────────┼──────────┼──────────┼──────────┤
│ ★ TOTAL GĐ3             │ 3.55 ms  │ 21.05 ms │ 7.55 ms  │
└───────────────────────────┴──────────┴──────────┴──────────┘
```

---

# 5. TÍNH TOÁN THỜI GIAN CHI TIẾT — TỔNG KẾT

## 5.1. Thời gian inference 1 ảnh (Non-pipelined, sequential)

```
T_total = T_GĐ1 + T_GĐ2 + T_GĐ3

╔══════════════════════════════════════════════════════════════════════╗
║  SEQUENTIAL (1 ảnh, không pipeline)                                  ║
╠═══════════════════╤══════════╤══════════╤══════════╤════════════════╣
║ Platform          │ T_GĐ1   │ T_GĐ2   │ T_GĐ3   │ T_total (1 ảnh)║
╠═══════════════════╪══════════╪══════════╪══════════╪════════════════╣
║ x86 + FPGA (PCIe)│ 2.5 ms  │ 9.9 ms  │ 3.5 ms  │ 15.9 ms        ║
║                   │          │          │          │ → 63 FPS       ║
╠═══════════════════╪══════════╪══════════╪══════════╪════════════════╣
║ MicroBlaze + FPGA │ 20.0 ms │ 9.9 ms  │ 21.0 ms │ 50.9 ms        ║
║ (Virtex-7 only)   │          │          │          │ → 20 FPS       ║
╠═══════════════════╪══════════╪══════════╪══════════╪════════════════╣
║ MicroBlaze (opt)  │ 10.0 ms │ 9.9 ms  │ 12.0 ms │ 31.9 ms        ║
║ (fast-path quant) │          │          │          │ → 31 FPS       ║
╠═══════════════════╪══════════╪══════════╪══════════╪════════════════╣
║ ARM Zynq + PL     │ 6.5 ms  │ 9.9 ms  │ 7.0 ms  │ 23.4 ms        ║
║                   │          │          │          │ → 43 FPS       ║
╚═══════════════════╧══════════╧══════════╧══════════╧════════════════╝
```

## 5.2. Thời gian FPS khi Pipeline 3 tầng

```
Throughput = 1 / max(T_GĐ1, T_GĐ2, T_GĐ3)
Latency   = T_GĐ1 + T_GĐ2 + T_GĐ3

╔══════════════════════════════════════════════════════════════════════╗
║  PIPELINED (3-stage overlap)                                         ║
╠═══════════════════╤══════════╤══════════╤══════════╤════════════════╣
║ Platform          │ T_GĐ1   │ T_GĐ2   │ T_GĐ3   │ Throughput     ║
╠═══════════════════╪══════════╪══════════╪══════════╪════════════════╣
║ x86 + FPGA       │ 2.5 ms  │ 9.9 ms  │ 3.5 ms  │ 1/9.9 ms       ║
║                   │          │ ★bottleneck        │ = 101 FPS ✓   ║
╠═══════════════════╪══════════╪══════════╪══════════╪════════════════╣
║ MicroBlaze        │ 20.0 ms │ 9.9 ms  │ 21.0 ms │ 1/21.0 ms      ║
║                   │★bottleneck│        │★bottleneck│ = 48 FPS ✗   ║
╠═══════════════════╪══════════╪══════════╪══════════╪════════════════╣
║ MicroBlaze (opt)  │ 10.0 ms │ 9.9 ms  │ 12.0 ms │ 1/12.0 ms      ║
║                   │          │          │★bottleneck│ = 83 FPS     ║
╠═══════════════════╪══════════╪══════════╪══════════╪════════════════╣
║ ARM Zynq          │ 6.5 ms  │ 9.9 ms  │ 7.0 ms  │ 1/9.9 ms       ║
║                   │          │★bottleneck│        │ = 101 FPS ✓   ║
╚═══════════════════╧══════════╧══════════╧══════════╧════════════════╝

★ Pipeline hoạt động khi 3 giai đoạn overlap:
   Frame N:   GĐ1 đang preprocess ảnh mới
   Frame N-1: GĐ2 đang compute trên IP
   Frame N-2: GĐ3 đang postprocess kết quả cũ
   → Throughput = 1 / max(3 stages)
```

## 5.3. Biểu đồ timeline pipeline

```
Time(ms):  0    5    10   15   20   25   30   35   40

x86+FPGA:
  GĐ1:  [==]                [==]                [==]
  GĐ2:       [========]          [========]
  GĐ3:                 [===]          [===]
  FPS:  ←─── 9.9ms ───→  ← 9.9ms →  = 101 FPS

MicroBlaze:
  GĐ1:  [==================]                    [==================]
  GĐ2:                      [========]
  GĐ3:                                [===================]
  FPS:  ←──────── 21.0ms ──────────→  = 48 FPS (GĐ3 bottleneck)

MicroBlaze (optimized):
  GĐ1:  [=========]              [=========]
  GĐ2:             [========]              [========]
  GĐ3:                       [===========]           [===========]
  FPS:  ←──── 12.0ms ────→  = 83 FPS (GĐ3 bottleneck)
```

---

# 6. KIẾN TRÚC NHÚNG TRÊN VIRTEX-7

## 6.1. Block Diagram trên Board Virtex-7

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                        Virtex-7 FPGA Board                               ║
║                                                                          ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │                        FPGA Fabric                                │   ║
║  │                                                                   │   ║
║  │  ┌─────────────┐     AXI-Lite        ┌──────────────────────┐   │   ║
║  │  │             │    (32b, control)    │                      │   │   ║
║  │  │  MicroBlaze │ ═══════════════════→ │    accel_top         │   │   ║
║  │  │  CPU Core   │                      │    (YOLOv10n IP)     │   │   ║
║  │  │  200 MHz    │     IRQ line         │                      │   │   ║
║  │  │             │ ←─────────────────── │  3,072 INT8 MACs     │   │   ║
║  │  │  ┌───────┐  │                      │  4 SuperClusters     │   │   ║
║  │  │  │ ILMB  │  │                      │  200 MHz             │   │   ║
║  │  │  │ DLMB  │  │                      └──────────┬───────────┘   │   ║
║  │  │  │ 128KB │  │                                 │                │   ║
║  │  │  └───────┘  │                          AXI4 Master             │   ║
║  │  └──────┬──────┘                          (256b, DMA)             │   ║
║  │         │                                        │                │   ║
║  │    AXI4 Master                                   │                │   ║
║  │    (32b or 64b)                                  │                │   ║
║  │         │                                        │                │   ║
║  │  ┌──────▼────────────────────────────────────────▼──────────┐    │   ║
║  │  │                 AXI Interconnect                           │    │   ║
║  │  │           (Crossbar / SmartConnect)                       │    │   ║
║  │  │                                                           │    │   ║
║  │  │  Port 0: MicroBlaze (32b) — lower priority               │    │   ║
║  │  │  Port 1: accel_top  (256b) — higher priority              │    │   ║
║  │  └──────────────────────┬────────────────────────────────────┘    │   ║
║  │                         │                                         │   ║
║  │                    AXI4 Slave                                     │   ║
║  │                    (256b)                                         │   ║
║  │                         │                                         │   ║
║  │  ┌──────────────────────▼────────────────────────────────────┐    │   ║
║  │  │              MIG DDR3 Controller                           │    │   ║
║  │  │         (Xilinx Memory Interface Generator)               │    │   ║
║  │  │         800 MHz DDR3, 64-bit interface                    │    │   ║
║  │  │         Effective BW: 12.8 GB/s                           │    │   ║
║  │  └──────────────────────┬────────────────────────────────────┘    │   ║
║  │                         │                                         │   ║
║  └─────────────────────────┼─────────────────────────────────────────┘   ║
║                             │                                            ║
╠═════════════════════════════╪════════════════════════════════════════════╣
║                             │  Physical DDR3 bus                         ║
║  ┌──────────────────────────▼──────────────────────────────┐            ║
║  │                    DDR3 SDRAM                            │            ║
║  │               1-2 GB, DDR3-1600                          │            ║
║  │                                                          │            ║
║  │   0x0000_0000: Descriptors (1 MB)                       │            ║
║  │   0x0010_0000: Weights     (2 MB)                       │            ║
║  │   0x0030_0000: Input X_int8 (1.23 MB)                   │            ║
║  │   0x0050_0000: Act buffers  (2 MB)                      │            ║
║  │   0x0080_0000: P3 output    (400 KB)                    │            ║
║  │   0x00A0_0000: P4 output    (200 KB)                    │            ║
║  │   0x00B0_0000: P5 output    (100 KB)                    │            ║
║  │   0x00C0_0000: Skip spill   (1 MB)                      │            ║
║  │   0x0100_0000: CPU workspace (remaining)                 │            ║
║  └──────────────────────────────────────────────────────────┘            ║
╚══════════════════════════════════════════════════════════════════════════╝
```

## 6.2. AXI Bus Topology

```
                    MicroBlaze
                    (AXI4 Master, 32-bit)
                        │
                        │ M_AXI_DP (data port)
                        │
           ┌────────────▼────────────┐
           │   AXI Interconnect       │
           │   (Vivado IP: axi_inter- │
           │    connect or smartconnect│
           ├──────────────────────────┤
           │  S0: MicroBlaze (32b)    │
           │  S1: accel_top  (256b)   │  ← AXI4 Master from IP
           │                          │
           │  M0: DDR3 MIG   (256b)   │  ← AXI4 Slave
           │  M1: accel_top  (32b)    │  ← AXI-Lite Slave (CSR)
           │  M2: UART       (32b)    │  ← (optional debug)
           │  M3: GPIO       (32b)    │  ← (optional LED/button)
           └──────────────────────────┘

Address Map:
  0x0000_0000 — 0x3FFF_FFFF : DDR3 (1 GB)
  0x4400_0000 — 0x4400_0FFF : accel_top CSR (AXI-Lite, 4 KB)
  0x4060_0000 — 0x4060_FFFF : UART Lite
  0x4000_0000 — 0x4000_FFFF : GPIO
```

---

# 7. GIAO TIẾP CPU ↔ DDR3 ↔ IP: FLOW CHI TIẾT

## 7.1. Initialization Flow (Power-on, 1 lần duy nhất)

```
╔══════════════════════════════════════════════════════════════════════╗
║  INITIALIZATION (chạy 1 lần khi power-on)                           ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  STEP 1: CPU khởi tạo DDR3                                          ║
║    MIG calibration completes (tự động bởi MIG IP)                   ║
║    CPU test DDR3 read/write OK                                       ║
║                                                                      ║
║  STEP 2: CPU load WEIGHTS vào DDR3                                  ║
║    Source: Flash/UART/PCIe từ host PC                                ║
║    Destination: DDR3 @ 0x0010_0000                                  ║
║    Size: ~1.8 MB                                                     ║
║    ★ Chỉ cần load 1 lần, weights không đổi giữa các frames         ║
║                                                                      ║
║    for (int i = 0; i < weight_size; i += 4) {                       ║
║        Xil_Out32(DDR3_WEIGHT_BASE + i, weight_data[i/4]);           ║
║    }                                                                 ║
║                                                                      ║
║  STEP 3: CPU tạo DESCRIPTORS và ghi vào DDR3                       ║
║    Tính toán tile decomposition cho 23 layers                        ║
║    Tạo: 1 NET_DESC + 23 LAYER_DESC + ~2000 TILE_DESC               ║
║    Destination: DDR3 @ 0x0000_0000                                  ║
║    Size: ~128 KB                                                     ║
║    ★ Chỉ cần tạo 1 lần (nếu input size cố định 640×640)           ║
║                                                                      ║
║  STEP 4: CPU preload SiLU LUT vào DDR3 (packed in descriptors)     ║
║    256 entries × INT8 = 256 bytes                                    ║
║                                                                      ║
║  STEP 5: CPU reset IP                                                ║
║    Xil_Out32(ACCEL_BASE + CSR_CTRL, 0x2);  // soft_reset            ║
║    usleep(10);                                                       ║
║    Xil_Out32(ACCEL_BASE + CSR_CTRL, 0x0);  // release reset         ║
║                                                                      ║
║  STEP 6: CPU verify IP                                               ║
║    uint32_t ver = Xil_In32(ACCEL_BASE + CSR_VERSION);               ║
║    assert(ver == 0x594F_0002);  // "YO" + version 2                 ║
║    uint32_t cap = Xil_In32(ACCEL_BASE + CSR_CAP0);                  ║
║    // cap should report: SC=4, Sub=4, Rows=3, Cols=4, Lanes=32     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

## 7.2. Per-Frame Inference Flow (chạy cho mỗi ảnh)

```
╔══════════════════════════════════════════════════════════════════════════╗
║             PER-FRAME INFERENCE — FLOW CHI TIẾT                         ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │  BƯỚC 1: CPU PREPROCESSING (Giai đoạn 1)                        │   ║
║  │                                                                   │   ║
║  │  1a. CPU đọc ảnh mới (từ camera/UART/Flash)                     │   ║
║  │      uint8_t image_bgr[640*640*3];                               │   ║
║  │      receive_image(image_bgr);                                   │   ║
║  │                                                                   │   ║
║  │  1b. CPU resize + normalize + quantize                           │   ║
║  │      int8_t X_int8[3*640*640];                                   │   ║
║  │      for (ch = 0; ch < 3; ch++)                                  │   ║
║  │        for (h = 0; h < 640; h++)                                 │   ║
║  │          for (w = 0; w < 640; w++) {                             │   ║
║  │            uint8_t p = image_bgr[h*640*3 + w*3 + (2-ch)];       │   ║
║  │            X_int8[ch*640*640 + h*640 + w] = (int8_t)p;          │   ║
║  │          }                                                       │   ║
║  │      // ★ NCHW layout, BGR→RGB reorder by (2-ch)               │   ║
║  │                                                                   │   ║
║  │  1c. CPU ghi X_int8 → DDR3                                      │   ║
║  │      memcpy((void*)DDR3_INPUT_BASE, X_int8, 3*640*640);         │   ║
║  │      // DDR3_INPUT_BASE = 0x0030_0000                            │   ║
║  │      // MicroBlaze: sử dụng Xil_Out32 loop hoặc DMA engine     │   ║
║  │                                                                   │   ║
║  └──────────────────────────────────────────────────────────────────┘   ║
║                              │                                           ║
║                              ▼                                           ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │  BƯỚC 2: KHỞI ĐỘNG IP (CPU → IP qua AXI-Lite)                  │   ║
║  │                                                                   │   ║
║  │  // Set descriptor base address                                   │   ║
║  │  Xil_Out32(ACCEL_BASE + 0x010, 0x00000000); // net_desc_base_lo │   ║
║  │  Xil_Out32(ACCEL_BASE + 0x014, 0x00000000); // net_desc_base_hi │   ║
║  │                                                                   │   ║
║  │  // Set layer range (all 23 layers)                               │   ║
║  │  Xil_Out32(ACCEL_BASE + 0x018, 0);   // layer_start = 0        │   ║
║  │  Xil_Out32(ACCEL_BASE + 0x01C, 22);  // layer_end = 22         │   ║
║  │                                                                   │   ║
║  │  // Enable interrupt                                              │   ║
║  │  Xil_Out32(ACCEL_BASE + 0x020, 0x1); // irq_mask enable         │   ║
║  │                                                                   │   ║
║  │  // ★ START inference!                                           │   ║
║  │  Xil_Out32(ACCEL_BASE + 0x000, 0x1); // CTRL.start = 1         │   ║
║  │                                                                   │   ║
║  └──────────────────────────────────────────────────────────────────┘   ║
║                              │                                           ║
║                              ▼                                           ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │  BƯỚC 3: IP TỰ ĐỘNG CHẠY (không cần CPU can thiệp)             │   ║
║  │                                                                   │   ║
║  │  IP nội bộ thực hiện:                                            │   ║
║  │    → desc_fetch_engine đọc NET_DESC từ DDR3                     │   ║
║  │    → Parse 23 LAYER_DESC, ~2000 TILE_DESC                       │   ║
║  │    → 4 SuperCluster × 4 Subcluster xử lý tiles                 │   ║
║  │    → DMA tự động đọc weights/activations từ DDR3                │   ║
║  │    → DMA tự động ghi results vào DDR3                           │   ║
║  │    → Barrier manager đồng bộ skip connections                   │   ║
║  │                                                                   │   ║
║  │  ★ Trong lúc này CPU RẢNH → có thể pipeline GĐ1 cho frame kế  │   ║
║  │  ★ Hoặc CPU có thể poll CSR_STATUS hoặc chờ IRQ                │   ║
║  │                                                                   │   ║
║  │  Thời gian: ~9.9 ms                                              │   ║
║  │                                                                   │   ║
║  └──────────────────────────────────────────────────────────────────┘   ║
║                              │                                           ║
║                              ▼                                           ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │  BƯỚC 4: CPU ĐĂNG KÝ KẾT QUẢ (Poll hoặc IRQ)                  │   ║
║  │                                                                   │   ║
║  │  // Option A: Polling (đơn giản)                                  │   ║
║  │  while (1) {                                                      │   ║
║  │      uint32_t status = Xil_In32(ACCEL_BASE + 0x004);            │   ║
║  │      if (status & 0x2) break;  // done bit                       │   ║
║  │  }                                                                │   ║
║  │                                                                   │   ║
║  │  // Option B: Interrupt (hiệu quả hơn)                           │   ║
║  │  // ISR handler set flag when IRQ fires                           │   ║
║  │  while (!inference_done_flag) {                                   │   ║
║  │      // CPU can do GĐ1 preprocessing here!                      │   ║
║  │  }                                                                │   ║
║  │                                                                   │   ║
║  │  // Clear interrupt                                               │   ║
║  │  Xil_Out32(ACCEL_BASE + 0x000, 0x4); // CTRL.irq_clr            │   ║
║  │                                                                   │   ║
║  └──────────────────────────────────────────────────────────────────┘   ║
║                              │                                           ║
║                              ▼                                           ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │  BƯỚC 5: CPU ĐỌC KẾT QUẢ TỪ DDR3 (Giai đoạn 3)                │   ║
║  │                                                                   │   ║
║  │  // Đọc P3 output                                                 │   ║
║  │  int8_t P3[64*80*80];   // 409,600 bytes                        │   ║
║  │  memcpy(P3, (void*)0x00800000, 64*80*80);                       │   ║
║  │                                                                   │   ║
║  │  // Đọc P4 output                                                 │   ║
║  │  int8_t P4[128*40*40];  // 204,800 bytes                        │   ║
║  │  memcpy(P4, (void*)0x00A00000, 128*40*40);                      │   ║
║  │                                                                   │   ║
║  │  // Đọc P5 output                                                 │   ║
║  │  int8_t P5[256*20*20];  // 102,400 bytes                        │   ║
║  │  memcpy(P5, (void*)0x00B00000, 256*20*20);                      │   ║
║  │                                                                   │   ║
║  └──────────────────────────────────────────────────────────────────┘   ║
║                              │                                           ║
║                              ▼                                           ║
║  ┌──────────────────────────────────────────────────────────────────┐   ║
║  │  BƯỚC 6: CPU POSTPROCESSING (Giai đoạn 3)                       │   ║
║  │                                                                   │   ║
║  │  6a. Dequantize: INT8 → float32                                   │   ║
║  │      for each element:                                            │   ║
║  │        float val = (P3[i] - zp_out) * scale_out;                │   ║
║  │                                                                   │   ║
║  │  6b. Detection Head (Conv1×1 + DFL + sigmoid)                    │   ║
║  │      → cls_scores[8400][80]  (80 COCO classes)                   │   ║
║  │      → bbox_preds[8400][4]   (x,y,w,h)                          │   ║
║  │                                                                   │   ║
║  │  6c. Confidence filter (threshold = 0.25)                        │   ║
║  │      → ~50-200 candidate boxes                                    │   ║
║  │                                                                   │   ║
║  │  6d. YOLOv10n: NMS-free (one-to-one matching)                    │   ║
║  │      → final_boxes[N][6]  (x1,y1,x2,y2,conf,class)             │   ║
║  │                                                                   │   ║
║  │  6e. Scale boxes back to original image coordinates               │   ║
║  │      → account for LetterBox padding/resize ratio                │   ║
║  │                                                                   │   ║
║  │  6f. Draw bounding boxes + labels on image                        │   ║
║  │      → output to display / UART / save to memory                 │   ║
║  │                                                                   │   ║
║  └──────────────────────────────────────────────────────────────────┘   ║
║                                                                          ║
║  ← QUAY LẠI BƯỚC 1 cho ảnh tiếp theo →                                 ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

## 7.3. Memory Transaction Diagram

```
Timeline một lần inference:

CPU                  AXI Bus              DDR3               IP (accel_top)
 │                     │                    │                    │
 │  ──── GĐ1 ─────    │                    │                    │
 │  preprocess image   │                    │                    │
 │  ...                │                    │                    │
 │  Write X_int8 ─────→│───── AXI Write ───→│  X_int8 stored    │
 │                     │                    │  @ 0x0030_0000    │
 │                     │                    │                    │
 │  Write CSR.start ──→│─── AXI-Lite Wr ──→│                   │← CSR received
 │                     │                    │                    │  start=1
 │                     │                    │                    │
 │  (idle / preproc    │                    │                    │  ── GĐ2 ──
 │   next frame)       │                    │                    │
 │                     │                    │←── AXI Read ──────│  read descs
 │                     │                    │                    │  read weights
 │                     │                    │←── AXI Read ──────│  read X_int8
 │                     │                    │                    │  compute...
 │                     │                    │                    │  compute...
 │                     │                    │←── AXI Write ─────│  write P3
 │                     │                    │←── AXI Write ─────│  write P4
 │                     │                    │←── AXI Write ─────│  write P5
 │                     │                    │                    │
 │  ← IRQ ───────────←│←─── IRQ line ─────│                   │← done, IRQ
 │                     │                    │                    │
 │  Read CSR.status ──→│─── AXI-Lite Rd ──→│                   │← status=done
 │  Clear IRQ ────────→│─── AXI-Lite Wr ──→│                   │
 │                     │                    │                    │
 │  ──── GĐ3 ─────    │                    │                    │
 │  Read P3 ──────────→│───── AXI Read ───→│  return P3 data   │
 │  Read P4 ──────────→│───── AXI Read ───→│  return P4 data   │
 │  Read P5 ──────────→│───── AXI Read ───→│  return P5 data   │
 │                     │                    │                    │
 │  postprocess...     │                    │                    │
 │  dequant, detect,   │                    │                    │
 │  draw bbox          │                    │                    │
 │  DONE               │                    │                    │
```

---

# 8. C CODE CHO MICROBLAZE

## 8.1. Header definitions

```c
/* ═══════════════════════════════════════════════════
 *  yolov10n_accel.h — Driver header for HW accelerator
 * ═══════════════════════════════════════════════════ */

#ifndef YOLOV10N_ACCEL_H
#define YOLOV10N_ACCEL_H

#include <stdint.h>
#include "xil_io.h"

/* ── Base Addresses ── */
#define ACCEL_BASE        0x44000000
#define DDR3_BASE         0x00000000

/* ── DDR3 Memory Map ── */
#define DDR3_DESC_BASE    (DDR3_BASE + 0x00000000)  /* 1 MB  descriptors  */
#define DDR3_WEIGHT_BASE  (DDR3_BASE + 0x00100000)  /* 2 MB  weights      */
#define DDR3_INPUT_BASE   (DDR3_BASE + 0x00300000)  /* 1.23 MB input      */
#define DDR3_ACT_PING     (DDR3_BASE + 0x00500000)  /* 2 MB  act buffer 0 */
#define DDR3_ACT_PONG     (DDR3_BASE + 0x00700000)  /* 2 MB  act buffer 1 */
#define DDR3_P3_BASE      (DDR3_BASE + 0x00800000)  /* 400 KB P3 output   */
#define DDR3_P4_BASE      (DDR3_BASE + 0x00A00000)  /* 200 KB P4 output   */
#define DDR3_P5_BASE      (DDR3_BASE + 0x00B00000)  /* 100 KB P5 output   */
#define DDR3_SKIP_BASE    (DDR3_BASE + 0x00C00000)  /* 1 MB  skip spill   */
#define DDR3_CPU_WORK     (DDR3_BASE + 0x01000000)  /* remaining for CPU  */

/* ── CSR Register Offsets ── */
#define CSR_CTRL           0x000  /* [0]start [1]soft_reset [2]irq_clr */
#define CSR_STATUS         0x004  /* [0]busy  [1]done [2]irq [3]error  */
#define CSR_VERSION        0x008
#define CSR_CAP0           0x00C
#define CSR_NET_DESC_LO    0x010
#define CSR_NET_DESC_HI    0x014
#define CSR_LAYER_START    0x018
#define CSR_LAYER_END      0x01C
#define CSR_IRQ_MASK       0x020
#define CSR_PERF_CTRL      0x030
#define CSR_PERF_CYCLE_LO  0x034
#define CSR_PERF_CYCLE_HI  0x038
#define CSR_PERF_TILE_DONE 0x03C

/* ── Output sizes ── */
#define P3_CHANNELS  64
#define P3_HEIGHT    80
#define P3_WIDTH     80
#define P3_SIZE      (P3_CHANNELS * P3_HEIGHT * P3_WIDTH)  /* 409,600 */

#define P4_CHANNELS  128
#define P4_HEIGHT    40
#define P4_WIDTH     40
#define P4_SIZE      (P4_CHANNELS * P4_HEIGHT * P4_WIDTH)  /* 204,800 */

#define P5_CHANNELS  256
#define P5_HEIGHT    20
#define P5_WIDTH     20
#define P5_SIZE      (P5_CHANNELS * P5_HEIGHT * P5_WIDTH)  /* 102,400 */

#define INPUT_SIZE   (3 * 640 * 640)  /* 1,228,800 */

/* ── Helper macros ── */
#define ACCEL_WR(off, val)  Xil_Out32(ACCEL_BASE + (off), (val))
#define ACCEL_RD(off)       Xil_In32(ACCEL_BASE + (off))

#endif
```

## 8.2. Main inference function

```c
/* ═══════════════════════════════════════════════════
 *  yolov10n_accel.c — Full inference driver
 * ═══════════════════════════════════════════════════ */

#include "yolov10n_accel.h"
#include "xil_cache.h"
#include <string.h>

/* ──────────── INITIALIZATION (1 time) ──────────── */
int accel_init(const uint8_t *weight_data, uint32_t weight_size,
               const uint8_t *desc_data,   uint32_t desc_size)
{
    /* Step 1: Soft reset IP */
    ACCEL_WR(CSR_CTRL, 0x2);
    for (volatile int i = 0; i < 100; i++);
    ACCEL_WR(CSR_CTRL, 0x0);

    /* Step 2: Verify IP */
    uint32_t ver = ACCEL_RD(CSR_VERSION);
    if ((ver >> 16) != 0x594F) return -1;  /* "YO" magic */

    /* Step 3: Load weights to DDR3 */
    memcpy((void *)DDR3_WEIGHT_BASE, weight_data, weight_size);

    /* Step 4: Load descriptors to DDR3 */
    memcpy((void *)DDR3_DESC_BASE, desc_data, desc_size);

    /* Flush cache to ensure DDR3 has latest data */
    Xil_DCacheFlushRange(DDR3_WEIGHT_BASE, weight_size);
    Xil_DCacheFlushRange(DDR3_DESC_BASE, desc_size);

    return 0;
}

/* ─── STAGE 1: Preprocess (CPU) ─── */
void preprocess_image(const uint8_t *image_bgr_640x640,
                      int8_t *X_int8)
{
    /* BGR→RGB reorder + uint8→int8 conversion
     * Layout: NCHW (channel-first)
     * scale ≈ 1/255, zp = 0 → q ≈ pixel value */
    for (int ch = 0; ch < 3; ch++) {
        int src_ch = 2 - ch;  /* BGR→RGB */
        for (int h = 0; h < 640; h++) {
            for (int w = 0; w < 640; w++) {
                uint8_t pixel = image_bgr_640x640[h * 640 * 3 + w * 3 + src_ch];
                /* Fast quantize: q = pixel (since scale ≈ 1/255 and input is [0,1]) */
                X_int8[ch * 640 * 640 + h * 640 + w] = (int8_t)pixel;
            }
        }
    }
}

/* ─── STAGE 2: Run HW Accelerator ─── */
int run_accelerator(const int8_t *X_int8)
{
    /* Write input tensor to DDR3 */
    memcpy((void *)DDR3_INPUT_BASE, X_int8, INPUT_SIZE);
    Xil_DCacheFlushRange(DDR3_INPUT_BASE, INPUT_SIZE);

    /* Configure CSR registers */
    ACCEL_WR(CSR_NET_DESC_LO, DDR3_DESC_BASE & 0xFFFFFFFF);
    ACCEL_WR(CSR_NET_DESC_HI, 0x00000000);
    ACCEL_WR(CSR_LAYER_START, 0);
    ACCEL_WR(CSR_LAYER_END,   22);
    ACCEL_WR(CSR_IRQ_MASK,    0x1);

    /* Reset performance counters */
    ACCEL_WR(CSR_PERF_CTRL, 0x1);

    /* ★ START! */
    ACCEL_WR(CSR_CTRL, 0x1);

    /* Wait for completion (polling) */
    uint32_t status;
    uint32_t timeout = 100000000;  /* ~500 ms at 200 MHz */
    do {
        status = ACCEL_RD(CSR_STATUS);
        if (status & 0x8) return -1;  /* error bit */
        timeout--;
    } while (!(status & 0x2) && timeout > 0);

    if (timeout == 0) return -2;  /* timeout */

    /* Clear done/irq */
    ACCEL_WR(CSR_CTRL, 0x4);

    /* Read performance counters */
    uint32_t cycles_lo = ACCEL_RD(CSR_PERF_CYCLE_LO);
    uint32_t cycles_hi = ACCEL_RD(CSR_PERF_CYCLE_HI);
    uint64_t total_cycles = ((uint64_t)cycles_hi << 32) | cycles_lo;
    /* total_cycles / 200MHz = time in seconds */

    return 0;
}

/* ─── STAGE 3: Read results + Postprocess ─── */
typedef struct {
    float x1, y1, x2, y2;
    float confidence;
    int   class_id;
} bbox_t;

int postprocess(bbox_t *detections, int max_detections)
{
    /* Invalidate cache to get fresh DDR3 data from IP */
    Xil_DCacheInvalidateRange(DDR3_P3_BASE, P3_SIZE);
    Xil_DCacheInvalidateRange(DDR3_P4_BASE, P4_SIZE);
    Xil_DCacheInvalidateRange(DDR3_P5_BASE, P5_SIZE);

    /* Read outputs from DDR3 */
    int8_t P3[P3_SIZE], P4[P4_SIZE], P5[P5_SIZE];
    memcpy(P3, (void *)DDR3_P3_BASE, P3_SIZE);
    memcpy(P4, (void *)DDR3_P4_BASE, P4_SIZE);
    memcpy(P5, (void *)DDR3_P5_BASE, P5_SIZE);

    /* ─── Dequantize INT8 → float ─── */
    /* Per-layer output scales (from calibration) */
    const float scale_p3 = 0.0235f;
    const float scale_p4 = 0.0312f;
    const float scale_p5 = 0.0418f;
    const int   zp = 0;

    /* ─── Detection Head (simplified) ─── */
    /* This is Qv10Detect: Conv1×1 projections for cls and bbox */
    /* In practice: load det_head weights, run Conv1×1 on CPU */

    int num_detections = 0;
    /* Process each scale level */
    int strides[3] = {8, 16, 32};
    int heights[3] = {80, 40, 20};
    int widths[3]  = {80, 40, 20};
    int8_t *feature_maps[3] = {P3, P4, P5};
    float scales[3] = {scale_p3, scale_p4, scale_p5};
    int channels[3] = {64, 128, 256};

    for (int s = 0; s < 3; s++) {
        int H = heights[s], W = widths[s];
        int C = channels[s];
        float sc = scales[s];
        int stride = strides[s];

        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                /* Dequantize feature vector at (h, w) */
                /* Run det head Conv1×1 → cls_score + bbox_pred */
                /* Apply sigmoid to cls_score */
                /* If max_cls_score > conf_threshold: */
                /*   Decode bbox using DFL */
                /*   Scale to image coordinates */
                /*   Store in detections[] */

                /* (Simplified — actual implementation requires
                    det_head weights and DFL computation) */
            }
        }
    }

    /* YOLOv10n: NMS-free, just filter by confidence */
    /* Sort by confidence, take top max_detections */

    return num_detections;
}

/* ─── MAIN LOOP ─── */
void inference_loop(void)
{
    int8_t   X_int8[INPUT_SIZE];
    uint8_t  image_bgr[640 * 640 * 3];
    bbox_t   detections[100];

    while (1) {
        /* Stage 1: Preprocess */
        receive_image(image_bgr);  /* from camera / UART */
        preprocess_image(image_bgr, X_int8);

        /* Stage 2: HW Accelerator */
        int ret = run_accelerator(X_int8);
        if (ret != 0) {
            /* handle error */
            continue;
        }

        /* Stage 3: Postprocess */
        int num_det = postprocess(detections, 100);

        /* Output results */
        for (int i = 0; i < num_det; i++) {
            print_detection(&detections[i]);
            /* or draw on framebuffer */
        }
    }
}
```

## 8.3. Pipelined version (for higher FPS)

```c
/* ═══════════════════════════════════════════════════
 *  Pipelined inference: overlap 3 stages
 *  Frame N:   Stage 1 (preprocess)
 *  Frame N-1: Stage 2 (HW accelerator)
 *  Frame N-2: Stage 3 (postprocess)
 * ═══════════════════════════════════════════════════ */

void inference_loop_pipelined(void)
{
    int8_t   X_buf[2][INPUT_SIZE];  /* double buffer for input */
    uint8_t  image_bgr[640 * 640 * 3];
    bbox_t   detections[100];

    int buf_idx = 0;      /* ping-pong index */
    int frame_count = 0;

    /* Prime the pipeline: preprocess first frame */
    receive_image(image_bgr);
    preprocess_image(image_bgr, X_buf[0]);

    /* Start accelerator on first frame */
    run_accelerator_async(X_buf[0]);  /* non-blocking start */
    frame_count = 1;

    while (1) {
        buf_idx = frame_count & 1;

        /* ─── Stage 1: Preprocess NEXT frame (while IP runs) ─── */
        receive_image(image_bgr);
        preprocess_image(image_bgr, X_buf[buf_idx]);

        /* ─── Wait for Stage 2 to complete on PREVIOUS frame ─── */
        wait_accelerator_done();  /* poll CSR or wait IRQ */

        /* ─── Stage 3: Postprocess PREVIOUS frame ─── */
        int num_det = postprocess(detections, 100);
        output_results(detections, num_det);

        /* ─── Stage 2: Start accelerator on THIS frame ─── */
        run_accelerator_async(X_buf[buf_idx]);

        frame_count++;
    }
}

void run_accelerator_async(const int8_t *X_int8)
{
    memcpy((void *)DDR3_INPUT_BASE, X_int8, INPUT_SIZE);
    Xil_DCacheFlushRange(DDR3_INPUT_BASE, INPUT_SIZE);

    ACCEL_WR(CSR_NET_DESC_LO, DDR3_DESC_BASE);
    ACCEL_WR(CSR_NET_DESC_HI, 0);
    ACCEL_WR(CSR_LAYER_START, 0);
    ACCEL_WR(CSR_LAYER_END,   22);
    ACCEL_WR(CSR_CTRL, 0x1);  /* START — returns immediately */
}

void wait_accelerator_done(void)
{
    while (!(ACCEL_RD(CSR_STATUS) & 0x2));  /* wait done bit */
    ACCEL_WR(CSR_CTRL, 0x4);               /* clear irq */
}
```

---

# 9. PIPELINE 3 TẦNG & FPS PHÂN TÍCH

## 9.1. Pipeline timing diagram cụ thể

```
Frame#  │  0     10     20     30     40     50     60     70 (ms)
        │  │      │      │      │      │      │      │      │
        │
Frame 0 │ [GĐ1: 10ms][    GĐ2: 9.9ms   ][GĐ3: 12ms]
        │                                              ↓ output F0
Frame 1 │          [GĐ1: 10ms][    GĐ2: 9.9ms   ][GĐ3: 12ms]
        │                                                       ↓ output F1
Frame 2 │                   [GĐ1: 10ms][    GĐ2: 9.9ms   ][GĐ3: 12ms]
        │
        │  ← steady state: 1 frame every max(10, 9.9, 12) = 12 ms
        │  → Throughput = 83 FPS (MicroBlaze optimized)
        │  → Latency = 10 + 9.9 + 12 = 31.9 ms (3 frames)
```

## 9.2. Tổng kết FPS theo platform

```
╔══════════════════════════════════════════════════════════════════════════╗
║         TỔNG KẾT HIỆU NĂNG — YOLOv10n INT8 Accelerator V2             ║
╠════════════════════╤═════════╤═════════╤════════╤═══════════╤══════════╣
║                    │  T_GĐ1  │  T_GĐ2  │ T_GĐ3 │ Throughput│ Latency  ║
║    Platform        │  (ms)   │  (ms)   │ (ms)  │ (FPS)     │ (ms)     ║
╠════════════════════╪═════════╪═════════╪════════╪═══════════╪══════════╣
║ SEQUENTIAL (no pipeline)                                               ║
╠════════════════════╪═════════╪═════════╪════════╪═══════════╪══════════╣
║ x86 + FPGA        │   2.5   │   9.9   │  3.5   │  63 FPS   │ 15.9 ms  ║
║ MicroBlaze         │  20.0   │   9.9   │ 21.0   │  20 FPS   │ 50.9 ms  ║
║ MicroBlaze (opt)   │  10.0   │   9.9   │ 12.0   │  31 FPS   │ 31.9 ms  ║
║ ARM Zynq           │   6.5   │   9.9   │  7.0   │  43 FPS   │ 23.4 ms  ║
╠════════════════════╪═════════╪═════════╪════════╪═══════════╪══════════╣
║ PIPELINED (3-stage overlap)                                            ║
╠════════════════════╪═════════╪═════════╪════════╪═══════════╪══════════╣
║ x86 + FPGA        │   2.5   │  ★9.9   │  3.5   │ 101 FPS ✓│ 15.9 ms  ║
║ MicroBlaze         │ ★20.0   │   9.9   │★21.0   │  48 FPS ✗│ 50.9 ms  ║
║ MicroBlaze (opt)   │  10.0   │   9.9   │★12.0   │  83 FPS  │ 31.9 ms  ║
║ ARM Zynq           │   6.5   │  ★9.9   │  7.0   │ 101 FPS ✓│ 23.4 ms  ║
╠════════════════════╪═════════╪═════════╪════════╪═══════════╪══════════╣
║ ★ = bottleneck stage for that platform                                 ║
╚════════════════════╧═════════╧═════════╧════════╧═══════════╧══════════╝

KẾT LUẬN:
  ✓ x86 + FPGA (PCIe):   101 FPS — bottleneck = HW accelerator (9.9ms)
  ✓ ARM Zynq + PL:       101 FPS — bottleneck = HW accelerator (9.9ms)
  ~ MicroBlaze (opt):     83 FPS — bottleneck = CPU postprocess (12ms)
  ✗ MicroBlaze (basic):   48 FPS — bottleneck = CPU pre/post (20-21ms)
```

---

# 10. CÁC KỊCH BẢN TRIỂN KHAI

## 10.1. Kịch bản A: Virtex-7 + MicroBlaze (Pure FPGA)

```
Ưu điểm: Standalone, không cần PC
Nhược: CPU yếu → bottleneck ở GĐ1/GĐ3

Flow:
  ① Power on → MicroBlaze boot từ Flash
  ② Load weights + descriptors vào DDR3
  ③ Camera → MicroBlaze preprocess → DDR3
  ④ Start IP → IP compute → P3/P4/P5 in DDR3
  ⑤ MicroBlaze postprocess → UART/VGA output

FPS đạt được: 48-83 FPS (tùy optimization level)
```

## 10.2. Kịch bản B: Virtex-7 + Host PC qua PCIe

```
Ưu điểm: CPU mạnh → GĐ1/GĐ3 rất nhanh
Nhược: Cần PC, PCIe latency

Flow:
  ① PC load weights/descriptors qua PCIe → DDR3
  ② PC preprocess ảnh (OpenCV, AVX2) → ghi X_int8 qua PCIe → DDR3
  ③ PC ghi CSR.start qua PCIe MMIO
  ④ IP compute → P3/P4/P5 in DDR3
  ⑤ PC đọc P3/P4/P5 qua PCIe → postprocess trên PC

FPS đạt được: 101 FPS ✓
```

## 10.3. Kịch bản C: Virtex-7 + External ARM (Zynq hoặc STM32)

```
Ưu điểm: ARM đủ mạnh, không cần PC
Nhược: Cần board thêm, giao tiếp

Flow:
  ① ARM preprocess ảnh → gửi X_int8 qua SPI/UART/AXI → DDR3
  ② ARM ghi CSR.start
  ③ IP compute
  ④ ARM đọc P3/P4/P5 → postprocess

FPS đạt được: ~101 FPS (nếu ARM đủ mạnh)
```

## 10.4. So sánh kịch bản

```
┌──────────────┬──────────┬──────────┬──────────┬────────┐
│ Kịch bản     │ Complexity│ Cost     │ FPS      │ Suitable│
├──────────────┼──────────┼──────────┼──────────┼────────┤
│ A: Pure FPGA │ Medium   │ Low      │ 48-83    │ Demo   │
│ B: + Host PC │ High     │ High     │ 101+ ✓   │ Lab    │
│ C: + Ext ARM │ Medium   │ Medium   │ 101+ ✓   │ Deploy │
└──────────────┴──────────┴──────────┴──────────┴────────┘
```

---

# 11. VIVADO BLOCK DESIGN SETUP

## 11.1. Tạo project trong Vivado

```
Step 1: File → New Project → RTL Project
  Target: xc7vx690tffg1761-2 (hoặc board cụ thể)

Step 2: Add IP Cores:
  ① MicroBlaze (200 MHz, 128KB local memory)
  ② MIG DDR3 (DDR3-1600, 64-bit, 800 MHz)
  ③ AXI Interconnect (2 masters, 2-3 slaves)
  ④ AXI UART Lite (debug, 115200 baud)
  ⑤ accel_top (custom IP — your design)
  ⑥ Concat + xlslice for IRQ

Step 3: Connect:
  MicroBlaze M_AXI_DP → AXI Interconnect S0
  accel_top m_axi     → AXI Interconnect S1
  AXI Interconnect M0 → MIG DDR3
  AXI Interconnect M1 → accel_top s_axil (CSR)
  AXI Interconnect M2 → AXI UART Lite
  accel_top.irq        → MicroBlaze interrupt

Step 4: Address Map:
  DDR3:      0x0000_0000 — 0x3FFF_FFFF (1 GB)
  accel_top: 0x4400_0000 — 0x4400_0FFF (4 KB CSR)
  UART:      0x4060_0000 — 0x4060_FFFF

Step 5: Generate Bitstream → Export Hardware → Launch SDK
```

## 11.2. IP Packaging (accel_top as Vivado IP)

```
Step 1: Tools → Create and Package IP → Package Current Project
Step 2: Set interface:
  - s_axil: AXI4-Lite Slave (CSR)
  - m_axi:  AXI4 Master (DMA to DDR3)
  - clk:    Associated clock (200 MHz)
  - rst_n:  Active-low reset
  - irq:    Interrupt output

Step 3: Set address space:
  - s_axil: 4 KB (0x000 — 0xFFF)
  - m_axi:  Full 40-bit address range (DDR3 access)

Step 4: Package and add to IP repository
Step 5: Use in Block Design as standard IP
```

---

_Tài liệu này mô tả đầy đủ quy trình inference end-to-end từ ảnh input đến bounding box output,
cùng với hướng dẫn chi tiết cách nhúng lên board Virtex-7 và giao tiếp CPU ↔ DDR3 ↔ IP._
