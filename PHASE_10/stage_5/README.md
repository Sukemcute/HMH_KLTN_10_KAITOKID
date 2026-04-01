# STAGE 5: Subcluster — 1 Phần Cứng Dùng Chung

> **Đây là bước QUAN TRỌNG NHẤT**: Chứng minh 1 phần cứng CỐ ĐỊNH có thể chạy
> TẤT CẢ primitive modes bằng cách thay đổi descriptor.

## Nguyên tắc cốt lõi

```
1 subcluster_datapath = 1 phần cứng CỐ ĐỊNH
  ├── GLB banks (input ×3, weight ×3, output ×4)
  ├── Address generators (input, weight, output)
  ├── Window generator (K=1,3,5,7)
  ├── PE cluster (3×4×32 = 384 MACs)
  ├── PPU (bias + requant + ReLU + clamp)
  ├── Router cluster (RIN/RWT/RPS + bypass)
  ├── Swizzle engine (upsample/concat/move)
  └── tile_fsm + shadow_reg_file (CONTROL)

Descriptor thay đổi config → cùng HW chạy:
  PE_RS3: Conv 3×3    (L0,L1,L3,L17, QC2f nội bộ)
  PE_OS1: Conv 1×1    (QC2f cv1/cv2, SCDown cv1)
  PE_DW3: DW Conv 3×3 (SCDown cv2)
  PE_DW7: DW Conv 7×7 (QC2fCIB DW7)
  PE_MP5: MaxPool 5×5 (SPPF)
  PE_PASS: Upsample/Concat/Move
```

## Kiến trúc kết nối

```
                    ┌───────────────────────────┐
                    │      tile_fsm              │
                    │  (đọc descriptor →         │
                    │   điều khiển toàn bộ)      │
                    └────────┬──────────────────┘
                             │ control signals
         ┌───────────────────┼───────────────────────┐
         ▼                   ▼                       ▼
  ┌──────────┐      ┌──────────────┐         ┌──────────┐
  │shadow_reg│      │  GLB Banks   │         │  Swizzle │
  │  file    │      │ IN:3 WT:3    │         │  Engine  │
  │(cfg latch)│     │ OUT:4        │         │(layout   │
  └────┬─────┘      └──────┬───────┘         │transform)│
       │ config             │ data            └────┬─────┘
       ▼                    ▼                      │
  ┌──────────┐      ┌──────────────┐               │
  │ Addr Gen │─────►│   Router     │               │
  │(IN/WT/OUT)│     │  Cluster     │               │
  └──────────┘      │(RIN/RWT/RPS)│               │
                    └──────┬───────┘               │
                           │ routed data           │
                    ┌──────▼───────┐               │
                    │  Window Gen  │               │
                    │ (K=1,3,5,7)  │               │
                    └──────┬───────┘               │
                           │ taps                  │
                    ┌──────▼───────┐               │
                    │  PE Cluster  │               │
                    │  3×4×32      │               │
                    │+column_reduce│               │
                    │+comp_tree    │               │
                    └──────┬───────┘               │
                           │ psum/pool             │
                    ┌──────▼───────┐               │
                    │     PPU      │──────────────►│
                    │bias+requant  │    output     │
                    │+ReLU+clamp   │               │
                    └──────────────┘               │
                                          GLB OUT ◄┘
```

## Files

```
stage_5/
├── rtl/
│   ├── subcluster_datapath.sv   ← MAIN: tích hợp tất cả modules (370+ dòng)
│   ├── pe_cluster.sv            ← 3×4 PE array + column_reduce + comparator_tree
│   ├── shadow_reg_file.sv       ← Config register snapshot từ descriptor
│   └── tile_fsm.sv              ← Tile execution FSM (276 dòng)
├── tb/
│   └── tb_subcluster_modes.sv   ← Test 5 pe_modes trên CÙNG 1 phần cứng
└── sim/
    └── compile_all.do
```

## 5 Tests (cùng 1 DUT, khác descriptor)

| Test | pe_mode | Mô phỏng | Config thay đổi | Expected |
|------|---------|-----------|----------------|----------|
| 1 | PE_RS3 | Conv 3×3 (L0) | kh=3,kw=3,stride=2,ReLU | ≥99.9% |
| 2 | PE_OS1 | Conv 1×1 (QC2f cv1) | kh=1,kw=1,stride=1,ReLU | ≥99% |
| 3 | PE_DW3 | DW Conv 3×3 (SCDown) | per-channel,stride=2,ReLU | ≥99.9% |
| 4 | PE_MP5 | MaxPool 5×5 (SPPF) | no PPU, comparator only | 100% |
| 5 | PE_PASS | Upsample 2× (L11) | swizzle active, no PE | 100% |

## Dependencies

```
Cần compile TẤT CẢ modules từ stage_2 + stage_4:
  stage_2/packages/     → accel_pkg, desc_pkg, csr_pkg
  stage_2/01-06/rtl/    → dsp_pair, pe_unit, column_reduce, comparator_tree, silu_lut, ppu
  stage_4/01-03/rtl/    → glb banks, addr_gen, window_gen, router, swizzle, metadata_ram
  stage_5/rtl/          → pe_cluster, shadow_reg_file, tile_fsm, subcluster_datapath
```

## Cách chạy

```bash
cd D:/HMH_KLTN/PHASE_10/stage_5/sim
vivado -mode batch -source compile_all.do
```

## Ý nghĩa khi ALL 5 PASS

```
✓ CÙNG 1 phần cứng chạy Conv3x3 ĐÚNG
✓ CÙNG 1 phần cứng chạy Conv1x1 ĐÚNG
✓ CÙNG 1 phần cứng chạy DWConv3x3 ĐÚNG
✓ CÙNG 1 phần cứng chạy MaxPool5x5 ĐÚNG
✓ CÙNG 1 phần cứng chạy Upsample ĐÚNG
→ Phần cứng SẴN SÀNG chạy toàn bộ YOLOv10n layers
→ Chỉ cần gửi đúng sequence descriptors (Stage 6)
```
