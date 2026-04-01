# STAGE 3: Primitive Engine Verification

> **Vai trò**: Verify rằng khi ghép Stage 2 atoms thành chuỗi tính toán primitive,
> kết quả toán học khớp Golden Python **bit-exact**.
>
> **QUAN TRỌNG**: Các engine ở đây là **golden computation verifiers**, KHÔNG phải phần cứng cuối.
> Phần cứng cuối = 1 subcluster (Stage 5) config qua descriptor.

## Tại sao cần Stage 3?

```
Stage 2: Verify TỪNG atom riêng lẻ
  dsp_pair: MAC đúng? ✓
  pe_unit:  32-lane đúng? ✓
  ppu:      rounding đúng? ✓

Stage 3: Verify CHUỖI atoms ghép lại = 1 primitive
  pe_unit × 3 + column_reduce + ppu = Conv3x3 đúng? ← STAGE 3 verify cái này
  pe_unit × 1 + ppu = Conv1x1 đúng?
  comparator_tree = MaxPool đúng?
  domain_align + concat = CONCAT đúng?

Stage 5: Verify subcluster thật (cùng HW, switch mode qua descriptor)
```

## 8 Primitive Engines

| # | Engine | Primitive | Atoms sử dụng | Dùng ở Layer |
|---|--------|-----------|----------------|-------------|
| P0 | conv3x3_engine | RS_DENSE_3x3 | 3×pe_unit + col_reduce + ppu | L0,1,3,17, QC2f nội bộ |
| P1 | conv1x1_engine | OS_1x1 | 1×pe_unit(OS1) + ppu | QC2f cv1/cv2, SCDown, SPPF |
| P2 | dwconv3x3_engine | DW_3x3 | 3×pe_unit(DW3) + ppu per-ch | SCDown cv2 (L5,7,20) |
| P3 | maxpool5x5_engine | MAXPOOL_5x5 | comparator_tree, no ppu | SPPF L9 (×3 cascade) |
| P5 | concat_engine | CONCAT | domain align requant | QConcat L12,15,18,21 |
| P6 | upsample_engine | UPSAMPLE_2X | address remap only | L11, L14 |
| P7 | ewise_add_engine | EWISE_ADD | domain align + add | QC2fCIB residual |
| P8 | dwconv7x7_engine | DW_7x7_MULTIPASS | 3×pe_unit, 3-pass, PSUM buf | QC2fCIB L22 |

## Cấu trúc

```
stage_3/
├── rtl/
│   ├── yolo_accel_pkg.sv        ← Package (import accel_pkg types)
│   ├── conv3x3_engine.sv        ← P0: 3 PE rows + col_reduce + PPU
│   ├── conv1x1_engine.sv        ← P1: 1 PE (broadcast weight) + PPU
│   ├── dwconv3x3_engine.sv      ← P2: 3 PE (per-channel) + PPU
│   ├── maxpool5x5_engine.sv     ← P3: comparator_tree, no PPU
│   ├── dwconv7x7_engine.sv      ← P8: 3 PE × 3 passes + PSUM + PPU
│   ├── concat_engine.sv         ← P5: domain align + channel interleave
│   ├── upsample_engine.sv       ← P6: address remap 2×
│   └── ewise_add_engine.sv      ← P7: domain align + add + requant
├── tb/
│   ├── tb_conv3x3_golden.sv     ← 4 tests (minimal, L0-style, large Cin, random)
│   ├── tb_conv1x1_golden.sv     ← 4 tests (QC2f cv1/cv2 style)
│   ├── tb_dwconv3x3_golden.sv   ← 3 tests (SCDown style)
│   ├── tb_maxpool5x5_golden.sv  ← 3 tests (SPPF cascade)
│   ├── tb_dwconv7x7_golden.sv   ← 3 tests (all-ones, single-ch, multi-ch ReLU)
│   ├── tb_concat_golden.sv      ← 3 tests (same scale, diff scale, L12-style)
│   ├── tb_upsample_golden.sv    ← 3 tests (minimal, multi-ch, L11-style)
│   └── tb_ewise_add_golden.sv   ← 4 tests (same scale, diff scale, saturation, L22)
├── sim/
│   └── compile_all.do           ← Compile + run all 8 engines
└── README.md                    ← (this file)
```

## Cách chạy

```bash
cd D:/HMH_KLTN/PHASE_10/stage_3/sim
vivado -mode batch -source compile_all.do
```

## Dependencies (từ stage_2)

```
stage_2/packages/accel_pkg.sv   ← pe_mode_e, act_mode_e types
stage_2/packages/desc_pkg.sv    ← post_profile_t for PPU config
stage_2/01_dsp_pair/rtl/        ← dsp_pair_int8 (used by pe_unit)
stage_2/02_pe_unit/rtl/         ← pe_unit (used by conv/dw engines)
stage_2/03_column_reduce/rtl/   ← column_reduce (used by conv3x3)
stage_2/04_comparator_tree/rtl/ ← comparator_tree (used by maxpool)
stage_2/05_silu_lut/rtl/        ← silu_lut (used by PPU)
stage_2/06_ppu/rtl/             ← ppu (used by conv/dw engines)
```

## Pass Criteria

| Engine | Expected Match | Max LSB | Source |
|--------|---------------|---------|--------|
| P0 Conv3x3 | ≥99.9% | ≤1 | L0 verification |
| P1 Conv1x1 | ≥99% | ≤2 | QC2f cv1/cv2 |
| P2 DWConv3x3 | ≥99.9% | ≤1 | SCDown cv2 |
| P3 MaxPool5x5 | 100% | 0 | SPPF L9 |
| P5 Concat | 100% | 0 | QConcat L12 |
| P6 Upsample | 100% | 0 | L11 |
| P7 EwiseAdd | 100% | 0 | QC2fCIB residual |
| P8 DWConv7x7 | ≥99.9% | ≤1 | QC2fCIB L22 |

## Sau khi ALL PASS → Stage 5 (Subcluster)

Stage 3 chứng minh: **mỗi phép tính primitive đúng khi atoms ghép lại**.
Stage 5 sẽ chứng minh: **cùng 1 phần cứng switch mode qua descriptor → tất cả primitives đều đúng**.
