# STAGE 2: RTL Compute Atoms — Build & Verify

> 6 compute atom modules = nền tảng tính toán của toàn bộ accelerator.
> Tất cả PHẢI pass 100% bit-exact trước khi ghép thành subcluster (Stage 5).

## Cấu trúc

```
stage_2/
├── packages/                    ← accel_pkg, desc_pkg, csr_pkg
├── 01_dsp_pair/rtl + tb/        ← 2-MAC DSP48E1 (NỀN TẢNG)
├── 02_pe_unit/rtl + tb/         ← 32-lane PE (16 DSP pairs)
├── 03_column_reduce/rtl + tb/   ← Sum 3 PE rows
├── 04_comparator_tree/rtl + tb/ ← MaxPool 25→1
├── 05_silu_lut/rtl + tb/        ← 256-entry activation LUT
├── 06_ppu/rtl + tb/             ← PPU: bias + requant + ReLU + clamp (CRITICAL)
└── sim/compile_all.do           ← Vivado build script
```

## Cách chạy

```bash
cd D:/HMH_KLTN/PHASE_10/stage_2/sim
vivado -mode batch -source compile_all.do
```

## Module Summary

| # | Module | RTL | TB | Key Tests | Pass Criteria |
|---|--------|-----|----|-----------|---------------|
| 01 | dsp_pair_int8 | 104L | ~450L | Exhaustive 65K, corners, 9-cycle acc, random 10K | 0 errors |
| 02 | pe_unit | 74L | ~585L | RS3/OS1/DW3 modes, 32-lane, timing | 0 errors |
| 03 | column_reduce | 49L | ~200L | Known pattern, overflow, random 100 | 0 errors |
| 04 | comparator_tree | 114L | ~300L | Boundaries, per-lane, random 200 | 0 errors |
| 05 | silu_lut | 31L | ~200L | Load/read, parallel, golden SiLU | 0 errors |
| 06 | ppu | 211L | ~934L | **Half-up rounding**, ReLU, ZP, stress 1000 | 0 errors |

## Dependency Graph

```
accel_pkg.sv → dsp_pair_int8.sv → pe_unit.sv
                                       ↓
desc_pkg.sv  → ppu.sv           column_reduce.sv
                                 comparator_tree.sv
                                 silu_lut.sv
```

## Tiếp theo: Khi ALL PASS → Stage 5 (Subcluster Integration)
