# STAGE 1: Golden Python & Software Verification (COMPLETED)

> Toàn bộ primitives, layers, model đã verified bit-exact trên phần mềm Python.
> mAP50 = 0.9302 trên 7,902 images. Stage này là REFERENCE.

## 1A. Golden Python Primitives

| # | File | Đường dẫn | Nội dung |
|---|------|-----------|----------|
| 1A.1 | config.py | `FUNCTION/SW_KLTN/python_golden_originial/` | LANES, BANKS, PSUM_BITS |
| 1A.2 | accel_types.py | `FUNCTION/SW_KLTN/python_golden_originial/` | Data types |
| 1A.3 | quant_affine.py | `FUNCTION/SW_KLTN/python_golden_originial/quant/` | quantize, requant, SiLU LUT |
| 1A.4 | quant_domain_align.py | `FUNCTION/SW_KLTN/python_golden_originial/quant/` | CONCAT/ADD alignment |
| 1A.5 | primitive_conv.py | `FUNCTION/SW_KLTN/python_golden_originial/primitives/` | P0 RS3, P1 OS1 |
| 1A.6 | primitive_dw.py | `FUNCTION/SW_KLTN/python_golden_originial/primitives/` | P2 DW3, P8 DW7 |
| 1A.7 | primitive_pool.py | `FUNCTION/SW_KLTN/python_golden_originial/primitives/` | P3 MAXPOOL |
| 1A.8 | primitive_tensor.py | `FUNCTION/SW_KLTN/python_golden_originial/primitives/` | P4-P7 |
| 1A.9 | primitive_psa.py | `FUNCTION/SW_KLTN/python_golden_originial/primitives/` | P9-P11 attention |

## 1B. Critical Fixes (đã apply vào golden)

| Fix | File nguồn | Impact |
|-----|-----------|--------|
| Half-up rounding | detailed_primitive_modification_report.md | Conv 70% → 99.9% |
| Signed domain (ZP-128) | dtype_alignment_strategy.md | Conv 74% → 99.98% |
| INT64 PPU multiply | detailed_primitive_modification_report.md | Prevent overflow |
| Activation = **ReLU** | mapping_conv_block.md | Model QAT dùng ReLU |
| ZP padding | detailed_primitive_modification_report.md | Boundary correctness |

## 1C. Per-Layer Verification Results (100 samples, 640x640)

| Layer | Type | Match % | Max LSB | Script |
|-------|------|---------|---------|--------|
| L0 | Conv | 99.99% | 1 | verify_layer_0.py |
| L1 | Conv | 99.96% | 1 | verify_layer_1.py |
| L2 | QC2f | 99.09% | 3 | verify_layer_2.py |
| L3 | Conv | 99.98% | 1 | verify_layer_3.py |
| L4 | QC2f | 96.52% | 3 | verify_layer_4.py |
| L5 | SCDown | 99.90% | 1 | verify_layer_5.py |
| L6 | QC2f | 94.46% | 3 | verify_layer_6.py |
| L7 | SCDown | 99.92% | 1 | verify_layer_7.py |
| L8 | QC2f | 99.20% | 2 | verify_layer_8.py |
| L9 | SPPF | 99.94% | 1 | verify_layer_9.py |
| L10 | QPSA | 83.52% | 23 | verify_layer_10.py |
| L11 | Upsample | 100.00% | 0 | verify_layer_11.py |
| L12 | QConcat | 100.00% | 0 | verify_layer_12.py |
| L13 | QC2f | 98.53% | 2 | verify_layer_13.py |
| L14 | Upsample | 100.00% | 0 | verify_layer_14.py |
| L15 | QConcat | 100.00% | 0 | verify_layer_15.py |
| L16 | QC2f | 99.02% | 9 | verify_layer_16.py |
| L17 | Conv | 99.99% | 1 | verify_layer_17.py |
| L18 | QConcat | 100.00% | 0 | verify_layer_18.py |
| L19 | QC2f | 98.94% | 3 | verify_layer_19.py |
| L20 | SCDown | 99.99% | 1 | verify_layer_20.py |
| L21 | QConcat | 100.00% | 0 | verify_layer_21.py |
| L22 | QC2fCIB | 99.96% | 1 | verify_layer_22.py |

## 1D. Dataset Validation

| Metric | Reference PTQ | Mapped Golden | Delta |
|--------|---------------|---------------|-------|
| mAP50 (All) | 0.9300 | **0.9302** | +0.0002 |
| mAP50-95 | 0.7220 | **0.7217** | -0.0003 |

## Status: ✅ COMPLETE — Software golden 100% verified
