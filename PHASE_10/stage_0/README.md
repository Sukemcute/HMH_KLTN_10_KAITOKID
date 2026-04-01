# STAGE 0: Nghiên Cứu & Đặc Tả (COMPLETED)

> Tất cả tài liệu nghiên cứu đã hoàn thành. Stage này là REFERENCE — không cần tạo code mới.

## Tài liệu nguồn (nằm trong các folder khác)

### Phân tích Model
| # | File | Đường dẫn | Nội dung |
|---|------|-----------|----------|
| 0.1 | MODEL_FORWARD_FLOW.md | `SW_KLTN/` | Luồng CPU → HW → CPU |
| 0.2 | MODEL_LAYERS_INT8_FLOW.md | `SW_KLTN/` | Trace INT8 dtype/shape L0-L22 |
| 0.3 | MODEL_BLOCKS_INT8_DETAIL.md | `SW_KLTN/` | 8 block types chi tiết |
| 0.4 | MODEL_LAYER_DEPENDENCIES.md | `SW_KLTN/` | Skip connections, barriers |

### Đặc tả Phần cứng (PHASE 0 Freeze)
| # | File | Đường dẫn | Nội dung |
|---|------|-----------|----------|
| 0.5 | 01_primitive_matrix.md | `SW_KLTN/PHASE0/` | 10 primitive P0-P9 |
| 0.6 | 02_layer_mapping.md | `SW_KLTN/PHASE0/` | Layer → Primitive mapping |
| 0.7 | 03_quant_policy.md | `SW_KLTN/PHASE0/` | INT8 quantization rules |
| 0.8 | 04_layout_addressing.md | `SW_KLTN/PHASE0/` | Banking, row slot, addressing |
| 0.9 | 05_descriptor_spec.md | `SW_KLTN/PHASE0/` | NET/LAYER/TILE descriptors |
| 0.10 | 06_execution_semantics.md | `SW_KLTN/PHASE0/` | Last-pass, barriers, PSUM/ACT |
| 0.11 | 07_golden_python_plan.md | `SW_KLTN/PHASE0/` | Golden Python design |
| 0.12 | 08_rtl_mapping_plan.md | `SW_KLTN/PHASE0/` | RTL module hierarchy |

### Kiến trúc & Strategy
| # | File | Đường dẫn | Nội dung |
|---|------|-----------|----------|
| 0.13 | HW_MAPPING_RESEARCH.md | `SW_KLTN/` | Primitive → RTL mapping |
| 0.14 | HW_ARCHITECTURE_V2_100FPS.md | `SW_KLTN/` | V2: LANES=32, Dual-RUNNING |
| 0.15 | HW_ACCELERATION_IMPL_FLOW.md | `SW_KLTN/` | 5-phase implementation |
| 0.16 | RTL_MODULE_SPEC.md | `SW_KLTN/` | 35+ RTL module specs |
| 0.17 | BUILD_STRATEGY.md | `SW_KLTN/PHASE_3/` | Bottom-up build order |
| 0.18 | TONG_HOP_NGHIEN_CUU.md | Root | Tổng hợp 11,561 dòng |

### Tổng hợp chiến lược RTL
| # | File | Đường dẫn | Nội dung |
|---|------|-----------|----------|
| 0.19 | RTL_BUILD_STRATEGY_FINAL.md | `HW_ACCEL/` | Master checklist + 10 golden rules |

## Status: ✅ COMPLETE — 19/19 documents
