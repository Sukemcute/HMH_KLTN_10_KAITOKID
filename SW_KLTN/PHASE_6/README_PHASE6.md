# PHASE 6 — Tài liệu kiến trúc RTL (YOLOv10n INT8, mapping → PHASE_3)

**Mục đích:** Bộ spec **top-down** từ SoC boundary → submodule, liên kết **primitive** (`HW_MAPPING_RESEARCH.md`) với **file RTL thực tế** trong `SW_KLTN/PHASE_3/`.

**Nguyên tắc:** PHASE_6 **không nhân đôi** toàn bộ RTL; mã nguồn tham chiếu vẫn là **PHASE_3**. PHASE_6 chứa **thiết kế / ánh xạ / khoảng trống implement**.

## Danh sách tài liệu

| File | Nội dung |
|------|-----------|
| `PH6_00_RTL_HIERARCHY_TOP_DOWN.md` | Cây module từ `accel_top` xuống lá; kết nối bus & luồng dữ liệu |
| `PH6_01_PRIMITIVE_TO_PE_MODE_AND_RTL.md` | P0–P9 → `pe_mode_e` / datapath → file `.sv` |
| `PH6_02_SUBCLUSTER_INTERNAL_DATAPATH.md` | Một subcluster: GLB → window → PE → PPU → swizzle → DMA |
| `PH6_03_INFERENCE_P3_P4_P5_AND_LAYER_MAP.md` | L0–L22 → primitive → điểm xuất P3/P4/P5 |
| `PH6_04_SKIP_BUFFER_BARRIER_DESCRIPTOR.md` | HOLD skip ~900KB, barrier, descriptor fields |
| `PH6_05_RTL_GAPS_AND_VERIFICATION.md` | Khối trong mapping chưa tách file / cần mở rộng so PHASE_3 |
| `PH6_06_MODULE_CATALOG_PHASE3.md` | Bảng tra cứu từng file `.sv` PHASE_3 + vai trò + primitive liên quan |
| `PH6_07_RESEARCH_COMPUTE_BLOCKS_PRIMITIVE_TO_LAYER.md` | **Nghiên cứu chuyên sâu:** xây khối tính toán primitive→layer, pha triển khai, kim tự tháp kiểm thử, golden PHASE1 |

## Tham chiếu ngoài PHASE_6

- Mapping chi tiết layer: `SW_KLTN/HW_MAPPING_RESEARCH.md`
- Cây file: `SW_KLTN/PHASE_5/RTL_IP_DIRECTORY_TREE.txt`
- Kiến trúc V2 / FPS: `SW_KLTN/HW_ARCHITECTURE_V2_100FPS.md`
