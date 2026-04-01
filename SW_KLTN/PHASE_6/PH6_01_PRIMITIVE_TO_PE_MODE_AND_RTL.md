# PH6-01 — Primitive (HW_MAPPING) ↔ `pe_mode_e` ↔ file RTL PHASE_3

Nguồn primitive: `HW_MAPPING_RESEARCH.md` §2, §5, §8.  
Nguồn enum: `PHASE_3/packages/accel_pkg.sv` (`PE_RS3`, `PE_OS1`, `PE_DW3`, `PE_DW7`, `PE_MP5`, `PE_GEMM`, `PE_PASS`).

---

## Bảng ánh xạ chính

| ID | Primitive (mapping) | `pe_mode_e` (gợi ý) | Module RTL PHASE_3 chính | Ghi chú |
|----|---------------------|---------------------|---------------------------|---------|
| P0 | RS_DENSE_3x3 | `PE_RS3` | `window_gen`, `pe_cluster` (pe_unit, column_reduce), `ppu` | Stride/pad trong `layer_desc`; multipass `num_cin_pass` / `num_k_pass`. |
| P1 | OS_1x1 | `PE_OS1` | Cùng datapath; `window_gen` 1×1 | Projection, SPPF cv1/cv2. |
| P2 | DW_3x3 | `PE_DW3` | `pe_cluster` (bỏ/tắt column_reduce theo mode DW) | SCDown nhánh depthwise. |
| P3 | MAXPOOL_5x5 | `PE_MP5` | `comparator_tree` (+ địa chỉ từ `window_gen` 5×5 nếu hỗ trợ) | Không qua PPU requant (INT8 max). |
| P4 | MOVE | `PE_PASS` hoặc FSM-only | `tensor_dma` + GLB; không MAC | Lưu skip / copy buffer. |
| P5 | CONCAT | — (không phải mode PE) | `router_cluster` + có thể `ppu` (requant domain) | QConcat L12/L15/L18/L21. |
| P6 | UPSAMPLE_NEAREST | `PE_PASS` + addr | `swizzle_engine` | L11, L14. |
| P7 | EWISE_ADD | PPU `ewise_en` | `ppu` | Residual nếu dùng. |
| P8 | DW_7x7_MULTIPASS | `PE_DW7` | `pe_cluster` + `tile_fsm` last_pass | 3 pass kernel rows → PSUM rồi PPU ở pass cuối. |
| P9 | GEMM_ATTN_BASIC | `PE_GEMM` | `pe_cluster` (MAC) + **mở rộng điều khiển** | QPSA: GEMM 400×400 @ 20×20; softmax LUT — **rủi ro lớn**, xem PH6-05. |

---

## Luồng dữ liệu theo primitive (đối chiếu HW_MAPPING §5)

| Primitive | Đọc từ | Qua | Ghi vào |
|-----------|--------|-----|---------|
| P0/P1 | glb_input + glb_weight | window_gen → PE → (column_reduce) → PPU | glb_output / DDR |
| P2 | glb_input + glb_weight (per-ch) | window_gen → PE (DW) → PPU | glb_output |
| P3 | glb_input | window/comparator | glb_output (ACT INT8) |
| P5 | hai bank / DDR skip | router (interleave channel) | glb_output |
| P6 | glb hoặc DDR | swizzle (dup addr) | glb/DDR |
| P8 | glb | PE DW multipass, PSUM namespace | PPU last pass → INT8 |
| P9 | glb | nhiều vòng OS_1x1 + GEMM + LUT | PPU |

---

## Việc compiler phải ghi vào descriptor

- **`template_id`** trong `layer_desc` → chọn `pe_mode_e`.  
- **`post_profile_id`** → bias / act / quant_mode cho `ppu`.  
- **`router_profile_id`** → nguồn A/B cho CONCAT, broadcast.  
- **`tile_flags`** → `hold_skip`, `barrier_before/after`, `need_swizzle`, spill.

*Tài liệu chi tiết layer: `HW_MAPPING_RESEARCH.md` §3; output P3/P4/P5: PH6-03.*
