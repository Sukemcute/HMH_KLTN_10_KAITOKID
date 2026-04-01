# PH6-04 — Skip tensor, barrier, descriptor (theo mapping)

Nguồn: `HW_MAPPING_RESEARCH.md` §4, §7.

---

## 1. Bảng HOLD / skip bắt buộc

| Tensor | Sinh | Tiêu thụ | Kích thước (INT8) |
|--------|------|----------|-------------------|
| F4 | L4 | L15 | 64×80×80 ≈ 409,600 B |
| F6 | L6 | L12 | 128×40×40 ≈ 204,800 B |
| F8 | L8 | L21 | 256×20×20 ≈ 102,400 B |
| F13 | L13 | L18 | 128×40×40 ≈ 204,800 B |

**Tổng ~921,600 B (~900 KiB)** — có thể vượt một GLB sub nhỏ → **spill DDR** (Option B trong `HW_ARCHITECTURE_V2_100FPS.md`) hoặc **phân mảnh** nhiều bank/SC.

---

## 2. Barrier (logical)

```
barrier_L12: L11_done ∧ F6_ready → start L12
barrier_L15: L14_done ∧ F4_ready → start L15
barrier_L18: L17_done ∧ F13_ready → start L18
barrier_L21: L20_done ∧ F8_ready → start L21
```

**RTL:** `barrier_manager` + `tile_flags` phát từ subcluster (`barrier_signal`, `barrier_signal_id`); `controller_system` tổng hợp scoreboard đọc MMIO.

---

## 3. Descriptor fields (gợi ý bổ sung so `desc_pkg`)

Các trường hiện có (`src_skip_tid`, `dst_tid`, `tile_flags`, …) cần gán nghĩa cố định:

| Semantic | Gợi ý |
|----------|--------|
| `hold_skip` / bit trong `tile_flags` | Output tile ghi vào vùng **skip arena** hoặc slot GLB cố định. |
| `barrier_before` | Tile concat chỉ chạy khi barrier grant. |
| `src_skip_off` | Địa chỉ DDR (hoặc offset) đọc nhánh skip thứ hai cho CONCAT. |

Compiler offline phải **không overlap** các vùng skip đồng thời sống.

---

## 4. CONCAT với domain khác scale

Theo mapping §5.5: nếu `scale_A ≠ scale_B`, nhánh cần **requant** qua PPU (hoặc pass PPU riêng) trước khi `router_cluster` ghép kênh — khai báo trong `layer_desc` / bảng quant per layer.

---

*Verify: golden Python `test_quant.py` + `test_model_forward.py` trong HW_MAPPING §9.*
