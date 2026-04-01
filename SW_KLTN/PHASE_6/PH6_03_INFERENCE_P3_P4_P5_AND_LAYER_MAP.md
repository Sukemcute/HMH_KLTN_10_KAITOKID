# PH6-03 — Inference INT8: input → L0–L22 → P3, P4, P5

**Giả định:** Input đã là `X_int8 [1,3,640,640]` + metadata trên CPU (theo `MODEL_FORWARD_FLOW.md`). Accelerator thực hiện **L0–L22**; **L23 Qv10Detect** trên CPU.

---

## Đầu vào / đầu ra accelerator

| Tensor | Shape (batch=1) | Layer sinh ra |
|--------|-----------------|---------------|
| Input | `[3, 640, 640]` INT8 | Trước L0 (DDR) |
| **P3** | `[64, 80, 80]` INT8 | **L16** (sau QC2f) |
| **P4** | `[128, 40, 40]` INT8 | **L19** (sau QC2f) |
| **P5** | `[256, 20, 20]` INT8 | **L22** (sau QC2fCIB) |

---

## Bảng layer → primitive → ý nghĩa RTL

| L | Block | Primitive chính (rút gọn) | Ghi chú |
|---|-------|---------------------------|---------|
| 0 | Conv | P0 RS s2 | Đầu backbone |
| 1 | Conv | P0 RS s2 | |
| 2 | QC2f | P1+P0+P5+P1 | Bottleneck nội bộ |
| 3 | Conv | P0 RS s2 | |
| 4 | QC2f | P1+P0+P5+P1 | **Skip → L15** |
| 5 | SCDown | P1+P2+P5 | 2 nhánh → concat |
| 6 | QC2f | P1+P0+P5+P1 | **Skip → L12** |
| 7 | SCDown | P1+P2+P5 | |
| 8 | QC2f | P1+P0+P5+P1 | **Skip → L21** |
| 9 | SPPF | P1+P3×3+P5+P1 | Pool không PPU |
| 10 | QPSA | P1+P9+P1 | GEMM + softmax approx |
| 11 | Upsample | P6 | |
| 12 | QConcat | P5 | Cần F6 + F11 |
| 13 | QC2f | P1+P0+P5+P1 | **Skip → L18** |
| 14 | Upsample | P6 | |
| 15 | QConcat | P5 | Cần F4 + F14 |
| 16 | QC2f | P1+P0+P5+P1 | **= P3** |
| 17 | Conv | P0 RS s2 | PAN down |
| 18 | QConcat | P5 | F17 + F13 |
| 19 | QC2f | P1+P0+P5+P1 | **= P4** |
| 20 | SCDown | P1+P2 | |
| 21 | QConcat | P5 | F20 + F8 |
| 22 | QC2fCIB | P1+P8+… | **= P5** |

---

## Compiler / descriptor

Mỗi hàng trên được bung thành **một hoặc nhiều `tile_desc`** + **`layer_desc`** với `template_id`, `post_profile_id`, `router_profile_id`, offset DDR (`src_in_off`, `src_w_off`, `src_skip_off`, `dst_off`).

**Ba tensor ra:**

- Ghi DDR tại **3 vùng** (hoặc ping-pong `act0`/`act1`) với pointer trong `net_desc` / bảng output do toolchain định nghĩa.  
- CPU đọc P3/P4/P5 + `scale_3/4/5`, `zp_3/4/5` cho head.

---

*Chi tiết toán & buffer: `HW_MAPPING_RESEARCH.md` §3–4.*
