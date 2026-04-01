# PH6-02 — Datapath nội bộ một Subcluster (theo mapping)

Mục tiêu: mô tả **chức năng từng khối nhỏ** và **thứ tự dữ liệu** để thực thi primitive P0–P9, khớp `HW_MAPPING_RESEARCH.md` §5–6.

---

## 1. tile_fsm — “bộ não pha”

**File:** `06_control/rtl/tile_fsm.sv`

| Pha | Hành động phần cứng (mapping) |
|-----|-------------------------------|
| PREFILL_WT | DMA đọc weight OIHW vào `glb_weight_bank`. |
| PREFILL_IN | DMA đọc activation (HWC) vào `glb_input_bank`. |
| PREFILL_SKIP | DMA đọc tensor skip (residual) nếu `tile_flags`. |
| RUN_COMPUTE | `glb_rd_en` + `pe_en`: MAC/pool theo `pe_mode`. |
| ACCUMULATE | Nhiều pass Cin/K — PSUM không qua PPU đến `last_pass`. |
| POST_PROCESS | `ppu_en`: bias + requant + SiLU + clamp. |
| SWIZZLE_STORE | `swizzle_engine` rồi DMA ghi `dst_off` (hoặc giữ GLB nếu HOLD_SKIP). |

---

## 2. shadow_reg_file — “ảnh chụp cấu hình”

**File:** `05_integration/rtl/shadow_reg_file.sv`

- Chốt `template_id` → `o_mode` (PE mode).  
- Chốt `q_in`, `q_out`, pad, stride, `num_*_pass` cho addr_gen và FSM.  
- Chốt `post_profile`, `router_profile` (router hiện có thể `'0` trong một số bản subcluster — cần nối đủ).

---

## 3. addr_gen_* + glb_* — banking (mapping §6)

| Quy tắc (nghiên cứu) | RTL |
|----------------------|-----|
| `bank_input = h mod 3` | `glb_input_bank` ×3 + logic trong addr_gen_input |
| `bank_output = out_row mod 4` | `glb_output_bank` ×4 |
| `row_slot`, `Q_in`, `Wblk` | `addr_gen_input.sv`, `addr_gen_output.sv` — trường `cfg_q_in`, `cfg_q_out` trong `layer_desc` |

**LANES:** `accel_pkg` — V2 dùng 32; mapping gốc viết 16 → đồng bộ với `pe_unit` packing.

---

## 4. window_gen

**File:** `04_data_movement/rtl/window_gen.sv`

- **P0/P1:** cửa sổ 3×3 hoặc 1×1 trên activation.  
- **P3:** mở rộng spec để 5×5 s=1 pad=2 (có thể tham số hóa kernel size trong layer_desc hoặc mode MP5).

---

## 5. pe_cluster

**File:** `05_integration/rtl/pe_cluster.sv`

- **pe_unit × 12:** mỗi unit × LANES MAC song song.  
- **column_reduce:** tích lũy theo Cin cho conv dày.  
- **comparator_tree:** max cho pool (P3).

**DW (P2/P8):** giảm hoặc bỏ column_reduce theo cấu hình mode; multipass P8 do FSM + `last_pass` điều khiển PPU.

---

## 6. ppu

**File:** `02_ppu/rtl/ppu.sv`

Thực hiện đúng chuỗi mapping §2.2: bias → `(acc * m_int) >>> shift` → SiLU LUT → clamp; hỗ trợ e-wise cho skip.

---

## 7. router_cluster

**File:** `04_data_movement/rtl/router_cluster.sv`

- **P5 CONCAT:** đọc lần lượt (hoặc song song) tensor A rồi B theo kênh; nếu scale khác → một nhánh qua PPU requant trước (mini-pipeline hoặc hai lần qua PPU).

---

## 8. swizzle_engine

**File:** `04_data_movement/rtl/swizzle_engine.sv`

- **P6:** nhân bản địa chỉ nguồn lên lưới 2H×2W.  
- Chuẩn bị layout ghi DDR (HWC).

---

## 9. Cổng ext_* ↔ tensor_dma

**File:** `05_integration/rtl/subcluster_wrapper.sv`

Handshake `ext_rd_*` / `ext_wr_*` là **ranh giới** giữa tile_fsm/DMA engine trong SC — đúng với mô hình “tensor post / move” (P4, P6) khi không dùng PE.

---

*Tích hợp đầy đủ: thay behavioral path trong `subcluster_wrapper` bằng nối dây các instance ở trên — xem `RTL_IP_DIRECTORY_TREE.txt` ghi chú.*
