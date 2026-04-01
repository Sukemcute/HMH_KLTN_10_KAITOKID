# Phân tích chuyên sâu: `tb_golden_check` timeout / mismatch — nguyên nhân & chỉnh sửa

Tài liệu này tóm tắt **root cause** đã xác minh trong RTL/TB và **các thay đổi đã áp dụng** (hoặc cần làm tiếp).

---

## 1. Chuỗi lỗi bạn thấy trên log

| Hiện tượng | Giải thích kỹ thuật |
|------------|---------------------|
| `VERSION = 0` | Đọc CSR sai chu kỳ: địa chỉ MMIO dùng cho decode **trễ một nhịp** so với handshake AR; dữ liệu đọc chỉ hợp lệ khi `mmio_re` bật nhưng TB lấy `rdata` ở pha **R** khi `mmio_re` đã tắt → trước đây `rdata` về 0 hoặc sai. |
| `TIMEOUT`, `IRQ` không lên | `status_done` chỉ lên khi `sched_inference_complete`. `global_scheduler` **không bao giờ** gán `inference_complete = 1` → hoàn tất inference không bao giờ báo về CSR. |
| `PERF_CYCLES = 0`, `TILES_DONE = 0` | `perf_cycle_cnt` chỉ tăng khi `status_busy`; nếu đọc STATUS sai hoặc `busy` không như mong đợi, log dễ gây hiểu nhầm. `perf_tile_done_cnt` trong RTL **chưa được cộng từ tile thật** (chỉ reset / không hook). |
| `RTL` toàn 0 ở `OUTPUT_BASE` | Vùng output DDR **chưa được DMA/SC ghi** (pipeline chưa chạy đúng hoặc chưa tới bước ghi). So với golden Python → mismatch 100%. |

---

## 2. Đã sửa trong repo (RTL)

### 2.1 `PHASE_3/07_system/rtl/accel_top.sv` — AXI-Lite đọc đúng

**Vấn đề:**

- `mmio_addr` dùng `mmio_addr_rd` đã đăng ký trong khi `mmio_re` là tín hiệu **cùng chu kỳ** với AR handshake → địa chỉ decode = giá trị **chu kỳ trước** (thường 0).
- `controller_system` chỉ drive `mmio_rdata` khi `mmio_re`; pha **R** (`rvalid` cao) thì `mmio_re` = 0 → `rdata` về 0 trước khi TB đọc.

**Cách làm:**

- `mmio_addr_mux`: khi `(arvalid && arready)` dùng **`s_axil_araddr`**; sau đó giữ `mmio_addr_rd` cho các chu kỳ tiếp theo nếu cần.
- **Latch** `mmio_rdata_o` vào `s_axil_rdata_latched` đúng chu kỳ AR handshake; `s_axil_rdata` xuất giá trị latched để ổn định trong khi `rvalid` cao.

### 2.2 `PHASE_3/07_system/rtl/controller_system.sv` — Kết thúc inference không bị treo

**Vấn đề:** `global_scheduler` chỉ xóa `inference_complete`, không bao giờ set 1.

**Cách làm (tạm thời, đủ để cosim không timeout vô hạn):**

- `sched_inference_complete = sched_inference_complete_gs | fetch_all_done`
- `fetch_all_done` từ `desc_fetch_engine` bật khi FSM vào `DF_DONE` (sau khi đã “đi hết” layer theo `layer_end` trong mô hình fetch).

**Hạn chế:** Đây là “**fetch descriptor xong**”, **chưa** chứng minh “**toàn bộ MAC + ghi DDR output xong**”. Khi GS/DMA hoàn chỉnh, nên chuyển sang: `inference_complete` chỉ bật khi **tile cuối + ghi bộ nhớ xong**, và tách khỏi `fetch_all_done`.

### 2.3 `global_scheduler.sv` — Việc cần làm sau (đề xuất)

Trong `global_scheduler`, bổ sung logic (một trong các hướng):

- Bật `inference_complete` một xung khi: đã xử lý xong **layer cuối** (`layer_id_reg == layer_end` từ CSR) **và** `tiles_dispatched` đạt `tiles_total` cho layer đó **và** không còn tile chờ từ fetch; **hoặc**
- Nhận `fetch_all_done` / `all_layers_done` từ controller và chỉ assert sau khi scheduler idle và scoreboard barrier (nếu có) ổn.

Hiện tại để **không đổi interface nhiều**, đã OR `fetch_all_done` ở `controller_system`.

---

## 3. Đã sửa trong repo (testbench)

### `PHASE_4/03_rtl_cosim/tb_golden_check.sv`

**Vấn đề:** `NET_DESC_BASE = 0` nhưng DDR tại `0x0` **không** được nạp `desc_net.hex` → engine đọc toàn 0 → không có tile hợp lệ / hành vi sai.

**Cách làm:** Sau khi nạp input/weight, gọi thêm:

- `desc_net.hex` → `DESC_BASE` (0x0)
- `desc_layers.hex` → `LAYER_TABLE_BASE` (0x100)
- `desc_tiles.hex` → `TILE_TABLE_BASE` (0x10000)

Địa chỉ này **khớp** `memory_map` trong `PHASE_4/01_export/generate_descriptors.py`.

**Lưu ý:** Chạy trước:

```bash
python PHASE_4/01_export/generate_descriptors.py --output PHASE_4/02_golden_data/
```

**CSR `LAYER_END`:** TB đang dùng `22`. Phải **khớp** `len(layers)-1` từ `quant_params`/descriptor thực tế. Nếu số layer export khác, chỉnh `CSR_LAYER_END` cho đúng.

---

## 4. Rủi ro còn lại (cần nghiên cứu tiếp)

### 4.1 Đồng bộ bitfield Python ↔ SystemVerilog

`generate_descriptors.py` dùng `pack_fields` (nối bit theo thứ tự field). RTL parse bằng `desc_buf[...]` cố định trong `desc_fetch_engine.sv` và `desc_pkg::` **packed struct**.

Nếu sau khi nạp descriptor vẫn sai hành vi: cần **so khớp từng field** (đặc biệt `layer_table_base`, `num_tile_hw`, `tile_table_offset`, `sc_mask`) giữa:

- `desc_fetch_engine.sv` (DF_PARSE_NET / DF_PARSE_LAYER)
- `generate_descriptors.py` (`build_*_descriptor`)

### 4.2 Thứ tự 256-bit trong file `.hex`

Python `descriptor_to_lines`: dòng 1 = 32 byte “thấp”, dòng 2 = 32 byte “cao” của 64-byte descriptor. RTL gán beat 0 vào `desc_buf[255:0]`, beat 1 vào `desc_buf[511:256]`. Nếu lệch thứ tự beat ↔ file, toàn bộ parse sai — cần một test nhỏ (một descriptor cố định, in ra hex từng beat).

### 4.3 `tensor_dma` không nằm trong `accel_top`

Comment đầu file nói có `tensor_dma` nhưng instance hiện **chưa** có. Nếu output P3/P4/P5 kỳ vọng đến từ DMA tập trung, cần **tích hợp** `tensor_dma` và arbiter AW/AR hoặc xác nhận đường ghi ra DDR nằm trong `supercluster_wrapper`.

### 4.4 Trọng số đầy đủ theo layer

TB mới chỉ nạp `weights_L0_conv.hex`. Các layer sau cần đúng vùng `weight_arena_base` + offset theo descriptor — nếu thiếu, compute/DMA có thể đọc 0 hoặc sai.

### 4.5 `perf_tile_done_cnt`

Trong `controller_system.sv`, thanh ghi `CSR_PERF_TILE_DONE` đọc `perf_tile_done_cnt` nhưng biến này **chưa được tăng** khi tile hoàn thành. Đề xuất: tăng khi `sched_layer_complete` hoặc khi mỗi SC báo `tiles_completed` tăng (tùy định nghĩa “tile done”).

---

## 5. Thứ tự kiểm tra đề nghị

1. `xvlog` / `xelab` `tb_accel_top` — kiểm tra đọc `VERSION` = `0x2026_0320` (smoke test PHASE_3).
2. `tb_golden_check` — sau timeout fix: có `IRQ` hoặc `STATUS` done trong thời gian hợp lý.
3. Waveform: `desc_fetch_engine` state, `axi_araddr`, `tile_valid`/`tile_accept`, `m_axi_awvalid` tại `OUTPUT_BASE`.
4. Khi output vẫn 0: trace từ `supercluster_wrapper` → AXI write.

---

*Tài liệu này bổ sung cho `SYSTEM_VERIFY_STEP4.md` và log debug cosim thực tế.*
