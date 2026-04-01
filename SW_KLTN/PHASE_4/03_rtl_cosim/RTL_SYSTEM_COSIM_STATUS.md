# RTL hệ thống vs `tb_golden_check` — trạng thái & việc cần làm

## Debug console (macro `ACCEL_DEBUG`, xvlog `-d ACCEL_DEBUG`)

RTL có các **checkpoint** `$display` (chỉ khi compile với macro `ACCEL_DEBUG`):

| Tag            | Module                 | Nội dung ngắn                                                    |
| -------------- | ---------------------- | ---------------------------------------------------------------- |
| `[CHK-CTRL]`   | `controller_system`    | CSR start, `sched_inference_complete`, `DF_DONE`, IRQ            |
| `[CHK-FETCH]`  | `desc_fetch_engine`    | PARSE_NET / PARSE_LAYER, DISPATCH tile (giới hạn log), `DF_DONE` |
| `[CHK-GS]`     | `global_scheduler`     | layer descriptor, tile vào GS (throttle), `LAYER_COMPLETE`       |
| `[CHK-SC-AXI]` | `supercluster_wrapper` | hoàn tất ghi AXI (BRESP), throttle                               |
| `[CHK-TILE]`   | `tile_fsm`             | mỗi subcluster: ISSUE tile (mặc định 64 tile đầu + mỗi 1024)     |

**Cosim:** trong `PHASE_4/03_rtl_cosim/run_cosim.tcl` mặc định `set ACCEL_DBG_DEFINE 1` → gọi `xvlog -sv -d ACCEL_DEBUG ...` (Vivado **không** dùng `+define+...` trên dòng lệnh `xvlog`). Để tắt log: `set ACCEL_DBG_DEFINE ""` rồi compile lại.

**Nếu `source run_cosim.tcl` báo không chạy được `xvlog`:** mở **Vivado Tcl Shell** (hoặc Tcl console sau khi `source` script Vivado), hoặc set biến môi trường `XILINX_VIVADO` trỏ tới thư mục cài Vivado (script sẽ gọi `%XILINX_VIVADO%/bin/xvlog.exe` trên Windows).

**Giảm log tile FSM (tùy chọn):** thêm trước khi compile module `tile_fsm`: `` `define ACCEL_TILE_LOG_MAX 32 `` (hoặc số khác) — xem `tile_fsm.sv`.

---

## Sửa bổ sung: `accel_top` — 4 SC dùng chung AXI write → deadlock

**Triệu chứng:** Sau vài tile, sim ~10M cycle không `sched_inference_complete` / IRQ; log dừng sau 1–2 `DISPATCH`.

**Nguyên nhân:** Cả 4 `supercluster_wrapper` nối cùng `m_axi_awready` / `m_axi_wready` / `m_axi_bvalid`. Chỉ SC0 được mux lên bus nhưng SC1–3 vẫn thấy handshake của giao dịch khác → FSM ghi kẹt, GS/`tile_desc_ready` không tiến.

**Đã sửa:** arbiter ghi (`axiw_busy` / `axiw_grant`) + chỉ SC đang được grant mới thấy `awready`/`wready`/`bvalid` tương ứng.

**Bổ sung (đọc DDR):** `m_axi_rready` trước đây luôn bằng `ctrl_axi_rready` từ `desc_fetch_engine` — ngoài các beat đọc descriptor thì `axi_rready` thường = 0 → **kênh R của SC DMA bị kẹt** dù `AR` đã mux tới SC. Đã thêm `ctrl_rd_busy` (giữ quyền đọc tới `RLAST`) và arbiter đọc cho SC (`axir_busy` / `axir_grant`) + gate `arready`/`rvalid`/`rlast` giống kênh ghi.

---

## Sửa bổ sung (2026-03): `DF_PARSE_LAYER` dùng `tile_total` cũ → bỏ qua tile layer đầu

**Triệu chứng cosim:** `PARSE_LAYER idx=0 num_tiles=100` nhưng vài chu kỳ sau đã `PARSE_LAYER idx=1`; `DISPATCH #0` có `layer_idx=1`; timeout 10M cycle; `OUTPUT_BASE` toàn 0.

**Nguyên nhân:** `nstate` trong `always_comb` dùng `tile_total > 0`, trong khi `tile_total <= ld.num_tile_hw` chỉ cập nhật ở **cùng** cạnh clock. `nstate` thấy **giá trị cũ** (0 ở layer đầu) → chuyển thẳng `DF_NEXT_LAYER` → **không fetch tile layer 0**; layer sau dùng `tile_total` còn sót → lệch với bảng tile → dễ kẹt handshake.

**Đã sửa:** nhánh `DF_PARSE_LAYER` dùng `layer_desc_t'(desc_buf...).num_tile_hw` (combinational từ buffer) để chọn `DF_FETCH_TILE_AR` vs `DF_NEXT_LAYER`.

---

## Sửa bổ sung (2026-03): `supercluster_wrapper` — GS kẹt `WAIT_ACCEPT` / timeout 10M cycle

**Triệu chứng:** Cosim chạy qua nhiều layer rồi dừng log; `TIMEOUT after 10M cycles`; `STATUS` busy; không thấy `[CHK-FETCH] DF_DONE`.

**Nguyên nhân:** `tile_accept` nối thẳng `arb_tile_consumed`. `local_arbiter` chỉ tạo `tile_consumed` khi có **sub `ROLE_IDLE`** để gán `FILLING`. Khi cả 4 sub đều bận (RUNNING / DRAINING / …), SC không bao giờ `accept` → `global_scheduler` kẹt `GS_WAIT_ACCEPT` → `tile_desc_ready` không lên → `desc_fetch` không đi tiếp → không tới `DF_DONE`.

**Đã sửa:** thêm **ingress 1 tile** (`ingress_valid` / `ingress_desc`): GS handshake `tile_accept = tile_valid && !ingress_valid`; arbiter lấy tile từ `ingress_*` (`tile_available = ingress_valid`). Như vậy mỗi SC có thể **ack ngay** khi buffer trống, không phụ thuộc sub đã rảnh.

**Bổ sung (cùng mục):** Không được xóa `ingress_valid` ngay lần `tile_consumed` **đầu tiên**. `local_arbiter` broadcast **cùng một** `tile_desc` lần lượt tới **NUM_SUBS** sub (mỗi lần gán sub idle → một xung `tile_consumed`). Nếu hạ `ingress_valid` sau consume đầu, `tile_available` tắt trước khi sub 1..NUM_SUBS-1 nhận descriptor → **kẹt**, thường lộ ngay sau `LAYER_COMPLETE` / layer mới. Cách đúng: đếm `ingress_disp_left` = `NUM_SUBS` khi latch từ GS; mỗi `arb_tile_consumed` giảm; chỉ khi về 0 mới `ingress_valid <= 0`.

**Lưu ý:** Đây là **độ sâu pipeline = 1 tile / SC** giữa GS và arbiter; backpressure vẫn đúng khi `ingress_valid=1` và chưa đủ NUM_SUBS lần consume.

---

## Sửa bổ sung: `desc_fetch_engine` — `tile_total` sai → không hề fetch tile

**Triệu chứng:** `PERF_CYCLES` ~125, `OUTPUT_BASE` toàn 0, log giống hệt trước khi sửa `supercluster` / descriptor.

**Nguyên nhân:** `tile_total` lấy bằng slice cố định `desc_buf[139:128]`, **không khớp** vị trí field `num_tile_hw` trong `desc_pkg::layer_desc_t` (và `generate_descriptors.py`). Hệ quả **`tile_total` thường = 0** → FSM bỏ qua mọi tile (`DF_PARSE_LAYER` → `DF_NEXT_LAYER` ngay) → **không tile nào tới SuperCluster** → không có DMA ghi.

**Đã sửa:** `DF_PARSE_NET` / `DF_PARSE_LAYER` dùng cast `net_desc_t'(...)` / `layer_desc_t'(...)` và gán `tile_total <= ld.num_tile_hw`, `tile_table_addr <= ld.tile_table_offset`.

**Bạn phải:** xóa `xsim.dir`, **compile lại** (`source run_cosim.tcl`), `xelab` lại `tb_golden_check`, rồi `xsim`.

---

## Đã xử lý trong repo (bring-up đường ghi DDR)

### 1. `supercluster_wrapper.sv` (PHASE_3/07_system)

- Trước đây: `m_axi_wvalid = 0`, `m_axi_wdata = 0` → **không bao giờ ghi DDR**.
- Nay: FSM **AW → W (1 beat, WLAST=1) → B** và **`dma_wr_done`** xung theo `bvalid/bready` tới đúng `tile_fsm`.
- Dữ liệu ghi hiện là **pattern stub** (địa chỉ + seq), **không** phải tensor P3/P4/P5 từ MAC/PPU.

### 2. `local_arbiter.sv`

- Trước đây: chỉ cấp **ghi** cho role `DRAINING` (sau `tile_done`), trong khi `dma_wr` xảy ra ở `TILE_SWIZZLE_STORE` khi vẫn `RUNNING` → **không bao giờ được grant ghi**.
- Nay: ưu tiên **`TILE_SWIZZLE_STORE` + `sub_dma_wr_req`** trước FILLING/DRAINING.

### 3. `generate_descriptors.py`

- `tile_flags`: bật **bit 5 (`need_spill`)** để `tile_fsm` kéo `dma_wr_req` trong `SWIZZLE_STORE`.
- `dst_off`: byte address = `output_arena_base (0x02000000) + global_tile_seq * 32` để **không ghi đè** vùng descriptor tại `0x0`.
- **Lưu ý:** Bố cục `+ tile * 32` chỉ phục vụ **chứng minh kênh AXI write**; **không** khớp thứ tự flatten NCHW của `golden_P3/P4/P5.hex`.

### 4. `subcluster_wrapper.sv` (05_integration) — chưa dùng trong `accel_top` hiện tại

- Vẫn có `ext_wr_valid = 0`, `ext_wr_data = 0`. Khi chuyển sang full compute unit, cần nối **swizzle / GLB output → AXI write**.

---

## Vì sao `tb_golden_check` vẫn có thể FAIL so sánh golden

| Hạng mục                      | Trạng thái                                                                                                   |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Ghi DDR tại `OUTPUT_BASE`     | Có (sau regenerate `desc_*.hex` + sim lại).                                                                  |
| Nội dung = golden P3/P4/P5    | **Chưa** — thiếu pipeline: đủ layer weight, MAC, PPU đúng param, swizzle/DMA đa beat, map địa chỉ đúng NCHW. |
| Chỉ nạp `weights_L0` trong TB | Đủ cho smoke control; **không** đủ inference 23 layer.                                                       |

---

## Việc bạn cần làm sau khi pull thay đổi

1. **Regenerate descriptor**

   ```bash
   cd PHASE_4
   python 01_export/generate_descriptors.py --output 02_golden_data/
   ```

2. Chạy lại Vivado / xsim: `source .../run_cosim.tcl` → `tb_golden_check`.

3. Kỳ vọng hợp lý **bước này**: vùng `OUTPUT_BASE` **không còn toàn 0** (có pattern stub); so file golden **vẫn có thể mismatch** cho đến khi hoàn thiện compute + map bộ nhớ.

---

## Lộ trình để PASS bit-exact (đề xuất)

Chi tiết từng bước (B0–B6), mốc kiểm tra và TB weight: xem **`PHASE_B_BITEXACT_ROADMAP.md`** (cùng thư mục).

Tóm tắt:

1. Nạp **đủ weight/bias** theo `weight_arena_base` + descriptor (hoặc mở rộng TB).
2. Thay stub `supercluster_wrapper` (chỉ `tile_fsm`) bằng **`subcluster_wrapper`** + sửa **`ext_wr_data/valid`** từ swizzle/GLB.
3. **`dst_off` / tile table**: tính theo layout tensor thật (NCHW, multi-beat, alignment) cho từng layer; riêng P3/P4/P5 phải trùng cách export `golden_*.hex`.
4. **`inference_complete`**: chỉ báo DONE khi **ghi xong output cuối**, không chỉ `fetch_all_done` (xem `COSIM_DEEP_DIVE_AND_FIXES.md`).

---

_Tài liệu này bổ sung `COSIM_DEEP_DIVE_AND_FIXES.md` sau các sửa RTL/Python ở trên._
