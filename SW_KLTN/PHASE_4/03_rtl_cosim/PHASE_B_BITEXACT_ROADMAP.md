# Phase B — Làm sao để cosim ra **đúng P3 / P4 / P5** (bit-exact với golden)

`tb_golden_check` **đã** so sánh đúng chỗ:

| Vùng DDR (byte) | Nội dung so |
|-----------------|-------------|
| `0x0200_0000` + 819 200 B | **P3** `[1,128,80,80]` |
| tiếp theo + 409 600 B | **P4** `[1,256,40,40]` |
| tiếp theo + 204 800 B | **P5** `[1,512,20,20]` |

Golden nằm trong `02_golden_data/golden_P3.hex` … (đồng thời TB nạp vào `0x0300_0000`… chỉ để tham chiếu).

**Hiện trạng RTL (`supercluster_wrapper`):** ghi AXI bằng **pattern stub**, không qua MAC/PPU/swizzle → **không thể** khớp golden.

**Hiện trạng descriptor (`generate_descriptors.py`):** `dst_off = output_arena_base + global_tile_seq * 32` chỉ phục vụ **chứng minh kênh ghi**; **không** trùng thứ tự flatten NCHW của tensor P3/P4/P5 mà `export_golden_data.py` xuất → dù có tính đúng số, **địa chỉ tile** vẫn có thể sai.

---

## Nguyên tắc

Để `STEP 4` in **ALL OUTPUTS BIT-EXACT**, cần **đồng thời**:

1. **Datapath số học thật** (đọc act/weight từ DDR → GLB → PE → PPU → swizzle → **dữ liệu 256b** lên `m_axi_wdata`).
2. **Địa chỉ ghi (`dst_off` + mọi tile)** khớp cách Python flatten tensor đầu ra vào `OUTPUT_BASE` (P3 nối tiếp P4 nối tiếp P5).
3. **DDR trong TB** có **đủ weight/bias** mọi layer theo `net_desc` / `weight_arena_base` (hiện TB chỉ nạp `weights_L0_conv.hex` — **không đủ** full 23 layer).

---

## Lộ trình đề xuất (theo thứ tự — nên làm từng bước, cosim sau mỗi bước)

### B0 — Giữ Phase A

- Control + AXI + không deadlock (đã có).
- Kỳ vọng: mismatch golden là **bình thường**.

### B1 — Nối **ghi DMA thật** từ khối compute có sẵn

File `PHASE_3/05_integration/rtl/subcluster_wrapper.sv`:

- Đang: `ext_wr_data = '0`, `ext_wr_valid = 1'b0` → **không ghi**.
- Cần: nối **swizzle_engine** (hoặc buffer đầu ra) → `ext_wr_data` / `ext_wr_valid`, handshake `ext_wr_grant` / `fsm_dma_wr_done` đúng timing (tương tự FSM ghi đã làm trong `supercluster_wrapper`).

Sau đó **thay** `tile_fsm` đơn lẻ trong `supercluster_wrapper` bằng **một instance `subcluster_wrapper`** (hoặc đưa toàn bộ SC sang cấu trúc `subcluster` + arbiter ext giống hiện tại nhưng port `ext_*` thay cho stub).

**Mốc kiểm tra B1:** vẫn có thể sai golden, nhưng `m_axi_wdata` phải **phụ thuộc** dữ liệu đọc từ DDR (không còn pattern `addr|seq|2AA` cố định).

### B2 — Đọc DDR đúng (weight + activation)

- `tile_fsm` + DMA read: đảm bảo **đủ beat** / `dma_rd_done` khớp burst (hiện một số chỗ tie-off `dma_rd_done` trong bring-up).
- `shadow_reg_file` / `layer_desc` / `tile_desc`: trường kích thước, stride, pad khớp `quant_params.json` / `desc_layers`.

**Mốc B2:** một **tile nhỏ** (một layer, một SC) cho phép so **một đoạn** bộ nhớ với Python (hoặc golden tạm) — có thể mở rộng TB chỉ so vài dòng hex.

### B3 — Align **`dst_off` / bảng tile** với golden P3|P4|P5

- Trong Python: xác định **layer index** (hoặc tensor) tương ứng P3, P4, P5 trong `export_golden_data.py` / trace.
- Tính **byte offset** mỗi tile trong tensor đầu ra (NCHW, contiguous) → ghi vào field `dst_off` (và `src_*` nếu cần) trong `generate_descriptors.py`.
- Regenerate `desc_*.hex`, chạy lại cosim.

**Mốc B3:** sau khi chạy hết các layer liên quan detect head, vùng `OUTPUT_BASE` **trùng layout** với cách TB so (P3‖P4‖P5 liên tiếp).

### B4 — Nạp **đủ weight** trong `tb_golden_check.sv`

- Dùng `quant_summary` / danh sách file `weights_L*.hex`, `bias_L*.hex` từ `export_golden_data.py`.
- Nạp vào DDR theo đúng `weight_arena_base` và offset trong descriptor (có thể cần thêm field hoặc bảng offset cố định trong TB cho đến khi RTL fetch weight theo descriptor đầy đủ).

**Mốc B4:** inference full layer không “thiếu weight”.

### B5 — Multi-beat / burst AXI (nếu một tile > 32 B)

- Golden mỗi dòng 32 byte; một lần ghi 256b = 32 B. Tensor lớn cần **nhiều beat** hoặc nhiều transaction với `awlen` / địa chỉ tăng dần.
- Mở rộng FSM ghi trong `supercluster_wrapper` (hoặc adapter) cho khớp.

### B6 — `inference_complete` / DONE

- Chỉ báo DONE khi **ghi xong output cuối** (hoặc đồng bộ GS + DMA), không chỉ dựa `fetch_all_done` nếu vẫn còn ghi treo.

---

## Ước lượng công việc

| Hạng mục | Độ khó | Ghi chú |
|----------|--------|---------|
| B1 nối ext_wr | Trung bình | Sửa 1 file + tích hợp SC |
| B2 đọc + FSM | Cao | Nhiều module |
| B3 dst_off Python | Trung bình | Toán địa chỉ + regenerate desc |
| B4 TB nạp weight | Thấp–TB | Nhiều file, script hóa được |
| B5 burst | Trung bình | AXI + FSM |

---

## Việc bạn có thể làm ngay (không cần sửa RTL)

1. Chạy full export:  
   `python PHASE_4/01_export/export_golden_data.py --image ... --output PHASE_4/02_golden_data/`
2. Regenerate descriptor:  
   `python PHASE_4/01_export/generate_descriptors.py --output PHASE_4/02_golden_data/`
3. Mở `quant_params.json` / `descriptor_summary.json` — đối chiếu **layer_id**, shape, với P3/P4/P5 trong `golden_outputs.json`.

---

## Tài liệu liên quan

- `RTL_SYSTEM_COSIM_STATUS.md` — trạng thái bring-up & stub.
- `PHASE_A_COSIM_DEBUG.md` — deadlock / Phase A.

_Khi B1–B3 xong, cùng một `tb_golden_check` sẽ tự chứng minh P3/P4/P5 mà không cần thêm “đọc txt”._
