# PH6-07 — Nghiên cứu chuyên sâu: xây khối tính toán từ Primitive → Layer (YOLOv10n INT8)

**Phạm vi:** Định hướng **nghiên cứu và triển khai** phần cứng kiểu PHASE_3, bám `HW_MAPPING_RESEARCH.md` và cấu trúc RTL thực tế trong `SW_KLTN/PHASE_3/`.  
**Giai đoạn ưu tiên:** Làm **đúng và kiểm thử đầy đủ** các **primitive** rồi mới **ghép layer**; **chưa bắt buộc** hoàn thiện toàn bộ **truyền nhận SoC** (AXI full graph, multi-tile scheduler production) ngay từ đầu — nhưng **không** được bỏ qua **khả năng nối dây** trong thiết kế khối (interface rõ ràng, testbench có thể thay “DMA giả” bằng port cố định).

**Tham chiếu chéo:** `PH6-01` (P→`pe_mode_e`), `PH6-02` (datapath subcluster), `PH6-04` (skip/barrier), `PH6-05` (gap + thứ tự verify), `PH6-06` (catalog file `.sv`).  
**Chuẩn vàng phần mềm:** `SW_KLTN/PHASE1/python_golden/` (ví dụ `primitives/primitive_conv.py` + `quant/quant_affine.py`).

---

## 1. Vấn đề cần giải quyết (problem statement)

Mạng qYOLOv10n PTQ được mô tả trong `HW_MAPPING_RESEARCH.md` như một **đồ thị các layer L0–L22**, trong khi phần cứng thực thi qua **một tập primitive P0–P9** và **một datapath cố định** (PE cluster, PPU, router, swizzle, GLB). Rủi ro chính:

1. **Semantic gap:** “Layer” trong PyTorch ≠ một lần kick PE; một layer có thể = **chuỗi primitive** + **buffer trung gian** + **requant giữa các nhánh**.  
2. **Numeric gap:** Sai một bit trong **làm tròn / thứ tự clamp / per-channel scale** → lệch dần qua nhiều layer.  
3. **Verification gap:** Chỉ chạy end-to-end full model sẽ **không định vị** được lỗi nằm ở primitive nào, descriptor nào, hay scheduler.

**Mục tiêu nghiên cứu của tài liệu này:** Đưa ra **khung phương pháp** để (a) xây và chứng minh **từng primitive** bit-true với golden INT8; (b) **tổ hợp có kiểm soát** thành **layer / block** như trong §3 mapping; (c) chuẩn bị sẵn **điểm nối** cho truyền nhận và descriptor sau này mà không đảo ngược kiến trúc khối.

---

## 2. Khái niệm cốt lõi: Primitive, Layer, Block

### 2.1 Primitive (đơn vị semantics toán học + quant)

Theo `HW_MAPPING_RESEARCH.md` §2.1, mỗi primitive có:

- **Input/output shape** chuẩn (tensor INT8, có thể có PSUM INT32 nội bộ trước requant).  
- **Stride / padding** (nếu có).  
- **Quant domain** (per-tensor, per-channel weight, giữ nguyên scale cho maxpool, align scale cho concat/ewise).  
- **Có đi qua PPU hay không** (bảng §2.1).

**Kiểm thử primitive** = một test có **golden duy nhất** cho **một** semantics đó, với vector nhỏ nhưng **đủ kích hoạt mọi nhánh quant** (ví dụ: có cả kênh dương/âm sau requant, có pad biên).

### 2.2 Layer (đơn vị trong graph YOLO)

Một **layer** trong §3 (ví dụ QC2f, SCDown, SPPF, QConcat) là **chuỗi có thứ tự** các primitive + **bộ nhớ trung gian** (tensor giữ giữa các bước).  
**Kiểm thử layer** = so sánh **tensor cuối của layer** với golden **sau khi** chạy **toàn bộ chuỗi** trong reference, **hoặc** so từng bước trung gian (checkpoint) để khoanh vùng lỗi.

### 2.3 Block (nhóm layer có pattern lặp)

Ví dụ: **QC2f** lặp lại nhiều resolution; **QConcat + QC2f** lặp ở neck. Nghiên cứu phần cứng: sau khi **một** QC2f nhỏ đã pass, các instance khác chủ yếu là **thay đổi shape/scale/weight**, không phải thay đổi RTL.

---

## 3. Nguồn sự thật (sources of truth) — thứ bậc ưu tiên

| Thứ tự | Nguồn | Dùng để |
|--------|--------|---------|
| 1 | `HW_MAPPING_RESEARCH.md` §2.2 | Định nghĩa **công thức quant** và **thứ tự bước** (MAC → bias → requant → act → clamp zp). |
| 2 | `PHASE1/python_golden` + `quant_affine` | **Golden có thể tái lập** (script, seed, export hex/bin). **Không** nhân bản công thức requant thủ công trong nhiều nơi — tránh drift. |
| 3 | `PHASE_3/packages` (`accel_pkg`, `desc_pkg`) | **Hợp đồng bit** giữa compiler và RTL (`pe_mode_e`, `post_profile_t`, v.v.). |
| 4 | RTL PHASE_3 (`ppu.sv`, `pe_unit.sv`, …) | **Implementation**; nếu lệch golden, hoặc sửa RTL hoặc sửa spec nếu golden sai (ghi rõ trong changelog). |

**Nguyên tắc bit-true:** Trước khi tranh luận “độ chính xác mô hình”, khóa **integer pipeline** giữa reference và RTL (cùng `M_int`, `shift`, `zp`, cùng rounding “half-up” như PPU nếu đó là spec).

---

## 4. Ánh xạ Primitive → khối RTL PHASE_3 (tóm tắt nghiên cứu)

Chi tiết bảng đầy đủ: `PH6_01_PRIMITIVE_TO_PE_MODE_AND_RTL.md`. Tóm tắt để định hướng **xây khối**:

| ID | Primitive | `pe_mode_e` / ghi chú | Khối RTL trọng tâm | PPU |
|----|-----------|------------------------|---------------------|-----|
| P0 | RS_DENSE_3x3 | `PE_RS3` | `window_gen`, `pe_cluster`, `column_reduce` | Có |
| P1 | OS_1x1 | `PE_OS1` | Cùng datapath, kernel 1×1 | Có |
| P2 | DW_3x3 | `PE_DW3` | PE DW, tắt/giảm reduce theo mode | Có |
| P3 | MAXPOOL_5x5 | `PE_MP5` | `comparator_tree` + window 5×5 | Không |
| P4 | MOVE | `PE_PASS` / DMA-FSM | `tensor_dma`, GLB, addr | Không |
| P5 | CONCAT | (profile router) | `router_cluster`, có thể + PPU requant | Theo nhánh |
| P6 | UPSAMPLE_NEAREST | `PE_PASS` + swizzle | `swizzle_engine` | Không |
| P7 | EWISE_ADD | PPU | `ppu` `ewise_en` | Có |
| P8 | DW_7x7_MULTIPASS | `PE_DW7` | PE + multipass + `NS_PSUM`/`NS_ACT` | Pass cuối |
| P9 | GEMM_ATTN_BASIC | `PE_GEMM` + điều khiển mở rộng | MAC + **thiếu module attn riêng** (xem PH6-05) | Có |

**Kết luận nghiên cứu:** P0–P8 là **đủ để cover backbone/neck phần lớn** nếu concat/router/swizzle ổn; **P9 là điểm rủi ro kiến trúc** — nên tách **kế hoạch verify** riêng (GEMM nhỏ → softmax approx → ghép QPSA).

---

## 5. Chiến lược xây dựng phần cứng theo pha (không cần full I/O ngay)

### Pha A — “Compute core đóng” (khuyến nghị bắt đầu từ đây)

**Mục tiêu:** Chứng minh **toán đúng** trên interface testbench, **không** phụ thuộc descriptor DDR hay SoC.

- **Đơn vị:** `pe_unit`, `dsp_pair_int8`, `column_reduce`, `comparator_tree`, `silu_lut`.  
- **Cách nối:** TB đứng trực tiếp trên port (array unpacked, valid) hoặc wrapper tối giản.  
- **Golden:** Một output channel / một cửa sổ pool / một lane — export từ `PHASE1/python_golden`.

**Tiêu chí xong pha:** Mọi mode PE liên quan P0–P3, P8 (từng phần) có **ít nhất một** test vector khớp từng phần tử INT8 (hoặc PSUM INT32 nếu test riêng stage).

### Pha B — “Datapath có bộ nhớ cục bộ”

**Mục tiêu:** Đúng **thứ tự byte, offset, halo**, giống một tile nhỏ.

- **Khối:** `window_gen`, `addr_gen_*`, `glb_*_bank` (hoặc model SRAM đơn giản tương đương).  
- **Golden:** Cùng tensor với pha A nhưng **đọc qua addr gen**; so memory dump sau khi fill.

**Tiêu chí xong pha:** Mismatch chỉ còn do **sai layout tensor** (NCHW vs block), không do MAC — phải **đóng băng layout** trong tài liệu compiler.

### Pha C — “PPU + router + swizzle”

**Mục tiêu:** Khóa **requant**, **SiLU LUT**, **concat cùng/khác scale**, **upsample**.

- **Khối:** `ppu.sv`, `router_cluster.sv`, `swizzle_engine.sv`.  
- **Golden:** `post_profile_t` / `router_profile_t` từ testbench (hardcode) trước khi có compiler.

**Tiêu chí xong pha:** Các rule §2.2 cho conv/DW/concat/upsample được **mô phỏng tối thiểu** trên shape nhỏ và khớp RTL.

### Pha D — “Subcluster shell” (`subcluster_wrapper`)

**Mục tiêu:** Một **tile** chạy hết đường ống như `PH6_02`.

- **Kiểm thử:** `tb_pe_cluster` mở rộng hoặc TB riêng gọi `tile_fsm` + shadow reg với **một** `layer_desc_t`/`tile_desc_t` cố định.  
- **Chưa cần:** `desc_fetch_engine` đọc DDR thật — có thể **inject struct** từ TB.

**Tiêu chí xong pha:** Ít nhất **một** primitive mỗi nhóm (conv, pool, DW, upsample, concat đơn giản) chạy qua **cùng shell** và khớp golden.

### Pha E — “Layer composition”

**Mục tiêu:** Chuỗi primitive như §3 (thu nhỏ).

- **Ví dụ thứ tự nghiên cứu:**  
  1. QC2f thu nhỏ: `OS_1x1 → RS_3x3 → CONCAT → OS_1x1` trên `H,W,C` nhỏ.  
  2. SPPF thu nhỏ: `OS_1x1 → MP5×3 → CONCAT 4 nhánh → OS_1x1`.  
  3. SCDown: hai nhánh `OS_1x1 → DW_3x3 s2` + CONCAT (đúng như §3.1 L5).  
- **Checkpoint:** Sau mỗi primitive trong chuỗi, có thể dump tensor trung gian và so với reference **cùng điểm dừng**.

**Tiêu chí xong pha:** Một layer phức tạp = **pass** khi **cả chuỗi** khớp; nếu fail, **đã có** checkpoint để biết bước nào.

### Pha F — Truyền nhận & descriptor đầy đủ (sau cùng trong roadmap này)

**Mục tiêu:** `controller_system`, `desc_fetch_engine`, CSR, DMA — đúng **binary layout** `net_desc_t`/`layer_desc_t`/`tile_desc_t`.

- Pha này **không thay thế** pha A–E; nó **đóng gói** đường đi đã chứng minh.

---

## 6. Kim tự tháp kiểm thử (test pyramid) — yêu cầu “đầy đủ”

### 6.1 Mức L0 — Unit RTL

- Mỗi file lá trong `01_compute_leaf`, `02_ppu` có TB riêng (repo đã có nhiều `tb_*.sv`).  
- **Pass criteria:** 100% mode được primitive sử dụng; coverage toggle cho `pe_mode_e` liên quan.

### 6.2 Mức L1 — Primitive integration

- Một test = **một** P0…P8 (P9 tách riêng).  
- **Vector:** nhỏ, cố định, lưu trong `PHASE1` hoặc `test_vectors/` (hex).  
- **Scoreboard:** so INT8 từng phần tử hoặc `max |a-b|` + số phần tử sai.

### 6.3 Mức L2 — Mini-layer / block

- Chuỗi primitive từ §3; shape giảm.  
- **Pass criteria:** output layer khớp reference; nếu có concat cross-scale, **bắt buộc** có case scale_A ≠ scale_B.

### 6.4 Mức L3 — Layer thật (subset L0–L22)

- Bắt đầu từ layer có ít phụ thuộc skip (ví dụ L0, L1), sau đó thêm **QConcat có skip** (L12, L15…) — cần **chiến lược buffer** `PH6_04`.

### 6.5 Mức L4 — E2E SoC

- `tb_accel_e2e` + DDR model; so P3/P4/P5 hoặc hash — chỉ khi L3 ổn.

---

## 7. Descriptor & truyền nhận: vai trò trong nghiên cứu

`desc_pkg.sv` **không** định nghĩa toán học primitive; nó định nghĩa **cách điều khiển** phần cứng. Do đó:

- **Giai đoạn primitive-first:** TB có thể **bỏ qua** `tile_table_offset` thật, inject struct trực tiếp.  
- **Giai đoạn layer:** Phải chứng minh **compiler** sinh `template_id` khớp `pe_mode_e`, `num_*_pass` khớp P8, `router_profile_id` khớp P5.  
- **Test riêng cho descriptor:** Đọc binary → `layer_desc_t` → so từng field với golden struct (không cần chạy MAC).

---

## 8. Rủi ro và giả định cần ghi nhận trong luận văn / báo cáo

1. **HW_MAPPING vs RTL:** Văn bản mapping có thể ghi `LANES=16` minh họa; RTL dùng `LANES=32` (`accel_pkg`) — **semantics không đổi**, **tiling đổi**; mọi golden phải **parameterize** theo package.  
2. **SCDown đầy đủ:** §3 có nhánh kép + CONCAT; verify layer L5/L7/L20 phải **bám đúng chuỗi** đã chọn, không chỉ một nhánh đơn giản hóa.  
3. **P9 QPSA:** GEMM 400×400 @ INT8 và softmax approx — **không** gộp vào tiêu chí “primitive cơ bản xong” cho đến khi có spec RTL softmax + test GEMM nhỏ.  
4. **Skip lớn (F4, F6, F8, F13):** Đúng layer = đúng **capacity GLB + thời gian giữ**; đây là kiểm thử **hệ thống bộ nhớ**, không chỉ MAC.

---

## 9. Tiêu chí hoàn thành (definition of done) theo cấp

| Cấp | Hoàn thành khi |
|-----|----------------|
| **Primitive** | Mỗi P0–P8 có ≥1 vector bit-true; tài liệu map P→RTL→TB; P9 có kế hoạch riêng. |
| **Layer** | ≥3 block đại diện (QC2f nhỏ, SPPF nhỏ, SCDown 2 nhánh) pass so với `PHASE1` hoặc script tương đương. |
| **Graph nhỏ** | Chuỗi L0→L3 hoặc tương đương không cần skip dài, pass end-to-end trên shell Pha D. |
| **Production path** | Descriptor + DMA + CSR pass trên `tb_accel_e2e` với cùng vector đã dùng ở Pha D. |

---

## 10. Kết luận nghiên cứu

- **Đúng primitive trước** là điều kiện **cần** để layer có ý nghĩa; **đúng layer** là điều kiện **cần** để graph YOLO có ý nghĩa.  
- **Chưa cần truyền nhận đầy đủ** ở giai đoạn đầu **nếu** interface khối tính toán được thiết kế để sau này **thay TB inject bằng descriptor/DMA** mà không đổi datapath.  
- **Đầy đủ kiểm thử** được hiểu là **kim tự tháp** L0–L4, không phải chỉ một testbench duy nhất.

---

*Tài liệu này là spec nghiên cứu / phương pháp; cập nhật khi RTL hoặc `HW_MAPPING_RESEARCH.md` thay đổi phiên bản.*
