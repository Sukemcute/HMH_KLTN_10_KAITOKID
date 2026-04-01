# PHASE 5 — Kiến trúc RTL cho qYOLOv10n INT8 (L0–L22): primitive, đường dữ liệu, controller

**Vai trò tài liệu:** Đóng vai phân tích **thiết kế phần cứng / RTL**, tổng hợp từ các nguồn nghiên cứu trong repo (không thay thế bản gốc — xem mục 9).  
**Phạm vi bộ tăng tốc:** Backbone + Neck **layer 0–22** INT8; **Qv10Detect (L23) + NMS** để CPU/phần mềm theo `MODEL_FORWARD_FLOW.md`.

---

## 1. Mục tiêu thiết kế bạn đặt ra

Bạn muốn **một khối tính toán** có thể:

- Thực thi **các primitive** tương ứng các khối YOLOv10n đã PTQ INT8.
- Chạy **toàn bộ L0–L22** **hiệu quả** chủ yếu bằng **cấu hình đường đi dữ liệu** (descriptor / router / template), không Resynthesize từng layer thành IP riêng.

**Chốt kiến trúc:** Tách rõ ba lớp:

| Lớp | Nội dung |
|-----|----------|
| **A. Điều khiển & metadata** | MMIO, CSR, bảng net/layer/tile trong DDR, fetch, scheduler, barrier. |
| **B. Vận chuyển tensor** | Một (hoặc vài) master AXI, DMA burst, GLB/skip spill, địa chỉ từ `tile_desc`. |
| **C. Primitive compute** | Mảng PE cố định (conv / depthwise / pool / pass-through) + **PPU** (bias, requant INT8, SiLU LUT, e-wise) — **cấu hình bằng `template_id`, `router_profile`, `post_profile` + kích thước từ `layer_desc`**. |

Lớp **C** không “biết” tên layer YOLO; nó chỉ biết **template** (ví dụ RS3, OS1, DW3, DW7, MP5, GEMM, PASS) đã map sẵn từ compiler.

---

## 2. Primitive phần mềm → template phần cứng

Theo `MODEL_BLOCKS_INT8_DETAIL.md` và `MODEL_LAYERS_INT8_FLOW.md`, các **khối cần hỗ trợ** trên đường INT8:

| Primitive (model) | Ý nghĩa tính toán | Gợi ý map RTL (như `accel_pkg::pe_mode_e` + datapath) |
|--------------------|-------------------|--------------------------------------------------------|
| **Conv** (fused BN) | INT8 conv → INT32 MAC → requant + SiLU/ReLU | **OS1** (1×1) / **RS3** (3×3 standard) tùy kernel stride; **PPU** cho act + quant. |
| **QC2f** | Chuỗi Conv 1×1 + bottleneck + **concat** kênh + Conv cuối | **Nhiều tile/phase** hoặc một **macro template**: lặp RS3/OS1, **router** concat; không cần một “opcode QC2f” riêng nếu compiler bung thành subgraph. |
| **SCDown** | Conv s=2 ∥ (MaxPool s=2 → Conv) → **concat** kênh | Hai nhánh conv/pool; có thể **hai pass** hoặc template **DW3** + pool (tùy biến thể đã fuse). |
| **SPPF** | Conv → MaxPool×3 (k=5,s=1) → concat 4 nhánh → Conv | **MP5** + concat + OS1; pool trên INT8 là so sánh, không đổi scale nếu cùng qconfig. |
| **QPSA** | Attention (conv + có thể softmax) | **Khó nhất:** (1) bung **GEMM**/conv INT8 + LUT xấp xỉ; (2) hoặc **tách nhỏ** dequant cục bộ (tốn băng thông/logic); (3) hoặc **offload** một phần — cần khớp **đúng inference** như cảnh báo trong `HW_ARCHITECTURE_V2_100FPS.md`. |
| **Upsample** | Nearest ×2 | **PE_PASS** / **địa chỉ đọc lại** (không MAC); chỉ thay đổi `addr_gen` / swizzle. |
| **QConcat** | Ghép kênh từ 2 tensor | **Không MAC** — **router** + đồng bộ scale (thường cùng qconfig); nếu khác scale cần **requant** trước concat (compiler chèn). |
| **QC2fCIB** | C2f + CIB (dw lớn, multi-pass) | **DW7** / nhiều **pass** (`num_cin_pass`, `num_k_pass` trong descriptor) như đã có trong `tile_fsm` / spec V2. |

**Kết luận RTL:** IP **một** mảng PE + **PPU** + **router/swizzle** đủ nếu **compiler** sinh đúng **chuỗi tile + template + offset DDR** cho từng layer con của QC2f/SCDown/SPPF.

---

## 3. Vì sao chỉ “cấu hình đường đi” vẫn chạy được L0–L22?

1. **Mọi layer** trong trace INT8 đều là **biến đổi tensor** có thể phân rã thành: **đọc activation/weight** → **MAC/pool/concat logic** → **ghi activation** (và giữ **skip** trong GLB hoặc DDR).
2. **`MODEL_LAYER_DEPENDENCIES.md`** chỉ ra:
   - Phần lớn layer **chỉ** `in(Lk) = L(k-1)` → scheduler **tuần tự** đơn giản.
   - **L12, L15, L18, L21** là **QConcat** cần **L6, L4, L13, L8** (v.v.) → phần cứng **bắt buộc** có:
     - **Bộ nhớ on-chip** hoặc **DDR skip arena** giữ feature đến khi concat;
     - **Router** chọn **tensor nguồn** theo `src_in_tid` / bảng trong `layer_desc` (đã có hướng trong `desc_pkg`).
3. **Descriptor-driven:** Mỗi tile mang **`src_in_off`, `src_w_off`, `src_skip_off`, `dst_off`** — đúng với thiết kế `tile_fsm` + `tensor_dma`: **chỉ đổi cấu hình**, không đổi dây DMA.

---

## 4. Phân tách trách nhiệm: quantization & nơi đón dữ liệu

Theo `MODEL_FORWARD_FLOW.md`:

| Bước | Ai làm | Ghi chú RTL |
|------|--------|-------------|
| LetterBox, RGB, normalize float | **CPU** | Không thuộc accelerator. |
| **Quantize đầu vào** (scale_in, zp_in) | **CPU** (hoặc DMA nhỏ) | IP nhận **INT8** đã quant; có thể ghi vào DDR trước khi kick IP. |
| Requant giữa layer | **PPU trong IP** | `m_int`, `shift`, `zp_out`, bias — nên nạp từ **DDR blob** hoặc CSR (PHASE_5 đã chuẩn bị CSR PPU table). |
| SiLU sau conv | **PPU** (LUT) | Khớp `Conv.forward_fuse` patch INT8 → float → SiLU → quant; HW tối ưu = **LUT / cố định điểm**. |

**“Nơi đón dữ liệu” trên IP (tầng vận chuyển):**

1. **MMIO** — không đón tensor; chỉ **CSR** (pointer `net_desc`, layer range, start, irq mask).
2. **AXI master** — đón **descriptor** + **weight** + **activation** + **skip** từ DDR vào **GLB** (hoặc buffer behavioral tạm).
3. **Sau prefill** — dữ liệu “sẵn sàng trước PE” nằm ở **GLB input/weight** (kiến trúc mục tiêu trong `RTL_IP_DIRECTORY_TREE` / PHASE_3).

---

## 5. Controller cần làm gì để “chỉ cấu hình” là chạy L0–L22?

**Controller (như `controller_system` hiện tại):**

- Giữ **CSR**, kick **`desc_fetch_engine`**, điều phối **global_scheduler** tới 4 SC.
- **Không** tính conv; chỉ đảm bảo **tile descriptor** tới đúng SC đúng lúc.

**Compiler / offline tool (ngoài RTL nhưng bắt buộc cho “chỉ config”):**

- Sinh **`net_desc` / `layer_desc` / `tile_desc`** khớp `desc_pkg`.
- Gán **`template_id`** ↔ primitive (RS3, OS1, DW3, …) từng tile.
- Tính **`num_cin_pass` / `num_k_pass`** cho layer nặng (QC2fCIB, QC2f lớn).
- Đặt **`src_skip_off`** / barrier / `tile_flags` cho **QConcat** và **skip retention** (GLB vs spill DDR — tham chiếu Option A/B trong `HW_ARCHITECTURE_V2_100FPS.md`).

Không có compiler, “chỉ đổi config” trên chip **không đủ**.

---

## 6. Kiến trúc phần cứng đề xuất (tổng thể, hiệu quả, không quá phức tạp)

Rút từ `HW_ARCHITECTURE_V2_100FPS.md` + luồng PHASE_3:

```
                    ┌─────────────────────────────────────────┐
  CPU ──AXI-Lite──► │ CSR / Controller / Desc fetch / Sched   │
                    └──────────────┬──────────────────────────┘
                                   │ tile_desc, layer_desc
                    ┌──────────────▼──────────────────────────┐
  DDR ◄════════════►│ 4 × SuperCluster                       │
      AXI4 256b     │   each: FIFO tile + arb + tensor_dma   │
                    │   4 × subcluster: GLB + tile_fsm + …     │
                    └──────────────┬──────────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────────┐
                    │ PE array (template-driven) + PPU + swizzle│
                    └─────────────────────────────────────────┘
```

**Nguyên tắc hiệu quả (V2):**

- **LANES=32**, **EXT 256b**, **2 sub RUNNING / SC** khi compute > fill+drain — tăng throughput mà không nhân đôi số PE theo chiều COL.
- **Một master AXI** mux: descriptor + tensor (và sau này PPU table) — đơn giản hơn nhiều master.
- **Skip lớn (F4 80×80)** có thể **spill DDR**; skip nhỏ giữ **GLB** — trade-off BRAM đã phân tích trong tài liệu V2.

---

## 7. Thứ tự thực thi trên chip (trả lời “truyền xong mới tính?”)

Đúng với **từng tile**:

1. **Cấu hình** (MMIO + descriptor trong DDR).  
2. **Fetch descriptor** (AXI read).  
3. **DMA prefill** weight / input / (skip) vào buffer nội bộ.  
4. **RUN_COMPUTE** / multi-pass trên PE.  
5. **PPU** (requant + act + e-wise nếu có).  
6. **Swizzle + ghi DDR** (output tile).

Giữa các layer L0…L22, **scheduler** xếp tile theo dependency; **QConcat** buộc có **bước chờ** feature cũ đã lưu.

---

## 8. Bảng tham chiếu nhanh L0–L22 (shape & primitive)

Nguồn: `MODEL_LAYERS_INT8_FLOW.md` (đã rút gọn).

| L | Module | Output shape (1,…,H,W) | MAC-heavy? |
|---|--------|------------------------|------------|
| 0 | Conv | 16,320,320 | Có |
| 1 | Conv | 32,160,160 | Có |
| 2 | QC2f | 32,160,160 | Có |
| 3 | Conv | 64,80,80 | Có |
| 4 | QC2f | 64,80,80 | Có — **skip cho L15** |
| 5 | SCDown | 128,40,40 | Có |
| 6 | QC2f | 128,40,40 | Có — **skip cho L12** |
| 7 | SCDown | 256,20,20 | Có |
| 8 | QC2f | 256,20,20 | Có — **skip cho L21** |
| 9 | SPPF | 256,20,20 | Có |
| 10 | QPSA | 256,20,20 | Có (đặc biệt) |
| 11 | Upsample | 256,40,40 | Không |
| 12 | QConcat | 384,40,40 | Không |
| 13 | QC2f | 128,40,40 | Có — **skip cho L18** |
| 14 | Upsample | 128,80,80 | Không |
| 15 | QConcat | 192,80,80 | Không |
| 16 | QC2f | 64,80,80 | Có |
| 17 | Conv | 64,40,40 | Có |
| 18 | QConcat | 192,40,40 | Không |
| 19 | QC2f | 128,40,40 | Có |
| 20 | SCDown | 128,20,20 | Có |
| 21 | QConcat | 384,20,20 | Không |
| 22 | QC2fCIB | 256,20,20 | Có |

**Đầu ra accelerator (trước L23):** ba nhánh P3/P4/P5 INT8 + metadata — CPU chạy `Qv10Detect` + NMS.

---

## 9. Nguồn nghiên cứu gốc trong repo (giữ nguyên, tra cứu chi tiết)

| Tài liệu | Đường dẫn |
|----------|-----------|
| Khối INT8 chi tiết | `SW_KLTN/MODEL_BLOCKS_INT8_DETAIL.md` |
| CPU vs HW vs head | `SW_KLTN/MODEL_FORWARD_FLOW.md` |
| Skip / concat dependency | `SW_KLTN/MODEL_LAYER_DEPENDENCIES.md` |
| Trace từng layer | `SW_KLTN/MODEL_LAYERS_INT8_FLOW.md` |
| Kiến trúc V2, FPS, BRAM/DSP, scheduler | `SW_KLTN/HW_ARCHITECTURE_V2_100FPS.md` |

**RTL mẫu data path:** `SW_KLTN/PHASE_3/` + `PHASE_5/07_system/rtl/accel_top.sv`, `PHASE_5/data_path/spec_sv/p5_dp_*.sv`.

---

## 10. Rủi ro & việc cần làm tiếp (RTL)

1. **QPSA:** Xác nhận với trace PTQ thực tế; nếu có softmax float, cần **chuẩn fixed-point** hoặc **LUT** có chứng minh sai số chấp nhận được.  
2. **Đồng bộ quant ở QConcat:** Đảm bảo compiler không ghép hai tensor khác scale (hoặc chèn requant).  
3. **PPU parameters:** Nối **DDR/CSR** (PHASE_5) thay vì chỉ port top để SoC khép kín.  
4. **Đồng bộ “hoàn tất inference”:** `sched_inference_complete` nên gắn **DMA + tile xong** (đã ghi chú trong `RTL_IP_DIRECTORY_TREE.txt`).

---

## 11. >100 FPS: kiến trúc trong tài liệu có *thực sự* đạt được không? & đề xuất mảng PE / dataflow

### 11.1. Phân biệt “mục tiêu thiết kế” và “đã chứng minh trên silicon”

- File **`HW_YOLOV10N_RTL_KIEN_TRUC_VA_PRIMITIVE.md`** (và mục 6) mô tả **lớp kiến trúc** (descriptor-driven + 4 SC + PE template + PPU + DMA). **Bản thân nó không chứa chứng cứ đo FPS** — nó **tương thích** với đường triển khai RTL PHASE_3/5.
- Con số **~100–115 FPS (chỉ accelerator / L0–L22)** xuất phát từ **`HW_ARCHITECTURE_V2_100FPS.md`**, với **giả định**:
  - **3072 MAC hoạt động mỗi chu kỳ** (LANES=32, 12 PE/sub, **2 sub RUNNING / SC**, 4 SC),
  - **~3.08 GMAC** cho backbone+neck,
  - **clock 200–220 MHz**, **utilization tổng ~50–55%** (gồm spatial util theo W + overhead thời gian ~27% trong mục 7.4),
  - RTL **thực sự** đạt dual-RUNNING, fill/drain overlap như phân tích, và **QPSA** không phá pipeline.

**Kết luận thẳng:** Đó là **mục tiêu khả thi trên giấy** cho **XC7VX690T-class + host CPU đủ mạnh**; **không** tương đương “chắc chắn >100 FPS” cho đến khi:

1. RTL khớp V2 (PHASE_3 hiện có thể vẫn gần V1 về scheduler/active subs — cần đối chiếu `accel_pkg` / `local_arbiter` / occupancy thực tế).  
2. Có **cosim / FPGA measured** `T_hw` (wall-clock) thay cho spreadsheet.  
3. **End-to-end:** Throughput = `1 / max(T_preprocess, T_hw, T_post)`. Với **MicroBlaze**, tài liệu V2 ghi rõ bottleneck CPU → **~50 FPS** dù HW đủ nhanh (`HW_ARCHITECTURE_V2_100FPS.md` §8.2–8.3).

### 11.2. Khi nào coi là “hiệu quả” cho >100 FPS?

Cần **đồng thời**:

| Điều kiện | Ý nghĩa |
|-----------|---------|
| Đủ **OP/s** | ~**300 GOPS** hiệu dụng cho ~3 GMAC/frame @ 100 FPS (INT8 MAC ≈ OP trong mô hình đơn giản). |
| **Utilization** không bị vỡ | Dual-RUNNING + tile schedule sao cho **compute ≥ fill+drain** (đã phân tích per-tier trong V2 §5.3, 7.3). |
| **Bộ nhớ** | BRAM đủ GLB hoặc chấp nhận spill skip; không để DMA nghẽn lâu. |
| **Compiler** | Tile + barrier khớp **L12/L15/L18/L21**; giảm tile boundary waste. |

Thiếu một trong các mục trên → FPS **tụt** dù “đúng primitive”.

### 11.3. Đề xuất mảng PE & cấu hình dataflow (dễ map từ compiler, vẫn hiệu quả)

**Giữ lõi hiện tại (khuyến nghị làm baseline):**  
**Một lưới PE đồng nhất** (3×4 × LANES), **đa chế độ** qua `template_id` + **router + window_gen** — vì YOLOv10n là tập **conv / DW / pool / concat / upsample** lặp lại, không cần ASIC riêng từng layer.

**Bổ sung để “dễ config” và giảm rủi ro hiệu năng:**

1. **Router / `router_profile_t` giàu ngữ nghĩa hơn**  
   - Thay vì chỉ “mask”, mô tả rõ **cạnh dataflow**: *tensor A → bank slot i*, *tensor B → concat port*, *output → bank j / DDR*.  
   - Compiler sinh **DAG nhỏ** mỗi layer; RTL chỉ **thực thi edge** đã mã hóa → giảm lỗi cấu hình và dễ verify với `MODEL_LAYER_DEPENDENCIES.md`.

2. **Slot skip có tên (logical buffer id)** cho **L4, L6, L8, L13**  
   - Map cố định: ví dụ `SKIP_SLOT_P3 = 0` (L4), … — descriptor chỉ ghi **slot id + DDR fallback** nếu spill.  
   - QConcat (L12, L15, L18, L21) chỉ cần **chọn 2 slot** → cấu hình ổn định, debug nhanh.

3. **Chuẩn hóa “macro-tile” cho QC2f / QC2fCIB**  
   - Một layer phần mềm = **chuỗi micro-tile** cùng `layer_id` với **sub_index** hoặc **phase_id** trong `tile_flags` — hardware vẫn là primitive lặp lại, compiler gánh **số pass** (đã có `num_cin_pass` / `num_k_pass`).

4. **Không tăng PE_COLS lên 8** (đúng hướng V2)  
   - Lợi ích H song song giảm dần khi H nhỏ; chi phí router/PPU/bank **tăng mạnh**. Ưu tiên **LANES** và **2 active sub / SC**.

5. **Phương án thay thế (chỉ khi baseline không đủ FPS)** — trade-off phức tạp:
   - **Hai engine khác biệt:** ví dụ **wide MAC** cho tầng H,W lớn (L0–L4) + **deep engine** cho C lớn, H,W nhỏ — tăng peak GOPS nhưng **tăng gấp đôi công việc schedule và wiring**; chỉ xem xét sau khi đo được tier nào là bottleneck thực tế (bảng MAC/tier §7.1 V2).

6. **QPSA**  
   - Nếu giữ INT8 end-to-end: cần **một primitive GEMM/LUT** có **latency cố định** và **không chiếm master AXI** trong suốt layer; nếu không, mọi phân tích FPS ở L10 nên **tăng margin** hoặc **đo riêng** layer này.

### 11.4. Tóm tắt một câu

- Kiến trúc trong tài liệu PHASE 5 là **hướng đi đúng** và **khớp** với phân tích V2 **để hướng tới** >100 FPS trên HW; **không** nên hiểu là đã **đảm bảo** FPS nếu chưa có RTL V2 đầy đủ + đo thực tế + pipeline CPU phù hợp.  
- **Mảng PE:** nên giữ **đồng nhất + template + router/slot skip rõ ràng**; cải tiến hiệu quả chủ yếu đến từ **scheduler 2-sub-active**, **compiler tile**, và **bộ nhớ/DMA**, hơn là tăng số loại PE khác hẳn.

---

*Tài liệu này là bản tổng hợp thiết kế RTL; mọi số liệu FPS/BRAM/DSP lấy từ `HW_ARCHITECTURE_V2_100FPS.md` và có thể chỉnh khi đổi FPGA hoặc clock.*
