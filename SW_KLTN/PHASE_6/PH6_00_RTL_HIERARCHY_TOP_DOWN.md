# PH6-00 — Hệ thống phân cấp RTL: từ top xuống lá

*Tài liệu cho người thiết kế RTL: mỗi cấp trả lời **làm gì**, **nối với ai**, **con bên trong là gì**.*

---

## Cấp 0 — Biên SoC (chip / FPGA wrapper)

| Khối | File RTL (PHASE_3) | Chức năng |
|------|-------------------|-----------|
| **accel_top** | `07_system/rtl/accel_top.sv` | Top IP: **AXI-Lite slave** (MMIO), **một AXI4 master 256b** mux giữa *đọc descriptor* và *DMA tensor* (4 SC); có thể thêm bank PPU (PHASE_5). |
| *(SoC ngoài IP)* | — | CPU, DDR controller, interrupt controller, clock/reset — không nằm trong repo accelerator. |

**Luồng:** CPU ghi CSR → `controller_system` đọc DDR (descriptor) → scheduler đẩy tile → mỗi SC dùng `tensor_dma` đọc/ghi tensor → kết quả P3/P4/P5 nằm DDR theo `dst_off` trong `tile_desc`.

---

## Cấp 1 — Bốn nhánh song song (SuperCluster)

| Khối | File RTL | Chức năng |
|------|----------|-----------|
| **supercluster_wrapper** | `07_system/rtl/supercluster_wrapper.sv` | **Một SC** = FIFO hàng đợi tile từ `global_scheduler` + **local_arbiter** (chọn sub nào dùng cổng DMA) + **một tensor_dma** + **4 × subcluster_wrapper**. |
| **tensor_dma** | `07_system/rtl/tensor_dma.sv` | Engine burst AXI: `rd_req/addr/len` → `m_axi_ar*/*r*`; `wr_*` → `m_axi_aw*/*w*/*b*`. |
| **local_arbiter** | `06_control/rtl/local_arbiter.sv` | Phân cổng `ext_*` (logical) giữa 4 sub khi tranh DMA. |

**Kết nối:** `accel_top` nối `ctrl_sc_tile[gi]`, `ctrl_sc_layer`, `ctrl_sc_valid/accept` vào SC[gi]; mux `m_axi_*` giữa `u_ctrl` (chỉ đọc) và 4 SC.

---

## Cấp 2 — Điều khiển tập trung (không nằm trong SC)

| Khối | File RTL | Chức năng |
|------|----------|-----------|
| **controller_system** | `07_system/rtl/controller_system.sv` | **CSR** (MMIO decode ở `accel_top`), **desc_fetch_engine**, **global_scheduler**, **barrier_manager**, status/IRQ/perf. |
| **desc_fetch_engine** | `06_control/rtl/desc_fetch_engine.sv` | Đọc `net_desc` / `layer_desc` / `tile_desc` từ DDR (64B/block). |
| **global_scheduler** | `06_control/rtl/global_scheduler.sv` | Handshake tile với 4 SC (`sc_tile_valid` / `sc_tile_accept`). |
| **barrier_manager** | `06_control/rtl/barrier_manager.sv` | Scoreboard cho đồng bộ concat/skip (tín hiệu từ SC). |

---

## Cấp 3 — Một Subcluster (đơn vị tính toán + GLB + DMA handshake)

| Khối | File RTL | Chức năng |
|------|----------|-----------|
| **subcluster_wrapper** | `05_integration/rtl/subcluster_wrapper.sv` | **Ghép** tile_fsm, shadow_reg_file, (mục tiêu) GLB + addr_gen + router + window_gen + pe_cluster + ppu + swizzle; cổng **ext_*** tới `tensor_dma`. *Hiện trạng: có nhánh behavioral — cần nối dây đủ theo spec.* |
| **tile_fsm** | `06_control/rtl/tile_fsm.sv` | FSM pha: LOAD_CFG → PREFILL_WT/IN/SKIP → RUN_COMPUTE → POST_PROCESS → SWIZZLE_STORE; sinh `dma_rd_addr/len`, `dma_wr_*`. |
| **shadow_reg_file** | `05_integration/rtl/shadow_reg_file.sv` | Chốt `tile_desc` + `layer_desc` + profile → cấu hình ổn định cho cả tile. |

---

## Cấp 4 — Bộ nhớ on-chip & địa chỉ (theo HW_MAPPING §6)

| Khối | File RTL | Chức năng |
|------|----------|-----------|
| **glb_input_bank** | `03_memory/rtl/glb_input_bank.sv` | Line buffer / slot input (theo banking `h mod 3` trong nghiên cứu mapping). |
| **glb_weight_bank** | `03_memory/rtl/glb_weight_bank.sv` | Cache trọng số tile. |
| **glb_output_bank** | `03_memory/rtl/glb_output_bank.sv` | PSUM (INT32) hoặc ACT (INT8) tùy pass. |
| **metadata_ram** | `03_memory/rtl/metadata_ram.sv` | Valid/slot/ring cho bank. |
| **addr_gen_input** | `03_memory/rtl/addr_gen_input.sv` | Địa chỉ đọc activation (Q_in, row_slot, Wblk). |
| **addr_gen_weight** | `03_memory/rtl/addr_gen_weight.sv` | Địa chỉ đọc weight. |
| **addr_gen_output** | `03_memory/rtl/addr_gen_output.sv` | Địa chỉ ghi output. |

---

## Cấp 5 — Di chuyển & cửa sổ

| Khối | File RTL | Chức năng |
|------|----------|-----------|
| **router_cluster** | `04_data_movement/rtl/router_cluster.sv` | Định tuyến tensor giữa bank ↔ PE; **CONCAT**, broadcast (theo `router_profile_t`). |
| **window_gen** | `04_data_movement/rtl/window_gen.sv` | Sinh cửa sổ **3×3 / 1×1** (và mở rộng 5×5 cho pool nếu cùng engine địa chỉ). |
| **swizzle_engine** | `04_data_movement/rtl/swizzle_engine.sv` | **UPSAMPLE_NEAREST**, reorder kênh/HW trước khi ghi DDR. |

---

## Cấp 6 — Tính toán (MAC + giảm chiều + pool)

| Khối | File RTL | Chức năng |
|------|----------|-----------|
| **pe_cluster** | `05_integration/rtl/pe_cluster.sv` | **12 × pe_unit** + **column_reduce** + **comparator_tree** — thực thi **RS_DENSE_3x3**, **OS_1x1**, **DW_3x3**, một phần **DW_7x7 multipass** (theo mode). |
| **pe_unit** | `01_compute_leaf/rtl/pe_unit.sv` | Lõi MAC song song theo **LANES** (32 trong spec V2). |
| **dsp_pair_int8** | `01_compute_leaf/rtl/dsp_pair_int8.sv` | Cặp INT8×INT8→INT32 (packing DSP nếu có). |
| **column_reduce** | `01_compute_leaf/rtl/column_reduce.sv` | Cộng dồn theo trục giảm cho conv dày (không dùng cho DW thuần). |
| **comparator_tree** | `01_compute_leaf/rtl/comparator_tree.sv` | **MAXPOOL_5x5** (max trên cửa sổ) — vai trò “pool_engine” trong mapping. |

---

## Cấp 7 — Hậu xử lý lượng tử & kích hoạt

| Khối | File RTL | Chức năng |
|------|----------|-----------|
| **ppu** | `02_ppu/rtl/ppu.sv` | Bias INT32 → requant (M_int, shift) → **SiLU LUT** / ReLU / None → clamp INT8; hỗ trợ e-wise (skip). |
| **silu_lut** | `01_compute_leaf/rtl/silu_lut.sv` | ROM/LUT có thể chia sẻ với PPU. |

---

## Cấp 8 — Gói & CSR (không phải module tích hợp)

| Khối | File | Chức năng |
|------|------|-----------|
| **accel_pkg** | `packages/accel_pkg.sv` | `pe_mode_e`, `tile_state_e`, kích thước PE/LANES. |
| **desc_pkg** | `packages/desc_pkg.sv` | `net_desc_t`, `layer_desc_t`, `tile_desc_t`, profiles. |
| **csr_pkg** | `packages/csr_pkg.sv` | Địa chỉ MMIO. |

---

## Sơ đồ kết nối khối (một Subcluster — mục tiêu đích)

```
ext_* (DMA) ──┬──► GLB input/weight  ◄── addr_gen_*
              │
              ▼
         router_cluster ◄──► window_gen ──► pe_cluster ──► ppu ──► swizzle_engine ──► ext_* (write)
              ▲
              └── metadata_ram (slots)
tile_fsm / shadow_reg_file điều khiển pha và load profile
```

---

## Tóm tắt một dòng mỗi cấp

0. **accel_top** — biên bus.  
1. **supercluster_wrapper** — 4 sub + 1 DMA + arbiter tile.  
2. **controller_system** — CSR + fetch descriptor + schedule + barrier.  
3. **subcluster_wrapper** — orchestration một “slice” compute.  
4. **glb_* + addr_gen_*** — bộ nhớ & địa chỉ theo banking.  
5. **router + window + swizzle** — dataflow & reshape.  
6. **pe_cluster (+ comparator_tree)** — primitive MAC/pool.  
7. **ppu** — quant + activation.  
8. **packages** — kiểu dữ liệu chung.
