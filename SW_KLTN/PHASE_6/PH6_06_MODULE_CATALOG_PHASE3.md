# PH6-06 — Catalog module RTL PHASE_3 (tham chiếu nhanh)

Mỗi dòng: **file** → **vai trò** → **thường được instance bởi** → **ghi chú YOLO/mapping**.

---

## packages/

| File | Vai trò | Ai dùng | Ghi chú |
|------|---------|---------|---------|
| `accel_pkg.sv` | Hằng số, `pe_mode_e`, `tile_state_e`, LANES, PE grid | Toàn IP | Map P0–P9 → `pe_mode_e` xem PH6-01 |
| `desc_pkg.sv` | `net/layer/tile` struct, profiles | controller, fetch, subcluster | Compiler sinh binary 64B |
| `csr_pkg.sv` | Offset MMIO | controller_system | |

---

## 07_system/rtl/

| File | Vai trò | Instance trong | Ghi chú |
|------|---------|----------------|---------|
| `accel_top.sv` | Top: Lite + mux m_axi + 4 SC + ctrl | SoC | |
| `controller_system.sv` | CSR, fetch, sched, barrier | accel_top | Không có tensor MAC |
| `tensor_dma.sv` | Burst AXI rd/wr | supercluster_wrapper | Tensor + có thể dùng cho PPU blob |
| `supercluster_wrapper.sv` | FIFO tile, arb, 4 sub, 1 DMA | accel_top | |

---

## 06_control/rtl/

| File | Vai trò | Instance trong | Ghi chú |
|------|---------|----------------|---------|
| `desc_fetch_engine.sv` | Đọc descriptor DDR | controller_system | |
| `global_scheduler.sv` | Tile → 4 SC | controller_system | |
| `local_arbiter.sv` | Chọn sub dùng DMA | supercluster_wrapper | |
| `tile_fsm.sv` | Pha tile, DMA req | subcluster_wrapper | Khớp HW_MAPPING §5 |
| `barrier_manager.sv` | Scoreboard barrier | controller_system | L12/L15/L18/L21 |

---

## 05_integration/rtl/

| File | Vai trò | Instance trong | Ghi chú |
|------|---------|----------------|---------|
| `subcluster_wrapper.sv` | Shell datapath + ext_* | supercluster_wrapper | Cần nối đủ GLB→PE |
| `shadow_reg_file.sv` | Snapshot desc | subcluster_wrapper | |
| `pe_cluster.sv` | 12×PE + reduce + comparator | subcluster_wrapper | P0–P3,P8 core |

---

## 04_data_movement/rtl/

| File | Vai trò | Instance trong | Ghi chú |
|------|---------|----------------|---------|
| `router_cluster.sv` | Route GLB↔PE, concat | subcluster_wrapper | P5 |
| `window_gen.sv` | Cửa sổ conv/pool | subcluster_wrapper | P0,P1,P3 |
| `swizzle_engine.sv` | Upsample, reorder | subcluster_wrapper | P6 |

---

## 03_memory/rtl/

| File | Vai trò | Instance trong | Ghi chú |
|------|---------|----------------|---------|
| `glb_input_bank.sv` | Buffer input | subcluster_wrapper | Banking §6 |
| `glb_weight_bank.sv` | Buffer weight | subcluster_wrapper | |
| `glb_output_bank.sv` | PSUM/ACT | subcluster_wrapper | Namespace PSUM vs ACT |
| `metadata_ram.sv` | Slot meta | subcluster_wrapper | |
| `addr_gen_input.sv` | Addr input | subcluster_wrapper | |
| `addr_gen_weight.sv` | Addr weight | subcluster_wrapper | |
| `addr_gen_output.sv` | Addr output | subcluster_wrapper | |

---

## 02_ppu/rtl/ (01 trong cây số cũ: 02_ppu)

| File | Vai trò | Instance trong | Ghi chú |
|------|---------|----------------|---------|
| `ppu.sv` | Bias, requant, act, e-wise | subcluster_wrapper | Sau mọi last_pass conv |

---

## 01_compute_leaf/rtl/

| File | Vai trò | Instance trong | Ghi chú |
|------|---------|----------------|---------|
| `pe_unit.sv` | MAC lanes | pe_cluster | |
| `dsp_pair_int8.sv` | 2 MAC / DSP trick | pe_unit | |
| `column_reduce.sv` | Giảm theo conv | pe_cluster | Tắt cho DW |
| `comparator_tree.sv` | Max | pe_cluster | SPPF P3 |
| `silu_lut.sv` | LUT | ppu hoặc shared | SiLU |

---

## 08_e2e/tb/

| File | Vai trò |
|------|---------|
| `tb_accel_e2e.sv` | Test hệ thống — mở rộng để so khớp PH6-03 |

---

*Cập nhật khi đổi tên thư mục 00_system vs 07_system trong repo.*
