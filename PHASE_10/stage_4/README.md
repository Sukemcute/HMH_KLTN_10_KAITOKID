# STAGE 4: Memory, Address Generation & Data Movement

> **Vai trò**: Verify hệ thống bộ nhớ on-chip (GLB) + address generators + data routing
> hoạt động đúng đắn. Đây là "hạ tầng" mà subcluster (Stage 5) sẽ dùng.

## Tại sao quan trọng?

```
Subcluster datapath = Stage 2 atoms + Stage 4 infrastructure
  GLB banks         → nơi chứa input/weight/output data
  Address generators → tính địa chỉ SRAM từ logical (h,w,c)
  Window generator   → tạo cửa sổ trượt K×K cho convolution
  Router cluster     → routing data giữa banks ↔ PE ↔ PPU
  Swizzle engine     → transform layout giữa layers (upsample, concat)
```

## 10 Modules + 1 Integration Test

### 01_memory/ — GLB Banks & Metadata

| Module | Lines | Chức năng | Key feature |
|--------|-------|-----------|-------------|
| glb_input_bank | 47 | Input SRAM: 32 subbanks × 2048 depth | Lane-masked writes |
| glb_weight_bank | 72 | Weight SRAM + 8-deep staging FIFO | FWFT discipline |
| glb_output_bank | 53 | Dual namespace: PSUM(32b) / ACT(8b) | Shared SRAM storage |
| metadata_ram | 68 | Slot validity + ring buffer pointers | Producer-consumer |

### 02_addr_gen/ — Address Generators

| Module | Lines | Chức năng | Key rule |
|--------|-------|-----------|----------|
| addr_gen_input | 75 | (h,w,c) → (bank_id, addr, padding) | bank = h mod 3 |
| addr_gen_weight | 75 | Mode-dependent weight addressing | RS3/OS1/DW3/DW7 |
| addr_gen_output | 46 | (h_out,w_out,cout) → (bank_id, addr) | bank = pe_col |

### 03_data_movement/ — Routing & Transform

| Module | Lines | Chức năng | Key feature |
|--------|-------|-----------|-------------|
| window_gen | 56 | Shift register: K=1,3,5,7 row taps | Configurable depth |
| router_cluster | 81 | RIN/RWT/RPS + bypass paths | Per-row source select |
| swizzle_engine | 179 | Layout transform (upsample/concat) | FSM-driven |

### 04_integration_test/ — Cross-module Verification

| Test | Modules tested | Chức năng |
|------|---------------|-----------|
| tb_addr_bank_integration | addr_gen_input + 3×glb_input_bank | Banking + padding end-to-end |

## Cấu trúc

```
stage_4/
├── 01_memory/
│   ├── rtl/  (4 modules)
│   └── tb/   (4 testbenches)
├── 02_addr_gen/
│   ├── rtl/  (3 modules)
│   └── tb/   (3 testbenches)
├── 03_data_movement/
│   ├── rtl/  (3 modules)
│   └── tb/   (2 testbenches)
├── 04_integration_test/
│   └── tb/   (1 integration testbench)
├── sim/compile_all.do
└── README.md
```

## Cách chạy

```bash
cd D:/HMH_KLTN/PHASE_10/stage_4/sim
vivado -mode batch -source compile_all.do
```

## Pass Criteria

| Module | Tests | Criteria |
|--------|-------|---------|
| glb_input_bank | 5 | Lane-mask, multi-addr, random 200 → 0 errors |
| glb_weight_bank | 6 | SRAM + FIFO full/empty/wrap → 0 errors |
| glb_output_bank | 5 | PSUM/ACT namespace switch → 0 errors |
| metadata_ram | 6 | Ring fill/drain/wrap/clear → 0 errors |
| addr_gen_input | 5 | Banking h mod 3, padding=zp_x → 0 errors |
| addr_gen_weight | 3 | Mode-dependent addressing → 0 errors |
| addr_gen_output | 3 | Bank selection + addr calc → 0 errors |
| window_gen | 6 | K=1,3,5,7 taps, flush, integrity → 0 errors |
| router_cluster | 5 | RIN/RWT/RPS routing, bypass → 0 errors |
| **addr+bank integration** | **3** | **End-to-end banking + padding** → **0 errors** |

## Dependencies

```
stage_2/packages/ → accel_pkg, desc_pkg (types)
stage_2/01_dsp_pair/ → dsp_pair_int8 (for integration test)
stage_2/02_pe_unit/ → pe_unit (for integration test)
```

## Sau khi ALL PASS → Stage 5 (Subcluster)

Stage 4 chứng minh: **bộ nhớ GLB đọc/ghi đúng, address tính đúng, window/routing/swizzle đúng**.
Stage 5 sẽ ghép Stage 2 + Stage 4 thành 1 subcluster hoàn chỉnh.
