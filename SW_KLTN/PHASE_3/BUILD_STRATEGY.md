# PHASE 3 — RTL Build & Verification Strategy
## YOLOv10n INT8 Accelerator V2 (LANES=32, Dual-RUNNING, 3,072 MACs)

> **Nguyên tắc**: Bottom-Up Build → Unit Test → Integration Test → System Test
> **Mỗi block PHẢI pass test trước khi tích hợp vào block cha**

---

## 1. TỔNG QUAN CHIẾN THUẬT

### 1.1. Dependency Graph (Build Order)

```
Level 0: PACKAGES (accel_pkg → desc_pkg → csr_pkg)
    │
    ├──────────────────────────────────────────────────────┐
    │                                                      │
Level 1: COMPUTE LEAF                              Level 1: MEMORY LEAF
    ├── dsp_pair_int8  ★FIRST                          ├── glb_input_bank
    ├── pe_unit (uses dsp_pair)                        ├── glb_weight_bank
    ├── column_reduce                                  ├── glb_output_bank
    ├── comparator_tree                                ├── metadata_ram
    └── silu_lut                                       ├── addr_gen_input
         │                                             ├── addr_gen_weight
         │                                             └── addr_gen_output
         │                                                  │
Level 2: PPU                    Level 2: DATA MOVEMENT      │
    └── ppu (uses silu_lut)         ├── window_gen          │
         │                          ├── router_cluster      │
         │                          └── swizzle_engine      │
         │                               │                  │
         └──────────────┬────────────────┘──────────────────┘
                        │
Level 3: INTEGRATION
    ├── pe_cluster (pe_unit × 12 + column_reduce + comparator_tree)
    ├── shadow_reg_file
    └── subcluster_wrapper (ALL above + tile_fsm)
                        │
Level 4: CONTROL
    ├── tile_fsm
    ├── barrier_manager
    ├── local_arbiter
    ├── desc_fetch_engine
    └── global_scheduler
                        │
Level 5: SYSTEM
    ├── supercluster_wrapper (4 × subcluster + arbiter)
    ├── tensor_dma
    ├── controller_system
    └── accel_top
                        │
Level 6: END-TO-END
    └── Full L0→L22 Inference Test
```

### 1.2. Nguyên tắc xây dựng mỗi Block

```
┌─────────────────────────────────────────────────────────────────────┐
│  BLOCK BUILD PROTOCOL (áp dụng cho MỌI module)                     │
│                                                                     │
│  Step 1: DEFINE I/O CONTRACT                                        │
│    - Input: data types, ranges, timing (valid/ready protocol)       │
│    - Output: data types, ranges, latency (cycles from input→output) │
│    - Config: parameters, modes supported                            │
│                                                                     │
│  Step 2: WRITE RTL                                                  │
│    - Implement logic theo spec                                      │
│    - Parameterize mọi thứ (LANES, DEPTH, WIDTH...)                 │
│    - Thêm assertions (SVA) cho critical invariants                 │
│                                                                     │
│  Step 3: WRITE TESTBENCH                                           │
│    - Self-checking testbench (tự so sánh output với golden)        │
│    - Edge cases: min/max values, boundary conditions               │
│    - Random stress test: N random vectors                          │
│    - Mode coverage: test ALL supported modes                       │
│                                                                     │
│  Step 4: SIMULATE & VERIFY                                         │
│    - 0 mismatches required → PASS                                  │
│    - Waveform check cho timing/pipeline behavior                   │
│    - Coverage check: tất cả modes/paths đã test                    │
│                                                                     │
│  Step 5: SIGN-OFF                                                  │
│    - Ghi nhận: module_name PASS/FAIL, date, notes                  │
│    - Nếu FAIL: fix → re-test → loop until PASS                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. STEP-BY-STEP CHI TIẾT

---

### STEP 0: Packages (Nền tảng)

**Files**: `accel_pkg.sv`, `desc_pkg.sv`, `csr_pkg.sv`

| Aspect       | Detail                                             |
|--------------|-----------------------------------------------------|
| **Mục đích** | Định nghĩa constants, types, structs dùng chung    |
| **Test**     | Compile-only (no simulation needed)                 |
| **Pass khi** | Tất cả modules import được, không lỗi compile       |

```
Compile command:
  vlog packages/accel_pkg.sv packages/desc_pkg.sv packages/csr_pkg.sv
  → 0 errors, 0 warnings = PASS
```

---

### STEP 1: Compute Leaf — Block đầu tiên và quan trọng nhất

#### 1A. `dsp_pair_int8` — ★ XÂY DỰNG ĐẦU TIÊN

**Lý do xây trước**: Đây là primitive nhỏ nhất, nền tảng của toàn bộ compute.
Nếu sai ở đây → toàn bộ hệ thống sai.

```
┌─────────────────────────────────────────────────────────┐
│ dsp_pair_int8 — I/O Contract                            │
│                                                         │
│ INPUT:                                                  │
│   clk, rst_n, en, clear                                 │
│   x_a  : signed [7:0]  — activation lane 2i             │
│   x_b  : signed [7:0]  — activation lane 2i+1           │
│   w    : signed [7:0]  — shared weight                  │
│                                                         │
│ PROCESSING:                                             │
│   Pipeline 4 stages:                                    │
│   S1: unsigned convert (x+128, w+128)                   │
│   S2: DSP48E1 multiply (pack 2 acts into A port)        │
│   S3: extract products + correction                     │
│   S4: accumulate (psum += product, or psum = product)   │
│                                                         │
│ OUTPUT:                                                 │
│   psum_a : signed [31:0] — accumulated MAC lane 2i      │
│   psum_b : signed [31:0] — accumulated MAC lane 2i+1    │
│   Latency: 4 cycles (pipeline)                          │
│                                                         │
│ CRITICAL INVARIANT:                                     │
│   psum_a == Σ(x_a[t] × w[t]) for all enabled cycles    │
│   psum_b == Σ(x_b[t] × w[t]) for all enabled cycles    │
│   Error tolerance: 0 (EXACT integer arithmetic)         │
└─────────────────────────────────────────────────────────┘

TEST PLAN:
  ① Exhaustive: all 256×256 = 65,536 (x,w) pairs → verify product
  ② Corner: (-128×-128), (-128×127), (127×127), (0×anything)
  ③ Accumulation: 9 MACs (typical conv3×3) → verify sum
  ④ Clear: mid-accumulation clear → psum resets correctly
  ⑤ Enable: en=0 → psum holds (no change)

PASS CRITERIA: 0 mismatches out of 65,536+ test vectors
```

#### 1B. `pe_unit` — Single PE (32 lanes)

```
┌─────────────────────────────────────────────────────────┐
│ pe_unit — I/O Contract                                  │
│                                                         │
│ INPUT:                                                  │
│   x_in[32]  : signed [7:0]  — 32 activations           │
│   w_in[32]  : signed [7:0]  — 32 weights               │
│   en, clear_psum, mode                                  │
│                                                         │
│ PROCESSING:                                             │
│   Instantiate 16 × dsp_pair_int8                        │
│   Lane 2i → dsp_pair[i].x_a                            │
│   Lane 2i+1 → dsp_pair[i].x_b                          │
│   Weight → dsp_pair[i].w (per-lane or broadcast)        │
│                                                         │
│ OUTPUT:                                                 │
│   psum_out[32] : signed [31:0]                          │
│   psum_valid                                            │
│   Latency: 4 cycles (inherited from dsp_pair)           │
│                                                         │
│ THỎA MÃN KHI NHÚNG VÀO PE_CLUSTER:                    │
│   psum_out[lane] == Σ(x_in[lane][t] × w_in[lane][t])   │
│   cho tất cả 32 lanes INDEPENDENT                       │
└─────────────────────────────────────────────────────────┘

TEST PLAN:
  ① 32-lane parallel MAC: random vectors × 10 cycles → verify all lanes
  ② Mode RS3: shared weight per DSP pair (lanes 2i/2i+1)
  ③ Mode OS1: broadcast weight (1×1 conv)
  ④ Clear/Enable control: verify independence
```

#### 1C. `column_reduce` — Cross-Row Reduction

```
┌─────────────────────────────────────────────────────────┐
│ column_reduce — I/O Contract                            │
│                                                         │
│ INPUT:                                                  │
│   pe_psum[3][4][32] : signed [31:0]                     │
│   (3 rows × 4 cols × 32 lanes)                         │
│                                                         │
│ PROCESSING:                                             │
│   For each (col, lane):                                 │
│     col_psum[col][lane] = Σ_{row=0}^{2} pe_psum[row]   │
│                                                         │
│ OUTPUT:                                                 │
│   col_psum[4][32] : signed [31:0]                       │
│   Latency: 1 cycle (combinational + register)           │
│                                                         │
│ THỎA MÃN:                                              │
│   col_psum[c][l] == pe_psum[0][c][l]                   │
│                    + pe_psum[1][c][l]                    │
│                    + pe_psum[2][c][l]                    │
└─────────────────────────────────────────────────────────┘

TEST: Random 3×4×32 INT32 values → verify sums. Edge: max INT32 overflow check.
```

#### 1D. `comparator_tree` — MAXPOOL

```
┌─────────────────────────────────────────────────────────┐
│ comparator_tree — I/O Contract                          │
│                                                         │
│ INPUT:  data_in[25][32] : signed [7:0] (5×5 × 32 lanes)│
│ OUTPUT: max_out[32]     : signed [7:0] (max per lane)   │
│ Latency: 5 cycles (pipelined tree)                      │
│                                                         │
│ THỎA MÃN:                                              │
│   max_out[l] == max(data_in[0..24][l])                  │
│   Per-lane INDEPENDENT (no cross-lane)                  │
└─────────────────────────────────────────────────────────┘

TEST:
  ① Known max at different positions (first, last, middle)
  ② All same values → output = that value
  ③ Signed edge: mix of -128 and 127
```

#### 1E. `silu_lut` — SiLU Lookup

```
┌─────────────────────────────────────────────────────────┐
│ silu_lut — I/O Contract                                 │
│                                                         │
│ INPUT (load): load_en, load_addr[8], load_data[8]       │
│ INPUT (read): idx[32] : [7:0] (32 parallel lookups)     │
│ OUTPUT:       out[32] : signed [7:0]                    │
│                                                         │
│ THỎA MÃN:                                              │
│   out[lane] == ROM[idx[lane]] after preload             │
│   32 simultaneous reads in 1 cycle                      │
└─────────────────────────────────────────────────────────┘

TEST:
  ① Preload 256 entries → read back each → verify
  ② 32 simultaneous reads with different indices
  ③ Boundary: idx=0 and idx=255
```

---

### STEP 2: PPU — Post-Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ ppu — I/O Contract                                                  │
│                                                                     │
│ INPUT:                                                              │
│   psum_in[32] : signed [31:0]  — raw accumulation from PE cluster  │
│   bias_val[32]: signed [31:0]  — per-channel bias                  │
│   m_int[32]   : signed [31:0]  — requant multiplier                │
│   shift[32]   : [5:0]          — requant shift amount              │
│   zp_out      : signed [7:0]   — output zero-point                 │
│   silu_lut[256]                 — preloaded LUT                    │
│   ewise_in[32]: signed [7:0]   — skip connection (optional)        │
│   cfg_post    : post_profile_t — mode config                       │
│                                                                     │
│ PROCESSING (4-stage pipeline):                                      │
│   Stage 1: biased  = psum_in + bias_val                            │
│   Stage 2: y_raw   = (biased × m_int) >>> shift (with rounding)    │
│   Stage 3: y_act   = SiLU(y_raw) or ReLU(y_raw) or y_raw          │
│   Stage 4: act_out = clamp(y_act + ewise + zp_out, -128, 127)     │
│                                                                     │
│ OUTPUT:                                                             │
│   act_out[32] : signed [7:0]  — final INT8 activation              │
│   Latency: 4 cycles                                                │
│                                                                     │
│ THỎA MÃN KHI NHÚNG VÀO SUBCLUSTER:                                │
│   act_out[ch] == clamp(SiLU(requant(psum[ch]+bias[ch])) + ewise, INT8)│
│   Must match Golden Python per-channel exactly                      │
│                                                                     │
│ CRITICAL:                                                           │
│   - Requant rounding: half_up (add 1<<(shift-1) before >>)         │
│   - SiLU index: clamp(y_raw + 128, 0, 255)                        │
│   - Final clamp: saturating arithmetic [-128, 127]                 │
└─────────────────────────────────────────────────────────────────────┘

TEST PLAN:
  ① Per-channel requant: known psum + known M/shift → verify INT8 output
  ② SiLU activation: verify LUT indexing matches Python precomputed
  ③ ReLU activation: negative→0, positive→pass-through
  ④ Element-wise add: with saturation at boundaries
  ⑤ Full pipeline: bias+requant+SiLU+ewise → compare Golden Python dump
```

---

### STEP 3: Memory Modules

#### Tất cả 7 modules cùng Level — Build song song

```
Module              | Input (Write)              | Output (Read)           | Key Invariant
--------------------|----------------------------|-------------------------|---------------------------
glb_input_bank      | 256b (32×INT8) + mask      | 256b (32×INT8)          | wr→rd same addr = same data
glb_weight_bank     | 256b + FIFO push           | 256b + FIFO pop         | FIFO FWFT ordering preserved
glb_output_bank     | 1024b PSUM or 256b ACT     | 1024b or 256b           | Namespace switch correct
metadata_ram        | set_valid + meta           | query_valid + meta      | Ring buffer no overflow
addr_gen_input      | (h,w,c) logical            | (bank_id, addr, pad?)   | h%3→bank, padding=zp_x
addr_gen_weight     | (kr, cin, cout) + mode     | (bank_id, addr)         | Per-mode address unique
addr_gen_output     | (h_out, w_out, cout)       | (bank_id, addr)         | bank = pe_col index
```

**CRITICAL TEST cho addr_gen_input**:
```
PHẢI kiểm tra:
  ① padding positions output zp_x (KHÔNG PHẢI 0!)
  ② Không có address collision: mọi (h,w,c) unique → unique physical addr
  ③ bank_id = h mod 3 (banking rule)
  ④ Stride support: stride=2 → skip alternate positions
```

---

### STEP 4: Data Movement

```
Module              | Input                      | Output                  | Key Invariant
--------------------|----------------------------|-------------------------|---------------------------
window_gen          | 32-wide INT8 stream        | K taps × 32-wide       | Shift register correct
router_cluster      | bank reads + profile       | PE act/wgt + bank write | Mode-dependent routing
swizzle_engine      | bank_output reads          | bank_input writes       | Upsample: 4 dst per src
```

**window_gen test quan trọng**:
```
Feed sequence: row0, row1, row2, row3...
K=3: output taps = [row_n, row_n-1, row_n-2]  (3 consecutive rows)
K=1: output taps = [row_n]                     (pass-through)
K=7: output taps = [row_n ... row_n-6]         (7 rows, DW7x7)

Verify: taps_valid asserted only after K rows accumulated
```

---

### STEP 5: Integration — PE Cluster

```
┌─────────────────────────────────────────────────────────────────────┐
│ pe_cluster — Integration I/O Contract                               │
│                                                                     │
│ INPUT:                                                              │
│   act_taps[3][32] : from window_gen → 3 PE rows                    │
│   wgt_data[3][32] : from router → 3 reduction lanes                │
│   psum_in[4][32]  : from bank_output (multi-pass accumulation)     │
│   mode            : PE_RS3/OS1/DW3/DW7/MP5/GEMM                   │
│                                                                     │
│ OUTPUT:                                                             │
│   psum_out[4][32] : 4 PE cols × 32 lanes → to bank_output/PPU     │
│   pool_out[32]    : MAXPOOL result (if mode=MP5)                   │
│                                                                     │
│ THỎA MÃN KHI NHÚNG VÀO SUBCLUSTER:                                │
│   RS3: psum_out[col][lane] = Σ over 3 kernel rows of (x × w)      │
│   OS1: psum_out[col][lane] = Σ over 3 cin_slices of (x × w)       │
│   DW3: psum_out[col][lane] = Σ over 3 kernel rows (per-channel)   │
│   MP5: pool_out[lane] = max(25 inputs per lane)                    │
└─────────────────────────────────────────────────────────────────────┘

TEST PLAN — Per Mode:
  RS3: Conv3×3 stride=1, Cin=8, Cout=4, H=8, W=32 → Golden Python
  OS1: Conv1×1, Cin=32, Cout=4, H=1, W=32 → Golden Python
  DW3: DWConv3×3, C=32, H=8, W=32 → Golden Python
  MP5: MaxPool5×5, C=32, H=8, W=32 → Golden Python
  Multi-pass: 2 Cin passes → psum accumulates correctly
```

---

### STEP 6: Control — FSM & Scheduling

```
Module              | Input                       | Output                   | Key Test
--------------------|-----------------------------|--------------------------|---------
tile_fsm            | tile_desc + signals          | control signals + state  | FSM transitions
barrier_manager     | signal + wait                | grant + scoreboard       | 4 YOLOv10n barriers
local_arbiter       | sub states + tile queue      | role assignments         | Dual-RUNNING rotation
desc_fetch_engine   | AXI read data                | parsed descriptors       | NET→LAYER→TILE parse
global_scheduler    | tile descs from fetch        | tile dispatch to 4 SCs   | sc_mask routing
```

**tile_fsm — FSM State Transition Test**:
```
IDLE → LOAD_CFG → PREFILL_WT → PREFILL_IN → WAIT_READY → 
RUN_COMPUTE → ACCUMULATE → (loop for multi-pass) →
POST_PROCESS → SWIZZLE_STORE → DONE → IDLE

Must test:
  ① Normal 1-pass tile (RS3, small)
  ② Multi-pass: 3 Cin passes → ACCUMULATE loops 3 times
  ③ DW7 3 K passes → ACCUMULATE loops 3 times
  ④ Barrier before: waits for barrier_grant
  ⑤ Barrier after: signals barrier_signal
  ⑥ NEED_SWIZZLE: waits for swizzle_done
  ⑦ NEED_SPILL: waits for dma_wr_done
```

---

### STEP 7: System Top

```
┌─────────────────────────────────────────────────────────────────────┐
│ accel_top — System I/O Contract                                     │
│                                                                     │
│ INPUT:                                                              │
│   AXI-Lite MMIO: CPU writes CSR (start, net_desc_base, etc.)       │
│   AXI4 Master: DDR3 responses (descriptors, weights, activations)  │
│                                                                     │
│ OUTPUT:                                                             │
│   AXI4 Master: DDR3 writes (output activations P3/P4/P5)           │
│   IRQ: inference complete                                           │
│                                                                     │
│ END-TO-END FLOW:                                                    │
│   1. CPU writes net_desc_base to CSR                                │
│   2. CPU writes start=1                                             │
│   3. Accelerator fetches NET_DESC → LAYER_DESCs → TILE_DESCs       │
│   4. Tiles dispatched to 4 SCs, processed L0→L22                   │
│   5. P3/P4/P5 written to DDR3                                      │
│   6. IRQ asserted → CPU reads P3/P4/P5                             │
│                                                                     │
│ PASS CRITERIA:                                                      │
│   P3[1,64,80,80]  bit-exact match Golden Python                    │
│   P4[1,128,40,40] bit-exact match Golden Python                    │
│   P5[1,256,20,20] bit-exact match Golden Python                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. FILE STRUCTURE

```
PHASE_3/
├── BUILD_STRATEGY.md          ← this file
├── packages/
│   ├── accel_pkg.sv           ← constants, types, enums
│   ├── desc_pkg.sv            ← descriptor structs
│   └── csr_pkg.sv             ← CSR register map
│
├── 01_compute_leaf/
│   ├── rtl/
│   │   ├── dsp_pair_int8.sv   ← 2-MAC DSP48E1 primitive
│   │   ├── pe_unit.sv         ← 32-lane PE (16 DSP pairs)
│   │   ├── column_reduce.sv   ← 3-row reduction
│   │   ├── comparator_tree.sv ← 25-input max (MAXPOOL)
│   │   └── silu_lut.sv        ← 256-entry LUT ROM
│   └── tb/
│       ├── tb_dsp_pair_int8.sv
│       ├── tb_pe_unit.sv
│       ├── tb_column_reduce.sv
│       ├── tb_comparator_tree.sv
│       └── tb_silu_lut.sv
│
├── 02_ppu/
│   ├── rtl/
│   │   └── ppu.sv
│   └── tb/
│       └── tb_ppu.sv
│
├── 03_memory/
│   ├── rtl/
│   │   ├── glb_input_bank.sv
│   │   ├── glb_weight_bank.sv
│   │   ├── glb_output_bank.sv
│   │   ├── metadata_ram.sv
│   │   ├── addr_gen_input.sv
│   │   ├── addr_gen_weight.sv
│   │   └── addr_gen_output.sv
│   └── tb/
│       ├── tb_glb_input_bank.sv
│       ├── tb_addr_gen_input.sv
│       └── ...
│
├── 04_data_movement/
│   ├── rtl/
│   │   ├── window_gen.sv
│   │   ├── router_cluster.sv
│   │   └── swizzle_engine.sv
│   └── tb/
│       └── ...
│
├── 05_integration/
│   ├── rtl/
│   │   ├── pe_cluster.sv
│   │   ├── shadow_reg_file.sv
│   │   └── subcluster_wrapper.sv
│   └── tb/
│       └── tb_pe_cluster.sv
│
├── 06_control/
│   ├── rtl/
│   │   ├── tile_fsm.sv
│   │   ├── barrier_manager.sv
│   │   ├── local_arbiter.sv
│   │   ├── desc_fetch_engine.sv
│   │   └── global_scheduler.sv
│   └── tb/
│       └── ...
│
├── 07_system/
│   ├── rtl/
│   │   ├── supercluster_wrapper.sv
│   │   ├── tensor_dma.sv
│   │   ├── controller_system.sv
│   │   └── accel_top.sv
│   └── tb/
│       └── ...
│
├── 08_e2e/
│   └── tb/
│       └── tb_accel_e2e.sv
│
└── sim_scripts/
    ├── compile_all.do         ← Vivado/ModelSim compile script
    └── run_step1.do           ← Run Step 1 tests
```

---

## 4. VERIFICATION CHECKLIST

```
STEP 0: Packages
  ☐ accel_pkg.sv compiles
  ☐ desc_pkg.sv compiles (imports accel_pkg)
  ☐ csr_pkg.sv compiles

STEP 1: Compute Leaf
  ☐ dsp_pair_int8:  65,536 pairs, 0 mismatch
  ☐ dsp_pair_int8:  accumulation 9 cycles, clear test
  ☐ pe_unit:        32-lane MAC, random 100 vectors
  ☐ column_reduce:  3×4×32 reduction, 1000 random
  ☐ comparator_tree: 25×32 max, position sweep
  ☐ silu_lut:       256-entry load + 32-port read

STEP 2: PPU
  ☐ ppu bias+requant: 100 vectors, half_up rounding
  ☐ ppu SiLU:       LUT index correctness
  ☐ ppu ReLU:       negative clamp
  ☐ ppu ewise_add:  saturation at ±128
  ☐ ppu full pipe:  match Golden Python (10 test cases)

STEP 3: Memory
  ☐ glb_input_bank:  write+read 32 subbanks, lane mask
  ☐ glb_weight_bank: SRAM + FIFO ordering
  ☐ glb_output_bank: PSUM↔ACT mode switch
  ☐ addr_gen_input:  padding=zp_x, no collision, bank=h%3
  ☐ addr_gen_weight: RS3/OS1/DW3/DW7 patterns
  ☐ addr_gen_output: bank=pe_col, slot rotation

STEP 4: Data Movement
  ☐ window_gen:      K1/K3/K5/K7 tap generation
  ☐ router_cluster:  RS3/OS1/DW3/bypass modes
  ☐ swizzle_engine:  upsample 2×, concat offset

STEP 5: Integration
  ☐ pe_cluster RS3:  Conv3×3 8×32 tile = Golden Python
  ☐ pe_cluster OS1:  Conv1×1 32-ch tile
  ☐ pe_cluster DW3:  DWConv3×3 32-ch tile
  ☐ pe_cluster MP5:  MaxPool5×5

STEP 6: Control
  ☐ tile_fsm:        FSM state transitions (all paths)
  ☐ tile_fsm:        multi-pass accumulation
  ☐ barrier_manager: 4 barriers signal/wait
  ☐ local_arbiter:   dual-RUNNING rotation
  ☐ desc_fetch:      parse NET/LAYER/TILE descriptors

STEP 7: System
  ☐ subcluster:      1 tile RS3 end-to-end
  ☐ supercluster:    4 subs, role rotation
  ☐ accel_top:       L0 inference bit-exact

STEP 8: End-to-End
  ☐ L0→L22:          P3/P4/P5 bit-exact Golden Python
  ☐ Barriers:        L4→L15, L6→L12, L8→L21, L13→L18
  ☐ Performance:     cycle count vs theoretical
```

---

## 5. SIMULATION TOOLS

```
Option A: Vivado Simulator (xvlog + xelab + xsim)
  Pro: Free, Xilinx-native, DSP48E1 simulation models
  Con: Slower than commercial

Option B: ModelSim/QuestaSim
  Pro: Fastest simulation, best debugging
  Con: License cost

Option C: Icarus Verilog (iverilog)
  Pro: Free, fast compile
  Con: Limited SystemVerilog support

RECOMMENDED: Vivado Simulator for RTL development
  → Has DSP48E1 behavioral model built-in
  → Directly targets XC7VX690T
```

---

*Mỗi STEP phải PASS 100% trước khi tiến sang STEP tiếp theo.*
*Nếu 1 module FAIL → fix → re-test → PASS → mới move on.*
