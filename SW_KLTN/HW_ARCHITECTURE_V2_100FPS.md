# Kiến Trúc V2: YOLOv10n INT8 Accelerator – Target >100 FPS
## Scale-Up Analysis & Architecture Redesign

> **Phiên bản**: V2 – Nâng cấp từ V1 để đạt >100 FPS trên Virtex 7
> **FPGA target**: XC7VX690T (primary), XC7VX485T (secondary)
> **Nguyên tắc bất di bất dịch**: CHẠY ĐÚNG INFERENCE NHƯ PHẦN MỀM

---

## 1. PHÂN TÍCH YÊU CẦU: TỪ 15 FPS LÊN >100 FPS

### 1.1. Hiệu năng V1 hiện tại

```
V1 Architecture:
  16 subclusters (4 SC × 4 Sub)
  Per sub: 3 rows × 4 cols × 16 lanes = 192 INT8 MACs
  Active: 4 subclusters (1 RUNNING per SC)
  Peak: 4 × 192 = 768 MACs/cycle

  @ 200 MHz, 45% utilization:
  Effective: 768 × 200M × 0.45 = 69.1 GOPS
  HW compute: 3.0 GMAC / 69.1 GOPS = 43.4 ms → 23 FPS (HW only)
  End-to-end: ~65-80 ms → 12-15 FPS
```

### 1.2. Target >100 FPS – Yêu cầu tính toán

```
YOLOv10n backbone+neck (L0-L22): ~3.0 GMACs
  (6.7 GFLOPs total model, head ~10% → backbone+neck ≈ 3.0 GMACs)

Để đạt 100 FPS trên HW:
  T_hw < 10 ms
  Required throughput: 3.0 GMAC / 10 ms = 300 GOPS

V1 cung cấp: 69.1 GOPS
→ CẦN TĂNG ≈ 4.3× THROUGHPUT
```

### 1.3. Phân rã bottleneck V1

```
V1 bottleneck analysis:
┌──────────────────────────────────────────────────────────────────────────────┐
│ Factor               │ V1 value   │ Lý do mất hiệu suất                    │
├──────────────────────┼────────────┼─────────────────────────────────────────┤
│ Total instantiated   │ 3,072 MACs │ 16 sub × 192 MAC → phần cứng đủ lớn   │
│ Active (computing)   │ 768 MACs   │ Chỉ 25% PEs đang compute (1/4 subs)   │ ← BOTTLENECK #1
│ Spatial utilization  │ ~77%       │ LANES=16 vs feature map widths          │
│ Temporal utilization │ ~58%       │ Fill/drain/barrier stalls               │ ← BOTTLENECK #2
│ Overall utilization  │ ~45%       │ 77% × 58% = 44.7%                      │
│ Effective MACs       │ 345 /cycle │ 768 × 0.45                              │
└──────────────────────┴────────────┴─────────────────────────────────────────┘

Insight: 75% phần cứng IDLE vì 4-phase scheduler chỉ cho 1 sub RUN per SC.
         Nếu tăng active subs lên 2 per SC → 2× compute ngay lập tức.
```

---

## 2. CHIẾN LƯỢC SCALE-UP: 4 TRỤC CẢI TIẾN

### 2.1. Bốn trục scaling khả thi

```
┌───────────────────────┬──────────────┬──────────────┬──────────┬─────────────┐
│ Trục                  │ V1           │ V2 (target)  │ Hệ số    │ Trả giá     │
├───────────────────────┼──────────────┼──────────────┼──────────┼─────────────┤
│ ① LANES (data width)  │ 16           │ 32           │ 2.0×     │ +DSP, +BRAM │
│ ② Active subs/SC      │ 1            │ 2            │ 2.0×     │ +scheduler  │
│ ③ Clock frequency     │ 200 MHz      │ 200-250 MHz  │ 1.0-1.25×│ timing      │
│ ④ Utilization         │ 45%          │ 50-55%       │ 1.1-1.22×│ SW compiler │
└───────────────────────┴──────────────┴──────────────┴──────────┴─────────────┘

Tổng hệ số tối đa: 2.0 × 2.0 × 1.25 × 1.22 = 6.1× → dư sức cho 4.3×
```

### 2.2. Tại sao chọn ① LANES=32 + ② Dual-RUNNING?

**LANES 16→32 (trục ①):**
- Mỗi PE xử lý 32 spatial positions/cycle thay vì 16
- EXT_PORT_WIDTH tăng 128b→256b (2 AXI beats)
- Không thay đổi logic control (cùng #rows, #cols)
- Internal data paths: 256-bit thay vì 128-bit

**Dual-RUNNING scheduler (trục ②):**
- 4-phase scheduler gốc: RUNNING + FILLING + DRAINING + HOLD_SKIP
- V2: **RUNNING×2 + FILLING + DRAINING/HOLD**
- 2 subclusters compute từ local GLB_BANK đồng thời → không tranh chấp external port
- External port chỉ phục vụ FILLING (read) và DRAINING (write)

**Tại sao KHÔNG tăng PE_COLS (4→8)?**
- PE_COLS=8 → cần 8 bank_output, 8 PPU lanes, phức tạp hóa routing
- Lợi ích H-parallelism nhỏ (H=20 vẫn chỉ cần ceil(20/4)=5 waves)
- Giữ PE_COLS=4 → giữ nguyên bank_output/PPU/router design từ V1

**Tại sao KHÔNG dùng LANES=64?**
- W=20 (feature maps của P5): spatial util = 20/64 = 31.25% → waste 69%!
- LANES=32 tại W=20: util = 20/32 = 62.5% → chấp nhận được
- Bus 512b (64×8) gây routing congestion nặng trên Virtex 7

---

## 3. KIẾN TRÚC V2 – THÔNG SỐ CHỐT

### 3.1. So sánh V1 vs V2

```
┌───────────────────────────┬────────────┬────────────┬───────────────────────┐
│ Parameter                 │ V1         │ V2         │ Ghi chú               │
├───────────────────────────┼────────────┼────────────┼───────────────────────┤
│ LANES                     │ 16         │ 32         │ 2× spatial parallel   │
│ PE_ROWS                   │ 3          │ 3          │ Giữ nguyên (K=3)      │
│ PE_COLS                   │ 4          │ 4          │ Giữ nguyên            │
│ MACs per subcluster       │ 192        │ 384        │ 2×                    │
│ SUPER_CLUSTERS            │ 4          │ 4          │ Giữ nguyên            │
│ SUBCLUSTERS_PER_SC        │ 4          │ 4          │ Giữ nguyên            │
│ Total subclusters         │ 16         │ 16         │ Giữ nguyên            │
│ Active subs per SC        │ 1          │ 2          │ Dual-RUNNING          │
│ Total active MACs         │ 768        │ 3,072      │ 4×                    │
│ EXT_PORT_WIDTH            │ 128b       │ 256b       │ 2× (for wider lanes)  │
│ PSUM_WIDTH                │ 32         │ 32         │ Giữ nguyên            │
│ ACT_WIDTH                 │ 8          │ 8          │ Giữ nguyên            │
│ SiLU_LUT_SIZE             │ 256        │ 256        │ Giữ nguyên            │
│ Clock target              │ 200 MHz    │ 200-250MHz │ ↑ nếu timing cho phép │
│ Target utilization        │ 45%        │ 50%        │ ↑ nhờ dual-pipeline   │
└───────────────────────────┴────────────┴────────────┴───────────────────────┘
```

### 3.2. Package định nghĩa V2

```systemverilog
package accel_pkg_v2;
  parameter LANES         = 32;        // ← thay đổi chính
  parameter PE_ROWS       = 3;
  parameter PE_COLS       = 4;
  parameter INPUT_BANKS   = 3;
  parameter OUTPUT_BANKS  = 4;
  parameter WEIGHT_BANKS  = 3;
  parameter PSUM_WIDTH    = 32;
  parameter ACT_WIDTH     = 8;
  parameter EXT_PORT_WIDTH = 256;      // ← thay đổi chính
  parameter SUPER_CLUSTERS = 4;
  parameter SUBS_PER_SC   = 4;
  parameter ACTIVE_PER_SC = 2;         // ← thay đổi chính
  parameter MACS_PER_SUB  = PE_ROWS * PE_COLS * LANES;  // = 384
  parameter TOTAL_ACTIVE_MACS = SUPER_CLUSTERS * ACTIVE_PER_SC * MACS_PER_SUB; // = 3,072
endpackage
```

---

## 4. DSP48E1 RESOURCE STRATEGY: 2 MACs PER DSP

### 4.1. Thách thức: DSP48E1 không hỗ trợ INT8 SIMD natively

```
DSP48E1 (Virtex 7): 25×18 → 43-bit multiplier
DSP48E2 (UltraScale+): native INT8 SIMD (2 MACs/DSP)
→ Trên Virtex 7, cần kỹ thuật unsigned-offset packing thủ công
```

### 4.2. Kỹ thuật Unsigned-Offset Packing (2 INT8 MACs / 1 DSP48E1)

```
Bước 1: Chuyển signed INT8 → unsigned INT8
  a_u = a_signed + 128    ∈ [0, 255]
  b_u = b_signed + 128    ∈ [0, 255]

Bước 2: Pack 2 activation values vào A port (25-bit)
  A[24:0] = {a2_u[7:0], 9'b0, a1_u[7:0]}
  ─── bit 24..17: a2_unsigned ───── bit 16..8: guard zeros ─── bit 7..0: a1_unsigned ───

Bước 3: Weight vào B port (18-bit)
  B[17:0] = {10'b0, w_u[7:0]}

Bước 4: DSP multiply
  P[42:0] = A × B = {a2_u × w_u} << 17  |  {a1_u × w_u}

  Max(a1_u × w_u) = 255 × 255 = 65,025 < 2^16 = 65,536
  → product 1 fits in P[15:0], product 2 in P[32:17]
  → NO CARRY OVERLAP (guard bit P[16] always 0) ✓

Bước 5: Extract & correct
  p1_u = P[15:0]   = a1_u × w_u
  p2_u = P[32:17]  = a2_u × w_u

  a1 × w = p1_u - 128×(a1_u + w_u) + 16384
  a2 × w = p2_u - 128×(a2_u + w_u) + 16384

  Correction: 2 add/sub operations in LUTs (~25 LUTs per pair)
```

### 4.3. Ưu/nhược điểm

```
✓ 2× MAC throughput per DSP slice
✓ Works on Virtex 7 DSP48E1 (không cần UltraScale)
✓ Correction logic nhỏ (~25 LUTs per DSP)
✗ Latency +1 cycle cho correction pipeline stage
✗ Cần precompute 128×(a_u + w_u) — trivially pipelined
```

### 4.4. Phương án dự phòng: Hybrid DSP + LUT MACs

```
Nếu DSP packing quá phức tạp cho implementation:

  LANES [0:15]:   192 DSPs per sub (1 MAC/DSP, giống V1)
  LANES [16:31]:  LUT-based MACs

  Per LUT-MAC: ~80 LUTs + 40 FFs (8×8 multiply + 32-bit accumulate)
  Per sub extra lanes: 12 PEs × 16 lanes × 80 LUTs = 15,360 LUTs
  Total 16 subs: 245,760 LUTs (57% of VX690T)

  DSP usage: 16 × 192 = 3,072 DSPs (85%)
  LUT usage: 246K + 100K control = 346K (80%)
  → Fits XC7VX690T ✓ (tight on LUTs)
```

---

## 5. DUAL-RUNNING SCHEDULER: CHI TIẾT

### 5.1. So sánh V1 vs V2 scheduler

```
V1 (4-phase, 1 RUNNING):                V2 (4-phase, 2 RUNNING):
┌─────┬─────────┬─────────┬──────┐     ┌─────┬─────────┬─────────┬──────┐
│Sub-0│RUNNING  │FILLING  │DRAIN │     │Sub-0│RUNNING  │DRAIN    │FILL  │
│Sub-1│FILLING  │DRAIN    │HOLD  │     │Sub-1│RUNNING  │FILL     │DRAIN │
│Sub-2│DRAIN    │HOLD     │RUN   │     │Sub-2│FILLING  │RUNNING  │RUN   │
│Sub-3│HOLD     │RUNNING  │FILL  │     │Sub-3│DRAIN    │RUNNING  │HOLD  │
│     │  T=0    │  T=1    │ T=2  │     │     │  T=0    │  T=1    │ T=2  │
└─────┴─────────┴─────────┴──────┘     └─────┴─────────┴─────────┴──────┘
Active: 1 sub                           Active: 2 subs
```

### 5.2. Rotation protocol

```
Timeline (per SuperCluster):

Phase 0:  Sub-0 = RUNNING(tile_A)     Sub-2 = FILLING(tile_C)
          Sub-1 = RUNNING(tile_B)     Sub-3 = DRAINING(tile_prev) or HOLD

     ──── Sub-0 finishes tile_A ────

Phase 1:  Sub-0 → DRAINING(tile_A)    Sub-3 → FILLING(tile_D)
          Sub-1 = RUNNING(tile_B)     Sub-2 → RUNNING(tile_C)

     ──── Sub-1 finishes tile_B ────

Phase 2:  Sub-1 → DRAINING(tile_B)    Sub-0 → FILLING(tile_E)
          Sub-2 = RUNNING(tile_C)     Sub-3 → RUNNING(tile_D)

     ... vòng lặp ...

Key insight: 2 subs compute đồng thời từ local GLB_BANK riêng.
             External port phục vụ 1 FILL + 1 DRAIN time-multiplexed.
```

### 5.3. External port bandwidth check

```
Per tile (typical L6: QC2f, Cin=128, Cout=128, H=40, W=40):
  Input data:  6 rows × 64ch_tile × ceil(40/32) × 32B = 24,576 B
  Weight data: 9 × 64 × 64 = 36,864 B (per Cin×Cout tile)
  Output data: 4 rows × 64ch × ceil(40/32) × 32B = 16,384 B

  Fill:  (24,576 + 36,864) B / (32 B/cycle × 200 MHz) = 9.6 µs
  Drain: 16,384 B / (32 B/cycle × 200 MHz) = 2.6 µs
  Fill + Drain = 12.2 µs

  Compute: 9×64×64×40×2_Wblk = 2,949,120 MACs / 384 MACs/cycle = 7,680 cycles = 38.4 µs

  Compute >> Fill+Drain → dual-RUNNING works perfectly ✓

Per tile (small, L10: QPSA GEMM, H=20, W=20):
  Fill: ~15 µs (more weight-heavy)
  Compute: ~25 µs

  Still compute > fill → works ✓
```

### 5.4. HOLD_SKIP management trong V2

```
V1: 1 sub dedicated HOLD_SKIP (có thể giữ skip tensor trong GLB)
V2: HOLD_SKIP role xoay vòng, không dedicated → cần linh hoạt hơn

Strategy cho V2:
  Option A (preferred): GLB_BANK đủ lớn → giữ skip data trong local GLB
    - Sub chuyển từ HOLD→RUNNING: trước khi compute, check GLB có skip data
    - Skip data chỉ overwrite vùng bank_output, bank_input preserved
    - Feasible vì skip tensors (F4/F6/F8/F13) nhỏ hơn GLB capacity

  Option B (fallback): Spill skip data to DDR3
    - Khi sub cần rotate out of HOLD: DMA write skip to DDR3 skip arena
    - Khi barrier release: DMA read skip back
    - Extra DDR3 traffic: 921 KB × 2 = 1.84 MB
    - At 12.8 GB/s: 0.14 ms overhead → negligible

Skip tensor sizes (unchanged from V1):
  F4_out [1,64,80,80]   = 409,600 B
  F6_out [1,128,40,40]  = 204,800 B
  F8_out [1,256,20,20]  = 102,400 B
  F13_out [1,128,40,40] = 204,800 B
  Total: 921,600 B ≈ 900 KB

GLB_BANK per sub (V2, LANES=32): ~200 KB
→ F4 (400KB) KHÔNG vừa 1 sub → phải split hoặc dùng Option B cho F4
→ F6/F8/F13 vừa trong 1 sub GLB → Option A works
```

---

## 6. TÀI NGUYÊN FPGA – ƯỚC TÍNH CHI TIẾT

### 6.1. XC7VX690T Resource Budget

```
┌───────────────────┬──────────┬──────────────┬─────────────────────────────────┐
│ Resource          │ Available│ V2 Usage     │ Chi tiết                        │
├───────────────────┼──────────┼──────────────┼─────────────────────────────────┤
│ DSP48E1           │ 3,600    │ 3,072 (85%)  │ 16 sub × 192 DSP (2 MAC/DSP)   │
│ BRAM36K           │ 1,470    │ ~850 (58%)   │ 16 × 48 GLB + 82 misc          │
│ LUT6              │ 433,200  │ ~280K (65%)  │ 92K DSP-correct + 188K control  │
│ FF                │ 866,400  │ ~320K (37%)  │ Pipeline regs + FSM             │
└───────────────────┴──────────┴──────────────┴─────────────────────────────────┘
```

### 6.2. DSP allocation breakdown

```
Per subcluster (384 MACs, using 2-MAC packing):
  PE_CLUSTER: 12 PEs × 16 lanes_per_DSP = 192 DSPs
  (Mỗi DSP xử lý 2 lanes: lane[2i] và lane[2i+1])

Total: 16 subclusters × 192 DSPs = 3,072 DSPs
PPU requant: ~16 DSPs per sub = 256 DSPs (shared timing)
Controller misc: ~32 DSPs
Grand total: ~3,360 DSPs (93% of 3,600) ✓
```

### 6.3. BRAM allocation breakdown

```
Per subcluster GLB_BANK (LANES=32):
  bank_input[3]:
    Each bank: R_need × Cin_tile × ceil(W_max/32) × 32 bytes
    Worst case (L16: H=80, Cin=192, W=80):
      6 × 48 × 3 × 32 = 27,648 B ≈ 28 KB per bank
      3 banks = 84 KB → ceil(84K / 4.5K) = 19 BRAM36K

  bank_weight[3]:
    Each bank: kernel_data per reduction lane
    Worst case: ~16 KB per bank → 3 × 16K = 48 KB
    11 BRAM36K

  bank_output[4]:
    Each bank: Cout_tile × ceil(W/32) × 32 × 4B (INT32 PSUM) or 1B (INT8 ACT)
    Worst case (PSUM mode): 4 × 32 × 3 × 32 × 4B = 49,152 B ≈ 48 KB
    11 BRAM36K

  psum_buffer: 12 PEs × 32 lanes × 4B = 1,536 B → shared with bank_output

  metadata_RAM + SiLU_LUT: ~4 KB → 1 BRAM36K

  Per sub total: ~42 BRAM36K

Total: 16 subs × 42 = 672 BRAM36K
Desc_RAM + DMA buffers + perf counters: ~80 BRAM36K
Grand total: ~752 BRAM36K (51% of 1,470) ✓
```

### 6.4. XC7VX485T Feasibility (secondary target)

```
┌───────────────────┬──────────┬────────────┬────────────────────────────────┐
│ Resource          │ Available│ V2 Usage   │ Fit?                           │
├───────────────────┼──────────┼────────────┼────────────────────────────────┤
│ DSP48E1           │ 2,800    │ 3,360      │ ✗ EXCEEDS by 560              │
│ BRAM36K           │ 1,030    │ ~752       │ ✓ 73%                         │
│ LUT6              │ 303,600  │ ~280K      │ ✗ 92% (very tight)            │
│ FF                │ 866,400  │ ~320K      │ ✓ 37%                         │
└───────────────────┴──────────┴────────────┴────────────────────────────────┘

→ XC7VX485T KHÔNG ĐỦ cho V2 full config
→ Giải pháp: V2-lite với LANES=24 hoặc 12 subclusters
   (xem mục 6.5)
```

### 6.5. V2-lite cho XC7VX485T

```
V2-lite: LANES=24, 12 subclusters (4 SC × 3 Sub), 2 active per SC

Per sub: 3 × 4 × 24 = 288 MACs
DSPs: 12 × 144 = 1,728 (62% of 2,800) ✓
Active: 8 × 288 = 2,304 MACs
BRAMs: 12 × 36 = 432 + 80 = 512 (50%) ✓
LUTs: ~210K (69%) ✓

Performance @ 250 MHz, 50%:
  2,304 × 250M × 0.50 = 288 GOPS → T_hw = 10.4 ms → 96 FPS ✗ (gần!)

@ 250 MHz, 55%: 317 GOPS → 9.5 ms → 105 FPS ✓ (chỉ đủ nếu tối ưu)
```

---

## 7. TÍNH TOÁN FPS CHI TIẾT PER-LAYER

### 7.1. MAC counts per resolution tier

```
┌────────────────────┬────────────┬───────┬────────────────────────────────────┐
│ Resolution tier    │ Layers     │ MACs  │ Chú thích                          │
├────────────────────┼────────────┼───────┼────────────────────────────────────┤
│ 640→320 (s2)       │ L0         │  44 M │ Conv3×3, Cin=3→16                  │
│ 320→160 (s2)       │ L1         │ 118 M │ Conv3×3, Cin=16→32                 │
│ 160×160            │ L2         │ 170 M │ QC2f(32→32): OS1+RS3×2+CAT+OS1    │
│ 160→80 (s2)        │ L3         │ 118 M │ Conv3×3, Cin=32→64                 │
│ 80×80              │ L4         │ 340 M │ QC2f(64→64)                        │
│ 80→40 (SCDown)     │ L5         │  60 M │ OS1×2 + DW3×2                      │
│ 40×40              │ L6         │ 320 M │ QC2f(128→128)                      │
│ 40→20 (SCDown)     │ L7         │ 120 M │ OS1×2 + DW3×2                      │
│ 20×20              │ L8         │ 550 M │ QC2f(256→256)                      │
│ 20×20              │ L9         │ 200 M │ SPPF: OS1+MP5×3+CAT+OS1           │
│ 20×20              │ L10        │ 360 M │ QPSA: GEMM_ATTN + OS1             │
│ 40×40 (upsample)   │ L11        │   0   │ address remap only                 │
│ 40×40 (concat)     │ L12        │   0   │ data movement only                 │
│ 40×40              │ L13        │ 220 M │ QC2f(384→128)                      │
│ 80×80 (upsample)   │ L14        │   0   │ address remap only                 │
│ 80×80 (concat)     │ L15        │   0   │ data movement only                 │
│ 80×80              │ L16        │ 180 M │ QC2f(192→64)                       │
│ 80→40 (s2)         │ L17        │  37 M │ Conv3×3, Cin=64→64                 │
│ 40×40 (concat)     │ L18        │   0   │ data movement only                 │
│ 40×40              │ L19        │ 110 M │ QC2f(192→128)                      │
│ 40→20 (SCDown)     │ L20        │  30 M │ OS1 + DW3                          │
│ 20×20 (concat)     │ L21        │   0   │ data movement only                 │
│ 20×20              │ L22        │ 100 M │ QC2fCIB: OS1+DW7×3pass+CAT+OS1    │
├────────────────────┼────────────┼───────┼────────────────────────────────────┤
│ TỔNG               │ L0-L22     │3,077M │ ≈ 3.08 GMACs (khớp 6.7 GFLOPs)   │
└────────────────────┴────────────┴───────┴────────────────────────────────────┘
```

### 7.2. Spatial utilization per tier (LANES=32, PE_COLS=4)

```
Spatial utilization = (valid_W_positions / padded_W_positions)
  padded_W = ceil(W / LANES) × LANES

┌──────────┬─────┬───────────────────┬───────────────┬───────────────────┐
│ W_out    │ Wblk│ Padded W          │ W spatial util│ Layers            │
├──────────┼─────┼───────────────────┼───────────────┼───────────────────┤
│ 320      │ 10  │ 320               │ 100.0%        │ L0                │
│ 160      │  5  │ 160               │ 100.0%        │ L1, L2            │
│ 80       │  3  │ 96                │  83.3%        │ L3,L4,L15,L16,L17 │
│ 40       │  2  │ 64                │  62.5%        │ L5,L6,L12,L13,    │
│          │     │                   │               │ L18,L19            │
│ 20       │  1  │ 32                │  62.5%        │ L7-L10,L20-L22    │
└──────────┴─────┴───────────────────┴───────────────┴───────────────────┘
```

### 7.3. Per-tier compute time (V2 @ 200 MHz)

```
Active MACs = 3,072 per cycle

T_tier = MACs_tier / (3,072 × spatial_util × 200M)

┌──────────────┬───────┬────────┬────────────────────────────┐
│ Tier         │ MACs  │ S_util │ T_compute (no overhead)    │
├──────────────┼───────┼────────┼────────────────────────────┤
│ 320 (L0)     │  44 M │ 100%   │  44M/(3072×1.0×200M)=0.07ms│
│ 160 (L1-L2)  │ 288 M │ 100%   │ 288M/(3072×1.0×200M)=0.47ms│
│ 80 (L3-L4,   │ 675 M │ 83.3%  │ 675M/(3072×0.833×200M)=1.32ms│
│    L15-L17)  │       │        │                             │
│ 40 (L5-L6,   │ 740 M │ 62.5%  │ 740M/(3072×0.625×200M)=1.93ms│
│    L12-L13,  │       │        │                             │
│    L18-L19)  │       │        │                             │
│ 20 (L7-L10,  │1,330 M│ 62.5%  │1330M/(3072×0.625×200M)=3.46ms│
│    L20-L22)  │       │        │                             │
├──────────────┼───────┼────────┼────────────────────────────┤
│ TỔNG         │3,077M │  —     │ 7.25 ms (pure compute)     │
└──────────────┴───────┴────────┴────────────────────────────┘
```

### 7.4. Temporal overhead estimation

```
Temporal overhead sources:
  1. Fill/drain pipeline gap:     10% (2-phase overlap, small gaps at tile boundaries)
  2. Descriptor fetch:             3% (pipelined with compute, occasional stalls)
  3. Barrier stalls (4 points):    5% (L12/L15/L18/L21 wait for skip tensors)
  4. DW_7x7 pass 3 inefficiency:  2% (1/3 PE rows used in final pass, L22 only)
  5. Tile boundary waste:          5% (edge tiles with partial valid data)
  6. QPSA softmax float path:     2% (L10 only, small tensor 20×20)
  ─────────────────────────────────────────────────────────────
  Total temporal overhead:        ~27%
  Temporal utilization:           ~73%

Hoặc equivalently:
  T_hw = T_pure_compute / temporal_util = 7.25 / 0.73 = 9.93 ms
```

### 7.5. FPS Summary (V2 trên XC7VX690T)

```
┌────────────────────┬──────────┬────────────┬───────────┬────────────┐
│ Scenario           │ Clock    │ Util total │ Eff GOPS  │ FPS (HW)   │
├────────────────────┼──────────┼────────────┼───────────┼────────────┤
│ Conservative       │ 200 MHz  │ 45%        │ 276.5     │ ~90        │
│ ★ Realistic V2     │ 200 MHz  │ 50%        │ 307.2     │ ~101       │
│ Optimized compiler │ 200 MHz  │ 55%        │ 337.9     │ ~111       │
│ Higher clock       │ 250 MHz  │ 50%        │ 384.0     │ ~125       │
│ ★ Target sweet spot│ 220 MHz  │ 52%        │ 351.5     │ ~115       │
└────────────────────┴──────────┴────────────┴───────────┴────────────┘

★ Kết luận: Với V2 @ 200-220 MHz, đạt 100-115 FPS trên accelerator.
```

---

## 8. END-TO-END PIPELINE ANALYSIS

### 8.1. Pipeline 3 tầng

```
Frame N:   ┌─────────────────┐
           │ CPU Preprocess   │
           │ (host PC, C++)  │──→ X_int8 to DDR3
           └─────────────────┘
                    ↓ pipeline overlap
Frame N-1: ┌──────────────────────────┐
           │ Accelerator L0-L22       │
           │ (DDR3 ← DMA → accel_top) │──→ P3/P4/P5 to DDR3
           └──────────────────────────┘
                    ↓ pipeline overlap
Frame N-2: ┌─────────────────────────┐
           │ CPU Postprocess          │
           │ dequant + Qv10Detect     │
           │ + decode bbox + draw     │
           └─────────────────────────┘

Throughput = 1 / max(T_preprocess, T_hw, T_postprocess)
Latency = T_preprocess + T_hw + T_postprocess (3 frames)
```

### 8.2. Timing breakdown per stage

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Stage            │ On host PC       │ On MicroBlaze     │ On ARM (Zynq)   │
│                  │ (x86, C++, AVX2) │ (200 MHz, no FPU) │ (667 MHz, NEON) │
├──────────────────┼──────────────────┼───────────────────┼─────────────────┤
│ LetterBox/Resize │ 1.0 ms           │ 8 ms              │ 3 ms            │
│ Normalize (÷255) │ 0.5 ms           │ 5 ms              │ 1.5 ms          │
│ quantize_affine  │ 0.8 ms           │ 7 ms              │ 2 ms            │
│ DMA write input  │ 0.1 ms (PCIe)    │ 0.1 ms (local)    │ 0.1 ms          │
│ ─────────────────┼──────────────────┼───────────────────┼─────────────────┤
│ T_preprocess     │ ≈ 2.5 ms         │ ≈ 20 ms           │ ≈ 6.5 ms        │
├──────────────────┼──────────────────┼───────────────────┼─────────────────┤
│ HW Accel L0-L22  │ 9.9 ms           │ 9.9 ms            │ 9.9 ms          │
├──────────────────┼──────────────────┼───────────────────┼─────────────────┤
│ DMA read P3/P4/P5│ 0.05 ms          │ 0.05 ms           │ 0.05 ms         │
│ dequantize       │ 0.3 ms           │ 3 ms              │ 1 ms            │
│ Qv10Detect head  │ 2.0 ms           │ 12 ms             │ 4 ms            │
│ decode bbox      │ 0.5 ms           │ 3 ms              │ 1 ms            │
│ draw bbox        │ 0.5 ms           │ 2 ms              │ 1 ms            │
│ ─────────────────┼──────────────────┼───────────────────┼─────────────────┤
│ T_postprocess    │ ≈ 3.5 ms         │ ≈ 20 ms           │ ≈ 7 ms          │
├──────────────────┼──────────────────┼───────────────────┼─────────────────┤
│ THROUGHPUT       │ 1/max(2.5,9.9,   │ 1/max(20,9.9,     │ 1/max(6.5,9.9,  │
│ (pipelined)      │  3.5) = 101 FPS  │  20) = 50 FPS     │  7) = 101 FPS   │
│ LATENCY          │ 15.9 ms          │ 49.9 ms           │ 23.4 ms         │
└──────────────────┴──────────────────┴───────────────────┴─────────────────┘
```

### 8.3. Kết luận end-to-end

```
★ Host PC (x86 + FPGA qua PCIe): 101 FPS throughput ✓   (bottleneck = HW accelerator)
★ Zynq ARM + PL:                  101 FPS throughput ✓   (bottleneck = HW accelerator)
✗ MicroBlaze:                      50 FPS throughput ✗   (bottleneck = CPU pre/postprocess)

Để đạt >100 FPS end-to-end, cần:
  ① HW accelerator V2 (đã thiết kế) 
  ② CPU đủ mạnh: x86 host hoặc ARM Cortex-A9 (Zynq)
  ③ Nếu chỉ có MicroBlaze: cần thêm HW preprocessing module (xem mục 8.4)
```

### 8.4. HW Preprocessing Module (optional, cho pure-FPGA)

```
Nếu không có host PC / ARM mạnh, thêm module HW preprocessing:

module hw_preprocess (
  input  [7:0] pixel_bgr [0:2],    // raw BGR uint8
  output [7:0] x_int8              // quantized INT8
);
  // Normalize + quantize: q = round(pixel / (255 * scale)) + zp
  // Với scale ≈ 0.00392, zp = 0:
  //   q = round(pixel / 0.9996) ≈ pixel (almost identity for this scale!)
  // Hardware: fixed-point multiply by 256/255 ≈ 1.00392
  //   q = (pixel * 256 + 128) >> 8    (biased rounding)
  // → 1 multiplier + 1 adder per pixel → trivially pipelined
endmodule

Throughput: 1 pixel/cycle @ 200 MHz = 200 Mpixels/s
Input size: 3 × 640 × 640 = 1,228,800 pixels
Time: 1.23M / 200M = 6.1 ms → still a bottleneck at 6.1 ms

Fix: 4 parallel pixel lanes → 0.8M cycles → 4.1 ms → 243 FPS preprocessing → not bottleneck

Resources: 4 × (1 DSP + ~20 LUTs) = 4 DSPs + 80 LUTs → negligible
```

---

## 9. MODULES THAY ĐỔI SO VỚI V1

### 9.1. Danh sách modules cần sửa đổi

```
┌──────────────────────────┬────────────────────────────────────────────────────┐
│ Module                   │ Thay đổi                                          │
├──────────────────────────┼────────────────────────────────────────────────────┤
│ accel_pkg.sv             │ LANES=32, EXT_PORT=256, ACTIVE_PER_SC=2           │
│                          │                                                    │
│ pe_lane_mac.sv           │ 32 lanes thay vì 16; mỗi PE instantiates 16 DSPs │
│                          │ (2 MAC/DSP) thay vì 16 DSPs (1 MAC/DSP)          │
│                          │ Thêm correction logic (~25 LUTs per DSP)          │
│                          │                                                    │
│ window_gen.sv            │ Shift register 32-wide thay vì 16-wide            │
│                          │ K1/K3/K5/K7 tap generator output 32 elements      │
│                          │                                                    │
│ column_reduce.sv         │ Sum 3 rows × 32 lanes thay vì 3 × 16             │
│                          │ Output: 4 columns × 32 INT32 values               │
│                          │                                                    │
│ ppu_lite.sv              │ 32 requant lanes parallel thay vì 16              │
│                          │ SiLU LUT: 32 read ports (multi-port ROM hoặc      │
│                          │ time-multiplex 2 cycles × 16 ports)               │
│                          │                                                    │
│ glb_input_bank.sv        │ 32 subbanks per bank thay vì 16                   │
│                          │ addr_gen: Wblk_total = ceil(W/32)                  │
│                          │                                                    │
│ glb_output_bank.sv       │ 32-wide write port per bank                       │
│                          │                                                    │
│ glb_weight_bank.sv       │ 32-wide read port per lane bank                   │
│                          │                                                    │
│ router_cluster.sv        │ RIN: 32-element vectors thay vì 16                │
│                          │ RWT: 32-element weight broadcast                   │
│                          │ RPS: 32-wide psum path                             │
│                          │                                                    │
│ swizzle_engine.sv        │ 32-wide re-layout engine                           │
│                          │                                                    │
│ local_arbiter.sv         │ ★ THAY ĐỔI LỚN: dual-RUNNING scheduling          │
│                          │ Priority: RUN0 = RUN1 > FILL > DRAIN > HOLD       │
│                          │ External port arb: FILL và DRAIN time-multiplex    │
│                          │                                                    │
│ tensor_dma.sv            │ AXI4 burst width 256b thay vì 128b                │
│                          │ Hoặc: 2 × 128b bursts per cycle                   │
│                          │                                                    │
│ accel_top.sv             │ AXI port 256b master; updated parameters           │
└──────────────────────────┴────────────────────────────────────────────────────┘
```

### 9.2. Modules KHÔNG thay đổi (critical for correctness)

```
★ KHÔNG THAY ĐỔI LOGIC TÍNH TOÁN:
  - pe_lane_mac: phép MAC INT8×INT8→INT32 GIỐNG HỆT (chỉ thêm lanes)
  - ppu_lite: bias + requant + SiLU LUT GIỐNG HỆT (chỉ thêm lanes)
  - column_reduce: sum 3 rows GIỐNG HỆT (chỉ rộng hơn)
  - comparator_tree: MAXPOOL logic GIỐNG HỆT
  - barrier_manager: dependency logic GIỐNG HỆT
  - tile_fsm: FSM states GIỐNG HỆT

→ Correctness of compute ĐƯỢC BẢO TOÀN vì:
  1. Cùng phép toán (MAC, requant, SiLU, maxpool) — chỉ parallelize nhiều hơn
  2. Cùng data type (INT8 in, INT32 psum, INT8 out)
  3. Cùng quantization parameters (scale, zp, M_int, shift)
  4. Cùng rounding mode (half_up)
  5. Kết quả INDEPENDENT giữa các lanes (no cross-lane interaction in MAC)
```

---

## 10. ĐẢM BẢO TÍNH ĐÚNG ĐẮN INFERENCE

### 10.1. Tại sao V2 cho kết quả GIỐNG HỆT V1 và Golden Python?

```
CHỨNG MINH: Scale-up lanes KHÔNG ảnh hưởng kết quả

Xét 1 output element y[cout, h_out, x_out]:

  y[cout, h_out, x_out] = PPU(Σ_{kh,kw,cin} x[cin, h_in+kh, x_out+kw] × w[cout, cin, kh, kw])

Trong V1 (LANES=16):
  - x_out ∈ {0, 1, ..., 15} xử lý song song trong cycle T
  - x_out ∈ {16, 17, ..., 31} xử lý trong cycle T+1

Trong V2 (LANES=32):
  - x_out ∈ {0, 1, ..., 31} xử lý song song trong CÙNG cycle T

Kết quả y[cout, h_out, x_out] KHÔNG PHỤ THUỘC vào lane nào xử lý nó,
vì mỗi lane tính HOÀN TOÀN ĐỘC LẬP (no cross-lane data dependency).

QED: V2 output === V1 output === Golden Python output (bit-exact)
```

### 10.2. Rủi ro thay đổi và cách kiểm soát

```
┌─────────────────────────┬──────────────────────────────────────────────────┐
│ Rủi ro                  │ Kiểm soát                                       │
├─────────────────────────┼──────────────────────────────────────────────────┤
│ DSP packing correction  │ Unit test: 2-MAC DSP output vs reference MAC    │
│ (unsigned offset error) │ Test ALL corner cases: (-128×-128), (-128×127), │
│                         │ (0×0), (127×127)                                │
│                         │                                                  │
│ 32-wide bank addressing │ Verify addr_gen cho W=20 (1 Wblk, 12 padding)  │
│ (padding elements)      │ Padding must fill with zp_x, not 0             │
│                         │                                                  │
│ Dual-RUNNING race cond. │ 2 RUNNING subs access DIFFERENT GLB_BANKs      │
│                         │ No shared memory → no race condition             │
│                         │ External port: FILL/DRAIN arbitrated by         │
│                         │ local_arbiter → serialized → no conflict         │
│                         │                                                  │
│ HOLD_SKIP rotation      │ Skip data integrity check after rotation         │
│                         │ Compare with Golden Python skip tensor dump      │
│                         │                                                  │
│ 256b AXI burst align    │ Ensure burst addresses 32B-aligned               │
│                         │ ARLEN calculation correct for wider bus           │
└─────────────────────────┴──────────────────────────────────────────────────┘
```

### 10.3. Verification flow (V2-specific additions)

```
Phase 0 (NEW): DSP packing unit test
  - Testbench drives all 65536 (x, w) INT8 pairs through 2-MAC DSP
  - Compare vs behavioral: for all a,b in [-128..127]: assert DSP_out == a*b
  - Must pass 100% (no tolerance)

Phase 1: Memory correctness (giống V1, nhưng LANES=32)
  - bank_input: verify modulo-3 với 32 subbanks
  - Wblk_total = ceil(W/32) thay vì ceil(W/16)
  - Padding positions [W..32*Wblk-1] must contain zp_x

Phase 2: Primitive unit tests (giống V1)
  - Golden Python numpy dumps → hex → testbench
  - Check: V2 output == V1 output == Golden Python (bit-exact)

Phase 3: Dual-RUNNING integration test (NEW)
  - Run 2 tiles simultaneously on same SC
  - Verify both outputs correct (no interference)
  - Check external port arbitration doesn't corrupt data

Phase 4: End-to-end L0-L22 (giống V1)
  - Input: X_int8[1,3,640,640] from Golden Python
  - Output: P3[1,64,80,80], P4[1,128,40,40], P5[1,256,20,20]
  - Compare bit-exact with Golden Python
```

---

## 11. SƠ ĐỒ KIẾN TRÚC V2 (TOP-LEVEL)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              accel_top_v2                                     │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────┐                │
│  │               controller_system (unchanged logic)        │                │
│  │  CSR/MMIO + desc_fetch + barrier_mgr + DMA + sched       │                │
│  └────────────────────────┬────────────────────────────────┘                │
│             cluster_cmd[4]│ cluster_sts[4]                                   │
│  ┌──────────────────────────────────────────────────────┐                   │
│  │                 Inter-cluster Fabrics                  │                   │
│  │  IACT_fabric(N/S/E/W 256b) + WGT_fabric(E/W 256b)   │  ← widened       │
│  │  PSUM_fabric(N/S, 32×32b)                             │  ← widened       │
│  └──────────┬───────────────┬──────────────┬────────────┘                   │
│             │               │              │                                  │
│  ┌──────────▼──┐   ┌────────▼────┐  ┌─────▼──────┐  ┌───────────┐         │
│  │ SC(0,0)     │   │ SC(0,1)     │  │ SC(1,0)    │  │ SC(1,1)   │         │
│  │ 4 subcluster│   │ 4 subcluster│  │ 4 subcluster│  │ 4 subcluster│        │
│  │ 2 RUNNING   │   │ 2 RUNNING   │  │ 2 RUNNING  │  │ 2 RUNNING │  ← NEW  │
│  │ Port0 256b  │   │ Port1 256b  │  │ Port2 256b │  │ Port3 256b│  ← wider│
│  └─────────────┘   └─────────────┘  └────────────┘  └───────────┘         │
│                                                                               │
│  ┌──────────────────┐  ┌───────────────────┐  ┌──────────────────────────┐ │
│  │  desc_ram         │  │  tensor_arena DMA  │  │  perf_mon / IRQ          │ │
│  │  (NET/LAYER/TILE) │  │  (AXI4 256b master)│  │  counters + CSR readout  │ │
│  └──────────────────┘  └───────────────────┘  └──────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 11.1. Subcluster V2 internal

```
ext_beat_in (256b) ──────────────────────────────────┐
                                                       │
cfg (shadow_regs) ────────┐                           │
                           ▼                           ▼
                    ┌──────────────────────────────────────┐
                    │              GLB_BANK_V2              │
                    │  bank_input[3]: 32 subbanks each     │  ← wider
                    │  bank_weight[3]: 32-wide read port   │  ← wider
                    │  bank_output[4]: 32-wide write port  │  ← wider
                    │  swizzle_engine_v2 (32-wide)         │
                    └────────────┬─────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────────────┐
                    │       ROUTER_CLUSTER_V2             │
                    │  RIN: 3 × 32-element vectors       │  ← wider
                    │  RWT: 3 × 32-element weight ports  │  ← wider
                    │  RPS: 4 × 32-element psum paths    │  ← wider
                    └────────────┬───────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────────────┐
                    │       WINDOW_GEN_V2                 │
                    │  Shift register: 32-wide × K_max   │  ← wider
                    └────────────┬───────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────────────┐
                    │       PE_CLUSTER_V2 (3×4×32)       │
                    │  12 PEs × 16 DSP48E1 (2 MAC/DSP)  │  ← 2×lanes
                    │  + correction LUTs per DSP pair    │
                    │  column_reduce: 32-wide             │
                    └────────────┬───────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────────────┐
                    │       PPU_V2                        │
                    │  32-lane requant + SiLU_LUT         │  ← wider
                    │  (2 cycles × 16-port LUT ROM,       │
                    │   or 32-port dual-read ROM)         │
                    └────────────┬───────────────────────┘
                                 │
                                 ▼
                    bank_output[4] / swizzle / ext_beat_out (256b)
```

---

## 12. PE_LANE_MAC V2: DSP PACKING DETAIL

### 12.1. Single PE (32 lanes, using 16 DSP48E1)

```
┌───────────────────────────────────────────────────────────────────┐
│ PE[row][col] — 32 lanes                                          │
│                                                                   │
│  x_in[31:0] (INT8 activations, 32 elements)                     │
│  w_in[31:0] (INT8 weights, 32 elements)                         │
│  psum_buf[31:0] (INT32 accumulators, 32 elements)               │
│                                                                   │
│  ┌─────────────┐  ┌─────────────┐       ┌─────────────┐         │
│  │ DSP_PAIR_0  │  │ DSP_PAIR_1  │  ...  │ DSP_PAIR_15 │         │
│  │ lane[0,1]   │  │ lane[2,3]   │       │ lane[30,31] │         │
│  │ 1 DSP48E1   │  │ 1 DSP48E1   │       │ 1 DSP48E1   │         │
│  │ + correct   │  │ + correct   │       │ + correct   │         │
│  └──────┬──────┘  └──────┬──────┘       └──────┬──────┘         │
│         │                │                      │                 │
│    psum[0], psum[1]  psum[2], psum[3]   psum[30], psum[31]      │
│         │                │                      │                 │
│  ┌──────▼────────────────▼──────────────────────▼──────────────┐ │
│  │                   psum_acc[31:0]                              │ │
│  │          (32 × INT32 accumulate registers)                   │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  Total per PE: 16 DSPs + ~400 LUTs (correction) + 32×32 FFs     │
│  Total per sub (12 PEs): 192 DSPs + ~4,800 LUTs                 │
└───────────────────────────────────────────────────────────────────┘
```

### 12.2. DSP_PAIR module

```systemverilog
module dsp_pair_int8 (
  input  logic        clk, rst_n, en,
  input  logic [7:0]  x_a, x_b,      // 2 signed INT8 activations (2 lanes)
  input  logic [7:0]  w,              // 1 signed INT8 weight (shared)
  output logic [31:0] mac_a, mac_b    // 2 INT32 accumulate outputs
);
  logic [7:0]  x_a_u, x_b_u, w_u;
  logic [24:0] dsp_A;
  logic [17:0] dsp_B;
  logic [42:0] dsp_P;
  logic [15:0] prod_a_u, prod_b_u;
  logic [31:0] correction_a, correction_b;

  assign x_a_u = x_a + 8'd128;
  assign x_b_u = x_b + 8'd128;
  assign w_u   = w   + 8'd128;

  assign dsp_A = {x_b_u, 9'b0, x_a_u};
  assign dsp_B = {10'b0, w_u};

  DSP48E1 #(/* OPMODE for A*B, accumulate */) dsp_inst (
    .CLK(clk), .A({5'b0, dsp_A[24:0]}), .B(dsp_B),
    .P(dsp_P), .CEP(en), .RSTP(rst_n)
  );

  assign prod_a_u = dsp_P[15:0];
  assign prod_b_u = dsp_P[32:17];

  assign correction_a = 32'(prod_a_u) - 32'(128) * (32'(x_a_u) + 32'(w_u)) + 32'd16384;
  assign correction_b = 32'(prod_b_u) - 32'(128) * (32'(x_b_u) + 32'(w_u)) + 32'd16384;

  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) begin
      mac_a <= '0;
      mac_b <= '0;
    end else if (en) begin
      mac_a <= mac_a + correction_a;
      mac_b <= mac_b + correction_b;
    end
endmodule
```

---

## 13. DUAL-RUNNING LOCAL_ARBITER V2

### 13.1. FSM states per subcluster

```systemverilog
typedef enum logic [2:0] {
  SUB_IDLE    = 3'h0,
  SUB_RUNNING = 3'h1,    // PE computing from local GLB
  SUB_FILLING = 3'h2,    // DMA → GLB (activation + weight)
  SUB_DRAINING= 3'h3,    // GLB → DMA (output activation)
  SUB_HOLD    = 3'h4     // holding skip tensor for barrier
} sub_state_e;
```

### 13.2. Role assignment logic

```systemverilog
always_comb begin
  // Priority: ensure 2 RUNNING at all times when tiles available
  num_running = count(sub_state == SUB_RUNNING);
  num_filling = count(sub_state == SUB_FILLING);

  // When a sub finishes RUNNING:
  if (sub_done_pulse) begin
    finished_sub → SUB_DRAINING;
    if (filled_sub_ready)
      filled_sub → SUB_RUNNING;    // promote FILLED → RUNNING
    // Find idle sub → SUB_FILLING (prefetch next tile)
    if (idle_sub_available && tiles_remaining)
      idle_sub → SUB_FILLING;
  end

  // External port allocation (time-multiplex):
  ext_port_grant =
    (num_filling > 0) ? FILL_FIRST :    // FILL has priority
    (num_draining > 0) ? DRAIN :         // then DRAIN
    IDLE;

  // Within FILL phase: alternate read bursts (weight / activation)
  // Within DRAIN phase: write bursts to DDR
end
```

### 13.3. Bandwidth sharing

```
External port: 256b/cycle @ 200 MHz = 6.4 GB/s per SC

Time budget between tile completions:
  Average tile compute: ~20 µs (varies 5-50 µs)
  Fill needed: ~12 µs (input + weight)
  Drain needed: ~5 µs (output)
  Fill + Drain: ~17 µs < 20 µs → fits in 1 tile compute window ✓

Edge case (small tiles at 20×20):
  Compute: ~10 µs
  Fill + Drain: ~8 µs → tight but works
  → temporal utilization ~80% for these tiles
```

---

## 14. DDR3 MEMORY MAP V2

```
DDR3 Base: 0x0000_0000  (unchanged layout, wider burst access)

├── 0x0000_0000 – 0x000F_FFFF (1 MB):    Descriptors (NET/LAYER/TILE)
├── 0x0010_0000 – 0x002F_FFFF (2 MB):    Weight arena
├── 0x0030_0000 – 0x0043_FFFF (1.25 MB): Input tensor X_int8[1,3,640,640]
├── 0x0050_0000 – 0x006F_FFFF (2 MB):    Activation double-buffer (ping/pong)
├── 0x0080_0000 – 0x009F_FFFF:           P3 output [1,64,80,80] = 409,600 B
├── 0x00A0_0000 – 0x00AF_FFFF:           P4 output [1,128,40,40] = 204,800 B
├── 0x00B0_0000 – 0x00B2_7FFF:           P5 output [1,256,20,20] = 102,400 B
├── 0x00C0_0000 – 0x00CF_FFFF:           Skip spill arena (F4 if needed)
└── 0x00D0_0000+:                         Debug / checkpoint buffers

AXI4 bus: 256b master @ 200 MHz
  Burst: ARLEN=15, ARSIZE=5 (32 bytes), burst = 16×32 = 512 B
  DDR3 effective BW: ~12.8 GB/s (unchanged, DDR3-1600)
  Total data per inference: ~43 MB → 43M/12.8G = 3.4 ms
  → Memory NOT bottleneck ✓
```

---

## 15. TỔNG KẾT & QUYẾT ĐỊNH

### 15.1. V2 Performance Card

```
╔══════════════════════════════════════════════════════════════════╗
║  YOLOv10n INT8 Accelerator V2 — Performance Card                ║
╠══════════════════════════════════════════════════════════════════╣
║  FPGA Target:    XC7VX690T                                      ║
║  Clock:          200 MHz (conservative) / 250 MHz (aggressive)  ║
║  Active MACs:    3,072 INT8 MACs/cycle                          ║
║  Peak:           614.4 GOPS @ 200 MHz / 768.0 GOPS @ 250 MHz   ║
║  Effective:      307.2 GOPS @ 200 MHz, 50% util                ║
║  ────────────────────────────────────────────────────────────── ║
║  Workload:       3.08 GMACs (YOLOv10n backbone+neck, 640×640)  ║
║  HW compute:     ~10.0 ms @ 200 MHz / ~8.0 ms @ 250 MHz       ║
║  HW FPS:         ~100 FPS @ 200 MHz / ~125 FPS @ 250 MHz       ║
║  ────────────────────────────────────────────────────────────── ║
║  End-to-end:     ~100 FPS (with x86 host or Zynq ARM)          ║
║  ────────────────────────────────────────────────────────────── ║
║  DSP48E1:        3,360 / 3,600 (93%)                            ║
║  BRAM36K:        752 / 1,470 (51%)                              ║
║  LUT6:           280K / 433K (65%)                              ║
║  FF:             320K / 866K (37%)                              ║
║  ────────────────────────────────────────────────────────────── ║
║  Correctness:    Bit-exact with Golden Python Phase 1           ║
║                  (same arithmetic, just wider parallelism)      ║
╚══════════════════════════════════════════════════════════════════╝
```

### 15.2. So sánh V1 vs V2

```
┌──────────────────────┬──────────────┬──────────────┬────────────┐
│ Metric               │ V1           │ V2           │ Tỷ lệ      │
├──────────────────────┼──────────────┼──────────────┼────────────┤
│ Active MACs          │ 768          │ 3,072        │ 4.0×       │
│ Effective GOPS       │ 69.1         │ 307.2        │ 4.4×       │
│ HW compute time      │ 44 ms        │ 10 ms        │ 4.4×       │
│ HW FPS               │ 23           │ 101          │ 4.4×       │
│ End-to-end FPS       │ 12-15        │ 100+         │ 6.7-8.3×   │
│ DSP usage            │ 85%          │ 93%          │ ↑          │
│ BRAM usage           │ 51%          │ 51%          │ =          │
│ LUT usage            │ ~45%         │ ~65%         │ ↑          │
│ RTL complexity       │ baseline     │ +30%         │ moderate   │
│ Correctness          │ bit-exact    │ bit-exact    │ preserved  │
└──────────────────────┴──────────────┴──────────────┴────────────┘
```

### 15.3. Recommended implementation path

```
Step 1: Implement V1 architecture FIRST (simpler, prove correctness)
  - LANES=16, 1 RUNNING per SC
  - Full verification against Golden Python
  - Establish baseline FPS (~23 FPS HW)

Step 2: Upgrade to V2 (parametric changes)
  - Change LANES parameter: 16 → 32
  - Widen all data paths
  - Add DSP packing (2 MAC/DSP)
  - Verify: V2 output === V1 output (bit-exact regression test)

Step 3: Enable dual-RUNNING scheduler
  - Modify local_arbiter for 2 RUNNING
  - Test concurrency: no data corruption
  - Measure actual utilization, tune tile sizes

Step 4: Clock optimization
  - Timing analysis at 200 MHz
  - If slack permits: push to 220-250 MHz
  - Re-verify at higher clock

Target: V2 @ 200-250 MHz → 100-125 FPS HW → >100 FPS end-to-end ✓
```

---

*V2 architecture kế thừa toàn bộ primitive set, descriptor format, và verification methodology từ V1.
Sự khác biệt DUY NHẤT là data parallelism (wider lanes) và scheduling (dual-RUNNING).
Phép toán arithmetic KHÔNG THAY ĐỔI → kết quả inference GIỐNG HỆT phần mềm.*
