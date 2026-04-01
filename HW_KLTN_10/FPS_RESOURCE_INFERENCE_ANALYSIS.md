# Phân Tích Chi Tiết: FPS, Resource FPGA, và Khả Năng Inference
## YOLOv10n INT8 Accelerator — Kiến trúc V3-VC707

> **Target**: VC707 Board (XC7VX485T) @ 200 MHz
> **Model**: qYOLOv10n PTQ, INT8, Backbone+Neck L0-L22
> **Kiến trúc**: V3-VC707 (Eyeriss v2 per-column routing)

---

## 1. CÔNG THỨC TÍNH FPS — TỪ MODEL ĐẾN FRAMERATE

### 1.1. Chuỗi công thức tổng quan

```
BƯỚC 1: Xác định lượng tính toán model
═══════════════════════════════════════

  YOLOv10n backbone+neck (L0-L22):

  ┌────────────────────┬────────────┬──────────────────────────────────┐
  │ Resolution tier    │ Layers     │ MACs (Multiply-Accumulate)       │
  ├────────────────────┼────────────┼──────────────────────────────────┤
  │ 640→320 (s2)       │ L0         │  44,000,000                      │
  │ 320→160 (s2)       │ L1         │ 118,000,000                      │
  │ 160×160            │ L2         │ 170,000,000                      │
  │ 160→80 (s2)        │ L3         │ 118,000,000                      │
  │ 80×80              │ L4         │ 340,000,000                      │
  │ 80→40 (SCDown)     │ L5         │  60,000,000                      │
  │ 40×40              │ L6         │ 320,000,000                      │
  │ 40→20 (SCDown)     │ L7         │ 120,000,000                      │
  │ 20×20              │ L8         │ 550,000,000                      │
  │ 20×20              │ L9         │ 200,000,000                      │
  │ 20×20              │ L10        │ 360,000,000                      │
  │ 40×40 (upsample)   │ L11        │   0 (address remap only)         │
  │ 40×40 (concat)     │ L12        │   0 (data movement only)         │
  │ 40×40              │ L13        │ 220,000,000                      │
  │ 80×80 (upsample)   │ L14        │   0                              │
  │ 80×80 (concat)     │ L15        │   0                              │
  │ 80×80              │ L16        │ 180,000,000                      │
  │ 80→40 (s2)         │ L17        │  37,000,000                      │
  │ 40×40 (concat)     │ L18        │   0                              │
  │ 40×40              │ L19        │ 110,000,000                      │
  │ 40→20 (SCDown)     │ L20        │  30,000,000                      │
  │ 20×20 (concat)     │ L21        │   0                              │
  │ 20×20              │ L22        │ 100,000,000                      │
  ├────────────────────┼────────────┼──────────────────────────────────┤
  │ TỔNG               │ L0-L22     │ 3,077,000,000 MACs               │
  └────────────────────┴────────────┴──────────────────────────────────┘

  Quy đổi:
    Total_MACs  = 3,077,000,000 = 3.077 GMACs
    Total_FLOPs = 3.077 × 2     = 6.154 GFLOPs  (1 MAC = 1 multiply + 1 add = 2 FLOPs)

  Lưu ý:
    "6.7 GFLOPs" thường thấy trong paper = TOÀN model (backbone + neck + head)
    Backbone+Neck riêng (L0-L22) ≈ 6.15 GFLOPs = 3.08 GMACs
    Detection Head (L23) ≈ 0.5 GFLOPs → chạy trên CPU, không tính
```

```
BƯỚC 2: Tính Peak Throughput của phần cứng
═══════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │  Peak_GOPS = MACs_per_cycle × Frequency                        │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘

  Phân rã MACs_per_cycle:

    Mỗi DSP pair:      2 MACs (2-MAC packing trên DSP48E1)
    Mỗi PE unit:       16 DSP pairs × 2 = 32 MACs
    Mỗi PE cluster:    12 PEs (3 rows × 4 cols) = 12 × 32 = 384 MACs
    Mỗi subcluster:    1 PE cluster = 384 MACs
    Active subclusters: 8 (Dual-RUNNING: 4 SC × 2 active per SC)

    MACs_per_cycle = 8 × 384 = 3,072

  Peak throughput:
    Peak_GOPS = 3,072 × 200,000,000 = 614,400,000,000 = 614.4 GOPS
```

```
BƯỚC 3: Tính Effective Throughput (trừ overhead)
═══════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │  Effective_GOPS = Peak_GOPS × Utilization                      │
  │                                                                 │
  │  Utilization = Spatial_util × Temporal_util                     │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘

  ── Spatial Utilization ──
  Phụ thuộc: feature map width vs LANES (32)
  Khi W < LANES: các lane thừa tính "padding" → lãng phí

  ┌───────────┬──────┬────────────┬──────────────┬────────────────────┐
  │ W_out     │ Wblk │ Padded W   │ Spatial util │ Layers             │
  ├───────────┼──────┼────────────┼──────────────┼────────────────────┤
  │ 320       │ 10   │ 320        │ 100.0%       │ L0                 │
  │ 160       │ 5    │ 160        │ 100.0%       │ L1, L2             │
  │ 80        │ 3    │ 96         │ 83.3%        │ L3,L4,L15,L16,L17  │
  │ 40        │ 2    │ 64         │ 62.5%        │ L5,L6,L12-L13,L18-19│
  │ 20        │ 1    │ 32         │ 62.5%        │ L7-L10,L20-L22     │
  └───────────┴──────┴────────────┴──────────────┴────────────────────┘

  Weighted average (theo MACs per tier):
    S_weighted = Σ(MACs_tier × S_tier) / Σ(MACs_tier)
               = (44M×1.0 + 288M×1.0 + 675M×0.833 + 740M×0.625 + 1330M×0.625) / 3077M
               = (44 + 288 + 562.3 + 462.5 + 831.3) / 3077
               = 2188.1 / 3077
               = 71.1%

  ── Temporal Utilization ──
  Các nguồn overhead:

  ┌─────────────────────────────────┬────────┬───────────────────────────────┐
  │ Nguồn overhead                  │ Loss   │ Lý do                         │
  ├─────────────────────────────────┼────────┼───────────────────────────────┤
  │ Fill/drain pipeline gap         │ 10%    │ Dual-RUNNING giảm nhưng       │
  │                                 │        │ không loại hoàn toàn          │
  │ Descriptor fetch                │ 3%     │ Pipelined, stall hiếm         │
  │ Barrier stalls (4 points)       │ 5%     │ L12/L15/L18/L21 wait skip    │
  │ DW_7x7 pass-3 inefficiency     │ 2%     │ 1/3 PE rows active (L22)     │
  │ Tile boundary waste             │ 3%     │ Edge tiles partial valid      │
  │ QPSA complex path              │ 2%     │ L10 sequential ops            │
  ├─────────────────────────────────┼────────┼───────────────────────────────┤
  │ Total overhead                  │ ~25%   │                               │
  │ Temporal utilization            │ ~75%   │                               │
  └─────────────────────────────────┴────────┴───────────────────────────────┘

  ── Overall Utilization ──
    Overall_util = Spatial_util × Temporal_util
                 = 71.1% × 75%
                 = 53.3%

  ── Effective Throughput ──
    Effective_GOPS = 614.4 × 0.533 = 327.5 GOPS
```

```
BƯỚC 4: Tính thời gian xử lý 1 frame (1 ảnh inference)
═══════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │  T_frame = Total_MACs / Effective_GOPS                          │
  │                                                                 │
  │  T_frame = 3,077,000,000 / 327,500,000,000                     │
  │          = 0.009396 giây                                        │
  │          = 9.40 ms                                              │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
```

```
BƯỚC 5: Tính FPS
═══════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │  ★ CÔNG THỨC CHÍNH:                                             │
  │                                                                 │
  │            MACs_per_cycle × Frequency × Utilization             │
  │  FPS = ──────────────────────────────────────────────           │
  │                       Total_MACs                                │
  │                                                                 │
  │            3,072 × 200,000,000 × 0.533                         │
  │      = ────────────────────────────────                        │
  │                  3,077,000,000                                  │
  │                                                                 │
  │            327,500,000,000                                      │
  │      = ──────────────────                                      │
  │           3,077,000,000                                         │
  │                                                                 │
  │      = 106.4 FPS  ✓  (> 100 FPS target)                       │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
```

### 1.2. Bảng Sensitivity — FPS theo các tham số

```
┌──────────┬──────────────┬────────────┬─────────┬─────────┬──────────┐
│ Clock    │ Active Subs  │ Util total │ Eff GOPS│ T_frame │ FPS      │
├──────────┼──────────────┼────────────┼─────────┼─────────┼──────────┤
│ 150 MHz  │ 8            │ 53%        │ 245.6   │ 12.53ms │ 79.8     │
│ 175 MHz  │ 8            │ 53%        │ 286.5   │ 10.74ms │ 93.1     │
│ 200 MHz  │ 6            │ 53%        │ 245.6   │ 12.53ms │ 79.8     │
│ 200 MHz  │ 8            │ 45%        │ 276.5   │ 11.13ms │ 89.8     │
│ 200 MHz  │ 8            │ 50%        │ 307.2   │ 10.02ms │ 99.8     │
│★200 MHz  │ 8            │ 53%        │ 327.5   │  9.40ms │ 106.4 ✓ │
│ 200 MHz  │ 8            │ 60%        │ 368.6   │  8.35ms │ 119.8    │
│ 220 MHz  │ 8            │ 53%        │ 360.2   │  8.54ms │ 117.1    │
│ 250 MHz  │ 8            │ 53%        │ 409.4   │  7.52ms │ 133.1    │
│ 200 MHz  │ 12           │ 53%        │ 491.2   │  6.26ms │ 159.7    │
└──────────┴──────────────┴────────────┴─────────┴─────────┴──────────┘

★ = Config mặc định V3-VC707

Kết luận:
  - 200 MHz + 8 active subs: cần utilization ≥ 50% → đạt ≥ 100 FPS
  - V3 architecture đạt ~53% util → 106 FPS (margin +6%)
  - Nếu optimize compiler tốt hơn (util 60%) → 120 FPS
```

### 1.3. End-to-End Pipeline (CPU + HW)

```
CPU + Accelerator pipeline (3 tầng overlap):

  Frame N:    │ CPU Preprocess │
              │ Letterbox+Quant│─── 2.5 ms ───
              │                │
  Frame N-1:  │                │ HW Accelerator │
              │                │ L0-L22 INT8    │─── 9.4 ms ───
              │                │                │
  Frame N-2:  │                │                │ CPU Postprocess│
              │                │                │ Detect+NMS     │── 3.5ms
              │                │                │                │

  Throughput = 1 / max(2.5, 9.4, 3.5) = 1 / 9.4 = 106 FPS ✓
  Latency = 2.5 + 9.4 + 3.5 = 15.4 ms (3 frame delay)
```

---

## 2. FPGA RESOURCE ESTIMATION CHI TIẾT

### 2.1. Breakdown Per Subcluster (×12 instances)

```
┌───────────────────────────┬──────┬───────┬────────┬────────┬─────────────────────────┐
│ Module                    │ DSP  │ BRAM  │ LUT    │ FF     │ Chi tiết                │
├───────────────────────────┼──────┼───────┼────────┼────────┼─────────────────────────┤
│ ═══ COMPUTE ═══           │      │       │        │        │                         │
│ PE Cluster (12 PEs)       │      │       │        │        │                         │
│  ├─ dsp_pair_int8 ×192    │ 192  │ 0     │ 4,800  │ 6,400  │ 16 DSP/PE, 25 LUT/pair │
│  ├─ column_reduce         │ 0    │ 0     │ 512    │ 256    │ 3→1 sum, 4 cols, 32 lns │
│  └─ comparator_tree       │ 0    │ 0     │ 800    │ 400    │ 25→1 max, 5 stages      │
│                           │      │       │        │        │                         │
│ ═══ POST-PROCESSING ═══   │      │       │        │        │                         │
│ PPU ×4 (parallel)         │ 0    │ 0     │ 3,200  │ 2,000  │ INT64 mult + ReLU ×4   │
│                           │      │       │        │        │                         │
│ ═══ MEMORY ═══            │      │       │        │        │                         │
│ GLB Input Bank ×3         │ 0    │ 6     │ 100    │ 50     │ 32 subbanks/bank        │
│ GLB Weight Bank ×3        │ 0    │ 12    │ 600    │ 300    │ ★ 4-read-port (V3)     │
│ GLB Output Bank ×4        │ 0    │ 16    │ 200    │ 100    │ PSUM(32b) + ACT(8b)    │
│ Metadata RAM              │ 0    │ 1     │ 50     │ 30     │ 16 slots ring buffer    │
│                           │      │       │        │        │                         │
│ ═══ ADDRESS GEN ═══       │      │       │        │        │                         │
│ addr_gen_input            │ 0    │ 0     │ 300    │ 150    │ h mod 3 + padding detect│
│ addr_gen_weight           │ 0    │ 0     │ 400    │ 200    │ ★ 4-col addr (V3)      │
│ addr_gen_output           │ 0    │ 0     │ 200    │ 100    │ bank = pe_col           │
│                           │      │       │        │        │                         │
│ ═══ DATA MOVEMENT ═══     │      │       │        │        │                         │
│ router_cluster_v2         │ 0    │ 0     │ 1,400  │ 700    │ ★ per-col weight (V3)  │
│ window_gen                │ 0    │ 0     │ 500    │ 1,800  │ 7×32 shift registers    │
│ swizzle_engine            │ 0    │ 0     │ 400    │ 300    │ upsample/concat/move    │
│                           │      │       │        │        │                         │
│ ═══ CONTROL ═══           │      │       │        │        │                         │
│ tile_fsm                  │ 0    │ 0     │ 300    │ 150    │ 10-state FSM            │
│ shadow_reg_file           │ 0    │ 0     │ 100    │ 500    │ Descriptor latch        │
│ compute_sequencer         │ 0    │ 0     │ 400    │ 300    │ (h,w,c,kw) inner loop   │
├───────────────────────────┼──────┼───────┼────────┼────────┼─────────────────────────┤
│ TOTAL per Subcluster      │ 192  │ 35    │ 14,262 │ 13,736 │                         │
│  ★ V3 additions          │ (+0) │ (+6)  │(+1,200)│ (+600) │ per-col routing         │
│ TOTAL per Sub (V3)        │ 192  │ 41    │ 15,462 │ 14,336 │                         │
└───────────────────────────┴──────┴───────┴────────┴────────┴─────────────────────────┘
```

### 2.2. Breakdown Per SuperCluster (×4 instances)

```
┌───────────────────────────┬──────┬───────┬────────┬────────┐
│ Module                    │ DSP  │ BRAM  │ LUT    │ FF     │
├───────────────────────────┼──────┼───────┼────────┼────────┤
│ 3× Subcluster (V3)       │ 576  │ 123   │ 46,386 │ 43,008 │
│ local_arbiter             │ 0    │ 0     │ 500    │ 300    │
│ tensor_dma                │ 0    │ 4     │ 1,500  │ 1,000  │
│ tile_ingress_fifo         │ 0    │ 2     │ 200    │ 100    │
├───────────────────────────┼──────┼───────┼────────┼────────┤
│ TOTAL per SuperCluster    │ 576  │ 129   │ 48,586 │ 44,408 │
└───────────────────────────┴──────┴───────┴────────┴────────┘
```

### 2.3. System Level Total

```
┌───────────────────────────┬──────┬───────┬─────────┬─────────┐
│ Module                    │ DSP  │ BRAM  │ LUT     │ FF      │
├───────────────────────────┼──────┼───────┼─────────┼─────────┤
│ 4× SuperCluster           │2,304 │ 516   │ 194,344 │ 177,632 │
│                           │      │       │         │         │
│ Controller System:        │      │       │         │         │
│  ├─ csr_register_bank     │ 0    │ 1     │ 500     │ 400     │
│  ├─ desc_fetch_engine     │ 0    │ 2     │ 800     │ 600     │
│  ├─ barrier_manager       │ 0    │ 0     │ 200     │ 100     │
│  └─ global_scheduler      │ 0    │ 1     │ 500     │ 400     │
│                           │      │       │         │         │
│ AXI Infrastructure:       │      │       │         │         │
│  ├─ AXI-Lite Slave        │ 0    │ 0     │ 1,000   │ 500     │
│  ├─ AXI4 Master Mux       │ 0    │ 4     │ 2,000   │ 1,500   │
│  └─ AXI Interconnect      │ 0    │ 4     │ 2,000   │ 1,000   │
│                           │      │       │         │         │
│ Clock/Reset/Misc          │ 0    │ 0     │ 1,500   │ 800     │
├───────────────────────────┼──────┼───────┼─────────┼─────────┤
│ ★ GRAND TOTAL             │2,304 │ 528   │ 202,844 │ 182,932 │
└───────────────────────────┴──────┴───────┴─────────┴─────────┘
```

### 2.4. So sánh với VC707 (XC7VX485T) Resources

```
┌───────────────────┬──────────┬──────────┬────────┬──────────────────────────┐
│ Resource          │ Available│ Used     │ %      │ Đánh giá                 │
├───────────────────┼──────────┼──────────┼────────┼──────────────────────────┤
│ DSP48E1           │ 2,800    │ 2,304    │ 82.3%  │ ⚠️ Sát nhưng OK         │
│ BRAM36K           │ 1,030    │ 528      │ 51.3%  │ ✅ Dư dả                │
│ LUT6              │ 303,600  │ 202,844  │ 66.8%  │ ✅ Vùng thoải mái       │
│ FF (Flip-Flop)    │ 607,200  │ 182,932  │ 30.1%  │ ✅ Rất dư               │
└───────────────────┴──────────┴──────────┴────────┴──────────────────────────┘

                    DSP:  ████████████████░░░░ 82.3%
                    BRAM: ██████████░░░░░░░░░░ 51.3%
                    LUT:  █████████████░░░░░░░ 66.8%
                    FF:   ██████░░░░░░░░░░░░░░ 30.1%

Kết luận: TẤT CẢ resources VỪA VC707.
  - DSP 82%: sát nhưng Virtex-7 DSP routing không phức tạp → OK
  - LUT 67%: vùng thoải mái (timing closure dễ dưới 80%)
  - BRAM 51%: dư → có thể tăng GLB depth nếu cần
  - FF 30%: rất dư → pipeline thêm nếu cần timing
```

### 2.5. Power Estimation (ước tính)

```
Virtex-7 XC7VX485T @ 200 MHz:
  DSP dynamic: 2,304 DSPs × ~3 mW/DSP = ~6.9 W
  BRAM dynamic: 528 BRAMs × ~1.5 mW/BRAM = ~0.8 W
  Logic dynamic: ~200K LUT × ~0.01 mW/LUT = ~2.0 W
  Clock tree: ~1.5 W
  Static: ~3.0 W
  ──────────────────────
  Total estimated: ~14.2 W

  VC707 board power budget: 50W+ → dư dả
```

---

## 3. KHẢ NĂNG INFERENCE TOÀN BỘ YOLOV10n

### 3.1. Phân tích coverage: Mỗi layer có được hỗ trợ?

```
┌───────┬──────────┬─────────────────────────────────────────┬──────────┐
│ Layer │ Block    │ Primitives cần                          │ HW mode  │
├───────┼──────────┼─────────────────────────────────────────┼──────────┤
│ L0    │ Conv     │ P0(Conv3x3, s=2) + PPU(ReLU)           │ PE_RS3   │
│ L1    │ Conv     │ P0(Conv3x3, s=2) + PPU(ReLU)           │ PE_RS3   │
│ L2    │ QC2f     │ P1(1x1) + P0(3x3)×2 + P5(CAT) + P1   │ 5 desc   │
│ L3    │ Conv     │ P0(Conv3x3, s=2) + PPU(ReLU)           │ PE_RS3   │
│ L4    │ QC2f     │ P1 + P0×2 + P5 + P1                    │ 5 desc   │
│ L5    │ SCDown   │ P1(1x1) + P2(DW3x3, s=2)              │ 2 desc   │
│ L6    │ QC2f     │ P1 + P0×2 + P5 + P1                    │ 5 desc   │
│ L7    │ SCDown   │ P1 + P2                                 │ 2 desc   │
│ L8    │ QC2f     │ P1 + P0×2 + P5 + P1                    │ 5 desc   │
│ L9    │ SPPF     │ P1 + P3(MP5)×3 + P5(CAT4) + P1        │ 6 desc   │
│ L10   │ QPSA     │ P1×N + GEMM(PE_OS1) + Softmax(LUT)    │ ~14 desc │
│ L11   │ Upsample │ P6(address remap 2×)                   │ PE_PASS  │
│ L12   │ QConcat  │ P5(CONCAT) + barrier sync              │ PE_PASS  │
│ L13   │ QC2f     │ P1 + P0×2 + P5 + P1                    │ 5 desc   │
│ L14   │ Upsample │ P6(address remap 2×)                   │ PE_PASS  │
│ L15   │ QConcat  │ P5(CONCAT) + barrier sync              │ PE_PASS  │
│ L16   │ QC2f     │ P1 + P0×2 + P5 + P1                    │ 5 desc   │
│ L17   │ Conv     │ P0(Conv3x3, s=2) + PPU(ReLU)           │ PE_RS3   │
│ L18   │ QConcat  │ P5 + barrier                            │ PE_PASS  │
│ L19   │ QC2f     │ P1 + P0×2 + P5 + P1                    │ 5 desc   │
│ L20   │ SCDown   │ P1 + P2                                 │ 2 desc   │
│ L21   │ QConcat  │ P5 + barrier                            │ PE_PASS  │
│ L22   │ QC2fCIB  │ P1+P2+P8(DW7)+P7(ADD)+P5+P1           │ ~9 desc  │
├───────┼──────────┼─────────────────────────────────────────┼──────────┤
│       │          │ Coverage:                               │ 23/23 ✅ │
└───────┴──────────┴─────────────────────────────────────────┴──────────┘

Tổng descriptors cho 1 inference: ~60 descriptors
Tất cả chạy trên CÙNG 1 phần cứng (subcluster), chỉ đổi descriptor config.
```

### 3.2. Primitive coverage — 14 primitives đều được hỗ trợ

```
┌──────┬──────────────────────┬──────────────────────┬──────────────────────┐
│ ID   │ Primitive            │ HW Implementation    │ Verified accuracy    │
├──────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ P0   │ RS_DENSE_3x3         │ pe_mode = PE_RS3     │ 99.98% (L0-L3-L17)  │
│ P1   │ OS_1x1               │ pe_mode = PE_OS1     │ 99%+ (QC2f cv1/cv2) │
│ P2   │ DW_3x3               │ pe_mode = PE_DW3     │ 99.94% (SCDown)     │
│ P3   │ MAXPOOL_5x5          │ pe_mode = PE_MP5     │ 99.93% (SPPF)       │
│ P4   │ MOVE                 │ DMA copy             │ 100%                │
│ P5   │ CONCAT               │ pe_mode = PE_PASS    │ 100% (QConcat)      │
│      │                      │ + router bypass      │                     │
│ P6   │ UPSAMPLE_NEAREST     │ pe_mode = PE_PASS    │ 100% (L11/L14)      │
│      │                      │ + swizzle_engine     │                     │
│ P7   │ EWISE_ADD            │ Domain align + add   │ 100% (residual)     │
│ P8   │ DW_7x7_MULTIPASS     │ pe_mode = PE_DW7     │ 99.96% (L22)        │
│      │                      │ 3 passes             │                     │
│ P9   │ GEMM_ATTN_BASIC      │ pe_mode = PE_OS1     │ 83.52% (QPSA L10)  │
│      │                      │ (matrix mul as conv) │                     │
│ P10  │ INT8_MATMUL          │ PE_OS1 reuse         │ (included in P9)    │
│ P11  │ SOFTMAX_APPROX       │ 256-entry LUT        │ (included in P9)    │
│ P12  │ REQUANT (PPU)        │ PPU pipeline          │ integrated          │
│ P13  │ SiLU_LUT             │ 256-entry LUT        │ NOT USED (ReLU)     │
│ P14  │ ReLU                 │ max(0, x) comparator │ integrated in PPU   │
├──────┼──────────────────────┼──────────────────────┼──────────────────────┤
│      │ Coverage:            │                      │ 14/14 ✅            │
└──────┴──────────────────────┴──────────────────────┴──────────────────────┘
```

### 3.3. Tại sao CHẮC CHẮN inference đúng — 3 tầng đảm bảo

```
TẦNG 1 — Toán học đã verified bằng Golden Python:
═══════════════════════════════════════════════════
  ✅ 14 primitives PASS (100% verified trên Python)
  ✅ 23 layers PASS (100 samples mỗi layer, 640×640)
  ✅ mAP50 = 0.9302 trên 7,902 images (full dataset)
  ✅ 6 critical fixes đã apply (rounding, signed domain, INT64, ...)
  → TOÁN ĐÚNG — đã chứng minh

TẦNG 2 — RTL implement CÙNG công thức:
═══════════════════════════════════════════════════
  ✅ dsp_pair_int8: signed INT8 × INT8 → INT32 (unsigned-offset DSP48E1 packing)
  ✅ PPU: half-up rounding: (acc × M_int + (1<<(sh-1))) >> sh
  ✅ PPU: INT64 multiply (biased × M_int = 64-bit product)
  ✅ PPU: ReLU activation (max(0, x), NOT SiLU)
  ✅ Padding: fill = zero_point_x (NOT 0)
  ✅ CONCAT: domain alignment (requant to common scale)
  ✅ EWISE_ADD: dequant → float add → requant (golden path)
  → RTL REPLICATE cùng arithmetic — nếu atoms đúng → layers đúng → inference đúng

TẦNG 3 — Descriptor-driven execution đảm bảo đúng sequence:
═══════════════════════════════════════════════════
  ✅ tile_fsm: CÙNG thứ tự phases với Golden Python
     (PREFILL_WT → PREFILL_IN → RUN_COMPUTE → PPU → SWIZZLE → DONE)
  ✅ Mỗi descriptor = 1 primitive call = cùng computation
  ✅ QC2f = 5 descriptors tuần tự → CÙNG sequence: OS1→RS3→RS3→CAT→OS1
  ✅ Barrier sync cho 4 skip connections (L12, L15, L18, L21)
  ✅ PSUM namespace cho DW_7x7 multipass (pass1→pass2→pass3→PPU)
  → EXECUTION ORDER đúng

  ★ Kết luận: NẾU 3 tầng đều PASS → inference CHẮC CHẮN đúng
```

### 3.4. Phạm vi accelerator vs CPU

```
┌────────────────────────────────────────────────────────────────────┐
│                        YOLOv10n Model                              │
│                                                                    │
│  ┌────────────────────────────────────────────┐                    │
│  │     ACCELERATOR (phần cứng, INT8)          │  9.4 ms            │
│  │                                            │  106 FPS           │
│  │  Backbone (L0-L10):                        │                    │
│  │    Conv → QC2f → Conv → QC2f → SCDown →    │                    │
│  │    QC2f → SCDown → QC2f → SPPF → QPSA     │                    │
│  │                                            │                    │
│  │  Neck FPN (L11-L16):                       │                    │
│  │    Upsample → QConcat → QC2f → Upsample    │                    │
│  │    → QConcat → QC2f                        │                    │
│  │                                            │                    │
│  │  Neck PAN (L17-L22):                       │                    │
│  │    Conv → QConcat → QC2f → SCDown →         │                    │
│  │    QConcat → QC2fCIB                       │                    │
│  │                                            │                    │
│  │  Output: P3[64,80,80] P4[128,40,40]        │                    │
│  │          P5[256,20,20] (INT8 + scale/zp)    │                    │
│  └───────────────────┬────────────────────────┘                    │
│                      │ DMA to DDR3                                 │
│                      ▼                                             │
│  ┌────────────────────────────────────────────┐                    │
│  │     CPU (phần mềm, float32)               │  3.5 ms            │
│  │                                            │                    │
│  │  Detection Head (L23: Qv10Detect):         │                    │
│  │    Dequantize P3/P4/P5: INT8 → float32    │                    │
│  │    3 nhánh Conv (bbox + class)             │                    │
│  │    Decode bounding boxes                   │                    │
│  │    NMS (Non-Maximum Suppression)           │                    │
│  │                                            │                    │
│  │  Output: [(x1,y1,x2,y2,conf,class), ...]  │                    │
│  └────────────────────────────────────────────┘                    │
│                                                                    │
│  Tại sao Head chạy CPU?                                            │
│    - L23 dùng float32 (không INT8)                                 │
│    - NMS là sequential (không parallelize hiệu quả)               │
│    - Compute nhỏ (~10% total model)                                │
│    - Dequant cần float → không phù hợp INT8 PE                    │
│                                                                    │
│  End-to-end throughput:                                             │
│    Pipeline overlap → bottleneck = max(CPU_pre, HW, CPU_post)     │
│    = max(2.5ms, 9.4ms, 3.5ms) = 9.4ms → 106 FPS                 │
└────────────────────────────────────────────────────────────────────┘
```

### 3.5. Accuracy expected

```
┌──────────────────────┬──────────────────────┬──────────────────────┐
│ Metric               │ Golden Python (SW)   │ RTL (HW) expected    │
├──────────────────────┼──────────────────────┼──────────────────────┤
│ Conv layers match    │ 99.98%               │ ≥99.9%               │
│ QC2f layers match    │ 97.31% (backbone)    │ ≥96%                 │
│                      │ 98.83% (neck)        │ ≥97%                 │
│ SCDown match         │ 99.94%               │ ≥99.9%               │
│ SPPF match           │ 99.93%               │ ≥99.9%               │
│ QPSA match           │ 83.63%               │ ≥83%                 │
│ Upsample/QConcat     │ 100.00%              │ 100%                 │
│ QC2fCIB match        │ 99.96%               │ ≥99.9%               │
│                      │                      │                      │
│ mAP50 (dataset)      │ 0.9302               │ ≥0.92 (< 1% loss)   │
│ mAP50-95             │ 0.7217               │ ≥0.71                │
└──────────────────────┴──────────────────────┴──────────────────────┘

RTL accuracy ≈ Golden Python accuracy vì:
  - Cùng công thức toán
  - Cùng rounding mode (half-up)
  - Cùng data type (signed INT8)
  - Cùng execution order (descriptor sequence)
  → mAP degradation dự kiến < 1%
```

---

## 4. TÓM TẮT 3 CÂU TRẢ LỜI

```
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║  CÂU 1: CÔNG THỨC FPS                                               ║
║                                                                      ║
║           MACs/cycle × Frequency × Utilization                      ║
║  FPS = ─────────────────────────────────────────                    ║
║                    Total_MACs                                        ║
║                                                                      ║
║       = (3,072 × 200M × 53.3%) / 3,077M = 106.4 FPS ✓             ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  CÂU 2: FPGA RESOURCES trên VC707 (XC7VX485T)                      ║
║                                                                      ║
║    DSP48E1:  2,304 / 2,800  = 82.3%  ✅                            ║
║    BRAM36K:    528 / 1,030  = 51.3%  ✅                            ║
║    LUT6:   202,844 / 303,600 = 66.8%  ✅                           ║
║    FF:     182,932 / 607,200 = 30.1%  ✅                           ║
║    → TẤT CẢ VỪA, power ~14W (board budget 50W+)                    ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  CÂU 3: CÓ INFERENCE ĐƯỢC TOÀN BỘ YOLOV10n KHÔNG?                  ║
║                                                                      ║
║    ✅ Backbone+Neck (L0-L22): 23/23 layers = 100% coverage         ║
║    ✅ 14/14 primitives mapped → CÙNG phần cứng                      ║
║    ✅ ~60 descriptors per inference                                  ║
║    ✅ 3 tầng đảm bảo: Toán đúng + RTL đúng + Sequence đúng         ║
║    ✅ Expected mAP50 ≥ 0.92 (< 1% loss)                            ║
║                                                                      ║
║    ⚠️ Detection Head (L23): chạy CPU (float32, NMS)                ║
║       → 3.5ms overhead, pipeline overlap → không ảnh hưởng FPS     ║
║                                                                      ║
║  ★ KẾT LUẬN: CÓ — inference đúng, >100 FPS, trên VC707            ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

---
---

# ════════════════════════════════════════════════════════════════
# PHẦN II: NGHIÊN CỨU TỐI ƯU V4-VC707 — HIGHER FREQUENCY, HIGHER FPS
# ════════════════════════════════════════════════════════════════

> **Mục tiêu**: Đạt FPS cao nhất có thể, resource ≤ 65-70% VC707
> **Phương pháp**: Tối ưu LANES + Clock + Subclusters + Pipelining
> **Kết quả**: ~183 FPS @ 250 MHz, ~65% resource utilization

---

## 5. PHÂN TÍCH VẤN ĐỀ V3 — TẠI SAO CẦN TỐI ƯU THÊM

### 5.1. Điểm yếu của V3-VC707

```
V3-VC707 hiện tại:
  FPS = 106 FPS @ 200 MHz (sensitivity: 124 FPS @ 62% util)
  DSP usage = 82.3% → VƯỢT target 60-70%
  Spatial utilization = 71.1% → CÒN DƯ LÃNG PHÍ LỚN

  ┌────────────────────────────────────────────────────────────────────────┐
  │ VẤN ĐỀ CHÍNH: LANES=32 KHÔNG KHỚP VỚI FEATURE MAP WIDTHS            │
  ├────────────────────────────────────────────────────────────────────────┤
  │                                                                        │
  │  YOLOv10n feature map widths: 320, 160, 80, 40, 20                   │
  │  LANES=32 → padded widths:   320, 160, 96, 64, 32                   │
  │                                                                        │
  │  W=80:  80/96  = 83.3% utilization  → 16.7% waste                    │
  │  W=40:  40/64  = 62.5% utilization  → 37.5% waste!                   │
  │  W=20:  20/32  = 62.5% utilization  → 37.5% waste!                   │
  │                                                                        │
  │  Tier 20×20 chiếm 1,330M MACs (43% total!) → lãng phí 37.5%         │
  │  Tier 40×40 chiếm 740M MACs (24% total!)   → lãng phí 37.5%         │
  │  → 67% compute nằm ở tiers bị lãng phí nặng                          │
  │                                                                        │
  │  Weighted spatial util = 71.1% → MẤT 28.9% throughput vì padding!    │
  └────────────────────────────────────────────────────────────────────────┘

  DSP = 82.3% → quá sát giới hạn:
    - Place & Route khó khăn khi DSP > 80%
    - Timing closure rủi ro cao
    - Không còn headroom cho debug logic

  Clock = 200 MHz:
    - Conservative cho Virtex-7 (DSP48E1 rated >500 MHz)
    - LANES=32 → fan-out lớn trên routing network
    - Critical path: router_cluster → pe_unit → column_reduce
    - Có thể push cao hơn nếu giảm fan-out
```

### 5.2. Phát hiện quan trọng: LANES=20 là "Magic Number"

```
╔══════════════════════════════════════════════════════════════════════╗
║  ★ PHÁT HIỆN THEN CHỐT:                                             ║
║                                                                      ║
║  Tất cả feature map widths của YOLOv10n đều CHIA HẾT cho 20:       ║
║                                                                      ║
║    320 ÷ 20 = 16  (exact)                                           ║
║    160 ÷ 20 = 8   (exact)                                           ║
║    80  ÷ 20 = 4   (exact)                                           ║
║    40  ÷ 20 = 2   (exact)                                           ║
║    20  ÷ 20 = 1   (exact)                                           ║
║                                                                      ║
║  → LANES=20: Spatial utilization = 100% cho MỌI tier!               ║
║  → KHÔNG còn padding waste!                                          ║
║  → Weighted spatial util: 71.1% → 100% (tăng 40.7%!)               ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 6. KIẾN TRÚC V4-VC707 — "NARROW-LANE HIGH-FREQUENCY"

### 6.1. Nguyên lý thiết kế

```
4 TRỤC TỐI ƯU ĐỒNG THỜI:
═══════════════════════════════════════════════════════════════

  ① LANES = 20 (từ 32)
     → 100% spatial utilization (từ 71.1%)
     → Giảm fan-out → cho phép clock cao hơn
     → Giảm DSP/subcluster: 120 (từ 192) → room cho nhiều subs hơn

  ② SUBCLUSTERS = 16 (từ 12)
     → 4 SC × 4 sub/SC (từ 4 SC × 3 sub/SC)
     → Triple-RUNNING: 3 active + 1 fill/drain per SC
     → 12 active subclusters (từ 8) = 50% MORE compute units!

  ③ CLOCK = 250 MHz (từ 200 MHz)
     → LANES nhỏ hơn → critical path ngắn hơn
     → Deeper pipeline (5-6 stage vs 4 stage)
     → DSP48E1 Virtex-7 rated >500 MHz → 250 MHz rất safe

  ④ DOUBLE-BUFFERED GLB
     → Ping-pong buffer: compute page A ↔ fill page B
     → Gần như loại bỏ fill/drain stall
     → Temporal util: 75% → 82%
```

### 6.2. Thông số chốt V4-VC707

```
╔══════════════════════════════════════════════════════════════════════╗
║              KIẾN TRÚC V4-VC707 — FINAL PARAMETERS                   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ┌───────────────────────────┬─────────────────────────────────────┐ ║
║  │ Parameter                 │ V4-VC707                            │ ║
║  ├───────────────────────────┼─────────────────────────────────────┤ ║
║  │ LANES                     │ 20 (★ magic number for YOLOv10n)   │ ║
║  │ PE_ROWS                   │ 3 (kh parallelism, giữ nguyên)    │ ║
║  │ PE_COLS                   │ 4 (cout parallelism, per-col wgt)  │ ║
║  │ DSP_PAIRS_PER_PE          │ 10 (= LANES/2)                    │ ║
║  │ MACs per PE               │ 20                                 │ ║
║  │ MACs per subcluster       │ 3 × 4 × 20 = 240 (ALL unique)    │ ║
║  │ DSPs per subcluster       │ 3 × 4 × 10 = 120                 │ ║
║  │                           │                                     │ ║
║  │ SUPER_CLUSTERS            │ 4                                   │ ║
║  │ SUBS_PER_SC               │ 4 (★ tăng từ 3)                   │ ║
║  │ ACTIVE_PER_SC             │ 3 (★ Triple-RUNNING)              │ ║
║  │ Total subclusters         │ 16                                  │ ║
║  │ Total active subclusters  │ 12                                  │ ║
║  │                           │                                     │ ║
║  │ Clock target              │ 250 MHz (★ tăng từ 200)           │ ║
║  │ Pipeline depth            │ 5-6 stages (★ sâu hơn V3)        │ ║
║  │ GLB buffering             │ Double-buffer (ping-pong)          │ ║
║  │ Activation                │ ReLU (max(0,x))                    │ ║
║  │ QPSA                      │ PE_OS1 mode + softmax LUT          │ ║
║  └───────────────────────────┴─────────────────────────────────────┘ ║
╚══════════════════════════════════════════════════════════════════════╝
```

### 6.3. Triple-RUNNING — Cơ chế hoạt động

```
V3 Dual-RUNNING (3 subs/SC):
═══════════════════════════════
  Cycle:  Sub-0     Sub-1     Sub-2
  T0:     COMPUTE   COMPUTE   FILL
  T1:     COMPUTE   DRAIN     COMPUTE
  T2:     FILL      COMPUTE   COMPUTE
  → 2 active per SC × 4 SC = 8 active total

V4 Triple-RUNNING (4 subs/SC):
═══════════════════════════════
  Cycle:  Sub-0     Sub-1     Sub-2     Sub-3
  T0:     COMPUTE   COMPUTE   COMPUTE   FILL+DRAIN
  T1:     FILL+DRAIN COMPUTE  COMPUTE   COMPUTE
  T2:     COMPUTE   FILL+DRAIN COMPUTE  COMPUTE
  T3:     COMPUTE   COMPUTE   FILL+DRAIN COMPUTE
  → 3 active per SC × 4 SC = 12 active total

  Tại sao FILL+DRAIN gộp được?
    1 sub không compute → DMA có toàn bộ bandwidth
    DMA fill (new tile inputs/weights) + DMA drain (prev tile outputs)
    Song song trên 2 AXI channels (read + write)
    LANES=20 → tiles nhỏ hơn → fill/drain nhanh hơn

  DMA bandwidth check:
    Per tile: ~15KB input + ~8KB weight + ~5KB output ≈ 28KB
    Tile duration: ~0.15ms (typical @ 250 MHz)
    Required BW: 28KB / 0.15ms = 187 MB/s per sub
    4 SC × 1 filling sub = 4 × 187 = 748 MB/s
    DDR3 bandwidth: 12,800 MB/s
    → Only 5.8% DDR BW utilized → DMA EASILY keeps up ✓
```

---

## 7. TÍNH TOÁN FPS V4 — CHI TIẾT TỪNG BƯỚC

### 7.1. Spatial Utilization — 100% (Perfect Alignment)

```
LANES = 20:

┌───────────┬──────┬────────────┬──────────────┬────────────────────┐
│ W_out     │ Wblk │ Padded W   │ Spatial util │ Layers             │
├───────────┼──────┼────────────┼──────────────┼────────────────────┤
│ 320       │ 16   │ 320        │ 100.0% ✓    │ L0                 │
│ 160       │ 8    │ 160        │ 100.0% ✓    │ L1, L2             │
│ 80        │ 4    │ 80         │ 100.0% ✓    │ L3,L4,L15,L16,L17  │
│ 40        │ 2    │ 40         │ 100.0% ✓    │ L5,L6,L12-13,L18-19│
│ 20        │ 1    │ 20         │ 100.0% ✓    │ L7-L10,L20-L22     │
└───────────┴──────┴────────────┴──────────────┴────────────────────┘

So sánh V3 (LANES=32):
┌───────────┬──────────────┬──────────────┬──────────────┐
│ W_out     │ V3 util      │ V4 util      │ Improvement  │
├───────────┼──────────────┼──────────────┼──────────────┤
│ 320       │ 100.0%       │ 100.0%       │ +0%          │
│ 160       │ 100.0%       │ 100.0%       │ +0%          │
│ 80        │ 83.3%        │ 100.0%       │ +20.0%       │
│ 40        │ 62.5%        │ 100.0%       │ +60.0%!      │
│ 20        │ 62.5%        │ 100.0%       │ +60.0%!      │
├───────────┼──────────────┼──────────────┼──────────────┤
│ Weighted  │ 71.1%        │ 100.0%       │ +40.6%!      │
└───────────┴──────────────┴──────────────┴──────────────┘

★ Cải thiện 40.6% spatial utilization = lợi ích LỚN NHẤT của V4
```

### 7.2. Temporal Utilization — Cải thiện nhờ Triple-RUN + Double-Buffer

```
┌─────────────────────────────────┬─────────┬─────────┬──────────────────────┐
│ Nguồn overhead                  │ V3 Loss │ V4 Loss │ Lý do cải thiện      │
├─────────────────────────────────┼─────────┼─────────┼──────────────────────┤
│ Fill/drain pipeline gap         │ 10%     │ 4%      │ Triple-RUN + Double  │
│                                 │         │         │ Buffer GLB overlap   │
│ Descriptor fetch                │ 3%      │ 2%      │ Deeper prefetch FIFO │
│ Barrier stalls (4 points)       │ 5%      │ 4%      │ More subs → faster   │
│                                 │         │         │ skip tensor produce  │
│ DW_7x7 pass-3 inefficiency     │ 2%      │ 2%      │ Không đổi            │
│ Tile boundary waste             │ 3%      │ 0%      │ ★ LANES=20: tiles   │
│                                 │         │         │ LUÔN đầy, KHÔNG pad! │
│ QPSA complex path              │ 2%      │ 2%      │ Không đổi            │
│ Clock domain crossing           │ 0%      │ 1%      │ 250MHz cần CDC care  │
├─────────────────────────────────┼─────────┼─────────┼──────────────────────┤
│ Total overhead                  │ ~25%    │ ~15%    │                      │
│ Temporal utilization            │ ~75%    │ ~85%    │                      │
└─────────────────────────────────┴─────────┴─────────┴──────────────────────┘

Conservative estimate: Temporal util = 82% (margin for unknowns)
```

### 7.3. Per-Tier Compute Time

```
Active MACs = 12 subclusters × 240 MACs/sub = 2,880 MACs/cycle
Clock = 250 MHz
Spatial utilization = 100% (all tiers!)

┌──────────────┬────────┬────────┬────────────────────────────────────────┐
│ Tier         │ MACs   │ S_util │ T_compute                              │
├──────────────┼────────┼────────┼────────────────────────────────────────┤
│ 320 (L0)     │  44 M  │ 100%   │ 44M / (2,880×1.0×250M) = 0.061 ms    │
│ 160 (L1-L2)  │ 288 M  │ 100%   │ 288M / (2,880×1.0×250M) = 0.400 ms   │
│ 80 (L3-L4,   │ 675 M  │ 100%   │ 675M / (2,880×1.0×250M) = 0.938 ms   │
│    L15-L17)  │        │        │                                        │
│ 40 (L5-L6,   │ 740 M  │ 100%   │ 740M / (2,880×1.0×250M) = 1.028 ms   │
│    L12-L19)  │        │        │                                        │
│ 20 (L7-L10,  │1,330 M │ 100%   │ 1,330M / (2,880×1.0×250M) = 1.847 ms │
│    L20-L22)  │        │        │                                        │
├──────────────┼────────┼────────┼────────────────────────────────────────┤
│ TỔNG         │3,077 M │ 100%   │ 4.274 ms (pure compute)               │
└──────────────┴────────┴────────┴────────────────────────────────────────┘

So sánh V3:
  V3 pure compute = 7.25 ms
  V4 pure compute = 4.27 ms  → giảm 41%!
```

### 7.4. FPS Calculation

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ★ CÔNG THỨC:                                                   │
│                                                                 │
│  T_hw = Pure_compute / Temporal_util                            │
│       = 4.274 ms / 0.82                                        │
│       = 5.21 ms                                                 │
│                                                                 │
│  FPS = 1 / T_hw                                                 │
│      = 1 / 0.00521                                              │
│      = 192 FPS                                                  │
│                                                                 │
│  ★ HOẶC DÙNG CÔNG THỨC TRỰC TIẾP:                               │
│                                                                 │
│         MACs_per_cycle × Frequency × Overall_util               │
│  FPS = ──────────────────────────────────────────               │
│                      Total_MACs                                 │
│                                                                 │
│         2,880 × 250,000,000 × 0.82                             │
│      = ─────────────────────────────                            │
│                3,077,000,000                                    │
│                                                                 │
│         590,400,000,000                                         │
│      = ────────────────                                         │
│          3,077,000,000                                          │
│                                                                 │
│      = 191.9 FPS  ≈ 192 FPS                                    │
│                                                                 │
│  Conservative (temporal util = 78%):                            │
│      = 2,880 × 250M × 0.78 / 3,077M = 183 FPS                 │
│                                                                 │
│  ★★★ TARGET: 183-192 FPS (conservative to moderate) ★★★         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.5. Sensitivity Table — V4 Performance Space

```
┌──────────┬──────────────┬────────────┬─────────┬─────────┬──────────┐
│ Clock    │ Active Subs  │ Util total │ Eff GOPS│ T_frame │ FPS      │
├──────────┼──────────────┼────────────┼─────────┼─────────┼──────────┤
│ 200 MHz  │ 12           │ 82%        │ 472.3   │ 6.51ms  │ 153.6    │
│ 220 MHz  │ 12           │ 82%        │ 519.6   │ 5.92ms  │ 168.9    │
│★250 MHz  │ 12           │ 78%        │ 561.6   │ 5.48ms  │ 182.5 ✓ │
│★250 MHz  │ 12           │ 82%        │ 590.4   │ 5.21ms  │ 191.9 ✓ │
│ 250 MHz  │ 12           │ 85%        │ 612.0   │ 5.03ms  │ 199.0    │
│ 280 MHz  │ 12           │ 80%        │ 645.1   │ 4.77ms  │ 209.7    │
│ 300 MHz  │ 12           │ 78%        │ 673.9   │ 4.56ms  │ 219.1    │
│ 250 MHz  │ 8 (Dual-RUN)│ 82%        │ 393.6   │ 7.82ms  │ 127.9    │
│ 200 MHz  │ 8 (Dual-RUN)│ 82%        │ 314.9   │ 9.77ms  │ 102.4    │
└──────────┴──────────────┴────────────┴─────────┴─────────┴──────────┘

★ = Cấu hình mặc định V4-VC707 (250 MHz, 12 active, Triple-RUNNING)

Key insight:
  - Ngay cả Dual-RUNNING (8 active) @ 250 MHz cũng đạt 128 FPS > V3
  - Fallback 200 MHz + Dual-RUNNING vẫn đạt 102 FPS = tương đương V3
  - V4 có margin lớn: có thể giảm clock/subs mà vẫn > 100 FPS
```

### 7.6. End-to-End Pipeline

```
CPU + Accelerator pipeline (3 tầng overlap):

  Frame N:    │ CPU Preprocess │
              │ Letterbox+Quant│─── 2.5 ms ───
              │                │
  Frame N-1:  │                │ HW Accelerator │
              │                │ L0-L22 INT8    │─── 5.21 ms ───
              │                │                │
  Frame N-2:  │                │                │ CPU Postprocess│
              │                │                │ Detect+NMS     │── 3.5ms
              │                │                │                │

  Throughput = 1 / max(2.5, 5.21, 3.5) = 1 / 5.21 = 192 FPS ✓
  Latency = 2.5 + 5.21 + 3.5 = 11.21 ms (giảm từ 15.4ms V3)

  ★ CPU postprocess (3.5ms) KHÔNG còn là bottleneck ngang HW!
    HW và CPU post bây giờ gần bằng nhau → pipeline rất cân bằng.
```

---

## 8. RESOURCE ESTIMATION V4 — CHI TIẾT

### 8.1. Breakdown Per Subcluster (×16 instances)

```
LANES=20, PE_ROWS=3, PE_COLS=4, Pipeline=5-stage

┌───────────────────────────┬──────┬───────┬────────┬────────┬─────────────────────────┐
│ Module                    │ DSP  │ BRAM  │ LUT    │ FF     │ Chi tiết                │
├───────────────────────────┼──────┼───────┼────────┼────────┼─────────────────────────┤
│ ═══ COMPUTE ═══           │      │       │        │        │                         │
│ PE Cluster v4 (12 PEs)    │      │       │        │        │                         │
│  ├─ dsp_pair_int8 ×120    │ 120  │ 0     │ 3,000  │ 4,800  │ 10 pairs/PE, 25 LUT/pr │
│  │  (★5-stage pipeline)   │      │       │        │(+1200) │ +1 reg stage for Fmax   │
│  ├─ column_reduce v4      │ 0    │ 0     │ 320    │ 200    │ 3→1 sum, 4 cols, 20 lns│
│  └─ comparator_tree v4    │ 0    │ 0     │ 500    │ 320    │ 25→1 max, 5 stages     │
│                           │      │       │        │        │                         │
│ ═══ POST-PROCESSING ═══   │      │       │        │        │                         │
│ PPU ×4 (parallel)         │ 0    │ 0     │ 2,400  │ 2,000  │ INT64 mult + ReLU ×4   │
│ (★5-stage pipeline)       │      │       │        │(+400)  │ +1 reg stage            │
│                           │      │       │        │        │                         │
│ ═══ MEMORY ═══            │      │       │        │        │                         │
│ glb_input_bank_db ×3      │ 0    │ 8     │ 150    │ 80     │ ★Double-buffer (A/B)   │
│ (★double-buffered)        │      │(+2)   │(+50)   │(+30)   │ Ping-pong pages        │
│ glb_weight_bank ×3        │ 0    │ 10    │ 500    │ 280    │ 4-read-port (per-col)  │
│ glb_output_bank ×4        │ 0    │ 12    │ 180    │ 90     │ PSUM(32b) + ACT(8b)    │
│ metadata_ram              │ 0    │ 1     │ 50     │ 30     │ 16 slots ring buffer    │
│                           │      │       │        │        │                         │
│ ═══ ADDRESS GEN ═══       │      │       │        │        │                         │
│ addr_gen_input            │ 0    │ 0     │ 280    │ 180    │ h mod 3 + pad detect    │
│ (★+1 pipeline reg)        │      │       │        │(+30)   │ Higher Fmax             │
│ addr_gen_weight           │ 0    │ 0     │ 380    │ 220    │ 4-col addr generation   │
│ addr_gen_output           │ 0    │ 0     │ 180    │ 100    │ bank = pe_col           │
│                           │      │       │        │        │                         │
│ ═══ DATA MOVEMENT ═══     │      │       │        │        │                         │
│ router_cluster_v2         │ 0    │ 0     │ 1,000  │ 700    │ Per-col weight routing  │
│ (★+1 pipeline reg)        │      │       │        │(+100)  │ Narrower → less mux     │
│ window_gen                │ 0    │ 0     │ 380    │ 1,400  │ 7×20 shift registers    │
│ swizzle_engine            │ 0    │ 0     │ 280    │ 250    │ upsample/concat/move    │
│                           │      │       │        │        │                         │
│ ═══ CONTROL ═══           │      │       │        │        │                         │
│ tile_fsm                  │ 0    │ 0     │ 300    │ 150    │ 10-state FSM            │
│ shadow_reg_file           │ 0    │ 0     │ 100    │ 500    │ Descriptor latch        │
│ compute_sequencer         │ 0    │ 0     │ 380    │ 300    │ (h,w,c,kw) inner loop   │
├───────────────────────────┼──────┼───────┼────────┼────────┼─────────────────────────┤
│ TOTAL per Subcluster V4   │ 120  │ 31    │ 10,380 │ 11,600 │                         │
│ So sánh V3 per Sub        │(192) │(41)   │(15,462)│(14,336)│                         │
│ Thay đổi                  │ -72  │ -10   │ -5,082 │ -2,736 │ Nhỏ hơn đáng kể!       │
└───────────────────────────┴──────┴───────┴────────┴────────┴─────────────────────────┘

★ Key: mỗi sub V4 NHỎ HƠN V3 (ít DSP, ít LUT, ít BRAM)
  nhưng tổng 16 subs vẫn trong budget vì per-sub cost giảm 37.5% DSP
```

### 8.2. Breakdown Per SuperCluster (×4 instances)

```
┌───────────────────────────┬──────┬───────┬────────┬────────┐
│ Module                    │ DSP  │ BRAM  │ LUT    │ FF     │
├───────────────────────────┼──────┼───────┼────────┼────────┤
│ 4× Subcluster (V4)       │ 480  │ 124   │ 41,520 │ 46,400 │
│ local_arbiter_v2          │ 0    │ 0     │ 600    │ 350    │
│ (★4-sub Triple-RUNNING)   │      │       │(+100)  │(+50)   │
│ tensor_dma (2-ch: R+W)   │ 0    │ 6     │ 2,000  │ 1,200  │
│ (★dual-channel for T-RUN)│      │(+2)   │(+500)  │(+200)  │
│ tile_ingress_fifo (deeper)│ 0    │ 3     │ 250    │ 150    │
│ (★deeper for 4 subs)      │      │(+1)   │(+50)   │(+50)   │
├───────────────────────────┼──────┼───────┼────────┼────────┤
│ TOTAL per SuperCluster    │ 480  │ 133   │ 44,370 │ 48,100 │
│ So sánh V3 per SC         │(576) │(129)  │(48,586)│(44,408)│
└───────────────────────────┴──────┴───────┴────────┴────────┘
```

### 8.3. System Level Total

```
┌───────────────────────────┬──────┬───────┬─────────┬─────────┐
│ Module                    │ DSP  │ BRAM  │ LUT     │ FF      │
├───────────────────────────┼──────┼───────┼─────────┼─────────┤
│ 4× SuperCluster (V4)      │1,920 │ 532   │ 177,480 │ 192,400 │
│                           │      │       │         │         │
│ Controller System:        │      │       │         │         │
│  ├─ csr_register_bank     │ 0    │ 1     │ 500     │ 400     │
│  ├─ desc_fetch_engine     │ 0    │ 2     │ 800     │ 600     │
│  ├─ barrier_manager       │ 0    │ 0     │ 200     │ 100     │
│  └─ global_scheduler      │ 0    │ 1     │ 600     │ 450     │
│     (★16-sub dispatch)     │      │       │(+100)   │(+50)    │
│                           │      │       │         │         │
│ AXI Infrastructure:       │      │       │         │         │
│  ├─ AXI-Lite Slave        │ 0    │ 0     │ 1,000   │ 500     │
│  ├─ AXI4 Master Mux       │ 0    │ 4     │ 2,000   │ 1,500   │
│  └─ AXI Interconnect      │ 0    │ 4     │ 2,000   │ 1,000   │
│                           │      │       │         │         │
│ Clock/Reset/Pipeline      │ 0    │ 0     │ 2,000   │ 1,200   │
│ (★deeper pipeline regs)    │      │       │(+500)   │(+400)   │
├───────────────────────────┼──────┼───────┼─────────┼─────────┤
│ ★ GRAND TOTAL V4          │1,920 │ 544   │ 186,580 │ 198,150 │
│   So sánh V3              │(2304)│(528)  │(202,844)│(182,932)│
│   Thay đổi                │ -384 │ +16   │ -16,264 │+15,218  │
└───────────────────────────┴──────┴───────┴─────────┴─────────┘
```

### 8.4. Resource Comparison: V4 vs VC707 Budget

```
┌───────────────────┬──────────┬──────────┬────────┬──────────────────────────┐
│ Resource          │ Available│ V4 Used  │ %      │ Đánh giá                 │
├───────────────────┼──────────┼──────────┼────────┼──────────────────────────┤
│ DSP48E1           │ 2,800    │ 1,920    │ 68.6%  │ ✅ Sweet spot (60-70%)  │
│ BRAM36K           │ 1,030    │ 544      │ 52.8%  │ ✅ Thoải mái            │
│ LUT6              │ 303,600  │ 186,580  │ 61.5%  │ ✅ Rất tốt cho timing   │
│ FF (Flip-Flop)    │ 607,200  │ 198,150  │ 32.6%  │ ✅ Dư dả               │
└───────────────────┴──────────┴──────────┴────────┴──────────────────────────┘

                    DSP:  █████████████░░░░░░░░ 68.6%  ★ TRONG TARGET
                    BRAM: ██████████░░░░░░░░░░░ 52.8%
                    LUT:  ████████████░░░░░░░░░ 61.5%  ★ TRONG TARGET
                    FF:   ██████░░░░░░░░░░░░░░░ 32.6%
```

### 8.5. So sánh tổng thể V3 vs V4

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                          V3-VC707  vs  V4-VC707                              ║
╠═════════════════════════╦═══════════════════╦════════════════════════════════╣
║ Metric                  ║ V3                ║ V4                    Δ       ║
╠═════════════════════════╬═══════════════════╬════════════════════════════════╣
║ LANES                   ║ 32                ║ 20               -37.5%      ║
║ Subclusters             ║ 12 (4SC×3)        ║ 16 (4SC×4)       +33%        ║
║ Active subs             ║ 8                 ║ 12               +50%        ║
║ Clock                   ║ 200 MHz           ║ 250 MHz          +25%        ║
║ MACs/cycle (active)     ║ 3,072             ║ 2,880            -6.3%       ║
║ Peak GOPS               ║ 614.4             ║ 720.0            +17.2%      ║
╠═════════════════════════╬═══════════════════╬════════════════════════════════╣
║ Spatial utilization     ║ 71.1%             ║ 100.0%           +40.6%! ★★  ║
║ Temporal utilization    ║ 75%               ║ 82%              +9.3%       ║
║ Overall utilization     ║ 53.3%             ║ 82%              +53.8%! ★★  ║
║ Effective GOPS          ║ 327.5             ║ 590.4            +80.3%! ★★  ║
╠═════════════════════════╬═══════════════════╬════════════════════════════════╣
║ T_frame                 ║ 9.40 ms           ║ 5.21 ms          -44.6%!     ║
║ ★ FPS                   ║ 106               ║ 192              +81.1%! ★★★ ║
╠═════════════════════════╬═══════════════════╬════════════════════════════════╣
║ DSP usage               ║ 82.3%             ║ 68.6%            -13.7%  ✓   ║
║ BRAM usage              ║ 51.3%             ║ 52.8%            +1.5%       ║
║ LUT usage               ║ 66.8%             ║ 61.5%            -5.3%   ✓   ║
║ FF usage                ║ 30.1%             ║ 32.6%            +2.5%       ║
║ Power (est.)            ║ ~14 W             ║ ~13 W            -7%         ║
╠═════════════════════════╬═══════════════════╬════════════════════════════════╣
║ Accuracy (mAP50)        ║ ≥0.92             ║ ≥0.92            Không đổi   ║
║ Correctness             ║ Bit-exact atoms   ║ Bit-exact atoms  Không đổi   ║
╚═════════════════════════╩═══════════════════╩════════════════════════════════╝

★ KẾT LUẬN:
  V4 đạt GẦN GẤP ĐÔI FPS (192 vs 106) trong khi DÙNG ÍT RESOURCE HƠN!
  Secret weapon: LANES=20 = perfect spatial alignment cho YOLOv10n.
```

### 8.6. Power Estimation V4

```
Virtex-7 XC7VX485T @ 250 MHz:
  DSP dynamic: 1,920 DSPs × ~3.5 mW/DSP (@250MHz) = ~6.7 W
  BRAM dynamic: 544 BRAMs × ~1.8 mW/BRAM = ~1.0 W
  Logic dynamic: ~187K LUT × ~0.012 mW/LUT = ~2.2 W
  Clock tree (250 MHz): ~1.8 W
  Static: ~3.0 W
  ──────────────────────
  Total estimated: ~14.7 W → tương đương V3 (~14.2 W)

  VC707 board power budget: 50W+ → dư dả
  Efficiency: 192 FPS / 14.7W = 13.1 FPS/W (vs V3: 106/14.2 = 7.5 FPS/W)
  → V4 hiệu năng năng lượng tốt hơn 75%!
```

---

## 9. TẠI SAO 250 MHZ KHẢ THI VỚI LANES=20

### 9.1. Phân tích critical path

```
V3 Critical Path (LANES=32, 200 MHz = 5.0ns):
═══════════════════════════════════════════════

  router_cluster → 32-to-1 mux tree → pe_unit → 16 dsp_pairs →
  column_reduce (3×32 adders) → PPU (64-bit multiply) → glb_write

  Critical components:
    - Router weight mux: 32-wide → log2(32)=5 levels → ~2.0ns
    - DSP48E1 cascade: 4 stage → 3.2ns (includes DSP)
    - Column reduce: 32-lane × 3-row → ~0.8ns
    - Total: ~6.0ns → needs 200 MHz pipelining to meet 5.0ns

V4 Critical Path (LANES=20, 250 MHz = 4.0ns):
═══════════════════════════════════════════════

  router_cluster → 20-to-1 mux tree → pe_unit → 10 dsp_pairs →
  column_reduce (3×20 adders) → PPU (64-bit multiply) → glb_write

  Improvements:
    - Router weight mux: 20-wide → log2(20)≈4.3 levels → ~1.5ns (-25%)
    - DSP48E1 cascade: 5 stage (deeper) → 3.2ns (same DSP, more pipeline)
    - Column reduce: 20-lane × 3-row → ~0.6ns (-25%)
    - Extra pipeline registers between stages → break long paths

  Total path per stage with 5-stage pipeline: ~3.5ns → 285 MHz achievable
  Target 250 MHz = 4.0ns → 12.5% timing margin ✓

  ★ DSP48E1 trên Virtex-7: rated FMAX > 500 MHz
    Logic là bottleneck, KHÔNG phải DSP
    LANES=20 giảm logic fan-out → giảm routing delay
```

### 9.2. Pipeline depth comparison

```
                    V3 (4-stage)          V4 (5-stage)
                    ═══════════           ═══════════
  Stage 1:          Weight fetch          Weight fetch + reg
  Stage 2:          PE multiply           PE multiply (DSP)
  Stage 3:          PE accumulate         PE accumulate (DSP)
  Stage 4:          Column reduce + PPU   Column reduce + reg
  Stage 5:          —                     PPU requant + write

  Cost: +1 cycle latency per tile
  Benefit: -20% critical path → +25% clock frequency
  Net: (+25% freq - ~1% extra latency) = +24% throughput ✓

  Additional registers (per sub):
    router → PE:       +20 FFs per lane × 12 PEs = +240 FFs
    column_reduce → PPU: +20 FFs per lane × 4 cols = +80 FFs
    addr_gen outputs:   +~50 FFs
    Total:              ~370 FFs per sub → 5,920 FFs for 16 subs
    → Trivial: 5,920 / 607,200 = 0.97% FF increase
```

---

## 10. DOUBLE-BUFFERED GLB — CHI TIẾT

### 10.1. Concept

```
SINGLE BUFFER (V3):
═══════════════════
  Time:  │── FILL ──│── COMPUTE ──│── DRAIN ──│── FILL ──│── COMPUTE ──│
  GLB:   │  Write   │    Read     │  Read     │  Write   │    Read     │
  PE:    │  IDLE    │   ACTIVE    │  IDLE     │  IDLE    │   ACTIVE    │

  Problem: PE idle during FILL and DRAIN → wasted cycles

DOUBLE BUFFER (V4):
═══════════════════
  Time:  │── FILL A ──│── COMPUTE A + FILL B ──│── COMPUTE B + FILL A ──│
  GLB-A: │  Write     │    Read                 │  (Write next tile)     │
  GLB-B: │  —         │    (Write next tile)    │    Read                │
  PE:    │  IDLE      │   ACTIVE ★              │   ACTIVE ★             │

  ★ PE active LIÊN TỤC sau initial fill! (trừ barrier stalls)

  Cost per GLB input bank:   +2 BRAM (page B)
  Cost per subcluster:       +6 BRAM (3 input banks × 2)
  Total cost: 16 × 6 = 96 BRAM → 9.3% of total BRAM
  → Included in resource estimation (Section 8)
```

### 10.2. Implementation

```
glb_input_bank_db.sv (double-buffered version):
  - 2 SRAM pages: page_A[DEPTH][LANES], page_B[DEPTH][LANES]
  - page_sel register: toggles on tile boundary
  - Read port: always reads from ACTIVE page (compute)
  - Write port: always writes to SHADOW page (DMA fill)
  - Swap: when tile_fsm transitions COMPUTE → DONE, toggle page_sel

  Port A (compute read): addr_gen → SRAM[active_page] → router
  Port B (DMA write):    tensor_dma → SRAM[shadow_page]

  Timing: 0-cycle swap overhead (just flip a MUX select bit)
```

---

## 11. INFERENCE CORRECTNESS — KHÔNG THAY ĐỔI

```
★ V4 DÙNG CÙNG COMPUTE ATOMS VỚI V3:
═══════════════════════════════════════

  ✅ dsp_pair_int8: CÙNG 2-MAC packing, CÙNG unsigned-offset trick
     Chỉ thay đổi: 10 instances/PE (thay vì 16) + deeper pipeline
     Kết quả toán: IDENTICAL (chỉ thêm latency)

  ✅ pe_unit: CÙNG weight × activation multiply-accumulate
     LANES=20 thay vì 32 → ít lanes hơn nhưng CÙNG phép tính/lane

  ✅ PPU: CÙNG half-up rounding, CÙNG INT64 multiply, CÙNG ReLU
     Deeper pipeline → thêm latency nhưng kết quả BIT-EXACT

  ✅ Padding: CÙNG zero_point_x fill

  ✅ CONCAT domain alignment: CÙNG requant_to_common logic

  ✅ EWISE_ADD: CÙNG golden path (float intermediate)

  ✅ Descriptor sequence: CÙNG 60 descriptors, CÙNG thứ tự

  → MỌI thứ khác là "structural" (bao nhiêu subs, clock mấy Hz)
    KHÔNG ảnh hưởng kết quả tính toán.

  ★ Expected accuracy: IDENTICAL với V3
    mAP50 ≥ 0.92, mAP50-95 ≥ 0.71
```

---
---

# ════════════════════════════════════════════════════════════════
# PHẦN III: HỆ THỐNG RTL MODULE HIERARCHY — COMPLETE LISTING
# ════════════════════════════════════════════════════════════════

> **Mục tiêu**: Liệt kê ĐẦY ĐỦ mọi module SystemVerilog từ TOP đến
> primitive atoms, kèm chức năng, interfaces, và mối quan hệ.
> **Kiến trúc**: V4-VC707 (LANES=20, 16 subs, 250 MHz)
> **Tham chiếu**: SW golden primitives P0-P14, Layers L0-L22

---

## 12. MODULE TREE — TỔNG QUAN

```
accel_top.sv                                    ← TOP-LEVEL IP
├── accel_pkg.sv                                ← Package: types, params, enums
├── desc_pkg.sv                                 ← Package: descriptor structs
├── csr_pkg.sv                                  ← Package: CSR address map
│
├── [INPUT STAGE]
│   ├── csr_register_bank.sv                    ← AXI-Lite CPU interface
│   ├── desc_fetch_engine.sv                    ← DDR3 → descriptor parser
│   ├── barrier_manager.sv                      ← 4-point skip sync
│   └── global_scheduler.sv                     ← Tile dispatch → 4 SCs
│
├── [PROCESSING STAGE] × 4 SuperClusters
│   └── supercluster_wrapper.sv
│       ├── local_arbiter_v2.sv                 ← Triple-RUNNING scheduler
│       ├── tensor_dma_v2.sv                    ← Dual-channel AXI4 DMA
│       ├── tile_ingress_fifo.sv                ← Tile descriptor FIFO
│       │
│       └── [SUBCLUSTER] × 4 per SC = 16 total
│           └── subcluster_datapath.sv          ← ★ CORE: all-in-one compute
│               │
│               ├── [CONTROL]
│               │   ├── tile_fsm.sv             ← Phase-level FSM
│               │   ├── shadow_reg_file.sv      ← Descriptor config latch
│               │   └── compute_sequencer.sv    ← Cycle-level iteration
│               │
│               ├── [MEMORY]
│               │   ├── glb_input_bank_db.sv ×3 ← ★ Double-buffered input
│               │   ├── glb_weight_bank.sv ×3   ← 4-read-port weight SRAM
│               │   ├── glb_output_bank.sv ×4   ← Dual-namespace PSUM/ACT
│               │   └── metadata_ram.sv         ← Slot validity ring buffer
│               │
│               ├── [ADDRESS GENERATION]
│               │   ├── addr_gen_input.sv       ← (h,w,c) → bank + addr
│               │   ├── addr_gen_weight.sv      ← Per-column weight addr
│               │   └── addr_gen_output.sv      ← Column → output bank
│               │
│               ├── [DATA MOVEMENT]
│               │   ├── router_cluster_v2.sv    ← Multicast + per-col wgt
│               │   ├── window_gen.sv           ← Sliding window K=1,3,5,7
│               │   └── swizzle_engine.sv       ← Layout transform
│               │
│               ├── [COMPUTE]
│               │   └── pe_cluster_v4.sv        ← ★ 3×4×20 PE array
│               │       ├── pe_unit.sv ×12      ← 20-lane processing element
│               │       │   └── dsp_pair_int8.sv ×10  ← 2-MAC DSP48E1
│               │       ├── column_reduce.sv    ← Sum 3 rows per column
│               │       └── comparator_tree.sv  ← MaxPool 25→1 per lane
│               │
│               └── [POST-PROCESSING]
│                   ├── ppu.sv ×4               ← ★ 4 parallel PPUs
│                   └── silu_lut.sv (optional)  ← Unused, kept for generality
│
├── [AXI INFRASTRUCTURE]
│   ├── axi_lite_slave.sv                       ← CPU register access
│   ├── axi4_master_mux.sv                      ← 4 SC → 1 DDR3 arbiter
│   └── axi_interconnect.sv                     ← AXI routing fabric
│
└── [CLOCK/RESET]
    ├── clk_wiz_250.sv                          ← MMCM: 200MHz in → 250MHz
    └── reset_sync.sv                           ← Async reset synchronizer

TOTAL: ~38 unique module types, ~320+ instances
```

---

## 13. CHI TIẾT TỪNG MODULE — PACKAGES

### 13.1. `accel_pkg.sv` — Global Parameters & Types

```
Chức năng: Định nghĩa TẤT CẢ parameters, types, enums cho toàn bộ design.
           Mọi module khác `import accel_pkg::*;`

Nội dung chính:

  // ═══ DATAPATH PARAMETERS ═══
  parameter LANES           = 20;        // ★ V4: spatial parallelism
  parameter PE_ROWS         = 3;         // kh parallelism
  parameter PE_COLS         = 4;         // cout parallelism (per-col weight)
  parameter DSP_PAIRS_PER_PE = LANES/2;  // = 10
  parameter MACS_PER_PE     = LANES;     // = 20 (2 MACs per DSP pair)
  parameter MACS_PER_SUB    = PE_ROWS * PE_COLS * MACS_PER_PE;  // = 240

  // ═══ HIERARCHY PARAMETERS ═══
  parameter N_SUPER_CLUSTERS    = 4;
  parameter N_SUBS_PER_SC       = 4;     // ★ V4: 4 (từ 3)
  parameter N_TOTAL_SUBS        = N_SUPER_CLUSTERS * N_SUBS_PER_SC;  // = 16
  parameter N_ACTIVE_PER_SC     = 3;     // ★ V4: Triple-RUNNING

  // ═══ MEMORY PARAMETERS ═══
  parameter GLB_INPUT_DEPTH     = 2048;  // per bank, per page
  parameter GLB_WEIGHT_DEPTH    = 1024;  // per bank
  parameter GLB_OUTPUT_DEPTH    = 512;   // per bank (PSUM + ACT)
  parameter GLB_INPUT_PAGES     = 2;     // ★ V4: double-buffer
  parameter WEIGHT_READ_PORTS   = PE_COLS;  // = 4

  // ═══ PIPELINE PARAMETERS ═══
  parameter DSP_PIPELINE_DEPTH  = 5;     // ★ V4: 5-stage (từ 4)
  parameter PPU_PIPELINE_DEPTH  = 5;     // ★ V4: 5-stage (từ 4)

  // ═══ PE MODES (descriptor-driven) ═══
  typedef enum logic [3:0] {
    PE_RS3   = 4'd0,   // Conv 3×3 (P0: RS_DENSE_3x3)
    PE_OS1   = 4'd1,   // Conv 1×1 (P1: OS_1x1)
    PE_DW3   = 4'd2,   // DW Conv 3×3 (P2: DW_3x3)
    PE_DW7   = 4'd3,   // DW Conv 7×7 multipass (P8: DW_7x7_MULTIPASS)
    PE_MP5   = 4'd4,   // MaxPool 5×5 (P3: MAXPOOL_5x5)
    PE_PASS  = 4'd5,   // Bypass (P4/P5/P6: MOVE/CONCAT/UPSAMPLE)
    PE_GEMM  = 4'd6    // GEMM for attention (P9/P10: QPSA matmul)
  } pe_mode_t;

  // ═══ ACTIVATION TYPES ═══
  typedef enum logic [1:0] {
    ACT_NONE  = 2'd0,  // Identity (no activation)
    ACT_RELU  = 2'd1,  // max(0, x) (P14)
    ACT_SILU  = 2'd2,  // LUT-based SiLU (P13, unused)
    ACT_RELU6 = 2'd3   // clamp to [0, 6_quant] (P14 variant)
  } act_type_t;

  // ═══ SWIZZLE MODES ═══
  typedef enum logic [1:0] {
    SWZ_NORMAL     = 2'd0,  // Identity pass-through
    SWZ_UPSAMPLE2X = 2'd1,  // Nearest-neighbor 2× (P6)
    SWZ_CONCAT     = 2'd2   // Channel concatenation (P5)
  } swizzle_mode_t;

  // ═══ TILE FSM STATES ═══
  typedef enum logic [3:0] {
    TS_IDLE        = 4'd0,
    TS_LOAD_DESC   = 4'd1,
    TS_PREFILL_WT  = 4'd2,
    TS_PREFILL_IN  = 4'd3,
    TS_COMPUTE     = 4'd4,
    TS_PE_DRAIN    = 4'd5,
    TS_PPU_RUN     = 4'd6,
    TS_SWIZZLE     = 4'd7,
    TS_WRITEBACK   = 4'd8,
    TS_DONE        = 4'd9
  } tile_state_t;

  // ═══ DATA TYPES ═══
  typedef logic signed [7:0]  int8_t;
  typedef logic signed [31:0] int32_t;
  typedef logic signed [63:0] int64_t;
```

### 13.2. `desc_pkg.sv` — Descriptor Structures

```
Chức năng: Định nghĩa 3-level descriptor hierarchy cho descriptor-driven execution.

  // ═══ NET DESCRIPTOR (1 per inference) ═══
  typedef struct packed {
    logic [31:0] magic;              // 0xACC10001
    logic [15:0] version;
    logic [7:0]  num_layers;         // 23 (L0-L22)
    logic [63:0] weight_arena_base;  // DDR3 address
    logic [63:0] act0_arena_base;    // DDR3 address
    logic [63:0] act1_arena_base;    // DDR3 address (double buffer)
  } net_desc_t;                      // 64 bytes

  // ═══ LAYER DESCRIPTOR (23 per inference, L0-L22) ═══
  typedef struct packed {
    pe_mode_t    pe_mode;           // PE_RS3, PE_OS1, PE_DW3, ...
    act_type_t   activation;        // ACT_RELU, ACT_NONE, ...
    logic [9:0]  cin, cout;         // Input/output channels
    logic [9:0]  hin, win;          // Input height/width
    logic [9:0]  hout, wout;        // Output height/width
    logic [3:0]  kh, kw;            // Kernel size
    logic [2:0]  stride;            // 1 or 2
    logic [2:0]  padding;           // 0, 1, 2, or 3
    logic [7:0]  num_tiles;         // Number of tiles for this layer
    logic [3:0]  num_cin_pass;      // For multi-pass cin accumulation
    logic [3:0]  num_k_pass;        // For DW7x7 multipass (=3)
    swizzle_mode_t swizzle;         // Post-compute layout transform
    logic [7:0]  router_profile_id; // Routing configuration
    logic [7:0]  post_profile_id;   // PPU configuration
  } layer_desc_t;                   // 32 bytes

  // ═══ TILE DESCRIPTOR (N per layer, ~60 total) ═══
  typedef struct packed {
    logic [7:0]  tile_id;
    logic [7:0]  layer_id;
    logic [3:0]  sc_mask;           // Which SCs handle this tile
    logic [9:0]  h_out0, wblk0;    // Output tile origin
    logic [9:0]  cin0, cout0;      // Channel tile origin
    logic [31:0] src_in_off;       // DDR offset: input activation
    logic [31:0] src_w_off;        // DDR offset: weights
    logic [31:0] src_skip_off;     // DDR offset: skip connection
    logic [31:0] dst_off;          // DDR offset: output
    // Flags:
    logic        first_tile;        // First tile of layer
    logic        last_tile;         // Last tile of layer
    logic        hold_skip;         // Hold output for skip connection
    logic        need_swizzle;      // Apply swizzle_engine
    logic        barrier_wait;      // Wait for barrier before start
    logic [3:0]  barrier_id;        // Which barrier to wait/signal
  } tile_desc_t;                    // 32 bytes
```

### 13.3. `csr_pkg.sv` — CSR Address Map

```
Chức năng: Ánh xạ địa chỉ CSR cho CPU access qua AXI-Lite.

  parameter CSR_CTRL         = 12'h000;  // [R/W] start, reset, irq_clear
  parameter CSR_STATUS       = 12'h004;  // [R]   busy, done, error
  parameter CSR_NET_DESC_LO  = 12'h010;  // [R/W] Descriptor base [31:0]
  parameter CSR_NET_DESC_HI  = 12'h014;  // [R/W] Descriptor base [63:32]
  parameter CSR_LAYER_START  = 12'h018;  // [R/W] First layer (0-22)
  parameter CSR_LAYER_END    = 12'h01C;  // [R/W] Last layer (0-22)
  parameter CSR_PERF_CYCLES  = 12'h020;  // [R]   Total cycle counter
  parameter CSR_PERF_STALLS  = 12'h024;  // [R]   Stall cycle counter
  parameter CSR_PERF_TILES   = 12'h028;  // [R]   Completed tile counter
  parameter CSR_IRQ_MASK     = 12'h02C;  // [R/W] Interrupt enable mask
  parameter CSR_VERSION      = 12'h030;  // [R]   IP version
  parameter CSR_CONFIG       = 12'h034;  // [R]   LANES, SUBS, CLOCK
```

---

## 14. CHI TIẾT TỪNG MODULE — INPUT STAGE

### 14.1. `csr_register_bank.sv`

```
Chức năng:    CPU-accessible Control/Status Registers qua AXI-Lite
Interface:    AXI-Lite Slave (32-bit addr, 32-bit data)
Dependency:   csr_pkg.sv

Ports:
  // AXI-Lite Slave
  input  axi_lite_if.slave    s_axi,
  // To controller
  output logic                ctrl_start,
  output logic                ctrl_soft_reset,
  output logic [63:0]         net_desc_base,
  output logic [7:0]          layer_start, layer_end,
  // From processing
  input  logic                status_busy, status_done, status_error,
  input  logic [31:0]         perf_cycles, perf_stalls, perf_tiles,
  // Interrupt
  output logic                irq

Behavior:
  CPU ghi CSR_CTRL.start=1 → ctrl_start pulse
  Processing complete → status_done=1 → assert irq
  CPU đọc status → clear irq

Resources: ~500 LUT, ~400 FF, 1 BRAM (optional for deep CSR)
```

### 14.2. `desc_fetch_engine.sv`

```
Chức năng:    Đọc 3-level descriptors từ DDR3, parse thành structs
Interface:    AXI4 Master (read only) → DDR3
Dependency:   desc_pkg.sv

Ports:
  // AXI4 Master (read)
  output axi4_if.master       m_axi_rd,
  // Input
  input  logic [63:0]         net_desc_base,
  input  logic [7:0]          layer_start, layer_end,
  input  logic                start,
  // Output to global_scheduler
  output net_desc_t           net_desc,
  output layer_desc_t         current_layer_desc,
  output tile_desc_t          tile_desc,
  output logic                tile_desc_valid,
  input  logic                tile_desc_accept,
  // Status
  output logic                all_layers_done

FSM:
  DF_IDLE → DF_FETCH_NET → DF_PARSE_NET → DF_FETCH_LAYER →
  DF_PARSE_LAYER → DF_FETCH_TILE → DF_DISPATCH_TILE →
  DF_NEXT_TILE → (DF_FETCH_TILE | DF_FETCH_LAYER | DF_IDLE)

  Prefetch: pipeline tile N+1 fetch while tile N is dispatching
  Burst: single-beat reads (descriptors are small)

Resources: ~800 LUT, ~600 FF, 2 BRAM (descriptor cache)
```

### 14.3. `barrier_manager.sv`

```
Chức năng:    Đồng bộ 4 skip connection barrier points trong YOLOv10n
Interface:    Signal/wait handshake

Ports:
  input  logic                clk, rst_n,
  // Producer side (signal when skip tensor ready)
  input  logic                barrier_signal_valid,
  input  logic [3:0]          barrier_signal_id,
  // Consumer side (wait until barrier released)
  input  logic                barrier_wait_valid,
  input  logic [3:0]          barrier_wait_id,
  output logic                barrier_grant,
  // Clear (per-inference reset)
  input  logic                barrier_clear_all

4 Barriers trong YOLOv10n:
  barrier_0: L6 → L12   (F6 skip, hold ~6 layers)
  barrier_1: L4 → L15   (F4 skip, hold ~11 layers)
  barrier_2: L13 → L18  (F13 skip, hold ~5 layers)
  barrier_3: L8 → L21   (F8 skip, hold ~13 layers)

Implementation:
  32-bit scoreboard register
  signal → set bit[id]
  wait → grant when bit[id] == 1, then clear bit

Resources: ~200 LUT, ~100 FF
```

### 14.4. `global_scheduler.sv`

```
Chức năng:    Nhận tiles từ desc_fetch, dispatch cho 4 SuperClusters
Interface:    tile input → SC tile outputs

Ports:
  input  tile_desc_t          tile_desc,
  input  layer_desc_t         layer_desc,
  input  logic                tile_valid,
  output logic                tile_accept,
  // Per-SC output
  output tile_desc_t          sc_tile_desc  [N_SUPER_CLUSTERS],
  output layer_desc_t         sc_layer_desc [N_SUPER_CLUSTERS],
  output logic                sc_tile_valid [N_SUPER_CLUSTERS],
  input  logic                sc_tile_accept[N_SUPER_CLUSTERS],
  // Status
  output logic                inference_complete

Dispatch logic:
  1. Đọc sc_mask[3:0] từ tile_desc
  2. Round-robin hoặc least-loaded SC selection
  3. Forward tile_desc + layer_desc cho SC được chọn
  4. Track: tiles_dispatched, layers_complete

Resources: ~600 LUT, ~450 FF, 1 BRAM (tile queue)
```

---

## 15. CHI TIẾT TỪNG MODULE — SUPERCLUSTER LEVEL

### 15.1. `supercluster_wrapper.sv` (×4 instances)

```
Chức năng:    Chứa 4 subclusters + arbiter + DMA, quản lý tile execution
              ★ V4: 4 subs per SC (tăng từ 3 cho Triple-RUNNING)

Ports:
  // From global_scheduler
  input  tile_desc_t          tile_desc,
  input  layer_desc_t         layer_desc,
  input  logic                tile_valid,
  output logic                tile_accept,
  // AXI4 Master (DDR3 access)
  output axi4_if.master       m_axi,
  // Barrier interface
  output logic                barrier_signal_valid,
  output logic [3:0]          barrier_signal_id,
  input  logic                barrier_wait_valid,
  input  logic [3:0]          barrier_wait_id,
  input  logic                barrier_grant

Sub-modules:
  ├── local_arbiter_v2       ← Triple-RUNNING scheduler (4 subs)
  ├── tensor_dma_v2          ← Dual-channel AXI4 (read + write parallel)
  ├── tile_ingress_fifo      ← 8-deep tile descriptor FIFO
  └── subcluster[0..3]       ← 4 compute units

Resources per SC: 480 DSP, 133 BRAM, ~44K LUT, ~48K FF
```

### 15.2. `local_arbiter_v2.sv`

```
Chức năng:    ★ Triple-RUNNING scheduler cho 4 subclusters
              3 active + 1 fill/drain tại mọi thời điểm

Ports:
  input  logic                clk, rst_n,
  // Sub status
  input  logic [3:0]          sub_compute_done,    // pulse when tile done
  input  logic [3:0]          sub_needs_fill,      // request input data
  input  logic [3:0]          sub_needs_drain,     // request output write
  // Role assignment
  output logic [1:0]          sub_role [4],        // COMPUTE/FILL/DRAIN/IDLE
  // DMA port grant
  output logic [1:0]          dma_read_grant,      // which sub gets DMA read
  output logic [1:0]          dma_write_grant      // which sub gets DMA write

Role enum:
  ROLE_COMPUTE = 2'd0   (PE active, processing tile)
  ROLE_FILL    = 2'd1   (DMA loading next tile's inputs/weights)
  ROLE_DRAIN   = 2'd2   (DMA writing prev tile's outputs)
  ROLE_IDLE    = 2'd3   (waiting for assignment)

State machine:
  ┌─────────────────────────────────────────────────────────────┐
  │ Phase │ Sub-0    │ Sub-1    │ Sub-2    │ Sub-3              │
  ├───────┼──────────┼──────────┼──────────┼────────────────────┤
  │ P0    │ COMPUTE  │ COMPUTE  │ COMPUTE  │ FILL+DRAIN         │
  │ P1    │ FILL+DRN │ COMPUTE  │ COMPUTE  │ COMPUTE            │
  │ P2    │ COMPUTE  │ FILL+DRN │ COMPUTE  │ COMPUTE            │
  │ P3    │ COMPUTE  │ COMPUTE  │ FILL+DRN │ COMPUTE            │
  └───────┴──────────┴──────────┴──────────┴────────────────────┘

  "FILL+DRAIN" = DMA read channel fills next tile +
                 DMA write channel drains prev tile (CONCURRENT)

Resources: ~600 LUT, ~350 FF
```

### 15.3. `tensor_dma_v2.sv`

```
Chức năng:    ★ Dual-channel AXI4 master: read + write can operate simultaneously
              Read: DDR3 → GLB (fill input/weight)
              Write: GLB → DDR3 (drain output/skip)

Ports:
  // AXI4 Master
  output axi4_if.master       m_axi,
  // Read channel (fill)
  input  logic [31:0]         fill_src_addr,
  input  logic [15:0]         fill_length,
  input  logic                fill_start,
  output logic                fill_done,
  output logic [LANES*8-1:0]  fill_data,          // 160 bits (20 bytes)
  output logic                fill_data_valid,
  // Write channel (drain)
  input  logic [31:0]         drain_dst_addr,
  input  logic [15:0]         drain_length,
  input  logic                drain_start,
  output logic                drain_done,
  input  logic [LANES*8-1:0]  drain_data,
  input  logic                drain_data_valid,
  output logic                drain_data_ready

AXI4 burst: max 16 beats × 20 bytes = 320 bytes (LANES=20)
  Hoặc: 32-byte AXI data width → 16 beats × 32B = 512 bytes

Bandwidth: 32 bytes/cycle × 250 MHz = 8.0 GB/s (each direction)
  DDR3 limit: 12.8 GB/s total → 4 SCs share → 3.2 GB/s/SC
  → Sufficient: fill ~200 MB/s + drain ~100 MB/s << 3.2 GB/s per SC

Resources: ~2,000 LUT, ~1,200 FF, 6 BRAM (FIFOs)
```

### 15.4. `tile_ingress_fifo.sv`

```
Chức năng:    Decouple global_scheduler speed từ subcluster processing speed
              FIFO chứa tile descriptors chờ xử lý

Ports:
  // Write (from SC wrapper)
  input  tile_desc_t          wr_data,
  input  logic                wr_valid,
  output logic                wr_ready,
  // Read (to local_arbiter → free sub)
  output tile_desc_t          rd_data,
  output logic                rd_valid,
  input  logic                rd_ready

Depth: 8 entries (★ V4: deeper cho 4 subs, mỗi sub cần lookahead)
Width: sizeof(tile_desc_t) = 256 bits

Resources: ~250 LUT, ~150 FF, 3 BRAM
```

---

## 16. CHI TIẾT TỪNG MODULE — SUBCLUSTER (★ CORE)

### 16.0. `subcluster_datapath.sv` — TOP INTEGRATION

```
Chức năng:    ★ CORE MODULE — 1 phần cứng cố định xử lý MỌI primitive.
              Wire up tất cả 21+ sub-module instances.
              Đây là "trái tim" của toàn bộ accelerator.

Ports:
  input  logic                clk, rst_n,
  // Tile descriptor input
  input  tile_desc_t          tile_desc,
  input  layer_desc_t         layer_desc,
  input  logic                tile_start,
  output logic                tile_done,
  // DMA interface (fill/drain)
  // ... (connect to tensor_dma via arbiter)
  // Barrier interface
  output logic                barrier_signal,
  input  logic                barrier_grant

Internal wiring (★ QUAN TRỌNG — kết nối 24 instances):

  ═══ CONTROL ═══
  tile_fsm ←→ shadow_reg_file ←→ compute_sequencer
       │ phase commands          │ cycle-level iteration
       ▼                         ▼
  ═══ ADDRESS ═══
  addr_gen_input → glb_input_bank_db[0..2]
  addr_gen_weight → glb_weight_bank[0..2]
  addr_gen_output → glb_output_bank[0..3]

  ═══ DATA FLOW (COMPUTE path) ═══
  glb_input_bank_db → router_cluster_v2 (RIN: multicast act) →
    → window_gen → pe_cluster_v4 (3×4×20 PEs) →
    → column_reduce (3→1 per col) →
    → ppu[0..3] (bias + requant + ReLU) →
    → glb_output_bank[0..3] (ACT namespace write)

  ═══ DATA FLOW (WEIGHT path) ═══
  glb_weight_bank → router_cluster_v2 (RWT: per-col routing) →
    → pe_cluster_v4 (4 different cout weights per column)

  ═══ DATA FLOW (BYPASS path — PE_PASS mode) ═══
  glb_input_bank_db → router_cluster_v2 (bypass) →
    → swizzle_engine → glb_output_bank

  ═══ DATA FLOW (MAXPOOL path) ═══
  glb_input_bank_db → window_gen (K=5) →
    → comparator_tree (25→1 max/lane) →
    → glb_output_bank (direct, no PPU)

  ═══ DATA FLOW (PSUM accumulation — DW7x7 multipass) ═══
  glb_output_bank (PSUM namespace read) →
    → pe_cluster_v4 (add to new partial sum) →
    → glb_output_bank (PSUM namespace write)
  Pass 3: → ppu → glb_output_bank (ACT namespace)
```

### 16.1. CONTROL MODULES

#### `tile_fsm.sv`

```
Chức năng:    Phase-level FSM — điều khiển luồng xử lý tile ở mức CAO.
              "Khi nào làm gì" (fill, compute, post-process, write)

Ports:
  input  logic                clk, rst_n,
  input  logic                tile_start,
  input  tile_desc_t          tile_desc,
  input  layer_desc_t         layer_desc,
  // To shadow_reg_file
  output logic                latch_config,
  // To compute_sequencer
  output logic                seq_start,
  input  logic                seq_done,
  // To DMA
  output logic                fill_request, drain_request,
  input  logic                fill_done, drain_done,
  // To PPU
  output logic                ppu_start,
  input  logic                ppu_done,
  // To swizzle
  output logic                swizzle_start,
  input  logic                swizzle_done,
  // Barrier
  output logic                barrier_signal,
  input  logic                barrier_grant,
  // Status
  output tile_state_t         current_state,
  output logic                tile_done

FSM States (10):
  TS_IDLE       → Waiting for tile_start
  TS_LOAD_DESC  → Latch descriptor into shadow_reg_file
  TS_PREFILL_WT → DMA load weights into GLB weight banks
  TS_PREFILL_IN → DMA load inputs into GLB input banks (page B)
  TS_COMPUTE    → compute_sequencer runs (h,w,c,kw loops)
  TS_PE_DRAIN   → Wait for PE pipeline to flush
  TS_PPU_RUN    → PPU processes accumulated PSUM → INT8
  TS_SWIZZLE    → Layout transform (upsample/concat) if needed
  TS_WRITEBACK  → DMA drain outputs to DDR3
  TS_DONE       → Signal tile_done, return to IDLE

  Multipass loop (DW7x7): TS_COMPUTE → TS_PE_DRAIN → TS_COMPUTE
    (num_k_pass times)

Resources: ~300 LUT, ~150 FF
```

#### `shadow_reg_file.sv`

```
Chức năng:    Latch descriptor fields thành stable config signals.
              Prevents descriptor changing during compute.

Ports:
  input  logic                clk, rst_n,
  input  logic                latch_en,          // from tile_fsm
  input  layer_desc_t         layer_desc,
  input  tile_desc_t          tile_desc,
  // Stable outputs (held until next latch)
  output pe_mode_t            cfg_pe_mode,
  output act_type_t           cfg_activation,
  output logic [9:0]          cfg_cin, cfg_cout,
  output logic [9:0]          cfg_hin, cfg_win, cfg_hout, cfg_wout,
  output logic [3:0]          cfg_kh, cfg_kw,
  output logic [2:0]          cfg_stride, cfg_padding,
  output logic [3:0]          cfg_num_cin_pass, cfg_num_k_pass,
  output swizzle_mode_t       cfg_swizzle,
  output logic [31:0]         cfg_quant_m_int,   // requant multiplier
  output logic [7:0]          cfg_quant_shift,   // requant shift
  output logic signed [7:0]   cfg_zp_x, cfg_zp_y, // zero points
  output logic signed [31:0]  cfg_bias [PE_COLS], // per-column bias
  // Tile-specific
  output logic [9:0]          cfg_h_out0, cfg_wblk0,
  output logic [9:0]          cfg_cin0, cfg_cout0,
  output logic                cfg_first_tile, cfg_last_tile,
  output logic                cfg_hold_skip, cfg_need_swizzle

Resources: ~100 LUT, ~500 FF
```

#### `compute_sequencer.sv`

```
Chức năng:    Cycle-level FSM — drives inner loops (h, w, cin, kw).
              "Chính xác cycle nào feed data gì cho PE"
              tile_fsm = KHI NÀO; sequencer = CÁI GÌ MỖI CYCLE

Ports:
  input  logic                clk, rst_n,
  input  logic                seq_start,
  output logic                seq_done,
  // Config (from shadow_reg_file)
  input  pe_mode_t            cfg_pe_mode,
  input  logic [9:0]          cfg_cin, cfg_cout, cfg_hout, cfg_wout,
  input  logic [3:0]          cfg_kh, cfg_kw,
  input  logic [2:0]          cfg_stride,
  // Address generation drives
  output logic [9:0]          iter_h, iter_w, iter_cin, iter_cout_group,
  output logic [3:0]          iter_kw, iter_kh_row,
  // PE control
  output logic                pe_enable,
  output logic                pe_clear_acc,      // start new accumulation
  output logic                pe_acc_valid,       // accumulation complete
  // PPU trigger
  output logic                ppu_trigger,
  output logic [9:0]          ppu_cout_base,     // which 4-cout group

Inner loop structure (mode-dependent):

  PE_RS3 (Conv 3×3):
    for h_out = 0..Hout-1:
      for wblk = 0..ceil(Wout/LANES)-1:          // ★ LANES=20
        for cout_group = 0..ceil(Cout/PE_COLS)-1: // step 4
          for cin = 0..Cin-1:
            for kw = 0..2:
              → feed PE: act_row[kh_row][h*s+kh_row][wblk*20+kw]
                         wgt[cout_group*4+col][cin][kh_row][kw]
          → PPU trigger (4 cout outputs)

  PE_OS1 (Conv 1×1):
    for h_out = 0..Hout-1:
      for wblk = 0..ceil(Wout/LANES)-1:
        for cout_group = 0..ceil(Cout/4)-1:
          for cin = 0..Cin-1:
            → feed 1 cycle per cin (no kw loop)
          → PPU trigger

  PE_DW3 (DW Conv 3×3):
    for h_out = 0..Hout-1:
      for wblk = 0..ceil(Wout/LANES)-1:
        for ch_group = 0..ceil(C/4)-1:           // 4 channels per iteration
          for kw = 0..2:
            → feed PE: act[ch_base+col][h*s+kh][w+kw]
                       wgt[ch_base+col][kh][kw]
          → PPU trigger (per-channel params)

  PE_DW7 (DW 7×7 multipass):
    for pass = 0..2:                              // 3 passes
      for h_out = 0..Hout-1:
        for wblk = 0..ceil(Wout/LANES)-1:
          for ch_group = 0..ceil(C/4)-1:
            for kw = 0..6:
              → feed PE rows with kh_rows of current pass
          if pass < 2: → write PSUM to GLB_OUT(PSUM namespace)
          if pass == 2: → PPU trigger → INT8 output

  PE_MP5 (MaxPool 5×5):
    PE cluster BYPASSED
    → window_gen feeds 25 values/lane → comparator_tree → output

  PE_PASS (Upsample/Concat/Move):
    PE cluster + PPU BYPASSED
    → swizzle_engine handles address remapping

Resources: ~380 LUT, ~300 FF
```

### 16.2. MEMORY MODULES

#### `glb_input_bank_db.sv` (×3 instances per sub)

```
Chức năng:    ★ Double-buffered input activation SRAM.
              Page A: compute reads; Page B: DMA fills simultaneously.
              V4 key: eliminates fill stall overhead.

Ports:
  input  logic                clk, rst_n,
  // Page control
  input  logic                page_swap,          // toggle on tile boundary
  // Compute read port (active page)
  input  logic [11:0]         rd_addr,
  output logic [LANES*8-1:0]  rd_data,            // 160 bits (20 bytes)
  // DMA write port (shadow page)
  input  logic [11:0]         wr_addr,
  input  logic [LANES*8-1:0]  wr_data,
  input  logic                wr_en,
  input  logic [LANES-1:0]    wr_lane_mask        // per-lane write enable

Implementation:
  SRAM page_A [GLB_INPUT_DEPTH][LANES × 8];   // 2048 × 160 bits
  SRAM page_B [GLB_INPUT_DEPTH][LANES × 8];
  logic active_page;                            // 0=A, 1=B

  Read: rd_data = active_page ? page_B[rd_addr] : page_A[rd_addr];
  Write: always writes to ~active_page (shadow)
  Swap: active_page <= ~active_page; (1-cycle, 0-stall)

Banking: bank_id = h mod 3 (3 banks cover 3 PE rows for kh=0,1,2)

Resources per bank: ~150 LUT, ~80 FF, 2.5 BRAM (×2 pages)
  ≈ 8 BRAM for 3 banks (double-buffered)
```

#### `glb_weight_bank.sv` (×3 instances per sub)

```
Chức năng:    Weight SRAM với 4 read ports (1 per PE column).
              ★ V4/V3 key feature: per-column weight = per-column cout.

Ports:
  input  logic                clk, rst_n,
  // 4 read ports (per column)
  input  logic [9:0]          rd_addr [PE_COLS],  // 4 different addresses
  output logic [LANES*8-1:0]  rd_data [PE_COLS],  // 4 × 160 bits
  // 1 write port (DMA fill)
  input  logic [9:0]          wr_addr,
  input  logic [LANES*8-1:0]  wr_data,
  input  logic                wr_en,
  // Staging FIFO
  output logic [LANES*8-1:0]  staged_data [PE_COLS],
  output logic                staged_valid

Implementation options:
  Option A (selected): 4× BRAM duplicate
    - 4 copies of weight data, each serves 1 column
    - Write: broadcast to all 4 copies
    - Read: each copy independent address
    - Cost: 4× BRAM per bank
    - Benefit: 0-cycle latency, full bandwidth

  Option B: Wide BRAM + demux
    - 1 BRAM 4× wider → read all 4 columns in 1 cycle
    - Cost: fewer BRAM but more LUT for demux
    - Limitation: all 4 columns must read same row (OK for RS3/OS1)

  8-deep staging FIFO: pre-fetches weights ahead of PE consumption
    → hides BRAM read latency

Resources per bank: ~500 LUT, ~280 FF, 3.3 BRAM
  ≈ 10 BRAM for 3 banks
```

#### `glb_output_bank.sv` (×4 instances per sub)

```
Chức năng:    Dual-namespace output SRAM: PSUM (INT32) + ACT (INT8).
              1 bank per PE column. Stores intermediate PSUM for multipass
              and final INT8 activations.

Ports:
  input  logic                clk, rst_n,
  // PSUM namespace (INT32, for multipass accumulation)
  input  logic [8:0]          psum_addr,
  input  logic [31:0]         psum_wr_data [LANES],  // 20 × 32 bits
  input  logic                psum_wr_en,
  output logic [31:0]         psum_rd_data [LANES],
  // ACT namespace (INT8, final output)
  input  logic [8:0]          act_addr,
  input  logic [7:0]          act_wr_data [LANES],   // 20 × 8 bits
  input  logic                act_wr_en,
  output logic [7:0]          act_rd_data [LANES],
  // DMA drain port (read ACT for writeback)
  input  logic [8:0]          drain_addr,
  output logic [LANES*8-1:0]  drain_data

Implementation:
  PSUM SRAM: 512 × (20 × 32) = 512 × 640 bits
  ACT SRAM:  512 × (20 × 8)  = 512 × 160 bits
  Namespace select: based on tile_fsm state

Resources per bank: ~180 LUT, ~90 FF, 3 BRAM
  ≈ 12 BRAM for 4 banks
```

#### `metadata_ram.sv`

```
Chức năng:    Slot validity + ring buffer management cho multipass PSUM.
              Tracks which GLB_output slots contain valid PSUM data.

Ports:
  input  logic                clk, rst_n,
  // Slot management
  input  logic [3:0]          slot_alloc_id,
  input  logic                slot_alloc_en,
  input  logic [3:0]          slot_free_id,
  input  logic                slot_free_en,
  output logic [15:0]         slot_valid_mask,
  // Ring buffer pointers
  output logic [3:0]          head_ptr, tail_ptr

Resources: ~50 LUT, ~30 FF, 1 BRAM
```

### 16.3. ADDRESS GENERATION MODULES

#### `addr_gen_input.sv`

```
Chức năng:    Chuyển đổi (h_in, w, cin) → (bank_id, sram_addr, is_padding)

Ports:
  input  logic                clk, rst_n,
  input  pe_mode_t            cfg_pe_mode,
  input  logic [9:0]          cfg_hin, cfg_win, cfg_cin,
  input  logic [2:0]          cfg_stride, cfg_padding,
  input  logic signed [7:0]   cfg_zp_x,           // zero-point for padding
  // From compute_sequencer
  input  logic [9:0]          iter_h, iter_w, iter_cin,
  input  logic [3:0]          iter_kh_row,
  // Output
  output logic [1:0]          bank_id,             // h_in mod 3
  output logic [11:0]         sram_addr,
  output logic                is_padding,          // out-of-bounds → fill zp_x
  output logic signed [7:0]   pad_value            // = cfg_zp_x (NOT 0!)

Address calculation:
  h_in = iter_h * cfg_stride + iter_kh_row - cfg_padding
  bank_id = h_in mod 3

  Padding detection:
    if h_in < 0 || h_in >= cfg_hin → is_padding = 1, pad_value = cfg_zp_x
    if w < 0 || w >= cfg_win → is_padding = 1, pad_value = cfg_zp_x

  ★ CRITICAL: Padding fills with zp_x (zero point), NOT 0!
    This ensures correct quantized convolution at image boundaries.

Resources: ~280 LUT, ~180 FF (★ +1 pipeline stage for 250 MHz)
```

#### `addr_gen_weight.sv`

```
Chức năng:    ★ Per-column weight address generation.
              4 khác addresses cho 4 PE columns = 4 khác output channels.

Ports:
  input  pe_mode_t            cfg_pe_mode,
  input  logic [9:0]          cfg_cin, cfg_cout,
  input  logic [3:0]          cfg_kw,
  // From compute_sequencer
  input  logic [9:0]          iter_cin, iter_cout_group,
  input  logic [3:0]          iter_kw, iter_kh_row,
  // Output: 4 different addresses (1 per PE column)
  output logic [9:0]          wgt_addr [PE_COLS],  // ★ 4 addresses
  output logic [1:0]          wgt_bank_id          // which weight bank

Per-mode address calculation:

  PE_RS3 (Conv 3×3):
    cout_base = iter_cout_group * 4
    for col = 0..3:
      wgt_addr[col] = (cout_base + col) * Cin * Kw + iter_cin * Kw + iter_kw
    wgt_bank_id = iter_kh_row  (bank 0=kh0, bank 1=kh1, bank 2=kh2)

  PE_OS1 (Conv 1×1):
    cout_base = iter_cout_group * 4
    for col = 0..3:
      wgt_addr[col] = (cout_base + col) * Cin + iter_cin
    wgt_bank_id = 0  (only 1 kh row)

  PE_DW3 (DW Conv 3×3):
    ch_base = iter_cout_group * 4  (channels, not cout)
    for col = 0..3:
      wgt_addr[col] = (ch_base + col) * Kw + iter_kw
    wgt_bank_id = iter_kh_row

  PE_DW7 (DW Conv 7×7 multipass):
    Similar to DW3 but with kh offset per pass

  PE_GEMM (Attention matmul):
    Reuses OS1 addressing (matrix as 1×1 conv)

Resources: ~380 LUT, ~220 FF
```

#### `addr_gen_output.sv`

```
Chức năng:    (h_out, w_out, cout) → (bank_id, sram_addr)
              bank_id = PE column index (direct mapping)

Ports:
  input  logic [9:0]          iter_h, iter_wblk, iter_cout_group,
  input  pe_mode_t            cfg_pe_mode,
  output logic [1:0]          bank_id [PE_COLS],   // = {0, 1, 2, 3}
  output logic [8:0]          sram_addr [PE_COLS]

  bank_id[col] = col  (column 0 → bank 0, etc.)
  sram_addr = h_out * Wblk_total + wblk

Resources: ~180 LUT, ~100 FF
```

### 16.4. DATA MOVEMENT MODULES

#### `router_cluster_v2.sv`

```
Chức năng:    ★ 3 routing networks: input multicast, per-column weight, output.
              Core Eyeriss-v2-inspired feature: activation sharing + weight splitting.

Ports:
  input  logic                clk, rst_n,
  input  pe_mode_t            cfg_pe_mode,
  // Input routing (RIN): GLB → PE rows
  input  logic signed [7:0]   glb_in_data [3][LANES],    // 3 banks × 20 lanes
  output logic signed [7:0]   pe_act [PE_ROWS][LANES],   // 3 rows × 20 lanes
  // Weight routing (RWT): GLB → PE rows × columns  ★ KEY CHANGE
  input  logic signed [7:0]   glb_wgt_data [3][PE_COLS][LANES], // 3 banks × 4 cols × 20
  output logic signed [7:0]   pe_wgt [PE_ROWS][PE_COLS][LANES], // 3×4×20
  // Output routing (RPS): PE columns → GLB output banks
  input  logic signed [31:0]  pe_psum [PE_COLS][LANES],  // 4 cols × 20 lanes
  output logic signed [31:0]  glb_out_psum [PE_COLS][LANES],
  // Bypass path (PE_PASS mode)
  input  logic signed [7:0]   bypass_in [LANES],
  output logic signed [7:0]   bypass_out [LANES],
  input  logic                bypass_en

Routing modes:

  COMPUTE mode (PE_RS3/OS1/DW3/DW7/GEMM):
    RIN: glb_in_data[bank] → MULTICAST → pe_act[row0] = pe_act[row1] = pe_act[row2]
         (cùng activation cho 3 rows, mỗi row xử lý khác kh)
    RWT: glb_wgt_data[bank][col] → pe_wgt[row][col]
         (★ mỗi column nhận KHÁC weight = khác cout)

  DW mode (PE_DW3/DW7):
    RIN: glb_in_data[bank] → pe_act[row] (mỗi col nhận KHÁC channel)
    RWT: per-channel weights

  BYPASS mode (PE_PASS):
    bypass_in → bypass_out (skip PE cluster entirely)
    Used for: MOVE (P4), CONCAT (P5), UPSAMPLE (P6)

Resources: ~1,000 LUT, ~700 FF (★ +1 pipeline reg for 250 MHz)
```

#### `window_gen.sv`

```
Chức năng:    Sliding window shift register: produces K tap outputs per cycle.
              Configurable K = 1 (conv1×1), 3 (conv3×3), 5 (maxpool5×5), 7 (DW7×7)

Ports:
  input  logic                clk, rst_n,
  input  logic [3:0]          cfg_k,              // kernel width: 1, 3, 5, 7
  input  logic                shift_in_valid,
  input  logic signed [7:0]   shift_in [LANES],   // 20 new values
  output logic signed [7:0]   taps [7][LANES],    // up to 7 taps × 20 lanes
  output logic                taps_valid,
  input  logic                flush               // clear all registers

Implementation:
  7 stages of 20-byte shift registers (total 7 × 20 = 140 FFs per lane concept)
  K=3: output taps[0..2], ignore taps[3..6]
  K=5: output taps[0..4] (for maxpool comparator tree)
  K=7: output taps[0..6] (for DW 7×7)
  K=1: output taps[0] only (for conv 1×1, bypass shift)

  Latency: K-1 cycles to fill + 1 cycle output valid

Resources: ~380 LUT, ~1,400 FF (shift registers)
```

#### `swizzle_engine.sv`

```
Chức năng:    Layout transform cho next layer input.
              3 modes: identity, upsample 2×, concat channel offset.

Ports:
  input  logic                clk, rst_n,
  input  swizzle_mode_t       cfg_mode,
  input  logic                start,
  output logic                done,
  // GLB read (source)
  output logic [8:0]          src_addr,
  input  logic signed [7:0]   src_data [LANES],
  // GLB write (destination)
  output logic [8:0]          dst_addr,
  output logic signed [7:0]   dst_data [LANES],
  output logic                dst_wr_en

Modes:
  SWZ_NORMAL:     dst[addr] = src[addr]  (identity, no-op in practice)

  SWZ_UPSAMPLE2X: Nearest-neighbor 2× upscale (P6: UPSAMPLE_NEAREST)
    dst[2h+dh][2w+dw][c] = src[h][w][c]  for dh,dw ∈ {0,1}
    No compute, purely address remapping
    Used in: L11 (20→40), L14 (40→80)

  SWZ_CONCAT:     Channel concatenation with domain alignment (P5: CONCAT)
    dst[h][w][0..Ca-1]    = requant(src_A, scale_A→scale_out)
    dst[h][w][Ca..Ca+Cb-1] = requant(src_B, scale_B→scale_out)
    Domain alignment: requant_to_common nếu scale khác nhau
    Used in: L12, L15, L18, L21 (QConcat)

Resources: ~280 LUT, ~250 FF
```

### 16.5. COMPUTE MODULES

#### `pe_cluster_v4.sv` — ★ CORE COMPUTE ARRAY

```
Chức năng:    ★ 3 rows × 4 columns × 20 lanes Processing Element array.
              Mỗi column tính KHÁC output channel (Eyeriss-inspired).
              CÙNG activation multicast cho 3 rows (kh parallelism).

Ports:
  input  logic                clk, rst_n,
  input  pe_mode_t            cfg_pe_mode,
  // Activation input (SHARED across columns — multicast)
  input  logic signed [7:0]   act_taps [PE_ROWS][LANES],    // 3 rows × 20 lanes
  // Weight input (DIFFERENT per column — per-col routing)
  input  logic signed [7:0]   wgt_data [PE_ROWS][PE_COLS][LANES], // 3×4×20
  // Control
  input  logic                pe_enable,
  input  logic                pe_clear_acc,
  // Output: partial sums per column (after column reduce)
  output logic signed [31:0]  col_psum [PE_COLS][LANES],    // 4 cols × 20 lanes
  output logic                col_psum_valid,
  // MaxPool output (bypass PE, from comparator_tree)
  output logic signed [7:0]   maxpool_out [LANES],
  output logic                maxpool_valid

Internal structure:

  generate
    for (r = 0; r < PE_ROWS; r++) begin : gen_row          // 3 rows
      for (c = 0; c < PE_COLS; c++) begin : gen_col         // 4 columns
        pe_unit #(.LANES(20)) u_pe (
          .clk      (clk),
          .rst_n    (rst_n),
          .pe_mode  (cfg_pe_mode),
          .x_in     (act_taps[r]),                           // ★ SHARED per row
          .w_in     (wgt_data[r][c]),                        // ★ DIFFERENT per col!
          .pe_enable(pe_enable),
          .clear_acc(pe_clear_acc),
          .psum_out (pe_psum[r][c])                          // [LANES] INT32
        );
      end
    end
  endgenerate

  // Column reduce: sum 3 rows per column → kh dimension reduction
  generate
    for (c = 0; c < PE_COLS; c++) begin : gen_col_reduce
      column_reduce #(.LANES(20), .N_ROWS(3)) u_col_red (
        .row_psum ({pe_psum[0][c], pe_psum[1][c], pe_psum[2][c]}),
        .col_sum  (col_psum[c])                              // final per-col output
      );
    end
  endgenerate

  // Comparator tree for MaxPool (PE_MP5 mode only)
  comparator_tree #(.LANES(20), .K(5)) u_comp_tree (
    .window_data (act_taps),  // 5×5 window values
    .max_out     (maxpool_out)
  );

★ DW MODE (PE_DW3, PE_DW7):
  4 columns = 4 KHÁC input channels (not 4 khác cout)
  Column 0 → channel c_base+0
  Column 1 → channel c_base+1
  Column 2 → channel c_base+2
  Column 3 → channel c_base+3
  Each column processes independently → 4× DW throughput

Resources: 120 DSP, ~3,820 LUT, ~5,320 FF
```

#### `pe_unit.sv` (×12 per cluster, ×192 total)

```
Chức năng:    Single Processing Element: 20 lanes × 2 MACs/lane = 20 MACs/cycle.
              Contains 10 dsp_pair_int8 instances.

Ports:
  input  logic                clk, rst_n,
  input  pe_mode_t            pe_mode,
  input  logic signed [7:0]   x_in [LANES],       // 20 activation values
  input  logic signed [7:0]   w_in [LANES],       // 20 weight values
  input  logic                pe_enable,
  input  logic                clear_acc,
  output logic signed [31:0]  psum_out [LANES]     // 20 accumulated results

Internal:
  generate
    for (l = 0; l < LANES; l += 2) begin : gen_dsp_pair
      dsp_pair_int8 u_dsp (
        .clk      (clk),
        .rst_n    (rst_n),
        .a0       (x_in[l]),
        .b0       (w_in[l]),
        .a1       (x_in[l+1]),
        .b1       (w_in[l+1]),
        .enable   (pe_enable),
        .clear    (clear_acc),
        .acc_out0 (psum_out[l]),
        .acc_out1 (psum_out[l+1])
      );
    end
  endgenerate

Mode-dependent behavior:
  PE_RS3: w_in = weight[cout][cin][kh][kw], accumulate over cin × kw
  PE_OS1: w_in = weight[cout][cin], broadcast to all lanes, accumulate over cin
  PE_DW3: w_in = weight[ch][kh][kw], per-channel, accumulate over kw
  PE_GEMM: similar to OS1 but for matrix multiply

Resources per PE: 10 DSP, ~250 LUT, ~400 FF
```

#### `dsp_pair_int8.sv` (×10 per PE, ×1,920 total)

```
Chức năng:    ★ ATOMIC COMPUTE UNIT — 2 signed INT8 MACs packed in 1 DSP48E1.
              This is the fundamental building block of all computation.

Ports:
  input  logic                clk, rst_n,
  input  logic signed [7:0]   a0, b0,              // MAC pair 0
  input  logic signed [7:0]   a1, b1,              // MAC pair 1
  input  logic                enable,
  input  logic                clear,               // reset accumulator
  output logic signed [31:0]  acc_out0, acc_out1   // accumulated results

Implementation (DSP48E1 2-MAC packing):
  ★ 5-stage pipeline (V4, increased from V3's 4-stage for 250 MHz):

  Stage 1: Input register
    a0_r <= a0; b0_r <= b0; a1_r <= a1; b1_r <= b1;

  Stage 2: Unsigned offset (signed → unsigned for DSP)
    a0_u = a0_r + 128;  b0_u = b0_r + 128;  // [0..255]
    a1_u = a1_r + 128;  b1_u = b1_r + 128;

  Stage 3: DSP48E1 multiply (18×25 bits)
    // Pack 2 multiplies: P = A_concat × B_concat
    // Low product:  a0_u × b0_u (bits [15:0] of 48-bit P)
    // High product: a1_u × b1_u (bits [33:18] of 48-bit P)
    // (Using unsigned-offset trick to exploit full DSP width)

  Stage 4: Correction + accumulate
    // Subtract offset: prod_signed = prod_unsigned - correction_term
    // correction = 128×b + 128×a + 128×128
    // Accumulate: acc += prod_signed

  Stage 5: Output register (★ V4 extra stage)
    acc_out0 <= acc[lane0];
    acc_out1 <= acc[lane1];

  Latency: 5 cycles (1 extra vs V3 for timing)
  Throughput: 2 MACs per cycle per DSP (sustained after pipeline fill)

Resources per instance: 1 DSP48E1, ~25 LUT, ~40 FF
  Total: 1,920 DSP48E1, ~48K LUT, ~76.8K FF (DSP-related only)
```

#### `column_reduce.sv`

```
Chức năng:    Sum 3 PE rows per column → kh dimension reduction.
              col_psum[col][lane] = Σ_{row=0..2} pe_psum[row][col][lane]

Ports:
  input  logic signed [31:0]  row_psum [PE_ROWS][LANES],  // 3 × 20
  output logic signed [31:0]  col_sum [LANES]              // 20

Implementation:
  Adder tree: 3 inputs → 1 output per lane
  Lane 0: col_sum[0] = row_psum[0][0] + row_psum[1][0] + row_psum[2][0]
  (parallel for all 20 lanes)

  2-level adder: sum01 = row[0] + row[1]; col_sum = sum01 + row[2]
  ★ Registered output for timing (1 extra cycle)

Resources: ~320 LUT, ~200 FF
```

#### `comparator_tree.sv`

```
Chức năng:    MaxPool 5×5: 25 inputs → 1 maximum per lane, 5-stage pipeline.
              Used only in PE_MP5 mode (SPPF L9).

Ports:
  input  logic signed [7:0]   window [25][LANES],  // 5×5 window × 20 lanes
  output logic signed [7:0]   max_out [LANES],     // 1 max per lane
  output logic                max_valid

Implementation:
  5-stage tree comparator:
    Stage 1: 25 → 13 (pairwise max, 1 passthrough)
    Stage 2: 13 → 7
    Stage 3: 7 → 4
    Stage 4: 4 → 2
    Stage 5: 2 → 1

  ★ Signed INT8 comparison: max = (a >= b) ? a : b
    (Important: signed, NOT unsigned!)

  Latency: 5 cycles
  Active only in PE_MP5 mode; otherwise gated for power saving.

Resources: ~500 LUT, ~320 FF
```

### 16.6. POST-PROCESSING MODULES

#### `ppu.sv` (×4 per subcluster — 1 per PE column)

```
Chức năng:    ★ Post-Processing Unit: Bias + Requantization + Activation + Clamp.
              Converts INT32 accumulator → INT8 output.
              4 PPU instances run in PARALLEL (1 per column = 1 per cout).

Ports:
  input  logic                clk, rst_n,
  // Input (from column_reduce)
  input  logic signed [31:0]  psum_in [LANES],     // 20 INT32 values
  input  logic                psum_valid,
  // Quantization parameters (from shadow_reg_file)
  input  logic signed [31:0]  bias_val,            // per-cout bias
  input  logic [31:0]         m_int,               // requant multiplier
  input  logic [7:0]          shift,               // requant shift amount
  input  logic signed [7:0]   zp_out,              // output zero point
  input  act_type_t           activation,           // ReLU, None, etc.
  // Output
  output logic signed [7:0]   act_out [LANES],     // 20 INT8 values
  output logic                act_valid

★ 5-stage pipeline (V4, for 250 MHz):

  Stage 1: BIAS
    biased[l] = psum_in[l] + bias_val
    (INT32 + INT32 → INT32)

  Stage 2: MULTIPLY (★ CRITICAL — INT64)
    product[l] = int64(biased[l]) × int64(m_int)
    (INT64 multiply — maps to DSP fabric or LUT-based multiplier)

  Stage 3: SHIFT + ROUND (★ half-up rounding, NOT floor!)
    round_offset = (1 << (shift - 1))
    requanted[l] = (product[l] + round_offset) >> shift
    (★ Half-up rounding: +0.5 before truncation)
    (This MUST match golden Python: (acc × M + (1<<(sh-1))) >> sh)

  Stage 4: ACTIVATION
    if activation == ACT_RELU:
      activated[l] = max(0, requanted[l])     // ReLU: simple comparator
    elif activation == ACT_NONE:
      activated[l] = requanted[l]              // Identity

  Stage 5: CLAMP + ZP_OUT
    with_zp[l] = activated[l] + zp_out
    act_out[l] = clamp(with_zp[l], -128, 127)  // Saturate to INT8 range

  ★ CRITICAL CORRECTNESS POINTS:
    1. INT64 multiply: biased × M_int MUST be 64-bit to prevent overflow
       (biased can be ~10^9, M_int can be ~10^9 → product ~10^18 > 2^32)
    2. Half-up rounding: NOT floor, NOT round-to-even
       This matches PyTorch CPU quantized inference behavior
    3. ReLU applied BEFORE adding zp_out
    4. Clamp is SIGNED [-128, +127]

Resources per PPU: ~600 LUT, ~500 FF, 0 DSP (LUT-based INT64 mult)
  Or: 4 DSP (if using DSP for INT64 mult) — but typically LUT-based
```

#### `silu_lut.sv` (optional, unused for current model)

```
Chức năng:    256-entry Look-Up Table for SiLU activation.
              NOT USED for current qYOLOv10n (which uses ReLU).
              Kept for generality / future model support.

Ports:
  input  logic signed [7:0]   x_in,
  output logic signed [7:0]   y_out

Implementation:
  ROM[256] initialized with SiLU(x) = x × sigmoid(x) values
  y_out = ROM[x_in + 128]  (offset to unsigned index)

Resources: 1 BRAM (or distributed LUT), ~50 LUT
Status: NOT instantiated in V4 (ReLU path only)
```

---

## 17. CHI TIẾT TỪNG MODULE — AXI & CLOCK

### 17.1. `axi_lite_slave.sv`

```
Chức năng:    AXI-Lite protocol handler for CPU register access.
Interface:    AXI-Lite Slave → csr_register_bank
Resources:    ~1,000 LUT, ~500 FF
```

### 17.2. `axi4_master_mux.sv`

```
Chức năng:    Arbitrate 4 SuperCluster AXI4 masters → 1 DDR3 interface.
              Round-robin or priority-based arbitration.
Interface:    4× AXI4 Slave (from SCs) → 1× AXI4 Master (to DDR3)
Resources:    ~2,000 LUT, ~1,500 FF, 4 BRAM (FIFOs)
```

### 17.3. `axi_interconnect.sv`

```
Chức năng:    AXI routing fabric connecting all masters/slaves.
              Routes AXI-Lite (CPU) and AXI4 (DMA) to correct targets.
Resources:    ~2,000 LUT, ~1,000 FF, 4 BRAM
```

### 17.4. `clk_wiz_250.sv`

```
Chức năng:    ★ V4: MMCM-based clock generator: 200 MHz input → 250 MHz output.
              Uses Xilinx MMCM (Mixed-Mode Clock Manager) primitive.

Ports:
  input  logic    clk_in_200,       // Board oscillator (VC707: 200 MHz)
  output logic    clk_out_250,      // Accelerator clock
  output logic    locked            // PLL lock indicator

Implementation:
  Xilinx MMCME2_ADV primitive:
    VCO = 200 × 5 = 1000 MHz (within VCO range 600-1200 MHz)
    CLKOUT0 = 1000 / 4 = 250 MHz
    Jitter < 100ps → more than adequate for 250 MHz (4ns period)

Resources: 1 MMCM primitive, ~100 LUT
```

### 17.5. `reset_sync.sv`

```
Chức năng:    Asynchronous reset synchronizer for clean reset release.
              2-FF synchronizer to prevent metastability.

Resources: ~20 FF
```

---

## 18. TỔNG HỢP MODULE INVENTORY

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    V4-VC707 MODULE INVENTORY                             ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  PACKAGES (3):                                                           ║
║    accel_pkg.sv, desc_pkg.sv, csr_pkg.sv                                ║
║                                                                          ║
║  TOP LEVEL (1):                                                          ║
║    accel_top.sv                                                          ║
║                                                                          ║
║  INPUT STAGE (4 modules):                                                ║
║    csr_register_bank.sv                                                  ║
║    desc_fetch_engine.sv                                                  ║
║    barrier_manager.sv                                                    ║
║    global_scheduler.sv                                                   ║
║                                                                          ║
║  SUPERCLUSTER LEVEL (4 modules × 4 instances = 16):                     ║
║    supercluster_wrapper.sv  ×4                                           ║
║    local_arbiter_v2.sv      ×4                                           ║
║    tensor_dma_v2.sv         ×4                                           ║
║    tile_ingress_fifo.sv     ×4                                           ║
║                                                                          ║
║  SUBCLUSTER LEVEL (19 module types × 16 instances):                     ║
║    ── Control (3) ──                                                     ║
║    tile_fsm.sv              ×16                                          ║
║    shadow_reg_file.sv       ×16                                          ║
║    compute_sequencer.sv     ×16                                          ║
║    ── Memory (4) ──                                                      ║
║    glb_input_bank_db.sv     ×48  (3 per sub)                            ║
║    glb_weight_bank.sv       ×48  (3 per sub)                            ║
║    glb_output_bank.sv       ×64  (4 per sub)                            ║
║    metadata_ram.sv          ×16                                          ║
║    ── Address Gen (3) ──                                                 ║
║    addr_gen_input.sv        ×16                                          ║
║    addr_gen_weight.sv       ×16                                          ║
║    addr_gen_output.sv       ×16                                          ║
║    ── Data Movement (3) ──                                               ║
║    router_cluster_v2.sv     ×16                                          ║
║    window_gen.sv            ×16                                          ║
║    swizzle_engine.sv        ×16                                          ║
║    ── Compute (4) ──                                                     ║
║    pe_cluster_v4.sv         ×16                                          ║
║    pe_unit.sv               ×192 (12 per cluster)                       ║
║    dsp_pair_int8.sv         ×1,920 (10 per PE)                          ║
║    column_reduce.sv         ×64  (4 per cluster)                        ║
║    comparator_tree.sv       ×16                                          ║
║    ── Post-Processing (2) ──                                             ║
║    ppu.sv                   ×64  (4 per sub)                            ║
║    silu_lut.sv              ×0   (optional, not instantiated)           ║
║                                                                          ║
║  AXI INFRASTRUCTURE (3 modules):                                         ║
║    axi_lite_slave.sv                                                     ║
║    axi4_master_mux.sv                                                    ║
║    axi_interconnect.sv                                                   ║
║                                                                          ║
║  CLOCK/RESET (2 modules):                                                ║
║    clk_wiz_250.sv                                                        ║
║    reset_sync.sv                                                         ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  TOTALS:                                                                 ║
║    Unique module types:    38                                            ║
║    Total instances:        ~2,600+                                       ║
║    DSP48E1 used:           1,920 (68.6% of 2,800)                       ║
║    BRAM36K used:           ~544  (52.8% of 1,030)                       ║
║    LUT6 used:              ~187K (61.5% of 303,600)                     ║
║    FF used:                ~198K (32.6% of 607,200)                     ║
║    Target clock:           250 MHz                                       ║
║    FPS:                    183-192                                        ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 19. PRIMITIVE → MODULE MAPPING

```
Bảng ánh xạ: Mỗi SW primitive chạy trên module HW nào

┌──────┬──────────────────────┬───────────────────────────────────────────────┐
│ ID   │ Primitive (SW)       │ HW Module Path                                │
├──────┼──────────────────────┼───────────────────────────────────────────────┤
│ P0   │ RS_DENSE_3x3         │ pe_cluster_v4 (PE_RS3 mode)                  │
│      │                      │ → pe_unit → dsp_pair_int8                    │
│      │                      │ → column_reduce → ppu                        │
│      │                      │ Router: multicast act + per-col weight       │
├──────┼──────────────────────┼───────────────────────────────────────────────┤
│ P1   │ OS_1x1               │ pe_cluster_v4 (PE_OS1 mode)                  │
│      │                      │ → pe_unit → dsp_pair_int8 (broadcast w)     │
│      │                      │ → column_reduce → ppu                        │
├──────┼──────────────────────┼───────────────────────────────────────────────┤
│ P2   │ DW_3x3               │ pe_cluster_v4 (PE_DW3 mode)                  │
│      │                      │ → 4 cols = 4 KHÁC channels                  │
│      │                      │ → ppu (per-channel M_int/shift)             │
├──────┼──────────────────────┼───────────────────────────────────────────────┤
│ P3   │ MAXPOOL_5x5          │ window_gen (K=5) → comparator_tree          │
│      │                      │ PE cluster BYPASSED, no PPU                  │
├──────┼──────────────────────┼───────────────────────────────────────────────┤
│ P4   │ MOVE                 │ tensor_dma_v2 (GLB ↔ DDR copy)             │
│      │                      │ PE cluster BYPASSED                          │
├──────┼──────────────────────┼───────────────────────────────────────────────┤
│ P5   │ CONCAT               │ router_cluster_v2 (bypass) + swizzle_engine │
│      │                      │ + domain alignment logic in swizzle         │
│      │                      │ PE cluster BYPASSED                          │
├──────┼──────────────────────┼───────────────────────────────────────────────┤
│ P6   │ UPSAMPLE_NEAREST     │ swizzle_engine (SWZ_UPSAMPLE2X mode)       │
│      │                      │ Address remap only, no compute              │
│      │                      │ PE cluster BYPASSED                          │
├──────┼──────────────────────┼───────────────────────────────────────────────┤
│ P7   │ EWISE_ADD            │ swizzle_engine (domain align + add)         │
│      │                      │ Or: dedicated adder in router bypass path   │
│      │                      │ PE cluster BYPASSED                          │
├──────┼──────────────────────┼───────────────────────────────────────────────┤
│ P8   │ DW_7x7_MULTIPASS     │ pe_cluster_v4 (PE_DW7 mode, 3 passes)      │
│      │                      │ + glb_output_bank (PSUM namespace hold)     │
│      │                      │ + tile_fsm multipass loop                   │
├──────┼──────────────────────┼───────────────────────────────────────────────┤
│ P9   │ GEMM_ATTN_BASIC      │ pe_cluster_v4 (PE_GEMM mode, reuse OS1)    │
│      │                      │ Multiple descriptors for Q×K^T, Score×V    │
├──────┼──────────────────────┼───────────────────────────────────────────────┤
│ P10  │ INT8_MATMUL          │ = PE_OS1 mode (matrix as 1×1 conv)          │
│      │                      │ Reshape logic in descriptor generation      │
├──────┼──────────────────────┼───────────────────────────────────────────────┤
│ P11  │ SOFTMAX_APPROX       │ silu_lut.sv (repurposed for exp LUT)       │
│      │                      │ 256-entry LUT for softmax approximation    │
├──────┼──────────────────────┼───────────────────────────────────────────────┤
│ P12  │ REQUANT (PPU)        │ ppu.sv: bias → INT64 mult → shift → clamp  │
│      │                      │ Integrated in every conv/DW output path    │
├──────┼──────────────────────┼───────────────────────────────────────────────┤
│ P13  │ SiLU_LUT             │ silu_lut.sv (NOT USED — model dùng ReLU)   │
├──────┼──────────────────────┼───────────────────────────────────────────────┤
│ P14  │ ReLU / ReLU6         │ ppu.sv Stage 4: max(0, x) comparator       │
│      │                      │ 1 comparator per lane, trivial logic       │
└──────┴──────────────────────┴───────────────────────────────────────────────┘

★ MỌI primitive chạy trên CÙNG 1 subcluster_datapath.
  Chỉ thay đổi descriptor config (pe_mode, params) → cùng HW khác behavior.
```

---

## 20. LAYER → DESCRIPTOR → MODULE EXECUTION FLOW

```
Ví dụ: Layer 2 (QC2f, [32,160,160] → [32,160,160])

  Descriptor 1: PE_OS1 (cv1: 32→64, ReLU)
    tile_fsm → TS_COMPUTE
    compute_sequencer: for h,wblk,cout_group,cin → pe_cluster(PE_OS1)
    → column_reduce → ppu[0..3] (4 cout cùng lúc) → glb_out

  Descriptor 2: PE_RS3 (bottleneck cv1: 16→16, 3×3, ReLU)
    tile_fsm → TS_COMPUTE
    compute_sequencer: for h,wblk,cout_group,cin,kw → pe_cluster(PE_RS3)
    → column_reduce → ppu[0..3] → glb_out

  Descriptor 3: PE_RS3 (bottleneck cv2: 16→16, 3×3, ReLU)
    Cùng flow với descriptor 2

  Descriptor 4: PE_PASS + CONCAT (domain alignment)
    tile_fsm → TS_SWIZZLE
    router_cluster bypass → swizzle_engine(SWZ_CONCAT) → glb_out
    Domain alignment: requant 3 branches to common scale

  Descriptor 5: PE_OS1 (cv2: 64→32, ReLU)
    tile_fsm → TS_COMPUTE
    Cùng flow PE_OS1 → ppu → glb_out

  CÙNG 1 subcluster xử lý 5 descriptors TUẦN TỰ.
  Chuyển mode chỉ cần latch descriptor mới vào shadow_reg_file.
  0 reconfiguration overhead (pipelined descriptor fetch).
```

---

## 21. KẾT LUẬN TỔNG THỂ

```
╔══════════════════════════════════════════════════════════════════════════╗
║                     V4-VC707 FINAL SUMMARY                               ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ★ PHÁT HIỆN THEN CHỐT:                                                 ║
║    LANES=20 hoàn toàn khớp với mọi feature map width của YOLOv10n.     ║
║    → Spatial utilization 100% (tăng từ 71.1%)                           ║
║    → Đây là single biggest optimization.                                ║
║                                                                          ║
║  ★ 4 TRỤC TỐI ƯU:                                                       ║
║    ① LANES=20: 100% spatial util (0% waste!)                            ║
║    ② 16 subclusters (Triple-RUNNING): 12 active (từ 8)                 ║
║    ③ 250 MHz clock (narrower lanes → shorter critical path)            ║
║    ④ Double-buffered GLB (eliminate fill stalls)                        ║
║                                                                          ║
║  ★ KẾT QUẢ:                                                             ║
║    FPS:          183-192 (tăng 72-81% từ V3's 106 FPS)                 ║
║    DSP:          68.6%  (giảm từ 82.3%, TRONG target 60-70%)           ║
║    BRAM:         52.8%  (tương đương V3)                                ║
║    LUT:          61.5%  (giảm từ 66.8%, better timing margin)          ║
║    FF:           32.6%  (tương đương V3)                                ║
║    Accuracy:     Không đổi (cùng compute atoms → bit-exact)            ║
║    Power:        ~14.7W (tương đương V3)                                ║
║    Efficiency:   13.1 FPS/W (tăng 75% từ V3's 7.5 FPS/W)              ║
║                                                                          ║
║  ★ RTL MODULES:                                                          ║
║    38 unique module types, ~2,600+ instances                            ║
║    14/14 primitives mapped → CÙNG phần cứng (descriptor-driven)        ║
║    23/23 layers covered → ~60 descriptors per inference                 ║
║    3 tầng đảm bảo: Toán đúng + RTL đúng + Sequence đúng               ║
║                                                                          ║
║  ★ INFERENCE ĐÚNG NHƯ PHẦN MỀM ✓                                        ║
║  ★ ~190 FPS REAL-TIME TRÊN VC707 ✓                                      ║
║  ★ 65% RESOURCE UTILIZATION ✓                                            ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```
