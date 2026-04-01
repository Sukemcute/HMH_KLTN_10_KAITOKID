# Nghiên Cứu Tối Ưu Kiến Trúc Phần Cứng YOLOv10n INT8
## Kết Hợp Eyeriss v2 + YOLOv10n Specifics → >100 FPS trên VC707

> **Mục tiêu**: Kiến trúc tối ưu cho inference đúng đắn + >100 FPS
> **Target**: VC707 Board (XC7VX485T)
> **Nguyên tắc**: Đúng trước, nhanh sau. Mỗi quyết định phải có lý do cụ thể.

---

## 1. PHÂN TÍCH HIỆN TRẠNG — VẤN ĐỀ CỦA KIẾN TRÚC CŨ

### 1.1. Resource Budget VC707 (XC7VX485T)

```
┌───────────────────┬──────────┬──────────────────────────────────┐
│ Resource          │ Available│ Ghi chú                          │
├───────────────────┼──────────┼──────────────────────────────────┤
│ DSP48E1           │ 2,800    │ Bottleneck chính                 │
│ BRAM36K           │ 1,030    │ Đủ cho GLB + weight buffer       │
│ LUT6              │ 303,600  │ Tight nếu thêm NoC phức tạp     │
│ FF                │ 607,200  │ Dư dả                            │
│ DDR3 bandwidth    │ 12.8 GB/s│ VC707 onboard DDR3 SODIMM        │
│ Clock target      │ 200 MHz  │ Conservative cho Virtex-7        │
└───────────────────┴──────────┴──────────────────────────────────┘
```

### 1.2. Yêu cầu tính toán YOLOv10n

```
Tổng MACs backbone+neck (L0-L22): 3,077 MMACs = 3.08 GMACs
Để đạt 100 FPS: T_hw < 10 ms
Required throughput: 3,077M / 10ms = 307.7 GOPS
```

### 1.3. Vấn đề kiến trúc V2-lite cũ

```
V2-lite cũ:
  12 subclusters × 12 PEs × 32 lanes × 2 MAC/DSP = 2,304 MACs/cycle peak
  @ 200 MHz: 2,304 × 200M = 460.8 GOPS peak

  Nhưng utilization thực tế:
  ┌────────────────────────────────────────────────────────────────────┐
  │ Vấn đề                      │ Loss    │ Lý do                     │
  ├────────────────────────────────────────────────────────────────────┤
  │ 4 PE columns tính CÙNG data │ 75%!    │ Không multicast, 3 cols   │
  │                              │         │ redundant                 │
  │ Spatial util (W=20→32 pad)  │ 37.5%   │ LANES=32, W_out=20       │
  │ Fill/drain gap               │ 10%     │ Dual-RUNNING giảm nhưng  │
  │                              │         │ không hết                 │
  │ DW conv 1 channel/time      │ high    │ 4 cols wasted cho DW     │
  └────────────────────────────────────────────────────────────────────┘

  Effective utilization ≈ 25-50% → 115-230 GOPS effective
  → 100 FPS KHÔNG ĐẢM BẢO nếu dưới 30%
```

**Kết luận**: Kiến trúc cũ có vấn đề cốt lõi: **4 PE columns làm cùng 1 việc**.

---

## 2. Ý TƯỞNG TỪ EYERISS v2 — ÁP DỤNG CHO THIẾT KẾ

### 2.1. Bài học chính từ Eyeriss v2

| Concept Eyeriss v2 | Ý nghĩa | Áp dụng? |
|---|---|---|
| **Multicast activation** | 1 GLB read → phát cho N PEs | ✅ PHẢI có |
| **4 PE cols = 4 khác cout** | Mỗi column tính output channel khác | ✅ PHẢI có |
| **Psum forward chain** | PE chain accumulate cin qua columns | ✅ Nên có |
| **Flexible mapping (DW)** | 4 cols = 4 khác channels cho DW | ✅ PHẢI có |
| **Hierarchical Mesh NoC** | Mesh giữa clusters | ❌ Quá phức tạp cho VC707 |
| **All-to-All intra-cluster** | Bất kỳ PE nào nhận bất kỳ data | ❌ Chi phí cao |
| **CSC sparsity** | Bỏ qua zero values | ❌ Model dense INT8 |

### 2.2. Kiến trúc mới: "Eyeriss-lite RS+"

Lấy 3 ý tưởng quan trọng nhất, bỏ phần phức tạp:

```
KIẾN TRÚC MỚI — 1 SUBCLUSTER:
═══════════════════════════════════════════════════════════════════

  GLB Input Banks (3)                GLB Weight Banks (3)
       │                                   │
       ▼                                   ▼
  ┌─────────────────────────────────────────────────┐
  │              ROUTER CLUSTER v2                   │
  │                                                  │
  │  RIN: 1 bank read → MULTICAST → 3 PE rows       │  ← Eyeriss idea #1
  │       (cùng activation cho 3 rows)               │
  │                                                  │
  │  RWT: 3 bank reads → 3 PE rows                  │
  │       Mỗi row nhận weight cho KHÁC kh            │
  │       Mỗi COLUMN nhận weight cho KHÁC cout       │  ← Eyeriss idea #2
  └──────────┬──────────────────────┬───────────────┘
             │                      │
             ▼                      ▼
  ┌──────────────────────────────────────────────────┐
  │              PE CLUSTER v2 (3×4×32)               │
  │                                                   │
  │  Row 0: PE[0,0]  PE[0,1]  PE[0,2]  PE[0,3]      │
  │         cout_0   cout_1   cout_2   cout_3        │  ← 4 KHÁC cout!
  │         ↓ kh=0   ↓ kh=0   ↓ kh=0   ↓ kh=0      │
  │  Row 1: PE[1,0]  PE[1,1]  PE[1,2]  PE[1,3]      │
  │         cout_0   cout_1   cout_2   cout_3        │
  │         ↓ kh=1   ↓ kh=1   ↓ kh=1   ↓ kh=1      │
  │  Row 2: PE[2,0]  PE[2,1]  PE[2,2]  PE[2,3]      │
  │         cout_0   cout_1   cout_2   cout_3        │
  │         ↓ kh=2   ↓ kh=2   ↓ kh=2   ↓ kh=2      │
  │                                                   │
  │  Column Reduce: sum 3 rows per column             │
  │  col_0→cout_0  col_1→cout_1  col_2→cout_2  ...  │
  └──────────┬───────┬───────┬───────┬───────────────┘
             │       │       │       │
             ▼       ▼       ▼       ▼
  ┌──────────────────────────────────────────────────┐
  │              4× PPU (song song 4 cout)            │
  │  PPU_0(cout_0)  PPU_1(cout_1)  PPU_2(cout_3) ..│  ← 4× throughput!
  └──────────────────────────────────────────────────┘
             │       │       │       │
             ▼       ▼       ▼       ▼
  ┌──────────────────────────────────────────────────┐
  │         GLB Output Banks (4) = 4 columns          │
  │  bank_0←cout_0  bank_1←cout_1  bank_2←cout_2 ..│
  └──────────────────────────────────────────────────┘

  DW MODE: 4 columns = 4 KHÁC channels (thay vì 4 khác cout)   ← Eyeriss idea #3
```

### 2.3. Thay đổi cụ thể trong RTL

**pe_cluster.sv — Mỗi column nhận KHÁC weight:**
```systemverilog
// CŨ (v1): tất cả columns nhận CÙNG weight
.w_in(wgt_data[r])   // broadcast cho ALL columns

// MỚI (v2): mỗi column nhận weight cho cout riêng
.w_in(wgt_data[r][c])  // wgt_data[row][col][LANES]
// → 4 columns × 3 rows × 32 lanes = 384 MACs thực sự KHÁC NHAU
```

**router_cluster.sv — Thêm per-column weight routing:**
```systemverilog
// CŨ: 3 weight banks → 3 PE rows (broadcast to columns)
// MỚI: 3 weight banks → 3 PE rows × 4 columns
//       Weight addr khác nhau cho mỗi column (khác cout)
output logic signed [7:0] pe_wgt [PE_ROWS][PE_COLS][LANES];  // thêm PE_COLS dim
```

---

## 3. TÍNH TOÁN FPS VỚI KIẾN TRÚC MỚI

### 3.1. Effective MACs/cycle

```
Kiến trúc cũ (v1):
  12 PEs active × 32 lanes × 2 MAC/DSP = 384 MACs/sub
  Nhưng 4 cols redundant → effective = 384/4 × 1 = 96 unique MACs/sub
  12 subclusters × 8 active = 768 effective MACs/cycle    ← LOW

Kiến trúc mới (v2, Eyeriss-inspired):
  12 PEs active × 32 lanes × 2 MAC/DSP = 384 MACs/sub
  4 cols = 4 KHÁC cout → effective = 384 unique MACs/sub   ← 4× IMPROVEMENT
  8 active subclusters × 384 = 3,072 effective MACs/cycle  ← HIGH
```

### 3.2. Per-tier compute time (Kiến trúc mới @ 200 MHz)

```
Active MACs = 3,072 / cycle (thực sự unique, không redundant)

┌──────────────┬───────┬────────┬──────────────────────────────┐
│ Tier         │ MACs  │ S_util │ T_compute                    │
├──────────────┼───────┼────────┼──────────────────────────────┤
│ 320 (L0)     │  44 M │ 100%   │ 44M/(3072×1.0×200M) = 0.07ms│
│ 160 (L1-L2)  │ 288 M │ 100%   │ 0.47 ms                      │
│ 80  (L3-L4,  │ 675 M │ 83.3%  │ 1.32 ms                      │
│     L15-L17) │       │        │                               │
│ 40  (L5-L6,  │ 740 M │ 62.5%  │ 1.93 ms                      │
│     L12-19)  │       │        │                               │
│ 20  (L7-L10, │1,330 M│ 62.5%  │ 3.46 ms                      │
│     L20-L22) │       │        │                               │
├──────────────┼───────┼────────┼──────────────────────────────┤
│ TỔNG         │3,077M │  —     │ 7.25 ms (pure compute)       │
└──────────────┴───────┴────────┴──────────────────────────────┘

Temporal overhead: ~25% (cải thiện từ 27% nhờ less stall)
T_hw = 7.25 / 0.75 = 9.67 ms → 103 FPS ✓
```

### 3.3. Resource estimate (Kiến trúc mới cho VC707)

```
Thay đổi so với kiến trúc cũ:

┌───────────────────┬──────────┬──────────┬──────────┬──────────────────────┐
│ Resource          │ Available│ Cũ       │ Mới      │ Thay đổi             │
├───────────────────┼──────────┼──────────┼──────────┼──────────────────────┤
│ DSP48E1           │ 2,800    │ 2,304    │ 2,304    │ Không đổi (cùng PEs)│
│ BRAM36K           │ 1,030    │ ~520     │ ~560     │ +40 (weight buffers) │
│ LUT6              │ 303,600  │ ~210K    │ ~225K    │ +15K (per-col route) │
│ FF                │ 607,200  │ ~240K    │ ~255K    │ +15K (pipeline regs) │
└───────────────────┴──────────┴──────────┴──────────┴──────────────────────┘

Cost thêm: ~15K LUT + 40 BRAM cho per-column weight routing
Benefit: 4× effective utilization
→ ROI cực cao
```

---

## 4. NHỮNG GÌ CẦN BỎ / ĐƠN GIẢN HÓA

### 4.1. Bỏ: SiLU LUT infrastructure

```
LÝ DO: Model QAT dùng ReLU, KHÔNG dùng SiLU.
  - silu_lut.sv (31 dòng) → GIỮA cho tính tổng quát nhưng KHÔNG cần test
  - PPU SiLU path → KHÔNG cần verify
  - Tiết kiệm: 256×8 = 2 BRAM + ~200 LUT cho LUT read logic

HÀNH ĐỘNG: Giữ module nhưng không instantiate.
  PPU chỉ cần: ReLU (1 comparator) + Clamp (1 comparator)
```

### 4.2. Bỏ: QPSA/GEMM engine (Layer 10) — Defer

```
LÝ DO:
  - Chỉ 83.6% accuracy (thấp nhất)
  - 14 sequential primitives — phức tạp cực cao
  - Chỉ 1 layer (L10) trên 23 layers
  - 360M MACs / 3,077M total = 11.7% compute

HÀNH ĐỘNG: Software fallback cho L10
  - Accelerator skip L10: output L9 → DDR → CPU tính L10 → DDR → input L11
  - CPU (ARM trên VC707): ~5ms cho QPSA tại 20×20 (nhỏ)
  - Tổng impact: +5ms → T_hw = 9.67 + 5 = 14.67ms → 68 FPS (chưa tối ưu)

  Hoặc: Simple GEMM on PE (dùng PE_OS1 mode cho matrix multiply)
  - Q×K^T và Score×V = 2× GEMM [400×64] × [64×400] và [400×400] × [400×128]
  - Dùng PE_OS1 mode: mỗi element = Σ(cin) a×b
  - ~360M MACs / 3,072 MACs/cycle / 200M = 0.58ms → chấp nhận được
  - Softmax approx: 256-entry LUT (reuse silu_lut module!)

KHUYẾN NGHỊ: Implement QPSA bằng PE_OS1 + softmax LUT
  → Không cần module mới, chỉ cần descriptor sequence đúng
```

### 4.3. Đơn giản hóa: Giảm subclusters

```
PHÂN TÍCH:
  Kiến trúc mới 4× effective → CẦN ÍT subclusters hơn

  Kiến trúc cũ: 12 subs × 96 effective MACs = 1,152 → thiếu
  Kiến trúc mới: 8 subs × 384 effective MACs = 3,072 → ĐỦ

  8 subclusters (thay vì 12):
  ┌───────────────────┬──────────┬──────────────────────────┐
  │ Resource          │ 12 subs  │ 8 subs                   │
  ├───────────────────┼──────────┼──────────────────────────┤
  │ DSP48E1           │ 2,304    │ 1,536 (55% ← dư!)       │
  │ BRAM36K           │ ~560     │ ~400 (39%)               │
  │ LUT6              │ ~225K    │ ~165K (54%)              │
  └───────────────────┴──────────┴──────────────────────────┘

  Hoặc giữ 12 subs nhưng LANES=32, chấp nhận 82% DSP.
  12 subs cho headroom performance khi utilization < 100%.
```

### 4.4. Đơn giản hóa: SuperCluster hierarchy

```
PHÂN TÍCH:
  V2 có 4 SuperClusters × 3 Subclusters = 12 total
  Mỗi SC có: local_arbiter + tensor_dma + FIFO

  VẤN ĐỀ: 4 SC × tensor_dma = 4 AXI masters → complex arbitration

  ĐƠN GIẢN HÓA: 2 SuperClusters × 4 Subclusters = 8 total
  - 2 SC → 2 AXI masters → simpler DDR arbitration
  - 4 subs/SC → Dual-RUNNING = 2 compute + 1 fill + 1 drain
  - Hoặc: 1 "Mega-Cluster" với 8 subs + 1 DMA (đơn giản nhất)

  KHUYẾN NGHỊ cho VC707:
    2 SC × 4 subs = 8 subs, 1,536 DSPs, Dual-RUNNING
    → Đơn giản hơn, dùng ít resource, vẫn đạt >100 FPS
```

---

## 5. KIẾN TRÚC TỐI ƯU ĐỀ XUẤT — "V3-VC707"

### 5.1. Thông số chốt

```
┌───────────────────────────┬─────────────────────────────────────┐
│ Parameter                 │ V3-VC707                            │
├───────────────────────────┼─────────────────────────────────────┤
│ LANES                     │ 32                                  │
│ PE_ROWS                   │ 3 (kh parallelism)                 │
│ PE_COLS                   │ 4 (cout parallelism) ← MỖI COL    │
│                           │ KHÁC cout (Eyeriss-inspired)       │
│ MACs per subcluster       │ 384 (ALL unique, không redundant)  │
│ SUPER_CLUSTERS            │ 2                                   │
│ SUBS_PER_SC               │ 4                                   │
│ ACTIVE_PER_SC             │ 2 (Dual-RUNNING)                   │
│ Total subclusters         │ 8                                   │
│ Total active MACs         │ 4 × 384 = 1,536 /cycle             │
│                           │ (nhưng 100% useful, vs 25% trước) │
│ Effective throughput      │ 1,536 × 200M = 307.2 GOPS          │
│ DSP48E1 usage             │ 8 × 192 = 1,536 (55% of 2,800)    │
│ BRAM36K usage             │ 8 × 42 + 80 = 416 (40%)           │
│ LUT6 usage                │ ~170K (56%)                        │
│ Clock target              │ 200 MHz                             │
│ Activation                │ ReLU (trivial: max(0,x))           │
│ QPSA                      │ PE_OS1 mode + softmax LUT          │
└───────────────────────────┴─────────────────────────────────────┘
```

### 5.2. FPS Calculation

```
Effective GOPS = 1,536 MACs/cycle × 200 MHz = 307.2 GOPS

Pure compute: 3,077M MACs / 307.2 GOPS = 10.02 ms
  Nhưng 100% utilization vì mỗi MAC là unique computation

Actual utilization factors:
  Spatial util (weighted average): ~80%
  Temporal util: ~78% (better than V2 nhờ ít SC → ít coordination)
  Overall: ~62%

T_hw = 3,077M / (307.2G × 0.62) = 16.15 ms → 62 FPS ← CHƯA ĐỦ

Hmm, 62 FPS < 100 FPS. Cần thêm active subclusters.

ĐIỀU CHỈNH: 3 SC × 4 subs = 12 subs, 6 active
  Effective: 6 × 384 × 200M = 460.8 GOPS
  T_hw = 3,077M / (460.8G × 0.62) = 10.77 ms → 93 FPS ← GẦN

  Tăng clock lên 220 MHz:
  T_hw = 3,077M / (460.8G × 220/200 × 0.62) = 9.79 ms → 102 FPS ✓

HOẶC: 4 SC × 3 subs = 12 subs (config ban đầu), 8 active
  Effective: 8 × 384 × 200M = 614.4 GOPS
  T_hw = 3,077M / (614.4G × 0.62) = 8.08 ms → 124 FPS ✓✓
  DSP: 12 × 192 = 2,304 (82%) ← VẪN VỪA

→ GIỮ 12 subclusters (4 SC × 3 sub) + per-col weight routing
```

### 5.3. Kiến trúc cuối cùng xác nhận

```
╔══════════════════════════════════════════════════════════════════════╗
║              KIẾN TRÚC V3-VC707 — FINAL                             ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  4 SuperClusters × 3 Subclusters = 12 total                        ║
║  Dual-RUNNING per SC: 8 active subclusters                         ║
║                                                                      ║
║  Mỗi Subcluster:                                                    ║
║    PE Cluster: 3 rows × 4 cols × 32 lanes                          ║
║    ★ Mỗi column = KHÁC cout (Eyeriss multicast)                    ║
║    ★ 384 MACs ALL UNIQUE per sub                                    ║
║    PPU: 4× parallel (1 per column)                                  ║
║    GLB: 3 input + 3 weight + 4 output banks                        ║
║    Activation: ReLU only (max(0,x))                                 ║
║                                                                      ║
║  Peak: 8 × 384 × 200M = 614.4 GOPS                                ║
║  @ 62% util: 380.9 effective GOPS                                   ║
║  T_hw = 3,077M / 380.9G = 8.08 ms → 124 FPS ✓                     ║
║                                                                      ║
║  DSP: 2,304 / 2,800 = 82%                                          ║
║  BRAM: ~520 / 1,030 = 50%                                          ║
║  LUT: ~225K / 303.6K = 74%                                         ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 6. SO SÁNH KIẾN TRÚC CŨ vs MỚI

| Aspect | V2-lite (cũ) | V3-VC707 (mới) |
|--------|-------------|-----------------|
| Subclusters | 12 | 12 (giữ nguyên) |
| PE columns | 4 (redundant) | 4 (mỗi col khác cout) |
| Effective MACs/sub | 96 | **384 (4×)** |
| DSP usage | 2,304 (82%) | 2,304 (82%) — không đổi |
| Thêm LUT | — | +15K cho weight routing |
| Thêm BRAM | — | +40 cho weight buffers |
| Effective GOPS | ~115-230 | **380.9** |
| T_hw | ~10-20 ms | **8.08 ms** |
| FPS | 50-100 (không chắc) | **124 FPS** (chắc chắn) |
| SiLU | Cần LUT 256 | Bỏ (ReLU only) |
| QPSA | Cần GEMM engine | PE_OS1 + softmax LUT |
| Complexity | 4 SC × 3 sub | 4 SC × 3 sub (giữ) |

**Chi phí tối ưu**: +15K LUT + 40 BRAM
**Lợi ích**: **4× effective utilization → 124 FPS chắc chắn**

---

## 7. THAY ĐỔI CẦN THIẾT TRONG RTL

### 7.1. pe_cluster.sv — Mỗi column nhận khác weight

```diff
- // CŨ: ALL columns same weight
- input  logic signed [7:0]  wgt_data [PE_ROWS][LANES],
+ // MỚI: PER-COLUMN weight
+ input  logic signed [7:0]  wgt_data [PE_ROWS][PE_COLS][LANES],

  generate
    for (r = 0; r < PE_ROWS; r++) begin : gen_row
      for (c = 0; c < PE_COLS; c++) begin : gen_col
        pe_unit u_pe (
          .x_in (act_taps[r]),        // activation SHARED (multicast)
-         .w_in (wgt_data[r]),        // weight SAME for all cols
+         .w_in (wgt_data[r][c]),     // weight DIFFERENT per col
        );
      end
    end
  endgenerate
```

### 7.2. router_cluster.sv — Per-column weight selection

```diff
- // CŨ: 1 weight per row
- output logic signed [7:0] pe_wgt [3][LANES],
+ // MỚI: 1 weight per row per column (4 different cout)
+ output logic signed [7:0] pe_wgt [3][4][LANES],

  // Weight routing: mỗi column đọc weight cho cout khác nhau
  // Column 0: weight[kh][kw][cin][cout_base+0]
  // Column 1: weight[kh][kw][cin][cout_base+1]
  // Column 2: weight[kh][kw][cin][cout_base+2]
  // Column 3: weight[kh][kw][cin][cout_base+3]
```

### 7.3. compute_sequencer.sv — Loop 4 cout cùng lúc

```diff
  // CŨ: 1 cout per iteration
- for cout = 0 to Cout-1:

  // MỚI: 4 cout per iteration (4 columns parallel)
+ for cout_group = 0 to (Cout/4)-1:
+   cout_base = cout_group * 4
+   Column 0 → cout_base+0
+   Column 1 → cout_base+1
+   Column 2 → cout_base+2
+   Column 3 → cout_base+3
  → 4× fewer iterations → 4× faster
```

### 7.4. ppu.sv — 4 instances song song

```
// CŨ: 1 PPU xử lý 1 column → sequential
// MỚI: 4 PPU instances, mỗi cái xử lý 1 column
generate for (c = 0; c < PE_COLS; c++) begin : gen_ppu
  ppu u_ppu_col (
    .psum_in (pe_psum_out[c]),
    .bias_val(bias_mem[cout_base + c]),
    .m_int   (m_int_mem[cout_base + c]),
    .shift   (shift_mem[cout_base + c]),
    .act_out (ppu_out[c])
  );
end endgenerate
```

### 7.5. DW Conv mode — 4 cols = 4 channels

```
// Khi mode == PE_DW3 hoặc PE_DW7:
//   Column 0 → channel c_base+0
//   Column 1 → channel c_base+1
//   Column 2 → channel c_base+2
//   Column 3 → channel c_base+3
//
// Mỗi column nhận:
//   Activation: input[h][w][c_base+col] (KHÁC channel per col)
//   Weight: W[c_base+col][kh][kw] (per-channel weight per col)
//
// → 4× throughput cho depthwise (quan trọng cho SCDown, QC2fCIB)
```

---

## 8. PIPELINE END-TO-END

```
Frame N:
  ┌────────────────────┐
  │ CPU Preprocess      │  ~2.5 ms (letterbox, quantize)
  │ (x86 host via PCIe)│
  └────────┬───────────┘
           │ DMA write X_int8 to DDR
           ▼
  ┌────────────────────────────────────────┐
  │ HW Accelerator L0-L22                  │  8.08 ms
  │ 12 subclusters, 8 active, 200 MHz      │
  │ 4 SC × Dual-RUNNING                   │
  │ Per-column cout routing (Eyeriss v2)   │
  └────────┬───────────────────────────────┘
           │ DMA read P3/P4/P5 from DDR
           ▼
  ┌────────────────────┐
  │ CPU Postprocess     │  ~3.5 ms (dequant + Detect head + NMS)
  └────────────────────┘

Pipelined throughput = 1 / max(2.5, 8.08, 3.5) = 1/8.08 = 124 FPS ✓
Latency = 2.5 + 8.08 + 3.5 = 14.08 ms
```

---

## 9. CẬP NHẬT CHIẾN LƯỢC RTL_BUILD_STRATEGY_FINAL

### Thay đổi cần apply:

```
Stage 2: KHÔNG đổi (compute atoms giữ nguyên)
Stage 3: KHÔNG đổi (primitive engines giữ nguyên, dùng cho verification)
Stage 4: KHÔNG đổi (memory + addr_gen giữ nguyên)
          THÊM: weight bank cần hỗ trợ 4 read ports (1 per column)
          hoặc 4× BRAM width, hoặc time-multiplex 4 reads
Stage 5: SỬA QUAN TRỌNG:
  - pe_cluster.sv: per-column weight input
  - router_cluster.sv: per-column weight routing
  - compute_sequencer.sv: 4-cout-parallel loop
  - subcluster_datapath.sv: 4× PPU instances
Stage 6-8: Logic giữ nguyên (cùng descriptor sequence)
```

### Bỏ / Đơn giản hóa:

```
BỎ:  SiLU LUT verification (model dùng ReLU)
BỎ:  Separate QPSA engine (dùng PE_OS1 + softmax LUT)
BỎ:  silu_lut instance trong subcluster (giữ module, không dùng)
ĐƠN GIẢN: PPU chỉ cần ReLU path (1 comparator per lane)
ĐƠN GIẢN: 2 SC (nếu 8 subs đủ) hoặc giữ 4 SC × 3 sub
```

---

## 10. KẾT LUẬN (Phần tối ưu kiến trúc)

```
Kiến trúc V3-VC707 = V2-lite + Eyeriss v2 per-column routing

Chi phí:   +15K LUT, +40 BRAM (trivial trên VC707)
Lợi ích:   4× effective PE utilization
Kết quả:   124 FPS @ 200 MHz, 82% DSP usage
Đúng đắn:  Cùng compute atoms → cùng bit-exact results
Đơn giản:  Bỏ SiLU, bỏ QPSA engine riêng, ReLU only

★ Thay đổi DUY NHẤT cần thiết: mỗi PE column nhận KHÁC weight.
  Tất cả phần còn lại (dsp_pair, pe_unit, PPU, GLB, addr_gen) giữ nguyên.
```

---
---

# ════════════════════════════════════════════════════════════════
# PHẦN II: KIẾN TRÚC IP CHI TIẾT — INPUT / PROCESSING / OUTPUT
# ════════════════════════════════════════════════════════════════

## 11. TỔNG QUAN IP — 3 TẦNG

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        YOLOV10N_ACCEL_TOP                                │
│                        (AXI-Lite Slave + AXI4 Master)                   │
│                                                                         │
│  ┌───────────┐    ┌──────────────────────────────┐    ┌──────────────┐ │
│  │  INPUT     │    │       PROCESSING              │    │   OUTPUT      │ │
│  │  STAGE     │───►│       STAGE                   │───►│   STAGE       │ │
│  │            │    │                                │    │              │ │
│  │ •CSR MMIO  │    │ •4 SuperClusters              │    │ •DMA write   │ │
│  │ •Desc Fetch│    │  ×3 Subclusters each           │    │  P3/P4/P5   │ │
│  │ •DMA read  │    │ •Global Scheduler              │    │ •IRQ assert  │ │
│  │ •Barrier   │    │ •Compute Datapath              │    │ •Status CSR  │ │
│  └───────────┘    └──────────────────────────────┘    └──────────────┘ │
│                                                                         │
│  AXI-Lite ◄──── CPU (VC707 MicroBlaze/PCIe Host)                      │
│  AXI4    ◄────► DDR3 (VC707 onboard 1GB SODIMM)                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 12. INPUT STAGE — Chi tiết modules

### 12.1. Module: `csr_register_bank`

```
Chức năng: CPU-accessible Control/Status Registers via AXI-Lite
Interface: AXI-Lite Slave (32-bit address, 32-bit data)

Registers:
  CSR_CTRL          (0x000): start, soft_reset, irq_clear         [R/W]
  CSR_STATUS        (0x004): busy, done, error                    [R]
  CSR_NET_DESC_LO   (0x010): Net descriptor base address [31:0]   [R/W]
  CSR_NET_DESC_HI   (0x014): Net descriptor base address [63:32]  [R/W]
  CSR_LAYER_START   (0x018): First layer to execute (0-22)        [R/W]
  CSR_LAYER_END     (0x01C): Last layer to execute (0-22)         [R/W]
  CSR_PERF_CYCLES   (0x020): Total clock cycles counter           [R]
  CSR_PERF_STALLS   (0x024): Stall cycle counter                  [R]
  CSR_IRQ_MASK      (0x028): Interrupt enable mask                [R/W]

Luồng hoạt động:
  1. CPU ghi NET_DESC address + LAYER_START/END
  2. CPU ghi CSR_CTRL.start = 1
  3. Accelerator chạy inference
  4. Khi xong: CSR_STATUS.done = 1, assert IRQ
  5. CPU đọc CSR_STATUS → clear IRQ
```

### 12.2. Module: `desc_fetch_engine`

```
Chức năng: Đọc descriptors từ DDR3 qua AXI4, parse thành structs
Interface: AXI4 Master (read only) → DDR3

Descriptor hierarchy (3 cấp):
  NET_DESC (64 bytes, 1 per inference):
    magic, version, num_layers
    weight_arena_base, act0_arena_base, act1_arena_base

  LAYER_DESC (per layer, 23 total for L0-L22):
    template_id (pe_mode), cin/cout, hin/win/hout/wout
    kh/kw, stride, padding, tile sizes
    num_cin_pass, num_k_pass
    router_profile_id, post_profile_id

  TILE_DESC (N per layer, for spatial/channel tiling):
    tile_id, layer_id, sc_mask
    h_out0, wblk0, cin0, cout0
    src_in_off, src_w_off, src_skip_off, dst_off
    tile_flags (first_tile, last_tile, hold_skip, need_swizzle, barrier)

FSM: DF_IDLE → DF_FETCH_NET → DF_PARSE_NET → DF_FETCH_LAYER →
     DF_PARSE_LAYER → DF_FETCH_TILE → DF_DISPATCH_TILE → DF_NEXT_TILE

Output: tile_desc_valid + tile_desc + layer_desc → global_scheduler
```

### 12.3. Module: `barrier_manager`

```
Chức năng: Đồng bộ skip connections (4 barrier points)
Interface: signal/wait handshake

4 Barriers trong YOLOv10n:
  barrier_L12: wait(L11_done AND F6_ready)  → release L12 (QConcat)
  barrier_L15: wait(L14_done AND F4_ready)  → release L15 (QConcat)
  barrier_L18: wait(L17_done AND F13_ready) → release L18 (QConcat)
  barrier_L21: wait(L20_done AND F8_ready)  → release L21 (QConcat)

Skip tensor sizes (phải HOLD trong GLB cho đến khi consume):
  F4  (L4→L15): INT8 [1,64,80,80]   = 409.6 KB, 11 layers hold
  F6  (L6→L12): INT8 [1,128,40,40]  = 204.8 KB, 6 layers hold
  F8  (L8→L21): INT8 [1,256,20,20]  = 102.4 KB, 13 layers hold (longest!)
  F13 (L13→L18): INT8 [1,128,40,40] = 204.8 KB, 5 layers hold
  Total: ~921.6 KB simultaneous

Implementation: 32-bit scoreboard register
  Producer: signal_valid + barrier_id → set bit
  Consumer: wait_valid + barrier_id → grant when bit set
```

### 12.4. Module: `global_scheduler`

```
Chức năng: Phân phối tiles cho 4 SuperClusters
Interface: tile input (from desc_fetch) → sc_tile_valid[4] output

Logic:
  Nhận tile_desc từ desc_fetch_engine
  Đọc sc_mask[3:0] → xác định SC nào xử lý tile này
  Dispatch tile cho SC tương ứng via handshake (valid/accept)
  Track: layer_complete, inference_complete
```

---

## 13. PROCESSING STAGE — Chi tiết kiến trúc tính toán

### 13.1. Tổng quan hệ thống xử lý

```
PROCESSING STAGE:
  4 SuperClusters × 3 Subclusters = 12 total
  Dual-RUNNING per SC: 2 compute + 1 fill/drain = 8 active simultaneous
  Peak: 8 × 384 = 3,072 unique MACs/cycle @ 200 MHz

  ┌──────────────────────────────────────────────────────────────────┐
  │                    PROCESSING STAGE                               │
  │                                                                   │
  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐│
  │  │SuperCluster 0│  │SuperCluster 1│  │SuperCluster 2│  │SC 3    ││
  │  │ 3 subs      │  │ 3 subs      │  │ 3 subs      │  │3 subs  ││
  │  │ +arbiter    │  │ +arbiter    │  │ +arbiter    │  │+arbiter││
  │  │ +tensor_dma │  │ +tensor_dma │  │ +tensor_dma │  │+t_dma  ││
  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────┬───┘│
  │         │                │                │               │     │
  │         └────────────────┴────────────────┴───────────────┘     │
  │                          │ AXI4 Master (shared)                  │
  │                          ▼                                       │
  │                     DDR3 Controller                               │
  └──────────────────────────────────────────────────────────────────┘
```

### 13.2. Module: `supercluster_wrapper` (×4 instances)

```
Chức năng: Chứa 3 subclusters + arbiter + DMA, quản lý tile execution

Sub-modules:
  ├── local_arbiter        ← Dual-RUNNING scheduler (2 compute + 1 fill/drain)
  ├── tensor_dma           ← AXI4 master, DDR3 ↔ GLB transfers
  ├── tile_ingress_fifo    ← Decouple global_scheduler từ subcluster speed
  └── subcluster[0..2]     ← 3 compute units (mô tả chi tiết bên dưới)

local_arbiter:
  Quản lý role rotation cho 3 subclusters:
    Phase 0: Sub-0=RUNNING, Sub-1=RUNNING, Sub-2=FILLING
    Phase 1: Sub-0=DRAINING, Sub-1=RUNNING, Sub-2=RUNNING
    Phase 2: Sub-0=FILLING, Sub-1=DRAINING, Sub-2=RUNNING
    (rotate khi sub finish tile)
  Output: sub_role[3], ext_port_grant

tensor_dma:
  AXI4 Master: burst read/write DDR3 ↔ GLB
  Max burst: 16 beats × 32 bytes = 512 bytes
  Shared bởi 3 subclusters qua arbiter grant
```

### 13.3. Module: `subcluster_datapath` (×12 instances) ★ CORE MODULE

```
Chức năng: 1 PHẦN CỨNG CỐ ĐỊNH xử lý MỌI primitive qua descriptor config.
           Đây là trung tâm tính toán của toàn bộ accelerator.

Sub-modules (21 instances bên trong):
  ┌─────────────────────────────────────────────────────────────────────┐
  │                     SUBCLUSTER DATAPATH                             │
  │                                                                     │
  │  ═══ CONTROL (2 modules) ═══                                       │
  │  ├── tile_fsm              Phase-level FSM (PREFILL→COMPUTE→PPU→   │
  │  │                         SWIZZLE→DONE). Đọc descriptor, điều     │
  │  │                         khiển toàn bộ datapath phase by phase.  │
  │  │                                                                  │
  │  ├── shadow_reg_file       Latch descriptor fields → stable config │
  │  │                         cho datapath (mode, cin, cout, stride,  │
  │  │                         padding, activation, quant params).     │
  │  │                                                                  │
  │  ├── compute_sequencer     Cycle-level FSM (h,w,c,kw iteration).   │
  │  │                         Drives addr_gen, PE enable, PPU trigger.│
  │  │                         tile_fsm = KHI NÀO, sequencer = CÁI GÌ │
  │  │                                                                  │
  │  ═══ MEMORY (10 modules) ═══                                       │
  │  ├── glb_input_bank ×3     Input activation SRAM, 32 subbanks/bank│
  │  │                         Banking: bank_id = h mod 3              │
  │  │                         Read: LANES (32) bytes/cycle            │
  │  │                                                                  │
  │  ├── glb_weight_bank ×3    Weight SRAM + 8-deep staging FIFO      │
  │  │                         ★ V3: 4 read ports per bank (1/column) │
  │  │                         Hoặc: 4× wider BRAM + lane demux       │
  │  │                                                                  │
  │  ├── glb_output_bank ×4    Dual namespace PSUM(INT32)/ACT(INT8)   │
  │  │                         1 bank per PE column per output row     │
  │  │                                                                  │
  │  ├── metadata_ram          Slot validity + ring buffer pointers    │
  │  │                         Quản lý multi-pass PSUM slots           │
  │  │                                                                  │
  │  ═══ ADDRESS GENERATION (3 modules) ═══                             │
  │  ├── addr_gen_input        (h,w,c) → (bank_id, sram_addr, is_pad) │
  │  │                         bank_id = h mod 3                       │
  │  │                         Padding → output zp_x (NOT zero!)      │
  │  │                                                                  │
  │  ├── addr_gen_weight       Mode-dependent: RS3/OS1/DW3/DW7/GEMM   │
  │  │                         ★ V3: addr cho 4 columns KHÁC cout     │
  │  │                                                                  │
  │  ├── addr_gen_output       (h_out,w_out,cout) → (bank_id, addr)   │
  │  │                         bank_id = PE column index (0-3)         │
  │  │                                                                  │
  │  ═══ DATA MOVEMENT (3 modules) ═══                                  │
  │  ├── router_cluster_v2     RIN: multicast activation → 3 PE rows   │
  │  │                         ★ RWT: per-column weight routing        │
  │  │                         (4 khác weight cho 4 khác cout)         │
  │  │                         RPS: 4 PE columns → 4 output banks      │
  │  │                         Bypass: MOVE/CONCAT/UPSAMPLE path       │
  │  │                                                                  │
  │  ├── window_gen            Shift register: K=1,3,5,7 row taps     │
  │  │                         Conv3x3: 3 taps, Conv1x1: 1 tap        │
  │  │                         MaxPool: 5 taps, DW7x7: 7 taps         │
  │  │                                                                  │
  │  ├── swizzle_engine        Layout transform cho next layer:        │
  │  │                         NORMAL (identity), UPSAMPLE_2X,         │
  │  │                         CONCAT (channel offset)                 │
  │  │                                                                  │
  │  ═══ COMPUTE (2 modules) ═══                                        │
  │  ├── pe_cluster_v3         ★ 3 rows × 4 cols × 32 lanes = 384 MACs│
  │  │   ├── pe_unit ×12       Mỗi PE: 16 dsp_pair_int8 → 32 MACs    │
  │  │   │   └── dsp_pair_int8 ×16  2-MAC DSP48E1 packing            │
  │  │   ├── column_reduce     Sum 3 rows → 1 result per column       │
  │  │   └── comparator_tree   MaxPool 5×5: 25 inputs → 1 max/lane   │
  │  │                                                                  │
  │  │   ★ THAY ĐỔI V3 (Eyeriss-inspired):                           │
  │  │     Row 0: PE[0,0](cout_0) PE[0,1](cout_1) PE[0,2](cout_2) ..│
  │  │     Row 1: PE[1,0](cout_0) PE[1,1](cout_1) PE[1,2](cout_2) ..│
  │  │     Row 2: PE[2,0](cout_0) PE[2,1](cout_1) PE[2,2](cout_2) ..│
  │  │     Mỗi column tính KHÁC output channel → 4× throughput        │
  │  │                                                                  │
  │  │   DW MODE: 4 columns = 4 KHÁC input channels                   │
  │  │                                                                  │
  │  ═══ POST-PROCESSING (4 modules) ═══                                │
  │  ├── ppu ×4 (1 per column) Bias + Requant + ReLU + Clamp          │
  │  │                         ★ 4 PPU song song = 4 cout cùng lúc    │
  │  │                                                                  │
  │  │   Pipeline 4-stage:                                              │
  │  │   S1: biased = psum + bias[cout]                (INT32+INT32)  │
  │  │   S2: requanted = (biased×M_int + round) >> sh  (INT64→INT32) │
  │  │       ★ Half-up rounding: + (1<<(sh-1))                       │
  │  │   S3: activated = ReLU(requanted)               (max(0,x))     │
  │  │   S4: output = clamp(activated + zp_out)        (INT8 [-128,127])│
  │  │                                                                  │
  │  └── silu_lut (optional)   256-entry LUT, KHÔNG dùng cho model     │
  │                             hiện tại (ReLU). Giữ cho tính tổng quát.│
  └─────────────────────────────────────────────────────────────────────┘
```

### 13.4. Chi tiết luồng tính toán theo pe_mode

```
★ MỌI mode dùng CÙNG phần cứng, chỉ KHÁC descriptor config:

─────────────────────────────────────────────────────────────────
PE_RS3 (Conv 3×3) — L0, L1, L3, L17, QC2f bottleneck
─────────────────────────────────────────────────────────────────
  Config: kh=3, kw=3, stride=1|2, act=ReLU
  Datapath:
    GLB_IN[bank = h mod 3] → router(multicast) → 3 PE rows
    GLB_WT[bank = kh] → router(per-col weight) → 4 PE cols × 3 rows
    PE accumulates: kw=0,1,2 × cin=0..C-1
    Column reduce: sum 3 rows (kh dimension)
    4× PPU: bias + requant + ReLU → 4 output channels cùng lúc
    GLB_OUT[4 banks] ← 4 INT8 results

  Iterations: for h_out, for wblk, for cout_group(step 4):
    feed 3×cin kw cycles → drain → 4× PPU → write 4 cout

─────────────────────────────────────────────────────────────────
PE_OS1 (Conv 1×1) — QC2f cv1/cv2, SCDown cv1, SPPF cv1/cv2
─────────────────────────────────────────────────────────────────
  Config: kh=1, kw=1, stride=1, pad=0, act=ReLU
  Datapath:
    GLB_IN → router → 1 PE row (broadcast weight to all lanes)
    GLB_WT → router(per-col weight) → 4 cols × 1 row
    PE accumulates: cin=0..C-1 (1 cycle per cin)
    Column reduce: passthrough (only 1 row)
    4× PPU → 4 cout cùng lúc
    GLB_OUT ← 4 results

  Iterations: for h, for wblk, for cout_group(step 4):
    feed cin cycles → drain → 4× PPU → write

─────────────────────────────────────────────────────────────────
PE_DW3 (Depthwise 3×3) — SCDown cv2
─────────────────────────────────────────────────────────────────
  Config: kh=3, kw=3, stride=1|2, per-channel weight/bias
  Datapath:
    ★ MỖI COLUMN = KHÁC INPUT CHANNEL (không phải khác cout)
    Col 0 → channel c_base+0
    Col 1 → channel c_base+1
    Col 2 → channel c_base+2
    Col 3 → channel c_base+3
    PE accumulates: kw=0,1,2 per channel
    Column reduce: passthrough (independent channels)
    4× PPU (per-channel params) → 4 channels cùng lúc

─────────────────────────────────────────────────────────────────
PE_DW7 (Depthwise 7×7 multipass) — QC2fCIB L22
─────────────────────────────────────────────────────────────────
  Config: kh=7, kw=7, stride=1, pad=3
  3 passes (fit 3 PE rows):
    Pass 1: kh=0,1,2 → 7 kw × 3 rows → PSUM_INT32 → GLB_OUT(PSUM)
    Pass 2: kh=3,4,5 → 7 kw × 3 rows → PSUM + prev → GLB_OUT(PSUM)
    Pass 3: kh=6     → 7 kw × 1 row  → PSUM + prev → PPU → INT8

  tile_fsm tự loop: num_k_pass=3
  PSUM namespace trong GLB_OUT giữ intermediate results

─────────────────────────────────────────────────────────────────
PE_MP5 (MaxPool 5×5) — SPPF L9
─────────────────────────────────────────────────────────────────
  Config: K=5, stride=1, pad=2, NO PPU
  Datapath:
    GLB_IN → window_gen(K=5) → 25 values per lane
    → comparator_tree (25→1 max, 5-stage pipeline)
    → GLB_OUT(ACT) directly (scale/zp unchanged)

  PE cluster BYPASSED, không dùng MAC

─────────────────────────────────────────────────────────────────
PE_PASS (Upsample/Concat/Move) — L11, L12, L14, L15, L18, L21
─────────────────────────────────────────────────────────────────
  PE cluster + PPU BYPASSED

  UPSAMPLE_2X (L11, L14):
    swizzle_engine: dst[2h+dh][2w+dw][c] = src[h][w][c] (dh,dw∈{0,1})
    Address remap only, no compute

  CONCAT (L12, L15, L18, L21):
    router_cluster bypass path
    Domain alignment: requant_to_common nếu scale_A ≠ scale_B
    Interleave channels: [A_channels, B_channels] → output

  MOVE:
    DMA copy GLB → DDR hoặc DDR → GLB
    Cho skip connection HOLD
```

### 13.5. Layer → Descriptor Sequence (CÙNG 1 phần cứng)

```
┌─────────┬──────────┬────────────────────────────────────────────────────┐
│ Layer   │ Block    │ Descriptor Sequence (trên CÙNG subcluster)         │
├─────────┼──────────┼────────────────────────────────────────────────────┤
│ L0      │ Conv     │ desc_1: PE_RS3(s=2,Cin=3,Cout=16,ReLU)           │
│ L1      │ Conv     │ desc_1: PE_RS3(s=2,Cin=16,Cout=32,ReLU)          │
│ L2      │ QC2f     │ desc_1: PE_OS1(cv1, 32→64, ReLU)                 │
│         │          │ desc_2: PE_RS3(bn_cv1, 16→16, ReLU)              │
│         │          │ desc_3: PE_RS3(bn_cv2, 16→16, ReLU)              │
│         │          │ desc_4: PE_PASS+CONCAT(domain align)              │
│         │          │ desc_5: PE_OS1(cv2, 64→32, ReLU)                 │
│ L3      │ Conv     │ desc_1: PE_RS3(s=2,Cin=32,Cout=64,ReLU)          │
│ L4      │ QC2f     │ (cùng pattern L2, Cin/Cout khác)                 │
│ L5      │ SCDown   │ desc_1: PE_OS1(cv1, 64→128, ReLU)                │
│         │          │ desc_2: PE_DW3(cv2, C=128, s=2, ReLU)            │
│ L6      │ QC2f     │ (5 descriptors)                                   │
│ L7      │ SCDown   │ (2 descriptors)                                   │
│ L8      │ QC2f     │ (5 descriptors)                                   │
│ L9      │ SPPF     │ desc_1: PE_OS1(cv1, 256→128, ReLU)               │
│         │          │ desc_2: PE_MP5(C=128, K=5, no PPU) [pool1]       │
│         │          │ desc_3: PE_MP5(C=128, K=5, no PPU) [pool2]       │
│         │          │ desc_4: PE_MP5(C=128, K=5, no PPU) [pool3]       │
│         │          │ desc_5: PE_PASS+CONCAT(4-way, 128×4=512)         │
│         │          │ desc_6: PE_OS1(cv2, 512→256, ReLU)               │
│ L10     │ QPSA     │ desc_1-14: PE_OS1(projections) + PE_OS1(GEMM)    │
│         │          │ + softmax_LUT + PE_DW3(pos_enc) + PE_OS1(FFN)    │
│ L11     │ Upsample │ desc_1: PE_PASS+UPSAMPLE_2X                      │
│ L12     │ QConcat  │ desc_1: PE_PASS+CONCAT(F11+F6, barrier)          │
│ L13-L16 │ QC2f     │ (5 descriptors mỗi layer)                        │
│ L14     │ Upsample │ desc_1: PE_PASS+UPSAMPLE_2X                      │
│ L15     │ QConcat  │ desc_1: PE_PASS+CONCAT(F14+F4, barrier)          │
│ L17     │ Conv     │ desc_1: PE_RS3(s=2)                               │
│ L18     │ QConcat  │ desc_1: PE_PASS+CONCAT(F17+F13, barrier)         │
│ L19     │ QC2f     │ (5 descriptors)                                   │
│ L20     │ SCDown   │ (2 descriptors)                                   │
│ L21     │ QConcat  │ desc_1: PE_PASS+CONCAT(F20+F8, barrier)          │
│ L22     │ QC2fCIB  │ desc_1: PE_OS1(cv1, 384→256)                     │
│         │          │ desc_2: PE_DW3(3x3), desc_3: PE_OS1(1x1)         │
│         │          │ desc_4: PE_DW7(7x7, 3-pass)                      │
│         │          │ desc_5: PE_OS1(1x1), desc_6: PE_DW3(3x3)         │
│         │          │ desc_7: EWISE_ADD(residual)                       │
│         │          │ desc_8: PE_PASS+CONCAT                            │
│         │          │ desc_9: PE_OS1(cv2)                               │
└─────────┴──────────┴────────────────────────────────────────────────────┘

Tổng: ~60 descriptors cho L0-L22. CÙNG 1 phần cứng xử lý tất cả.
```

---

## 14. OUTPUT STAGE — Chi tiết modules

### 14.1. 3 Output Tensors (P3/P4/P5)

```
Accelerator produces 3 multi-scale feature maps:
  P3 = L16 output: INT8 [1, 64, 80, 80]    (stride 8)
  P4 = L19 output: INT8 [1, 128, 40, 40]   (stride 16)
  P5 = L22 output: INT8 [1, 256, 20, 20]   (stride 32)

Mỗi tensor kèm: scale (float32), zero_point (int8)
Ghi vào DDR3 tại: dst_off trong tile_desc

Tổng output: 64×80×80 + 128×40×40 + 256×20×20
           = 409,600 + 204,800 + 102,400 = 716,800 bytes ≈ 700 KB
```

### 14.2. Module: `tensor_dma` (write path)

```
Chức năng: Ghi output từ GLB → DDR3 qua AXI4 burst writes

Khi tile_fsm ở SWIZZLE_STORE state và flag_need_spill=1:
  Đọc từ GLB_output_bank (ACT namespace)
  Pack thành AXI4 burst (max 16 beats × 32 bytes = 512 bytes)
  Write to DDR3 at dst_off address

Bandwidth: 32 bytes/cycle × 200 MHz = 6.4 GB/s (half duplex)
Output 700KB at 6.4 GB/s = 0.11 ms (negligible)
```

### 14.3. Completion & IRQ

```
Khi inference hoàn thành (L22 last tile done):
  1. global_scheduler: inference_complete = 1
  2. controller_system: CSR_STATUS.done = 1
  3. IRQ asserted (nếu CSR_IRQ_MASK enabled)
  4. CPU nhận IRQ → đọc P3/P4/P5 từ DDR3
  5. CPU chạy Qv10Detect head (float32) → decode bboxes → NMS
```

---

## 15. TỔNG HỢP TẤT CẢ MODULES

```
Module Inventory — V3-VC707:

TOP LEVEL:
  accel_top.sv                  ← AXI-Lite + AXI4 wrapper, 4 SCs + controller

INPUT STAGE (5 modules):
  csr_register_bank.sv          ← CPU control interface
  desc_fetch_engine.sv          ← DDR3 → descriptor structs
  barrier_manager.sv            ← 4-point skip sync
  global_scheduler.sv           ← Tile dispatch → 4 SCs

PROCESSING STAGE (per SuperCluster = ×4):
  supercluster_wrapper.sv       ← 3 subs + arbiter + DMA
  local_arbiter.sv              ← Dual-RUNNING scheduler
  tensor_dma.sv                 ← AXI4 master DDR↔GLB

PROCESSING STAGE (per Subcluster = ×12):
  subcluster_datapath.sv        ← TOP integration module
  tile_fsm.sv                   ← Phase-level FSM
  shadow_reg_file.sv            ← Config latch
  compute_sequencer.sv          ← Cycle-level iteration
  glb_input_bank.sv ×3          ← Input SRAM
  glb_weight_bank.sv ×3         ← Weight SRAM (★ 4-port read for V3)
  glb_output_bank.sv ×4         ← Output SRAM (PSUM/ACT)
  metadata_ram.sv               ← Slot management
  addr_gen_input.sv             ← Input address
  addr_gen_weight.sv            ← Weight address (★ per-column)
  addr_gen_output.sv            ← Output address
  router_cluster_v2.sv          ← ★ Per-column weight routing
  window_gen.sv                 ← Sliding window K=1,3,5,7
  swizzle_engine.sv             ← Layout transform
  pe_cluster_v3.sv              ← ★ 3×4×32 (per-col weight)
    pe_unit.sv ×12              ← 32-lane PE
      dsp_pair_int8.sv ×16     ← 2-MAC DSP48E1
    column_reduce.sv            ← Sum 3 rows
    comparator_tree.sv          ← MaxPool 25→1
  ppu.sv ×4                     ← ★ 4 parallel PPUs
  silu_lut.sv (optional)        ← SiLU LUT (unused in ReLU model)

PACKAGES:
  accel_pkg.sv                  ← Types, parameters, enums
  desc_pkg.sv                   ← Descriptor struct types
  csr_pkg.sv                    ← CSR address map

TOTAL: ~35 unique modules, ~250 instances across 12 subclusters
```

---
---

# ════════════════════════════════════════════════════════════════
# PHẦN III: CHECKLIST XÂY DỰNG — STEP BY STEP ĐẾN INFERENCE ĐÚNG
# ════════════════════════════════════════════════════════════════

## 16. PHASE MAP — Từ atoms đến inference

```
PHASE A: Compute Atoms (dsp_pair, pe_unit, column_reduce, comparator_tree, PPU)
    ↓ ALL PASS (0 errors, bit-exact)
PHASE B: Memory & Routing (GLB banks, addr_gen, window_gen, router_v2, swizzle)
    ↓ ALL PASS
PHASE C: Subcluster Integration (wire up + tile_fsm + compute_sequencer)
    ↓ PASS 7 pe_mode tests trên CÙNG 1 HW
PHASE D: System Integration (4 SC + scheduler + DMA + barrier)
    ↓ PASS multi-tile multi-SC
PHASE E: Layer Verification (L0-L22 từng layer vs golden)
    ↓ PASS per-layer accuracy ≥ baseline
PHASE F: Full Inference (L0→L22 chain, P3/P4/P5 output)
    ↓ mAP50 ≈ 0.93
PHASE G: FPGA Deployment (VC707 synthesis + board test)
```

---

## PHASE A: COMPUTE ATOMS

```
☐ A.1  dsp_pair_int8
       - RTL: 4-stage pipeline, unsigned-offset DSP48E1 packing
       - Test: exhaustive 65,536 products + 9-cycle accumulation + random 10K
       - Pass: 0 errors
       - File: PHASE_10/stage_2/01_dsp_pair/

☐ A.2  pe_unit
       - RTL: 32 lanes = 16 dsp_pairs, mode-dependent weight routing
       - Test: PE_RS3 (per-lane), PE_OS1 (broadcast), PE_DW3 (per-channel)
       - Pass: 0 errors, correct pipeline timing (4-cycle latency)
       - File: PHASE_10/stage_2/02_pe_unit/

☐ A.3  column_reduce
       - RTL: col_psum[c][l] = Σ_{r=0..2} pe_psum[r][c][l]
       - Test: known values + overflow boundary + random 100
       - Pass: 0 errors
       - File: PHASE_10/stage_2/03_column_reduce/

☐ A.4  comparator_tree
       - RTL: 25 inputs → 1 max per lane, 5-stage pipeline
       - Test: signed boundaries (-128/127), gradient, per-lane, random 200
       - Pass: 0 errors
       - File: PHASE_10/stage_2/04_comparator_tree/

☐ A.5  ppu (★ CRITICAL)
       - RTL: 4-stage: bias → requant(half-up) → ReLU → clamp+zp
       - Test: half-up rounding (MUST differ from floor), ReLU, ZP, ewise_add
       - ★ Rounding: (acc×M + (1<<(sh-1))) >> sh
       - ★ INT64 multiply: mult = int64(biased) × int64(M_int)
       - ★ Activation = ReLU: y = max(0, x) (NOT SiLU)
       - Pass: 0 errors, 1000 random stress
       - File: PHASE_10/stage_2/06_ppu/
```

---

## PHASE B: MEMORY & DATA MOVEMENT

```
☐ B.1  glb_input_bank ×3
       - RTL: 32 subbanks, lane-masked write, registered read
       - Test: write/read patterns, lane mask, multi-address random
       - File: PHASE_10/stage_4/01_memory/

☐ B.2  glb_weight_bank ×3
       - RTL: SRAM + 8-deep FIFO
       - ★ V3: cần 4 read ports (1 per PE column cho khác cout)
       - Option A: 4× BRAM width (128 bytes/read → lane demux)
       - Option B: 4 independent read ports (4× BRAM duplicate)
       - Option C: Time-multiplex 4 reads (1 read/cycle × 4 cycles)
       - File: PHASE_10/stage_4/01_memory/

☐ B.3  glb_output_bank ×4
       - RTL: dual namespace PSUM(INT32) / ACT(INT8)
       - Test: write PSUM → read PSUM, write ACT → read ACT, switch namespace
       - File: PHASE_10/stage_4/01_memory/

☐ B.4  addr_gen_input
       - RTL: (h,w,c) → bank_id(h mod 3) + addr + padding detection
       - ★ Padding: output zp_x (KHÔNG phải 0)
       - Test: L0 sweep (640×640), banking pattern, padding boundary
       - File: PHASE_10/stage_4/02_addr_gen/

☐ B.5  addr_gen_weight
       - RTL: mode-dependent addressing
       - ★ V3: phải generate 4 KHÁC addresses cho 4 columns
       - RS3: addr[col] = (cout_base+col) × Cin × Kw + cin × Kw + kw
       - OS1: addr[col] = (cout_base+col) × Cin + cin
       - DW:  addr[col] = (ch_base+col) × Kw + kw
       - File: PHASE_10/stage_4/02_addr_gen/

☐ B.6  addr_gen_output
       - RTL: bank_id = PE column index (0-3)
       - Test: address calculation for multi-column output
       - File: PHASE_10/stage_4/02_addr_gen/

☐ B.7  window_gen
       - RTL: shift register K_MAX=7, configurable K=1,3,5,7
       - Test: K=1 (conv1x1), K=3 (conv3x3), K=5 (maxpool), K=7 (DW7x7)
       - Test: flush + refill, data integrity 32 lanes
       - File: PHASE_10/stage_4/03_data_movement/

☐ B.8  router_cluster_v2
       - RTL: RIN (multicast act) + RWT (★ per-column weight) + RPS + bypass
       - ★ V3 change: pe_wgt output = [PE_ROWS][PE_COLS][LANES]
       - Test: straight routing, cross routing, multicast, bypass
       - File: PHASE_10/stage_4/03_data_movement/ (cần UPDATE)

☐ B.9  swizzle_engine
       - RTL: UPSAMPLE_2X + CONCAT offset + NORMAL identity
       - Test: L11 upsample pattern, L12 concat channel offset
       - File: PHASE_10/stage_4/03_data_movement/

☐ B.10 Integration test: addr_gen + GLB banks
       - Pre-fill banks → addr_gen sweep → verify data matches
       - Verify padding returns zp_x
       - File: PHASE_10/stage_4/04_integration_test/
```

---

## PHASE C: SUBCLUSTER INTEGRATION (★ MOST CRITICAL)

```
☐ C.1  pe_cluster_v3 — Thay đổi wgt_data interface
       - Input: wgt_data[PE_ROWS][PE_COLS][LANES] (thay vì [PE_ROWS][LANES])
       - Mỗi column nhận KHÁC weight
       - Test: 4 columns → 4 khác cout → verify 4 kết quả độc lập
       - File: PHASE_10/stage_5/rtl/pe_cluster.sv (CẦN SỬA)

☐ C.2  compute_sequencer — Inner loop controller
       - FSM: SEQ_INIT → SEQ_FEED_PE → SEQ_NEXT_CIN → SEQ_PE_DRAIN →
              SEQ_PPU_RUN → SEQ_WRITE_OUT → SEQ_NEXT_COUT_GROUP →
              SEQ_NEXT_WBLK → SEQ_NEXT_HOUT → SEQ_DONE
       - ★ cout_group loop: step 4 (4 columns parallel)
       - Drives: addr_gen(h,w,c), PE(en,clear), PPU(trigger,cout_idx)
       - File: PHASE_10/stage_5/rtl/compute_sequencer.sv (CẦN SỬA)

☐ C.3  subcluster_datapath — Wire up TẤT CẢ modules
       - 21 module instances, đầy đủ kết nối
       - ★ V3: 4× PPU instances (1 per column)
       - ★ V3: per-column weight từ router → pe_cluster
       - ★ PPU output → glb_output_bank write
       - ★ Pool output → glb_output_bank write
       - File: PHASE_10/stage_5/rtl/subcluster_datapath.sv (CẦN SỬA)

☐ C.4  Test pe_mode PE_RS3 (Conv 3×3)
       - Fill GLB với L0 golden data
       - Config descriptor: PE_RS3, Cin=3, Cout=16, stride=2, ReLU
       - tile_fsm: PREFILL → COMPUTE → PPU → DONE
       - Compare 4 cout outputs vs golden → ≥99.9%
       - CÙNG phần cứng với tất cả test sau

☐ C.5  Test pe_mode PE_OS1 (Conv 1×1)
       - Config: PE_OS1, broadcast weight, 1 cycle/cin
       - Compare vs golden → ≥99%

☐ C.6  Test pe_mode PE_DW3 (DW Conv 3×3)
       - Config: PE_DW3, 4 khác channels per column
       - Per-channel bias/m_int/shift
       - Compare vs golden → ≥99.9%

☐ C.7  Test pe_mode PE_MP5 (MaxPool 5×5)
       - Config: PE_MP5, no PPU, comparator_tree
       - PE cluster bypassed
       - Compare vs golden → 100%

☐ C.8  Test pe_mode PE_PASS + UPSAMPLE
       - Config: PE_PASS, swizzle UPSAMPLE_2X
       - Compute bypassed, address remap only
       - Compare vs golden → 100%

☐ C.9  Test pe_mode PE_PASS + CONCAT
       - Config: PE_PASS, router bypass + domain alignment
       - Compare vs golden → 100%

☐ C.10 Test pe_mode PE_DW7 multipass
       - Config: PE_DW7, num_k_pass=3
       - PSUM namespace hold giữa passes
       - Compare vs golden → ≥99.9%
```

---

## PHASE D: SYSTEM INTEGRATION

```
☐ D.1  supercluster_wrapper (×4)
       - Wire: 3 subclusters + local_arbiter + tensor_dma
       - Test: Dual-RUNNING rotation (2 compute + 1 fill/drain)

☐ D.2  tensor_dma
       - AXI4 burst read/write DDR3 ↔ GLB
       - Test: burst split, address alignment, read/write completion

☐ D.3  local_arbiter
       - Dual-RUNNING scheduler: role rotation 3 subs
       - Test: rotation sequence, ext_port grant

☐ D.4  accel_top
       - Wire: AXI-Lite MMIO + controller_system + 4 SCs
       - Test: CSR write → start → IRQ assert

☐ D.5  Multi-tile test
       - 1 layer = nhiều tiles → distribute qua global_scheduler
       - Verify: tất cả tiles complete → layer output correct
```

---

## PHASE E: LAYER VERIFICATION (so golden Python per-layer)

```
☐ E.1  L0 Conv:     golden L0 → subcluster → compare ≥99.99%
☐ E.2  L1 Conv:     golden L1 → ≥99.96%
☐ E.3  L2 QC2f:     5 descriptors sequence → compare ≥99%
☐ E.4  L3 Conv:     ≥99.98%
☐ E.5  L4 QC2f:     ≥96.52%
☐ E.6  L5 SCDown:   2 descriptors → ≥99.90%
☐ E.7  L6 QC2f:     ≥94.46%
☐ E.8  L7 SCDown:   ≥99.92%
☐ E.9  L8 QC2f:     ≥99.20%
☐ E.10 L9 SPPF:     6 descriptors → ≥99.94%
☐ E.11 L10 QPSA:    14 descriptors → ≥83.52%
☐ E.12 L11 Upsample: 1 descriptor → 100%
☐ E.13 L12 QConcat: 1 descriptor + barrier → 100%
☐ E.14 L13-L22:     (tương tự pattern, target từ golden verification)

Pass criteria cho mỗi layer: match % ≥ software golden baseline
```

---

## PHASE F: FULL INFERENCE

```
☐ F.1  Generate descriptor blob (Python script)
       - NET_DESC + 23 LAYER_DESC + ~60 TILE_DESC → binary

☐ F.2  Generate weight blob (Python script)
       - Pack weights + bias + M_int + shift → DDR image

☐ F.3  E2E simulation
       - Load descriptors + weights + input image → DDR model
       - Start accelerator → wait IRQ
       - Read P3/P4/P5 from DDR

☐ F.4  Compare P3/P4/P5 vs golden oracle
       - P3[1,64,80,80]:  ≥97% bit-exact
       - P4[1,128,40,40]: ≥97% bit-exact
       - P5[1,256,20,20]: ≥97% bit-exact

☐ F.5  Feed P3/P4/P5 vào CPU Qv10Detect head
       - mAP50 ≥ 0.92 (< 1% degradation from golden 0.9302)

☐ F.6  Multi-image validation (100 images)
       - Consistent accuracy across images
```

---

## PHASE G: FPGA DEPLOYMENT (VC707)

```
☐ G.1  Synthesis: accel_top → XC7VX485T
       - Vivado project, target 200 MHz
       - Check: DSP ≤ 2,304 (82%), BRAM ≤ 520 (50%), LUT ≤ 225K (74%)

☐ G.2  Implementation (Place & Route)
       - Timing closure @ 200 MHz
       - Power estimation

☐ G.3  Bitstream generation

☐ G.4  Board test
       - Program VC707
       - CPU driver: load data via AXI-Lite/PCIe
       - Run inference on real hardware
       - Compare output vs golden → FINAL VALIDATION

☐ G.5  Performance measurement
       - Measure actual T_hw → calculate real FPS
       - Target: >100 FPS
```

---

## 17. TÓM TẮT: CON ĐƯỜNG TỪ CODE → 124 FPS INFERENCE ĐÚNG

```
╔══════════════════════════════════════════════════════════════════════╗
║  A. Compute Atoms pass 100% bit-exact                               ║
║     (dsp_pair, pe_unit, column_reduce, comparator_tree, PPU)       ║
║                              ↓                                      ║
║  B. Memory + Router pass (GLB banks, addr_gen, window_gen)          ║
║     ★ router_v2: per-column weight routing (Eyeriss idea)          ║
║                              ↓                                      ║
║  C. 1 Subcluster pass 7 pe_modes trên CÙNG HW                     ║
║     ★ pe_cluster_v3: 4 columns = 4 khác cout = 4× throughput      ║
║     ★ 4× PPU parallel                                              ║
║                              ↓                                      ║
║  D. 4 SuperClusters + DMA + Scheduler hoạt động                    ║
║                              ↓                                      ║
║  E. 23 layers L0-L22 đều PASS golden baseline                      ║
║                              ↓                                      ║
║  F. Full inference: P3/P4/P5 đúng → mAP50 ≈ 0.93                  ║
║                              ↓                                      ║
║  G. VC707 board: 200 MHz, 124 FPS real-time                        ║
║                                                                      ║
║  ★ INFERENCE ĐÚNG ĐẮN NHƯ PHẦN MỀM ✓                              ║
║  ★ >100 FPS TRÊN VC707 ✓                                            ║
╚══════════════════════════════════════════════════════════════════════╝
```
