# PHASE_10 — YOLOv10n INT8 Accelerator RTL Build

> Target: Xilinx VC707 (XC7VX485T)
> Mục tiêu: Primitive PASS → Layer PASS → Inference PASS (bit-exact với Golden Python)

## Cấu trúc tổng thể

```
PHASE_10/
├── stage_0/         ✅ Nghiên cứu & Đặc tả (19 docs, REFERENCE ONLY)
├── stage_1/         ✅ Golden Python & SW Verification (mAP50=0.9302, REFERENCE ONLY)
├── stage_2/         🔧 RTL Compute Atoms (6 modules + TB, CẦN CHẠY VERIFY)
├── stage_5/         ⬜ Subcluster Integration (1 phần cứng chung, config qua descriptor)
├── stage_6/         ⬜ Layer Sequence Verify (chuỗi descriptors trên cùng HW)
├── stage_7/         ⬜ System Integration (4 SC + controller + DMA)
├── stage_8/         ⬜ Full Inference E2E (L0-L22 → P3/P4/P5)
└── stage_9/         ⬜ FPGA Deployment (VC707 board)
```

> Stage 3-4 đã bỏ. Lý do: Stage 2 verify atoms đầy đủ → nhảy thẳng Stage 5 (subcluster).
> conv3x3_engine.sv và các file tương tự trong HW_ACCEL/ là golden test helpers, KHÔNG phải phần cứng cuối.

## Triết lý kiến trúc

```
1 SUBCLUSTER = 1 phần cứng CỐ ĐỊNH
  GLB → Window_Gen → PE_Cluster → PPU → Output
  ↑ Config bởi descriptor (mode, stride, cin, cout, act, ...)

L0:  descriptor{PE_RS3, K=3, s=2, Cin=3, Cout=16, act=ReLU}   → cùng HW
L5:  descriptor{PE_DW3, K=3, s=2, C=128, act=ReLU}             → cùng HW
L9:  descriptor{PE_MP5, K=5, no PPU}                            → cùng HW
L11: descriptor{PE_PASS, swizzle=UPSAMPLE_2X}                   → cùng HW
```

## Quy tắc nền tảng

1. Signed INT8 [-128, 127], ZP_hw = ZP_pytorch - 128
2. Half-up rounding: `(acc*M + (1<<(sh-1))) >> sh`
3. INT64 cho PPU multiply
4. Activation = **ReLU** (model QAT), SiLU LUT giữ cho tính tổng quát
5. Padding fill = zero_point_x (KHÔNG phải 0)

## Hiện tại: Chạy Stage 2

```bash
cd D:/HMH_KLTN/PHASE_10/stage_2/sim
vivado -mode batch -source compile_all.do
# → 6 modules phải ALL PASS trước khi tiếp
```
