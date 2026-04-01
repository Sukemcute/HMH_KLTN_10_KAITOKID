# Project Overview: qYOLOv10n INT8 Hardware Accelerator

This document provides a high-level overview of the graduation project, translated and summarized from the original Vietnamese specifications.

## 1. Project Goal
The objective is to design and implement a hardware accelerator for the **quantized YOLOv10n (INT8)** model. The accelerator handles the Backbone and Neck sections (Layers 0–22), offloading the most computationally intensive parts of the object detection pipeline from the CPU.

## 2. System Architecture
The system is partitioned between a CPU (Host) and the Accelerator (Hardware):

### CPU Responsibilities (Software)
- **Pre-processing:** Image loading, Letterbox resizing, and normalization to `[0.0, 1.0]`.
- **Input Quantization:** Converting `float32` input to `INT8` using specific scales and zero-points.
- **Post-processing:** Receiving `P3, P4, P5` feature maps from hardware, running the Detection Head (float32), and performing Non-Maximum Suppression (NMS).

### Accelerator Responsibilities (Hardware)
- **Backbone & Neck Execution:** Processing Layers 0 through 22.
- **Compute Engine:** Executing 14 specialized primitives (Conv, Depthwise, Pooling, Attention, etc.).
- **Memory Management:** Efficiently handling 4 major skip connections (SKIP-A to SKIP-D) using dedicated GLB (Global Line Buffer) banks.
- **Output:** Producing three multi-scale feature maps in INT8 format.

## 3. Development Phases
The project follows a "V-Model" inspired implementation flow:
- **Phase 0:** Specification Freeze (Mapping, Quantization, Memory Layout).
- **Phase 1:** Golden Python (Reference model for all 14 primitives).
- **Phase 2:** Block-level Integration (C2f, SCDown, SPPF models).
- **Phase 3:** Full Model Simulation (Layer 0–22 Runner).
- **Phase 4:** RTL (Hardware) Implementation and bit-exact verification.

## 4. Key Technical Constraints
- **Data Type:** Signed INT8 for activations, INT8 symmetric for weights.
- **Accumulation:** INT32 internal accumulators to prevent overflow.
- **Requantization:** Fixed-point arithmetic (Scale/Shift) instead of floating point.
- **Activation:** SiLU and ReLU implemented via Look-Up Tables (LUT).
- **Memory Layout:** 3-bank input and 4-bank output system to optimize spatial parallelization.

---
*Reference files: HW_ACCELERATION_IMPL_FLOW.md, MODEL_FORWARD_FLOW.md, PHASE0/03_quant_policy.md*
