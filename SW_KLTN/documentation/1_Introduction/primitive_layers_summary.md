# Summary of 14 Primitive Layers (qYOLOv10n Accelerator)

This document summarizes the 14 core primitive operations identified in Phase 1 of the project. These primitives are the fundamental building blocks used to map complex YOLOv10n layers into hardware-executable operations.

## 1. Convolution & Depthwise (Core Compute)

| ID | Name | Type | Description |
|---|---|---|---|
| **P0** | **RS_DENSE_3x3** | Conv 3x3 | Regular Strided Dense Convolution. Used for downsampling (stride 2) and main feature extraction. Supports asymmetric padding with zero-point correction. |
| **P1** | **OS_1x1** | Conv 1x1 | Output-Stationary Pointwise Convolution. Used for channel projection, expansion/compression, and inside bottleneck blocks. |
| **P2** | **DW_3x3** | Depthwise 3x3 | Standard Depthwise Convolution. Processes each channel independently. Key component of the SCDown (Spatial-Channel Downsampling) module. |
| **P8** | **DW_7x7_MULTIPASS** | Depthwise 7x7 | Optimized 7x7 Depthwise Conv split into 3 horizontal passes (3 rows + 3 rows + 1 row) to maintain a small line buffer in hardware. |

## 2. Tensor & Spatial Operations

| ID | Name | Type | Description |
|---|---|---|---|
| **P3** | **MAXPOOL_5x5** | Pooling | 5x5 Max Pooling with stride 1 and padding 2. Used in the SPPF block. It is a "pass-through" operation (quantization parameters remain unchanged). |
| **P6** | **UPSAMPLE_NEAREST** | Upsample | Nearest-neighbor upsampling by a factor of 2. Achieved via address remapping in hardware; metadata (scale/zp) is preserved. |
| **P5** | **CONCAT** | Join | Channel-wise concatenation. Includes "Domain Alignment" logic to requantize different input tensors into a common output scale. |
| **P4** | **MOVE** | Buffer Copy | Explicit tensor copy operation. Essential for managing "Skip Connections" where feature maps must be held in memory for many subsequent layers. |
| **P7** | **EWISE_ADD** | Element-wise | Addition of two tensors with domain alignment. Primarily used for residual connections in the model backbone. |

## 3. Attention & Advanced Math

| ID | Name | Type | Description |
|---|---|---|---|
| **P9** | **GEMM_ATTN_BASIC** | Attention | Top-level primitive for Self-Attention (QPSA). Orchestrates the sequence of projections, matrix multiplications, and softmax. |
| **P10**| **INT8_MATMUL** | Matrix Mult | The core engine for Attention. Performs signed INT8 x INT8 multiplication with INT32/INT64 accumulation. |
| **P11**| **SOFTMAX_APPROX** | Activation | A hardware-optimized approximation of the Softmax function, typically implemented using Look-Up Tables (LUT) or piecewise linear functions. |

## 4. Hardware Post-Processing (PPU)

| ID | Name | Type | Description |
|---|---|---|---|
| **P12**| **REQUANT (PPU)** | Requantization | The Post-Processing Unit logic: `(Accumulator * M_int >> shift) + zp_out`. Essential for converting high-precision sums back to INT8. |
| **P13**| **SiLU_LUT** | Activation | Sigmoid Linear Unit activation implemented via a Look-Up Table (LUT) for high efficiency in INT8 space. |
| **P14**| **ReLU / ReLU6** | Activation | Simple clipping logic to `[0, 127]` or `[0, 6]` (in quantized space). Integrated directly into the Requantization path. |

---
**Note:** These primitives are implemented in `PHASE1/python_golden/primitives/` and validated against the specifications in `PHASE0/01_primitive_matrix.md`.