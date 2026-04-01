# Mapping: YOLO QC2f Block to Hardware Primitives

This document describes the strategy for mapping the complex `QC2f` (Quantized C2f) block into hardware primitives.

## 1. Structure of the QC2f Block
The `QC2f` block is a Faster Implementation of the CSP (Cross Stage Partial) Bottleneck. In the quantized model, it consists of:
- **`cv1` (`OS_1x1`):** A 1x1 convolution that expands channels.
- **Split:** The output of `cv1` is split into two halves along the channel dimension.
- **`m` (ModuleList of `QBottleneck`):** A sequence of bottleneck blocks where each block processes the output of the previous one.
- **`fl` (`CONCAT`):** Concatenates the two initial split halves and the outputs of all bottlenecks.
- **`cv2` (`OS_1x1`):** A final 1x1 convolution to project the concatenated features to the output dimension.

## 2. Primitive Decomposition Flow

| Step | Operation | Primitive ID | Description |
| :--- | :--- | :--- | :--- |
| **1** | Expansion | **P1** | `OS_1x1` expands input to `2 * c_` channels. |
| **2** | Split | - | Software/Address-based split into `y[0]` and `y[1]`. |
| **3** | Bottleneck Part 1 | **P0** | `RS_DENSE_3x3` inside `QBottleneck`. |
| **4** | Bottleneck Part 2 | **P0** | `RS_DENSE_3x3` inside `QBottleneck`. |
| **5** | Concatenation | **P5** | `CONCAT` with **Domain Alignment** for all intermediate results. |
| **6** | Projection | **P1** | `OS_1x1` maps back to the output channel size. |

## 3. Implementation: `block_qc2f`
The mapping is implemented in `PHASE1/python_golden/blocks/block_qc2f.py`. It orchestrates the sequence of primitives and handles the complex metadata passing (scales and zero-points) required for the `CONCAT` domain alignment.

## 4. Key Considerations
- **Domain Alignment:** Crucial for the `CONCAT` step. Since different branches (split vs. bottlenecks) have different quantization parameters, they must be aligned to a common domain (specified by the model's `FloatFunctional` parameters) before concatenation.
- **Memory Management:** In hardware, this block requires significant buffering to hold the initial split (`y[0]`, `y[1]`) while the bottlenecks are being computed.

---
**Status:** QC2f mapping logic is implemented and verified via functional dry-run.
