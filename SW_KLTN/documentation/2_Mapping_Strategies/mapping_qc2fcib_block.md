# Mapping Strategy: QC2fCIB Block

This document describes the strategy for mapping the `QC2fCIB` block, which replaces standard bottlenecks with high-depth `CIB` (Conditional Identity Block) modules.

## 1. Block Overview
The `QC2fCIB` block is an advanced feature fusion module used in the model's neck (e.g., Layer 22). It combines the split-and-merge strategy of C2f with the dense spatial processing of CIB.

### Architecture:
1.  **Expansion (cv1):** 1x1 Pointwise convolution.
2.  **Split:** Input is halved along the channel dimension.
3.  **QCIB (Stacked):** The processed branch goes through one or more `QCIB` modules.
    - **QCIB Internal Sequence:**
        - DW 3x3 -> PW 1x1 -> **DW 7x7** -> PW 1x1 -> DW 3x3.
        - Residual Addition.
4.  **Fusion (Concat):** Merges the original identity branch with the processed branch.
5.  **Compression (cv2):** 1x1 Pointwise convolution.

## 2. Primitive Mapping

| Component | Operation | Primitive | Logic Description |
| :--- | :--- | :--- | :--- |
| **Projections** | 1x1 Conv | **P1** (`OS_1x1`) | Handles initial expansion, internal PWs, and final compression. |
| **Spatial 3x3** | 3x3 DW Conv | **P2** (`DW_3x3`) | Standard depthwise spatial processing. |
| **Spatial 7x7** | 7x7 DW Conv | **P2 (Generic)** | Uses the generic `dw_conv_int` primitive for larger spatial kernels. |
| **Shortcut** | Add | **P7** (`EWISE_ADD`) | Performs residual fusion inside each QCIB. |
| **Merge** | Concat | **P5** (`CONCAT`) | Joins the split feature maps. |

## 3. Mathematical Execution Flow

### The QCIB Chain
The core innovation in this block is the 5-layer convolutional sequence inside each `QCIB`. 
- **DW 7x7 Integration:** This is the largest spatial kernel in the model. In hardware, this is mapped to the same P2 logic unit but with an expanded line-buffer and window-slide mechanism to accommodate the 7x7 receptive field.
- **Requantization Chain:** Each of the 5 internal convolutions has its own Scale/ZP. The Golden Model ensures that zero-points are correctly subtracted at every stage to prevent bias drift.

### Identity Preservation
- Just like the standard `QC2f`, the first branch is bypassed directly to the `Concat` stage.
- The `shortcut` logic inside `QCIB` uses the **Golden Path** (Float intermediate) to ensure bit-accuracy with PyTorch's residual connection.

## 4. Key Constraints for RTL
- **Buffer Depth:** The DW 7x7 primitive requires a minimum of **7 line buffers** to process the spatial window, compared to the 3 required for standard layers.
- **Latency:** Because a single `QC2fCIB` contains 7 convolutions in sequence (cv1 -> 5 in QCIB -> cv2), it has higher latency than standard blocks. The hardware should prioritize throughput by pipelining these stages.

## 5. Verification Status
The QC2fCIB mapping was verified in **Layer 22** with **100.00% bit-exact match**. This confirms that our generic DW primitive and deep sequential logic are perfectly stable.
