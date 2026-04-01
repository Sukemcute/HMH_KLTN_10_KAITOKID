# Mapping Strategy: SPPF Block

This document describes the strategy for mapping the `SPPF` (Spatial Pyramid Pooling - Fast) block to hardware primitives.

## 1. Block Overview
The `SPPF` block is designed to capture multi-scale context by pooling features at different resolutions and fusing them. It is computationally efficient as it pools the previous pooled result rather than the original input.

### Architecture:
1.  **cv1 (Pointwise):** 1x1 Convolution for initial expansion.
2.  **m (Max Pooling):** Three sequential 5x5 MaxPool layers with stride 1.
3.  **Concat:** Merges cv1 output and all 3 pooled outputs.
4.  **cv2 (Pointwise):** 1x1 Convolution for feature fusion and compression.

## 2. Primitive Mapping

| Stage | Operation | Primitive | Logic Description |
| :--- | :--- | :--- | :--- |
| **Expansion** | 1x1 Conv | **P1** (`OS_1x1`) | Prepares hidden dimension. Uses ReLU. |
| **Pooling 1** | 5x5 MaxPool | **P10** (`MAXPOOL`) | First scale. Stride=1, Pad=2. |
| **Pooling 2** | 5x5 MaxPool | **P10** (`MAXPOOL`) | Second scale (operates on Pool 1). |
| **Pooling 3** | 5x5 MaxPool | **P10** (`MAXPOOL`) | Third scale (operates on Pool 2). |
| **Fusion** | Concat | **P5** (`CONCAT`) | Joins 4 tensors. Scale alignment required. |
| **Compression**| 1x1 Conv | **P1** (`OS_1x1`) | Final result generation. |

## 3. Mathematical Execution Flow

### Sequential Pooling Advantage
Instead of running 5x5, 9x9, and 13x13 pools in parallel (slow), SPPF runs three 5x5 pools in a chain.
- $P_1 = MaxPool(cv1)$
- $P_2 = MaxPool(P_1)$
- $P_3 = MaxPool(P_2)$
The hardware only needs a single P10 unit that can be reused three times.

### Quantization Invariance
Max pooling is a comparison-based operation. It does **not** change the scale or zero-point of the data.
- $Scale_{P1} = Scale_{P2} = Scale_{P3} = Scale_{cv1}$
This allows the subsequent Concat operation to be extremely efficient, as no domain alignment (scaling) is needed between the branches.

## 4. Key Constraints for RTL
- **Data Reuse:** The output of `cv1` must be held in a buffer while pooling occurs, as it is needed later for the final concatenation.
- **Comparison Logic:** The P10 primitive must perform signed integer comparison to correctly handle the `int8` domain.

## 5. Verification Status
The SPPF mapping was verified in **Layer 9** with **100.00% bit-exact match**. This confirms that sequential pooling and multi-way concatenation are correctly handled by the Golden Model.
