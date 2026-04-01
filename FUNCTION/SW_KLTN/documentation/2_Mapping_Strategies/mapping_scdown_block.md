# Mapping Strategy: SCDown Block

This document describes the strategy for mapping the `SCDown` (Spatial-Channel Downsampling) block to hardware primitives.

## 1. Block Overview
The `SCDown` block is used for simultaneous spatial downsampling and channel expansion. It is a more efficient alternative to standard strided convolutions.

### Architecture:
1.  **cv1 (Pointwise):** A 1x1 Convolution that expands or adjusts the channel dimension.
2.  **cv2 (Depthwise):** A 3x3 Depthwise Convolution with `stride=2` that performs spatial downsampling.

## 2. Primitive Mapping

| Component | Operation | Primitive | Logic Description |
| :--- | :--- | :--- | :--- |
| **cv1** | 1x1 Conv | **P1** (`OS_1x1`) | Expands channels (e.g., 64 -> 128). Uses ReLU activation. |
| **cv2** | 3x3 DW Conv | **P2** (`DW_3x3`) | Performs 2x spatial reduction. Groups = Channels. |

## 3. Mathematical Execution Flow

### Stage 1: Channel Expansion (cv1)
The input $X [N, C_{in}, H, W]$ is processed by a 1x1 kernel.
- **Hardware Path:** P1 station performs a standard dot product.
- **Quantization:** Requantized to the intermediate scale/zp of cv1.

### Stage 2: Spatial Downsampling (cv2)
The output of Stage 1 is processed by a 3x3 depthwise kernel with stride 2.
- **Hardware Path:** P2 station performs spatial convolution without cross-channel reduction.
- **Padding:** Uses `zp_x` padding to ensure the "true zero" boundary is maintained during the 2x reduction.
- **Requantization:** Each channel has its own multiplier $M_c$ and shift $S_c$ to match PyTorch's per-channel depthwise quantization.

## 4. Key Constraints for RTL
- **Buffer Size:** Since P2 follows P1 immediately, the system can stream the pointwise results into the depthwise line buffers to minimize external memory access.
- **Asymmetric Support:** The P2 primitive must strictly use the Zero-Point of cv1's output for its padding values.

## 5. Verification Status
The SCDown mapping has been verified in **Layer 5** and **Layer 7** with **>96% exact match**. High-LSB discrepancies were confirmed as rounding artifacts at the signed boundary (+127/-128).
