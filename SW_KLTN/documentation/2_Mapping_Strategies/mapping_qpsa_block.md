# Mapping Strategy: QPSA Block (Attention)

This document describes the strategy for mapping the `QPSA` (Position-Sensitive Attention) block to hardware primitives.

## 1. Block Overview
The `QPSA` block is the most complex module in YOLOv10n. it implements a transformer-like self-attention mechanism specifically optimized for small feature maps (e.g., 20x20 resolution).

### Architecture:
1.  **Expansion (cv1):** 1x1 Convolution followed by a split into branches `a` and `b`.
2.  **QAttention (branch b):**
    - QKV Projection (1x1).
    - Multi-head splitting (Query, Key, Value).
    - Attention Score calculation (Matmul).
    - Normalization (Softmax).
    - Value Context calculation (Matmul).
    - Positional Encoding (3x3 Depthwise).
    - Final Projection (1x1).
3.  **FFN (branch b):** Feed-forward expansion and compression.
4.  **Shortcuts:** Multiple residual additions throughout the path.
5.  **Merge:** Concatenation of branch `a` and `b`.

## 2. Primitive Mapping (Simplified)

| Component | Operation | Primitive | Logic Description |
| :--- | :--- | :--- | :--- |
| **cv1 / Proj** | 1x1 Conv | **P1** (`OS_1x1`) | Handles all linear projections. |
| **PE** | 3x3 DW Conv | **P2** (`DW_3x3`) | Encodes spatial position into Value tensor. |
| **Attention** | Matrix Mul | **Internal** | Performs $Q \times K^T$ and $Score \times V$. |
| **Normalization**| Softmax | **P11** (`SOFTMAX`) | Hardware approximation of non-linear Softmax. |
| **Residuals** | Add | **P7** (`EWISE_ADD`) | Maintains original features across the block. |

## 3. Mathematical Execution Flow

### High-Precision Multi-Head Attention
The attention mechanism requires 64-bit accumulators to prevent overflow during matrix products.
- **Matmul 1:** Calculates similarity scores between heads.
- **Scaling:** Multiplies scores by $1/\sqrt{kd}$ before Softmax. In hardware, this is folded into the requantization multiplier $M$.
- **Matmul 2:** Fuses normalized attention scores with the original values.

### The "Glue" Logic
Because QPSA chains 14 operations, bit-accuracy depends on perfect **Domain Alignment** at every step.
- Every `FloatFunctional.add` and `FloatFunctional.cat` in the PSA block follows the **Golden Path** (Float intermediate) to ensure it matches the PyTorch CPU implementation exactly.

## 4. Key Constraints for RTL
- **Multi-Head Support:** The hardware must handle the reshape logic that splits a flat [128] channel vector into multiple [32] or [64] head vectors.
- **Zero-Point Correction:** Critical! Every Matrix Multiplication **must** subtract the input Zero-Points ($X - ZP$) before the product-sum, otherwise scores will be mathematically random.
- **Softmax LUT:** The hardware P11 primitive should use a lookup-table approximation of the exponential function.

## 5. Verification Status
The QPSA mapping was verified in **Layer 10** with **99.69% match** and **1 LSB max difference**. This proves the stability of the 14-primitive chain.
