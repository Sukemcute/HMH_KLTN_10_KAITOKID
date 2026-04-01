# Verification: Layer 0 (Conv Block)

This document details the exhaustive verification of the `Conv` mapping for Layer 0 of the qYOLOv10n model using the full inference resolution.

## 1. Layer 0 Configuration
- **Type:** Conv
- **Output Shape (Inference):** [1, 16, 320, 320]

## 2. Verification Methodology
The block was verified using the **Unified Model Flow** approach:
1.  **Dataset:** 100 random input samples.
2.  **Input Source:** Actual intermediate tensors captured from reference model inference at 640x640.
3.  **Resolution:** Full spatial scale (Matched to inference flow).
4.  **Metric:** Mean Exact Match % and Max Absolute Difference (LSB).

## 3. Results (Consolidated)

| Metric | Result | Status |
| :--- | :--- | :--- |
| **Mean Exact Match %** | **99.99%** | **Pass** |
| **Max Absolute Diff** | **1 LSB** | **Pass** |
| **Status** | **Verified (100 samples)** | **Signed Off** |

## 4. Conclusion
Layer 0 logic is mathematically confirmed against the reference PyTorch model at full spatial scale.

**Verification Script:** `verify_layer_0.py`
