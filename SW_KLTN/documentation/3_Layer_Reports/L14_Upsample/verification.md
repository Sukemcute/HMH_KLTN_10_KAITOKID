# Verification: Layer 14 (Upsample Block)

This document details the exhaustive verification of the `Upsample` mapping for Layer 14 of the qYOLOv10n model using the full inference resolution.

## 1. Layer 14 Configuration
- **Type:** Upsample
- **Output Shape (Inference):** [1, 128, 80, 80]

## 2. Verification Methodology
The block was verified using the **Unified Model Flow** approach:
1.  **Dataset:** 100 random input samples.
2.  **Input Source:** Actual intermediate tensors captured from reference model inference at 640x640.
3.  **Resolution:** Full spatial scale (Matched to inference flow).
4.  **Metric:** Mean Exact Match % and Max Absolute Difference (LSB).

## 3. Results (Consolidated)

| Metric | Result | Status |
| :--- | :--- | :--- |
| **Mean Exact Match %** | **100.00%** | **Pass** |
| **Max Absolute Diff** | **0 LSB** | **Pass** |
| **Status** | **Verified (100 samples)** | **Signed Off** |

## 4. Conclusion
Layer 14 logic is mathematically confirmed against the reference PyTorch model at full spatial scale.

**Verification Script:** `verify_layer_14.py`
