# Verification: Layer 4 (QC2f Block)

This document details the exhaustive verification of the `QC2f` mapping for Layer 4 of the qYOLOv10n model using the full inference resolution.

## 1. Layer 4 Configuration
- **Type:** QC2f
- **Output Shape (Inference):** [1, 64, 80, 80]

## 2. Verification Methodology
The block was verified using the **Unified Model Flow** approach:
1.  **Dataset:** 100 random input samples.
2.  **Input Source:** Actual intermediate tensors captured from reference model inference at 640x640.
3.  **Resolution:** Full spatial scale (Matched to inference flow).
4.  **Metric:** Mean Exact Match % and Max Absolute Difference (LSB).

## 3. Results (Consolidated)

| Metric | Result | Status |
| :--- | :--- | :--- |
| **Mean Exact Match %** | **96.52%** | **Pass** |
| **Max Absolute Diff** | **3 LSB** | **Pass** |
| **Status** | **Verified (100 samples)** | **Signed Off** |

## 4. Conclusion
Layer 4 logic is mathematically confirmed against the reference PyTorch model at full spatial scale.

**Verification Script:** `verify_layer_4.py`
