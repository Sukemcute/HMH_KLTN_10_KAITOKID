# Summary: Phase 1 Layer-by-Layer Verification

This directory contains the definitive bit-accuracy and functional verification records for the qYOLOv10n Backbone and Neck (Layers 0-22).

## 1. Per-Layer Verification Folders (L0 - L22)
Each folder (e.g., `L0_Conv/`) contains:
- **`verification.md`**: A formal report of bit-accuracy against the PyTorch reference using 100 random samples at actual inference spatial resolutions (640x640 flow).
- **`verify_layer_i.py`**: A standalone verification script that replicates the test by extracting real intermediate inputs from the reference model.

### Consolidated Results Table (640x640 Flow)

| Layer | Type | Mean Match % | Max Diff | Status |
| :--- | :--- | :--- | :--- | :--- |
| 0-1, 3, 17 | Standard Conv | **>99.9%** | 1 LSB | PASS |
| 2, 4, 6, 8 | QC2f (Backbone) | **~97.3%** | 4 LSB | PASS |
| 5, 7, 20 | SCDown | **>99.9%** | 2 LSB | PASS |
| 9 | SPPF | **>99.9%** | 1 LSB | PASS |
| 10 | QPSA (Attention) | **~83.6%** | 25 LSB | PASS* |
| 11, 14 | Upsample | **100.0%** | 0 LSB | PASS |
| 12, 15, 18, 21 | QConcat | **100.0%** | 0 LSB | PASS |
| 13, 16, 19 | QC2f (Neck) | **~98.8%** | 9 LSB | PASS |
| 22 | QC2fCIB (Neck) | **>99.9%** | 1 LSB | PASS |

*\* QPSA variance is stable and attributed to 14-stage sequential rounding noise.*

## 2. High-Level Summary Reports
Apart from the individual layer folders, this directory includes:
- **`full_model_flow_640_report.md`**: A comprehensive summary of the "Unified Model Flow" verification, confirming that skip connections and spatial dimensions are perfectly synchronized.
- **`dataset_validation_report.md`**: The final functional sign-off, showing the Mapped Golden Model achieving an **mAP50 of 0.9302** on the full 7,902-image dataset.

## 3. Verification Suite Summary
The following core scripts in the root directory provide the tooling for end-to-end and trace-level verification:

| Script | Scope | Purpose |
| :--- | :--- | :--- |
| **`exhaustive_verify_model_flow.py`** | Full Model | Iterates through all layers, extracting real inputs from a reference pass to verify each Golden block in isolation but with realistic shapes. |
| **`verify_mapped_features.py`** | Full Model | Performs a cumulative bit-accuracy trace. Unlike the flow script, it passes the *mapped* output of one layer as input to the next to measure error propagation. |
| **`test_mapped_model.py`** | End-to-End | Runs a single inference on both models and compares the final detection tensors (boxes/scores) to confirm functional equivalence. |
| **`val_mapped_model.py`** | Dataset | Integrates the Mapped Golden Model into the Ultralytics pipeline to validate accuracy (mAP) across the entire dataset. |
