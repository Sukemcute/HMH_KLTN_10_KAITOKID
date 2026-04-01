# Technical Report: Dtype Trace of Quantized YOLOv10n

This document provides a comprehensive analysis of the data types (`dtype`) utilized across the quantized YOLOv10n architecture. This trace is essential for understanding memory alignment and hardware acceleration compatibility (e.g., for FBGEMM/QNNPACK engines).

## 1. Executive Summary
The quantized YOLOv10n model primarily operates in the **8-bit integer domain**. It uses **`quint8`** (Unsigned 8-bit) for activations to maximize the dynamic range of ReLU-based features and **`qint8`** (Signed 8-bit) for weights. The model transitions back to **`float32`** only at the final stages of the detection head to preserve coordinate precision.

---

## 2. Global Execution Flow
The following table summarizes the high-level transitions of data types throughout the inference lifecycle:

| Stage | Component | Input Dtype | Output Dtype |
| :--- | :--- | :--- | :--- |
| **Entry** | Preprocessing | `float32` | `float32` |
| **Quantization** | `model.quant` (Quantize) | `float32` | **`quint8`** |
| **Feature Map** | Backbone (Layers 0-9) | `quint8` | `quint8` |
| **Neck/Fusion** | Neck (Layers 10-22) | `quint8` | `quint8` |
| **Detection** | Head (Layer 23) - Pre-Dequant | `quint8` | `quint8` |
| **Dequantization** | `model.dequant` (DeQuantize) | `quint8` | **`float32`** |
| **Post-Process** | BBox Decoding / NMS | `float32` | `float32` |

---

## 3. Detailed Component Trace

### 3.1. Standard Convolution Blocks (Conv)
Most layers in the backbone (e.g., Layer 0, 1, 3) are standard `Conv` blocks fused during quantization.
- **`nnq.Conv2d`**: The core operation.
    - **Input**: `quint8`
    - **Weight**: `qint8`
    - **Bias**: `float32` (Quantized internally during accumulation)
    - **Output**: `quint8`
- **`nnq.ReLU`**: Performed in-place or as a separate op on **`quint8`**.

### 3.2. Advanced Blocks (QC2f, QBottleneck, SCDown)
These blocks maintain the integer pipeline using `torch.nn.quantized.FloatFunctional` to handle non-linear operations.
- **`chunk` / `split`**: Tensors are sliced but remain **`quint8`**.
- **`FloatFunctional.cat`**: Concatenates multiple **`quint8`** tensors into a single **`quint8`** tensor.
- **`FloatFunctional.add`**: Used for residual/shortcut connections. It re-quantizes the sum back to **`quint8`**.

### 3.3. Attention Mechanism (QPSA / QAttention)
Layer 10 introduces the position-sensitive attention, which is fully quantized:
- **Matrix Multiplication**: `FloatFunctional.matmul` operates on **`quint8`**.
- **Softmax**: Uses the specialized `torch.ao.nn.quantized.modules.activation.Softmax`.
    - **Input/Output**: **`quint8`**.
- **Scaling**: `FloatFunctional.mul_scalar` maintains **`quint8`** via internal rescaling factors.

### 3.4. Upsampling & Concatenation
- **`nn.Upsample`**: Uses nearest-neighbor interpolation directly on **`quint8`** values.
- **`QConcat`**: Specialized module to ensure concatenated quantized tensors share compatible scales, outputting **`quint8`**.

---

## 4. The Detection Head (Qv10Detect)
The head (Layer 23) serves as the "bridge" back to the floating-point domain.

1.  **Integer Processing**: The initial convolutions (`cv2`, `cv3`, `one2one_cv2`, `one2one_cv3`) are all `nnq.Conv2d` operating on **`quint8`**.
2.  **Transition**: The `self.dequant` module is invoked.
    - It converts the integer feature maps into **`float32`**.
3.  **Floating-Point Logic**: 
    - **DFL (Distribution Focal Loss)**: Performed on dequantized `float32` tensors.
    - **BBox Decoding**: All coordinate math (anchors, strides, etc.) is in **`float32`**.
    - **Final Sigmoid**: Performed on the `float32` class scores.

---

## 5. Summary Table (Sub-Layer Level)

| Layer Type | Sub-Component | Dtype |
| :--- | :--- | :--- |
| **All Quantized** | Activations | `quint8` |
| **All Quantized** | Weights | `qint8` |
| **Conv2d** | `conv` | `quint8` (In/Out) |
| **ReLU** | `act` | `quint8` |
| **C2f** | `cv1`, `cv2`, `m` | `quint8` |
| **PSA** | `attn.sm` (Softmax) | `quint8` |
| **Detect** | `cv2`, `cv3` | `quint8` |
| **Detect** | `dfl` | `float32` |
| **Detect** | `outputs` | `float32` |

---
*Report generated for Ultralytics Quantization Framework.*
