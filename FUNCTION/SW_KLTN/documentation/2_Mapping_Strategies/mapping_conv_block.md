# Mapping: YOLO Conv Block to Hardware Primitives

This document describes the strategy for mapping the standard YOLO `Conv` block (Convolution + BatchNorm + Activation) into the primitive layers defined for the hardware accelerator, based on the actual architecture of the quantized YOLOv10n model.

## 1. Structure of the Quantized Conv Block
Based on the model examination, a `Conv` block in the quantized domain typically consists of:
- **`QuantizedConv2d`:** An integer-based convolution with Batch Normalization already fused into the weights and bias. It contains its own `scale` and `zero_point` for output quantization.
- **`act` (Activation):** Usually `nn.ReLU(inplace=True)`, but can also be `nn.Identity()` or `nn.SiLU()`.

Example (Layer 0):
```python
(0): Conv(
  (conv): QuantizedConv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), scale=0.3527, zero_point=62, padding=(1, 1))
  (act): ReLU(inplace=True)
)
```

## 2. Mapping Strategy

A single YOLO `Conv` block is mapped to **exactly one** primitive layer call.

### A. Dense Convolution (Standard)
| Kernel Size | Primitive ID | Primitive Name | Mapping Logic |
| :--- | :--- | :--- | :--- |
| **3 x 3** | **P0** | `RS_DENSE_3x3` | Standard 3x3 convolution with activation. |
| **1 x 1** | **P1** | `OS_1x1` | Pointwise convolution (stride=1, padding=0). |

### B. Depthwise Convolution
In some blocks like `SCDown`, the `Conv` module is configured as a depthwise convolution (`groups == channels`).
| Kernel Size | Primitive ID | Primitive Name | Mapping Logic |
| :--- | :--- | :--- | :--- |
| **3 x 3** | **P2** | `DW_3x3` | Depthwise 3x3 convolution. |
| **7 x 7** | **P8** | `DW_7x7_MULTIPASS` | Large kernel depthwise convolution. |

## 3. Parameter Alignment

To achieve equivalent results, the parameters from the PyTorch `QuantizedConv2d` are mapped as follows:

| PyTorch Parameter | Primitive Argument | Implementation Note |
| :--- | :--- | :--- |
| `weight` | `W_int8` | Extracted from `conv.weight()`. |
| `bias` | `B_int32` | Extracted from `conv.bias()`. |
| `scale` | `scale_y` | Used for the requantization step. |
| `zero_point` | `zp_y` | Used for the requantization step. |
| `stride` | `stride` | Matches `conv.stride`. |
| `act` type | `activation` | "relu", "silu", or "none" (for Identity). |

## 4. Execution Flow within the Primitive
The primitive implementation (e.g., `rs_dense_3x3`) encapsulates the entire fused logic:
1.  **Padding:** Uses input $ZP_x$.
2.  **MAC:** Standard integer multiply-accumulate.
3.  **Correction:** $Acc = Acc_{raw} - (ZP_x \times \sum W) + Bias$.
4.  **Requant:** $Y = (Acc \times M_{int} \gg shift) + ZP_y$.
5.  **Activation:** Applied to the INT8 result before storage.

---
**Conclusion:** The standard YOLO `Conv` block is functionally equivalent to a single call of our Convolution/Depthwise primitives, provided the correct quantization parameters and activation modes are passed.
