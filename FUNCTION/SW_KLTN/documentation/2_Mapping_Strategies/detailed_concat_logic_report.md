# Technical Report: Deep Dive into Concatenation Mapping

## 1. Executive Summary
This report documents the verification of the `concat` primitive logic used for joining multiple feature maps in the YOLOv10n model (specifically in `QC2f` and Head layers). The implementation has been verified to be **100.00% bit-exact** compared to PyTorch's `FloatFunctional.cat`.

## 2. Implementation Architecture
The `concat` operation is more than a simple memory join; it requires **Domain Alignment** because the input tensors often come from different branches with distinct scales and zero-points.

### Core Components
1. **`concat` (Primitive):** The high-level API that receives a list of tensors and their individual quantization parameters.
2. **`align_and_concat` (Domain Logic):** Orchestrates the normalization process. It identifies the target domain (either provided by the model's observer or calculated via strategy).
3. **`requant_to_common` (Normalization):** The heavy lifter that converts each input tensor to the target scale/ZP *before* memory concatenation.

## 3. The Mapping Strategy
To achieve functional equivalence with PyTorch, the Golden Model follows the "Golden Path" of transformation:

$$X_{target} = \text{quantize}(\text{dequantize}(X_{src}, S_{src}, ZP_{src}), S_{target}, ZP_{target})$$

This ensures that even if one branch has a scale of $0.1$ and another has $0.25$, they are mathematically aligned so that the resulting concatenated tensor has a consistent numeric meaning across all channels.

## 4. Verification Methodology
The verification was performed using real parameters extracted from **Layer 12 (QConcat)** of the YOLOv10n model.

- **Target Scale:** 0.1471...
- **Input Scales:** Mixed (0.25 and 0.10 used for testing)
- **Input Data:** Randomized INT8 tensors (32x32 resolution)
- **Domain Mapping:** Consistent signed mapping ($Value - 128$) was applied to both parameters and data to ensure valid comparison between PyTorch's `uint8` and our `int8` domains.

## 5. Verification Results
| Metric | Result |
| :--- | :--- |
| **Max Absolute Difference** | **0 LSB** |
| **Match Percentage** | **100.00%** |
| **Logic Integrity** | **Verified** |

## 6. Conclusion
The Concatenation mapping is robust and provides a perfect bit-accurate representation of the CPU-based glue logic in the quantized model. It is now fully trusted for use in complex blocks like `QC2f` and for joining skip connections in the model neck.

**Status:** VERIFIED.
