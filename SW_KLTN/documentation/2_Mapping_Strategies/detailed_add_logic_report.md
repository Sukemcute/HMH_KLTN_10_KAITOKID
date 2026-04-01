# Technical Report: Deep Dive into Addition Mapping & Logic Repair

## 1. Executive Summary
This report documents the identification and repair of the `ewise_add` primitive logic. Initial attempts at verification against PyTorch showed massive discrepancies (up to 255 LSB). The root cause was identified as a combination of a formula error in zero-point handling and a fundamental domain mismatch between signed and unsigned 8-bit integers. Following the fix, the implementation now achieves **100.00% bit-exact matching** with the reference model.

## 2. Original Implementation Analysis
The original code in `quant_domain_align.py` attempted to optimize the addition by staying in the integer domain.

**Original (Broken) Logic:**
```python
# The implementation incorrectly assumed that subtracting ZP once was sufficient
sum_int16 = A_common.astype(np.int16) + B_common.astype(np.int16)
sum_clamped = np.clip(sum_int16 - common_zp, INT8_MIN, INT8_MAX).astype(np.int16)
Y_int8 = np.clip(sum_clamped + zp_out, INT8_MIN, INT8_MAX).astype(np.int8)
```

### Mathematical Flaw
Quantized addition is defined as:
$Y_{float} = (A_{int} - ZP_A) \times S_A + (B_{int} - ZP_B) \times S_B$

If scales are aligned ($S_A = S_B = S_{out}$), the correct integer result is:
$Y_{int} = (A_{int} + B_{int} - ZP_A - ZP_B) + ZP_{out}$

**The Error:** The original code only subtracted the zero-point **once**, effectively leaving a "ghost" offset in the data.

## 3. Root Cause: The Signed vs. Unsigned Domain Mismatch
The most confusing errors (255 LSB) arose from how the zero-points from PyTorch (Unsigned) were interpreted in our Golden Model (Signed).

### The Example of Failure
Consider a pixel with value **150** and a zero-point of **63**.

1. **PyTorch (Unsigned uint8):**
   - Data ($X_{u8}$): 150
   - ZP ($ZP_{u8}$): 63
   - Math: $150 - 63 = \mathbf{87}$

2. **Incorrect Golden Model Interpretation (Bit-casting):**
   - Data ($X_{s8}$): -106 (This is the signed view of 150)
   - ZP ($ZP_{s8}$): 63
   - Math: $-106 - 63 = \mathbf{-169}$
   - **Result:** Massive error. The distance between 87 and -169 is exactly 256, causing an 8-bit wrap-around error.

## 4. The Fix: Consistent Domain Mapping
To solve this, we implemented a "Numeric Universe Shift." We must treat every value (data and parameters) using the same consistent offset.

### The Fix Strategy
We map the PyTorch domain to our domain using:
$X_{signed} = X_{uint8} - 128$
$ZP_{signed} = ZP_{uint8} - 128$

**Verification of the fix using the example above:**
- $X_{s8} = 150 - 128 = \mathbf{22}$
- $ZP_{s8} = 63 - 128 = \mathbf{-65}$
- Math: $22 - (-65) = \mathbf{87}$
- **Result:** Perfect match with PyTorch.

## 5. Final Verified Implementation
The `align_and_add` function was rewritten to use the "Golden Path" (Dequantize -> Float Add -> Quantize). This ensures that even if scales are not identical, the result matches the CPU's `FloatFunctional` behavior exactly.

```python
def align_and_add(A_int8, scale_A, zp_A, B_int8, scale_B, zp_B, scale_out, zp_out):
    # Step 1: Dequantize using signed parameters
    A_float = (A_int8.astype(np.float64) - zp_A) * scale_A
    B_float = (B_int8.astype(np.float64) - zp_B) * scale_B
    
    # Step 2: Sum in float domain
    sum_float = A_float + B_float
    
    # Step 3: Re-quantize to the target output domain
    return quantize_affine(sum_float, scale_out, zp_out, dtype="int8")
```

## 6. Verification Results
| Test | Result |
| :--- | :--- |
| **Tensor Size** | 16 channels x 32 x 32 |
| **Max Absolute Difference** | **0 LSB** |
| **Match Percentage** | **100.00%** |

**Conclusion:** The Addition mapping is now robust and bit-accurate.
