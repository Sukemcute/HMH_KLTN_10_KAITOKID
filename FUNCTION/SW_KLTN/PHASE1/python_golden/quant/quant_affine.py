"""
quant_affine.py – Core affine quantization math (Phase 1A)

Nguồn chân lý duy nhất cho mọi phép tính quantize/dequantize/requant.
Mọi primitive khác PHẢI import từ đây, KHÔNG tự implement lại.

Rule: Mọi function trả INT8 tensor PHẢI trả (tensor, scale, zp) cùng lúc.
"""

import math
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    INT8_MIN, INT8_MAX, INT32_MIN, INT32_MAX,
    ROUNDING_MODE, QUANT_WEIGHT_ZP,
)

# ─── Rounding ────────────────────────────────────────────────────────────────

def _round(x: np.ndarray) -> np.ndarray:
    """Áp dụng rounding theo cấu hình ROUNDING_MODE."""
    if ROUNDING_MODE == "half_up":
        return np.floor(np.asarray(x, dtype=np.float64) + 0.5).astype(np.int64)
    else:  # "half_even" – banker's rounding
        return np.round(np.asarray(x, dtype=np.float64)).astype(np.int64)


# ─── Quantize / Dequantize ───────────────────────────────────────────────────

def quantize_affine(
    x_float: np.ndarray,
    scale: float,
    zp: int,
    dtype: str = "int8",
) -> np.ndarray:
    """
    Quantize float → INT8 (hoặc uint8).

    x_int = clamp(round(x_float / scale) + zp, min_val, max_val)

    Args:
        x_float : float32/float64 array, bất kỳ shape.
        scale   : float > 0
        zp      : int, zero-point
        dtype   : "int8" | "uint8" | "int32"

    Returns:
        Quantized numpy array với dtype tương ứng.
    """
    assert scale > 0, f"scale phải > 0, nhận được {scale}"

    if dtype == "int8":
        min_val, max_val, out_dtype = INT8_MIN, INT8_MAX, np.int8
    elif dtype == "uint8":
        min_val, max_val, out_dtype = 0, 255, np.uint8
    elif dtype == "int32":
        min_val, max_val, out_dtype = INT32_MIN, INT32_MAX, np.int32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    x_scaled = _round(np.asarray(x_float, dtype=np.float64) / scale)
    x_int = np.clip(x_scaled + zp, min_val, max_val).astype(out_dtype)
    return x_int


def dequantize_affine(
    x_int: np.ndarray,
    scale: float,
    zp: int,
) -> np.ndarray:
    """
    Dequantize INT8 → float64.

    x_float = (x_int - zp) * scale

    KHÔNG dùng trong execution path; chỉ dùng để debug / verify / test.
    """
    return (x_int.astype(np.float64) - zp) * float(scale)


# ─── Fixed-point decomposition ───────────────────────────────────────────────

def _fixed_point_decompose_scalar(M: float) -> tuple:
    """
    Decompose scalar M thành (M_int, shift) sao cho:
        M ≈ M_int / 2^shift
        M_int fits INT32, shift ∈ [0, 31]
        M_int / 2^shift ≈ M với sai số < 1e-5 (relative)

    Chiến lược: chọn shift để M_int nằm trong [2^29, 2^30) → precision tốt nhất.
    Hoạt động đúng cho cả M < 1 và M ≥ 1.
    """
    if M <= 0.0:
        return 0, 0

    log2_M = math.log2(M)
    # Target: M * 2^shift ≈ 2^30  →  shift ≈ 30 - floor(log2_M)
    shift = int(math.floor(30 - log2_M))
    shift = max(0, min(shift, 31))
    M_int = int(round(M * (2 ** shift)))
    M_int = min(max(M_int, 0), INT32_MAX)
    return M_int, shift


def make_requant_params(
    scale_in: float,
    scale_w_per_ch: np.ndarray,
    scale_out: float,
) -> tuple:
    """
    Tính per-channel requant parameters cho một conv layer.

    M[cout] = scale_in * scale_w[cout] / scale_out
    → (M_int[cout], shift[cout]) via fixed-point decompose

    Args:
        scale_in      : scalar float – scale của activation input
        scale_w_per_ch: [Cout] float – per-channel weight scale
        scale_out     : scalar float – scale của activation output

    Returns:
        M_int_arr : np.ndarray [Cout] int64  – fixed-point multipliers
        shift_arr : np.ndarray [Cout] int32  – bit shifts
    """
    assert scale_in > 0 and scale_out > 0
    scale_w = np.asarray(scale_w_per_ch, dtype=np.float64)
    M = scale_in * scale_w / scale_out  # [Cout]

    M_int_list, shift_list = [], []
    for m in M.flat:
        m_int, sh = _fixed_point_decompose_scalar(float(m))
        M_int_list.append(m_int)
        shift_list.append(sh)

    return (
        np.array(M_int_list, dtype=np.int64),
        np.array(shift_list, dtype=np.int32),
    )


def post_process_int32_to_int8(
    acc_int32: np.ndarray,
    M_int: np.ndarray,
    shift: np.ndarray,
    zp_out: int,
) -> np.ndarray:
    """
    Post-process INT32 accumulator → INT8.

    y_raw[cout] = (acc_int32[...,cout,...] * M_int[cout]) >> shift[cout]
    y_int8      = clamp(y_raw + zp_out, -128, 127)

    Args:
        acc_int32 : array shape [N, Cout, H, W] hoặc [Cout] – INT32/INT64
        M_int     : [Cout] int64 – per-channel multipliers
        shift     : [Cout] int32 – per-channel shifts
        zp_out    : int – output zero-point

    Returns:
        np.ndarray int8, cùng shape với acc_int32
    """
    acc = acc_int32.astype(np.int64)
    M_int_arr = np.asarray(M_int, dtype=np.int64)
    shift_arr = np.asarray(shift, dtype=np.int32)
    Cout = M_int_arr.shape[0]

    y_raw = np.zeros_like(acc, dtype=np.int64)

    if acc.ndim == 1:
        # Shape: [Cout]
        for c in range(Cout):
            sh = int(shift_arr[c])
            offset = (1 << (sh - 1)) if sh > 0 else 0
            y_raw[c] = (acc[c] * M_int_arr[c] + offset) >> sh
    elif acc.ndim == 4:
        # Shape: [N, Cout, H, W]
        for c in range(Cout):
            sh = int(shift_arr[c])
            offset = (1 << (sh - 1)) if sh > 0 else 0
            y_raw[:, c, :, :] = (acc[:, c, :, :] * M_int_arr[c] + offset) >> sh
    elif acc.ndim == 3:
        # Shape: [N, Cout, L] – dùng cho GEMM
        for c in range(Cout):
            sh = int(shift_arr[c])
            offset = (1 << (sh - 1)) if sh > 0 else 0
            y_raw[:, c, :] = (acc[:, c, :] * M_int_arr[c] + offset) >> sh
    elif acc.ndim == 2:
        # Shape: [N, Cout]
        for c in range(Cout):
            sh = int(shift_arr[c])
            offset = (1 << (sh - 1)) if sh > 0 else 0
            y_raw[:, c] = (acc[:, c] * M_int_arr[c] + offset) >> sh
    else:
        raise ValueError(f"Unsupported acc ndim: {acc.ndim}")

    # Final conversion to INT8 with saturation
    # Logic: Rounding is already done via 'offset' and '>> sh'.
    # We just need to add the output zero-point and clamp.
    y_int32 = y_raw.astype(np.int32) + zp_out
    y_int8 = np.clip(y_int32, INT8_MIN, INT8_MAX).astype(np.int8)
    return y_int8


# ─── SiLU Activation ─────────────────────────────────────────────────────────

def build_silu_lut(scale_y: float, zp_y: int) -> np.ndarray:
    """
    Xây dựng 256-entry SiLU LUT cho hardware.

    Index i ∈ [0, 255] ánh xạ tới INT8 value (i - 128).
    LUT[i] = quantize_affine(SiLU(dequantize_affine(i-128, scale_y, zp_y)),
                              scale_y, zp_y)

    Returns:
        np.ndarray [256] int8
    """
    lut = np.zeros(256, dtype=np.int8)
    for i in range(256):
        int8_val = i - 128  # -128 to 127
        float_val = (int8_val - zp_y) * float(scale_y)
        silu_val = float_val * (1.0 / (1.0 + math.exp(-float_val)))
        lut[i] = quantize_affine(np.array([silu_val]), scale_y, zp_y, dtype="int8")[0]
    return lut


def apply_relu(Y_int8: np.ndarray, zp_y: int) -> np.ndarray:
    """
    Apply ReLU in INT8 domain.
    Y_int8 = max(Y_int8, zp_y)
    """
    return np.maximum(Y_int8, zp_y).astype(np.int8)

def apply_silu_float(
    y_int8: np.ndarray,
    scale_y: float,
    zp_y: int,
) -> np.ndarray:
    """
    Áp dụng SiLU via float path (golden accuracy).

    Dequant → SiLU float → requant với cùng scale/zp.
    Đây là "nguồn chân lý" cho activation SiLU.

    Args:
        y_int8  : INT8 array (pre-activation)
        scale_y : scale của tensor (cùng scale cho pre và post activation)
        zp_y    : zero-point

    Returns:
        np.ndarray int8, cùng shape
    """
    y_float = dequantize_affine(y_int8, scale_y, zp_y)
    silu_float = y_float * (1.0 / (1.0 + np.exp(-y_float)))
    return quantize_affine(silu_float, scale_y, zp_y, dtype="int8")


def apply_silu_lut(
    y_int8: np.ndarray,
    lut: np.ndarray,
) -> np.ndarray:
    """
    Áp dụng SiLU qua LUT (hardware-faithful).

    Args:
        y_int8 : INT8 array – index = value + 128
        lut    : [256] int8 – từ build_silu_lut()

    Returns:
        np.ndarray int8
    """
    idx = (y_int8.astype(np.int32) + 128).clip(0, 255).astype(np.uint8)
    return lut[idx].astype(np.int8)
