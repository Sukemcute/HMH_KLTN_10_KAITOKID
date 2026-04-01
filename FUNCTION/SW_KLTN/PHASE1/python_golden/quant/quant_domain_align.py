"""
quant_domain_align.py – Domain alignment cho CONCAT / EWISE_ADD (Phase 1A)

⚠️ RỦI RO SỐ 1 của toàn dự án.
Mọi primitive dùng CONCAT/ADD PHẢI gọi qua đây, KHÔNG tự viết lại logic align.

Nguồn chân lý duy nhất cho common-domain requant của CONCAT và ADD.
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import INT8_MIN, INT8_MAX
from quant.quant_affine import quantize_affine, dequantize_affine


# ─── Common Scale Strategy ────────────────────────────────────────────────────

def compute_common_scale(
    scale_list: list,
    zp_list: list = None,
    strategy: str = "max",
) -> tuple:
    """
    Chọn common scale/zp cho CONCAT hoặc ADD.

    Chiến lược:
        "max"     : common_scale = max(scale_list)  ← giữ precision tốt nhất
        "min"     : common_scale = min(scale_list)  ← tránh clamp ở nhánh nhỏ
        "offline" : dùng scale_list[0] (defined offline – preferred for HW)

    Args:
        scale_list : list of floats
        zp_list    : list of ints (nếu None → tất cả 0)
        strategy   : "max" | "min" | "offline"

    Returns:
        (common_scale, common_zp)
    """
    assert len(scale_list) >= 2, "Cần ít nhất 2 scale"
    if zp_list is None:
        zp_list = [0] * len(scale_list)

    if strategy == "max":
        idx = int(np.argmax(scale_list))
    elif strategy == "min":
        idx = int(np.argmin(scale_list))
    elif strategy == "offline":
        idx = 0
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return float(scale_list[idx]), int(zp_list[idx])


def requant_to_common(
    x_int8: np.ndarray,
    scale_src: float,
    zp_src: int,
    scale_dst: float,
    zp_dst: int,
) -> np.ndarray:
    """
    Requantize tensor từ (scale_src, zp_src) → (scale_dst, zp_dst).

    Dùng integer arithmetic thông qua float intermediate (golden accuracy).
    Nếu scale_src == scale_dst và zp_src == zp_dst → trả về copy (identity path).

    Returns:
        np.ndarray int8
    """
    # Identity path: không requant thêm
    if abs(scale_src - scale_dst) < 1e-12 and zp_src == zp_dst:
        return x_int8.copy()

    x_float = dequantize_affine(x_int8, scale_src, zp_src)
    return quantize_affine(x_float, scale_dst, zp_dst, dtype="int8")


# ─── Align and Concat ────────────────────────────────────────────────────────

def align_and_concat(
    tensors_int8: list,
    scales: list,
    zps: list,
    axis: int = 1,
    strategy: str = "max",
    scale_out: float = None,
    zp_out: int = None,
) -> tuple:
    """
    Align tất cả tensor về common domain rồi concatenate.

    Bước 1: Xác định common domain qua compute_common_scale() hoặc dùng scale_out
    Bước 2: Requant tất cả tensor về common domain
    Bước 3: numpy.concatenate theo axis

    Args:
        tensors_int8 : list of np.ndarray int8
        scales       : list of float – per-tensor scale
        zps          : list of int   – per-tensor zero-point
        axis         : int – concat axis (thường = 1 cho channel)
        strategy     : chiến lược chọn common scale
        scale_out    : optional fixed output scale
        zp_out       : optional fixed output zero-point

    Returns:
        (Y_int8, common_scale, common_zp)
    """
    assert len(tensors_int8) == len(scales) == len(zps), \
        f"Length mismatch: {len(tensors_int8)}, {len(scales)}, {len(zps)}"
    assert len(tensors_int8) >= 2

    if scale_out is not None and zp_out is not None:
        common_scale, common_zp = float(scale_out), int(zp_out)
    else:
        common_scale, common_zp = compute_common_scale(scales, zps, strategy=strategy)

    aligned = []
    for t, s, z in zip(tensors_int8, scales, zps):
        aligned.append(requant_to_common(t, s, z, common_scale, common_zp))

    Y_int8 = np.concatenate(aligned, axis=axis).astype(np.int8)
    return Y_int8, common_scale, common_zp


# ─── Align and Add ───────────────────────────────────────────────────────────

def align_and_add(
    A_int8: np.ndarray,
    scale_A: float,
    zp_A: int,
    B_int8: np.ndarray,
    scale_B: float,
    zp_B: int,
    scale_out: float = None,
    zp_out: int = 0,
    strategy: str = "max",
) -> tuple:
    """
    Element-wise ADD với domain alignment và saturation clamp.

    Args:
        A_int8, B_int8 : INT8 tensors cùng shape
        scale_A, zp_A  : quant params của A
        scale_B, zp_B  : quant params của B
        scale_out      : output scale (None → dùng common_scale)
        zp_out         : output zero-point
        strategy       : chiến lược chọn common scale

    Returns:
        (Y_int8, scale_out, zp_out)
    """
    assert A_int8.shape == B_int8.shape, \
        f"Shape mismatch: {A_int8.shape} vs {B_int8.shape}"

    # 1. Xác định target domain
    if scale_out is not None:
        target_scale, target_zp = float(scale_out), int(zp_out)
    else:
        target_scale, target_zp = compute_common_scale(
            [scale_A, scale_B], [zp_A, zp_B], strategy=strategy
        )

    # 2. Golden Math: Add in float domain then re-quantize
    # This is functionally equivalent to how PyTorch QFunctional.add works
    A_float = dequantize_affine(A_int8, scale_A, zp_A)
    B_float = dequantize_affine(B_int8, scale_B, zp_B)
    
    sum_float = A_float + B_float
    Y_int8 = quantize_affine(sum_float, target_scale, target_zp, dtype="int8")

    return Y_int8, float(target_scale), int(target_zp)
