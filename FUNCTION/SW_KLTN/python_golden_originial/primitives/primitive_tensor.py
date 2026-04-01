"""
primitive_tensor.py – MOVE, CONCAT, UPSAMPLE_NEAREST, EWISE_ADD (Phase 1D)

Tensor manipulation primitives.
CONCAT và EWISE_ADD PHẢI gọi qua quant_domain_align.py.
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant.quant_domain_align import align_and_concat, align_and_add


# ─── MOVE ────────────────────────────────────────────────────────────────────

def move(
    X_int8: np.ndarray,
    scale: float,
    zp: int,
) -> tuple:
    """
    MOVE – Copy tensor, giữ nguyên quant metadata.

    Dùng cho skip connections: lưu output của layer vào buffer.

    Returns:
        (X_int8.copy(), scale, zp)
    """
    return X_int8.copy(), float(scale), int(zp)


# ─── UPSAMPLE_NEAREST ────────────────────────────────────────────────────────

def upsample_nearest(
    X_int8: np.ndarray,
    scale_x: float,
    zp_x: int,
    scale_factor: int = 2,
) -> tuple:
    """
    UPSAMPLE_NEAREST – Nearest-neighbor upsampling × scale_factor.

    Quy tắc:
        Y[c, i*sf + di, j*sf + dj] = X[c, i, j]
        cho di in [0, sf), dj in [0, sf)

    scale_out = scale_in, zp_out = zp_in  ← KHÔNG đổi metadata.

    Args:
        X_int8       : [N, C, H, W] int8
        scale_x      : float – pass-through
        zp_x         : int   – pass-through
        scale_factor : int   – thường = 2

    Returns:
        (Y_int8, scale_x, zp_x)
    """
    sf = scale_factor
    N, C, H, W = X_int8.shape

    Y = np.zeros((N, C, H * sf, W * sf), dtype=np.int8)

    for di in range(sf):
        for dj in range(sf):
            Y[:, :, di::sf, dj::sf] = X_int8

    return Y, float(scale_x), int(zp_x)


# ─── CONCAT ──────────────────────────────────────────────────────────────────

def concat(
    tensors: list,
    scales: list,
    zps: list,
    axis: int = 1,
    strategy: str = "max",
) -> tuple:
    """
    CONCAT – Concatenate INT8 tensors với domain alignment.

    Gọi align_and_concat() từ quant_domain_align.py.
    KHÔNG tự implement lại alignment logic.

    Args:
        tensors  : list of np.ndarray int8 – các tensor cùng H, W
        scales   : list of float
        zps      : list of int
        axis     : int – thường = 1 (channel axis)
        strategy : str – "max" | "min" | "offline"

    Returns:
        (Y_int8, common_scale, common_zp)
    """
    return align_and_concat(tensors, scales, zps, axis=axis, strategy=strategy)


# ─── EWISE_ADD ───────────────────────────────────────────────────────────────

def ewise_add(
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
    EWISE_ADD – Element-wise ADD với saturation và domain alignment.

    Gọi align_and_add() từ quant_domain_align.py.
    KHÔNG tự implement lại alignment logic.

    Dùng cho: Residual connections (dự phòng, không có trong YOLOv10n neck).

    Returns:
        (Y_int8, scale_out, zp_out)
    """
    return align_and_add(
        A_int8, scale_A, zp_A,
        B_int8, scale_B, zp_B,
        scale_out=scale_out,
        zp_out=zp_out,
        strategy=strategy,
    )
