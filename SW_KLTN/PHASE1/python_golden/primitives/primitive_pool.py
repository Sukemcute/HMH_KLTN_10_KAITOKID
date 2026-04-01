"""
primitive_pool.py – MAXPOOL_5x5 (Phase 1D)

Max pooling trên INT8: chỉ so sánh số nguyên, KHÔNG thay đổi scale/zp.
Dùng cho SPPF (lặp 3 lần liên tiếp).
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def maxpool_5x5(
    X_int8: np.ndarray,
    scale_x: float,
    zp_x: int,
    padding: int = 2,
) -> tuple:
    """
    MAXPOOL_5x5 – Max pooling 5×5, stride=1, padding=2.

    Thao tác so sánh số nguyên → KHÔNG đổi scale/zp.
    Output H, W giữ nguyên so với input (stride=1, padding=2 cho kernel=5).

    Args:
        X_int8   : [N, C, H, W] int8
        scale_x  : float – pass-through (KHÔNG thay đổi)
        zp_x     : int   – pass-through (KHÔNG thay đổi)
        padding  : int   – default 2 (để giữ H, W không đổi)

    Returns:
        (Y_int8, scale_x, zp_x)  ← scale/zp KHÔNG đổi
    """
    N, C, H, W = X_int8.shape
    kernel = 5
    stride = 1

    # Pad với INT8_MIN (= -128) để max không bị ảnh hưởng bởi padding
    # (giá trị pad_val = INT8_MIN sẽ luôn thua trong max comparison)
    pad_val = np.iinfo(np.int8).min  # -128

    if padding > 0:
        X_pad = np.pad(
            X_int8.astype(np.int16),
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode='constant',
            constant_values=int(pad_val),
        )
    else:
        X_pad = X_int8.astype(np.int16)

    H_pad, W_pad = X_pad.shape[2], X_pad.shape[3]
    Hout = (H_pad - kernel) // stride + 1
    Wout = (W_pad - kernel) // stride + 1

    Y = np.full((N, C, Hout, Wout), fill_value=pad_val, dtype=np.int16)

    for kh in range(kernel):
        for kw in range(kernel):
            x_slice = X_pad[
                :, :,
                kh:kh + Hout * stride:stride,
                kw:kw + Wout * stride:stride,
            ]
            Y = np.maximum(Y, x_slice)

    return Y.astype(np.int8), float(scale_x), int(zp_x)
