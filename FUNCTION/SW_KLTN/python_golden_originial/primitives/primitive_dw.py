"""
primitive_dw.py – DW_3x3 và DW_7x7_MULTIPASS (Phase 1C)

Depthwise convolution với per-channel requant.

Key differences vs RS_DENSE_3x3:
  - groups = Cin → mỗi channel xử lý độc lập (không cross-channel reduce)
  - Per-channel scale_w, bias, M_int, shift
  - last_pass luôn = True cho DW_3x3 (không accumulate cross-channel)
  - DW_7x7_MULTIPASS: tích lũy PSUM qua 3 pass, requant chỉ ở pass cuối
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import INT8_MIN, INT8_MAX, ACT_SILU, ACT_NONE, DW7x7_SPLIT
from quant.quant_affine import (
    make_requant_params,
    post_process_int32_to_int8,
    apply_silu_float,
    dequantize_affine,
)


# ─── Internal: Depthwise Conv ────────────────────────────────────────────────

def _dw_conv_channel(
    x_ch: np.ndarray,
    w_ch: np.ndarray,
    stride: int,
    pad: int,
    zp_x: int,
) -> np.ndarray:
    """
    Depthwise conv cho 1 channel.

    x_ch : [H, W] int64 – single channel input
    w_ch : [kH, kW] int64 – single channel kernel
    stride, pad, zp_x

    Returns: [Hout, Wout] int64 – raw MAC (trước zp correction)
    """
    H, W = x_ch.shape
    kH, kW = w_ch.shape

    if pad > 0:
        x_pad = np.pad(
            x_ch, ((pad, pad), (pad, pad)),
            mode='constant', constant_values=int(zp_x),
        )
    else:
        x_pad = x_ch

    Hout = (x_pad.shape[0] - kH) // stride + 1
    Wout = (x_pad.shape[1] - kW) // stride + 1
    acc = np.zeros((Hout, Wout), dtype=np.int64)

    for kh in range(kH):
        for kw in range(kW):
            x_slice = x_pad[kh:kh + Hout * stride:stride, kw:kw + Wout * stride:stride]
            acc += x_slice * int(w_ch[kh, kw])

    return acc


def _dw_conv_channel_partial(
    x_ch: np.ndarray,
    w_ch_rows: np.ndarray,
    row_start: int,
    stride: int,
    pad_top: int,
    pad_side: int,
    H_full: int,
    zp_x: int,
) -> np.ndarray:
    """
    Depthwise partial conv cho một subset rows của kernel 7×7.

    x_ch      : [H, W] int64 – toàn bộ input channel (không pad)
    w_ch_rows : [n_rows, 7] int64 – subset kernel rows
    row_start : int – kernel row index bắt đầu (0, 3, hoặc 6)
    stride    : int
    pad_top   : int – padding ở top của full 7×7 kernel (thường = 3)
    pad_side  : int – padding ở sides (thường = 3)
    H_full    : int – H_out của full convolution
    zp_x      : int

    Returns: [H_out, W_out] int64 – partial accumulation
    """
    H_in, W_in = x_ch.shape
    n_rows, kW = w_ch_rows.shape

    # Pad input với zp_x
    x_pad = np.pad(
        x_ch,
        ((pad_top, pad_top), (pad_side, pad_side)),
        mode='constant',
        constant_values=int(zp_x),
    )

    W_out = (W_in + 2 * pad_side - kW) // stride + 1
    acc = np.zeros((H_full, W_out), dtype=np.int64)

    for local_row_idx, kh_abs in enumerate(range(row_start, row_start + n_rows)):
        for kw in range(kW):
            x_slice = x_pad[kh_abs:kh_abs + H_full * stride:stride,
                            kw:kw + W_out * stride:stride]
            if x_slice.shape == (H_full, W_out):
                acc += x_slice * int(w_ch_rows[local_row_idx, kw])

    return acc


# ─── DW_3x3 ──────────────────────────────────────────────────────────────────

def dw_3x3(
    X_int8: np.ndarray,
    W_int8_per_ch: np.ndarray,
    B_int32_per_ch: np.ndarray,
    scale_x: float,
    zp_x: int,
    scale_w_per_ch: np.ndarray,
    scale_y: float,
    zp_y: int,
    stride: int = 1,
    activation: str = "none",
    dump: bool = False,
) -> tuple:
    """
    DW_3x3 – Depthwise 3×3 Convolution với per-channel requant.

    Dùng cho: SCDown nhánh DW (stride=2), các DW bên trong block.

    Thuật toán per-channel:
      for c in range(C):
        M[c] = scale_x * scale_w[c] / scale_y
        acc[c] = Σ_{kh,kw} x[c,h_in,w_in] * W[c,kh,kw]
               - zp_x * Σ_{kh,kw} W[c,kh,kw]      ← zero-point correction
               + B_int32[c]
        y[c] = clamp(round(acc[c] * M[c]) + zp_y, -128, 127)

    Args:
        X_int8          : [N, C, H, W] int8
        W_int8_per_ch   : [C, 3, 3] int8  (groups=C)
        B_int32_per_ch  : [C] int32
        scale_x         : float
        zp_x            : int
        scale_w_per_ch  : [C] float – per-channel weight scale
        scale_y         : float
        zp_y            : int
        stride          : 1 hoặc 2
        activation      : "none" | "silu"
        dump            : bool

    Returns:
        (Y_int8, scale_y, zp_y)
    """
    N, C, H, W = X_int8.shape
    assert W_int8_per_ch.shape == (C, 3, 3), \
        f"DW_3x3 weight phải [C,3,3], nhận {W_int8_per_ch.shape}"
    assert B_int32_per_ch.shape == (C,)

    pad = 1  # kernel=3, 'same' padding

    X_i64 = X_int8.astype(np.int64)
    W_i64 = W_int8_per_ch.astype(np.int64)
    scale_w = np.asarray(scale_w_per_ch, dtype=np.float64)

    # Per-channel requant params
    M_int_arr, shift_arr = make_requant_params(scale_x, scale_w, scale_y)

    # Compute output shape
    Hout = (H + 2 * pad - 3) // stride + 1
    Wout = (W + 2 * pad - 3) // stride + 1
    acc_all = np.zeros((N, C, Hout, Wout), dtype=np.int64)

    for c in range(C):
        partial_sum_w_c = int(W_i64[c].sum())
        for n in range(N):
            raw_mac = _dw_conv_channel(X_i64[n, c], W_i64[c], stride, pad, zp_x)
            acc_all[n, c] = raw_mac - zp_x * partial_sum_w_c + int(B_int32_per_ch[c])

    # Requant per-channel
    Y_int8 = post_process_int32_to_int8(acc_all, M_int_arr, shift_arr, zp_y)

    # Activation
    if activation == "silu":
        Y_int8 = apply_silu_float(Y_int8, scale_y, zp_y)

    if dump:
        return Y_int8, scale_y, zp_y, {"acc_all": acc_all.copy()}

    return Y_int8, scale_y, zp_y


# ─── DW_7x7_MULTIPASS ────────────────────────────────────────────────────────

def dw_7x7_multipass(
    X_int8: np.ndarray,
    W_int8_per_ch: np.ndarray,
    B_int32_per_ch: np.ndarray,
    scale_x: float,
    zp_x: int,
    scale_w_per_ch: np.ndarray,
    scale_y: float,
    zp_y: int,
    stride: int = 1,
    split: tuple = None,
    activation: str = "none",
    dump: bool = False,
) -> tuple:
    """
    DW_7x7_MULTIPASS – Depthwise 7×7 chia làm nhiều pass.

    Kernel 7×7 split thành 3 passes: [3 rows] + [3 rows] + [1 row]
    Pass 1, 2: accumulate PSUM (INT32, no requant)
    Pass 3 (last_pass): PSUM + bias → requant → INT8

    QUAN TRỌNG: OUTPUT phải == monolithic DW_7x7 result.

    Args:
        X_int8          : [N, C, H, W] int8
        W_int8_per_ch   : [C, 7, 7] int8
        B_int32_per_ch  : [C] int32
        scale_x         : float
        zp_x            : int
        scale_w_per_ch  : [C] float
        scale_y         : float
        zp_y            : int
        stride          : int (thường = 1 cho QC2fCIB)
        split           : tuple (rows_p1, rows_p2, rows_p3), default (3,3,1)
        activation      : "none" | "silu"
        dump            : bool – True trả về PSUM sau từng pass

    Returns:
        (Y_int8, scale_y, zp_y)
        Nếu dump=True: (Y_int8, scale_y, zp_y, psum_traces)
            psum_traces = {"psum_after_p1": ..., "psum_after_p2": ...}
    """
    if split is None:
        split = DW7x7_SPLIT  # (3, 3, 1)

    N, C, H, W = X_int8.shape
    assert W_int8_per_ch.shape == (C, 7, 7), \
        f"DW_7x7 weight phải [C,7,7], nhận {W_int8_per_ch.shape}"

    pad = 3  # kernel=7, 'same' padding

    X_i64 = X_int8.astype(np.int64)
    W_i64 = W_int8_per_ch.astype(np.int64)
    scale_w = np.asarray(scale_w_per_ch, dtype=np.float64)

    # Compute output shape
    Hout = (H + 2 * pad - 7) // stride + 1
    Wout = (W + 2 * pad - 7) // stride + 1

    # Per-channel requant params
    M_int_arr, shift_arr = make_requant_params(scale_x, scale_w, scale_y)

    # PSUM: INT32 accumulator (không requant đến last_pass)
    PSUM = np.zeros((N, C, Hout, Wout), dtype=np.int64)

    row_boundaries = [0, split[0], split[0] + split[1], 7]
    psum_traces = {}

    for pass_idx, (row_start, row_end) in enumerate(
        zip(row_boundaries[:-1], row_boundaries[1:])
    ):
        is_last_pass = (pass_idx == len(split) - 1)
        n_rows = row_end - row_start

        for c in range(C):
            w_rows = W_i64[c, row_start:row_end, :]  # [n_rows, 7]

            for n in range(N):
                partial = _dw_conv_channel_partial(
                    X_i64[n, c],
                    w_rows,
                    row_start=row_start,
                    stride=stride,
                    pad_top=pad,
                    pad_side=pad,
                    H_full=Hout,
                    zp_x=zp_x,
                )
                PSUM[n, c] += partial

        if dump and not is_last_pass:
            psum_traces[f"psum_after_p{pass_idx + 1}"] = PSUM.copy()

    # last_pass: + bias → requant → INT8
    bias = B_int32_per_ch.astype(np.int64).reshape(1, C, 1, 1)

    # Subtract zp_x * sum(W) per channel (full kernel)
    partial_sum_w_full = W_i64.sum(axis=(1, 2))  # [C]
    zp_correction = (zp_x * partial_sum_w_full).reshape(1, C, 1, 1)

    PSUM_final = PSUM - zp_correction + bias  # [N, C, Hout, Wout]

    Y_int8 = post_process_int32_to_int8(PSUM_final, M_int_arr, shift_arr, zp_y)

    if activation == "silu":
        Y_int8 = apply_silu_float(Y_int8, scale_y, zp_y)

    if dump:
        psum_traces["psum_after_p2"] = psum_traces.get(
            "psum_after_p2", PSUM.copy()
        )
        return Y_int8, scale_y, zp_y, psum_traces

    return Y_int8, scale_y, zp_y


# ─── Monolithic DW_7x7 (reference, dùng để verify multipass == monolithic) ──

def dw_7x7_monolithic(
    X_int8: np.ndarray,
    W_int8_per_ch: np.ndarray,
    B_int32_per_ch: np.ndarray,
    scale_x: float,
    zp_x: int,
    scale_w_per_ch: np.ndarray,
    scale_y: float,
    zp_y: int,
    stride: int = 1,
) -> tuple:
    """
    DW_7x7 monolithic (reference) – dùng để verify multipass == monolithic.

    Implements full 7×7 depthwise conv trong một lần (không split).
    """
    N, C, H, W = X_int8.shape
    assert W_int8_per_ch.shape == (C, 7, 7)

    pad = 3
    X_i64 = X_int8.astype(np.int64)
    W_i64 = W_int8_per_ch.astype(np.int64)
    scale_w = np.asarray(scale_w_per_ch, dtype=np.float64)

    Hout = (H + 2 * pad - 7) // stride + 1
    Wout = (W + 2 * pad - 7) // stride + 1

    M_int_arr, shift_arr = make_requant_params(scale_x, scale_w, scale_y)
    acc_all = np.zeros((N, C, Hout, Wout), dtype=np.int64)

    for c in range(C):
        partial_sum_w_c = int(W_i64[c].sum())
        for n in range(N):
            raw_mac = _dw_conv_channel(X_i64[n, c], W_i64[c], stride, pad, zp_x)
            acc_all[n, c] = raw_mac - zp_x * partial_sum_w_c + int(B_int32_per_ch[c])

    return post_process_int32_to_int8(acc_all, M_int_arr, shift_arr, zp_y), scale_y, zp_y
