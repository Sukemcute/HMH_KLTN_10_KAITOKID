"""
primitive_conv.py – RS_DENSE_3x3 và OS_1x1 (Phase 1B)

Primitive convolution với đầy đủ:
  - Zero-point correction (precomputed offline)
  - Per-channel requant (M_int, shift)
  - SiLU activation (float path = golden accuracy)
  - Padding với zp_x để xử lý đúng asymmetric quantization

Rule: Không bao giờ tự implement requant hay SiLU ở đây –
      PHẢI import từ quant_affine.py.
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import INT8_MIN, INT8_MAX, ACT_SILU, ACT_NONE, ACT_RELU, ACT_RELU6
from quant.quant_affine import (
    make_requant_params,
    post_process_int32_to_int8,
    apply_silu_float,
    build_silu_lut,
    apply_silu_lut,
    dequantize_affine,
    quantize_affine,
)
from types import SimpleNamespace


# ─── Internal: Integer Conv2D ────────────────────────────────────────────────

def _conv2d_int(
    X_pad: np.ndarray,
    W: np.ndarray,
    stride: int,
) -> np.ndarray:
    """
    2D cross-correlation trên input đã PAD sẵn.

    X_pad : [N, Cin, H_pad, W_pad] int64 (đã pad với zp_x, không pad thêm)
    W     : [Cout, Cin, kH, kW] int64
    stride: int

    Returns: [N, Cout, Hout, Wout] int64
    """
    N, Cin, H_pad, W_pad = X_pad.shape
    Cout, _, kH, kW = W.shape
    Hout = (H_pad - kH) // stride + 1
    Wout = (W_pad - kW) // stride + 1

    Y = np.zeros((N, Cout, Hout, Wout), dtype=np.int64)

    # Loop qua kernel positions để tối ưu memory
    # print(f"    _conv2d_int: Processing {kH}x{kW} kernel...")
    for kh in range(kH):
        for kw in range(kW):
            # X_slice: [N, Cin, Hout, Wout]
            X_slice = X_pad[:, :, kh:kh + Hout * stride:stride,
                                  kw:kw + Wout * stride:stride]
            # W_kh_kw: [Cout, Cin]
            W_kh_kw = W[:, :, kh, kw].astype(np.int64)
            # Y += einsum('oi, nihw -> nohw')
            Y += np.tensordot(W_kh_kw, X_slice.astype(np.int64), axes=([1], [1])).transpose(1, 0, 2, 3)

    return Y


def _compute_pad(kernel: int, stride: int, H_in: int) -> int:
    """
    Tính padding để giữ 'same' semantics:
    H_out = ceil(H_in / stride)
    Matching PyTorch autopad: pad = kernel // 2
    """
    return kernel // 2


# ─── Core: Shared conv + requant + activation ────────────────────────────────

def _conv_requant_act(
    X_int8: np.ndarray,
    W_int8: np.ndarray,
    B_int32: np.ndarray,
    scale_x: float,
    zp_x: int,
    scale_w: np.ndarray,
    zp_w: int,
    scale_y: float,
    zp_y: int,
    stride: int,
    kernel: int,
    padding: int,
    activation: int,
    dump: bool = False,
) -> np.ndarray:
    """
    Core: conv INT8 → requant → activation → INT8.

    Thuật toán (theo spec, đầy đủ zero-point correction):
      1. Pad X với zp_x (đảm bảo padded pixel = "true 0" trong float space)
      2. Raw MAC: acc_raw = Σ X_pad_int * W_int
      3. Zero-point subtract + bias:
         acc = acc_raw - zp_x * Σ_w W_int + B_int32
         (chú ý: zp_w = 0 cho symmetric weight nên không cần Σ_x correction)
      4. Requant per-channel: y_int8 = post_process_int32_to_int8(acc, M, shift, zp_y)
      5. Activation (SiLU / ReLU / None)

    Returns:
        np.ndarray int8, shape [N, Cout, Hout, Wout]
    """
    assert zp_w == 0, f"zp_w phải = 0 (symmetric weight), nhận được {zp_w}"

    X_i64 = X_int8.astype(np.int64)
    W_i64 = W_int8.astype(np.int64)
    W_scale = np.asarray(scale_w, dtype=np.float64)

    N, Cin, H, W_in = X_i64.shape
    Cout = W_i64.shape[0]

    # 1. Pad với zp_x (không phải 0)
    if padding > 0:
        X_pad = np.pad(
            X_i64,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode='constant',
            constant_values=int(zp_x),
        )
    else:
        X_pad = X_i64

    # 2. Raw MAC
    acc_raw = _conv2d_int(X_pad, W_i64, stride)  # [N, Cout, Hout, Wout] int64

    # 3. Zero-point correction (precomputed) + bias
    #    partial_sum_w[cout] = Σ_{cin,kh,kw} W_int8[cout,cin,kh,kw]
    partial_sum_w = W_i64.sum(axis=(1, 2, 3))  # [Cout]
    zp_correction = (zp_x * partial_sum_w).reshape(1, Cout, 1, 1)
    bias = B_int32.astype(np.int64).reshape(1, Cout, 1, 1)
    acc = acc_raw - zp_correction + bias  # [N, Cout, Hout, Wout] int64

    # 4. Per-channel requant
    M_int_arr, shift_arr = make_requant_params(scale_x, W_scale, scale_y)
    y_pre_act = post_process_int32_to_int8(acc, M_int_arr, shift_arr, zp_y)

    if dump:
        return y_pre_act, scale_y, zp_y, {
            "acc_raw": acc_raw,
            "zp_correction": zp_correction,
            "bias": bias,
            "acc": acc
        }

    # 5. Activation
    if activation == ACT_SILU:
        y_out = apply_silu_float(y_pre_act, scale_y, zp_y)
    elif activation == ACT_RELU:
        float_zero = quantize_affine(np.array([0.0]), scale_y, zp_y, "int8")[0]
        y_out = np.clip(y_pre_act, float_zero, INT8_MAX).astype(np.int8)
    elif activation == ACT_RELU6:
        float_zero = quantize_affine(np.array([0.0]), scale_y, zp_y, "int8")[0]
        float_six = quantize_affine(np.array([6.0]), scale_y, zp_y, "int8")[0]
        y_out = np.clip(y_pre_act, float_zero, float_six).astype(np.int8)
    else:  # ACT_NONE
        y_out = y_pre_act

    return y_out


# ─── Public API ──────────────────────────────────────────────────────────────

def rs_dense_3x3(
    X_int8: np.ndarray,
    W_int8: np.ndarray,
    B_int32: np.ndarray,
    scale_x: float,
    zp_x: int,
    scale_w: np.ndarray,
    zp_w: int,
    scale_y: float,
    zp_y: int,
    stride: int = 1,
    padding: str = "same",
    activation: str = "silu",
    dump: bool = False,
) -> tuple:
    """
    RS_DENSE_3x3 – Regular Strided Dense Convolution 3×3.

    Dùng cho: Conv layers 0, 1, 3, 17 (stride=2) và bottleneck bên trong QC2f.

    Args:
        X_int8  : [N, Cin, H, W] int8
        W_int8  : [Cout, Cin, 3, 3] int8
        B_int32 : [Cout] int32
        scale_x : float – input scale
        zp_x    : int   – input zero-point
        scale_w : [Cout] float – per-channel weight scale
        zp_w    : int   – weight zero-point (PHẢI = 0)
        scale_y : float – output scale
        zp_y    : int   – output zero-point
        stride  : 1 hoặc 2
        padding : "same" (= kernel//2) hoặc int
        activation: "silu" | "relu" | "relu6" | "none"
        dump    : bool – trả về intermediate tensors

    Returns:
        (Y_int8, scale_y, zp_y)
        Nếu dump=True: (Y_int8, scale_y, zp_y, intermediates_dict)
    """
    assert W_int8.shape[2] == 3 and W_int8.shape[3] == 3, \
        f"rs_dense_3x3 cần kernel 3×3, nhận {W_int8.shape}"

    if isinstance(padding, str) and padding == "same":
        pad = _compute_pad(3, stride, X_int8.shape[2])
    else:
        pad = int(padding)

    act_id = {"silu": ACT_SILU, "relu": ACT_RELU, "relu6": ACT_RELU6, "none": ACT_NONE,
              None: ACT_NONE}.get(activation, ACT_NONE)

    Y_int8 = _conv_requant_act(
        X_int8, W_int8, B_int32,
        scale_x, zp_x, scale_w, zp_w, scale_y, zp_y,
        stride=stride, kernel=3, padding=pad, activation=act_id,
    )

    if dump:
        intermediates = {
            "X_input": X_int8.copy(),
            "W_int8": W_int8.copy(),
            "B_int32": B_int32.copy(),
            "pad": pad,
            "stride": stride,
        }
        return Y_int8, scale_y, zp_y, intermediates

    return Y_int8, scale_y, zp_y


def os_1x1(
    X_int8: np.ndarray,
    W_int8: np.ndarray,
    B_int32: np.ndarray,
    scale_x: float,
    zp_x: int,
    scale_w: np.ndarray,
    zp_w: int,
    scale_y: float,
    zp_y: int,
    activation: str = "none",
    dump: bool = False,
) -> tuple:
    """
    OS_1x1 – Output-Stationary 1×1 Convolution (Pointwise).

    Dùng cho: Projection trong QC2f, SPPF cv1/cv2, SCDown channel adjust, PSA.

    Gọi _conv_requant_act với kernel=1, stride=1, padding=0.

    Returns:
        (Y_int8, scale_y, zp_y)
    """
    assert W_int8.shape[2] == 1 and W_int8.shape[3] == 1, \
        f"os_1x1 cần kernel 1×1, nhận {W_int8.shape}"

    act_id = {"silu": ACT_SILU, "relu": ACT_RELU, "relu6": ACT_RELU6, "none": ACT_NONE,
              None: ACT_NONE}.get(activation, ACT_NONE)

    Y_int8 = _conv_requant_act(
        X_int8, W_int8, B_int32,
        scale_x, zp_x, scale_w, zp_w, scale_y, zp_y,
        stride=1, kernel=1, padding=0, activation=act_id,
    )

    if dump:
        return Y_int8, scale_y, zp_y, {"X_input": X_int8.copy()}

    return Y_int8, scale_y, zp_y
