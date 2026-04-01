"""
primitive_psa.py – GEMM_ATTN_BASIC (Phase 1E)

Quantized attention cho QPSA block tại 20×20.

Pipeline:
  Q = OS_1x1(X)           → [N, Hq, HW]  INT8
  K = OS_1x1(X)           → [N, Hk, HW]  INT8
  V = OS_1x1(X)           → [N, Hv, HW]  INT8
  Attn = Q × K^T           → [N, HW, HW]  INT8 (after requant)
  Attn_soft = softmax(Attn / sqrt(Hq))   INT8-approx
  Out = Attn_soft × V      → [N, HW, Hv]  INT8
  output_proj = OS_1x1(Out) → [N, C, H, W] INT8

Ưu tiên số 1: CHỨC NĂNG ĐÚNG (không phải tốc độ).
"""

import numpy as np
import sys, os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import INT8_MIN, INT8_MAX
from quant.quant_affine import (
    make_requant_params,
    post_process_int32_to_int8,
    quantize_affine,
    dequantize_affine,
)
from primitives.primitive_conv import os_1x1


# ─── Internal: INT8 GEMM ─────────────────────────────────────────────────────

def _int8_matmul(
    A_int8: np.ndarray,
    B_int8: np.ndarray,
) -> np.ndarray:
    """
    INT8 × INT8 → INT32 Matrix Multiplication.

    A : [N, M, K] int8
    B : [N, K, L] int8  (B đã được transpose nếu cần)
    Returns: [N, M, L] int64
    """
    A = A_int8.astype(np.int64)
    B = B_int8.astype(np.int64)
    return np.matmul(A, B)


# ─── Softmax Approximation ───────────────────────────────────────────────────

def _softmax_int8_approx(
    attn_int8: np.ndarray,
    scale_attn: float,
    zp_attn: int,
    scale_out: float,
    zp_out: int,
) -> np.ndarray:
    """
    INT8 softmax approximation (float path = golden accuracy).

    Dequant → float softmax → requant.
    Đây là cách đúng nhất cho golden Python; hardware sẽ dùng LUT/piecewise.

    attn_int8 : [N, HW, HW] int8
    Returns   : [N, HW, HW] int8
    """
    # Dequant về float
    attn_float = dequantize_affine(attn_int8, scale_attn, zp_attn)

    # Numerically stable softmax (trừ max trên axis cuối)
    attn_max = attn_float.max(axis=-1, keepdims=True)
    attn_exp = np.exp(attn_float - attn_max)
    attn_sum = attn_exp.sum(axis=-1, keepdims=True)
    soft_float = attn_exp / attn_sum

    # Requant về INT8
    return quantize_affine(soft_float, scale_out, zp_out, dtype="int8")


# ─── GEMM_ATTN_BASIC ─────────────────────────────────────────────────────────

def gemm_attn_basic(
    X_int8: np.ndarray,
    scale_x: float,
    zp_x: int,
    # Projection weights (INT8)
    W_Q: np.ndarray,
    W_K: np.ndarray,
    W_V: np.ndarray,
    W_out: np.ndarray,
    # Projection biases (INT32)
    B_Q: np.ndarray,
    B_K: np.ndarray,
    B_V: np.ndarray,
    B_out: np.ndarray,
    # Quant params dict
    scale_params: dict,
) -> tuple:
    """
    GEMM_ATTN_BASIC – Quantized Self-Attention.

    Dùng cho Layer 10 (QPSA): X shape [1, 256, 20, 20]

    scale_params dict PHẢI có các keys:
        scale_Q,  zp_Q      : Q projection output
        scale_K,  zp_K      : K projection output
        scale_V,  zp_V      : V projection output
        scale_wQ, zp_wQ     : weight scale/zp cho Q (thường zp=0)
        scale_wK, zp_wK     : ...
        scale_wV, zp_wV     : ...
        scale_wOut, zp_wOut : output projection weight
        scale_Attn, zp_Attn : Attn_raw requant output
        scale_Soft, zp_Soft : softmax output
        scale_AttnV, zp_AttnV : Attn×V output
        scale_out, zp_out   : final output

    Args:
        X_int8 : [N, C, H, W] int8 – input
        ...

    Returns:
        (Y_int8, scale_out, zp_out)
        Y_int8: [N, C, H, W] int8 – cùng shape với input
    """
    N, C, H, W = X_int8.shape
    HW = H * W

    sp = scale_params  # shorthand

    # ── Step 1: Q, K, V Projections (OS_1x1) ──────────────────────────────
    # W_Q shape: [Hq, C, 1, 1]
    Hq = W_Q.shape[0]
    Hk = W_K.shape[0]
    Hv = W_V.shape[0]

    Q_int8, sq, zq = os_1x1(
        X_int8, W_Q, B_Q,
        scale_x, zp_x,
        sp["scale_wQ"], sp.get("zp_wQ", 0),
        sp["scale_Q"], sp["zp_Q"],
    )
    K_int8, sk, zk = os_1x1(
        X_int8, W_K, B_K,
        scale_x, zp_x,
        sp["scale_wK"], sp.get("zp_wK", 0),
        sp["scale_K"], sp["zp_K"],
    )
    V_int8, sv, zv = os_1x1(
        X_int8, W_V, B_V,
        scale_x, zp_x,
        sp["scale_wV"], sp.get("zp_wV", 0),
        sp["scale_V"], sp["zp_V"],
    )

    # ── Step 2: Reshape → sequence dimension ──────────────────────────────
    # [N, Hq, H, W] → [N, HW, Hq]
    Q_seq = Q_int8.reshape(N, Hq, HW).transpose(0, 2, 1)  # [N, HW, Hq]
    K_seq = K_int8.reshape(N, Hk, HW).transpose(0, 2, 1)  # [N, HW, Hk]
    V_seq = V_int8.reshape(N, Hv, HW).transpose(0, 2, 1)  # [N, HW, Hv]

    # ── Step 3: Attention matrix Q × K^T ─────────────────────────────────
    # [N, HW, Hq] × [N, Hk, HW] (K^T) → [N, HW, HW]
    K_t = K_seq.transpose(0, 2, 1)  # [N, Hk, HW]
    Attn_raw = _int8_matmul(Q_seq, K_t)  # [N, HW, HW] int64

    # Scale factor: 1/sqrt(Hq) applied before requant
    sqrt_Hq = math.sqrt(float(Hq))
    scale_Attn_eff = float(sp["scale_Q"]) * float(sp["scale_K"]) / (
        float(sp["scale_Attn"]) * sqrt_Hq
    )

    # Requant Attn → INT8
    scale_Attn_w = np.array([scale_Attn_eff] * HW, dtype=np.float64)
    # Reshape Attn_raw for per-"channel" requant: [N, HW, HW] → treat HW as "Cout"
    Attn_perm = Attn_raw.transpose(0, 2, 1)  # [N, HW, HW] với "Cout"=HW on axis1
    M_attn, shift_attn = make_requant_params(1.0, scale_Attn_w, 1.0)
    Attn_i8_perm = post_process_int32_to_int8(
        Attn_perm, M_attn, shift_attn, sp["zp_Attn"]
    )
    Attn_i8 = Attn_i8_perm.transpose(0, 2, 1)  # [N, HW, HW]

    # ── Step 4: Softmax approximation ────────────────────────────────────
    Attn_soft = _softmax_int8_approx(
        Attn_i8,
        sp["scale_Attn"], sp["zp_Attn"],
        sp["scale_Soft"], sp["zp_Soft"],
    )  # [N, HW, HW]

    # ── Step 5: Attn_soft × V → Out ──────────────────────────────────────
    # [N, HW, HW] × [N, HW, Hv] → [N, HW, Hv]
    Out_raw = _int8_matmul(Attn_soft, V_seq)  # int64

    # Requant Out → INT8
    scale_AttnV_w = np.array(
        [float(sp["scale_Soft"]) * float(sp["scale_V"]) / float(sp["scale_AttnV"])] * Hv,
        dtype=np.float64,
    )
    Out_perm = Out_raw.transpose(0, 2, 1)  # [N, Hv, HW]
    M_out, shift_out = make_requant_params(1.0, scale_AttnV_w, 1.0)
    Out_i8_perm = post_process_int32_to_int8(
        Out_perm, M_out, shift_out, sp["zp_AttnV"]
    )
    Out_i8 = Out_i8_perm.transpose(0, 2, 1)  # [N, HW, Hv]

    # ── Step 6: Reshape back → [N, Hv, H, W] ────────────────────────────
    Out_spatial = Out_i8.transpose(0, 2, 1).reshape(N, Hv, H, W)  # [N, Hv, H, W]

    # ── Step 7: Output projection OS_1x1 ─────────────────────────────────
    Y_int8, scale_out, zp_out = os_1x1(
        Out_spatial, W_out, B_out,
        float(sp["scale_AttnV"]), int(sp["zp_AttnV"]),
        sp["scale_wOut"], sp.get("zp_wOut", 0),
        sp["scale_out"], sp["zp_out"],
    )

    return Y_int8, float(scale_out), int(zp_out)
