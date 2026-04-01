
import numpy as np
import sys, os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primitives.primitive_conv import os_1x1
from primitives.primitive_dw import dw_3x3
from primitives.primitive_tensor import ewise_add, concat
from primitives.primitive_psa import _int8_matmul, _softmax_int8_approx
from quant.quant_affine import post_process_int32_to_int8, make_requant_params

def block_qattention(
    X_int8: np.ndarray,
    scale_x: float,
    zp_x: int,
    qkv_params: dict,
    proj_params: dict,
    pe_params: dict,
    sm_params: dict,
    matmul1_params: dict,
    matmul2_params: dict,
    add_pe_params: dict,
    num_heads: int,
    dump: bool = False,
) -> tuple:
    """
    Mapping for QAttention module inside QPSA.
    Simulates every intermediate QFunctional requantization step.
    """
    B, C, H, W = X_int8.shape
    N = H * W
    
    # 1. QKV Projection [B, C, H, W] -> [B, 2*kd+hd, H, W]
    split_sizes = qkv_params.pop("split_sizes", [32, 32, 64]) 
    if dump:
        y_qkv, s_qkv, z_qkv, d_qkv = os_1x1(X_int8, dump=True, **qkv_params)
    else:
        y_qkv, s_qkv, z_qkv = os_1x1(X_int8, **qkv_params)
    
    # 2. Reshape and Split
    kd = matmul1_params["key_dim"]
    hd = matmul2_params["head_dim"]
    y_reshaped = y_qkv.reshape(B, num_heads, kd * 2 + hd, N)
    q = y_reshaped[:, :, :kd, :]
    k = y_reshaped[:, :, kd:kd*2, :]
    v = y_reshaped[:, :, kd*2:, :]
    
    # 3. Matmul 1: q.transpose @ k
    from quant.quant_affine import quantize_affine, dequantize_affine
    q_t = q.transpose(0, 1, 3, 2)
    # Both q and k have zp_qkv (z_qkv)
    amul_raw = _int8_matmul(q_t, k, zp_A=z_qkv, zp_B=z_qkv)
    
    # Requant 1: To mul_fn domain (s_q * s_k / s_mul_fn)
    amul_float = amul_raw.astype(np.float64) * (s_qkv * s_qkv)
    amul_i8 = quantize_affine(amul_float, matmul1_params["scale_out"], matmul1_params["zp_out"], dtype="int8")
    
    # 4. Mul Scalar: attn = amul * scale
    amul_f = dequantize_affine(amul_i8, matmul1_params["scale_out"], matmul1_params["zp_out"])
    attn_float = amul_f * matmul1_params["head_scale"]
    attn_score = quantize_affine(attn_float, matmul1_params["scale_out"], matmul1_params["zp_out"], dtype="int8")
    
    # 5. Softmax
    attn_soft = _softmax_int8_approx(
        attn_score,
        matmul1_params["scale_out"], matmul1_params["zp_out"],
        sm_params["scale_out"], sm_params["zp_out"]
    )
    
    # 6. Matmul 2: v @ attn_soft.transpose
    attn_soft_t = attn_soft.transpose(0, 1, 3, 2)
    # v has z_qkv, attn_soft has sm_params["zp_out"]
    x_raw = _int8_matmul(v, attn_soft_t, zp_A=z_qkv, zp_B=sm_params["zp_out"])
    
    # Requant 2: To mul_fn domain
    x_f = x_raw.astype(np.float64) * (s_qkv * sm_params["scale_out"])
    x_attn_i8 = quantize_affine(x_f, matmul2_params["scale_out"], matmul2_params["zp_out"], dtype="int8")
    
    # 7. Reshape back to spatial
    x_spatial = x_attn_i8.reshape(B, C, H, W)
    
    # 8. Positional Encoding (3x3 DW) applied to V!
    # PyTorch: self.pe(v.reshape(B, C, H, W))
    v_spatial = v.reshape(B, C, H, W)
    if dump:
        y_pe, s_pe, z_pe, d_pe = dw_3x3(v_spatial, dump=True, **pe_params)
    else:
        y_pe, s_pe, z_pe = dw_3x3(v_spatial, **pe_params)
    
    # 9. Add PE (QFunctional Add)
    # x = x_attn + y_pe
    x_fused, s_fused, z_fused = ewise_add(
        x_spatial, matmul2_params["scale_out"], matmul2_params["zp_out"],
        y_pe, s_pe, z_pe,
        scale_out = add_pe_params["scale_out"],
        zp_out = add_pe_params["zp_out"]
    )
    
    # 10. Final Projection
    if dump:
        y_out, s_out, z_out, d_proj = os_1x1(x_fused, dump=True, **proj_params)
        return y_out, s_out, z_out, {
            "qkv_out": y_qkv,
            "attn_score": attn_score,
            "attn_soft": attn_soft,
            "x_attn": x_spatial,
            "y_pe": y_pe,
            "x_fused": x_fused
        }
    else:
        y_out, s_out, z_out = os_1x1(x_fused, **proj_params)
        return y_out, s_out, z_out

def block_qpsa(
    X_int8: np.ndarray,
    cv1_params: dict,
    attn_params: dict,
    ffn_params: list,
    ffn_add_params: dict,
    attn_add_params: dict,
    cv2_params: dict,
    concat_params: dict,
    dump: bool = False,
) -> tuple:
    """
    Mapping for QPSA Block.
    """
    # 1. Initial Split
    if dump:
        y_cv1, s_cv1, z_cv1, d_cv1 = os_1x1(X_int8, dump=True, **cv1_params)
    else:
        y_cv1, s_cv1, z_cv1 = os_1x1(X_int8, **cv1_params)
    
    mid = y_cv1.shape[1] // 2
    a = y_cv1[:, :mid, :, :]
    b = y_cv1[:, mid:, :, :]
    
    # 2. Attention path
    if dump:
        y_attn, s_attn, z_attn, d_attn = block_qattention(b, s_cv1, z_cv1, dump=True, **attn_params)
    else:
        y_attn, s_attn, z_attn = block_qattention(b, s_cv1, z_cv1, **attn_params)
    
    # Shortcut 1: b = b + attn(b)
    b_attn, s_b_attn, z_b_attn = ewise_add(
        b, s_cv1, z_cv1,
        y_attn, s_attn, z_attn,
        **attn_add_params
    )
    
    # 3. FFN path
    y_f1, s_f1, z_f1 = os_1x1(b_attn, **ffn_params[0])
    y_f2, s_f2, z_f2 = os_1x1(y_f1, **ffn_params[1])
    
    # Shortcut 2: b = b + ffn(b)
    b_final, s_b_final, z_b_final = ewise_add(
        b_attn, s_b_attn, z_b_attn,
        y_f2, s_f2, z_f2,
        **ffn_add_params
    )
    
    # 4. Concat and Projection
    concat_params.pop("scales", None)
    concat_params.pop("zps", None)
    y_cat, s_cat, z_cat = concat([a, b_final], scales=[s_cv1, s_b_final], zps=[z_cv1, z_b_final], **concat_params)
    
    if dump:
        y_out, s_out, z_out, d_cv2 = os_1x1(y_cat, dump=True, **cv2_params)
        return y_out, s_out, z_out, {
            "cv1_out": y_cv1,
            "attn_out": y_attn,
            "b_attn": b_attn,
            "b_final": b_final,
            "d_attn": d_attn
        }
    else:
        y_out, s_out, z_out = os_1x1(y_cat, **cv2_params)
        return y_out, s_out, z_out
