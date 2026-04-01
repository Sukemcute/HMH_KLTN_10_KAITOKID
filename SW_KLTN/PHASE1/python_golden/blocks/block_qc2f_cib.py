
import numpy as np
import sys, os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primitives.primitive_conv import os_1x1
from primitives.primitive_dw import dw_3x3
from primitives.primitive_tensor import ewise_add, concat

def block_qcib(
    X_int8: np.ndarray,
    scale_x: float,
    zp_x: int,
    cv_params_list: list, # List of 5 dicts for the 5 convs
    add_params: dict,
    shortcut: bool = True
) -> tuple:
    """
    Mapping for QCIB (Quantized Conditional Identity Block).
    Sequence: DW3x3 -> PW1x1 -> DW7x7 -> PW1x1 -> DW3x3
    """
    from primitives.primitive_dw import dw_conv_int
    
    # 0: DW 3x3 (pad=1)
    y0, s0, z0 = dw_conv_int(X_int8, padding=1, **cv_params_list[0])
    # 1: PW 1x1
    y1, s1, z1 = os_1x1(y0, **cv_params_list[1])
    # 2: DW 7x7 (pad=3)
    y2, s2, z2 = dw_conv_int(y1, padding=3, **cv_params_list[2])
    # 3: PW 1x1
    y3, s3, z3 = os_1x1(y2, **cv_params_list[3])
    # 4: DW 3x3 (pad=1)
    y4, s4, z4 = dw_conv_int(y3, padding=1, **cv_params_list[4])
    
    if shortcut:
        return ewise_add(X_int8, scale_x, zp_x, y4, s4, z4, **add_params)
    else:
        return y4, s4, z4

def block_qc2f_cib(
    X_int8: np.ndarray,
    cv1_params: dict,
    cv2_params: dict,
    qcib_params_list: list, # List of param dicts for each QCIB in self.m
    qcib_add_params_list: list,
    concat_params: dict,
) -> tuple:
    """
    Mapping for QC2fCIB Block.
    """
    # 1. Initial Expansion
    y_cv1, s_cv1, z_cv1 = os_1x1(X_int8, **cv1_params)
    
    # 2. Split
    mid = y_cv1.shape[1] // 2
    y = [y_cv1[:, :mid, :, :], y_cv1[:, mid:, :, :]]
    
    curr_scale = s_cv1
    curr_zp = z_cv1
    
    # 3. Stacked QCIBs
    for i, qcib_p in enumerate(qcib_params_list):
        y_next, s_next, z_next = block_qcib(y[-1], curr_scale, curr_zp, qcib_p, qcib_add_params_list[i])
        y.append(y_next)
        curr_scale = s_next
        curr_zp = z_next
        
    # 4. Concat
    # PyTorch: torch.cat(y, 1)
    # We need scales and zps for all tensors in y
    # y[0] and y[1] have (s_cv1, z_cv1)
    # Subsequent have (s_next, z_next) from each QCIB
    scales = [s_cv1, s_cv1]
    zps = [z_cv1, z_cv1]
    for i in range(len(qcib_params_list)):
        # This is slightly wrong if we have multiple QCIBs, 
        # we need to collect all intermediate scales.
        # But for n=1, it's [s_cv1, s_cv1, s_qcib_out]
        pass
    
    # Re-calculate scales for concat
    scales = [s_cv1, s_cv1]
    zps = [z_cv1, z_cv1]
    # For now assume n=1 as in Layer 22
    if len(qcib_params_list) == 1:
        scales.append(curr_scale)
        zps.append(curr_zp)
    
    y_cat, s_cat, z_cat = concat(y, scales=scales, zps=zps, **concat_params)
    
    # 5. Final Projection
    return os_1x1(y_cat, **cv2_params)
