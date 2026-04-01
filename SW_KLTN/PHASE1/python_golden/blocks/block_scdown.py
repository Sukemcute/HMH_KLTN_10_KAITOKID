
import numpy as np
import sys, os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primitives.primitive_conv import os_1x1
from primitives.primitive_dw import dw_3x3

def block_scdown(
    X_int8: np.ndarray,
    cv1_params: dict,
    cv2_params: dict,
    dump: bool = False,
) -> tuple:
    """
    Mapping for SCDown:
    1. cv1: OS_1x1 (Pointwise Conv)
    2. cv2: DW_3x3 (Depthwise Conv)
    """
    # 1. Pointwise Conv (1x1)
    if dump:
        y1, s1, z1, d1 = os_1x1(X_int8, dump=True, **cv1_params)
    else:
        y1, s1, z1 = os_1x1(X_int8, **cv1_params)
    
    # 2. Depthwise Conv (3x3)
    # SCDown cv2 typically has stride=2 and no activation
    if dump:
        y2, s2, z2, d2 = dw_3x3(y1, dump=True, **cv2_params)
        return y2, s2, z2, {"cv1": d1, "cv2": d2}
    else:
        y2, s2, z2 = dw_3x3(y1, **cv2_params)
        return y2, s2, z2
