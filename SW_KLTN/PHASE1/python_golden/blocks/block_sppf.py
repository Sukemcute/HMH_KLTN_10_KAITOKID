
import numpy as np
import sys, os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primitives.primitive_conv import os_1x1
from primitives.primitive_pool import maxpool_5x5
from primitives.primitive_tensor import concat

def block_sppf(
    X_int8: np.ndarray,
    cv1_params: dict,
    cv2_params: dict,
    pool_params: dict = None,
    concat_params: dict = None,
    dump: bool = False,
) -> tuple:
    """
    Mapping for SPPF:
    1. cv1: OS_1x1 (Expansion)
    2. m: 3 consecutive MAXPOOL_5x5
    3. concat: Join all 4 intermediate results
    4. cv2: OS_1x1 (Compression)
    """
    # 1. Initial 1x1 Conv
    if dump:
        y_cv1, s_cv1, z_cv1, d1 = os_1x1(X_int8, dump=True, **cv1_params)
    else:
        y_cv1, s_cv1, z_cv1 = os_1x1(X_int8, **cv1_params)
    
    # 2. Sequential Max Pooling
    # y = [y_cv1]
    # y.extend(self.m(y[-1]) for _ in range(3))
    y_pooled = [y_cv1]
    curr_y = y_cv1
    for i in range(3):
        curr_y, _, _ = maxpool_5x5(curr_y, s_cv1, z_cv1)
        y_pooled.append(curr_y)
        
    # 3. Concatenate
    # All pooled outputs have same scale/zp as cv1 output
    if concat_params is None:
        # If not provided, assume pass-through concat (scales match)
        concat_params = {
            "scales": [s_cv1] * 4,
            "zps": [z_cv1] * 4,
            "scale_out": s_cv1,
            "zp_out": z_cv1
        }
    
    y_cat, s_cat, z_cat = concat(y_pooled, **concat_params)
    
    # 4. Final 1x1 Conv
    if dump:
        y_out, s_out, z_out, d2 = os_1x1(y_cat, dump=True, **cv2_params)
        return y_out, s_out, z_out, {"cv1": d1, "cv2": d2}
    else:
        y_out, s_out, z_out = os_1x1(y_cat, **cv2_params)
        return y_out, s_out, z_out
