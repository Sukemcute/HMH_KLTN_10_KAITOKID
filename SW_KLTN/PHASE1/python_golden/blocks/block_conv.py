
import numpy as np
import sys, os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primitives.primitive_conv import rs_dense_3x3, os_1x1

def block_conv(
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
    activation: str = "relu",
    dump: bool = False,
) -> tuple:
    """
    Mapping for the YOLO Conv block.
    Selects the appropriate primitive based on kernel size (3x3 or 1x1).
    """
    k = W_int8.shape[2]  # Kernel size
    
    if k == 3:
        return rs_dense_3x3(
            X_int8, W_int8, B_int32,
            scale_x, zp_x, scale_w, zp_w, scale_y, zp_y,
            stride=stride, padding=padding, activation=activation, dump=dump
        )
    elif k == 1:
        return os_1x1(
            X_int8, W_int8, B_int32,
            scale_x, zp_x, scale_w, zp_w, scale_y, zp_y,
            activation=activation, dump=dump
        )
    else:
        raise ValueError(f"Unsupported kernel size {k} for standard Conv block mapping.")
