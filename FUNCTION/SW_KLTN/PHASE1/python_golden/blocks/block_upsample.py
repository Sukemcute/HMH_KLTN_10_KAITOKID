
import numpy as np
import sys, os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primitives.primitive_tensor import upsample_nearest

def block_upsample(
    X_int8: np.ndarray,
    scale_x: float,
    zp_x: int,
    scale_factor: int = 2,
) -> tuple:
    """
    Mapping for Upsample block.
    In YOLOv10n, this is always nearest neighbor upsampling.
    """
    return upsample_nearest(X_int8, scale_x, zp_x, scale_factor=scale_factor)
