
import numpy as np
import sys, os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primitives.primitive_tensor import concat

def block_concat(
    tensors: list,
    scales: list,
    zps: list,
    concat_params: dict,
) -> tuple:
    """
    Mapping for QConcat block.
    """
    return concat(
        tensors,
        scales=scales,
        zps=zps,
        **concat_params
    )
