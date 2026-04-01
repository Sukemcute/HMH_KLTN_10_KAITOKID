
import numpy as np
import sys, os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from primitives.primitive_conv import rs_dense_3x3, os_1x1
from primitives.primitive_tensor import concat, ewise_add

def block_qbottleneck(
    X_int8: np.ndarray,
    cv1_params: dict,
    cv2_params: dict,
    shortcut: bool = False,
    add_params: dict = None,
    dump: bool = False,
) -> tuple:
    """
    Mapping for QBottleneck: cv1 (3x3) -> cv2 (3x3) [+ shortcut]
    """
    # 1. First Conv (3x3)
    if dump:
        y1, s1, z1, d1 = rs_dense_3x3(X_int8, dump=True, **cv1_params)
    else:
        y1, s1, z1 = rs_dense_3x3(X_int8, **cv1_params)
    
    # 2. Second Conv (3x3)
    if dump:
        y2, s2, z2, d2 = rs_dense_3x3(y1, dump=True, **cv2_params)
    else:
        y2, s2, z2 = rs_dense_3x3(y1, **cv2_params)
    
    if shortcut:
        # Addition parameters:
        # A is input X_int8
        # B is output of cv2 (y2)
        out, sout, zout = ewise_add(
            A_int8 = X_int8,
            scale_A = cv1_params["scale_x"],
            zp_A = cv1_params["zp_x"],
            B_int8 = y2,
            scale_B = s2,
            zp_B = z2,
            scale_out = add_params["scale_out"],
            zp_out = add_params["zp_out"],
            strategy = "offline" # We use the fixed scale from the model
        )
        if dump:
            return out, sout, zout, {"cv1": d1, "cv2": d2}
        return out, sout, zout
    
    if dump:
        return y2, s2, z2, {"cv1": d1, "cv2": d2}
    return y2, s2, z2

def block_qc2f(
    X_int8: np.ndarray,
    cv1_params: dict,
    cv2_params: dict,
    bottleneck_list_params: list,
    concat_params: dict,
    dump: bool = False,
) -> tuple:
    """
    Mapping for QC2f:
    1. cv1 (1x1) -> split into (a, b)
    2. b -> m[0] -> m[1] -> ... -> m[n]
    3. concat(a, b, m[0], m[1], ...)
    4. cv2 (1x1)
    """
    # 1. Initial 1x1 Conv
    y_cv1, s_cv1, z_cv1 = os_1x1(X_int8, **cv1_params)
    
    # 2. Split (chunk)
    # y_cv1 shape is [N, 2*C, H, W]
    mid = y_cv1.shape[1] // 2
    y_split = [y_cv1[:, :mid, :, :], y_cv1[:, mid:, :, :]]
    
    # 3. Bottlenecks
    y_m = []
    current_input = y_split[1]
    for b_params in bottleneck_list_params:
        out, s_out, z_out = block_qbottleneck(
            current_input, 
            cv1_params=b_params["cv1_params"],
            cv2_params=b_params["cv2_params"],
            shortcut=b_params["shortcut"],
            add_params=b_params.get("add_params")
        )
        y_m.append(out)
        current_input = out
        
    # 4. Concat all
    # Tensors to concat: y_split[0], y_split[1], y_m[0], y_m[1], ...
    to_concat = y_split + y_m
    print(f"  QC2f: Concatenating {len(to_concat)} tensors...")
    for i, t in enumerate(to_concat):
        print(f"    Tensor {i} shape: {t.shape}")
    
    # Pass all input scales/zps to concat primitive
    y_cat, s_cat, z_cat = concat(
        to_concat, 
        scales=concat_params["scales"], 
        zps=concat_params["zps"], 
        strategy=concat_params.get("strategy", "offline"),
        scale_out=concat_params.get("scale_out"),
        zp_out=concat_params.get("zp_out")
    )
    
    # 5. Final 1x1 Conv
    y_out, s_out, z_out = os_1x1(y_cat, **cv2_params)
    
    return y_out, s_out, z_out
