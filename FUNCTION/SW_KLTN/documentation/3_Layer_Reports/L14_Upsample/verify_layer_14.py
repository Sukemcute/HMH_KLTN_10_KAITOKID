
import torch
import numpy as np
import os
import sys

# Set paths
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'ultralytics'))
sys.path.insert(0, os.path.join(os.getcwd(), 'PHASE1/python_golden'))

from ultralytics.quant.utils import load_ptq_model_from_state_dict
from ultralytics.myimplm import collect_results
from exhaustive_verify_model_flow import get_conv_params, get_qc2f_params, get_qpsa_params, get_qc2fcib_params, to_signed_int8
from blocks.block_conv import block_conv
from blocks.block_qc2f import block_qc2f
from blocks.block_scdown import block_scdown
from blocks.block_sppf import block_sppf
from blocks.block_qpsa import block_qpsa
from blocks.block_upsample import block_upsample
from blocks.block_concat import block_concat
from blocks.block_qc2f_cib import block_qc2f_cib

def verify_layer_14():
    print("=== Layer 14 (Upsample) Verification (100 Samples, 640x640 Flow) ===")
    base_weights = 'ultralytics/ultralytics/qyolov10n.yaml'
    quant_state_dict = 'ultralytics/ultralytics/quant/quant_state_dict/alpr_ptq_state_dict.pt'
    model = load_ptq_model_from_state_dict(base_weights, quant_state_dict)
    model.model.eval()
    
    concat_map = {12: [11, 6], 15: [14, 4], 18: [17, 13], 21: [20, 10]}
    matches, diffs = [], []

    for s in range(100):
        input_tensor = torch.rand(1, 3, 640, 640)
        res = collect_results(model, input_tensor)
        
        m = model.model.model[14]
        node = getattr(res, "Layer14")
        ref_out_signed = to_signed_int8(node().int_repr().numpy())
        
        # Extract realistic inputs
        if 14 == 0:
            ref_in = model.model.quant(input_tensor)
            in_signed, in_scale, in_zp = to_signed_int8(ref_in.int_repr().numpy()), float(ref_in.q_scale()), int(ref_in.q_zero_point())-128
        elif 14 in concat_map:
            in_signed = [to_signed_int8(getattr(res, f"Layer{idx}")().int_repr().numpy()) for idx in concat_map[14]]
            in_scale = [float(getattr(res, f"Layer{idx}")().q_scale()) for idx in concat_map[14]]
            in_zp = [int(getattr(res, f"Layer{idx}")().q_zero_point())-128 for idx in concat_map[14]]
        else:
            ref_in = getattr(res, "Layer13")()
            in_signed, in_scale, in_zp = to_signed_int8(ref_in.int_repr().numpy()), float(ref_in.q_scale()), int(ref_in.q_zero_point())-128

        # Run Golden Block
        my_out, _, _ = block_upsample(in_signed, in_scale, in_zp, scale_factor=2)

        diff = np.abs(my_out.astype(np.int16) - ref_out_signed.astype(np.int16))
        matches.append((diff == 0).sum() / diff.size * 100)
        diffs.append(np.max(diff))

    print(f"Mean Match: {np.mean(matches):.2f}%")
    print(f"Max Diff:   {np.max(diffs)} LSB")

if __name__ == "__main__":
    verify_layer_14()
