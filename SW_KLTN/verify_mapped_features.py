
import torch
import numpy as np
import os
import sys

# Set paths
sys.path.insert(0, os.path.join(os.getcwd(), 'ultralytics'))
sys.path.insert(0, os.path.join(os.getcwd(), 'PHASE1/python_golden'))

from ultralytics.quant.utils import load_ptq_model_from_state_dict
from ultralytics.myimplm import collect_results
from model.qyolov10n_mapped import QYOLOv10nMapped, to_signed_int8

def test_features():
    print("=== Mapped Feature Verification (L16, L19, L22) ===")
    
    base_weights = 'ultralytics/ultralytics/qyolov10n.yaml'
    quant_state_dict = 'ultralytics/ultralytics/quant/quant_state_dict/alpr_ptq_state_dict.pt'
    ref_model = load_ptq_model_from_state_dict(base_weights, quant_state_dict)
    ref_model.model.eval()
    
    mapped_model = QYOLOv10nMapped(ref_model.model)
    input_tensor = torch.rand(1, 3, 640, 640)
    
    print("Running Reference Inference...")
    res = collect_results(ref_model.model, input_tensor)
    ref_features = {
        16: to_signed_int8(res.Layer16().int_repr().numpy()),
        19: to_signed_int8(res.Layer19().int_repr().numpy()),
        22: to_signed_int8(res.Layer22().int_repr().numpy())
    }
    
    print("Running Mapped Golden Inference...")
    # Manual run to capture intermediate features
    ref_in = mapped_model.fused_model.quant(input_tensor)
    current_x = to_signed_int8(ref_in.int_repr().numpy())
    
    saved_features = {} 
    from blocks.block_conv import block_conv
    from blocks.block_qc2f import block_qc2f
    from blocks.block_scdown import block_scdown
    from blocks.block_sppf import block_sppf
    from blocks.block_qpsa import block_qpsa
    from blocks.block_upsample import block_upsample
    from blocks.block_concat import block_concat
    from blocks.block_qc2f_cib import block_qc2f_cib

    for i in range(23):
        m = mapped_model.layers[i]
        type_name = type(m).__name__
        p = mapped_model.param_map[i]
        
        if type_name == "Conv":
            current_x, _, _ = block_conv(current_x, padding="same", **p)
        elif type_name == "QC2f":
            current_x, _, _ = block_qc2f(current_x, **p)
        elif type_name == "SCDown":
            current_x, _, _ = block_scdown(current_x, p["cv1_params"], p["cv2_params"])
        elif type_name == "SPPF":
            current_x, _, _ = block_sppf(current_x, p["cv1_p"], p["cv2_p"], concat_params=p["concat_p"])
        elif type_name == "QPSA":
            current_x, _, _ = block_qpsa(current_x, **p)
        elif type_name == "Upsample":
            current_x, _, _ = block_upsample(current_x, **p)
        elif type_name == "QConcat":
            if i == 12: inputs = [saved_features[11], saved_features[6]]
            elif i == 15: inputs = [saved_features[14], saved_features[4]]
            elif i == 18: inputs = [saved_features[17], saved_features[13]]
            elif i == 21: inputs = [saved_features[20], saved_features[10]]
            current_x, _, _ = block_concat(inputs, p["scales"], p["zps"], p["concat_params"])
        elif type_name == "QC2fCIB":
            current_x, _, _ = block_qc2f_cib(current_x, **p)
        
        saved_features[i] = current_x
        
        # Compare every layer
        ref_out = getattr(res, f"Layer{i}")()
        ref_out_signed = to_signed_int8(ref_out.int_repr().numpy())
        diff = np.abs(current_x.astype(np.int16) - ref_out_signed.astype(np.int16))
        match = (diff == 0).sum() / diff.size * 100
        print(f"Layer {i:<2} ({type_name:<8}): Match={match:6.2f}%, MaxDiff={np.max(diff)} LSB")
        if i == 22: break

if __name__ == "__main__":
    test_features()
