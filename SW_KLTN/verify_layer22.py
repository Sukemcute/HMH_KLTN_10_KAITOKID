
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

def test_layer22():
    print("=== Layer 22 Functional Verification (640x640) ===")
    
    # 1. Load Reference Model
    base_weights = 'ultralytics/ultralytics/qyolov10n.yaml'
    quant_state_dict = 'ultralytics/ultralytics/quant/quant_state_dict/alpr_ptq_state_dict.pt'
    ref_model = load_ptq_model_from_state_dict(base_weights, quant_state_dict)
    ref_model.model.eval()
    
    # 2. Build Mapped Model
    mapped_model = QYOLOv10nMapped(ref_model.model)
    
    # 3. Prepare Input
    input_tensor = torch.rand(1, 3, 640, 640)
    
    print("Running Reference Inference...")
    ref_res = collect_results(ref_model.model, input_tensor)
    ref_out_22 = ref_res.Layer22()
    ref_out_signed = to_signed_int8(ref_out_22.int_repr().numpy())
    
    print("Running Mapped Golden Inference...")
    # We need to capture Layer 22 from the mapped model.
    # Let's manually run the backbone through the mapped model's layers.
    
    # 0. Quantize Input
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
        if i == 22: break

    mapped_out_22 = current_x
    
    # 4. Compare Results
    diff = np.abs(mapped_out_22.astype(np.int16) - ref_out_signed.astype(np.int16))
    match_rate = (diff == 0).sum() / diff.size * 100
    max_lsb = np.max(diff)
    
    print("\n" + "="*40)
    print("LAYER 22 COMPARISON (END-OF-NECK)")
    print("="*40)
    print(f"Match Rate: {match_rate:.2f}%")
    print(f"Max Diff:   {max_lsb} LSB")
    
    if match_rate > 90:
        print("\nStatus: SUCCESS (Backbone/Neck logic is correct)")
    else:
        print("\nStatus: FAIL (Accumulated error is too high)")

if __name__ == "__main__":
    test_layer22()
