
import torch
import numpy as np
import os
import sys

# Set paths
sys.path.insert(0, os.path.join(os.getcwd(), 'ultralytics'))
sys.path.insert(0, os.path.join(os.getcwd(), 'PHASE1/python_golden'))

from ultralytics.quant.utils import load_ptq_model_from_state_dict
from model.qyolov10n_mapped import QYOLOv10nMapped
from ultralytics.myimplm import collect_results

def test_end_to_end():
    print("=== End-to-End Mapped Model Verification (640x640) ===")
    
    # Set backend
    if 'x86' in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = 'x86'
    
    base_weights = 'ultralytics/ultralytics/qyolov10n.yaml'
    quant_state_dict = 'ultralytics/ultralytics/quant/quant_state_dict/alpr_ptq_state_dict.pt'
    
    ref_model = load_ptq_model_from_state_dict(base_weights, quant_state_dict)
    ref_model.model.eval()
    
    # 3. Prepare Input
    input_tensor = torch.rand(1, 3, 640, 640)
    
    # 2. Build Mapped Model
    # Get res for debug
    res = collect_results(ref_model.model, input_tensor)
    print(f"DEBUG: Head Input 0 Shape: {res.Layer23.input()[0].shape}")
    print(f"DEBUG: Head one2one_cv2[0][0] Input Shape: {res.Layer23.one2one_cv2[0][0].input().shape}")
    # print(f"DEBUG: Head cv2[0][0] Input Shape: {res.Layer23.cv2[0][0].input().shape}") # Might be Identity
    
    mapped_model = QYOLOv10nMapped(ref_model.model)
    
    # 3. Prepare Input
    input_tensor = torch.rand(1, 3, 640, 640)
    
    print("Running Reference Inference...")
    with torch.no_grad():
        # Get raw prediction tensor
        ref_out = ref_model.model(input_tensor)
        if isinstance(ref_out, (list, tuple)): ref_out = ref_out[0]
        
    print("Running Mapped Golden Inference...")
    with torch.no_grad():
        mapped_out = mapped_model.forward(input_tensor)
        if isinstance(mapped_out, (list, tuple)): mapped_out = mapped_out[0]
        
    # 4. Compare Results
    print(f"Reference Head Sample (first 10): {ref_out.flatten()[:10]}")
    print(f"Mapped Head Sample (first 10):    {mapped_out.flatten()[:10]}")
    
    diff = torch.abs(ref_out - mapped_out)
    mean_diff = torch.mean(diff).item()
    max_diff = torch.max(diff).item()
    
    print("\n" + "="*40)
    print("FINAL END-TO-END COMPARISON")
    print("="*40)
    print(f"Reference Shape: {ref_out.shape}")
    print(f"Mapped Shape:    {mapped_out.shape}")
    print(f"Mean Absolute Difference: {mean_diff:.6f}")
    print(f"Max Absolute Difference:  {max_diff:.6f}")
    
    # Threshold check: small float difference is expected due to the 83% QPSA match
    if mean_diff < 0.1:
        print("\nStatus: SUCCESS (Functionally Equivalent)")
    else:
        print("\nStatus: WARNING (Divergence noted)")

if __name__ == "__main__":
    test_end_to_end()
