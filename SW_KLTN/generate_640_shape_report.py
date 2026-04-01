
import torch
import numpy as np
import os
import sys

# Set paths
sys.path.insert(0, os.path.join(os.getcwd(), 'ultralytics'))

from ultralytics.quant.utils import load_ptq_model_from_state_dict
from ultralytics.myimplm import collect_results

def main():
    base_weights = 'ultralytics/ultralytics/qyolov10n.yaml'
    quant_state_dict = 'ultralytics/ultralytics/quant/quant_state_dict/alpr_ptq_state_dict.pt'
    model = load_ptq_model_from_state_dict(base_weights, quant_state_dict)
    model.model.eval()
    
    input_tensor = torch.rand(1, 3, 640, 640)
    res = collect_results(model, input_tensor)
    
    print("# Layer Tensor Shapes (640x640 Inference)")
    print("| Layer | Module Type | Input Shape(s) | Output Shape |")
    print("| :--- | :--- | :--- | :--- |")
    
    for i, m in enumerate(model.model.model):
        node = getattr(res, f"Layer{i}")
        inp = node.input()
        out = node()
        
        type_name = type(m).__name__
        
        if isinstance(inp, (list, tuple)):
            inp_str = " + ".join([str(list(t.shape)) for t in inp])
        else:
            inp_str = str(list(inp.shape))
            
        if isinstance(out, (list, tuple)):
            out_str = " + ".join([str(list(t.shape)) for t in out if isinstance(t, torch.Tensor)])
        else:
            out_str = str(list(out.shape))
        
        print(f"| {i} | {type_name} | {inp_str} | {out_str} |")

if __name__ == "__main__":
    main()
