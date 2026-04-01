
import torch
import numpy as np
import os
import sys

# Set paths to include ultralytics and python_golden
sys.path.insert(0, os.path.join(os.getcwd(), 'ultralytics'))
sys.path.insert(0, os.path.join(os.getcwd(), 'PHASE1/python_golden'))

from ultralytics.quant.utils import load_ptq_model_from_state_dict
from model.qyolov10n_mapped import QYOLOv10nMapped

def val_mapped_model():
    print("=== End-to-End Validation of Mapped Golden Model ===")
    
    # 1. Configuration Paths
    base_weights = 'ultralytics/ultralytics/qyolov10n.yaml'
    quant_state_dict = '/home/hahuynh/Data/ProcessData/src_code/ultralytics/ultralytics/quant/quant_state_dict/alpr_ptq_state_dict.pt'
    data_config = 'ultralytics/ultralytics/data_config_local.yaml'
    
    if not os.path.exists(quant_state_dict):
        print(f"ERROR: quant_state_dict not found at {quant_state_dict}")
        # Fallback to local workspace path if absolute path fails
        quant_state_dict = 'ultralytics/ultralytics/quant/quant_state_dict/alpr_ptq_state_dict.pt'
        print(f"Trying local fallback: {quant_state_dict}")

    # 2. Load Reference Model
    print(f"Loading reference model...")
    ref_model = load_ptq_model_from_state_dict(
        base_weights = base_weights,
        quant_state_dict = quant_state_dict
    )
    ref_model.model.eval()
    
    # 3. Build Mapped Golden Model
    print("Initializing Mapped Golden Model (Backbone/Neck mapping)...")
    mapped_oracle = QYOLOv10nMapped(ref_model.model)
    
    # 4. Patch the reference model
    # We replace _predict_once to ensure our bit-accurate mapping is used.
    def golden_predict_once(x, profile=False, visualize=False, embed=None):
        # x is the float input tensor
        return mapped_oracle.forward(x)

    print("Patching DetectionModel._predict_once with Golden Oracle...")
    ref_model.model._predict_once = golden_predict_once
    
    # 5. Run Validation
    print("Starting validation on dataset...")
    # split='val' to use the validation set
    results = ref_model.val(data=data_config, imgsz=640, split='val')
    
    print("\n" + "="*40)
    print("MAPPED MODEL VALIDATION RESULTS")
    print("="*40)
    
    try:
        # Try to extract metrics safely
        map50 = results.results_dict.get('metrics/mAP50(B)', results.results_dict.get('mAP50', 0))
        map50_95 = results.results_dict.get('metrics/mAP50-95(B)', results.results_dict.get('mAP50-95', 0))
        print(f"mAP50:    {map50:.4f}")
        print(f"mAP50-95: {map50_95:.4f}")
    except Exception as e:
        print(f"Could not print results summary: {e}")
        print("Raw results dictionary:", results.results_dict)

if __name__ == "__main__":
    val_mapped_model()
