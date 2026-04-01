
import torch
import numpy as np
import os
import sys

# Set paths
sys.path.insert(0, os.path.join(os.getcwd(), 'ultralytics'))

from ultralytics.quant.utils import load_ptq_model_from_state_dict

def extract_l10_details():
    base_weights = 'ultralytics/ultralytics/qyolov10n.yaml'
    quant_state_dict = 'ultralytics/ultralytics/quant/quant_state_dict/qat_sttd.pt'
    model = load_ptq_model_from_state_dict(base_weights, quant_state_dict)
    model.model.eval()
    
    l10 = model.model.model[10]
    
    with open('Layer10_Params.txt', 'w') as f:
        f.write("=== Layer 10 (QPSA) Detail Extraction ===\n\n")
        
        f.write(f"PSA c: {l10.c}\n")
        f.write(f"PSA cv1: {l10.cv1.conv}\n")
        f.write(f"PSA cv2: {l10.cv2.conv}\n")
        
        attn = l10.attn
        f.write(f"\n--- Attention ---\n")
        f.write(f"num_heads: {attn.num_heads}\n")
        f.write(f"head_dim: {attn.head_dim}\n")
        f.write(f"key_dim: {attn.key_dim}\n")
        f.write(f"scale factor: {attn.scale}\n")
        f.write(f"qkv conv: {attn.qkv.conv}\n")
        f.write(f"proj conv: {attn.proj.conv}\n")
        f.write(f"pe conv: {attn.pe.conv}\n")
        f.write(f"softmax: {attn.sm}\n")
        f.write(f"mul_fn (matmul/add): {attn.mul_fn}\n")
        
        f.write(f"\n--- FFN ---\n")
        f.write(f"ffn[0]: {l10.ffn[0].conv}\n")
        f.write(f"ffn[1]: {l10.ffn[1].conv}\n")
        
        f.write(f"\n--- PSA Globals ---\n")
        f.write(f"fl (cat): {l10.fl}\n")

if __name__ == "__main__":
    extract_l10_details()
