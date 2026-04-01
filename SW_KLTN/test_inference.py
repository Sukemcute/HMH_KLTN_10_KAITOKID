from ultralytics import YOLO
import torch
import cv2
import numpy as np

# Load the quantized model
from ultralytics.quant import load_ptq_model_from_state_dict

try:
    model = load_ptq_model_from_state_dict(
        base_weights = 'qyolov10n.yaml',
        quant_state_dict = './Ultralytics-dev/ultralytics/quant/quant_state_dict/qat_sttd.pt'
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Generate a dummy image to test inference
img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.imwrite("dummy.jpg", img)

try:
    print("Trying default predict()...")
    results = model.predict("dummy.jpg")
    print("Inference successful! Boxes:", len(results[0].boxes))
except Exception as e:
    print(f"Error during default inference: {e}")

