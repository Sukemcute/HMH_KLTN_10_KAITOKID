"""
Inference bằng model YOLOv10 đã quantization (PTQ).
Chạy: python run_quant_inference.py
Hoặc đổi IMAGE_PATH / OUTPUT_PATH bên dưới rồi chạy.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Ultralytics-dev"))

from ultralytics import YOLO
from ultralytics.quant import load_ptq_model_from_state_dict

# --- Cấu hình ---
BASE_WEIGHTS = ROOT / "Ultralytics-dev" / "ultralytics" / "qyolov10n.yaml"
QUANT_STATE_DICT = ROOT / "Ultralytics-dev" / "ultralytics" / "quant" / "quant_state_dict" / "qat_sttd.pt"
IMAGE_PATH = ROOT / "img1.jpg"   # Ảnh đầu vào (đổi nếu cần)
OUTPUT_PATH = ROOT / "output_quant.jpg"  # Ảnh kết quả
CONF_THRESHOLD = 0.25
DEVICE = "cpu"  # Model quantize chỉ chạy trên CPU


def main():
    if not BASE_WEIGHTS.exists():
        raise FileNotFoundError(f"Không tìm thấy: {BASE_WEIGHTS}")
    if not QUANT_STATE_DICT.exists():
        raise FileNotFoundError(f"Không tìm thấy: {QUANT_STATE_DICT}")
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {IMAGE_PATH}")

    print("Đang load model PTQ...")
    model = load_ptq_model_from_state_dict(
        base_weights=str(BASE_WEIGHTS),
        quant_state_dict=str(QUANT_STATE_DICT),
    )
    print("Đã load model xong.")

    print("Đang chạy inference...")
    results = model.predict(
        source=str(IMAGE_PATH),
        device=DEVICE,
        conf=CONF_THRESHOLD,
        verbose=False,
    )

    # Lấy kết quả ảnh đầu tiên
    r = results[0]
    boxes = r.boxes

    print(f"Số object phát hiện: {len(boxes)}")
    if len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().int().numpy()
        names = r.names
        for i in range(len(boxes)):
            name = names.get(int(cls[i]), str(cls[i]))
            print(f"  [{i+1}] {name}: conf={conf[i]:.2f}, box=[x1={xyxy[i][0]:.0f}, y1={xyxy[i][1]:.0f}, x2={xyxy[i][2]:.0f}, y2={xyxy[i][3]:.0f}]")

    # Lưu ảnh có vẽ box
    r.save(filename=str(OUTPUT_PATH))
    print(f"Đã lưu ảnh kết quả: {OUTPUT_PATH}")
    return results


if __name__ == "__main__":
    main()
