"""
CPU Postprocessing: Read P3/P4/P5 from RTL simulation output (hex files),
dequantize to float, run NMS, draw bounding boxes.

This simulates what the CPU does AFTER the FPGA IP finishes inference:
  1. Read P3/P4/P5 INT8 tensors from DDR3 (here from hex files)
  2. Dequantize: float_val = (int8_val - zero_point) * scale
  3. Decode boxes (xywh → xyxy)
  4. Run Non-Maximum Suppression (NMS)
  5. Scale boxes back to original image coordinates
  6. Draw bounding boxes and save output image

Usage (golden từ Python — kiểm tra pipeline CPU sau P3/P4/P5):
  cd PHASE_4/04_postprocess
  python cpu_postprocess.py --hex_dir ../02_golden_data --image ../../img1.jpg --output out_from_golden.jpg

Usage (sau khi có hex từ RTL sim / board — đặt cùng tên golden_P3.hex … trong một thư mục):
  python cpu_postprocess.py --hex_dir ../rtl_sim_dump --image ../../img1.jpg --output out_from_fpga.jpg

Shapes & quant lấy từ quant_params.json → golden_outputs (không hard-code 64/128/256).

Lưu ý: decode trong file này là bản đơn giản hóa; với YOLOv10 end-to-end nên dùng
model.predict() làm chuẩn, còn P3/P4/P5 từ IP thì ưu tiên so khớp tensor với golden
(verify_full_model_outputs.py / compare_golden_vs_rtl.py) trước khi tin bbox.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import cv2


def load_hex_to_int8(hex_path, shape=None):
    """Read hex file back to numpy INT8 array."""
    values = []
    with open(hex_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('#'):
                continue
            for i in range(0, len(line), 2):
                byte_hex = line[i:i + 2]
                val = int(byte_hex, 16)
                values.append(val)
    arr = np.array(values, dtype=np.uint8)
    if shape is not None:
        total = 1
        for s in shape:
            total *= s
        arr = arr[:total].reshape(shape)
    return arr


def dequantize(int8_arr, scale, zero_point):
    """Dequantize: float = (int8 - zp) * scale."""
    return (int8_arr.astype(np.float32) - zero_point) * scale


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def decode_yolov10_outputs(p3, p4, p5, num_classes=80):
    """
    Chỉ hợp lệ khi MỖI scale có cùng số kênh C = 4 + num_classes (tensor dạng detection thô).

    Golden YOLOv10n hiện tại: P3/P4/P5 là feature đầu vào detect head với C = 128 / 256 / 512
    (khác nhau giữa các scale) → KHÔNG thể np.concatenate theo trục anchor như code cũ.
    Trả về None → main() sẽ bỏ qua NMS hoặc dùng --boxes-from-model.
    """
    per_scale = []
    channels = []
    for feat, _stride in [(p3, 8), (p4, 16), (p5, 32)]:
        if feat.ndim != 4:
            return None
        b, c, h, w = feat.shape
        # [1, h*w, c]
        p = feat.reshape(b, c, h * w).transpose(0, 2, 1)
        per_scale.append(p)
        channels.append(c)

    if len(set(channels)) != 1:
        return None
    c0 = channels[0]
    if c0 != 4 + num_classes:
        return None
    return np.concatenate(per_scale, axis=1)  # [1, sum(HW), C]


def nms(boxes, scores, iou_threshold=0.45):
    """Standard NMS on numpy arrays."""
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def postprocess(predictions, conf_threshold=0.25, iou_threshold=0.45, num_classes=80):
    """
    Full postprocessing: decode, filter, NMS.
    predictions: [1, num_anchors, 4+num_classes]
    Returns: list of (x1, y1, x2, y2, conf, class_id)
    """
    if predictions is None or predictions.shape[1] == 0:
        return []

    preds = predictions[0]  # [num_anchors, 4+nc]

    # Box decode (assuming cx, cy, w, h format)
    box_preds = preds[:, :4]
    class_scores = sigmoid(preds[:, 4:])

    # Get max class score and class id
    max_scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    # Filter by confidence
    mask = max_scores > conf_threshold
    if not mask.any():
        return []

    filtered_boxes = box_preds[mask]
    filtered_scores = max_scores[mask]
    filtered_classes = class_ids[mask]

    # Convert cxcywh → xyxy
    cx, cy, w, h = filtered_boxes[:, 0], filtered_boxes[:, 1], filtered_boxes[:, 2], filtered_boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # Per-class NMS
    results = []
    for cls_id in np.unique(filtered_classes):
        cls_mask = filtered_classes == cls_id
        cls_boxes = boxes_xyxy[cls_mask]
        cls_scores = filtered_scores[cls_mask]
        keep = nms(cls_boxes, cls_scores, iou_threshold)
        for k in keep:
            results.append((*cls_boxes[k], cls_scores[k], int(cls_id)))

    results.sort(key=lambda x: x[4], reverse=True)
    return results


def scale_boxes_to_original(detections, letterbox_info, imgsz=640):
    """Scale detection boxes from letterbox coordinates to original image."""
    scale = letterbox_info["scale"]
    pad_top = letterbox_info["pad_top"]
    pad_left = letterbox_info["pad_left"]

    scaled = []
    for x1, y1, x2, y2, conf, cls_id in detections:
        x1 = (x1 - pad_left) / scale
        y1 = (y1 - pad_top) / scale
        x2 = (x2 - pad_left) / scale
        y2 = (y2 - pad_top) / scale
        scaled.append((x1, y1, x2, y2, conf, cls_id))
    return scaled


def draw_info_banner(image, lines):
    """Vài dòng chữ góc trên (BGR)."""
    y = 24
    for line in lines:
        cv2.putText(
            image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA
        )
        y += 26
    return image


def detections_from_ultralytics_predict(image_path: str, conf: float):
    """Dùng model quantized đầy đủ để lấy bbox (minh họa — không chứng minh IP)."""
    root = Path(__file__).resolve().parents[2]
    exp = root / "PHASE_4" / "01_export"
    if str(exp) not in sys.path:
        sys.path.insert(0, str(exp))
    from export_common import load_quant_model

    model = load_quant_model()
    results = model.predict(source=str(image_path), device="cpu", conf=conf, verbose=False)
    r = results[0]
    out = []
    if r.boxes is None or len(r.boxes) == 0:
        return out
    xyxy = r.boxes.xyxy.cpu().numpy()
    sc = r.boxes.conf.cpu().numpy()
    cl = r.boxes.cls.cpu().int().numpy()
    for i in range(len(r.boxes)):
        out.append((float(xyxy[i, 0]), float(xyxy[i, 1]), float(xyxy[i, 2]), float(xyxy[i, 3]), float(sc[i]), int(cl[i])))
    return out


def draw_boxes(image, detections, class_names=None):
    """Draw bounding boxes on image."""
    for x1, y1, x2, y2, conf, cls_id in detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = (0, 255, 255)  # cyan
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{int(cls_id)} {conf:.2f}"
        if class_names and int(cls_id) in class_names:
            label = f"{class_names[int(cls_id)]} {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image


def main():
    parser = argparse.ArgumentParser(description="CPU postprocessing for RTL output")
    parser.add_argument("--hex_dir", type=str, required=True, help="Directory with golden/RTL hex files")
    parser.add_argument("--image", type=str, required=True, help="Original input image")
    parser.add_argument("--output", type=str, default="output_rtl.jpg", help="Output image with boxes")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--boxes-from-model",
        action="store_true",
        help="Bỏ qua decode từ P3/P4/P5 hex; vẽ bbox bằng model.predict (cùng weights). "
        "Dùng để có ảnh minh họa; xác nhận IP vẫn cần so hex P3/P4/P5 với golden.",
    )
    args = parser.parse_args()

    hex_dir = Path(args.hex_dir)

    print("[1/5] Loading quant params...")
    params_path = hex_dir / "quant_params.json"
    if not params_path.exists():
        print(f"ERROR: Need {params_path} (copy from 02_golden_data next to your RTL hex).")
        return
    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    # P3/P4/P5: prefer golden_outputs inside quant_params.json (matches export_golden_data)
    golden_feats = params.get("golden_outputs")
    if not golden_feats:
        go = hex_dir / "golden_outputs.json"
        if go.exists():
            with open(go, "r", encoding="utf-8") as f:
                golden_feats = json.load(f).get("features", [])
    if not golden_feats:
        print("ERROR: quant_params.json missing 'golden_outputs' and no golden_outputs.json")
        return

    by_name = {item["name"]: item for item in golden_feats if "name" in item}
    for need in ("P3", "P4", "P5"):
        if need not in by_name:
            print(f"ERROR: golden metadata missing entry for {need}")
            return

    print("[2/5] Loading P3/P4/P5 hex files...")
    p3_spec, p4_spec, p5_spec = by_name["P3"], by_name["P4"], by_name["P5"]
    p3_shape = list(p3_spec["shape"])
    p4_shape = list(p4_spec["shape"])
    p5_shape = list(p5_spec["shape"])

    p3_data = load_hex_to_int8(hex_dir / p3_spec["file"])
    p4_data = load_hex_to_int8(hex_dir / p4_spec["file"])
    p5_data = load_hex_to_int8(hex_dir / p5_spec["file"])

    print(f"  P3: {p3_data.shape[0]} bytes, shape {p3_shape}")
    print(f"  P4: {p4_data.shape[0]} bytes, shape {p4_shape}")
    print(f"  P5: {p5_data.shape[0]} bytes, shape {p5_shape}")

    try:
        p3 = p3_data[: int(np.prod(p3_shape))].reshape(p3_shape)
        p4 = p4_data[: int(np.prod(p4_shape))].reshape(p4_shape)
        p5 = p5_data[: int(np.prod(p5_shape))].reshape(p5_shape)
    except ValueError as e:
        print(f"  ERROR: Could not reshape: {e}")
        return

    print("[3/5] Dequantizing...")
    p3_scale = float(p3_spec["scale"])
    p3_zp = int(p3_spec["zero_point"])
    p4_scale = float(p4_spec["scale"])
    p4_zp = int(p4_spec["zero_point"])
    p5_scale = float(p5_spec["scale"])
    p5_zp = int(p5_spec["zero_point"])

    p3_float = dequantize(p3, p3_scale, p3_zp)
    p4_float = dequantize(p4, p4_scale, p4_zp)
    p5_float = dequantize(p5, p5_scale, p5_zp)

    print(f"  P3 float range: [{p3_float.min():.3f}, {p3_float.max():.3f}]")
    print(f"  P4 float range: [{p4_float.min():.3f}, {p4_float.max():.3f}]")
    print(f"  P5 float range: [{p5_float.min():.3f}, {p5_float.max():.3f}]")

    print("[4/5] Detection / visualization...")
    detections = []
    if args.boxes_from_model:
        print("  Mode: --boxes-from-model (Ultralytics predict on image)")
        try:
            detections = detections_from_ultralytics_predict(args.image, args.conf)
        except Exception as ex:
            print(f"  ERROR: model.predict failed: {ex}")
            print("  Tip: pip install dill ; run from repo with Ultralytics-dev on path")
    else:
        predictions = decode_yolov10_outputs(p3_float, p4_float, p5_float)
        if predictions is None:
            c3, c4, c5 = p3_float.shape[1], p4_float.shape[1], p5_float.shape[1]
            print(
                f"  Skip simple decode: P3/P4/P5 channels differ ({c3},{c4},{c5}) "
                f"or are not 4+nc — đây là feature map trước head, không phải tensor 4+80."
            )
            print("  Để vẽ bbox: thêm --boxes-from-model  HOẶC triển khai decode đúng head YOLOv10.")
        else:
            detections = postprocess(predictions, conf_threshold=args.conf)
    print(f"  Detections to draw: {len(detections)}")

    print("[5/5] Drawing bounding boxes...")
    image = cv2.imread(str(args.image))
    if image is None:
        print(f"Cannot read image: {args.image}")
        return

    letterbox_path = hex_dir / "letterbox_info.json"
    if letterbox_path.exists() and detections and not args.boxes_from_model:
        with open(letterbox_path, "r", encoding="utf-8") as f:
            letterbox_info = json.load(f)
        detections = scale_boxes_to_original(detections, letterbox_info)
    # predict() trả box theo ảnh gốc — không scale letterbox

    if not detections and not args.boxes_from_model:
        image = draw_info_banner(
            image,
            [
                "P3/P4/P5: no simple bbox decode (128/256/512 ch)",
                "Confirm IP: compare hex vs golden",
                "Optional: --boxes-from-model for viz",
            ],
        )
    image = draw_boxes(image, detections)
    cv2.imwrite(str(args.output), image)
    print(f"  Saved: {args.output}")

    for i, (x1, y1, x2, y2, conf, cls_id) in enumerate(detections):
        print(f"  [{i+1}] class={int(cls_id)} conf={conf:.4f} box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")


if __name__ == "__main__":
    main()
