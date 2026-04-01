"""
Verify RTL detection results against Python quantized model results.

Compares:
  1. The detection output (bounding boxes, confidence, class) from the
     Python quantized model (reference)
  2. The detection output from the RTL simulation (via cpu_postprocess.py)

Usage:
  python verify_detection.py \
      --reference ../../output_quant.jpg \
      --rtl_output output_rtl.jpg \
      --ref_json ref_detections.json \
      --rtl_json rtl_detections.json
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent


def run_python_reference(image_path):
    """Run the quantized model and return detection results."""
    sys.path.insert(0, str(ROOT / "Ultralytics-dev"))
    from ultralytics.quant import load_ptq_model_from_state_dict

    base_weights = ROOT / "Ultralytics-dev" / "ultralytics" / "qyolov10n.yaml"
    quant_sd = ROOT / "Ultralytics-dev" / "ultralytics" / "quant" / "quant_state_dict" / "qat_sttd.pt"

    model = load_ptq_model_from_state_dict(
        base_weights=str(base_weights),
        quant_state_dict=str(quant_sd),
    )

    results = model.predict(source=str(image_path), device="cpu", conf=0.25, verbose=False)
    r = results[0]
    boxes = r.boxes
    detections = []
    if len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().int().numpy()
        for i in range(len(boxes)):
            detections.append({
                "class": int(cls[i]),
                "conf": float(conf[i]),
                "box": [float(x) for x in xyxy[i]]
            })
    return detections


def compute_iou(box1, box2):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-8)


def compare_detections(ref_dets, rtl_dets, iou_threshold=0.5):
    """Compare two sets of detections."""
    print(f"\n{'='*60}")
    print(f"Detection Comparison: {len(ref_dets)} ref vs {len(rtl_dets)} rtl")
    print(f"{'='*60}")

    matched = 0
    matched_pairs = []
    used_rtl = set()

    for i, ref in enumerate(ref_dets):
        best_iou = 0
        best_j = -1
        for j, rtl in enumerate(rtl_dets):
            if j in used_rtl:
                continue
            if ref["class"] != rtl["class"]:
                continue
            iou = compute_iou(ref["box"], rtl["box"])
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_threshold and best_j >= 0:
            matched += 1
            used_rtl.add(best_j)
            matched_pairs.append((i, best_j, best_iou))
            conf_diff = abs(ref["conf"] - rtl_dets[best_j]["conf"])
            print(f"  MATCH: ref[{i}] ↔ rtl[{best_j}] | class={ref['class']} "
                  f"IoU={best_iou:.3f} conf_diff={conf_diff:.4f}")
        else:
            print(f"  MISS:  ref[{i}] class={ref['class']} conf={ref['conf']:.3f} "
                  f"box={[f'{x:.0f}' for x in ref['box']]} (best_iou={best_iou:.3f})")

    for j, rtl in enumerate(rtl_dets):
        if j not in used_rtl:
            print(f"  EXTRA: rtl[{j}] class={rtl['class']} conf={rtl['conf']:.3f}")

    precision = matched / len(rtl_dets) if rtl_dets else 0
    recall = matched / len(ref_dets) if ref_dets else 0
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"\n  Matched: {matched}/{len(ref_dets)}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")

    if matched == len(ref_dets) and matched == len(rtl_dets):
        print(f"\n  ★ PERFECT DETECTION MATCH ★")
    elif recall >= 0.9:
        print(f"\n  ◉ GOOD MATCH (recall >= 90%)")
    else:
        print(f"\n  ✗ SIGNIFICANT DIFFERENCES")

    return matched, len(ref_dets), len(rtl_dets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=str(ROOT / "img1.jpg"))
    parser.add_argument("--ref_json", type=str, default=None, help="Saved reference detections")
    parser.add_argument("--rtl_json", type=str, default=None, help="RTL detection results")
    args = parser.parse_args()

    # Get reference detections
    if args.ref_json and Path(args.ref_json).exists():
        with open(args.ref_json) as f:
            ref_dets = json.load(f)
    else:
        print("Running Python quantized model for reference...")
        ref_dets = run_python_reference(args.image)
        ref_path = Path(args.image).parent / "PHASE_4" / "02_golden_data" / "ref_detections.json"
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ref_path, 'w') as f:
            json.dump(ref_dets, f, indent=2)
        print(f"Saved reference detections to {ref_path}")

    print(f"\nReference detections ({len(ref_dets)}):")
    for i, d in enumerate(ref_dets):
        print(f"  [{i}] class={d['class']} conf={d['conf']:.4f} box={[f'{x:.0f}' for x in d['box']]}")

    # Get RTL detections (if available)
    if args.rtl_json and Path(args.rtl_json).exists():
        with open(args.rtl_json) as f:
            rtl_dets = json.load(f)
        compare_detections(ref_dets, rtl_dets)
    else:
        print("\nNo RTL detections available yet.")
        print("After RTL simulation and cpu_postprocess.py, run:")
        print("  python verify_detection.py --ref_json ref_detections.json --rtl_json rtl_detections.json")


if __name__ == "__main__":
    main()
