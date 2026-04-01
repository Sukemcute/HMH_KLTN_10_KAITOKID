"""
Re-run quantized inference and compare P3 / P4 / P5 to golden_*.hex.

Must use the same image as export_golden_data.py (path stored in letterbox_info.json).

Usage (from PHASE_4/):
  python 03_rtl_cosim/verify_full_model_outputs.py
  python 03_rtl_cosim/verify_full_model_outputs.py --golden-dir 02_golden_data/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PHASE4 = Path(__file__).resolve().parents[1]
_EXPORT = PHASE4 / "01_export"
if str(_EXPORT) not in sys.path:
    sys.path.insert(0, str(_EXPORT))

from export_common import find_detect_feature_inputs  # noqa: E402
from export_common import load_quant_model  # noqa: E402
from export_common import preprocess_image  # noqa: E402
from export_common import trace_quantized_model  # noqa: E402


def load_hex_uint8(path: Path) -> np.ndarray:
    vals: list[int] = []
    with open(path, "r", encoding="ascii") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for i in range(0, len(line), 2):
                vals.append(int(line[i : i + 2], 16))
    return np.array(vals, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden-dir", type=str, default=str(PHASE4 / "02_golden_data"))
    parser.add_argument("--image", type=str, default="", help="Override image path (default: letterbox_info.json)")
    args = parser.parse_args()
    gdir = Path(args.golden_dir).resolve()

    golden_json = gdir / "golden_outputs.json"
    if not golden_json.exists():
        raise FileNotFoundError(f"Run export_golden_data.py first: missing {golden_json}")

    with open(golden_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    features = meta["features"]

    image_path = args.image.strip() if args.image else ""
    if not image_path:
        letterbox = gdir / "letterbox_info.json"
        if letterbox.exists():
            with open(letterbox, "r", encoding="utf-8") as f:
                info = json.load(f)
            image_path = info.get("image_path") or ""

    if not image_path or not Path(str(image_path)).expanduser().exists():
        raise FileNotFoundError(
            "Missing image: add letterbox_info.json with image_path in golden-dir, or use --image PATH"
        )

    print("[1/3] Loading model + image...")
    try:
        model = load_quant_model()
    except Exception as ex:
        print(f"ERROR: load_quant_model failed: {ex}")
        print("Try: pip install dill")
        sys.exit(1)
    image_tensor, _ = preprocess_image(image_path)

    print("[2/3] Tracing forward (same as export)...")
    trace = trace_quantized_model(model, image_tensor)
    feats = find_detect_feature_inputs(trace["layers"])
    if feats is None:
        raise RuntimeError("Could not find 3 detect feature tensors (see export_common.find_detect_feature_inputs)")

    names = ("P3", "P4", "P5")
    print("[3/3] Compare to golden hex...")
    print("")
    all_ok = True
    for feat_meta, t in zip(features, feats):
        name = feat_meta["name"]
        fpath = gdir / feat_meta["file"]
        if not fpath.exists():
            print(f"  [MISS] {name}: file missing {fpath}")
            all_ok = False
            continue
        g = load_hex_uint8(fpath).reshape(feat_meta["shape"])
        p = t.int_repr().cpu().numpy().astype(np.uint8)
        match = int(np.sum(p == g))
        total = p.size
        if match == total:
            print(f"  [PASS] {name}: {match}/{total} bit-exact")
        else:
            all_ok = False
            diff = p.astype(np.int16) - g.astype(np.int16)
            print(f"  [FAIL] {name}: {match}/{total} exact, max|err|={np.max(np.abs(diff))}")

    print("")
    if all_ok:
        print("Full model head outputs (P3/P4/P5) match golden hex.")
    else:
        print("Some mismatches — re-run export_golden_data with same image/weights.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
