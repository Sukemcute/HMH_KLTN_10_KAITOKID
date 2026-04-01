"""
Convert golden data to RTL-compatible formats.

1. Input activations:  NCHW -> HWC (each row: W*C contiguous bytes)
2. Weights per layer:  OIHW (already correct, just concatenate & align)
3. Golden P3/P4/P5:    NCHW -> HWC for comparison

Invocation:
    python 01_export/convert_for_rtl.py --golden 02_golden_data/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from export_common import ensure_dir, write_hex_lines, uint8_hex_lines


def read_hex_file(path: Path) -> np.ndarray:
    """Read a hex file into a flat uint8 numpy array."""
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for i in range(0, len(line), 2):
                data.append(int(line[i:i+2], 16))
    return np.array(data, dtype=np.uint8)


def nchw_to_hwc(data: np.ndarray, shape: list[int]) -> np.ndarray:
    """Convert NCHW flat bytes to HWC layout."""
    n, c, h, w = shape
    tensor = data[:n * c * h * w].reshape(n, c, h, w)
    # Transpose: NCHW -> NHWC
    tensor_hwc = tensor.transpose(0, 2, 3, 1)  # [N, H, W, C]
    return tensor_hwc.reshape(-1)


def convert_input_act(golden_dir: Path, params: dict):
    """Convert input_act.hex from NCHW to HWC."""
    input_path = golden_dir / "input_act.hex"
    if not input_path.exists():
        print(f"  SKIP: {input_path} not found")
        return

    qi = params.get("quant_input", {})
    shape = qi.get("shape", [1, 3, 640, 640])
    data = read_hex_file(input_path)
    hwc = nchw_to_hwc(data, shape)
    out_path = golden_dir / "input_act_hwc.hex"
    write_hex_lines(out_path, uint8_hex_lines(hwc))
    print(f"  input_act_hwc.hex: {hwc.size} bytes, shape={shape} -> HWC")


def convert_golden_outputs(golden_dir: Path, params: dict):
    """Convert golden_P3/P4/P5 from NCHW to HWC."""
    for feat in params.get("golden_outputs", []):
        name = feat.get("name", "")
        shape = feat.get("shape", [])
        fname = feat.get("file", "")
        if not fname or len(shape) != 4:
            continue
        src = golden_dir / fname
        if not src.exists():
            continue
        data = read_hex_file(src)
        hwc = nchw_to_hwc(data, shape)
        out_name = fname.replace(".hex", "_hwc.hex")
        write_hex_lines(golden_dir / out_name, uint8_hex_lines(hwc))
        print(f"  {out_name}: {hwc.size} bytes from {fname}")


def convert_layer_activations(golden_dir: Path, params: dict):
    """Convert per-layer output activations from NCHW to HWC."""
    lbl_dir = golden_dir / "layer_by_layer"
    for layer in params.get("layers", []):
        out_meta = layer.get("output")
        if not isinstance(out_meta, dict):
            continue
        shape = out_meta.get("shape", [])
        if len(shape) != 4:
            continue
        idx = layer["index"]
        src = lbl_dir / f"act_L{idx:02d}_output.hex"
        if not src.exists():
            src = lbl_dir / f"L{idx:02d}_output.hex"
        if not src.exists():
            src = golden_dir / f"L{idx:02d}_output.hex"
        if not src.exists():
            continue
        data = read_hex_file(src)
        hwc = nchw_to_hwc(data, shape)
        out_path = lbl_dir / f"L{idx:02d}_output_hwc.hex"
        ensure_dir(str(lbl_dir))
        write_hex_lines(out_path, uint8_hex_lines(hwc))
        print(f"  L{idx:02d}_output_hwc.hex: {hwc.size} bytes")


def concat_all_weights(golden_dir: Path, params: dict):
    """Concatenate all layer weights into all_weights.hex with 32-byte alignment."""
    all_lines = []
    offsets = {}
    byte_cursor = 0

    for layer in params.get("layers", []):
        idx = layer["index"]
        for w in layer.get("weights", []):
            wfile = golden_dir / w["file"]
            if not wfile.exists():
                print(f"  WARNING: {wfile} not found")
                continue
            with open(wfile) as f:
                lines = [l.strip() for l in f if l.strip()]
            if idx not in offsets:
                offsets[idx] = byte_cursor
            all_lines.extend(lines)
            byte_cursor += len(lines) * 32
            # Align to 32 bytes
            remainder = byte_cursor % 32
            if remainder:
                pad = (32 - remainder) // 32 + 1
                all_lines.extend(["00" * 32] * pad)
                byte_cursor += pad * 32

    out = golden_dir / "all_weights.hex"
    with open(out, "w") as f:
        for line in all_lines:
            f.write(line + "\n")
    print(f"  all_weights.hex: {byte_cursor} bytes, {len(offsets)} layers")

    # Save offsets
    offsets_path = golden_dir / "weight_offsets.json"
    with open(offsets_path, "w") as f:
        json.dump({str(k): v for k, v in offsets.items()}, f, indent=2)
    return offsets


def main():
    parser = argparse.ArgumentParser(description="Convert golden data to RTL-compatible formats")
    parser.add_argument("--golden", required=True, help="Golden data directory")
    args = parser.parse_args()

    golden_dir = Path(args.golden).resolve()
    params_file = golden_dir / "quant_params.json"
    if not params_file.exists():
        print(f"ERROR: {params_file} not found. Run export_golden_data.py first.")
        return

    with open(params_file) as f:
        params = json.load(f)

    print("=== Converting golden data for RTL ===")

    print("\n[1/4] Converting input activations NCHW -> HWC...")
    convert_input_act(golden_dir, params)

    print("\n[2/4] Converting golden P3/P4/P5 outputs NCHW -> HWC...")
    convert_golden_outputs(golden_dir, params)

    print("\n[3/4] Converting per-layer activations NCHW -> HWC...")
    convert_layer_activations(golden_dir, params)

    print("\n[4/4] Concatenating all weights...")
    concat_all_weights(golden_dir, params)

    print("\nDone!")


if __name__ == "__main__":
    main()
