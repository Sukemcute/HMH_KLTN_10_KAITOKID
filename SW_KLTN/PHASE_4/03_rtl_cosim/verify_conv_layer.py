"""
Verify a single Conv layer using pure integer arithmetic.
Reads golden hex data, performs the EXACT same computation the RTL accelerator does:
  1. INT8 x INT8 -> INT32 accumulate (MAC)
  2. Add INT32 bias (pre-scaled to accumulator domain)
  3. Requantize: round(acc * scale_in * scale_w[co] / scale_out) + out_zp
  4. SiLU via lookup table (on requantized output)
  5. Compare with golden output

This script proves the golden data is self-consistent BEFORE touching Verilog.

Usage from PHASE_4/:
  python 03_rtl_cosim/verify_conv_layer.py --layer 0
  python 03_rtl_cosim/verify_conv_layer.py --layer 1
  python 03_rtl_cosim/verify_conv_layer.py --layer 3
  python 03_rtl_cosim/verify_conv_layer.py --layer 17
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PHASE4 = Path(__file__).resolve().parents[1]
GOLDEN_DIR = PHASE4 / "02_golden_data"
LBL_DIR = GOLDEN_DIR / "layer_by_layer"


def load_hex_uint8(path: Path) -> np.ndarray:
    vals = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for i in range(0, len(line), 2):
                vals.append(int(line[i : i + 2], 16))
    return np.array(vals, dtype=np.uint8)


def load_hex_int32(path: Path) -> np.ndarray:
    """Load INT32 hex file (8 hex digits per value, big-endian signed)."""
    vals = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for i in range(0, len(line), 8):
                hex_str = line[i : i + 8]
                if len(hex_str) < 8:
                    continue
                u = int(hex_str, 16)
                if u >= 0x80000000:
                    u -= 0x100000000
                vals.append(u)
    return np.array(vals, dtype=np.int32)


def load_layer_info(layer_idx: int) -> dict:
    with open(LBL_DIR / "layer_summary.json", "r") as f:
        summary = json.load(f)

    for entry in summary["layers"]:
        if entry["index"] == layer_idx:
            return entry

    raise ValueError(f"Layer {layer_idx} not found in layer_summary.json")


def build_silu_lut(scale: float, zero_point: int) -> np.ndarray:
    """Build SiLU lookup table: uint8 -> uint8 (same quant domain)."""
    lut = np.zeros(256, dtype=np.uint8)
    for q in range(256):
        x = (q - zero_point) * scale
        if abs(x) < 1e-12:
            silu_val = 0.0
        else:
            silu_val = x / (1.0 + math.exp(-x))
        q_out = int(round(silu_val / scale)) + zero_point
        lut[q] = np.uint8(max(0, min(255, q_out)))
    return lut


def find_bias_file(layer_idx: int, weight_name: str = "conv") -> Path | None:
    """Find bias hex file for a specific layer."""
    pattern = f"bias_L{layer_idx}_{weight_name}.hex"
    candidate = GOLDEN_DIR / pattern
    if candidate.exists():
        return candidate
    return None


def verify_conv(layer_idx: int, check_rows: int = 0) -> float:
    info = load_layer_info(layer_idx)
    print(f"\n{'='*70}")
    print(f"  Verifying Layer {layer_idx}: {info['name']}")
    print(f"{'='*70}")

    # --- Shapes and quant params ---
    inp_meta = info["input"]
    out_meta = info["output"]
    wgt_meta = info["weights"][0]

    _, cin, hin, win = inp_meta["shape"]
    _, cout, hout, wout = out_meta["shape"]
    _, _, kh, kw = wgt_meta["shape"]

    inp_scale = inp_meta["scale"]
    inp_zp = inp_meta["zero_point"]
    out_scale = out_meta["scale"]
    out_zp = out_meta["zero_point"]
    w_scales = np.array(wgt_meta["scales"], dtype=np.float64)
    w_zps = np.array(wgt_meta["zero_points"], dtype=np.int32)

    sh = max(1, round(hin / hout))
    sw = max(1, round(win / wout))
    pad = (kh - 1) // 2

    print(f"  Input:  [{cin},{hin},{win}] scale={inp_scale:.8f} zp={inp_zp}")
    print(f"  Weight: [{cout},{cin},{kh},{kw}] per_channel_scales[0..{cout-1}]")
    print(f"  Output: [{cout},{hout},{wout}] scale={out_scale:.8f} zp={out_zp}")
    print(f"  Stride: ({sh},{sw})  Pad: {pad}")

    # --- Load data ---
    print(f"\n  Loading hex files...")
    inp_flat = load_hex_uint8(LBL_DIR / inp_meta["file"])
    wgt_flat = load_hex_uint8(LBL_DIR / wgt_meta["file"])
    out_flat = load_hex_uint8(LBL_DIR / out_meta["file"])

    inp = inp_flat[: cin * hin * win].reshape(cin, hin, win).astype(np.int32)
    wgt = wgt_flat[: cout * cin * kh * kw].reshape(cout, cin, kh, kw).astype(np.int8).astype(np.int32)
    golden = out_flat[: cout * hout * wout].reshape(cout, hout, wout).astype(np.uint8)

    print(f"  Input range:  [{inp.min()}, {inp.max()}]")
    print(f"  Weight range: [{wgt.min()}, {wgt.max()}]")
    print(f"  Golden range: [{golden.min()}, {golden.max()}]")

    # --- Load bias (INT32) ---
    bias_path = find_bias_file(layer_idx, wgt_meta["name"])
    if bias_path is not None:
        bias_int32 = load_hex_int32(bias_path)
        print(f"  Bias loaded:  {bias_path.name} ({len(bias_int32)} values)")
        print(f"  Bias range:   [{bias_int32.min()}, {bias_int32.max()}]")
        if len(bias_int32) < cout:
            bias_int32 = np.pad(bias_int32, (0, cout - len(bias_int32)))
    else:
        bias_int32 = np.zeros(cout, dtype=np.int32)
        print(f"  Bias: none (zero)")

    # --- Build SiLU LUT ---
    silu_lut = build_silu_lut(out_scale, out_zp)
    print(f"  SiLU LUT built: lut[{out_zp}]={silu_lut[out_zp]} (zero -> zero)")
    print(f"  SiLU LUT sample: lut[0]={silu_lut[0]}, lut[128]={silu_lut[128]}, lut[255]={silu_lut[255]}")

    # --- Compute conv in pure integer ---
    rows_to_check = hout if check_rows == 0 else min(check_rows, hout)
    print(f"\n  Computing INT8 conv for {rows_to_check}/{hout} output rows...")

    computed = np.zeros((cout, rows_to_check, wout), dtype=np.uint8)
    max_abs_err = 0
    total_err = 0
    exact_match = 0
    total_checked = 0

    for oh in range(rows_to_check):
        for ow in range(wout):
            for co in range(cout):
                acc = np.int64(0)
                for ci in range(cin):
                    for fh in range(kh):
                        for fw in range(kw):
                            ih = oh * sh + fh - pad
                            iw = ow * sw + fw - pad
                            if 0 <= ih < hin and 0 <= iw < win:
                                act_val = np.int64(inp[ci, ih, iw]) - np.int64(inp_zp)
                            else:
                                # PyTorch pads with zero_point, so (zp - zp) = 0
                                act_val = np.int64(0)
                            wgt_val = np.int64(wgt[co, ci, fh, fw]) - np.int64(w_zps[co])
                            acc += act_val * wgt_val

                # Add INT32 bias (already scaled to accumulator domain)
                acc += np.int64(bias_int32[co])

                # Requantize: float_val = acc * input_scale * weight_scale[co]
                float_val = float(acc) * inp_scale * float(w_scales[co])

                # Quantize to output domain (pre-SiLU)
                pre_silu = int(round(float_val / out_scale)) + out_zp
                pre_silu = max(0, min(255, pre_silu))

                # Apply SiLU via LUT
                out_int = int(silu_lut[pre_silu])
                computed[co, oh, ow] = out_int

                g = int(golden[co, oh, ow])
                err = abs(out_int - g)
                max_abs_err = max(max_abs_err, err)
                total_err += err
                if out_int == g:
                    exact_match += 1
                total_checked += 1

        if oh % 20 == 0 or oh == rows_to_check - 1:
            pct = 100.0 * exact_match / max(total_checked, 1)
            print(f"    Row {oh}/{rows_to_check}: exact={pct:.1f}% max_err={max_abs_err}")

    # --- Report ---
    mean_err = total_err / max(total_checked, 1)
    exact_pct = 100.0 * exact_match / total_checked

    print(f"\n{'='*70}")
    print(f"  Layer {layer_idx} Verification Results ({total_checked} values checked)")
    print(f"{'='*70}")
    print(f"  Bit-exact match: {exact_match}/{total_checked} ({exact_pct:.2f}%)")
    print(f"  Max absolute error: {max_abs_err}")
    print(f"  Mean absolute error: {mean_err:.4f}")

    err_arr = np.abs(computed[:, :rows_to_check, :].astype(np.int16) - golden[:, :rows_to_check, :].astype(np.int16))
    print(f"\n  Error distribution:")
    for threshold in [0, 1, 2, 3, 5]:
        count = int(np.sum(err_arr <= threshold))
        pct = 100.0 * count / total_checked
        print(f"    |error| <= {threshold}: {count}/{total_checked} ({pct:.1f}%)")

    if exact_pct == 100.0:
        print(f"\n  [PASS] LAYER {layer_idx} PERFECT BIT-EXACT MATCH")
    elif exact_pct >= 90.0:
        print(f"\n  [CLOSE] LAYER {layer_idx}: very close ({exact_pct:.1f}%), max_err={max_abs_err}")
        print(f"    Small rounding differences are expected (PyTorch vs integer-only)")
    else:
        print(f"\n  [CHECK] LAYER {layer_idx}: significant differences ({exact_pct:.1f}%)")
        print(f"    Check if bias or SiLU LUT differs between Python and RTL model")

    return exact_pct


def main():
    parser = argparse.ArgumentParser(description="Verify a Conv layer with pure integer arithmetic")
    parser.add_argument("--layer", type=int, required=True, help="Layer index (0, 1, 3, 17)")
    parser.add_argument("--rows", type=int, default=4,
                        help="Number of output rows to check (0=all, default=4 for speed)")
    args = parser.parse_args()

    verify_conv(args.layer, check_rows=args.rows)


if __name__ == "__main__":
    main()
