"""
Concatenate all per-layer weight hex files into a single all_weights.hex.

The weights are laid out contiguously at WEIGHT_BASE, each 32-byte aligned.
Also exports ppu_params.json with bias_int32, m_int, shift, zp_out, silu_lut
per layer for testbench loading.

Usage:
    python concat_weights.py --golden-dir 02_golden_data/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from export_common import ensure_dir


BYTES_PER_LINE = 32  # 256 bits = 32 bytes per hex line


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden-dir", required=True)
    args = parser.parse_args()

    golden = Path(args.golden_dir).resolve()
    qp_path = golden / "quant_params.json"
    if not qp_path.exists():
        raise FileNotFoundError(f"quant_params.json not found at {qp_path}")

    with qp_path.open() as f:
        params = json.load(f)

    all_lines = []
    weight_offsets = {}
    byte_cursor = 0

    for layer in params.get("layers", []):
        lid = layer["index"]
        for winfo in layer.get("weights", []):
            wfile = golden / winfo["file"]
            if not wfile.exists():
                print(f"  WARNING: {wfile} not found, skipping")
                continue

            with wfile.open() as f:
                lines = [l.strip() for l in f if l.strip()]

            if lid not in weight_offsets:
                weight_offsets[lid] = byte_cursor

            # Pad to 32-byte alignment if needed
            all_lines.extend(lines)
            byte_cursor += len(lines) * BYTES_PER_LINE
            remainder = byte_cursor % BYTES_PER_LINE
            if remainder:
                pad_lines = (BYTES_PER_LINE - remainder) // BYTES_PER_LINE + 1
                all_lines.extend(["00" * BYTES_PER_LINE] * pad_lines)
                byte_cursor += pad_lines * BYTES_PER_LINE

    out_path = golden / "all_weights.hex"
    with out_path.open("w") as f:
        for line in all_lines:
            f.write(line + "\n")

    print(f"Wrote {len(all_lines)} lines ({byte_cursor} bytes) to {out_path}")
    print(f"Weight offsets: {weight_offsets}")

    # Save weight offsets for descriptor generator
    offsets_path = golden / "weight_offsets.json"
    with offsets_path.open("w") as f:
        json.dump(
            {str(k): {"byte_offset": v, "hex_offset": hex(v)} for k, v in weight_offsets.items()},
            f,
            indent=2,
        )
    print(f"Wrote weight offsets to {offsets_path}")


if __name__ == "__main__":
    main()
