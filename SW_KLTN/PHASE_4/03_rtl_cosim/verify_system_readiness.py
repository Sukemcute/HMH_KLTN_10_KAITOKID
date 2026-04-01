"""
PHASE 4 - Step 4A: System readiness check (software-only, no Vivado).

Before tb_accel_top / tb_golden_check:
  - Required golden files exist; byte count matches golden_outputs in JSON
  - Descriptor hex: default WARN if missing; use --strict-descriptors to FAIL (RTL TB cần desc_*.hex)
  - Suggests next commands

From PHASE_4/:
  python 03_rtl_cosim/verify_system_readiness.py
  python 03_rtl_cosim/verify_system_readiness.py --golden-dir 02_golden_data
  python 03_rtl_cosim/verify_system_readiness.py --strict-descriptors
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PHASE4 = Path(__file__).resolve().parents[1]


def file_size(p: Path) -> int:
    return p.stat().st_size if p.is_file() else -1


def count_hex_bytes(path: Path) -> int:
    n = 0
    with open(path, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            for i in range(0, len(line), 2):
                if i + 2 <= len(line):
                    n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="PHASE 4 step 4A: system readiness (software only)")
    ap.add_argument("--golden-dir", type=str, default=str(PHASE4 / "02_golden_data"))
    ap.add_argument(
        "--strict-descriptors",
        action="store_true",
        help="Fail if desc_net/layers/tiles.hex missing (needed for tb_golden_check / DDR descriptors)",
    )
    args = ap.parse_args()
    gdir = Path(args.golden_dir).resolve()

    print("=" * 60)
    print("  PHASE 4 - Step 4A: System readiness (software-only)")
    print("=" * 60)
    print(f"  Golden dir: {gdir}\n")

    ok = True
    required = [
        "input_act.hex",
        "quant_params.json",
        "golden_P3.hex",
        "golden_P4.hex",
        "golden_P5.hex",
    ]
    for name in required:
        p = gdir / name
        if p.is_file():
            print(f"  [OK] {name}")
        else:
            print(f"  [MISS] {name}")
            ok = False

    qp = gdir / "quant_params.json"
    if not qp.is_file():
        sys.exit(1)

    with open(qp, "r", encoding="utf-8") as f:
        params = json.load(f)

    feats = params.get("golden_outputs")
    if not feats:
        print("\n  [WARN] quant_params.json missing 'golden_outputs'")
        ok = False
    else:
        print("\n  --- Tensor size vs hex byte count ---")
        for item in feats:
            name = item.get("name", "?")
            fn = item.get("file")
            shape = item.get("shape", [])
            if not fn or len(shape) != 4:
                continue
            exp = int(shape[0] * shape[1] * shape[2] * shape[3])
            hp = gdir / fn
            if not hp.is_file():
                print(f"  [MISS] {name}: {fn}")
                ok = False
                continue
            got = count_hex_bytes(hp)
            if got == exp:
                print(f"  [OK] {name} {shape}: {got} bytes")
            else:
                print(f"  [BAD] {name} {shape}: expected {exp} bytes, hex has {got}")
                ok = False

    descs = ["desc_net.hex", "desc_layers.hex", "desc_tiles.hex"]
    print("\n  --- Descriptor hex (for desc_fetch / DDR) ---")
    for d in descs:
        p = gdir / d
        if p.is_file():
            print(f"  [OK] {d}")
        else:
            print(f"  [WARN] {d} missing (run generate_descriptors.py)")
            if args.strict_descriptors:
                ok = False
                print(f"  [FAIL] strict-descriptors: {d} required")

    lbl = gdir / "layer_by_layer" / "layer_summary.json"
    print("\n  --- Layer-by-layer (debug) ---")
    if lbl.is_file():
        print(f"  [OK] {lbl.name}")
    else:
        print("  [WARN] layer_by_layer/ not exported (export_layer_by_layer.py)")

    print("\n  --- Next steps ---")
    print("  - Python vs golden:  python 03_rtl_cosim/verify_full_model_outputs.py")
    print("  - Complex blocks:    python 03_rtl_cosim/verify_complex_blocks.py --all")
    print("  - PPU RTL:           Vivado: source run_ppu_golden.tcl")
    print("  - Full SoC RTL:      source run_cosim.tcl -> tb_accel_top / tb_golden_check")
    print("  - After sim:         compare_golden_vs_rtl.py --rtl <dump.hex>")

    print("\n" + "=" * 60)
    if ok:
        print("  [PASS] 4A - Golden and metadata OK (ready to hook RTL).")
    else:
        print("  [FAIL] 4A - Fix missing files or byte count before system RTL test.")
    print("=" * 60 + "\n")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
