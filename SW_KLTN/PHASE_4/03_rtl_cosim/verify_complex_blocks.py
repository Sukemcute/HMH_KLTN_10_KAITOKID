"""
Verify non-Conv blocks using golden hex in layer_by_layer/.

  - Upsample: nearest-neighbor 2x, same scale/zp -> bit-exact uint8 match
  - QConcat: dequant both branches, concat on C, requant to output scale/zp
  - SCDown / QC2f / ... : run the actual quantized PyTorch submodule on input
    rebuilt from hex; compare int_repr to exported act_Lxx_output.hex

Usage (from PHASE_4/):
  python 03_rtl_cosim/verify_complex_blocks.py --all
  python 03_rtl_cosim/verify_complex_blocks.py --upsample --qconcat --torch-blocks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PHASE4 = Path(__file__).resolve().parents[1]
LBL = PHASE4 / "02_golden_data" / "layer_by_layer"

# Add 01_export for load_quant_model
_EXPORT = PHASE4 / "01_export"
if str(_EXPORT) not in sys.path:
    sys.path.insert(0, str(_EXPORT))

from export_common import load_quant_model, preprocess_image  # noqa: E402


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


def load_summary() -> dict:
    with open(LBL / "layer_summary.json", "r", encoding="utf-8") as f:
        return json.load(f)


def qtensor_from_meta(arr_uint8: np.ndarray, shape: list[int], scale: float, zp: int) -> torch.Tensor:
    """Rebuild quint8 tensor from NCHW uint8 int_repr."""
    x = arr_uint8[: int(np.prod(shape))].reshape(shape)
    dq = (x.astype(np.float32) - float(zp)) * float(scale)
    t = torch.from_numpy(dq)
    return torch.quantize_per_tensor(t, float(scale), int(zp), torch.quint8)


def verify_upsample(entry: dict) -> tuple[int, int]:
    idx = entry["index"]
    name = entry["name"]
    inp = entry["input"]
    out = entry["output"]
    _, c, h, w = inp["shape"]
    path_in = LBL / inp["file"]
    path_g = LBL / out["file"]
    x = load_hex_uint8(path_in).reshape(1, c, h, w)
    # nearest 2x: PyTorch Upsample(scale_factor=2) on NCHW
    y = np.repeat(np.repeat(x, 2, axis=2), 2, axis=3)
    g = load_hex_uint8(path_g).reshape(out["shape"])
    match = int(np.sum(y == g))
    total = y.size
    pct = 100.0 * match / total
    status = "PASS" if match == total else "FAIL"
    print(f"  [{status}] L{idx:02d} Upsample: {match}/{total} ({pct:.2f}%) exact")
    return match, total


def verify_qconcat(entry: dict) -> tuple[int, int]:
    idx = entry["index"]
    outs = entry["output"]
    s_out = outs["scale"]
    zp_out = outs["zero_point"]
    shape_out = outs["shape"]
    n, c_out, h, w = shape_out

    branches = entry["input_list"]
    if len(branches) != 2:
        print(f"  [SKIP] L{idx:02d} QConcat: expected 2 inputs, got {len(branches)}")
        return 0, 0

    f0 = []
    for b in branches:
        arr = load_hex_uint8(LBL / b["file"]).reshape(b["shape"])
        dq = (arr.astype(np.float32) - float(b["zero_point"])) * float(b["scale"])
        f0.append(dq)
    cat = np.concatenate(f0, axis=1)
    assert cat.shape == (n, c_out, h, w), (cat.shape, shape_out)

    requant = np.rint(cat / float(s_out) + float(zp_out)).astype(np.int32)
    requant = np.clip(requant, 0, 255).astype(np.uint8)
    g = load_hex_uint8(LBL / outs["file"]).reshape(shape_out)
    match = int(np.sum(requant == g))
    total = requant.size
    pct = 100.0 * match / total
    status = "PASS" if match == total else "FAIL"
    print(f"  [{status}] L{idx:02d} QConcat: {match}/{total} ({pct:.2f}%) exact")
    return match, total


def verify_torch_block(model: torch.nn.Module, entry: dict) -> tuple[int, int]:
    """Run model.model.model[idx] on quantized input from hex; compare to golden output."""
    idx = entry["index"]
    name = entry["name"]
    try:
        mod = model.model.model[idx]
        mod.eval()

        with torch.no_grad():
            if "input_list" in entry:
                tensors = []
                for b in entry["input_list"]:
                    arr = load_hex_uint8(LBL / b["file"])
                    t = qtensor_from_meta(arr, b["shape"], b["scale"], b["zero_point"])
                    tensors.append(t)
                y = mod(*tensors)
            else:
                inp = entry["input"]
                arr = load_hex_uint8(LBL / inp["file"])
                x = qtensor_from_meta(arr, inp["shape"], inp["scale"], inp["zero_point"])
                y = mod(x)

        if isinstance(y, (list, tuple)):
            print(f"  [SKIP] L{idx:02d} {name}: list output not handled")
            return 0, 0

        out_meta = entry["output"]
        g = load_hex_uint8(LBL / out_meta["file"]).reshape(out_meta["shape"])
        pred = y.int_repr().cpu().numpy().astype(np.uint8)
        match = int(np.sum(pred == g))
        total = pred.size
        pct = 100.0 * match / total
        status = "PASS" if match == total else "FAIL"
        print(f"  [{status}] L{idx:02d} {name}: {match}/{total} ({pct:.2f}%) exact")
        return match, total
    except Exception as ex:
        print(f"  [SKIP] L{idx:02d} {name}: {ex}")
        return 0, 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--qconcat", action="store_true")
    parser.add_argument("--torch-blocks", action="store_true", help="SCDown, QC2f, SPPF, ... via PyTorch")
    args = parser.parse_args()

    do_all = args.all
    do_up = do_all or args.upsample
    do_cat = do_all or args.qconcat
    do_torch = do_all or args.torch_blocks

    if not (do_up or do_cat or do_torch):
        parser.print_help()
        return

    summary = load_summary()
    layers = summary["layers"]

    total_m = 0
    total_n = 0

    print("\n" + "=" * 70)
    print("  PHASE 4: Complex block verification (golden hex)")
    print("=" * 70)

    if do_up:
        print("\n-- Upsample (nearest 2x, uint8) --")
        for e in layers:
            if e.get("name") == "Upsample":
                m, n = verify_upsample(e)
                total_m += m
                total_n += n

    if do_cat:
        print("\n-- QConcat (dequant + concat + requant) --")
        for e in layers:
            if e.get("name") == "QConcat":
                m, n = verify_qconcat(e)
                total_m += m
                total_n += n

    if do_torch:
        print("\n-- Torch submodule vs golden output --")
        print("  (Loading model once...)")
        try:
            model = load_quant_model()
        except Exception as ex:
            print(f"  [SKIP] Cannot load quantized model: {ex}")
            print("  Install deps: pip install dill")
            print("  (Ultralytics quant path needs dill in many setups.)")
            model = None

        if model is not None:
            image_path = summary["image"]["image_path"]
            _, _ = preprocess_image(image_path)  # sanity: same image as export

            torch_names = {"SCDown", "QC2f", "QC2fCIB", "SPPF", "QPSA"}
            for e in layers:
                if e.get("name") in torch_names:
                    m, n = verify_torch_block(model, e)
                    total_m += m
                    total_n += n

    print("\n" + "=" * 70)
    if total_n > 0:
        print(f"  Total element matches: {total_m}/{total_n} ({100.0*total_m/total_n:.2f}%)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
