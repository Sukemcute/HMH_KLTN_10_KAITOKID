"""
Export PHASE 4 golden data from the quantized YOLOv10n model.

This script is the main entry point for producing:
  - input_act.hex
  - weights_L*_*.hex
  - bias_L*_*.hex
  - quant_params.json
  - silu_lut.hex
  - golden_P3.hex / golden_P4.hex / golden_P5.hex
  - golden_outputs.json

Recommended invocation from `PHASE_4/`:
  python 01_export/export_golden_data.py --image ../img1.jpg --output 02_golden_data/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from export_common import compute_bias_int32
from export_common import compute_silu_lut
from export_common import ensure_dir
from export_common import export_tensor_hex
from export_common import find_detect_feature_inputs
from export_common import int32_hex_lines
from export_common import iter_quantized_weight_modules
from export_common import load_quant_model
from export_common import preprocess_image
from export_common import quantized_weight_meta
from export_common import tensor_meta
from export_common import trace_quantized_model
from export_common import write_hex_lines
from export_common import write_json


def export_all(image_path: str | Path, output_dir: str | Path) -> dict:
    output_root = ensure_dir(output_dir)

    print("[1/6] Loading quantized model...")
    model = load_quant_model()

    print("[2/6] Preprocessing image...")
    image_tensor, letterbox_info = preprocess_image(image_path)
    write_json(output_root / "letterbox_info.json", letterbox_info)

    print("[3/6] Tracing quantized forward path...")
    trace = trace_quantized_model(model, image_tensor)
    quant_input = trace["quant_input"]
    traced_layers = trace["layers"]

    input_export = export_tensor_hex(output_root / "input_act.hex", quant_input)
    print(
        f"  QuantStub output: {input_export['shape']} -> {input_export['num_lines']} lines "
        f"(scale={quant_input.q_scale():.8f}, zp={quant_input.q_zero_point()})"
    )

    print("[4/6] Exporting quantized weights, bias and layer metadata...")
    quant_summary = {
        "image": letterbox_info,
        "input_tensor": tensor_meta(image_tensor),
        "quant_input": tensor_meta(quant_input),
        "files": {
            "input_act": "input_act.hex",
            "letterbox_info": "letterbox_info.json",
        },
        "layers": [],
    }

    total_weight_files = 0
    total_bias_files = 0

    for entry in traced_layers:
        layer_idx = entry["index"]
        layer_name = entry["name"]
        layer_input = entry["input"]
        layer_output = entry["output"]

        layer_record = {
            "index": layer_idx,
            "name": layer_name,
            "from": entry["from"],
            "save_output": entry["save_output"],
        }

        if isinstance(layer_input, torch.Tensor):
            layer_record["input"] = tensor_meta(layer_input)
        elif isinstance(layer_input, (list, tuple)):
            layer_record["input_list"] = [tensor_meta(t) for t in layer_input if isinstance(t, torch.Tensor)]

        if isinstance(layer_output, torch.Tensor):
            layer_record["output"] = tensor_meta(layer_output)
        elif isinstance(layer_output, (list, tuple)):
            layer_record["output_list"] = [tensor_meta(t) for t in layer_output if isinstance(t, torch.Tensor)]

        exported_weights = []
        for child_name, child_module, weight_tensor in iter_quantized_weight_modules(
            model.model.model[layer_idx]
        ):
            weight_file = f"weights_L{layer_idx}_{child_name}.hex"
            export_tensor_hex(output_root / weight_file, weight_tensor)
            weight_record = {
                "name": child_name,
                "file": weight_file,
                **quantized_weight_meta(weight_tensor),
            }
            total_weight_files += 1

            bias_int32 = None
            bias_file = None
            try:
                bias_tensor = child_module.bias()
            except Exception:
                bias_tensor = None

            if isinstance(layer_input, torch.Tensor):
                bias_int32 = compute_bias_int32(bias_tensor, layer_input, weight_tensor)
            elif isinstance(layer_input, (list, tuple)) and layer_input:
                first_qtensor = next((t for t in layer_input if isinstance(t, torch.Tensor) and t.is_quantized), None)
                bias_int32 = compute_bias_int32(bias_tensor, first_qtensor, weight_tensor)

            if bias_int32 is not None:
                bias_file = f"bias_L{layer_idx}_{child_name}.hex"
                write_hex_lines(output_root / bias_file, int32_hex_lines(bias_int32))
                weight_record["bias_file"] = bias_file
                weight_record["bias_shape"] = list(bias_int32.shape)
                total_bias_files += 1

            exported_weights.append(weight_record)

        if exported_weights:
            layer_record["weights"] = exported_weights
            print(f"  L{layer_idx:02d} {layer_name}: exported {len(exported_weights)} weight groups")

        quant_summary["layers"].append(layer_record)

    print("[5/6] Exporting SiLU LUT from QuantStub domain...")
    silu_lut = compute_silu_lut(float(quant_input.q_scale()), int(quant_input.q_zero_point()))
    export_tensor_hex(output_root / "silu_lut.hex", silu_lut, bytes_per_line=256)

    print("[6/6] Exporting golden P3/P4/P5 tensors...")
    detect_inputs = find_detect_feature_inputs(traced_layers)
    golden_outputs = []
    if detect_inputs is None:
        print("  Warning: could not locate 3 detect-head input features automatically.")
    else:
        for name, tensor in zip(("P3", "P4", "P5"), detect_inputs):
            export_info = export_tensor_hex(output_root / f"golden_{name}.hex", tensor)
            record = {
                "name": name,
                "file": f"golden_{name}.hex",
                **tensor_meta(tensor),
                "num_lines": export_info["num_lines"],
            }
            golden_outputs.append(record)
            print(f"  {name}: {record['shape']} -> {record['num_lines']} lines")

    quant_summary["golden_outputs"] = golden_outputs
    write_json(output_root / "golden_outputs.json", {"features": golden_outputs})
    write_json(output_root / "quant_params.json", quant_summary)

    print(f"\nExport completed: {output_root}")
    print(f"  Weight files: {total_weight_files}")
    print(f"  Bias files:   {total_bias_files}")
    print(f"  Layers:       {len(traced_layers)}")
    return quant_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export golden quantized tensors for PHASE 4")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, required=True, help="Output directory, e.g. 02_golden_data/")
    args = parser.parse_args()

    export_all(args.image, args.output)


if __name__ == "__main__":
    main()
