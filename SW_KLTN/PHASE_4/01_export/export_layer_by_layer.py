"""
Export detailed layer-by-layer tensors for PHASE 4 debugging.

Outputs inside `layer_by_layer/`:
  - quant_input.hex
  - act_Lxx_input.hex / act_Lxx_output.hex
  - act_Lxx_input_k.hex / act_Lxx_output_k.hex for list inputs/outputs
  - weight_Lxx_name.hex
  - layer_summary.json

Recommended invocation from `PHASE_4/`:
  python 01_export/export_layer_by_layer.py --image ../img1.jpg --output 02_golden_data/layer_by_layer/
"""

from __future__ import annotations

import argparse

import torch

from export_common import ensure_dir
from export_common import export_tensor_hex
from export_common import iter_quantized_weight_modules
from export_common import load_quant_model
from export_common import preprocess_image
from export_common import quantized_weight_meta
from export_common import sanitize_name
from export_common import tensor_meta
from export_common import trace_quantized_model
from export_common import write_json


def export_layer_view(image_path: str, output_dir: str) -> dict:
    out_dir = ensure_dir(output_dir)

    print("[1/4] Loading quantized model...")
    model = load_quant_model()

    print("[2/4] Preprocessing image...")
    image_tensor, letterbox_info = preprocess_image(image_path)

    print("[3/4] Replaying quantized forward pass layer-by-layer...")
    trace = trace_quantized_model(model, image_tensor)
    quant_input = trace["quant_input"]
    layers = trace["layers"]

    print("[4/4] Exporting per-layer tensors...")
    summary = {
        "image": letterbox_info,
        "quant_input": tensor_meta(quant_input),
        "layers": [],
    }

    quant_input_info = export_tensor_hex(out_dir / "quant_input.hex", quant_input)
    summary["quant_input"]["file"] = "quant_input.hex"
    summary["quant_input"]["num_lines"] = quant_input_info["num_lines"]
    print(f"  Quant input: {quant_input_info['shape']} -> {quant_input_info['num_lines']} lines")

    for entry in layers:
        layer_idx = entry["index"]
        layer_name = entry["name"]
        layer_input = entry["input"]
        layer_output = entry["output"]

        record = {
            "index": layer_idx,
            "name": layer_name,
            "from": entry["from"],
            "save_output": entry["save_output"],
        }

        if isinstance(layer_input, torch.Tensor):
            file_name = f"act_L{layer_idx:02d}_input.hex"
            export_info = export_tensor_hex(out_dir / file_name, layer_input)
            record["input"] = {**tensor_meta(layer_input), "file": file_name, "num_lines": export_info["num_lines"]}
        elif isinstance(layer_input, (list, tuple)):
            record["input_list"] = []
            for input_idx, input_tensor in enumerate(layer_input):
                if not isinstance(input_tensor, torch.Tensor):
                    continue
                file_name = f"act_L{layer_idx:02d}_input_{input_idx}.hex"
                export_info = export_tensor_hex(out_dir / file_name, input_tensor)
                record["input_list"].append(
                    {
                        "slot": input_idx,
                        **tensor_meta(input_tensor),
                        "file": file_name,
                        "num_lines": export_info["num_lines"],
                    }
                )

        if isinstance(layer_output, torch.Tensor):
            file_name = f"act_L{layer_idx:02d}_output.hex"
            export_info = export_tensor_hex(out_dir / file_name, layer_output)
            record["output"] = {**tensor_meta(layer_output), "file": file_name, "num_lines": export_info["num_lines"]}
        elif isinstance(layer_output, (list, tuple)):
            record["output_list"] = []
            for output_idx, output_tensor in enumerate(layer_output):
                if not isinstance(output_tensor, torch.Tensor):
                    continue
                file_name = f"act_L{layer_idx:02d}_output_{output_idx}.hex"
                export_info = export_tensor_hex(out_dir / file_name, output_tensor)
                record["output_list"].append(
                    {
                        "slot": output_idx,
                        **tensor_meta(output_tensor),
                        "file": file_name,
                        "num_lines": export_info["num_lines"],
                    }
                )

        weight_records = []
        for child_name, _child_module, weight_tensor in iter_quantized_weight_modules(model.model.model[layer_idx]):
            safe_child_name = sanitize_name(child_name)
            file_name = f"weight_L{layer_idx:02d}_{safe_child_name}.hex"
            export_info = export_tensor_hex(out_dir / file_name, weight_tensor)
            weight_records.append(
                {
                    "name": child_name,
                    "file": file_name,
                    "num_lines": export_info["num_lines"],
                    **quantized_weight_meta(weight_tensor),
                }
            )
        if weight_records:
            record["weights"] = weight_records

        summary["layers"].append(record)

        if "output" in record:
            output_shape = record["output"]["shape"]
            print(f"  L{layer_idx:02d} {layer_name:20s}: output {output_shape}")
        elif "output_list" in record:
            print(f"  L{layer_idx:02d} {layer_name:20s}: output list x{len(record['output_list'])}")
        else:
            print(f"  L{layer_idx:02d} {layer_name:20s}: exported")

    write_json(out_dir / "layer_summary.json", summary)
    print(f"\nExported {len(summary['layers'])} layer records to {out_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export layer-by-layer quantized tensors")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, required=True, help="Output directory, e.g. 02_golden_data/layer_by_layer/")
    args = parser.parse_args()

    export_layer_view(args.image, args.output)


if __name__ == "__main__":
    main()
