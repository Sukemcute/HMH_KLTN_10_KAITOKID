"""
Export per-layer PPU parameters for RTL co-simulation.

Produces per-layer files:
  ppu_bias_Lxx.hex   - INT32 bias values (32 per line, hex)
  ppu_m_int_Lxx.hex  - INT32 requant multiplier
  ppu_shift_Lxx.hex  - UINT6 shift values
  ppu_zp_out_Lxx.hex - INT8 output zero point
  silu_lut_Lxx.hex   - 256-entry INT8 SiLU LUT

Also produces ppu_params.json with metadata.

Invocation:
    python 01_export/export_ppu_params.py --image ../img1.jpg --output 02_golden_data/
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from export_common import (
    compute_bias_int32,
    compute_silu_lut,
    ensure_dir,
    iter_quantized_weight_modules,
    load_quant_model,
    preprocess_image,
    trace_quantized_model,
    write_hex_lines,
    write_json,
)

LANES = 32


def decompose_multiplier(m_float: float, max_shift: int = 31):
    """Decompose m_float into (m_int, shift) such that m_float ≈ m_int * 2^(-shift).

    Uses the standard TFLite/QNNPACK approach.
    """
    if m_float == 0.0:
        return 0, 0
    shift = 0
    while m_float < 0.5 and shift < max_shift:
        m_float *= 2.0
        shift += 1
    m_int = int(round(m_float * (1 << 30)))
    shift += 30
    if shift > max_shift:
        m_int >>= (shift - max_shift)
        shift = max_shift
    return m_int, shift


def compute_requant_params(
    input_scale: float,
    weight_scales: np.ndarray,
    output_scale: float,
    num_channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel (m_int, shift) requantization parameters."""
    m_ints = np.zeros(num_channels, dtype=np.int32)
    shifts = np.zeros(num_channels, dtype=np.int32)

    for ch in range(num_channels):
        w_scale = float(weight_scales[ch]) if ch < len(weight_scales) else float(weight_scales[-1])
        effective_scale = (input_scale * w_scale) / output_scale if output_scale != 0 else 0
        m, s = decompose_multiplier(effective_scale)
        m_ints[ch] = m
        shifts[ch] = s

    return m_ints, shifts


def int32_to_hex(arr: np.ndarray, values_per_line: int = 8) -> list[str]:
    flat = arr.flatten().astype(np.int32).view(np.uint32)
    lines = []
    for i in range(0, len(flat), values_per_line):
        chunk = flat[i:i + values_per_line]
        if len(chunk) < values_per_line:
            chunk = np.pad(chunk, (0, values_per_line - len(chunk)), constant_values=0)
        lines.append("".join(f"{int(v):08X}" for v in chunk))
    return lines


def uint8_to_hex(arr: np.ndarray, bytes_per_line: int = 32) -> list[str]:
    flat = arr.flatten().astype(np.uint8)
    lines = []
    for i in range(0, len(flat), bytes_per_line):
        chunk = flat[i:i + bytes_per_line]
        if len(chunk) < bytes_per_line:
            chunk = np.pad(chunk, (0, bytes_per_line - len(chunk)), constant_values=0)
        lines.append("".join(f"{int(v):02X}" for v in chunk))
    return lines


def pad_to_lanes(arr: np.ndarray, lanes: int = LANES) -> np.ndarray:
    if len(arr) >= lanes:
        return arr[:lanes]
    return np.pad(arr, (0, lanes - len(arr)), constant_values=0)


def export_ppu_params(image_path: str, output_dir: str) -> dict:
    out = ensure_dir(output_dir)
    print("[1/4] Loading quantized model...")
    model = load_quant_model()

    print("[2/4] Preprocessing and tracing...")
    image_tensor, _ = preprocess_image(image_path)
    trace = trace_quantized_model(model, image_tensor)
    traced = trace["layers"]
    quant_input = trace["quant_input"]

    print("[3/4] Computing PPU parameters per layer...")
    layer_params = []

    for entry in traced:
        idx = entry["index"]
        name = entry["name"]
        layer_input = entry["input"]
        layer_output = entry["output"]
        layer_module = model.model.model[idx]

        # Determine input/output scales and zero points
        input_scale, input_zp = None, None
        output_scale, output_zp = None, None

        if isinstance(layer_input, torch.Tensor) and layer_input.is_quantized:
            input_scale = float(layer_input.q_scale())
            input_zp = int(layer_input.q_zero_point())
        elif isinstance(layer_input, (list, tuple)):
            for t in layer_input:
                if isinstance(t, torch.Tensor) and t.is_quantized:
                    input_scale = float(t.q_scale())
                    input_zp = int(t.q_zero_point())
                    break

        if isinstance(layer_output, torch.Tensor) and layer_output.is_quantized:
            output_scale = float(layer_output.q_scale())
            output_zp = int(layer_output.q_zero_point())
        elif isinstance(layer_output, (list, tuple)):
            for t in layer_output:
                if isinstance(t, torch.Tensor) and t.is_quantized:
                    output_scale = float(t.q_scale())
                    output_zp = int(t.q_zero_point())
                    break

        # Find the FIRST quantized conv in this layer for weight scales
        weight_children = list(iter_quantized_weight_modules(layer_module))
        if not weight_children or input_scale is None or output_scale is None:
            layer_params.append({
                "index": idx, "name": name,
                "has_ppu_params": False,
            })
            continue

        child_name, child_mod, weight_tensor = weight_children[0]
        cout = weight_tensor.int_repr().shape[0]

        # Weight scales
        if weight_tensor.qscheme() in (torch.per_channel_affine, torch.per_channel_affine_float_qparams):
            w_scales = weight_tensor.q_per_channel_scales().cpu().numpy().astype(np.float64)
        else:
            w_scales = np.full(cout, float(weight_tensor.q_scale()), dtype=np.float64)

        # Bias INT32
        bias_tensor = None
        try:
            bias_tensor = child_mod.bias()
        except Exception:
            pass
        if isinstance(layer_input, torch.Tensor):
            bias_int32 = compute_bias_int32(bias_tensor, layer_input, weight_tensor)
        elif isinstance(layer_input, (list, tuple)):
            first_qt = next((t for t in layer_input if isinstance(t, torch.Tensor) and t.is_quantized), None)
            bias_int32 = compute_bias_int32(bias_tensor, first_qt, weight_tensor)
        else:
            bias_int32 = None

        if bias_int32 is None:
            bias_int32 = np.zeros(cout, dtype=np.int32)

        # Requant M_int, shift
        m_ints, shifts = compute_requant_params(input_scale, w_scales, output_scale, cout)

        # SiLU LUT (uses OUTPUT scale/zp for the layer)
        silu_lut = compute_silu_lut(output_scale, output_zp if output_zp else 0)

        # Pad to LANES for first 32 (hardware PPU), but export ALL channels
        bias_padded = pad_to_lanes(bias_int32.astype(np.int32))
        m_int_padded = pad_to_lanes(m_ints.astype(np.int32))
        shift_padded = pad_to_lanes(shifts.astype(np.int32))

        # Full-channel arrays for behavioral model (all cout channels)
        bias_full = bias_int32.astype(np.int32)
        m_int_full = m_ints.astype(np.int32)
        shift_full = shifts.astype(np.int32)

        # Write hex files — full channel count for behavioral model
        bias_file = f"ppu_bias_L{idx:02d}.hex"
        write_hex_lines(out / bias_file, int32_to_hex(bias_full))

        m_int_file = f"ppu_m_int_L{idx:02d}.hex"
        write_hex_lines(out / m_int_file, int32_to_hex(m_int_full))

        shift_file = f"ppu_shift_L{idx:02d}.hex"
        shift_u8 = np.clip(shift_full, 0, 63).astype(np.uint8)
        write_hex_lines(out / shift_file, uint8_to_hex(shift_u8))

        zp_file = f"ppu_zp_out_L{idx:02d}.hex"
        zp_val = np.array([output_zp if output_zp else 0], dtype=np.int8).view(np.uint8)
        write_hex_lines(out / zp_file, uint8_to_hex(zp_val))

        lut_file = f"silu_lut_L{idx:02d}.hex"
        write_hex_lines(out / lut_file, uint8_to_hex(silu_lut))

        layer_params.append({
            "index": idx,
            "name": name,
            "has_ppu_params": True,
            "cout": cout,
            "input_scale": input_scale,
            "output_scale": output_scale,
            "output_zp": output_zp,
            "bias_file": bias_file,
            "m_int_file": m_int_file,
            "shift_file": shift_file,
            "zp_file": zp_file,
            "lut_file": lut_file,
        })
        print(f"  L{idx:02d} {name}: cout={cout}, input_s={input_scale:.6f}, "
              f"output_s={output_scale:.6f}, output_zp={output_zp}")

    print("[4/4] Writing summary...")
    summary = {"layers": layer_params}
    write_json(out / "ppu_params.json", summary)
    print(f"Done. PPU params for {sum(1 for l in layer_params if l.get('has_ppu_params'))} layers.")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Export PPU parameters for RTL co-simulation")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output directory (same as golden data)")
    args = parser.parse_args()
    export_ppu_params(args.image, args.output)


if __name__ == "__main__":
    main()
