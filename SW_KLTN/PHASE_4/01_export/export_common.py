"""
Shared helpers for PHASE 4 export scripts.

These utilities keep the Python side consistent across:
  - export_golden_data.py
  - export_layer_by_layer.py
  - generate_descriptors.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
ULTRA_ROOT = ROOT / "Ultralytics-dev"
DEFAULT_BASE_WEIGHTS = ULTRA_ROOT / "ultralytics" / "qyolov10n.yaml"
DEFAULT_QUANT_STATE_DICT = ULTRA_ROOT / "ultralytics" / "quant" / "quant_state_dict" / "qat_sttd.pt"


def ensure_ultralytics_on_path() -> None:
    ultra_str = str(ULTRA_ROOT)
    if ultra_str not in sys.path:
        sys.path.insert(0, ultra_str)


def resolve_existing_path(path_str: str | Path) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


def ensure_dir(path_str: str | Path) -> Path:
    path = Path(path_str).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, data: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)


def sanitize_name(name: str) -> str:
    sanitized = []
    for ch in name:
        if ch.isalnum() or ch in ("_", "-"):
            sanitized.append(ch)
        else:
            sanitized.append("_")
    text = "".join(sanitized).strip("_")
    return text or "self"


def load_quant_model(
    base_weights: str | Path | None = None,
    quant_state_dict: str | Path | None = None,
):
    ensure_ultralytics_on_path()
    from ultralytics.quant import load_ptq_model_from_state_dict

    base = resolve_existing_path(base_weights or DEFAULT_BASE_WEIGHTS)
    quant_sd = resolve_existing_path(quant_state_dict or DEFAULT_QUANT_STATE_DICT)

    model = load_ptq_model_from_state_dict(
        base_weights=str(base),
        quant_state_dict=str(quant_sd),
    )
    model.model.eval()
    return model


def preprocess_image(image_path: str | Path, imgsz: int = 640, pad_value: int = 114):
    path = resolve_existing_path(image_path)
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    orig_h, orig_w = image_bgr.shape[:2]
    scale = min(imgsz / orig_h, imgsz / orig_w)
    resized_h = int(round(orig_h * scale))
    resized_w = int(round(orig_w * scale))
    resized = cv2.resize(image_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_h = imgsz - resized_h
    pad_w = imgsz - resized_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    letterboxed = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value),
    )
    image_rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).contiguous().float() / 255.0

    meta = {
        "image_path": str(path),
        "imgsz": imgsz,
        "orig_shape": [orig_h, orig_w],
        "resized_shape": [resized_h, resized_w],
        "letterbox_shape": [imgsz, imgsz],
        "scale": scale,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "pad_value": pad_value,
    }
    return tensor, meta


def uint8_hex_lines(arr: np.ndarray, bytes_per_line: int = 32) -> list[str]:
    flat = np.ascontiguousarray(arr).reshape(-1).astype(np.uint8)
    lines: list[str] = []
    for i in range(0, flat.size, bytes_per_line):
        chunk = flat[i : i + bytes_per_line]
        if chunk.size < bytes_per_line:
            chunk = np.pad(chunk, (0, bytes_per_line - chunk.size), constant_values=0)
        lines.append("".join(f"{int(v):02X}" for v in chunk))
    return lines


def int32_hex_lines(arr: np.ndarray, values_per_line: int = 8) -> list[str]:
    flat = np.ascontiguousarray(arr).reshape(-1).astype(np.int32).view(np.uint32)
    lines: list[str] = []
    for i in range(0, flat.size, values_per_line):
        chunk = flat[i : i + values_per_line]
        if chunk.size < values_per_line:
            chunk = np.pad(chunk, (0, values_per_line - chunk.size), constant_values=0)
        lines.append("".join(f"{int(v):08X}" for v in chunk))
    return lines


def write_hex_lines(path: str | Path, lines: list[str]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="ascii") as f:
        for line in lines:
            f.write(line + "\n")


def tensor_meta(tensor: torch.Tensor | None) -> dict[str, Any]:
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return {}

    meta: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "is_quantized": bool(tensor.is_quantized),
    }
    if tensor.is_quantized:
        int_repr = tensor.int_repr()
        meta["scale"] = float(tensor.q_scale())
        meta["zero_point"] = int(tensor.q_zero_point())
        meta["int_min"] = int(int_repr.min())
        meta["int_max"] = int(int_repr.max())
    else:
        tensor_f = tensor.detach().float()
        meta["float_min"] = float(tensor_f.min())
        meta["float_max"] = float(tensor_f.max())
    return meta


def export_tensor_hex(
    path: str | Path,
    tensor: torch.Tensor | np.ndarray,
    bytes_per_line: int = 32,
) -> dict[str, Any]:
    if isinstance(tensor, torch.Tensor):
        if tensor.is_quantized:
            arr = tensor.int_repr().cpu().numpy().astype(np.uint8)
        else:
            arr = tensor.detach().cpu().numpy()
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(np.rint(arr), -128, 255).astype(np.int16)
                arr = np.where(arr < 0, arr + 256, arr).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
    else:
        arr = np.asarray(tensor)
        if np.issubdtype(arr.dtype, np.signedinteger):
            arr = arr.astype(np.int16)
            arr = np.where(arr < 0, arr + 256, arr).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

    lines = uint8_hex_lines(arr, bytes_per_line=bytes_per_line)
    write_hex_lines(path, lines)
    return {
        "file": Path(path).name,
        "shape": list(arr.shape),
        "num_values": int(arr.size),
        "num_lines": len(lines),
        "bytes_per_line": bytes_per_line,
    }


def trace_quantized_model(model, image_tensor: torch.Tensor) -> dict[str, Any]:
    net = model.model
    save_indices = set(getattr(net, "save", []))
    traced_layers: list[dict[str, Any]] = []

    with torch.no_grad():
        x = net.quant(image_tensor)
        quant_input = x
        y: list[Any] = []

        for idx, module in enumerate(net.model):
            from_idx = getattr(module, "f", -1)
            module_idx = getattr(module, "i", idx)
            module_name = type(module).__name__

            if from_idx != -1:
                if isinstance(from_idx, int):
                    layer_input = y[from_idx]
                else:
                    layer_input = [x if j == -1 else y[j] for j in from_idx]
            else:
                layer_input = x

            layer_output = module(layer_input)
            x = layer_output
            y.append(x if module_idx in save_indices else None)

            traced_layers.append(
                {
                    "index": module_idx,
                    "name": module_name,
                    "from": from_idx,
                    "save_output": module_idx in save_indices,
                    "input": layer_input,
                    "output": layer_output,
                }
            )

    return {
        "quant_input": quant_input,
        "layers": traced_layers,
        "final_output": x,
    }


def iter_quantized_weight_modules(layer_module: torch.nn.Module):
    for child_name, child in layer_module.named_modules():
        if child_name == "":
            continue
        if not hasattr(child, "weight"):
            continue
        try:
            weight = child.weight()
        except Exception:
            continue
        if isinstance(weight, torch.Tensor) and weight.is_quantized:
            yield child_name, child, weight


def quantized_weight_meta(weight: torch.Tensor) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "shape": list(weight.int_repr().shape),
        "qscheme": str(weight.qscheme()).replace("torch.", ""),
    }
    if weight.qscheme() in (torch.per_channel_affine, torch.per_channel_affine_float_qparams):
        meta["scales"] = weight.q_per_channel_scales().cpu().numpy().tolist()
        meta["zero_points"] = weight.q_per_channel_zero_points().cpu().numpy().tolist()
        meta["axis"] = int(weight.q_per_channel_axis())
    else:
        meta["scale"] = float(weight.q_scale())
        meta["zero_point"] = int(weight.q_zero_point())
    return meta


def compute_bias_int32(
    bias_tensor: torch.Tensor | None,
    input_tensor: torch.Tensor | None,
    weight_tensor: torch.Tensor,
) -> np.ndarray | None:
    if bias_tensor is None or input_tensor is None:
        return None
    if not isinstance(input_tensor, torch.Tensor) or not input_tensor.is_quantized:
        return None

    bias_fp32 = bias_tensor.detach().cpu().numpy().astype(np.float64)
    input_scale = float(input_tensor.q_scale())

    if weight_tensor.qscheme() in (torch.per_channel_affine, torch.per_channel_affine_float_qparams):
        weight_scales = weight_tensor.q_per_channel_scales().cpu().numpy().astype(np.float64)
        if weight_scales.shape[0] != bias_fp32.shape[0]:
            weight_scales = np.resize(weight_scales, bias_fp32.shape[0])
    else:
        weight_scales = np.full_like(bias_fp32, float(weight_tensor.q_scale()), dtype=np.float64)

    denom = input_scale * weight_scales
    denom = np.where(denom == 0.0, 1.0, denom)
    return np.rint(bias_fp32 / denom).astype(np.int32)


def compute_silu_lut(scale: float, zero_point: int) -> np.ndarray:
    lut = np.zeros(256, dtype=np.uint8)
    for q in range(256):
        x = (q - zero_point) * scale
        silu = x / (1.0 + math.exp(-x))
        q_out = int(round(silu / scale) + zero_point)
        lut[q] = np.uint8(max(0, min(255, q_out)))
    return lut


def find_detect_feature_inputs(traced_layers: list[dict[str, Any]]) -> list[torch.Tensor] | None:
    for entry in reversed(traced_layers):
        layer_input = entry["input"]
        if not isinstance(layer_input, (list, tuple)) or len(layer_input) != 3:
            continue
        if all(isinstance(t, torch.Tensor) and t.ndim == 4 for t in layer_input):
            return list(layer_input)
    return None

