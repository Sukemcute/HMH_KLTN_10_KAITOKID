"""
Generate NET/LAYER/TILE descriptor files for PHASE 4.

Phase B version: correct DMA addresses, proper src_in_off/src_w_off/dst_off,
and ping-pong activation buffer management.

Reads `quant_params.json` and emits:
    desc_net.hex, desc_layers.hex, desc_tiles.hex, descriptor_summary.json

Invocation:
    python 01_export/generate_descriptors.py --output 02_golden_data/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from export_common import ensure_dir
from export_common import write_hex_lines
from export_common import write_json


DESC_BYTES = 64

# P3/P4/P5 detect head inputs come from these layers (YOLOv10n specific)
P3_LAYER = 16
P4_LAYER = 19
P5_LAYER = 22
DETECT_LAYER = 23


def load_quant_params(output_dir: Path, params_path: str | None) -> dict:
    candidate = Path(params_path).resolve() if params_path else (output_dir / "quant_params.json").resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            f"quant_params.json not found. Expected: {candidate}\n"
            "Run export_golden_data.py first."
        )
    with candidate.open("r", encoding="utf-8") as f:
        return json.load(f)


def guess_template_id(layer_name: str) -> int:
    """Map layer class name to PE mode (pe_mode_e encoding)."""
    name = layer_name.lower()
    if "detect" in name:
        return 1   # PE_OS1 (detect head uses 1x1 convs)
    if "concat" in name or "upsample" in name or "identity" in name:
        return 6   # PE_PASS
    if "pool" in name or "sppf" in name:
        return 4   # PE_MP5
    return 0       # PE_RS3 (default conv)


def guess_post_profile_id(layer_name: str, has_skip: bool = False) -> int:
    """Map layer to PPU post-processing profile.

    Profile IDs (must match subcluster_wrapper.sv lookup table):
      0 = Bias + PerChannel Requant + SiLU        (most Conv layers)
      1 = Bias + PerChannel Requant + No activation (detect head, pass-through)
      2 = Bias + PerChannel Requant + SiLU + Ewise  (C2f residual branches)
      3 = Bias + PerChannel Requant + ReLU
    """
    name = layer_name.lower()
    if "detect" in name:
        return 1
    if "concat" in name or "upsample" in name or "identity" in name:
        return 1
    if has_skip:
        return 2
    return 0


def _weight_bytes_for(w: dict) -> int:
    s = w.get("shape", [])
    if len(s) == 4:
        return s[0] * s[1] * s[2] * s[3]
    if len(s) == 2:
        return s[0] * s[1]
    return 0


def decompose_layer(item: dict, hin: int, win: int, hout: int, wout: int,
                    cin: int, cout: int) -> list[dict]:
    """Decompose a complex block into primitive sub-layer dicts.

    Each returned dict has: sub_name, cin_total, cout_total, hin, win, hout,
    wout, kh, kw, sh, sw, weight_bytes, post_profile_id, template_id,
    weight_files (list of hex filenames for this sub-conv).

    Returns a single-element list for simple Conv layers.
    """
    weights = item.get("weights", [])
    name = item["name"]
    name_lower = name.lower()

    # Sort weight entries by child name for predictable ordering
    weights_sorted = sorted(weights, key=lambda w: w.get("name", ""))

    # --- Simple Conv (0 or 1 weight) -------
    if len(weights) <= 1:
        wb = _weight_bytes_for(weights[0]) if weights else 0
        return [{
            "sub_name": name,
            "cin_total": cin, "cout_total": cout,
            "hin": hin, "win": win, "hout": hout, "wout": wout,
            "kh": 3 if cin > 1 else 1,
            "kw": 3 if cin > 1 else 1,
            "sh": max(1, round(hin / max(hout, 1))),
            "sw": max(1, round(win / max(wout, 1))),
            "weight_bytes": wb,
            "weight_files": [w.get("file", "") for w in weights],
            "post_profile_id": 0,
            "template_id": 0,
        }]

    # Build a lookup: child_name → weight_entry
    wmap = {}
    for w in weights:
        key = w.get("name", "").replace(".", "_").lower()
        wmap[key] = w

    # Helper: make a sub-layer dict from a weight entry
    def _make_sub(tag: str, w: dict, h_in: int, w_in: int, h_out: int, w_out: int,
                  profile: int = 0, tmpl: int = 0) -> dict:
        s = w.get("shape", [1, 1, 1, 1])
        c_out = s[0] if len(s) >= 1 else 1
        c_in  = s[1] if len(s) >= 2 else 1
        k_h   = s[2] if len(s) >= 3 else 1
        k_w   = s[3] if len(s) >= 4 else 1
        s_h = max(1, round(h_in / max(h_out, 1)))
        s_w = max(1, round(w_in / max(w_out, 1)))
        if k_h == 1 and k_w == 1:
            tmpl = 1  # PE_OS1 for 1x1 convs
        return {
            "sub_name": f"{name}.{tag}",
            "cin_total": c_in, "cout_total": c_out,
            "hin": h_in, "win": w_in, "hout": h_out, "wout": w_out,
            "kh": k_h, "kw": k_w, "sh": s_h, "sw": s_w,
            "weight_bytes": _weight_bytes_for(w),
            "weight_files": [w.get("file", "")],
            "post_profile_id": profile,
            "template_id": tmpl,
        }

    # Try to find weights by common YOLOv10 naming patterns
    def _find(pattern: str):
        for key, w in wmap.items():
            if pattern in key:
                return w
        return None

    # --- QC2f / QC2fCIB decomposition ---
    if "c2f" in name_lower or "c2fcib" in name_lower:
        cv1 = _find("cv1")
        cv2 = _find("cv2")
        # Bottleneck convs: m_0_cv1, m_0_cv2, etc.
        bneck_pairs = []
        for i in range(8):  # up to 8 bottleneck blocks
            b_cv1 = _find(f"m_{i}_cv1") or _find(f"m_{i}_0")
            b_cv2 = _find(f"m_{i}_cv2") or _find(f"m_{i}_1")
            if b_cv1:
                bneck_pairs.append((b_cv1, b_cv2))
            else:
                break

        subs = []
        if cv1:
            subs.append(_make_sub("cv1", cv1, hin, win, hout, wout, profile=0))
        for idx, (bcv1, bcv2) in enumerate(bneck_pairs):
            subs.append(_make_sub(f"m{idx}.cv1", bcv1, hout, wout, hout, wout, profile=0))
            if bcv2:
                subs.append(_make_sub(f"m{idx}.cv2", bcv2, hout, wout, hout, wout, profile=0))
        if cv2:
            subs.append(_make_sub("cv2", cv2, hout, wout, hout, wout, profile=0))

        if subs:
            subs[-1]["cout_total"] = cout  # last conv produces final output channels
            return subs

    # --- SCDown decomposition ---
    if "scdown" in name_lower:
        cv1 = _find("cv1")
        cv2 = _find("cv2")
        subs = []
        if cv1:
            subs.append(_make_sub("cv1", cv1, hin, win, hin, win, profile=0))
        if cv2:
            subs.append(_make_sub("cv2", cv2, hin, win, hout, wout, profile=0))
        if subs:
            return subs

    # --- SPPF decomposition ---
    if "sppf" in name_lower:
        cv1 = _find("cv1")
        cv2 = _find("cv2")
        subs = []
        if cv1:
            subs.append(_make_sub("cv1", cv1, hin, win, hin, win, profile=0))
        # MaxPool is handled as PE_MP5 pass-through (no weights)
        subs.append({
            "sub_name": f"{name}.maxpool_concat",
            "cin_total": cv1["shape"][0] if cv1 else cin,
            "cout_total": (cv1["shape"][0] if cv1 else cin) * 4,
            "hin": hin, "win": win, "hout": hin, "wout": win,
            "kh": 5, "kw": 5, "sh": 1, "sw": 1,
            "weight_bytes": 0,
            "weight_files": [],
            "post_profile_id": 1,
            "template_id": 4,  # PE_MP5
        })
        if cv2:
            subs.append(_make_sub("cv2", cv2, hin, win, hout, wout, profile=0))
        if subs:
            return subs

    # --- QPSA decomposition ---
    if "psa" in name_lower:
        subs = []
        for w in weights_sorted:
            wname = w.get("name", "unknown")
            tag = wname.replace(".", "_")
            subs.append(_make_sub(tag, w, hin, win, hout, wout, profile=0))
        if subs:
            return subs

    # --- Fallback: one sub-layer per weight ---
    subs = []
    for w in weights_sorted:
        wname = w.get("name", "unknown")
        tag = wname.replace(".", "_")
        subs.append(_make_sub(tag, w, hin, win, hout, wout, profile=0))
    return subs if subs else [{
        "sub_name": name,
        "cin_total": cin, "cout_total": cout,
        "hin": hin, "win": win, "hout": hout, "wout": wout,
        "kh": 1, "kw": 1, "sh": 1, "sw": 1,
        "weight_bytes": 0, "weight_files": [],
        "post_profile_id": 1, "template_id": 6,
    }]


def derive_hw_layers(params: dict) -> list[dict]:
    """Convert traced layer metadata into hardware layer descriptors.

    Complex blocks (QC2f, SCDown, SPPF, QPSA, QC2fCIB) are decomposed
    into primitive sub-layers.  Each sub-layer gets a unique sequential id.
    """
    layers = []
    hw_id = 0

    for item in params.get("layers", []):
        output_meta = item.get("output")
        if not isinstance(output_meta, dict):
            output_list = item.get("output_list", [])
            output_meta = output_list[0] if output_list else None
        if not output_meta or not isinstance(output_meta, dict):
            continue
        output_shape = output_meta.get("shape", [])
        if len(output_shape) != 4:
            continue

        input_meta = item.get("input")
        if not isinstance(input_meta, dict):
            input_list = item.get("input_list", [])
            input_meta = input_list[0] if input_list else {}

        input_shape = input_meta.get("shape", output_shape)
        if len(input_shape) != 4:
            input_shape = output_shape

        _, cin, hin, win = input_shape
        _, cout, hout, wout = output_shape

        tmpl_name = item["name"].lower()
        if tmpl_name in ("concat", "qconcat", "upsample", "identity"):
            kh, kw, sh, sw = 1, 1, 1, 1
            wb = 0
            layers.append({
                "id": hw_id,
                "orig_id": item["index"],
                "name": item["name"],
                "from": item.get("from", -1),
                "save_output": item.get("save_output", False),
                "template_id": guess_template_id(item["name"]),
                "post_profile_id": guess_post_profile_id(item["name"]),
                "cin_total": int(cin),
                "cout_total": int(cout),
                "hin": int(hin), "win": int(win),
                "hout": int(hout), "wout": int(wout),
                "kh": kh, "kw": kw, "sh": sh, "sw": sw,
                "weight_bytes": wb,
            })
            hw_id += 1
            continue

        sub_layers = decompose_layer(
            item, int(hin), int(win), int(hout), int(wout), int(cin), int(cout)
        )

        for si, sub in enumerate(sub_layers):
            is_last_sub = (si == len(sub_layers) - 1)
            layers.append({
                "id": hw_id,
                "orig_id": item["index"],
                "name": sub["sub_name"],
                "from": item.get("from", -1) if si == 0 else -1,
                "save_output": item.get("save_output", False) if is_last_sub else False,
                "template_id": sub["template_id"],
                "post_profile_id": sub["post_profile_id"],
                "cin_total": sub["cin_total"],
                "cout_total": sub["cout_total"],
                "hin": sub["hin"], "win": sub["win"],
                "hout": sub["hout"], "wout": sub["wout"],
                "kh": sub["kh"], "kw": sub["kw"],
                "sh": sub["sh"], "sw": sub["sw"],
                "weight_bytes": sub["weight_bytes"],
            })
            hw_id += 1

    return layers


def pack_fields(field_defs: list[tuple[int, int]], total_bits: int = DESC_BYTES * 8) -> bytes:
    value = 0
    used_bits = 0
    for width, field_value in field_defs:
        mask = (1 << width) - 1
        value = (value << width) | (field_value & mask)
        used_bits += width
    if used_bits > total_bits:
        raise ValueError(f"Descriptor uses {used_bits} bits > {total_bits}")
    return value.to_bytes(DESC_BYTES, byteorder="big", signed=False)


def descriptor_to_lines(desc_bytes: bytes) -> list[str]:
    if len(desc_bytes) != DESC_BYTES:
        raise ValueError("Descriptor must be 64 bytes")
    low_256 = desc_bytes[32:64]
    high_256 = desc_bytes[0:32]
    return [low_256.hex().upper(), high_256.hex().upper()]


def build_net_descriptor(memory_map: dict, num_layers: int) -> bytes:
    field_defs = [
        (16, 0xAC10),
        (8, 0x01),
        (8, num_layers),
        (64, memory_map["layer_table_base"]),
        (64, memory_map["weight_arena_base"]),
        (64, memory_map["act0_arena_base"]),
        (64, memory_map["act1_arena_base"]),
        (64, memory_map["aux_arena_base"]),
    ]
    return pack_fields(field_defs)


def compute_padding(layer: dict) -> tuple[int, int, int, int]:
    """Compute padding from kernel size (same-padding for stride-1, valid for others)."""
    kh, kw = layer["kh"], layer["kw"]
    sh, sw = layer["sh"], layer["sw"]
    tmpl = layer.get("template_id", 0)
    if tmpl == 6:  # PE_PASS
        return 0, 0, 0, 0
    pad_h = kh - 1
    pad_w = kw - 1
    pad_top = pad_h // 2
    pad_bot = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return pad_top, pad_bot, pad_left, pad_right


def build_layer_descriptor(layer: dict, tile_table_offset: int, num_tiles: int) -> bytes:
    tile_cin = min(layer["cin_total"], 255)
    tile_cout = min(layer["cout_total"], 255)
    tile_w_blks = max(1, (layer["wout"] + 31) // 32)
    # For behavioral model: single pass (model handles all channels internally)
    num_cin_pass = 1
    num_k_pass = 1
    pad_top, pad_bot, pad_left, pad_right = compute_padding(layer)

    # Field order MUST match desc_pkg.sv layer_desc_t struct packed (MSB first)
    field_defs = [
        (4,  layer["template_id"]),     # template_id
        (5,  layer["id"] & 0x1F),       # layer_id
        (9,  layer["cin_total"]),        # cin_total
        (9,  layer["cout_total"]),       # cout_total
        (10, layer["hin"]),              # hin
        (10, layer["win"]),              # win
        (10, layer["hout"]),             # hout
        (10, layer["wout"]),             # wout
        (4,  layer["kh"]),              # kh
        (4,  layer["kw"]),              # kw
        (3,  layer["sh"]),              # sh
        (3,  layer["sw"]),              # sw
        (4,  pad_top),                  # pad_top
        (4,  pad_bot),                  # pad_bot
        (4,  pad_left),                 # pad_left
        (4,  pad_right),                # pad_right
        (8,  tile_cin),                 # tile_cin
        (8,  tile_cout),                # tile_cout
        (6,  tile_w_blks),              # tile_w_blks
        (12, num_tiles),                # num_tile_hw
        (4,  0),                        # r_need
        (4,  0),                        # r_new
        (4,  0),                        # q_in
        (4,  0),                        # q_out
        (4,  num_cin_pass),             # num_cin_pass
        (4,  num_k_pass),               # num_k_pass
        (8,  0),                        # router_profile_id
        (8,  layer.get("post_profile_id", 0)),  # post_profile_id
        (5,  0),                        # src_in_tid
        (5,  0),                        # src_w_tid
        (5,  0),                        # src_skip_tid
        (5,  0),                        # dst_tid
        (64, tile_table_offset),        # tile_table_offset
        (16, 0x0001),                   # layer_flags
    ]
    return pack_fields(field_defs)


def build_tile_descriptor(
    layer: dict,
    tile_id: int,
    h_out0: int,
    wblk0: int,
    valid_h: int,
    valid_w: int,
    *,
    num_tiles_in_layer: int,
    src_in_off: int,
    src_w_off: int,
    src_skip_off: int,
    dst_off: int,
    has_skip: bool = False,
    global_tile_seq: int = 0,
) -> bytes:
    first_tile = 1 if tile_id == 0 else 0
    last_tile = 1 if tile_id == num_tiles_in_layer - 1 else 0
    # bit0: first_tile, bit1: last_tile, bit3: has_skip, bit4: need_swizzle, bit5: need_spill
    tile_flags = first_tile | (last_tile << 1) | (int(has_skip) << 3) | (1 << 5)

    # Compute halos from kernel/stride/padding
    kh, kw = layer["kh"], layer["kw"]
    sh, sw = layer["sh"], layer["sw"]
    halo_top = min((kh - 1) // 2, h_out0 * sh)
    halo_bot = (kh - 1) - halo_top
    halo_left = min((kw - 1) // 2, wblk0 * sw)
    halo_right = (kw - 1) - halo_left

    # Route each tile to exactly one SC via round-robin
    sc_mask_val = 1 << (global_tile_seq % 4)

    field_defs = [
        (16, tile_id),
        (5, layer["id"]),
        (4, sc_mask_val),
        (10, h_out0),
        (10, wblk0),
        (9, 0),
        (9, 0),
        (6, valid_h),
        (6, valid_w),
        (4, halo_top),
        (4, halo_bot),
        (4, halo_left),
        (4, halo_right),
        (32, src_in_off & 0xFFFFFFFF),
        (32, src_w_off & 0xFFFFFFFF),
        (32, src_skip_off & 0xFFFFFFFF),
        (32, dst_off & 0xFFFFFFFF),
        (10, 0),
        (10, 0),
        (10, h_out0),
        (10, 0),
        (4, 0),
        (4, 1),
        (4, 0),
        (4, 1),
        (16, tile_flags),
    ]
    return pack_fields(field_defs)


def generate_tiles_for_layer(layer: dict, tile_h: int, tile_w: int) -> list[dict]:
    tiles = []
    tile_id = 0
    for h0 in range(0, layer["hout"], tile_h):
        for w0 in range(0, layer["wout"], tile_w):
            valid_h = min(tile_h, layer["hout"] - h0)
            valid_w = min(tile_w, layer["wout"] - w0)
            tiles.append({
                "tile_id": tile_id,
                "h_out0": h0,
                "wblk0": w0 // max(1, tile_w),
                "valid_h": valid_h,
                "valid_w": valid_w,
            })
            tile_id += 1
    return tiles


def compute_act_size(layer: dict, is_input: bool = True) -> int:
    """Compute byte size of the activation tensor (NCHW, INT8)."""
    if is_input:
        return layer["cin_total"] * layer["hin"] * layer["win"]
    else:
        return layer["cout_total"] * layer["hout"] * layer["wout"]


def emit_descriptor_file(path: Path, descriptors: list[bytes]) -> int:
    lines = []
    for desc in descriptors:
        lines.extend(descriptor_to_lines(desc))
    write_hex_lines(path, lines)
    return len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate descriptor hex files from exported quant metadata"
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--params", default=None, help="Optional path to quant_params.json")
    parser.add_argument("--tile-h", type=int, default=32, help="Output tile height")
    parser.add_argument("--tile-w", type=int, default=32, help="Output tile width")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output)
    params = load_quant_params(output_dir, args.params)
    layers = derive_hw_layers(params)

    memory_map = {
        "net_desc_base":     0x0000_0000,
        "layer_table_base":  0x0000_0100,
        "tile_table_base":   0x0001_0000,
        "weight_arena_base": 0x0010_0000,
        "input_base":        0x0100_0000,    # must match TB INPUT_BASE
        "act0_arena_base":   0x0100_0000,
        "act1_arena_base":   0x0180_0000,
        "aux_arena_base":    0x01F0_0000,
        "output_arena_base": 0x0200_0000,    # P3||P4||P5 contiguous
    }

    # Build weight offset table: accumulate byte offsets in weight_arena
    weight_offsets = {}
    w_cursor = memory_map["weight_arena_base"]
    for layer in layers:
        weight_offsets[layer["id"]] = w_cursor
        w_cursor += layer["weight_bytes"]
        w_cursor = (w_cursor + 31) & ~31  # align to 32B

    # Build activation buffer plan: ping-pong between act0 and act1.
    act_base = [memory_map["act0_arena_base"], memory_map["act1_arena_base"]]
    input_base = memory_map["input_base"]

    layer_input_base = {}
    layer_output_base = {}
    aux_cursor = memory_map["aux_arena_base"]

    # Identify P3/P4/P5 by orig_id (the LAST sub-layer of the original block)
    p3_p4_p5_hw_ids = set()
    for layer in layers:
        if layer.get("orig_id") in (P3_LAYER, P4_LAYER, P5_LAYER) and layer["save_output"]:
            p3_p4_p5_hw_ids.add(layer["id"])

    for i, layer in enumerate(layers):
        lid = layer["id"]
        if i == 0:
            layer_input_base[lid] = input_base
        else:
            prev_id = layers[i - 1]["id"]
            layer_input_base[lid] = layer_output_base.get(prev_id, act_base[i % 2])

        if lid in p3_p4_p5_hw_ids:
            layer_output_base[lid] = memory_map["output_arena_base"]
        elif layer["save_output"]:
            layer_output_base[lid] = aux_cursor
            act_size = compute_act_size(layer, is_input=False)
            aux_cursor += (act_size + 31) & ~31
        else:
            layer_output_base[lid] = act_base[(i + 1) % 2]

    # Compute P3/P4/P5 output offsets within output_arena
    p3_offset = 0
    p4_offset = 0
    p5_offset = 0
    for layer in layers:
        oid = layer.get("orig_id", layer["id"])
        if oid == P3_LAYER and layer["save_output"]:
            p3_offset = 0
            p4_offset = compute_act_size(layer, is_input=False)
        elif oid == P4_LAYER and layer["save_output"]:
            p5_offset = p4_offset + compute_act_size(layer, is_input=False)

    p_offsets_hw = {}
    for layer in layers:
        oid = layer.get("orig_id", layer["id"])
        if oid == P3_LAYER and layer["save_output"]:
            p_offsets_hw[layer["id"]] = p3_offset
        elif oid == P4_LAYER and layer["save_output"]:
            p_offsets_hw[layer["id"]] = p4_offset
        elif oid == P5_LAYER and layer["save_output"]:
            p_offsets_hw[layer["id"]] = p5_offset

    print("=== Generating descriptors (Phase B) ===")
    print(f"  Layers: {len(layers)}")

    net_desc = build_net_descriptor(memory_map, len(layers))
    net_lines = emit_descriptor_file(output_dir / "desc_net.hex", [net_desc])

    layer_descs = []
    tile_descs = []
    tile_summary = []
    tile_cursor_bytes = memory_map["tile_table_base"]
    global_tile_seq = 0

    for layer in layers:
        lid = layer["id"]
        tiles = generate_tiles_for_layer(layer, args.tile_h, args.tile_w)
        n_tiles = len(tiles)
        layer_descs.append(build_layer_descriptor(layer, tile_cursor_bytes, n_tiles))

        in_base = layer_input_base.get(lid, act_base[0])
        w_base = weight_offsets.get(lid, memory_map["weight_arena_base"])
        out_base = layer_output_base.get(lid, act_base[1])

        if lid in p_offsets_hw:
            out_base = memory_map["output_arena_base"] + p_offsets_hw[lid]

        for tile in tiles:
            h0 = tile["h_out0"]
            w0_pixels = tile["wblk0"] * args.tile_w
            vh = tile["valid_h"]
            vw = tile["valid_w"]

            # NCHW byte offset within the activation tensor for this tile's
            # spatial region.  For a single DMA that loads full rows of the
            # tile's height range (all channels), we compute the offset to the
            # first byte of the first needed input row.
            sh, sw = layer["sh"], layer["sw"]
            h_in_start = max(0, h0 * sh - (layer["kh"] - 1) // 2)
            w_in_start = max(0, w0_pixels * sw - (layer["kw"] - 1) // 2)
            src_in_off = in_base + h_in_start * layer["win"] * layer["cin_total"]

            # Output NCHW offset: each tile writes to its position in the output tensor
            dst_off = out_base + h0 * layer["wout"] * layer["cout_total"]

            tile_descs.append(
                build_tile_descriptor(
                    layer,
                    tile["tile_id"],
                    tile["h_out0"],
                    tile["wblk0"],
                    tile["valid_h"],
                    tile["valid_w"],
                    num_tiles_in_layer=n_tiles,
                    src_in_off=src_in_off,
                    src_w_off=w_base,
                    src_skip_off=0,
                    dst_off=dst_off,
                    global_tile_seq=global_tile_seq,
                )
            )
            global_tile_seq += 1

        tile_summary.append({
            "layer_id": lid,
            "name": layer["name"],
            "num_tiles": len(tiles),
            "tile_table_offset": tile_cursor_bytes,
            "output_shape": [layer["cout_total"], layer["hout"], layer["wout"]],
            "in_base": hex(in_base),
            "w_base": hex(w_base),
            "out_base": hex(out_base),
        })
        tile_cursor_bytes += len(tiles) * DESC_BYTES

    layer_lines = emit_descriptor_file(output_dir / "desc_layers.hex", layer_descs)
    tile_lines = emit_descriptor_file(output_dir / "desc_tiles.hex", tile_descs)

    summary = {
        "memory_map": memory_map,
        "tile_h": args.tile_h,
        "tile_w": args.tile_w,
        "num_net_desc": 1,
        "num_layer_desc": len(layer_descs),
        "num_tile_desc": len(tile_descs),
        "net_lines": net_lines,
        "layer_lines": layer_lines,
        "tile_lines": tile_lines,
        "layers": layers,
        "tile_summary": tile_summary,
        "weight_offsets": {str(k): hex(v) for k, v in weight_offsets.items()},
    }
    write_json(output_dir / "descriptor_summary.json", summary)

    # Write hw_layer_mapping.json for PPU param export coordination
    hw_mapping = []
    for layer in layers:
        hw_mapping.append({
            "hw_id": layer["id"],
            "orig_id": layer.get("orig_id", layer["id"]),
            "name": layer["name"],
            "cout_total": layer["cout_total"],
            "cin_total": layer["cin_total"],
            "template_id": layer["template_id"],
            "post_profile_id": layer.get("post_profile_id", 0),
        })
    write_json(output_dir / "hw_layer_mapping.json", hw_mapping)

    print(f"  desc_net.hex:    {net_lines} lines")
    print(f"  desc_layers.hex: {layer_lines} lines")
    print(f"  desc_tiles.hex:  {tile_lines} lines")
    print(f"  Total tiles:     {len(tile_descs)}")
    print(f"  Weight arena used: {w_cursor - memory_map['weight_arena_base']} bytes")


if __name__ == "__main__":
    main()
