"""
Export RTL-specific parameters for each Conv layer.
Generates $readmemh-compatible files for the Verilog PPU testbench.

For each layer, creates rtl_test_L{idx}/ with:
  m_int.hex        - requant multiplier per channel (INT32)
  shift.hex        - requant shift per channel (UINT8)
  silu_lut.hex     - 256-entry SiLU LUT (signed INT8, hex)
  silu_lut_quant.hex - 256-entry SiLU LUT for tb_single_layer ($readmemh), same as verify_conv_layer.py
  zp_out.hex       - output zero point (single value)
  bias.hex         - bias per channel (INT32)
  psum_pos{p}.hex  - pre-computed INT32 psum for test position p
  golden_pos{p}.hex - expected uint8 output for test position p

Usage from PHASE_4/:
  python 03_rtl_cosim/export_rtl_params.py --layer 0
  python 03_rtl_cosim/export_rtl_params.py --layer 1
  python 03_rtl_cosim/export_rtl_params.py --layer 3
  python 03_rtl_cosim/export_rtl_params.py --layer 17
  python 03_rtl_cosim/export_rtl_params.py --all
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

PHASE4 = Path(__file__).resolve().parents[1]
GOLDEN_DIR = PHASE4 / "02_golden_data"
LBL_DIR = GOLDEN_DIR / "layer_by_layer"
RTL_DIR = PHASE4 / "03_rtl_cosim"


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
    vals = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for i in range(0, len(line), 8):
                s = line[i : i + 8]
                if len(s) < 8:
                    continue
                u = int(s, 16)
                if u >= 0x80000000:
                    u -= 0x100000000
                vals.append(u)
    return np.array(vals, dtype=np.int32)


def write_readmemh(path: Path, values, fmt: str = "02X"):
    """Write one hex value per line, compatible with $readmemh."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for v in values:
            f.write(format(int(v) & ((1 << (len(fmt.replace("0", "").replace("X", "")) * 4 if "X" in fmt else 32)) - 1), fmt) + "\n")


def write_int32_readmemh(path: Path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for v in values:
            iv = int(v)
            if iv < 0:
                iv = iv + (1 << 32)
            f.write(f"{iv:08X}\n")


def write_uint8_readmemh(path: Path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for v in values:
            f.write(f"{int(v) & 0xFF:02X}\n")


def write_int8_readmemh(path: Path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for v in values:
            iv = int(v)
            if iv < 0:
                iv = iv + 256
            f.write(f"{iv & 0xFF:02X}\n")


def load_layer_info(layer_idx: int) -> dict:
    with open(LBL_DIR / "layer_summary.json", "r") as f:
        summary = json.load(f)
    for entry in summary["layers"]:
        if entry["index"] == layer_idx:
            return entry
    raise ValueError(f"Layer {layer_idx} not found")


def compute_m_int_shift(m_float: float, max_shift: int = 30) -> tuple[int, int]:
    """Compute fixed-point multiplier and shift for requantization."""
    if m_float == 0:
        return 0, 0
    for s in range(max_shift, 0, -1):
        m = int(round(m_float * (1 << s)))
        if abs(m) < (1 << 31):
            return m, s
    return int(round(m_float)), 0


def build_rtl_silu_lut(scale: float) -> np.ndarray:
    """Build SiLU LUT for RTL: indexed by (signed_value + 128), output is signed int8."""
    lut = np.zeros(256, dtype=np.int8)
    for j in range(256):
        y = j - 128
        x = y * scale
        if abs(x) < 1e-12:
            silu = 0.0
        else:
            silu = x / (1.0 + math.exp(-x))
        q = int(round(silu / scale))
        lut[j] = np.int8(max(-128, min(127, q)))
    return lut


def build_silu_lut_quant_domain(scale: float, zero_point: int) -> np.ndarray:
    """
    Same indexing as verify_conv_layer.build_silu_lut / export golden:
    index q = uint8 value before SiLU (after requant+clip); output = uint8 after SiLU.
    Used by tb_single_layer.sv ($readmemh) to avoid $exp vs math.exp drift.
    """
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


def compute_psum_for_position(inp, wgt, inp_zp, w_zps, cin, kh, kw, sh, sw, pad, hin, win, oh, ow, cout):
    """Compute INT32 psums for all output channels at one spatial position."""
    psums = np.zeros(cout, dtype=np.int64)
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
                        act_val = np.int64(0)
                    wgt_val = np.int64(wgt[co, ci, fh, fw]) - np.int64(w_zps[co])
                    acc += act_val * wgt_val
        psums[co] = acc
    return psums.astype(np.int32)


def export_layer(layer_idx: int):
    info = load_layer_info(layer_idx)
    print(f"\n{'='*60}")
    print(f"  Exporting RTL params for Layer {layer_idx}: {info['name']}")
    print(f"{'='*60}")

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

    out_dir = RTL_DIR / f"rtl_test_L{layer_idx:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- M_int and shift per channel ---
    m_ints = []
    shifts = []
    for co in range(cout):
        m_float = inp_scale * float(w_scales[co]) / out_scale
        m, s = compute_m_int_shift(m_float)
        m_ints.append(m)
        shifts.append(s)

    write_int32_readmemh(out_dir / "m_int.hex", m_ints)
    write_uint8_readmemh(out_dir / "shift.hex", shifts)
    print(f"  m_int range: [{min(m_ints)}, {max(m_ints)}]")
    print(f"  shift range: [{min(shifts)}, {max(shifts)}]")

    # --- RTL SiLU LUT ---
    silu_lut = build_rtl_silu_lut(out_scale)
    write_int8_readmemh(out_dir / "silu_lut.hex", silu_lut)
    print(f"  SiLU LUT: lut[128]={silu_lut[128]} (zero), lut[0]={silu_lut[0]}, lut[255]={silu_lut[255]}")

    # --- SiLU LUT for tb_single_layer / verify_conv match (quant domain index 0..255) ---
    silu_q = build_silu_lut_quant_domain(out_scale, out_zp)
    write_uint8_readmemh(out_dir / "silu_lut_quant.hex", silu_q)
    print(f"  silu_lut_quant.hex: tb_single_layer / verify_conv_layer.py compatible")

    # --- zp_out ---
    write_uint8_readmemh(out_dir / "zp_out.hex", [out_zp])
    print(f"  zp_out: {out_zp}")

    # --- Bias ---
    bias_path = GOLDEN_DIR / f"bias_L{layer_idx}_{wgt_meta['name']}.hex"
    if bias_path.exists():
        bias = load_hex_int32(bias_path)
        if len(bias) < cout:
            bias = np.pad(bias, (0, cout - len(bias)))
    else:
        bias = np.zeros(cout, dtype=np.int32)
    write_int32_readmemh(out_dir / "bias.hex", bias)
    print(f"  Bias: {len(bias)} values, range [{bias.min()}, {bias.max()}]")

    # --- Load golden input/weight/output ---
    inp_flat = load_hex_uint8(LBL_DIR / inp_meta["file"])
    wgt_flat = load_hex_uint8(LBL_DIR / wgt_meta["file"])
    out_flat = load_hex_uint8(LBL_DIR / out_meta["file"])

    inp = inp_flat[: cin * hin * win].reshape(cin, hin, win).astype(np.int32)
    wgt = wgt_flat[: cout * cin * kh * kw].reshape(cout, cin, kh, kw).astype(np.int8).astype(np.int32)
    golden = out_flat[: cout * hout * wout].reshape(cout, hout, wout).astype(np.uint8)

    # --- Test positions ---
    positions = [
        (0, 0),
        (0, wout // 2),
        (hout // 2, wout // 2),
        (hout - 1, wout - 1),
    ]

    for pi, (oh, ow) in enumerate(positions):
        psums = compute_psum_for_position(
            inp, wgt, inp_zp, w_zps, cin, kh, kw, sh, sw, pad, hin, win, oh, ow, cout
        )
        golden_slice = golden[:, oh, ow]

        write_int32_readmemh(out_dir / f"psum_pos{pi}.hex", psums)
        write_uint8_readmemh(out_dir / f"golden_pos{pi}.hex", golden_slice)

        # Verify pipeline manually
        ok = 0
        for co in range(cout):
            biased = int(psums[co]) + int(bias[co])
            mult = biased * m_ints[co]
            s = shifts[co]
            if s > 0:
                rounded = mult + (1 << (s - 1))
            else:
                rounded = mult
            shifted = rounded >> s
            if shifted > 32767:
                shifted = 32767
            elif shifted < -32768:
                shifted = -32768
            idx = max(0, min(255, shifted + 128))
            silu_val = int(silu_lut[idx])
            result = silu_val + out_zp
            result = max(-128, min(127, result))
            if result == int(golden_slice[co]):
                ok += 1

        print(f"  Pos{pi} ({oh},{ow}): psum range [{psums.min()},{psums.max()}] | "
              f"RTL pipeline verify: {ok}/{cout} exact")

    # --- Export cout value for testbench ---
    with open(out_dir / "params.txt", "w") as f:
        f.write(f"cout={cout}\n")
        f.write(f"hout={hout}\n")
        f.write(f"wout={wout}\n")
        f.write(f"zp_out={out_zp}\n")
        f.write(f"num_positions={len(positions)}\n")
        for pi, (oh, ow) in enumerate(positions):
            f.write(f"pos{pi}={oh},{ow}\n")

    print(f"\n  Files exported to: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, help="Layer index")
    parser.add_argument("--all", action="store_true", help="Export all Conv layers")
    args = parser.parse_args()

    if args.all:
        for idx in [0, 1, 3, 17]:
            export_layer(idx)
    elif args.layer is not None:
        export_layer(args.layer)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
