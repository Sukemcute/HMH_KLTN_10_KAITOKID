#!/usr/bin/env python3
"""
cosim_vector_gen.py -- Generate HW-layout-packed golden vectors for all primitives.

For each primitive:
  1. Build input / weight / quant-param fixture
  2. Call the Python golden primitive
  3. Pack input/weight/expected-output into RTL SRAM layout (.memh)
  4. Write quant param file

Output paths:
  - Stage 8 primitives: .../stage_100/work/vectors/<prim>/
  - Stage 11 blocks:    .../stage_11_block_verify/generated/<block>/

Usage
-----
  python cosim_vector_gen.py --prim os1       # generate OS1 vectors
  python cosim_vector_gen.py --prim rs3       # generate RS3 vectors
  python cosim_vector_gen.py --prim dw3       # ...
  python cosim_vector_gen.py --prim mp5
  python cosim_vector_gen.py --prim upsample
  python cosim_vector_gen.py --prim concat
  python cosim_vector_gen.py --prim ewise_add
  python cosim_vector_gen.py --prim move
  python cosim_vector_gen.py --prim dw7
  python cosim_vector_gen.py --prim all       # generate ALL primitives
  python cosim_vector_gen.py --prim blocks    # generate block-level vectors
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_SW_ROOT = os.path.dirname(_THIS)
_GOLDEN = os.path.join(_SW_ROOT, "python_golden_originial")
if _GOLDEN not in sys.path:
    sys.path.insert(0, _GOLDEN)

import numpy as np

from primitives.primitive_conv   import rs_dense_3x3, os_1x1
from primitives.primitive_dw     import dw_3x3, dw_7x7_multipass
from primitives.primitive_pool   import maxpool_5x5
from primitives.primitive_tensor import move, upsample_nearest, concat, ewise_add
from quant.quant_affine          import make_requant_params

import sw_layout as lay

# ── Default output root (under HW tree) ──

def _hw_root() -> str:
    return os.path.normpath(os.path.join(_SW_ROOT, "..", "..", "HW_KLTN_10"))


def _stage8_gen(prim_name: str) -> str:
    # All USE_GOLDEN_VECTORS memh files: stage_100/work/vectors/ (single observation hub)
    d = os.path.join(
        _hw_root(),
        "stage_100",
        "work",
        "vectors",
        prim_name,
    )
    os.makedirs(d, exist_ok=True)
    return d


def _stage11_gen(block_name: str) -> str:
    d = os.path.join(_hw_root(), "stage_11_block_verify", "generated", block_name)
    os.makedirs(d, exist_ok=True)
    return d


def _write_manifest(out_dir: str, info: dict) -> None:
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════
#  PRIMITIVE GENERATORS
# ═══════════════════════════════════════════════════════════════

def gen_os1(seed: int = 42) -> str:
    """OS1 Conv 1x1 -- simplest compute primitive."""
    rng = np.random.RandomState(seed)
    cin, cout, hin, win = 4, 4, 2, 20
    hout, wout = hin, win  # stride=1, no padding

    X = rng.randint(-10, 11, size=(1, cin, hin, win)).astype(np.int8)
    W = rng.randint(-5, 6, size=(cout, cin, 1, 1)).astype(np.int8)
    B = np.zeros(cout, dtype=np.int32)
    scale_w = np.ones(cout, dtype=np.float64)

    Y, _, _ = os_1x1(X, W, B, 1.0, 0, scale_w, 0, 1.0, 0, activation="relu")

    M_int, shift = make_requant_params(1.0, scale_w, 1.0)

    out_dir = _stage8_gen("os1")

    in_banks  = lay.pack_input_to_banks(X, zp_x=0)
    wt_banks  = lay.pack_weight_os1(W)

    for bk in range(3):
        lay.write_bank_memh(os.path.join(out_dir, f"input_bank{bk}.memh"), in_banks[bk])
        lay.write_bank_memh(os.path.join(out_dir, f"weight_bank{bk}.memh"), wt_banks[bk])

    for bk in range(4):
        lay.write_expected_memh(
            os.path.join(out_dir, f"expected_out_bank{bk}.memh"),
            Y, cout, hout, wout, bk,
        )

    lay.write_quant_params(
        os.path.join(out_dir, "quant_params.txt"), B, M_int, shift, 0, cout,
    )

    _write_manifest(out_dir, {
        "primitive": "os1", "pe_mode": "PE_OS1",
        "cin": cin, "cout": cout, "hin": hin, "win": win,
        "hout": hout, "wout": wout, "stride": 1, "padding": 0,
        "activation": "relu", "seed": seed,
    })
    print(f"[OS1]  Generated in {out_dir}")
    return out_dir


def gen_rs3(seed: int = 42) -> str:
    """RS3 Conv 3x3 -- with 3-bank input layout."""
    rng = np.random.RandomState(seed)
    cin, cout, hin, win = 1, 4, 5, 24
    stride, padding = 1, 1
    hout = (hin + 2 * padding - 3) // stride + 1  # 5
    wout = (win + 2 * padding - 3) // stride + 1  # 24

    X = rng.randint(-5, 6, size=(1, cin, hin, win)).astype(np.int8)
    W = rng.randint(-3, 4, size=(cout, cin, 3, 3)).astype(np.int8)
    B = np.zeros(cout, dtype=np.int32)
    scale_w = np.ones(cout, dtype=np.float64)

    Y, _, _ = rs_dense_3x3(X, W, B, 1.0, 0, scale_w, 0, 1.0, 0,
                            stride=stride, padding=padding, activation="relu")

    M_int, shift = make_requant_params(1.0, scale_w, 1.0)

    out_dir = _stage8_gen("rs3")

    in_banks = lay.pack_input_to_banks(X, zp_x=0)
    wt_banks = lay.pack_weight_rs3(W)

    for bk in range(3):
        lay.write_bank_memh(os.path.join(out_dir, f"input_bank{bk}.memh"), in_banks[bk])
        lay.write_bank_memh(os.path.join(out_dir, f"weight_bank{bk}.memh"), wt_banks[bk])

    for bk in range(4):
        lay.write_expected_memh(
            os.path.join(out_dir, f"expected_out_bank{bk}.memh"),
            Y, cout, hout, wout, bk,
        )

    lay.write_quant_params(
        os.path.join(out_dir, "quant_params.txt"), B, M_int, shift, 0, cout,
    )

    _write_manifest(out_dir, {
        "primitive": "rs3", "pe_mode": "PE_RS3",
        "cin": cin, "cout": cout, "hin": hin, "win": win,
        "hout": hout, "wout": wout, "stride": stride, "padding": padding,
        "activation": "relu", "seed": seed,
    })
    print(f"[RS3]  Generated in {out_dir}")
    return out_dir


def gen_dw3(seed: int = 42) -> str:
    """DW3 Depthwise 3x3."""
    rng = np.random.RandomState(seed)
    ch, hin, win = 4, 4, 20
    stride, padding = 1, 1
    hout = (hin + 2 * padding - 3) // stride + 1
    wout = (win + 2 * padding - 3) // stride + 1

    X = rng.randint(-8, 9, size=(1, ch, hin, win)).astype(np.int8)
    W = rng.randint(-3, 4, size=(ch, 3, 3)).astype(np.int8)
    B = np.zeros(ch, dtype=np.int32)
    scale_w = np.ones(ch, dtype=np.float64)

    Y, _, _ = dw_3x3(X, W, B, 1.0, 0, scale_w, 1.0, 0,
                      stride=stride, activation="none")

    M_int, shift = make_requant_params(1.0, scale_w, 1.0)

    out_dir = _stage8_gen("dw3")

    in_banks = lay.pack_input_to_banks(X, zp_x=0)
    wt_banks = lay.pack_weight_dw3(W)

    for bk in range(3):
        lay.write_bank_memh(os.path.join(out_dir, f"input_bank{bk}.memh"), in_banks[bk])
        lay.write_bank_memh(os.path.join(out_dir, f"weight_bank{bk}.memh"), wt_banks[bk])

    for bk in range(4):
        lay.write_expected_memh(
            os.path.join(out_dir, f"expected_out_bank{bk}.memh"),
            Y, ch, hout, wout, bk,
        )

    lay.write_quant_params(
        os.path.join(out_dir, "quant_params.txt"), B, M_int, shift, 0, ch,
    )

    _write_manifest(out_dir, {
        "primitive": "dw3", "pe_mode": "PE_DW3",
        "cin": ch, "cout": ch, "hin": hin, "win": win,
        "hout": hout, "wout": wout, "stride": stride, "padding": padding,
        "activation": "none", "seed": seed,
    })
    print(f"[DW3]  Generated in {out_dir}")
    return out_dir


def gen_mp5(seed: int = 42) -> str:
    """MaxPool 5x5."""
    rng = np.random.RandomState(seed)
    ch, hin, win = 4, 5, 20

    X = rng.randint(-20, 21, size=(1, ch, hin, win)).astype(np.int8)

    Y, _, _ = maxpool_5x5(X, 1.0, 0, padding=2)
    hout, wout = Y.shape[2], Y.shape[3]

    out_dir = _stage8_gen("mp5")

    in_banks = lay.pack_input_to_banks(X, zp_x=0)

    for bk in range(3):
        lay.write_bank_memh(os.path.join(out_dir, f"input_bank{bk}.memh"), in_banks[bk])

    for bk in range(4):
        lay.write_expected_memh(
            os.path.join(out_dir, f"expected_out_bank{bk}.memh"),
            Y, ch, hout, wout, bk,
        )

    _write_manifest(out_dir, {
        "primitive": "mp5", "pe_mode": "PE_MP5",
        "cin": ch, "cout": ch, "hin": hin, "win": win,
        "hout": hout, "wout": wout,
        "activation": "none", "seed": seed,
    })
    print(f"[MP5]  Generated in {out_dir}")
    return out_dir


def gen_upsample(seed: int = 42) -> str:
    """Upsample nearest 2x."""
    rng = np.random.RandomState(seed)
    ch, hin, win = 4, 2, 20

    X = rng.randint(-30, 31, size=(1, ch, hin, win)).astype(np.int8)
    Y, _, _ = upsample_nearest(X, 1.0, 0, scale_factor=2)
    hout, wout = Y.shape[2], Y.shape[3]

    out_dir = _stage8_gen("upsample")

    in_banks = lay.pack_input_to_banks(X, zp_x=0)
    for bk in range(3):
        lay.write_bank_memh(os.path.join(out_dir, f"input_bank{bk}.memh"), in_banks[bk])

    for bk in range(4):
        lay.write_expected_memh(
            os.path.join(out_dir, f"expected_out_bank{bk}.memh"),
            Y, ch, hout, wout, bk,
        )

    _write_manifest(out_dir, {
        "primitive": "upsample", "pe_mode": "PE_PASS", "swizzle": "SWZ_UPSAMPLE2X",
        "cin": ch, "cout": ch, "hin": hin, "win": win,
        "hout": hout, "wout": wout, "seed": seed,
    })
    print(f"[UPSAMPLE] Generated in {out_dir}")
    return out_dir


def gen_concat(seed: int = 42) -> str:
    """Concat two tensors along channel axis."""
    rng = np.random.RandomState(seed)
    c1, c2, hin, win = 4, 4, 2, 20

    A = rng.randint(-20, 21, size=(1, c1, hin, win)).astype(np.int8)
    B_t = rng.randint(-20, 21, size=(1, c2, hin, win)).astype(np.int8)

    Y, _, _ = concat([A, B_t], [1.0, 1.0], [0, 0], axis=1, strategy="max")
    cout = Y.shape[1]
    hout, wout = Y.shape[2], Y.shape[3]

    out_dir = _stage8_gen("concat")

    in_banks_a = lay.pack_input_to_banks(A, zp_x=0)
    in_banks_b = lay.pack_input_to_banks(B_t, zp_x=0)
    for bk in range(3):
        lay.write_bank_memh(os.path.join(out_dir, f"input_a_bank{bk}.memh"), in_banks_a[bk])
        lay.write_bank_memh(os.path.join(out_dir, f"input_b_bank{bk}.memh"), in_banks_b[bk])

    for bk in range(4):
        lay.write_expected_memh(
            os.path.join(out_dir, f"expected_out_bank{bk}.memh"),
            Y, cout, hout, wout, bk,
        )

    _write_manifest(out_dir, {
        "primitive": "concat", "pe_mode": "PE_PASS", "swizzle": "SWZ_CONCAT",
        "c_a": c1, "c_b": c2, "cout": cout, "hin": hin, "win": win,
        "hout": hout, "wout": wout, "seed": seed,
    })
    print(f"[CONCAT] Generated in {out_dir}")
    return out_dir


def gen_ewise_add(seed: int = 42) -> str:
    """Element-wise add."""
    rng = np.random.RandomState(seed)
    ch, hin, win = 4, 1, 20

    A = rng.randint(-30, 31, size=(1, ch, hin, win)).astype(np.int8)
    B_t = rng.randint(-30, 31, size=(1, ch, hin, win)).astype(np.int8)

    Y, _, _ = ewise_add(A, 1.0, 0, B_t, 1.0, 0, scale_out=1.0, zp_out=0, strategy="max")
    hout, wout = Y.shape[2], Y.shape[3]

    out_dir = _stage8_gen("ewise_add")

    in_banks_a = lay.pack_input_to_banks(A, zp_x=0)
    in_banks_b = lay.pack_input_to_banks(B_t, zp_x=0)
    for bk in range(3):
        lay.write_bank_memh(os.path.join(out_dir, f"input_a_bank{bk}.memh"), in_banks_a[bk])
        lay.write_bank_memh(os.path.join(out_dir, f"input_b_bank{bk}.memh"), in_banks_b[bk])

    for bk in range(4):
        lay.write_expected_memh(
            os.path.join(out_dir, f"expected_out_bank{bk}.memh"),
            Y, ch, hout, wout, bk,
        )

    _write_manifest(out_dir, {
        "primitive": "ewise_add", "pe_mode": "PE_PASS", "swizzle": "SWZ_EWISE_ADD",
        "cin": ch, "cout": ch, "hin": hin, "win": win,
        "hout": hout, "wout": wout, "seed": seed,
    })
    print(f"[EWISE_ADD] Generated in {out_dir}")
    return out_dir


def gen_move(seed: int = 42) -> str:
    """Move / copy."""
    rng = np.random.RandomState(seed)
    ch, hin, win = 4, 2, 20

    X = rng.randint(-50, 51, size=(1, ch, hin, win)).astype(np.int8)
    Y, _, _ = move(X, 1.0, 0)
    hout, wout = Y.shape[2], Y.shape[3]

    out_dir = _stage8_gen("move")

    in_banks = lay.pack_input_to_banks(X, zp_x=0)
    for bk in range(3):
        lay.write_bank_memh(os.path.join(out_dir, f"input_bank{bk}.memh"), in_banks[bk])

    for bk in range(4):
        lay.write_expected_memh(
            os.path.join(out_dir, f"expected_out_bank{bk}.memh"),
            Y, ch, hout, wout, bk,
        )

    _write_manifest(out_dir, {
        "primitive": "move", "pe_mode": "PE_PASS",
        "cin": ch, "cout": ch, "hin": hin, "win": win,
        "hout": hout, "wout": wout, "seed": seed,
    })
    print(f"[MOVE] Generated in {out_dir}")
    return out_dir


def gen_dw7(seed: int = 42) -> str:
    """DW7 depthwise 7x7 multipass (3 passes)."""
    rng = np.random.RandomState(seed)
    ch, hin, win = 4, 8, 20
    stride = 1
    padding = 3
    hout = (hin + 2 * padding - 7) // stride + 1
    wout = (win + 2 * padding - 7) // stride + 1

    X = rng.randint(-5, 6, size=(1, ch, hin, win)).astype(np.int8)
    W = rng.randint(-2, 3, size=(ch, 7, 7)).astype(np.int8)
    B = np.zeros(ch, dtype=np.int32)
    scale_w = np.ones(ch, dtype=np.float64)

    Y, _, _ = dw_7x7_multipass(X, W, B, 1.0, 0, scale_w, 1.0, 0,
                                stride=stride, activation="none")

    M_int, shift = make_requant_params(1.0, scale_w, 1.0)

    out_dir = _stage8_gen("dw7")

    in_banks = lay.pack_input_to_banks(X, zp_x=0)
    wt_banks = lay.pack_weight_dw7(W)

    for bk in range(3):
        lay.write_bank_memh(os.path.join(out_dir, f"input_bank{bk}.memh"), in_banks[bk])
        lay.write_bank_memh(os.path.join(out_dir, f"weight_bank{bk}.memh"), wt_banks[bk])

    for bk in range(4):
        lay.write_expected_memh(
            os.path.join(out_dir, f"expected_out_bank{bk}.memh"),
            Y, ch, hout, wout, bk,
        )

    lay.write_quant_params(
        os.path.join(out_dir, "quant_params.txt"), B, M_int, shift, 0, ch,
    )

    _write_manifest(out_dir, {
        "primitive": "dw7", "pe_mode": "PE_DW7",
        "cin": ch, "cout": ch, "hin": hin, "win": win,
        "hout": hout, "wout": wout, "stride": stride, "padding": padding,
        "activation": "none", "num_k_pass": 3, "seed": seed,
    })
    print(f"[DW7]  Generated in {out_dir}")
    return out_dir


# ═══════════════════════════════════════════════════════════════
#  BLOCK-LEVEL GENERATORS  (Stage 11 style -- chain of primitives)
# ═══════════════════════════════════════════════════════════════

def gen_block_conv(seed: int = 42) -> str:
    """Block 11.1 Conv: single RS3 + ReLU."""
    return gen_rs3(seed)  # reuses RS3


def gen_block_scdown(seed: int = 42) -> str:
    """Block 11.3 SCDown: OS1 -> DW3(stride=2)."""
    rng = np.random.RandomState(seed)
    cin, cout_cv1, hin, win = 4, 8, 4, 20

    X = rng.randint(-5, 6, size=(1, cin, hin, win)).astype(np.int8)
    W_cv1 = rng.randint(-3, 4, size=(cout_cv1, cin, 1, 1)).astype(np.int8)
    B_cv1 = np.zeros(cout_cv1, dtype=np.int32)
    sw_cv1 = np.ones(cout_cv1, dtype=np.float64)

    Y1, _, _ = os_1x1(X, W_cv1, B_cv1, 1.0, 0, sw_cv1, 0, 1.0, 0, activation="relu")

    ch_dw = cout_cv1
    W_dw = rng.randint(-2, 3, size=(ch_dw, 3, 3)).astype(np.int8)
    B_dw = np.zeros(ch_dw, dtype=np.int32)
    sw_dw = np.ones(ch_dw, dtype=np.float64)

    Y2, _, _ = dw_3x3(Y1, W_dw, B_dw, 1.0, 0, sw_dw, 1.0, 0,
                       stride=2, activation="relu")

    out_dir = _stage11_gen("scdown")

    in_banks = lay.pack_input_to_banks(X, zp_x=0)
    wt_cv1_banks = lay.pack_weight_os1(W_cv1)
    wt_dw_banks  = lay.pack_weight_dw3(W_dw)

    for bk in range(3):
        lay.write_bank_memh(os.path.join(out_dir, f"input_bank{bk}.memh"), in_banks[bk])
        lay.write_bank_memh(os.path.join(out_dir, f"weight_cv1_bank{bk}.memh"), wt_cv1_banks[bk])
        lay.write_bank_memh(os.path.join(out_dir, f"weight_dw_bank{bk}.memh"), wt_dw_banks[bk])

    hout, wout, cout = Y2.shape[2], Y2.shape[3], Y2.shape[1]
    for bk in range(4):
        lay.write_expected_memh(
            os.path.join(out_dir, f"expected_out_bank{bk}.memh"),
            Y2, cout, hout, wout, bk,
        )

    _write_manifest(out_dir, {
        "block": "scdown", "steps": ["os1_relu", "dw3_s2_relu"],
        "cin": cin, "cout": cout, "hin": hin, "win": win,
        "hout": hout, "wout": wout, "seed": seed,
    })
    print(f"[BLOCK SCDown] Generated in {out_dir}")
    return out_dir


def gen_block_sppf(seed: int = 42) -> str:
    """Block 11.4 SPPF: OS1 -> MP5 x3 -> concat -> OS1."""
    rng = np.random.RandomState(seed)
    cin, cout_cv1, hin, win = 4, 4, 4, 20

    X = rng.randint(-10, 11, size=(1, cin, hin, win)).astype(np.int8)

    W_cv1 = rng.randint(-3, 4, size=(cout_cv1, cin, 1, 1)).astype(np.int8)
    B_cv1 = np.zeros(cout_cv1, dtype=np.int32)
    sw_cv1 = np.ones(cout_cv1, dtype=np.float64)
    Y1, s1, z1 = os_1x1(X, W_cv1, B_cv1, 1.0, 0, sw_cv1, 0, 1.0, 0, activation="relu")

    P1, sp1, zp1 = maxpool_5x5(Y1, s1, z1, padding=2)
    P2, sp2, zp2 = maxpool_5x5(P1, sp1, zp1, padding=2)
    P3, sp3, zp3 = maxpool_5x5(P2, sp2, zp2, padding=2)

    Ycat, sc, zc = concat([Y1, P1, P2, P3], [s1, sp1, sp2, sp3], [z1, zp1, zp2, zp3])

    cout_cv2 = 4
    W_cv2 = rng.randint(-2, 3, size=(cout_cv2, Ycat.shape[1], 1, 1)).astype(np.int8)
    B_cv2 = np.zeros(cout_cv2, dtype=np.int32)
    sw_cv2 = np.ones(cout_cv2, dtype=np.float64)
    Yfinal, _, _ = os_1x1(Ycat, W_cv2, B_cv2, sc, zc, sw_cv2, 0, 1.0, 0, activation="relu")

    out_dir = _stage11_gen("sppf")
    hout, wout, cout = Yfinal.shape[2], Yfinal.shape[3], Yfinal.shape[1]

    in_banks = lay.pack_input_to_banks(X, zp_x=0)
    for bk in range(3):
        lay.write_bank_memh(os.path.join(out_dir, f"input_bank{bk}.memh"), in_banks[bk])

    for bk in range(4):
        lay.write_expected_memh(
            os.path.join(out_dir, f"expected_out_bank{bk}.memh"),
            Yfinal, cout, hout, wout, bk,
        )

    _write_manifest(out_dir, {
        "block": "sppf",
        "steps": ["os1_relu", "mp5", "mp5", "mp5", "concat", "os1_relu"],
        "cin": cin, "cout": cout, "hin": hin, "win": win,
        "hout": hout, "wout": wout, "seed": seed,
    })
    print(f"[BLOCK SPPF] Generated in {out_dir}")
    return out_dir


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

PRIM_MAP = {
    "os1":       gen_os1,
    "rs3":       gen_rs3,
    "dw3":       gen_dw3,
    "mp5":       gen_mp5,
    "upsample":  gen_upsample,
    "concat":    gen_concat,
    "ewise_add": gen_ewise_add,
    "move":      gen_move,
    "dw7":       gen_dw7,
}

BLOCK_MAP = {
    "conv":   gen_block_conv,
    "scdown": gen_block_scdown,
    "sppf":   gen_block_sppf,
}


def main() -> int:
    p = argparse.ArgumentParser(description="Generate HW-layout golden vectors for cosim")
    p.add_argument("--prim", type=str, default="all",
                   help="Primitive name (os1/rs3/dw3/mp5/upsample/concat/ewise_add/move/dw7/all/blocks)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.prim == "all":
        for name, fn in PRIM_MAP.items():
            fn(args.seed)
        for name, fn in BLOCK_MAP.items():
            fn(args.seed)
    elif args.prim == "blocks":
        for name, fn in BLOCK_MAP.items():
            fn(args.seed)
    elif args.prim in PRIM_MAP:
        PRIM_MAP[args.prim](args.seed)
    elif args.prim in BLOCK_MAP:
        BLOCK_MAP[args.prim](args.seed)
    else:
        print(f"Unknown primitive: {args.prim}")
        print(f"Available: {', '.join(list(PRIM_MAP) + list(BLOCK_MAP) + ['all', 'blocks'])}")
        return 1

    print("\nDone. Use +USE_GOLDEN_VECTORS in xsim to enable byte-exact comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
