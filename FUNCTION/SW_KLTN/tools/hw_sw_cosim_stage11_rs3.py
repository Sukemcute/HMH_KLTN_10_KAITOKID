#!/usr/bin/env python3
"""
HW/SW cosimulation helper — Stage 11.1-style RS3 conv (ident quant + ReLU).

Documentation alignment
-----------------------
  - SW primitives:  documentation/4_Primitives_Implementation/01_conv_primitives.md
  - Implementation: python_golden_originial/primitives/primitive_conv.py :: rs_dense_3x3
  - HW checklist:    HW_KLTN_10/RTL_V4_BUILD_CHECKLIST.md (Stage 8 golden vector flow)

Flow
----
  1) Build the SAME INT8 tensors the Stage 11 testbench conceptually uses:
       X [1, Cin, Hin, Win]  filled with one value (default 2)
       W [Cout, Cin, 3, 3]   filled with one value (default 1)
  2) Run the software primitive rs_dense_3x3 (identity scales, ReLU).
  3) Write generated files under HW_KLTN_10/stage_11_block_verify/generated/
     for optional $readmemh in SystemVerilog.

NOTE: Full-frame numpy conv (np.pad + sliding) may differ from ONE hardware tile
that uses 3 line-buffer banks + per-row addresses (addr_gen_input sram_addr_row[]).
When +strict fails, compare waveforms / dump intermediates — do not assume row index
of the full tensor equals iter_h of the tile without mapping.

Usage
-----
  python hw_sw_cosim_stage11_rs3.py
  python hw_sw_cosim_stage11_rs3.py --x-val 2 --w-val 1 --hin 5 --win 24 --cout 4
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

# Repo roots
_THIS = os.path.dirname(os.path.abspath(__file__))
_SW_ROOT = os.path.dirname(_THIS)
_GOLDEN = os.path.join(_SW_ROOT, "python_golden_originial")
if _GOLDEN not in sys.path:
    sys.path.insert(0, _GOLDEN)

import numpy as np

from primitives.primitive_conv import rs_dense_3x3


def default_hw_generated_dir() -> str:
    """stage_11_block_verify/generated next to HW_KLTN_10."""
    return os.path.normpath(
        os.path.join(_SW_ROOT, "..", "..", "HW_KLTN_10", "stage_11_block_verify", "generated")
    )


def run_sw_golden(
    cin: int,
    cout: int,
    hin: int,
    win: int,
    x_val: int,
    w_val: int,
    stride: int,
    padding: int,
    out_row: int,
    out_w: int,
) -> tuple[np.ndarray, int]:
    """
    Returns (Y full tensor int8, byte at channel 0 position [0,0,out_row,out_w]).
    padding: integer pad per side (same as layer_desc.padding in Stage 11 make_conv_rs3_desc).
    """
    x_val = int(np.clip(x_val, -128, 127))
    w_val = int(np.clip(w_val, -128, 127))
    x = np.full((1, cin, hin, win), x_val, dtype=np.int8)
    w = np.full((cout, cin, 3, 3), w_val, dtype=np.int8)
    b = np.zeros(cout, dtype=np.int32)
    scale_w = np.ones(cout, dtype=np.float64)
    y, _, _ = rs_dense_3x3(
        x,
        w,
        b,
        scale_x=1.0,
        zp_x=0,
        scale_w=scale_w,
        zp_w=0,
        scale_y=1.0,
        zp_y=0,
        stride=stride,
        padding=padding,
        activation="relu",
    )
    byte = int(y[0, 0, out_row, out_w])
    return y, byte


def write_memh_byte(path: str, value: int) -> None:
    """Single address 0, one byte in hex (for $readmemh line format)."""
    v = int(np.clip(value, -128, 127)) & 0xFF
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{v:02x}\n")


def copy_for_xsim_cwd(src: str, sim_cwd: str | None) -> None:
    """Optional second copy next to xsim so default filename works without plusargs."""
    if not sim_cwd:
        return
    dst = os.path.join(sim_cwd, os.path.basename(src))
    shutil.copy2(src, dst)
    print(f"  Also copied -> {dst} (xsim default SW_GOLDEN_MEMH)")


def write_report(path: str, y: np.ndarray, lane0_row: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# SW golden rs_dense_3x3 (ident quant, ReLU)\n")
        f.write(f"# Y shape {y.shape} dtype {y.dtype}\n")
        f.write(f"# Lane0 slice Y[0,:, {lane0_row}, 0..min(19,W-1)]\n")
        _, c, h, w = y.shape
        ww = min(20, w)
        for ci in range(c):
            vals = [int(y[0, ci, lane0_row, wi]) for wi in range(ww)]
            f.write(f"cout{ci}: " + " ".join(str(v) for v in vals) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description="Export SW golden vectors for HW cosim (Stage 11 RS3)")
    p.add_argument("--cin", type=int, default=1)
    p.add_argument("--cout", type=int, default=4)
    p.add_argument("--hin", type=int, default=5)
    p.add_argument("--win", type=int, default=24)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--padding", type=int, default=1, help="Per-side pad (Stage 11 make_conv_rs3_desc uses 1)")
    p.add_argument("--x-val", type=int, default=2)
    p.add_argument("--w-val", type=int, default=1)
    p.add_argument("--out-row", type=int, default=0, help="Full-frame output row index for report slice")
    p.add_argument("--out-w", type=int, default=0, help="Column index for single-byte memh")
    p.add_argument("--out-dir", type=str, default="", help="Override generated/ output directory")
    p.add_argument(
        "--xsim-cwd",
        type=str,
        default="",
        help="If set, copy .memh here as golden_stage11_rs3_lane0_c0.memh for tb default path",
    )
    args = p.parse_args()

    out_dir = args.out_dir or default_hw_generated_dir()
    os.makedirs(out_dir, exist_ok=True)

    y, b00 = run_sw_golden(
        args.cin,
        args.cout,
        args.hin,
        args.win,
        args.x_val,
        args.w_val,
        args.stride,
        args.padding,
        args.out_row,
        args.out_w,
    )

    memh = os.path.join(out_dir, "golden_stage11_rs3_lane0_c0.memh")
    report = os.path.join(out_dir, "golden_stage11_rs3_report.txt")
    write_memh_byte(memh, b00)
    write_report(report, y, args.out_row)
    if args.xsim_cwd:
        copy_for_xsim_cwd(memh, args.xsim_cwd)

    print("=== HW/SW cosim vector export (primitive: rs_dense_3x3) ===")
    print(f"  Y shape: {y.shape}")
    print(f"  SW byte Y[0,0,{args.out_row},{args.out_w}] = {b00} (signed int8)")
    print(f"  Wrote: {memh}")
    print(f"  Wrote: {report}")
    print("")
    print("Next: copy .memh to xsim cwd OR pass +SW_GOLDEN_MEMH (see COSIM_SW_HW.txt).")
    print("  xsim ... -R --testplusarg USE_SW_GOLDEN")
    print("")
    print("If SW byte != HW ext_rd_act_data[0], check tile/line-buffer mapping;")
    print("  full-frame conv row index may not equal hardware iter_h without offset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
