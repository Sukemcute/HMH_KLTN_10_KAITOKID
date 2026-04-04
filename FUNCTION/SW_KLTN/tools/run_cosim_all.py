#!/usr/bin/env python3
"""
run_cosim_all.py -- Master regression runner for HW/SW cosim verification.

Steps:
  1. Generate ALL golden vectors (primitive + block level)
  2. Print summary of generated files
  3. Print instructions for running xsim with golden vectors

Usage:
  python run_cosim_all.py                  # Generate vectors only
  python run_cosim_all.py --run-stage8     # Generate + attempt xsim for Stage 8
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))

# Import the vector generator
sys.path.insert(0, _THIS)
import cosim_vector_gen as gen


def print_header(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def generate_all(seed: int = 42) -> dict:
    """Generate all golden vectors. Returns dict of {name: out_dir}."""
    results = {}

    print_header("GENERATING PRIMITIVE-LEVEL GOLDEN VECTORS (Stage 8)")

    for name, fn in gen.PRIM_MAP.items():
        try:
            out_dir = fn(seed)
            results[f"prim_{name}"] = {"dir": out_dir, "status": "OK"}
        except Exception as e:
            results[f"prim_{name}"] = {"dir": "", "status": f"ERROR: {e}"}
            print(f"  [ERROR] {name}: {e}")

    print_header("GENERATING BLOCK-LEVEL GOLDEN VECTORS (Stage 11)")

    for name, fn in gen.BLOCK_MAP.items():
        try:
            out_dir = fn(seed)
            results[f"block_{name}"] = {"dir": out_dir, "status": "OK"}
        except Exception as e:
            results[f"block_{name}"] = {"dir": "", "status": f"ERROR: {e}"}
            print(f"  [ERROR] {name}: {e}")

    return results


def print_summary(results: dict) -> None:
    print_header("VECTOR GENERATION SUMMARY")

    ok_count = sum(1 for v in results.values() if v["status"] == "OK")
    err_count = len(results) - ok_count

    for name, info in sorted(results.items()):
        status = info["status"]
        d = info["dir"]
        if status == "OK":
            print(f"  [OK]    {name:25s} -> {d}")
        else:
            print(f"  [FAIL]  {name:25s} -> {status}")

    print(f"\n  Total: {ok_count} OK, {err_count} FAILED out of {len(results)}")


def print_instructions() -> None:
    hw_root = gen._hw_root()
    print_header("HOW TO RUN SIMULATION")

    s100 = os.path.join(hw_root, "stage_100", "work")
    print(f"""
  Stage 8 (Primitive-level byte-exact, USE_GOLDEN_VECTORS):
  --------------------------------------------------------
  1. Memh output: {os.path.join(s100, 'vectors', '<prim>')}
  2. Run: {os.path.join(s100, 'run_stage100_golden.cmd')}  (CWD = stage_100/work for xsim)
  3. Or: python run_cosim_all.py then compile/sim with CWD = stage_100/work,
     plusargs: USE_GOLDEN_VECTORS  [optional GV_ROOT=vectors]

  Stage 8 (Quick regression, no golden files):
  --------------------------------------
  Run:  xsim <sim_snapshot> -R
     (original 10 tests: FSM + basic output checks)

  Stage 11 (Block-level, existing tests):
  --------------------------------------
  1. cd {os.path.join(hw_root, 'stage_11_block_verify')}
  2. Run:  xsim <sim_snapshot> -R
     (7 block tests: Conv, QC2f, SCDown, SPPF, QConcat, Upsample, QC2fCIB)

  Stage 11 (Block golden vectors):
  --------------------------------------
  Run:  xsim <sim_snapshot> -R --testplusarg USE_BLOCK_GOLDEN
     (loads generated/<block>/expected_out_bank*.memh for comparison)

  Combined (Stage 11 + SW golden for 11.1):
  --------------------------------------
  Run:  xsim <sim_snapshot> -R --testplusarg USE_SW_GOLDEN
""")


def main() -> int:
    p = argparse.ArgumentParser(description="Master cosim regression runner")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--generate-only", action="store_true",
                   help="Only generate vectors, don't attempt simulation")
    args = p.parse_args()

    print_header("HW/SW COSIM REGRESSION RUNNER")
    print(f"  Seed: {args.seed}")

    results = generate_all(args.seed)
    print_summary(results)
    print_instructions()

    err_count = sum(1 for v in results.values() if v["status"] != "OK")
    if err_count > 0:
        print(f"\n[WARNING] {err_count} vector generation(s) failed.")
        return 1

    print("[SUCCESS] All vectors generated. Ready for simulation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
