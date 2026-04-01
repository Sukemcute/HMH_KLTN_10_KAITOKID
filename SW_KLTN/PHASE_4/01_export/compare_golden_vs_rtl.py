"""
Compare golden Python outputs with RTL simulation outputs.

After running the RTL simulation, the testbench writes output activations
to hex files. This script compares them with the golden Python reference
and reports:
  - Bit-exact match percentage
  - Maximum absolute error
  - Mean absolute error
  - Histogram of error distribution

Usage:
  python compare_golden_vs_rtl.py \\
      --golden ../02_golden_data/golden_P3.hex \\
      --rtl    ../02_golden_data/rtl_dump/golden_P3.hex \\
      --name P3 --shape 1,128,80,80

  # P4: --shape 1,256,40,40   P5: --shape 1,512,20,20

  # --rtl phải là đường dẫn THẬT tới file hex dump từ RTL/sim (không copy "E:\\path\\to\\...").
  # Chưa có RTL: tự so golden với chính nó để kiểm tra script:
  #   python compare_golden_vs_rtl.py --golden ../02_golden_data/golden_P3.hex --rtl ../02_golden_data/golden_P3.hex --name P3 --shape 1,128,80,80
"""

import argparse
import sys
import numpy as np
from pathlib import Path


def load_hex_to_int8(hex_path):
    """Read hex file back to numpy uint8 array."""
    values = []
    with open(hex_path, "r", encoding="ascii", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('#'):
                continue
            for i in range(0, len(line), 2):
                byte_hex = line[i:i + 2]
                if len(byte_hex) == 2:
                    try:
                        val = int(byte_hex, 16)
                        values.append(val)
                    except ValueError:
                        pass
    return np.array(values, dtype=np.uint8)


def compare(golden_path, rtl_path, name, shape=None):
    print(f"\n{'='*60}")
    print(f"Comparing {name}: {golden_path} vs {rtl_path}")
    print(f"{'='*60}")

    golden = load_hex_to_int8(golden_path)
    rtl = load_hex_to_int8(rtl_path)

    print(f"  Golden size: {len(golden)} bytes")
    print(f"  RTL size:    {len(rtl)} bytes")

    min_len = min(len(golden), len(rtl))
    if min_len == 0:
        print("  ERROR: One or both files are empty!")
        return

    golden = golden[:min_len].astype(np.int8)
    rtl = rtl[:min_len].astype(np.int8)

    # Bit-exact comparison
    exact_match = np.sum(golden == rtl)
    match_pct = 100.0 * exact_match / min_len
    print(f"\n  Bit-exact matches: {exact_match} / {min_len} ({match_pct:.2f}%)")

    # Error analysis
    diff = np.abs(golden.astype(np.int16) - rtl.astype(np.int16))
    max_err = diff.max()
    mean_err = diff.mean()
    print(f"  Max absolute error:  {max_err}")
    print(f"  Mean absolute error: {mean_err:.4f}")

    # Error histogram
    print(f"\n  Error distribution:")
    for threshold in [0, 1, 2, 3, 5, 10, 20]:
        count = np.sum(diff <= threshold)
        pct = 100.0 * count / min_len
        print(f"    |error| <= {threshold:2d}: {count:8d} / {min_len} ({pct:6.2f}%)")

    # First 10 mismatches
    mismatch_indices = np.where(golden != rtl)[0]
    if len(mismatch_indices) > 0:
        print(f"\n  First 10 mismatches (of {len(mismatch_indices)}):")
        for idx in mismatch_indices[:10]:
            print(f"    [{idx}]: golden={golden[idx]:4d}  rtl={rtl[idx]:4d}  diff={diff[idx]:3d}")

    # Spatial analysis if shape is provided
    if shape is not None and len(shape) == 4:
        b, c, h, w = shape
        total = b * c * h * w
        if total <= min_len:
            golden_4d = golden[:total].reshape(shape)
            rtl_4d = rtl[:total].reshape(shape)
            diff_4d = np.abs(golden_4d.astype(np.int16) - rtl_4d.astype(np.int16))

            # Per-channel error
            print(f"\n  Per-channel analysis (top 5 worst):")
            ch_errors = []
            for ch in range(c):
                ch_max = diff_4d[0, ch].max()
                ch_mean = diff_4d[0, ch].mean()
                ch_errors.append((ch, ch_max, ch_mean))
            ch_errors.sort(key=lambda x: x[2], reverse=True)
            for ch, ch_max, ch_mean in ch_errors[:5]:
                print(f"    Channel {ch:3d}: max_err={ch_max:3d}, mean_err={ch_mean:.3f}")

    if match_pct == 100.0:
        print(f"\n  [PASS] {name}: PERFECT BIT-EXACT MATCH")
    elif match_pct >= 99.0:
        print(f"\n  [CLOSE] {name}: NEAR-EXACT ({match_pct:.2f}%), max_err={max_err}")
    else:
        print(f"\n  [FAIL] {name}: SIGNIFICANT DIFFERENCES ({match_pct:.2f}%)")

    return match_pct


def main():
    parser = argparse.ArgumentParser(
        description="So sanh tensor hex: golden (Python) vs dump RTL/sim."
    )
    parser.add_argument("--golden", type=str, required=True, help="File hex chuan (golden)")
    parser.add_argument(
        "--rtl",
        type=str,
        required=True,
        help="File hex tu RTL/xsim/board (duong dan that, KHONG dung vi du E:\\path\\to\\...)",
    )
    parser.add_argument("--name", type=str, default="output")
    parser.add_argument("--shape", type=str, default=None, help="e.g. 1,128,80,80")
    args = parser.parse_args()

    gp = Path(args.golden).expanduser().resolve()
    rp = Path(args.rtl).expanduser().resolve()

    if not gp.is_file():
        print(f"ERROR: Khong tim thay file golden:\n  {gp}")
        sys.exit(1)
    if not rp.is_file():
        print(f"ERROR: Khong tim thay file RTL (--rtl):\n  {rp}")
        print()
        print('  "--rtl" phai la duong dan THAT toi file dump (sau khi chay simulation / doc DDR).')
        print('  Chuoi "E:\\path\\to\\rtl_P3.hex" trong huong dan chi la VI DU — khong phai file tren may ban.')
        print()
        print("  Chua co RTL: tu kiem tra script (golden vs cung file golden) phai ra 100%:")
        sh = args.shape if args.shape else "1,128,80,80"
        print(f'    python compare_golden_vs_rtl.py --golden "{gp}" --rtl "{gp}" --name {args.name} --shape {sh}')
        sys.exit(1)

    shape = None
    if args.shape:
        shape = [int(x) for x in args.shape.split(",")]

    compare(str(gp), str(rp), args.name, shape)


if __name__ == "__main__":
    main()
