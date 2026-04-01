#!/bin/bash
# ============================================================================
# Stage 1: Compile & simulate all compute atom testbenches
# Tool: iverilog + vvp (open-source) or Vivado xsim
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$SCRIPT_DIR/../stage_0_packages/rtl"
COMP_DIR="$SCRIPT_DIR/rtl"
TB_DIR="$SCRIPT_DIR/tb"
OUT_DIR="$SCRIPT_DIR/work"

mkdir -p "$OUT_DIR"

echo "════════════════════════════════════════════════════"
echo " STAGE 1: Compute Atom Tests"
echo "════════════════════════════════════════════════════"

PASS=0
FAIL=0

# Common package files (must be compiled first)
PKGS="$PKG_DIR/accel_pkg.sv"

run_test() {
    local name=$1
    local rtl_files=$2
    local tb_file=$3

    echo ""
    echo "──────────────────────────────────────────────"
    echo " Running: $name"
    echo "──────────────────────────────────────────────"

    if command -v iverilog &> /dev/null; then
        iverilog -g2012 -o "$OUT_DIR/${name}.vvp" \
            $PKGS $rtl_files $tb_file 2>&1
        if [ $? -ne 0 ]; then
            echo "✗ COMPILE FAIL: $name"
            FAIL=$((FAIL + 1))
            return
        fi
        vvp "$OUT_DIR/${name}.vvp" 2>&1 | tee "$OUT_DIR/${name}.log"
        if grep -q "ALL PASS" "$OUT_DIR/${name}.log"; then
            echo "✓ $name: PASS"
            PASS=$((PASS + 1))
        else
            echo "✗ $name: FAIL"
            FAIL=$((FAIL + 1))
        fi
    else
        echo "No iverilog found. Please install or use Vivado xsim."
        FAIL=$((FAIL + 1))
    fi
}

# ──── Test 1: dsp_pair_int8 ────
run_test "tb_dsp_pair_int8" \
    "$COMP_DIR/dsp_pair_int8.sv" \
    "$TB_DIR/tb_dsp_pair_int8.sv"

# ──── Test 2: pe_unit ────
run_test "tb_pe_unit" \
    "$COMP_DIR/dsp_pair_int8.sv $COMP_DIR/pe_unit.sv" \
    "$TB_DIR/tb_pe_unit.sv"

# ──── Test 3: column_reduce ────
run_test "tb_column_reduce" \
    "$COMP_DIR/column_reduce.sv" \
    "$TB_DIR/tb_column_reduce.sv"

# ──── Test 4: comparator_tree ────
run_test "tb_comparator_tree" \
    "$COMP_DIR/comparator_tree.sv" \
    "$TB_DIR/tb_comparator_tree.sv"

# ──── Summary ────
echo ""
echo "════════════════════════════════════════════════════"
echo " STAGE 1 SUMMARY"
echo "   PASS: $PASS / $((PASS + FAIL))"
echo "   FAIL: $FAIL / $((PASS + FAIL))"
if [ $FAIL -eq 0 ]; then
    echo "   ★★★ ALL STAGE 1 TESTS PASS ★★★"
else
    echo "   ✗✗✗ $FAIL TEST(S) FAILED ✗✗✗"
fi
echo "════════════════════════════════════════════════════"
