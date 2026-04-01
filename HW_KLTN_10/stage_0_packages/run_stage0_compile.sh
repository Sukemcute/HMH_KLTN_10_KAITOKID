#!/bin/bash
# ============================================================================
# Stage 0: Compile packages only (no simulation needed)
# Tool: Any SystemVerilog simulator (iverilog, VCS, Questa, Xcelium)
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$SCRIPT_DIR/rtl"

echo "════════════════════════════════════════════════════"
echo " STAGE 0: Package Compilation Check"
echo "════════════════════════════════════════════════════"

# Try iverilog first (open-source), fallback to other tools
if command -v iverilog &> /dev/null; then
    echo "Using: iverilog"
    iverilog -g2012 -o /dev/null \
        "$PKG_DIR/accel_pkg.sv" \
        "$PKG_DIR/desc_pkg.sv" \
        "$PKG_DIR/csr_pkg.sv"
    echo "✓ All 3 packages compile clean."
elif command -v xvlog &> /dev/null; then
    echo "Using: Vivado xvlog"
    xvlog --sv "$PKG_DIR/accel_pkg.sv" "$PKG_DIR/desc_pkg.sv" "$PKG_DIR/csr_pkg.sv"
    echo "✓ All 3 packages compile clean."
else
    echo "No SystemVerilog compiler found. Install iverilog or use Vivado."
    exit 1
fi

echo ""
echo "★ STAGE 0 PASS — Packages verified."
