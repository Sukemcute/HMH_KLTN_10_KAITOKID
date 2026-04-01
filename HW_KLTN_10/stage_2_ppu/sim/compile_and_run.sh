#!/bin/bash
# ============================================================================
# Stage 2: PPU — Compile & Simulate
# Usage: cd stage_2_ppu/sim && bash compile_and_run.sh
# ============================================================================

PKG="../../../HW_KLTN_10/stage_0_packages/rtl"
RTL="../rtl"
TB="../tb"

echo "╔══════════════════════════════════════════════╗"
echo "║  STAGE 2: PPU — Compile & Simulate           ║"
echo "╚══════════════════════════════════════════════╝"

# --- VCS Compile ---
echo "[1/2] VCS Compile..."
vcs -sverilog -full64 -debug_access+all \
    -timescale=1ns/1ps \
    +incdir+${PKG} \
    ${PKG}/accel_pkg.sv \
    ${RTL}/ppu.sv \
    ${TB}/tb_ppu.sv \
    -top tb_ppu \
    -o simv_ppu \
    -l compile_ppu.log \
    +lint=all \
    2>&1 | tail -5

if [ $? -ne 0 ]; then
    echo "COMPILE FAILED! See compile_ppu.log"
    exit 1
fi
echo "Compile: OK"

# --- Simulate + FSDB dump ---
echo "[2/2] Simulate..."
./simv_ppu -l sim_ppu.log +fsdbfile+ppu.fsdb +fsdb+all 2>&1 | tail -20

echo ""
echo "════════════════════════════════════════"
grep -E "PASS|FAIL|★|errors" sim_ppu.log | tail -15
echo "════════════════════════════════════════"
echo "Waveform: ppu.fsdb"
echo "Debug: verdi -ssf ppu.fsdb -sverilog ${PKG}/accel_pkg.sv ${RTL}/ppu.sv ${TB}/tb_ppu.sv &"
