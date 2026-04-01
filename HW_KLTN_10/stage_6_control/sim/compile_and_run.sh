#!/bin/bash
# Stage 6: Control — Compile & Simulate ALL 3 modules
PKG="../../stage_0_packages/rtl"
RTL="../rtl"
TB="../tb"

echo "╔══════════════════════════════════════════════╗"
echo "║  STAGE 6: CONTROL — ALL 3 MODULES              ║"
echo "╚══════════════════════════════════════════════╝"

PASS=0; FAIL=0

run_test() {
    local NAME=$1; local DEPS=$2
    echo ""; echo "─── $NAME ───"
    vcs -sverilog -full64 -debug_access+all -timescale=1ns/1ps \
        ${PKG}/accel_pkg.sv ${PKG}/desc_pkg.sv $DEPS \
        -top tb_${NAME} -o simv_${NAME} -l compile_${NAME}.log 2>&1 | tail -3
    [ $? -ne 0 ] && { echo "  COMPILE FAILED!"; FAIL=$((FAIL+1)); return; }
    ./simv_${NAME} -l sim_${NAME}.log +fsdbfile+${NAME}.fsdb +fsdb+all 2>&1 | tail -5
    grep -q "ALL.*PASS\|0 total errors\|★.*PASS" sim_${NAME}.log && { echo "  ★ PASS"; PASS=$((PASS+1)); } || { echo "  ✗ FAIL"; FAIL=$((FAIL+1)); }
}

run_test "tile_fsm"          "${RTL}/tile_fsm.sv ${TB}/tb_tile_fsm.sv"
run_test "shadow_reg_file"   "${RTL}/shadow_reg_file.sv ${TB}/tb_shadow_reg_file.sv"
run_test "compute_sequencer" "${RTL}/compute_sequencer.sv ${TB}/tb_compute_sequencer.sv"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  STAGE 6: $PASS PASS, $FAIL FAIL               ║"
echo "╚══════════════════════════════════════════════╝"
[ $FAIL -gt 0 ] && { echo "FIX before Stage 7!"; exit 1; } || echo "★ ALL PASSED → Stage 7"
