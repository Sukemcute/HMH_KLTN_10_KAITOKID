#!/bin/bash
# ============================================================================
# Stage 5: Data Movement — Compile & Simulate ALL 3 modules
# Usage: cd stage_5_data_movement/sim && bash compile_and_run.sh
# ============================================================================

PKG="../../stage_0_packages/rtl"
RTL="../rtl"
TB="../tb"

echo "╔══════════════════════════════════════════════╗"
echo "║  STAGE 5: DATA MOVEMENT — ALL 3 MODULES       ║"
echo "╚══════════════════════════════════════════════╝"

PASS_COUNT=0
FAIL_COUNT=0

run_test() {
    local NAME=$1
    local RTL_FILE=$2
    local TB_FILE=$3

    echo ""
    echo "─── $NAME ───"
    vcs -sverilog -full64 -debug_access+all -timescale=1ns/1ps \
        ${PKG}/accel_pkg.sv ${RTL}/${RTL_FILE} ${TB}/${TB_FILE} \
        -top tb_${NAME} -o simv_${NAME} -l compile_${NAME}.log \
        2>&1 | tail -3

    if [ $? -ne 0 ]; then
        echo "  COMPILE FAILED!"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return
    fi

    ./simv_${NAME} -l sim_${NAME}.log +fsdbfile+${NAME}.fsdb +fsdb+all 2>&1 | tail -5

    if grep -q "ALL.*PASS\|0 total errors\|★.*PASS" sim_${NAME}.log; then
        echo "  ★ PASS"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "  ✗ FAIL — check sim_${NAME}.log"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

run_test "router_cluster_v2" "router_cluster_v2.sv" "tb_router_cluster_v2.sv"
run_test "window_gen"        "window_gen.sv"         "tb_window_gen.sv"
run_test "swizzle_engine"    "swizzle_engine.sv"     "tb_swizzle_engine.sv"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  STAGE 5 SUMMARY: $PASS_COUNT PASS, $FAIL_COUNT FAIL      ║"
echo "╚══════════════════════════════════════════════╝"

if [ $FAIL_COUNT -gt 0 ]; then
    echo "FIX failures before Stage 6!"
    exit 1
else
    echo "★ ALL STAGE 5 PASSED — Ready for Stage 6"
fi
