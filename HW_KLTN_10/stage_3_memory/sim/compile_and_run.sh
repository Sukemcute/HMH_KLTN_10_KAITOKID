#!/bin/bash
# ============================================================================
# Stage 3: Memory — Compile & Simulate ALL 4 modules
# Usage: cd stage_3_memory/sim && bash compile_and_run.sh
# ============================================================================

PKG="../../stage_0_packages/rtl"
RTL="../rtl"
TB="../tb"

echo "╔══════════════════════════════════════════════╗"
echo "║  STAGE 3: MEMORY — Compile & Simulate ALL    ║"
echo "╚══════════════════════════════════════════════╝"

PASS_COUNT=0
FAIL_COUNT=0

run_test() {
    local NAME=$1
    local RTL_FILE=$2
    local TB_FILE=$3

    echo ""
    echo "─── $NAME ───"

    # Compile
    vcs -sverilog -full64 -debug_access+all \
        -timescale=1ns/1ps \
        ${PKG}/accel_pkg.sv \
        ${RTL}/${RTL_FILE} \
        ${TB}/${TB_FILE} \
        -top tb_${NAME} \
        -o simv_${NAME} \
        -l compile_${NAME}.log \
        2>&1 | tail -3

    if [ $? -ne 0 ]; then
        echo "  COMPILE FAILED!"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return
    fi

    # Simulate
    ./simv_${NAME} -l sim_${NAME}.log +fsdbfile+${NAME}.fsdb +fsdb+all 2>&1 | tail -5

    # Check results
    if grep -q "ALL.*PASS\|★.*PASS\|0 total errors" sim_${NAME}.log; then
        echo "  ★ PASS"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "  ✗ FAIL — check sim_${NAME}.log"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# --- Run all 4 memory module tests ---
run_test "glb_input_bank_db" "glb_input_bank_db.sv" "tb_glb_input_bank_db.sv"
run_test "glb_weight_bank"   "glb_weight_bank.sv"   "tb_glb_weight_bank.sv"
run_test "glb_output_bank"   "glb_output_bank.sv"   "tb_glb_output_bank.sv"
run_test "metadata_ram"      "metadata_ram.sv"       "tb_metadata_ram.sv"

# --- Summary ---
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  STAGE 3 SUMMARY: $PASS_COUNT PASS, $FAIL_COUNT FAIL      ║"
echo "╚══════════════════════════════════════════════╝"

if [ $FAIL_COUNT -gt 0 ]; then
    echo "⚠️  FIX failures before proceeding to Stage 4!"
    exit 1
else
    echo "★ ALL STAGE 3 MODULES PASSED — Ready for Stage 4"
fi
