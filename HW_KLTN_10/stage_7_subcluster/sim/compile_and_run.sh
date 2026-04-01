#!/bin/bash
# Stage 7: Subcluster — Compile & Simulate pe_cluster_v4 + subcluster_datapath
# Dependencies: ALL previous stages (0-6)

S0="../../stage_0_packages/rtl"
S1="../../stage_1_compute/rtl"
S2="../../stage_2_ppu/rtl"
S3="../../stage_3_memory/rtl"
S4="../../stage_4_addr_gen/rtl"
S5="../../stage_5_data_movement/rtl"
S6="../../stage_6_control/rtl"
S7="../rtl"
TB="../tb"

echo "╔══════════════════════════════════════════════╗"
echo "║  STAGE 7: SUBCLUSTER — PE CLUSTER + INTEGRATION║"
echo "╚══════════════════════════════════════════════╝"

PASS=0; FAIL=0

# ── Test 1: pe_cluster_v4 (depends on pe_unit + column_reduce + comparator_tree) ──
echo ""; echo "─── pe_cluster_v4 ───"
vcs -sverilog -full64 -debug_access+all -timescale=1ns/1ps \
    ${S0}/accel_pkg.sv ${S0}/desc_pkg.sv \
    ${S1}/dsp_pair_int8.sv ${S1}/pe_unit.sv ${S1}/column_reduce.sv ${S1}/comparator_tree.sv \
    ${S7}/pe_cluster_v4.sv ${TB}/tb_pe_cluster_v4.sv \
    -top tb_pe_cluster_v4 -o simv_pe_cluster -l compile_pe_cluster.log 2>&1 | tail -3

if [ $? -eq 0 ]; then
    ./simv_pe_cluster -l sim_pe_cluster.log +fsdbfile+pe_cluster.fsdb +fsdb+all 2>&1 | tail -5
    grep -q "ALL.*PASS\|0 total errors\|★.*PASS" sim_pe_cluster.log && { echo "  ★ PASS"; PASS=$((PASS+1)); } || { echo "  ✗ FAIL"; FAIL=$((FAIL+1)); }
else
    echo "  COMPILE FAILED!"; FAIL=$((FAIL+1))
fi

# ── Test 2: subcluster_datapath (depends on ALL previous stages) ──
echo ""; echo "─── subcluster_datapath (smoke test) ───"
vcs -sverilog -full64 -debug_access+all -timescale=1ns/1ps \
    ${S0}/accel_pkg.sv ${S0}/desc_pkg.sv ${S0}/csr_pkg.sv \
    ${S1}/dsp_pair_int8.sv ${S1}/pe_unit.sv ${S1}/column_reduce.sv ${S1}/comparator_tree.sv \
    ${S2}/ppu.sv \
    ${S3}/glb_input_bank_db.sv ${S3}/glb_weight_bank.sv ${S3}/glb_output_bank.sv ${S3}/metadata_ram.sv \
    ${S4}/addr_gen_input.sv ${S4}/addr_gen_weight.sv ${S4}/addr_gen_output.sv \
    ${S5}/router_cluster_v2.sv ${S5}/window_gen.sv ${S5}/swizzle_engine.sv \
    ${S6}/tile_fsm.sv ${S6}/shadow_reg_file.sv ${S6}/compute_sequencer.sv \
    ${S7}/pe_cluster_v4.sv ${S7}/subcluster_datapath.sv \
    ${TB}/tb_subcluster_datapath.sv \
    -top tb_subcluster_datapath -o simv_subcluster -l compile_subcluster.log 2>&1 | tail -5

if [ $? -eq 0 ]; then
    ./simv_subcluster -l sim_subcluster.log +fsdbfile+subcluster.fsdb +fsdb+all 2>&1 | tail -10
    grep -q "ALL.*PASS\|0 total errors\|★.*PASS\|SMOKE TEST.*PASS" sim_subcluster.log && { echo "  ★ PASS"; PASS=$((PASS+1)); } || { echo "  ✗ FAIL"; FAIL=$((FAIL+1)); }
else
    echo "  COMPILE FAILED — check compile_subcluster.log"; FAIL=$((FAIL+1))
fi

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  STAGE 7: $PASS PASS, $FAIL FAIL               ║"
echo "╚══════════════════════════════════════════════╝"
[ $FAIL -gt 0 ] && { echo "FIX before Stage 8!"; exit 1; } || echo "★ ALL PASSED → Stage 8 (Primitive Verification)"
