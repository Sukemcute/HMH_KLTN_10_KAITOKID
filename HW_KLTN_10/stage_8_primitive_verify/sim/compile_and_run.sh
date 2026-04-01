#!/usr/bin/env bash
# Stage 8 — Primitive verification on subcluster_datapath (+ Stage 5 swizzle regression)
# Requires: Synopsys VCS (same as Stage 7). On Windows, use WSL or adjust paths.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HW="$(cd "$ROOT/.." && pwd)"
S0="$HW/stage_0_packages/rtl"
S1="$HW/stage_1_compute/rtl"
S2="$HW/stage_2_ppu/rtl"
S3="$HW/stage_3_memory/rtl"
S4="$HW/stage_4_addr_gen/rtl"
S5="$HW/stage_5_data_movement/rtl"
S6="$HW/stage_6_control/rtl"
S7="$HW/stage_7_subcluster/rtl"
S8R="$ROOT/rtl"
TB="$ROOT/tb"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  STAGE 8: PRIMITIVE VERIFICATION (7 pe_modes / 1 HW)    ║"
echo "╚══════════════════════════════════════════════════════════╝"

echo ""; echo "─── tb_stage8_subcluster_primitives ───"
vcs -sverilog -full64 -debug_access+all -timescale=1ns/1ps \
    "$S0/accel_pkg.sv" "$S0/desc_pkg.sv" "$S0/csr_pkg.sv" \
    "$S8R/stage8_pkg.sv" \
    "$S1/dsp_pair_int8.sv" "$S1/pe_unit.sv" "$S1/column_reduce.sv" "$S1/comparator_tree.sv" \
    "$S2/ppu.sv" \
    "$S3/glb_input_bank_db.sv" "$S3/glb_weight_bank.sv" "$S3/glb_output_bank.sv" "$S3/metadata_ram.sv" \
    "$S4/addr_gen_input.sv" "$S4/addr_gen_weight.sv" "$S4/addr_gen_output.sv" \
    "$S5/router_cluster_v2.sv" "$S5/window_gen.sv" "$S5/swizzle_engine.sv" \
    "$S6/tile_fsm.sv" "$S6/shadow_reg_file.sv" "$S6/compute_sequencer.sv" \
    "$S7/pe_cluster_v4.sv" "$S7/subcluster_datapath.sv" \
    "$TB/tb_stage8_subcluster_primitives.sv" \
    -top tb_stage8_subcluster_primitives -o simv_s8 -l compile_s8.log 2>&1 | tail -8

if [[ $? -eq 0 ]]; then
  ./simv_s8 -l sim_s8.log +fsdbfile+stage8.fsdb +fsdb+all 2>&1 | tail -25
  if grep -q "STAGE 8 ALL PRIMITIVE TESTS PASSED" sim_s8.log; then
    echo "  ★ Stage-8 subcluster primitive TB: PASS"
  else
    echo "  ✗ Stage-8 subcluster primitive TB: review sim_s8.log"; exit 1
  fi
else
  echo "  COMPILE FAILED — see compile_s8.log"; exit 1
fi

echo ""
echo "Next (manual): cd $HW/stage_5_data_movement/sim && bash compile_and_run.sh"
echo "  → bit-exact upsample/concat per SW_KLTN (checklist §8.5/8.6 complement)"
echo ""
echo "★ STAGE 8: subcluster primitive TB complete when sim_s8 reports all PASS"
