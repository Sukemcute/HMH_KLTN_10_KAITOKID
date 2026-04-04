#!/bin/bash
# Stage 11 — Block Verification compile & run
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HW="$(cd "$SCRIPT_DIR/../.." && pwd)"

# All RTL files in dependency order
PKG="$HW/stage_0_packages/rtl/accel_pkg.sv \
     $HW/stage_0_packages/rtl/desc_pkg.sv \
     $HW/stage_0_packages/rtl/csr_pkg.sv"

S1_RTL="$HW/stage_1_compute/rtl/dsp_pair_int8.sv \
        $HW/stage_1_compute/rtl/pe_unit.sv \
        $HW/stage_1_compute/rtl/column_reduce.sv \
        $HW/stage_1_compute/rtl/comparator_tree.sv"
S2_RTL="$HW/stage_2_ppu/rtl/ppu.sv"
S3_RTL="$HW/stage_3_memory/rtl/glb_input_bank_db.sv \
        $HW/stage_3_memory/rtl/glb_weight_bank.sv \
        $HW/stage_3_memory/rtl/glb_output_bank.sv \
        $HW/stage_3_memory/rtl/metadata_ram.sv"
S4_RTL="$HW/stage_4_addr_gen/rtl/addr_gen_input.sv \
        $HW/stage_4_addr_gen/rtl/addr_gen_weight.sv \
        $HW/stage_4_addr_gen/rtl/addr_gen_output.sv"
S5_RTL="$HW/stage_5_data_movement/rtl/router_cluster_v2.sv \
        $HW/stage_5_data_movement/rtl/window_gen.sv \
        $HW/stage_5_data_movement/rtl/swizzle_engine.sv"
S6_RTL="$HW/stage_6_control/rtl/tile_fsm.sv \
        $HW/stage_6_control/rtl/shadow_reg_file.sv \
        $HW/stage_6_control/rtl/compute_sequencer.sv"
S7_RTL="$HW/stage_7_subcluster/rtl/pe_cluster_v4.sv \
        $HW/stage_7_subcluster/rtl/subcluster_datapath.sv"

S11_PKG="$HW/stage_11_block_verify/rtl/stage11_pkg.sv"
S11_TB="$HW/stage_11_block_verify/tb/tb_stage11_block_verify.sv"

ALL_RTL="$PKG $S1_RTL $S2_RTL $S3_RTL $S4_RTL $S5_RTL $S6_RTL $S7_RTL $S11_PKG"

echo "===== Stage 11 — Block Verification ====="
echo "Compiling..."
xvlog --sv $ALL_RTL $S11_TB 2>&1 | tail -10
echo "Elaborating..."
xelab work.tb_stage11_block_verify -s sim_stage11 2>&1 | tail -10
echo "Running simulation..."
xsim sim_stage11 -R 2>&1
echo ""
echo "===== Stage 11 Complete ====="
