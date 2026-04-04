#!/bin/bash
# Stage 9 — SuperCluster compile & run
# Usage: bash compile_and_run.sh [test_name]
#   test_name: arbiter | dma | sc_wrapper (default: all)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# sim -> stage_9_supercluster -> HW_KLTN_10 (RTL root)
HW="$SCRIPT_DIR/../.."

# Package files
PKG="$HW/stage_0_packages/rtl/accel_pkg.sv \
     $HW/stage_0_packages/rtl/desc_pkg.sv \
     $HW/stage_0_packages/rtl/csr_pkg.sv"

# Stage 1-7 RTL (needed for subcluster_datapath)
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

# Stage 9 RTL
S9_RTL="$HW/stage_9_supercluster/rtl/local_arbiter_v2.sv \
        $HW/stage_9_supercluster/rtl/tensor_dma_v2.sv \
        $HW/stage_9_supercluster/rtl/tile_ingress_fifo.sv \
        $HW/stage_9_supercluster/rtl/supercluster_wrapper.sv"

ALL_RTL="$PKG $S1_RTL $S2_RTL $S3_RTL $S4_RTL $S5_RTL $S6_RTL $S7_RTL $S9_RTL"

TEST=${1:-all}

run_test() {
  local name=$1
  local tb=$2
  echo "===== Compiling + Running $name ====="
  xvlog --sv $ALL_RTL $tb 2>&1 | tail -5
  xelab work.$name -s sim_$name 2>&1 | tail -5
  xsim sim_$name -R 2>&1
  echo ""
}

case "$TEST" in
  arbiter)
    run_test tb_local_arbiter_v2 "$HW/stage_9_supercluster/tb/tb_local_arbiter_v2.sv"
    ;;
  dma)
    run_test tb_tensor_dma_v2 "$HW/stage_9_supercluster/tb/tb_tensor_dma_v2.sv"
    ;;
  sc_wrapper)
    run_test tb_supercluster_wrapper "$HW/stage_9_supercluster/tb/tb_supercluster_wrapper.sv"
    ;;
  all)
    run_test tb_local_arbiter_v2 "$HW/stage_9_supercluster/tb/tb_local_arbiter_v2.sv"
    run_test tb_tensor_dma_v2 "$HW/stage_9_supercluster/tb/tb_tensor_dma_v2.sv"
    run_test tb_supercluster_wrapper "$HW/stage_9_supercluster/tb/tb_supercluster_wrapper.sv"
    ;;
esac
echo "===== Stage 9 Tests Complete ====="
