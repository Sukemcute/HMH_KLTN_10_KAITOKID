#!/bin/bash
# Stage 10 — System Integration compile & run
# Usage: bash compile_and_run.sh [test_name]
#   test_name: barrier | csr | top (default: all)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# sim -> stage_10_system -> HW_KLTN_10 (RTL root)
HW="$SCRIPT_DIR/../.."

# Package files
PKG="$HW/stage_0_packages/rtl/accel_pkg.sv \
     $HW/stage_0_packages/rtl/desc_pkg.sv \
     $HW/stage_0_packages/rtl/csr_pkg.sv"

# All lower-stage RTL
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
S9_RTL="$HW/stage_9_supercluster/rtl/local_arbiter_v2.sv \
        $HW/stage_9_supercluster/rtl/tensor_dma_v2.sv \
        $HW/stage_9_supercluster/rtl/tile_ingress_fifo.sv \
        $HW/stage_9_supercluster/rtl/supercluster_wrapper.sv"

# Stage 10 RTL
S10_RTL="$HW/stage_10_system/rtl/reset_sync.sv \
         $HW/stage_10_system/rtl/clk_wiz_250.sv \
         $HW/stage_10_system/rtl/axi_lite_slave.sv \
         $HW/stage_10_system/rtl/csr_register_bank.sv \
         $HW/stage_10_system/rtl/barrier_manager.sv \
         $HW/stage_10_system/rtl/desc_fetch_engine.sv \
         $HW/stage_10_system/rtl/global_scheduler.sv \
         $HW/stage_10_system/rtl/axi4_master_mux.sv \
         $HW/stage_10_system/rtl/accel_top.sv"

ALL_RTL="$PKG $S1_RTL $S2_RTL $S3_RTL $S4_RTL $S5_RTL $S6_RTL $S7_RTL $S9_RTL $S10_RTL"

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
  barrier)
    run_test tb_barrier_manager "$HW/stage_10_system/tb/tb_barrier_manager.sv"
    ;;
  csr)
    run_test tb_csr_register_bank "$HW/stage_10_system/tb/tb_csr_register_bank.sv"
    ;;
  top)
    run_test tb_accel_top "$HW/stage_10_system/tb/tb_accel_top.sv"
    ;;
  all)
    run_test tb_barrier_manager "$HW/stage_10_system/tb/tb_barrier_manager.sv"
    run_test tb_csr_register_bank "$HW/stage_10_system/tb/tb_csr_register_bank.sv"
    run_test tb_accel_top "$HW/stage_10_system/tb/tb_accel_top.sv"
    ;;
esac
echo "===== Stage 10 Tests Complete ====="
