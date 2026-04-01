# =============================================================================
# Vivado xsim simulation script for tb_golden_check (L0 single-layer verify)
#
# Usage (from PHASE_4/03_rtl_cosim/):
#   vivado -mode batch -source run_vivado_sim.tcl
#
# Or for full model:
#   vivado -mode batch -source run_vivado_sim.tcl -tclargs FULL_MODEL
# =============================================================================

set script_dir [file dirname [file normalize [info script]]]
set rtl_root   [file normalize "$script_dir/../../PHASE_3"]
set tb_dir     [file normalize "$script_dir"]
set sim_dir    [file normalize "$script_dir/vivado_work"]

set full_model 0
if {[llength $argv] > 0 && [lindex $argv 0] eq "FULL_MODEL"} {
    set full_model 1
    puts ">>> FULL MODEL mode enabled"
} else {
    puts ">>> L0 SINGLE-LAYER mode (default)"
}

file mkdir $sim_dir

# =============================================================================
# Step 1: Compile packages first (order matters)
# =============================================================================
puts "\n=== Step 1: Compiling packages ==="

set pkg_files [list \
    "$rtl_root/packages/accel_pkg.sv" \
    "$rtl_root/packages/desc_pkg.sv"  \
    "$rtl_root/packages/csr_pkg.sv"   \
]

# =============================================================================
# Step 2: Compile RTL sources (leaf to top)
# =============================================================================
puts "\n=== Step 2: Compiling RTL sources ==="

set rtl_files [list \
    "$rtl_root/01_compute_leaf/rtl/dsp_pair_int8.sv"    \
    "$rtl_root/01_compute_leaf/rtl/column_reduce.sv"    \
    "$rtl_root/01_compute_leaf/rtl/comparator_tree.sv"  \
    "$rtl_root/01_compute_leaf/rtl/pe_unit.sv"          \
    "$rtl_root/01_compute_leaf/rtl/silu_lut.sv"         \
    "$rtl_root/02_ppu/rtl/ppu.sv"                       \
    "$rtl_root/03_memory/rtl/addr_gen_input.sv"         \
    "$rtl_root/03_memory/rtl/addr_gen_output.sv"        \
    "$rtl_root/03_memory/rtl/addr_gen_weight.sv"        \
    "$rtl_root/03_memory/rtl/glb_input_bank.sv"         \
    "$rtl_root/03_memory/rtl/glb_output_bank.sv"        \
    "$rtl_root/03_memory/rtl/glb_weight_bank.sv"        \
    "$rtl_root/03_memory/rtl/metadata_ram.sv"           \
    "$rtl_root/04_data_movement/rtl/router_cluster.sv"  \
    "$rtl_root/04_data_movement/rtl/swizzle_engine.sv"  \
    "$rtl_root/04_data_movement/rtl/window_gen.sv"      \
    "$rtl_root/05_integration/rtl/pe_cluster.sv"        \
    "$rtl_root/05_integration/rtl/shadow_reg_file.sv"   \
    "$rtl_root/05_integration/rtl/subcluster_wrapper.sv"\
    "$rtl_root/06_control/rtl/barrier_manager.sv"       \
    "$rtl_root/06_control/rtl/desc_fetch_engine.sv"     \
    "$rtl_root/06_control/rtl/global_scheduler.sv"      \
    "$rtl_root/06_control/rtl/local_arbiter.sv"         \
    "$rtl_root/06_control/rtl/tile_fsm.sv"              \
    "$rtl_root/07_system/rtl/tensor_dma.sv"             \
    "$rtl_root/07_system/rtl/controller_system.sv"      \
    "$rtl_root/07_system/rtl/supercluster_wrapper.sv"   \
    "$rtl_root/07_system/rtl/accel_top.sv"              \
]

# =============================================================================
# Step 3: Compile testbench
# =============================================================================
puts "\n=== Step 3: Compiling testbench ==="

set tb_files [list \
    "$tb_dir/tb_golden_check.sv" \
]

# =============================================================================
# Build xvlog command
# =============================================================================
set all_files [concat $pkg_files $rtl_files $tb_files]

set define_flags ""
if {$full_model} {
    set define_flags "-d COSIM_FULL_MODEL"
}

puts "\n=== Running xvlog (SystemVerilog compilation) ==="
set xvlog_cmd "xvlog -sv -work work $define_flags -log $sim_dir/xvlog.log"
foreach f $all_files {
    append xvlog_cmd " \"$f\""
}
puts "Command: $xvlog_cmd"

cd $sim_dir
eval exec $xvlog_cmd

# =============================================================================
# Step 4: Elaborate
# =============================================================================
puts "\n=== Step 4: Elaboration (xelab) ==="
set xelab_cmd "xelab work.tb_golden_check -s sim_snapshot -timescale 1ns/1ps -log $sim_dir/xelab.log -debug typical -relax"
puts "Command: $xelab_cmd"
eval exec $xelab_cmd

# =============================================================================
# Step 5: Run simulation
# =============================================================================
puts "\n=== Step 5: Running simulation (xsim) ==="
set xsim_cmd "xsim sim_snapshot -runall -log $sim_dir/xsim.log"
puts "Command: $xsim_cmd"
eval exec $xsim_cmd

puts "\n=== Simulation complete. Check $sim_dir/xsim.log for results ==="
