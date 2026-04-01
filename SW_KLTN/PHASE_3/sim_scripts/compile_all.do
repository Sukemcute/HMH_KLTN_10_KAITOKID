# compile_all.do — Vivado xvlog compile script for all PHASE_3 modules
# Usage in Vivado Tcl: source compile_all.do
# Or from shell: vivado -mode batch -source compile_all.do

set BASE "E:/KLTN_HMH_FINAL/SW_KLTN/PHASE_3"

# Clean previous artifacts
file delete -force xsim.dir .Xil xvlog.pb

# ═══════════════════════════════════════════════
# STEP 0: Packages (compile order matters!)
# ═══════════════════════════════════════════════
exec xvlog -sv $BASE/packages/accel_pkg.sv
exec xvlog -sv $BASE/packages/desc_pkg.sv
exec xvlog -sv $BASE/packages/csr_pkg.sv

# ═══════════════════════════════════════════════
# STEP 1: Compute Leaf
# ═══════════════════════════════════════════════
exec xvlog -sv $BASE/01_compute_leaf/rtl/dsp_pair_int8.sv
exec xvlog -sv $BASE/01_compute_leaf/rtl/pe_unit.sv
exec xvlog -sv $BASE/01_compute_leaf/rtl/column_reduce.sv
exec xvlog -sv $BASE/01_compute_leaf/rtl/comparator_tree.sv
exec xvlog -sv $BASE/01_compute_leaf/rtl/silu_lut.sv

# ═══════════════════════════════════════════════
# STEP 2: PPU
# ═══════════════════════════════════════════════
exec xvlog -sv $BASE/02_ppu/rtl/ppu.sv

# ═══════════════════════════════════════════════
# STEP 3: Memory
# ═══════════════════════════════════════════════
exec xvlog -sv $BASE/03_memory/rtl/glb_input_bank.sv
exec xvlog -sv $BASE/03_memory/rtl/glb_weight_bank.sv
exec xvlog -sv $BASE/03_memory/rtl/glb_output_bank.sv
exec xvlog -sv $BASE/03_memory/rtl/metadata_ram.sv
exec xvlog -sv $BASE/03_memory/rtl/addr_gen_input.sv
exec xvlog -sv $BASE/03_memory/rtl/addr_gen_weight.sv
exec xvlog -sv $BASE/03_memory/rtl/addr_gen_output.sv

# ═══════════════════════════════════════════════
# STEP 4: Data Movement
# ═══════════════════════════════════════════════
exec xvlog -sv $BASE/04_data_movement/rtl/window_gen.sv
exec xvlog -sv $BASE/04_data_movement/rtl/router_cluster.sv
exec xvlog -sv $BASE/04_data_movement/rtl/swizzle_engine.sv

# ═══════════════════════════════════════════════
# STEP 5: Integration
# ═══════════════════════════════════════════════
exec xvlog -sv $BASE/05_integration/rtl/pe_cluster.sv
exec xvlog -sv $BASE/05_integration/rtl/shadow_reg_file.sv
exec xvlog -sv $BASE/05_integration/rtl/subcluster_wrapper.sv

# ═══════════════════════════════════════════════
# STEP 6: Control
# ═══════════════════════════════════════════════
exec xvlog -sv $BASE/06_control/rtl/barrier_manager.sv
exec xvlog -sv $BASE/06_control/rtl/tile_fsm.sv
exec xvlog -sv $BASE/06_control/rtl/local_arbiter.sv
exec xvlog -sv $BASE/06_control/rtl/desc_fetch_engine.sv
exec xvlog -sv $BASE/06_control/rtl/global_scheduler.sv

# ═══════════════════════════════════════════════
# STEP 7: System
# ═══════════════════════════════════════════════
exec xvlog -sv $BASE/07_system/rtl/tensor_dma.sv
exec xvlog -sv $BASE/07_system/rtl/controller_system.sv
exec xvlog -sv $BASE/07_system/rtl/supercluster_wrapper.sv
exec xvlog -sv $BASE/07_system/rtl/accel_top.sv

puts "═══════════════════════════════════════════════════"
puts "  ALL RTL MODULES COMPILED SUCCESSFULLY"
puts "═══════════════════════════════════════════════════"
