# compile_all.do — Vivado xvlog/xelab/xsim script for HW_ACCEL engines + testbenches
# Usage:  vivado -mode batch -source compile_all.do
#     or: source compile_all.do   (from Vivado Tcl console)

set ROOT "D:/HMH_KLTN"
set P3   "$ROOT/SW_KLTN/PHASE_3"
set HW   "$ROOT/HW_ACCEL"

# Clean previous artifacts
file delete -force xsim.dir .Xil xvlog.pb xelab.pb xsim.pb webtalk*

# ═══════════════════════════════════════════════════
# STEP 0: PHASE_3 Packages (compile order matters)
# ═══════════════════════════════════════════════════
puts "--- Step 0: PHASE_3 packages ---"
exec xvlog -sv $P3/packages/accel_pkg.sv
exec xvlog -sv $P3/packages/desc_pkg.sv

# ═══════════════════════════════════════════════════
# STEP 1: HW_ACCEL Package
# ═══════════════════════════════════════════════════
puts "--- Step 1: HW_ACCEL package ---"
exec xvlog -sv $HW/packages/yolo_accel_pkg.sv

# ═══════════════════════════════════════════════════
# STEP 2: PHASE_3 Compute-Leaf Modules
# ═══════════════════════════════════════════════════
puts "--- Step 2: PHASE_3 leaf modules ---"
exec xvlog -sv $P3/01_compute_leaf/rtl/dsp_pair_int8.sv
exec xvlog -sv $P3/01_compute_leaf/rtl/pe_unit.sv
exec xvlog -sv $P3/01_compute_leaf/rtl/column_reduce.sv
exec xvlog -sv $P3/01_compute_leaf/rtl/comparator_tree.sv
exec xvlog -sv $P3/01_compute_leaf/rtl/silu_lut.sv

# ═══════════════════════════════════════════════════
# STEP 3: PHASE_3 PPU
# ═══════════════════════════════════════════════════
puts "--- Step 3: PPU ---"
exec xvlog -sv $P3/02_ppu/rtl/ppu.sv

# ═══════════════════════════════════════════════════
# STEP 4: HW_ACCEL Engine RTL
# ═══════════════════════════════════════════════════
puts "--- Step 4: HW_ACCEL engines ---"
exec xvlog -sv $HW/rtl/conv3x3_engine.sv
exec xvlog -sv $HW/rtl/conv1x1_engine.sv
exec xvlog -sv $HW/rtl/dwconv3x3_engine.sv
exec xvlog -sv $HW/rtl/maxpool5x5_engine.sv
exec xvlog -sv $HW/rtl/dwconv7x7_engine.sv

# ═══════════════════════════════════════════════════
# STEP 5: HW_ACCEL Testbenches
# ═══════════════════════════════════════════════════
puts "--- Step 5: HW_ACCEL testbenches ---"
exec xvlog -sv $HW/tb/tb_conv3x3_golden.sv
exec xvlog -sv $HW/tb/tb_conv1x1_golden.sv
exec xvlog -sv $HW/tb/tb_dwconv3x3_golden.sv
exec xvlog -sv $HW/tb/tb_maxpool5x5_golden.sv

puts ""
puts "═══════════════════════════════════════════════════"
puts "  ALL FILES COMPILED SUCCESSFULLY"
puts "═══════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════
# STEP 6: Elaborate & Simulate each testbench
# ═══════════════════════════════════════════════════

# --- Conv3x3 ---
puts "\n--- Elaborate: tb_conv3x3_golden ---"
exec xelab -debug typical -timescale 1ns/1ps \
    -top tb_conv3x3_golden \
    -snapshot snap_conv3x3 \
    -log elab_conv3x3.log

puts "--- Simulate: tb_conv3x3_golden ---"
exec xsim snap_conv3x3 \
    -runall \
    -log sim_conv3x3.log

# --- Conv1x1 ---
puts "\n--- Elaborate: tb_conv1x1_golden ---"
exec xelab -debug typical -timescale 1ns/1ps \
    -top tb_conv1x1_golden \
    -snapshot snap_conv1x1 \
    -log elab_conv1x1.log

puts "--- Simulate: tb_conv1x1_golden ---"
exec xsim snap_conv1x1 \
    -runall \
    -log sim_conv1x1.log

# --- DW Conv3x3 ---
puts "\n--- Elaborate: tb_dwconv3x3_golden ---"
exec xelab -debug typical -timescale 1ns/1ps \
    -top tb_dwconv3x3_golden \
    -snapshot snap_dwconv3x3 \
    -log elab_dwconv3x3.log

puts "--- Simulate: tb_dwconv3x3_golden ---"
exec xsim snap_dwconv3x3 \
    -runall \
    -log sim_dwconv3x3.log

# --- MaxPool5x5 ---
puts "\n--- Elaborate: tb_maxpool5x5_golden ---"
exec xelab -debug typical -timescale 1ns/1ps \
    -top tb_maxpool5x5_golden \
    -snapshot snap_maxpool5x5 \
    -log elab_maxpool5x5.log

puts "--- Simulate: tb_maxpool5x5_golden ---"
exec xsim snap_maxpool5x5 \
    -runall \
    -log sim_maxpool5x5.log

puts ""
puts "═══════════════════════════════════════════════════"
puts "  ALL SIMULATIONS COMPLETE"
puts "═══════════════════════════════════════════════════"
puts "  Logs: sim_conv3x3.log, sim_conv1x1.log,"
puts "        sim_dwconv3x3.log, sim_maxpool5x5.log"
puts "═══════════════════════════════════════════════════"
