# ============================================================================
# PHASE_10/stage_3 — Compile & Simulate All Primitive Engines
# Purpose: Verify computation correctness of each primitive type
#          These are "golden computation verifiers", NOT final hardware.
#          Final hardware = 1 subcluster (Stage 5) config qua descriptor.
# Target: Vivado xvlog / xelab / xsim
# Usage: vivado -mode batch -source compile_all.do
# ============================================================================

set S2 "D:/HMH_KLTN/PHASE_10/stage_2"
set S3 "D:/HMH_KLTN/PHASE_10/stage_3"

file delete -force xsim.dir .Xil xvlog.pb xelab.pb xsim.pb webtalk*

puts "================================================================"
puts "  PHASE_10/stage_3: PRIMITIVE ENGINE VERIFICATION"
puts "================================================================"

# ═══════════ STEP 1: Packages (from stage_2) ═══════════
puts "\n--- Packages ---"
exec xvlog -sv $S2/packages/accel_pkg.sv
exec xvlog -sv $S2/packages/desc_pkg.sv
exec xvlog -sv $S2/packages/csr_pkg.sv
exec xvlog -sv $S3/rtl/yolo_accel_pkg.sv

# ═══════════ STEP 2: Compute atoms (from stage_2) ═══════════
puts "\n--- Stage 2 atoms (dependencies) ---"
exec xvlog -sv $S2/01_dsp_pair/rtl/dsp_pair_int8.sv
exec xvlog -sv $S2/02_pe_unit/rtl/pe_unit.sv
exec xvlog -sv $S2/03_column_reduce/rtl/column_reduce.sv
exec xvlog -sv $S2/04_comparator_tree/rtl/comparator_tree.sv
exec xvlog -sv $S2/05_silu_lut/rtl/silu_lut.sv
exec xvlog -sv $S2/06_ppu/rtl/ppu.sv

# ═══════════ STEP 3: Primitive engines (stage_3 RTL) ═══════════
puts "\n--- Stage 3 engines ---"
exec xvlog -sv $S3/rtl/conv3x3_engine.sv
exec xvlog -sv $S3/rtl/conv1x1_engine.sv
exec xvlog -sv $S3/rtl/dwconv3x3_engine.sv
exec xvlog -sv $S3/rtl/maxpool5x5_engine.sv
exec xvlog -sv $S3/rtl/dwconv7x7_engine.sv
exec xvlog -sv $S3/rtl/concat_engine.sv
exec xvlog -sv $S3/rtl/upsample_engine.sv
exec xvlog -sv $S3/rtl/ewise_add_engine.sv

# ═══════════ STEP 4: Testbenches ═══════════
puts "\n--- Testbenches ---"
exec xvlog -sv $S3/tb/tb_conv3x3_golden.sv
exec xvlog -sv $S3/tb/tb_conv1x1_golden.sv
exec xvlog -sv $S3/tb/tb_dwconv3x3_golden.sv
exec xvlog -sv $S3/tb/tb_maxpool5x5_golden.sv
exec xvlog -sv $S3/tb/tb_dwconv7x7_golden.sv
exec xvlog -sv $S3/tb/tb_concat_golden.sv
exec xvlog -sv $S3/tb/tb_upsample_golden.sv
exec xvlog -sv $S3/tb/tb_ewise_add_golden.sv

puts "\n================================================================"
puts "  ALL FILES COMPILED"
puts "================================================================"

# ═══════════ STEP 5: Elaborate & Simulate ═══════════

set tests {
    {tb_conv3x3_golden   snap_p0  "P0 RS_DENSE_3x3"}
    {tb_conv1x1_golden   snap_p1  "P1 OS_1x1"}
    {tb_dwconv3x3_golden snap_p2  "P2 DW_3x3"}
    {tb_maxpool5x5_golden snap_p3 "P3 MAXPOOL_5x5"}
    {tb_dwconv7x7_golden snap_p8  "P8 DW_7x7_MULTIPASS"}
    {tb_concat_golden    snap_p5  "P5 CONCAT"}
    {tb_upsample_golden  snap_p6  "P6 UPSAMPLE_NEAREST"}
    {tb_ewise_add_golden snap_p7  "P7 EWISE_ADD"}
}

set idx 1
set total [llength $tests]
foreach t $tests {
    set tb   [lindex $t 0]
    set snap [lindex $t 1]
    set desc [lindex $t 2]
    puts "\n--- \[$idx/$total\] $desc ($tb) ---"
    exec xelab -debug typical -timescale 1ns/1ps -top $tb -snapshot $snap -log elab_${snap}.log
    exec xsim $snap -runall -log sim_${snap}.log
    incr idx
}

puts "\n================================================================"
puts "  ALL $total PRIMITIVE SIMULATIONS COMPLETE"
puts "================================================================"
puts "  Logs: sim_snap_p0.log .. sim_snap_p7.log"
puts "  Check: grep -E 'PASS|FAIL|ERROR' sim_snap_*.log"
puts "================================================================"
