# ============================================================================
# PHASE_10/stage_4 — Compile & Simulate Memory + Data Movement + Integration
# Purpose: Verify GLB banks, address generators, window gen, router, swizzle
# Target: Vivado xvlog / xelab / xsim
# Usage: vivado -mode batch -source compile_all.do
# ============================================================================

set S2 "D:/HMH_KLTN/PHASE_10/stage_2"
set S4 "D:/HMH_KLTN/PHASE_10/stage_4"

file delete -force xsim.dir .Xil xvlog.pb xelab.pb xsim.pb webtalk*

puts "================================================================"
puts "  PHASE_10/stage_4: MEMORY & DATA MOVEMENT VERIFICATION"
puts "================================================================"

# ═══════════ STEP 1: Packages ═══════════
puts "\n--- Packages ---"
exec xvlog -sv $S2/packages/accel_pkg.sv
exec xvlog -sv $S2/packages/desc_pkg.sv
exec xvlog -sv $S2/packages/csr_pkg.sv

# ═══════════ STEP 2: Stage 2 atoms (dependencies for integration) ═══════════
puts "\n--- Stage 2 atoms ---"
exec xvlog -sv $S2/01_dsp_pair/rtl/dsp_pair_int8.sv
exec xvlog -sv $S2/02_pe_unit/rtl/pe_unit.sv

# ═══════════ STEP 3: Stage 4 RTL ═══════════
puts "\n--- Memory modules ---"
exec xvlog -sv $S4/01_memory/rtl/glb_input_bank.sv
exec xvlog -sv $S4/01_memory/rtl/glb_weight_bank.sv
exec xvlog -sv $S4/01_memory/rtl/glb_output_bank.sv
exec xvlog -sv $S4/01_memory/rtl/metadata_ram.sv

puts "\n--- Address generators ---"
exec xvlog -sv $S4/02_addr_gen/rtl/addr_gen_input.sv
exec xvlog -sv $S4/02_addr_gen/rtl/addr_gen_weight.sv
exec xvlog -sv $S4/02_addr_gen/rtl/addr_gen_output.sv

puts "\n--- Data movement ---"
exec xvlog -sv $S4/03_data_movement/rtl/window_gen.sv
exec xvlog -sv $S4/03_data_movement/rtl/router_cluster.sv
exec xvlog -sv $S4/03_data_movement/rtl/swizzle_engine.sv

# ═══════════ STEP 4: Testbenches ═══════════
puts "\n--- Testbenches ---"
exec xvlog -sv $S4/01_memory/tb/tb_glb_input_bank.sv
exec xvlog -sv $S4/01_memory/tb/tb_glb_weight_bank.sv
exec xvlog -sv $S4/01_memory/tb/tb_glb_output_bank.sv
exec xvlog -sv $S4/01_memory/tb/tb_metadata_ram.sv
exec xvlog -sv $S4/02_addr_gen/tb/tb_addr_gen_input.sv
exec xvlog -sv $S4/02_addr_gen/tb/tb_addr_gen_weight.sv
exec xvlog -sv $S4/02_addr_gen/tb/tb_addr_gen_output.sv
exec xvlog -sv $S4/03_data_movement/tb/tb_window_gen.sv
exec xvlog -sv $S4/03_data_movement/tb/tb_router_cluster.sv
exec xvlog -sv $S4/04_integration_test/tb/tb_addr_bank_integration.sv

puts "\n================================================================"
puts "  ALL FILES COMPILED"
puts "================================================================"

# ═══════════ STEP 5: Simulate ═══════════
set tests {
    {tb_glb_input_bank       snap_gin   "GLB Input Bank"}
    {tb_glb_weight_bank      snap_gwt   "GLB Weight Bank"}
    {tb_glb_output_bank      snap_gout  "GLB Output Bank"}
    {tb_metadata_ram          snap_meta  "Metadata RAM"}
    {tb_addr_gen_input        snap_agi   "Addr Gen Input"}
    {tb_addr_gen_weight       snap_agw   "Addr Gen Weight"}
    {tb_addr_gen_output       snap_ago   "Addr Gen Output"}
    {tb_window_gen            snap_wgen  "Window Gen"}
    {tb_router_cluster        snap_rtr   "Router Cluster"}
    {tb_addr_bank_integration snap_intg  "Addr+Bank Integration"}
}

set idx 1
set total [llength $tests]
foreach t $tests {
    set tb [lindex $t 0]; set snap [lindex $t 1]; set desc [lindex $t 2]
    puts "\n--- \[$idx/$total\] $desc ---"
    exec xelab -debug typical -timescale 1ns/1ps -top $tb -snapshot $snap -log elab_${snap}.log
    exec xsim $snap -runall -log sim_${snap}.log
    incr idx
}

puts "\n================================================================"
puts "  ALL $total SIMULATIONS COMPLETE"
puts "================================================================"
puts "  Check: grep -E 'PASS|FAIL' sim_snap_*.log"
puts "================================================================"
