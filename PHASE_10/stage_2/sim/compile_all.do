# ============================================================================
# PHASE_10 — Compile & Simulate All Compute Atoms
# Target: Vivado xvlog / xelab / xsim
# Usage: vivado -mode batch -source compile_all.do
# ============================================================================

set ROOT "D:/HMH_KLTN/PHASE_10/stage_2"

# Clean
file delete -force xsim.dir .Xil xvlog.pb xelab.pb xsim.pb webtalk*

puts "================================================================"
puts "  PHASE_10: STAGE 2 — COMPUTE ATOMS BUILD & VERIFY"
puts "================================================================"

# ═══════════ STEP 1: Packages ═══════════
puts "\n--- Compiling packages ---"
exec xvlog -sv $ROOT/packages/accel_pkg.sv
exec xvlog -sv $ROOT/packages/desc_pkg.sv
exec xvlog -sv $ROOT/packages/csr_pkg.sv

# ═══════════ STEP 2: RTL Modules ═══════════
puts "\n--- Compiling RTL modules ---"
exec xvlog -sv $ROOT/01_dsp_pair/rtl/dsp_pair_int8.sv
exec xvlog -sv $ROOT/02_pe_unit/rtl/pe_unit.sv
exec xvlog -sv $ROOT/03_column_reduce/rtl/column_reduce.sv
exec xvlog -sv $ROOT/04_comparator_tree/rtl/comparator_tree.sv
exec xvlog -sv $ROOT/05_silu_lut/rtl/silu_lut.sv
exec xvlog -sv $ROOT/06_ppu/rtl/ppu.sv

# ═══════════ STEP 3: Testbenches ═══════════
puts "\n--- Compiling testbenches ---"
exec xvlog -sv $ROOT/01_dsp_pair/tb/tb_dsp_pair_int8.sv
exec xvlog -sv $ROOT/02_pe_unit/tb/tb_pe_unit.sv
exec xvlog -sv $ROOT/03_column_reduce/tb/tb_column_reduce.sv
exec xvlog -sv $ROOT/04_comparator_tree/tb/tb_comparator_tree.sv
exec xvlog -sv $ROOT/05_silu_lut/tb/tb_silu_lut.sv
exec xvlog -sv $ROOT/06_ppu/tb/tb_ppu.sv

puts "\n================================================================"
puts "  ALL FILES COMPILED SUCCESSFULLY"
puts "================================================================"

# ═══════════ STEP 4: Elaborate & Simulate Each TB ═══════════

# --- 01: dsp_pair_int8 ---
puts "\n--- [1/6] dsp_pair_int8 ---"
exec xelab -debug typical -timescale 1ns/1ps -top tb_dsp_pair_int8 -snapshot snap_dsp -log elab_dsp.log
exec xsim snap_dsp -runall -log sim_dsp_pair.log

# --- 02: pe_unit ---
puts "\n--- [2/6] pe_unit ---"
exec xelab -debug typical -timescale 1ns/1ps -top tb_pe_unit -snapshot snap_pe -log elab_pe.log
exec xsim snap_pe -runall -log sim_pe_unit.log

# --- 03: column_reduce ---
puts "\n--- [3/6] column_reduce ---"
exec xelab -debug typical -timescale 1ns/1ps -top tb_column_reduce -snapshot snap_colred -log elab_colred.log
exec xsim snap_colred -runall -log sim_column_reduce.log

# --- 04: comparator_tree ---
puts "\n--- [4/6] comparator_tree ---"
exec xelab -debug typical -timescale 1ns/1ps -top tb_comparator_tree -snapshot snap_comp -log elab_comp.log
exec xsim snap_comp -runall -log sim_comparator_tree.log

# --- 05: silu_lut ---
puts "\n--- [5/6] silu_lut ---"
exec xelab -debug typical -timescale 1ns/1ps -top tb_silu_lut -snapshot snap_silu -log elab_silu.log
exec xsim snap_silu -runall -log sim_silu_lut.log

# --- 06: ppu ---
puts "\n--- [6/6] ppu (CRITICAL) ---"
exec xelab -debug typical -timescale 1ns/1ps -top tb_ppu -snapshot snap_ppu -log elab_ppu.log
exec xsim snap_ppu -runall -log sim_ppu.log

puts "\n================================================================"
puts "  ALL 6 SIMULATIONS COMPLETE"
puts "================================================================"
puts "  Logs:"
puts "    sim_dsp_pair.log"
puts "    sim_pe_unit.log"
puts "    sim_column_reduce.log"
puts "    sim_comparator_tree.log"
puts "    sim_silu_lut.log"
puts "    sim_ppu.log"
puts "================================================================"
puts "  Grep for PASS/FAIL: grep -E 'PASS|FAIL' sim_*.log"
puts "================================================================"
