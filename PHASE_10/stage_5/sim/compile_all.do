# ============================================================================
# PHASE_10/stage_5 — SUBCLUSTER INTEGRATION: 1 HW config qua descriptor
# Compile & simulate subcluster_datapath with all pe_mode tests
# ============================================================================

set S2 "D:/HMH_KLTN/PHASE_10/stage_2"
set S4 "D:/HMH_KLTN/PHASE_10/stage_4"
set S5 "D:/HMH_KLTN/PHASE_10/stage_5"

file delete -force xsim.dir .Xil xvlog.pb xelab.pb xsim.pb webtalk*

puts "================================================================"
puts "  STAGE 5: SUBCLUSTER — 1 HARDWARE, N DESCRIPTORS"
puts "================================================================"

# ═══════════ Packages ═══════════
puts "\n--- Packages ---"
exec xvlog -sv $S2/packages/accel_pkg.sv
exec xvlog -sv $S2/packages/desc_pkg.sv
exec xvlog -sv $S2/packages/csr_pkg.sv

# ═══════════ Stage 2: Compute Atoms ═══════════
puts "\n--- Compute atoms ---"
exec xvlog -sv $S2/01_dsp_pair/rtl/dsp_pair_int8.sv
exec xvlog -sv $S2/02_pe_unit/rtl/pe_unit.sv
exec xvlog -sv $S2/03_column_reduce/rtl/column_reduce.sv
exec xvlog -sv $S2/04_comparator_tree/rtl/comparator_tree.sv
exec xvlog -sv $S2/05_silu_lut/rtl/silu_lut.sv
exec xvlog -sv $S2/06_ppu/rtl/ppu.sv

# ═══════════ Stage 4: Memory & Data Movement ═══════════
puts "\n--- Memory & routing ---"
exec xvlog -sv $S4/01_memory/rtl/glb_input_bank.sv
exec xvlog -sv $S4/01_memory/rtl/glb_weight_bank.sv
exec xvlog -sv $S4/01_memory/rtl/glb_output_bank.sv
exec xvlog -sv $S4/01_memory/rtl/metadata_ram.sv
exec xvlog -sv $S4/02_addr_gen/rtl/addr_gen_input.sv
exec xvlog -sv $S4/02_addr_gen/rtl/addr_gen_weight.sv
exec xvlog -sv $S4/02_addr_gen/rtl/addr_gen_output.sv
exec xvlog -sv $S4/03_data_movement/rtl/window_gen.sv
exec xvlog -sv $S4/03_data_movement/rtl/router_cluster.sv
exec xvlog -sv $S4/03_data_movement/rtl/swizzle_engine.sv

# ═══════════ Stage 5: Subcluster Integration ═══════════
puts "\n--- Subcluster modules ---"
exec xvlog -sv $S5/rtl/pe_cluster.sv
exec xvlog -sv $S5/rtl/shadow_reg_file.sv
exec xvlog -sv $S5/rtl/tile_fsm.sv
exec xvlog -sv $S5/rtl/compute_sequencer.sv
exec xvlog -sv $S5/rtl/subcluster_datapath.sv

# ═══════════ Testbench ═══════════
puts "\n--- Testbench ---"
exec xvlog -sv $S5/tb/tb_subcluster_modes.sv

puts "\n================================================================"
puts "  ALL FILES COMPILED"
puts "================================================================"

# ═══════════ Elaborate & Simulate ═══════════
puts "\n--- Elaborate: subcluster all-modes test ---"
exec xelab -debug typical -timescale 1ns/1ps \
    -top tb_subcluster_modes \
    -snapshot snap_subcluster \
    -log elab_subcluster.log

puts "\n--- Simulate: 5 pe_mode tests ---"
exec xsim snap_subcluster -runall -log sim_subcluster.log

puts "\n================================================================"
puts "  SUBCLUSTER SIMULATION COMPLETE"
puts "================================================================"
puts "  Log: sim_subcluster.log"
puts "  Check: grep -E 'PASS|FAIL|TEST' sim_subcluster.log"
puts "================================================================"
