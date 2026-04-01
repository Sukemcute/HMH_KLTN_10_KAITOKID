# =============================================================================
# PHASE 4 - RTL Co-Simulation Script (Vivado xsim / xvlog)
# =============================================================================
# Usage (Vivado Tcl console):
#   cd E:/KLTN_HMH_FINAL/SW_KLTN/PHASE_4/03_rtl_cosim
#   source run_cosim.tcl
#
# If you see "couldn't execute xvlog": open Vivado from "Vivado Tcl Shell"
# or set XILINX_VIVADO to your install (e.g. C:/Xilinx/Vivado/2023.2).
# =============================================================================

set PHASE3 "E:/KLTN_HMH_FINAL/SW_KLTN/PHASE_3"
set PHASE4 "E:/KLTN_HMH_FINAL/SW_KLTN/PHASE_4"

# RTL console checkpoints when macro ACCEL_DEBUG is defined (ifdef in RTL).
# Vivado xvlog does NOT accept ModelSim-style "+define+NAME" (it treats that as a file path).
# When non-empty, we pass: xvlog -sv -d ACCEL_DEBUG <file.sv>
# Set to "" for quiet simulation (avoids huge logs on full netlists).
set ACCEL_DBG_DEFINE 1

# Phase A deadlock diagnosis: adds hierarchical monitor in tb_golden_check.sv only.
# See PHASE_A_COSIM_DEBUG.md. Compile with: -d ACCEL_PHASE_A_DBG on that TB file only.
set PHASE_A_DBG_DEFINE 1

# Resolve xvlog: Vivado GUI Tcl sometimes does not have tools on PATH (Windows).
proc _xvlog_bin {} {
  if {[info exists ::env(XILINX_VIVADO)] && $::env(XILINX_VIVADO) ne ""} {
    set vroot [file normalize $::env(XILINX_VIVADO)]
    set win [file join $vroot bin xvlog.exe]
    set uni [file join $vroot bin xvlog]
    if {[file exists $win]} { return $win }
    if {[file exists $uni]} { return $uni }
  }
  return xvlog
}

proc xvlog_sv {path} {
  global ACCEL_DBG_DEFINE PHASE_A_DBG_DEFINE
  set xv [_xvlog_bin]
  if {![file exists $path]} {
    error "xvlog_sv: file not found: $path"
  }
  set vargs [list $xv -sv]
  if {[string trim $ACCEL_DBG_DEFINE] ne ""} {
    lappend vargs -d ACCEL_DEBUG
  }
  # Phase A monitor (tb_golden_check only — avoids touching all RTL compiles)
  if {[string equal -nocase [file tail $path] "tb_golden_check.sv"]} {
    if {[info exists PHASE_A_DBG_DEFINE] && [string trim $PHASE_A_DBG_DEFINE] ne ""} {
      lappend vargs -d ACCEL_PHASE_A_DBG
    }
  }
  lappend vargs $path
  if {[catch {exec {*}$vargs} err]} {
    puts stderr "\n*** xvlog FAILED ***\n  path: $path\n  xvlog: $xv\n  $err\n"
    error $err
  }
}

# ----- Clean previous build -----
file delete -force xsim.dir
file delete -force .Xil
file delete -force xvlog.pb

# =========== STEP 0: Compile all PHASE 3 RTL ===========
puts "=== Compiling PHASE 3 packages ==="
xvlog_sv $PHASE3/packages/accel_pkg.sv
xvlog_sv $PHASE3/packages/desc_pkg.sv
xvlog_sv $PHASE3/packages/csr_pkg.sv

puts "=== Compiling PHASE 3 compute leaf ==="
xvlog_sv $PHASE3/01_compute_leaf/rtl/dsp_pair_int8.sv
xvlog_sv $PHASE3/01_compute_leaf/rtl/pe_unit.sv
xvlog_sv $PHASE3/01_compute_leaf/rtl/column_reduce.sv
xvlog_sv $PHASE3/01_compute_leaf/rtl/comparator_tree.sv
xvlog_sv $PHASE3/01_compute_leaf/rtl/silu_lut.sv

puts "=== Compiling PHASE 3 PPU ==="
xvlog_sv $PHASE3/02_ppu/rtl/ppu.sv

puts "=== Compiling PHASE 3 memory ==="
xvlog_sv $PHASE3/03_memory/rtl/glb_input_bank.sv
xvlog_sv $PHASE3/03_memory/rtl/glb_output_bank.sv
xvlog_sv $PHASE3/03_memory/rtl/glb_weight_bank.sv
xvlog_sv $PHASE3/03_memory/rtl/addr_gen_input.sv
xvlog_sv $PHASE3/03_memory/rtl/addr_gen_output.sv
xvlog_sv $PHASE3/03_memory/rtl/addr_gen_weight.sv
xvlog_sv $PHASE3/03_memory/rtl/metadata_ram.sv

puts "=== Compiling PHASE 3 data movement ==="
xvlog_sv $PHASE3/04_data_movement/rtl/window_gen.sv
xvlog_sv $PHASE3/04_data_movement/rtl/router_cluster.sv
xvlog_sv $PHASE3/04_data_movement/rtl/swizzle_engine.sv

puts "=== Compiling PHASE 3 integration ==="
xvlog_sv $PHASE3/05_integration/rtl/pe_cluster.sv
xvlog_sv $PHASE3/05_integration/rtl/shadow_reg_file.sv
xvlog_sv $PHASE3/05_integration/rtl/subcluster_wrapper.sv

puts "=== Compiling PHASE 3 control ==="
xvlog_sv $PHASE3/06_control/rtl/tile_fsm.sv
xvlog_sv $PHASE3/06_control/rtl/barrier_manager.sv
xvlog_sv $PHASE3/06_control/rtl/local_arbiter.sv
xvlog_sv $PHASE3/06_control/rtl/desc_fetch_engine.sv
xvlog_sv $PHASE3/06_control/rtl/global_scheduler.sv

puts "=== Compiling PHASE 3 system ==="
xvlog_sv $PHASE3/07_system/rtl/tensor_dma.sv
xvlog_sv $PHASE3/07_system/rtl/controller_system.sv
xvlog_sv $PHASE3/07_system/rtl/supercluster_wrapper.sv
xvlog_sv $PHASE3/07_system/rtl/accel_top.sv

# =========== STEP 1: Compile PHASE 4 testbenches ===========
puts "\n=== Compiling PHASE 4 co-simulation testbenches ==="
if {[info exists PHASE_A_DBG_DEFINE] && [string trim $PHASE_A_DBG_DEFINE] ne ""} {
  puts "  (tb_golden_check: ACCEL_PHASE_A_DBG enabled — see PHASE_A_COSIM_DEBUG.md)"
}
xvlog_sv $PHASE4/03_rtl_cosim/tb_golden_check.sv
xvlog_sv $PHASE4/03_rtl_cosim/tb_single_layer.sv
puts "=== Compiling PHASE 3 accel_top smoke TB (CSR / AXI-Lite) ==="
xvlog_sv $PHASE3/07_system/tb/tb_accel_top.sv

# =========== STEP 2: Elaborate ===========
puts "\n=== Choose which testbench to run ==="
puts "  Option A: Single-layer test (recommended first)"
puts "    exec xelab tb_single_layer -s sim_single_layer"
puts "    exec xsim sim_single_layer -runall"
puts ""
puts "  Option B: Full golden co-simulation"
puts "    exec xelab tb_golden_check -s sim_golden"
puts "    exec xsim sim_golden -runall"
puts ""
puts "  Option C: CSR / AXI-Lite smoke (tb_accel_top from PHASE_3)"
puts "    exec xelab tb_accel_top -s sim_accel_smoke"
puts "    exec xsim sim_accel_smoke -runall"
puts ""
puts "  Run the exec commands manually to select."

# For automated single-layer test:
# exec xelab tb_single_layer -s sim_single_layer
# exec xsim sim_single_layer -runall
