# =============================================================
#  PPU Golden Verification TCL Script
#
#  Usage: In Vivado Tcl console:
#    source E:/KLTN_HMH_FINAL/SW_KLTN/PHASE_4/03_rtl_cosim/run_ppu_golden.tcl
#
#  Then:
#    run_ppu_test 0
#    run_ppu_test 1
#    run_ppu_test 3
#    run_ppu_test 17
# =============================================================

set BASE "E:/KLTN_HMH_FINAL/SW_KLTN"

proc compile_ppu {} {
    global BASE
    puts "=== Cleaning ==="
    file delete -force xsim.dir
    file delete -force .Xil
    file delete -force xvlog.pb

    puts "=== Compiling packages ==="
    exec xvlog -sv $BASE/PHASE_3/packages/accel_pkg.sv
    exec xvlog -sv $BASE/PHASE_3/packages/desc_pkg.sv

    puts "=== Compiling PPU RTL ==="
    exec xvlog -sv $BASE/PHASE_3/02_ppu/rtl/ppu.sv

    puts "=== Compiling testbench ==="
    exec xvlog -sv $BASE/PHASE_4/03_rtl_cosim/tb_ppu_golden.sv

    puts "=== Compilation done ==="
}

proc run_ppu_test {layer_idx} {
    set snap "sim_ppu_L${layer_idx}"
    puts ""
    puts "========================================="
    puts " Elaborating Layer $layer_idx -> $snap"
    puts "========================================="
    exec xelab tb_ppu_golden -s $snap -generic_top "LAYER_IDX=$layer_idx"

    puts "========================================="
    puts " Running simulation..."
    puts "========================================="
    exec xsim $snap -runall
}

compile_ppu

puts ""
puts "============================================="
puts " Ready! Run tests with:"
puts "   run_ppu_test 0"
puts "   run_ppu_test 1"
puts "   run_ppu_test 3"
puts "   run_ppu_test 17"
puts "============================================="
