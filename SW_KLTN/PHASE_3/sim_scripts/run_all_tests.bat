@echo off
REM run_all_tests.bat — Compile and run all PHASE_3 testbenches sequentially
REM Usage: Open Vivado Tcl shell, then: source run_all_tests.tcl
REM Or run from Windows CMD: vivado -mode batch -source run_all_tests.tcl

SET BASE=E:/KLTN_HMH_FINAL/SW_KLTN/PHASE_3
echo ============================================
echo  PHASE 3 — Full Test Suite
echo ============================================

REM --- Step 0: Compile all RTL ---
echo [COMPILE] All RTL modules...
call :compile_all
if errorlevel 1 goto :error

REM --- Step 1: Compute Leaf Tests ---
call :run_test "01_compute_leaf" "tb_dsp_pair_int8" "sim_dsp"
call :run_test "01_compute_leaf" "tb_pe_unit" "sim_pe"
call :run_test "01_compute_leaf" "tb_column_reduce" "sim_colred"
call :run_test "01_compute_leaf" "tb_comparator_tree" "sim_comptree"
call :run_test "01_compute_leaf" "tb_silu_lut" "sim_silu"

REM --- Step 2: PPU ---
call :run_test "02_ppu" "tb_ppu" "sim_ppu"

REM --- Step 3: Memory ---
call :run_test "03_memory" "tb_glb_input_bank" "sim_glb_in"
call :run_test "03_memory" "tb_addr_gen_input" "sim_agi"
call :run_test "03_memory" "tb_metadata_ram" "sim_meta"
call :run_test "03_memory" "tb_glb_output_bank" "sim_glb_out"
call :run_test "03_memory" "tb_glb_weight_bank" "sim_glb_wt"
call :run_test "03_memory" "tb_addr_gen_output" "sim_ago"
call :run_test "03_memory" "tb_addr_gen_weight" "sim_agw"

REM --- Step 4: Data Movement ---
call :run_test "04_data_movement" "tb_window_gen" "sim_win"
call :run_test "04_data_movement" "tb_router_cluster" "sim_router"
call :run_test "04_data_movement" "tb_swizzle_engine" "sim_swizzle"

REM --- Step 5: Integration ---
call :run_test "05_integration" "tb_pe_cluster" "sim_peclust"
call :run_test "05_integration" "tb_shadow_reg_file" "sim_shadow"

REM --- Step 6: Control ---
call :run_test "06_control" "tb_barrier_manager" "sim_barrier"
call :run_test "06_control" "tb_tile_fsm" "sim_tfsm"
call :run_test "06_control" "tb_local_arbiter" "sim_arb"
call :run_test "06_control" "tb_desc_fetch_engine" "sim_fetch"
call :run_test "06_control" "tb_global_scheduler" "sim_gsched"

REM --- Step 7: System ---
call :run_test "07_system" "tb_tensor_dma" "sim_dma"
call :run_test "07_system" "tb_accel_top" "sim_top"

echo ============================================
echo  ALL TESTS COMPLETE
echo ============================================
goto :eof

:compile_all
xvlog -sv %BASE%/packages/accel_pkg.sv
xvlog -sv %BASE%/packages/desc_pkg.sv
xvlog -sv %BASE%/packages/csr_pkg.sv
for %%d in (01_compute_leaf 02_ppu 03_memory 04_data_movement 05_integration 06_control 07_system) do (
  for %%f in (%BASE%/%%d/rtl/*.sv) do xvlog -sv %%f
)
goto :eof

:run_test
echo [TEST] %~2
xvlog -sv %BASE%/%~1/tb/%~2.sv
xelab %~2 -s %~3
xsim %~3 -runall
echo.
goto :eof

:error
echo [ERROR] Compilation failed!
exit /b 1
