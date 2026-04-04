@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM ── Add Vivado to PATH ──
call C:\Xilinx\Vivado\2022.2\settings64.bat

echo ============================================================
echo  Stage 8: Primitive Cosimulation  (RS3 / OS1 / DW3 / MP5)
echo ============================================================

REM ── Cleanup stale artefacts ──
taskkill /F /IM xsim.exe 2>nul
rd /s /q xsim.dir 2>nul
rd /s /q .Xil 2>nul
del xvlog_s8.log xelab_s8.log sim_stage8.log 2>nul

REM ── Step 0: Generate golden vectors ──
echo.
echo [STEP 0] Generating golden test vectors ...
pushd golden
python gen_vectors.py --out ..\vectors
if errorlevel 1 (
    echo ERROR: gen_vectors.py failed
    exit /b 1
)
popd
echo [STEP 0] DONE

REM ── Step 1: Compile RTL + TB ──
echo.
echo [STEP 1] xvlog — compiling all SystemVerilog ...

set HW=..\
set SRC=

REM Packages first
set SRC=%SRC% %HW%stage_0_packages\rtl\accel_pkg.sv
set SRC=%SRC% %HW%stage_0_packages\rtl\desc_pkg.sv
set SRC=%SRC% %HW%stage_0_packages\rtl\csr_pkg.sv
set SRC=%SRC% %HW%stage_0_packages\rtl\rtl_trace_pkg.sv

REM Stage 1: Compute atoms
set SRC=%SRC% %HW%stage_1_compute\rtl\dsp_pair_int8.sv
set SRC=%SRC% %HW%stage_1_compute\rtl\pe_unit.sv
set SRC=%SRC% %HW%stage_1_compute\rtl\column_reduce.sv
set SRC=%SRC% %HW%stage_1_compute\rtl\comparator_tree.sv

REM Stage 2: PPU
set SRC=%SRC% %HW%stage_2_ppu\rtl\ppu.sv

REM Stage 3: Memory
set SRC=%SRC% %HW%stage_3_memory\rtl\glb_input_bank_db.sv
set SRC=%SRC% %HW%stage_3_memory\rtl\glb_weight_bank.sv
set SRC=%SRC% %HW%stage_3_memory\rtl\glb_output_bank.sv
set SRC=%SRC% %HW%stage_3_memory\rtl\metadata_ram.sv

REM Stage 4: Address gen
set SRC=%SRC% %HW%stage_4_addr_gen\rtl\addr_gen_input.sv
set SRC=%SRC% %HW%stage_4_addr_gen\rtl\addr_gen_weight.sv
set SRC=%SRC% %HW%stage_4_addr_gen\rtl\addr_gen_output.sv

REM Stage 5: Data movement
set SRC=%SRC% %HW%stage_5_data_movement\rtl\router_cluster_v2.sv
set SRC=%SRC% %HW%stage_5_data_movement\rtl\window_gen.sv
set SRC=%SRC% %HW%stage_5_data_movement\rtl\swizzle_engine.sv

REM Stage 6: Control
set SRC=%SRC% %HW%stage_6_control\rtl\tile_fsm.sv
set SRC=%SRC% %HW%stage_6_control\rtl\shadow_reg_file.sv
set SRC=%SRC% %HW%stage_6_control\rtl\compute_sequencer.sv

REM Stage 7: Integration
set SRC=%SRC% %HW%stage_7_subcluster\rtl\pe_cluster_v4.sv
set SRC=%SRC% %HW%stage_7_subcluster\rtl\subcluster_datapath.sv

REM Stage 8: Testbench
set SRC=%SRC% tb\tb_stage8_cosim.sv

cmd /c "xvlog --sv -d S8_DBG -d RTL_TRACE %SRC% -log xvlog_s8.log"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: xvlog failed — see xvlog_s8.log
    exit /b 1
)
echo [STEP 1] xvlog DONE

REM ── Step 2: Elaborate ──
echo.
echo [STEP 2] xelab — elaborating tb_stage8_cosim ...
cmd /c "xelab work.tb_stage8_cosim -s sim_s8 -timescale 1ns/1ps -log xelab_s8.log"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: xelab failed — see xelab_s8.log
    exit /b 1
)
echo [STEP 2] xelab DONE

REM ── Step 3: Simulate ──
echo.
echo [STEP 3] xsim — running simulation ...
cmd /c "xsim sim_s8 -R -log sim_stage8.log"
echo [STEP 3] xsim DONE

REM ── Step 4: Check results ──
echo.
echo ============================================================
if exist sim_stage8.log (
    findstr /C:"ALL TESTS PASSED" sim_stage8.log >nul 2>&1
    if not errorlevel 1 (
        echo   *** ALL TESTS PASSED ***
    ) else (
        echo   *** SOME TESTS FAILED — see sim_stage8.log ***
        findstr /C:"FAIL" sim_stage8.log
    )
) else (
    echo   ERROR: sim_stage8.log not found
)
echo ============================================================
