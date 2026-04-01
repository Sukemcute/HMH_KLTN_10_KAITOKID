@echo off
REM =============================================================================
REM Vivado xsim simulation for tb_golden_check
REM
REM Usage:
REM   run_vivado_sim.bat          (L0 single-layer verify, default)
REM   run_vivado_sim.bat FULL     (full model L0-L22)
REM   run_vivado_sim.bat DEBUG    (L0 single-layer + Phase-A debug)
REM Stuck xsim.dir / XSIM 43-3345 (xsimk.exe locked): close Vivado GUI, then either
REM   set CLEANXSIM=1 ^& run_vivado_sim.bat
REM or this script now tries taskkill on xsim/xsimk before xelab.
REM =============================================================================

REM Auto-detect Vivado installation
if exist "C:\Xilinx\Vivado\2022.2\bin\xvlog.bat" (
    set VIVADO_BIN=C:\Xilinx\Vivado\2022.2\bin
) else if exist "C:\Xilinx\Vivado\2023.2\bin\xvlog.bat" (
    set VIVADO_BIN=C:\Xilinx\Vivado\2023.2\bin
) else if exist "D:\Vivado\2022.2\bin\xvlog.bat" (
    set VIVADO_BIN=D:\Vivado\2022.2\bin
) else (
    echo ERROR: Cannot find Vivado installation. Set VIVADO_BIN manually.
    pause
    exit /b 1
)
echo Using Vivado: %VIVADO_BIN%
set PATH=%VIVADO_BIN%;%PATH%

set SCRIPT_DIR=%~dp0
set RTL_ROOT=%SCRIPT_DIR%..\..\PHASE_3
set TB_DIR=%SCRIPT_DIR%
set SIM_DIR=%SCRIPT_DIR%vivado_work

REM Parse arguments
set DEFINES=
set MODE=L0
if /i "%1"=="FULL" (
    set DEFINES=-d COSIM_FULL_MODEL
    set MODE=FULL_MODEL
)
if /i "%1"=="DEBUG" (
    set DEFINES=-d ACCEL_PHASE_A_DBG -d ACCEL_DEBUG -d COSIM_DMA_AUDIT
    set MODE=L0_DEBUG
)
if /i "%1"=="FULLDEBUG" (
    set DEFINES=-d COSIM_FULL_MODEL -d ACCEL_PHASE_A_DBG -d ACCEL_DEBUG -d COSIM_DMA_AUDIT
    set MODE=FULL_DEBUG
)

echo ============================================================
echo  Vivado xsim Co-Simulation: %MODE%
echo ============================================================

REM Create work directory
if not exist "%SIM_DIR%" mkdir "%SIM_DIR%"
cd /d "%SIM_DIR%"

REM Optional: full clean if xsim.dir is locked/corrupt — set CLEANXSIM=1 before running
if "%CLEANXSIM%"=="1" if exist "xsim.dir" (
  echo CLEANXSIM=1: removing xsim.dir ...
  rd /s /q "xsim.dir" 2>nul
)

REM =============================================================================
REM  Step 1: Compile all SystemVerilog sources with xvlog
REM =============================================================================
echo.
echo === Step 1: xvlog - Compiling SystemVerilog ===

call xvlog -sv -work work %DEFINES% ^
  "%RTL_ROOT%\packages\accel_pkg.sv" ^
  "%RTL_ROOT%\packages\desc_pkg.sv" ^
  "%RTL_ROOT%\packages\csr_pkg.sv" ^
  "%RTL_ROOT%\01_compute_leaf\rtl\dsp_pair_int8.sv" ^
  "%RTL_ROOT%\01_compute_leaf\rtl\column_reduce.sv" ^
  "%RTL_ROOT%\01_compute_leaf\rtl\comparator_tree.sv" ^
  "%RTL_ROOT%\01_compute_leaf\rtl\pe_unit.sv" ^
  "%RTL_ROOT%\01_compute_leaf\rtl\silu_lut.sv" ^
  "%RTL_ROOT%\02_ppu\rtl\ppu.sv" ^
  "%RTL_ROOT%\03_memory\rtl\addr_gen_input.sv" ^
  "%RTL_ROOT%\03_memory\rtl\addr_gen_output.sv" ^
  "%RTL_ROOT%\03_memory\rtl\addr_gen_weight.sv" ^
  "%RTL_ROOT%\03_memory\rtl\glb_input_bank.sv" ^
  "%RTL_ROOT%\03_memory\rtl\glb_output_bank.sv" ^
  "%RTL_ROOT%\03_memory\rtl\glb_weight_bank.sv" ^
  "%RTL_ROOT%\03_memory\rtl\metadata_ram.sv" ^
  "%RTL_ROOT%\04_data_movement\rtl\router_cluster.sv" ^
  "%RTL_ROOT%\04_data_movement\rtl\swizzle_engine.sv" ^
  "%RTL_ROOT%\04_data_movement\rtl\window_gen.sv" ^
  "%RTL_ROOT%\05_integration\rtl\pe_cluster.sv" ^
  "%RTL_ROOT%\05_integration\rtl\shadow_reg_file.sv" ^
  "%RTL_ROOT%\05_integration\rtl\subcluster_wrapper.sv" ^
  "%RTL_ROOT%\06_control\rtl\barrier_manager.sv" ^
  "%RTL_ROOT%\06_control\rtl\desc_fetch_engine.sv" ^
  "%RTL_ROOT%\06_control\rtl\global_scheduler.sv" ^
  "%RTL_ROOT%\06_control\rtl\local_arbiter.sv" ^
  "%RTL_ROOT%\06_control\rtl\tile_fsm.sv" ^
  "%RTL_ROOT%\07_system\rtl\tensor_dma.sv" ^
  "%RTL_ROOT%\07_system\rtl\controller_system.sv" ^
  "%RTL_ROOT%\07_system\rtl\supercluster_wrapper.sv" ^
  "%RTL_ROOT%\07_system\rtl\accel_top.sv" ^
  "%TB_DIR%\tb_golden_check.sv" ^
  -log xvlog.log 2>&1

if %ERRORLEVEL% neq 0 (
    echo.
    echo *** xvlog FAILED! Check %SIM_DIR%\xvlog.log ***
    echo.
    type xvlog.log
    pause
    exit /b 1
)
echo xvlog OK

REM =============================================================================
REM  Step 2: Elaborate with xelab
REM =============================================================================
echo.
REM Release locks on xsimk.exe (XSIM 43-3345) — harmless if no sim is running.
taskkill /F /IM xsim.exe  >nul 2>&1
taskkill /F /IM xsimk.exe >nul 2>&1
timeout /t 1 /nobreak >nul 2>&1
REM Do NOT delete all of xsim.dir here — it holds work library from xvlog (XSIM 43-3225).
REM Only remove prior elaboration snapshot (fixes XSIM 43-3356 xsim.type write lock).
if exist "xsim.dir\sim_snapshot" (
  echo Removing previous xsim.dir\sim_snapshot ...
  rd /s /q "xsim.dir\sim_snapshot" 2>nul
)
echo.
echo === Step 2: xelab - Elaboration ===

call xelab work.tb_golden_check ^
  -s sim_snapshot ^
  -timescale 1ns/1ps ^
  -debug typical ^
  -relax ^
  -log xelab.log 2>&1

if %ERRORLEVEL% neq 0 (
    echo.
    echo *** xelab FAILED! Check %SIM_DIR%\xelab.log ***
    echo.
    type xelab.log
    pause
    exit /b 1
)
echo xelab OK

REM =============================================================================
REM  Step 3: Run simulation with xsim
REM =============================================================================
echo.
echo === Step 3: xsim - Running simulation ===

call xsim sim_snapshot -runall -log xsim.log 2>&1

if %ERRORLEVEL% neq 0 (
    echo.
    echo *** xsim FAILED! Check %SIM_DIR%\xsim.log ***
    echo.
    type xsim.log
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Simulation finished. Log: %SIM_DIR%\xsim.log
echo ============================================================
echo.

REM Show key results from log
findstr /i "MISMATCH BIT-EXACT TILES_DONE TIMEOUT LAYER-DONE IRQ LAYER_COMPLETE" xsim.log

pause
