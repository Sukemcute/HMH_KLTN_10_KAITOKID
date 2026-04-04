@echo off
setlocal enabledelayedexpansion
REM Stage 7: Compile + run 2 integration testbenches (pe_cluster_v4 + subcluster_datapath)
set "VIVADO_ROOT=C:\Xilinx\Vivado\2022.2"
if not exist "%VIVADO_ROOT%\settings64.bat" (echo Set VIVADO_ROOT & exit /b 1)
call "%VIVADO_ROOT%\settings64.bat"

set "HW=%~dp0.."
if "%HW:~-1%"=="\" set "HW=%HW:~0,-1%"
set "S0=%HW%\stage_0_packages\rtl"
set "S1=%HW%\stage_1_compute\rtl"
set "S2=%HW%\stage_2_ppu\rtl"
set "S3=%HW%\stage_3_memory\rtl"
set "S4=%HW%\stage_4_addr_gen\rtl"
set "S5=%HW%\stage_5_data_movement\rtl"
set "S6=%HW%\stage_6_control\rtl"
set "S7=%HW%\stage_7_subcluster"
set "PASS=0"
set "TOTAL=2"

cd /d "%S7%"

REM ── 1/2 tb_pe_cluster_v4 (needs S0 + S1 RTL) ──
echo.
echo ============================================================
echo  1/2  tb_pe_cluster_v4
echo ============================================================
call xvlog --sv -d RTL_TRACE "%S0%\accel_pkg.sv" "%S0%\desc_pkg.sv" "%S0%\csr_pkg.sv" "%S0%\rtl_trace_pkg.sv" "%S1%\dsp_pair_int8.sv" "%S1%\pe_unit.sv" "%S1%\column_reduce.sv" "%S1%\comparator_tree.sv" "%S7%\rtl\pe_cluster_v4.sv" "%S7%\tb\tb_pe_cluster_v4.sv"
if errorlevel 1 (echo [FAIL] xvlog & goto :TB2)
call xelab work.tb_pe_cluster_v4 -s sim_pec -timescale 1ns/1ps
if errorlevel 1 (echo [FAIL] xelab & goto :TB2)
call xsim sim_pec -R -log sim_pec.log
if exist sim_pec.log (findstr /C:"ALL" sim_pec.log | findstr /C:"PASS" >nul && set /a PASS+=1)

:TB2
REM ── 2/2 tb_subcluster_datapath (needs ALL stages 0-7 RTL) ──
echo.
echo ============================================================
echo  2/2  tb_subcluster_datapath  (full integration)
echo ============================================================
call xvlog --sv -d RTL_TRACE "%S0%\accel_pkg.sv" "%S0%\desc_pkg.sv" "%S0%\csr_pkg.sv" "%S0%\rtl_trace_pkg.sv" "%S1%\dsp_pair_int8.sv" "%S1%\pe_unit.sv" "%S1%\column_reduce.sv" "%S1%\comparator_tree.sv" "%S2%\ppu.sv" "%S3%\glb_input_bank_db.sv" "%S3%\glb_weight_bank.sv" "%S3%\glb_output_bank.sv" "%S3%\metadata_ram.sv" "%S4%\addr_gen_input.sv" "%S4%\addr_gen_weight.sv" "%S4%\addr_gen_output.sv" "%S5%\router_cluster_v2.sv" "%S5%\window_gen.sv" "%S5%\swizzle_engine.sv" "%S6%\tile_fsm.sv" "%S6%\shadow_reg_file.sv" "%S6%\compute_sequencer.sv" "%S7%\rtl\pe_cluster_v4.sv" "%S7%\rtl\subcluster_datapath.sv" "%S7%\tb\tb_subcluster_datapath.sv"
if errorlevel 1 (echo [FAIL] xvlog & goto :SUMMARY)
call xelab work.tb_subcluster_datapath -s sim_scd -timescale 1ns/1ps
if errorlevel 1 (echo [FAIL] xelab & goto :SUMMARY)
call xsim sim_scd -R -log sim_scd.log
if exist sim_scd.log (findstr /C:"ALL" sim_scd.log | findstr /C:"PASS" >nul && set /a PASS+=1)

:SUMMARY
echo.
echo ============================================================
echo  STAGE 7 SUMMARY:  !PASS! / !TOTAL! testbenches PASSED
echo ============================================================
if !PASS!==!TOTAL! (echo [STAGE 7 PASS]) else (echo [STAGE 7 FAIL])
pause
exit /b 0
