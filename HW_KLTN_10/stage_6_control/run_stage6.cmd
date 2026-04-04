@echo off
setlocal enabledelayedexpansion
REM Stage 6: Compile + run 3 control testbenches (includes pe_clear_acc fix)
set "VIVADO_ROOT=C:\Xilinx\Vivado\2022.2"
if not exist "%VIVADO_ROOT%\settings64.bat" (echo Set VIVADO_ROOT & exit /b 1)
call "%VIVADO_ROOT%\settings64.bat"

set "HW=%~dp0.."
if "%HW:~-1%"=="\" set "HW=%HW:~0,-1%"
set "S0=%HW%\stage_0_packages\rtl"
set "S6=%HW%\stage_6_control"
set "PASS=0"
set "TOTAL=3"

cd /d "%S6%"

REM ── 1/3 tb_tile_fsm ──
echo.
echo ============================================================
echo  1/3  tb_tile_fsm
echo ============================================================
call xvlog --sv "%S0%\accel_pkg.sv" "%S0%\desc_pkg.sv" "%S0%\csr_pkg.sv" "%S6%\rtl\tile_fsm.sv" "%S6%\tb\tb_tile_fsm.sv"
if errorlevel 1 (echo [FAIL] xvlog & goto :TB2)
REM -mt off: avoid Vivado 2022.2 xelab EXCEPTION_ACCESS_VIOLATION on this TB
call xelab -mt off work.tb_tile_fsm -s sim_tfsm -timescale 1ns/1ps
if errorlevel 1 (echo [FAIL] xelab & goto :TB2)
call xsim sim_tfsm -R -log sim_tfsm.log
if exist sim_tfsm.log (findstr /C:"ALL" sim_tfsm.log | findstr /C:"PASS" >nul && set /a PASS+=1)

:TB2
echo.
echo ============================================================
echo  2/3  tb_shadow_reg_file
echo ============================================================
call xvlog --sv "%S0%\accel_pkg.sv" "%S0%\desc_pkg.sv" "%S0%\csr_pkg.sv" "%S6%\rtl\shadow_reg_file.sv" "%S6%\tb\tb_shadow_reg_file.sv"
if errorlevel 1 (echo [FAIL] xvlog & goto :TB3)
call xelab work.tb_shadow_reg_file -s sim_srf -timescale 1ns/1ps
if errorlevel 1 (echo [FAIL] xelab & goto :TB3)
call xsim sim_srf -R -log sim_srf.log
if exist sim_srf.log (findstr /C:"ALL" sim_srf.log | findstr /C:"PASS" >nul && set /a PASS+=1)

:TB3
echo.
echo ============================================================
echo  3/3  tb_compute_sequencer  (pe_clear_acc fix applied)
echo ============================================================
call xvlog --sv "%S0%\accel_pkg.sv" "%S0%\desc_pkg.sv" "%S0%\csr_pkg.sv" "%S6%\rtl\compute_sequencer.sv" "%S6%\tb\tb_compute_sequencer.sv"
if errorlevel 1 (echo [FAIL] xvlog & goto :SUMMARY)
call xelab work.tb_compute_sequencer -s sim_cseq -timescale 1ns/1ps
if errorlevel 1 (echo [FAIL] xelab & goto :SUMMARY)
call xsim sim_cseq -R -log sim_cseq.log
if exist sim_cseq.log (findstr /C:"ALL" sim_cseq.log | findstr /C:"PASS" >nul && set /a PASS+=1)

:SUMMARY
echo.
echo ============================================================
echo  STAGE 6 SUMMARY:  !PASS! / !TOTAL! testbenches PASSED
echo ============================================================
if !PASS!==!TOTAL! (echo [STAGE 6 PASS]) else (echo [STAGE 6 FAIL])
pause
exit /b 0
