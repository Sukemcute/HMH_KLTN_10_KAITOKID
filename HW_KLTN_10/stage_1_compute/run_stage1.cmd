@echo off
setlocal
REM Stage 1: Compile + run 4 compute-atom testbenches (dsp_pair, pe_unit, col_reduce, comp_tree)
set "VIVADO_ROOT=C:\Xilinx\Vivado\2022.2"
if not exist "%VIVADO_ROOT%\settings64.bat" (echo Set VIVADO_ROOT & exit /b 1)
call "%VIVADO_ROOT%\settings64.bat"

set "HW=%~dp0.."
if "%HW:~-1%"=="\" set "HW=%HW:~0,-1%"
set "S0=%HW%\stage_0_packages\rtl"
set "S1=%HW%\stage_1_compute\rtl"
set "TB=%HW%\stage_1_compute\tb"
set "PASS=0"
set "TOTAL=4"

cd /d "%~dp0"

REM ── 1/4 tb_dsp_pair_int8 ──
echo.
echo ============================================================
echo  1/4  tb_dsp_pair_int8
echo ============================================================
call xvlog --sv "%S0%\accel_pkg.sv" "%S1%\dsp_pair_int8.sv" "%TB%\tb_dsp_pair_int8.sv"
if errorlevel 1 (echo [FAIL] xvlog tb_dsp_pair_int8 & goto :TB2)
call xelab work.tb_dsp_pair_int8 -s sim_dsp -timescale 1ns/1ps
if errorlevel 1 (echo [FAIL] xelab tb_dsp_pair_int8 & goto :TB2)
call xsim sim_dsp -R -log sim_dsp.log
findstr /C:"ALL" sim_dsp.log | findstr /C:"PASS" >nul && set /a PASS+=1

:TB2
REM ── 2/4 tb_pe_unit ──
echo.
echo ============================================================
echo  2/4  tb_pe_unit
echo ============================================================
call xvlog --sv "%S0%\accel_pkg.sv" "%S1%\dsp_pair_int8.sv" "%S1%\pe_unit.sv" "%TB%\tb_pe_unit.sv"
if errorlevel 1 (echo [FAIL] xvlog tb_pe_unit & goto :TB3)
call xelab work.tb_pe_unit -s sim_pe -timescale 1ns/1ps
if errorlevel 1 (echo [FAIL] xelab tb_pe_unit & goto :TB3)
call xsim sim_pe -R -log sim_pe.log
findstr /C:"ALL" sim_pe.log | findstr /C:"PASS" >nul && set /a PASS+=1

:TB3
REM ── 3/4 tb_column_reduce ──
echo.
echo ============================================================
echo  3/4  tb_column_reduce
echo ============================================================
call xvlog --sv "%S0%\accel_pkg.sv" "%S1%\column_reduce.sv" "%TB%\tb_column_reduce.sv"
if errorlevel 1 (echo [FAIL] xvlog tb_column_reduce & goto :TB4)
call xelab work.tb_column_reduce -s sim_col -timescale 1ns/1ps
if errorlevel 1 (echo [FAIL] xelab tb_column_reduce & goto :TB4)
call xsim sim_col -R -log sim_col.log
findstr /C:"ALL" sim_col.log | findstr /C:"PASS" >nul && set /a PASS+=1

:TB4
REM ── 4/4 tb_comparator_tree ──
echo.
echo ============================================================
echo  4/4  tb_comparator_tree
echo ============================================================
call xvlog --sv "%S0%\accel_pkg.sv" "%S1%\comparator_tree.sv" "%TB%\tb_comparator_tree.sv"
if errorlevel 1 (echo [FAIL] xvlog tb_comparator_tree & goto :SUMMARY)
call xelab work.tb_comparator_tree -s sim_comp -timescale 1ns/1ps
if errorlevel 1 (echo [FAIL] xelab tb_comparator_tree & goto :SUMMARY)
call xsim sim_comp -R -log sim_comp.log
findstr /C:"ALL" sim_comp.log | findstr /C:"PASS" >nul && set /a PASS+=1

:SUMMARY
echo.
echo ============================================================
echo  STAGE 1 SUMMARY:  %PASS% / %TOTAL% testbenches PASSED
echo ============================================================
if %PASS%==%TOTAL% (echo [STAGE 1 PASS]) else (echo [STAGE 1 FAIL])
pause
exit /b 0
