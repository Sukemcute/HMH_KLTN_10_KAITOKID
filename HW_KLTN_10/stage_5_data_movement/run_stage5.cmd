@echo off
setlocal enabledelayedexpansion
REM Stage 5: Compile + run 3 data-movement testbenches
set "VIVADO_ROOT=C:\Xilinx\Vivado\2022.2"
if not exist "%VIVADO_ROOT%\settings64.bat" (echo Set VIVADO_ROOT & exit /b 1)
call "%VIVADO_ROOT%\settings64.bat"

set "HW=%~dp0.."
if "%HW:~-1%"=="\" set "HW=%HW:~0,-1%"
set "S0=%HW%\stage_0_packages\rtl"
set "S5=%HW%\stage_5_data_movement"
set "PASS=0"
set "TOTAL=3"

cd /d "%S5%"

REM ── 1/3 tb_router_cluster_v2 ──
echo.
echo ============================================================
echo  1/3  tb_router_cluster_v2
echo ============================================================
call xvlog --sv "%S0%\accel_pkg.sv" "%S0%\desc_pkg.sv" "%S0%\csr_pkg.sv" "%S5%\rtl\router_cluster_v2.sv" "%S5%\tb\tb_router_cluster_v2.sv"
if errorlevel 1 (echo [FAIL] xvlog & goto :TB2)
call xelab work.tb_router_cluster_v2 -s sim_rcv2 -timescale 1ns/1ps
if errorlevel 1 (echo [FAIL] xelab & goto :TB2)
call xsim sim_rcv2 -R -log sim_rcv2.log
findstr /C:"ALL" sim_rcv2.log | findstr /C:"PASS" >nul && set /a PASS+=1

:TB2
echo.
echo ============================================================
echo  2/3  tb_window_gen
echo ============================================================
call xvlog --sv "%S0%\accel_pkg.sv" "%S0%\desc_pkg.sv" "%S0%\csr_pkg.sv" "%S5%\rtl\window_gen.sv" "%S5%\tb\tb_window_gen.sv"
if errorlevel 1 (echo [FAIL] xvlog & goto :TB3)
call xelab work.tb_window_gen -s sim_wg -timescale 1ns/1ps
if errorlevel 1 (echo [FAIL] xelab & goto :TB3)
call xsim sim_wg -R -log sim_wg.log
findstr /C:"ALL" sim_wg.log | findstr /C:"PASS" >nul && set /a PASS+=1

:TB3
echo.
echo ============================================================
echo  3/3  tb_swizzle_engine
echo ============================================================
call xvlog --sv "%S0%\accel_pkg.sv" "%S0%\desc_pkg.sv" "%S0%\csr_pkg.sv" "%S5%\rtl\swizzle_engine.sv" "%S5%\tb\tb_swizzle_engine.sv"
if errorlevel 1 (echo [FAIL] xvlog & goto :SUMMARY)
call xelab work.tb_swizzle_engine -s sim_swz -timescale 1ns/1ps
if errorlevel 1 (echo [FAIL] xelab & goto :SUMMARY)
call xsim sim_swz -R -log sim_swz.log
findstr /C:"ALL" sim_swz.log | findstr /C:"PASS" >nul && set /a PASS+=1

:SUMMARY
echo.
echo ============================================================
echo  STAGE 5 SUMMARY:  !PASS! / !TOTAL! testbenches PASSED
echo ============================================================
if !PASS!==!TOTAL! (echo [STAGE 5 PASS]) else (echo [STAGE 5 FAIL])
pause
exit /b 0
