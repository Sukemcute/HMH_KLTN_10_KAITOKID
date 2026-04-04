@echo off
setlocal enabledelayedexpansion
REM Stage 4: Compile + run 3 address-generator testbenches
set "VIVADO_ROOT=C:\Xilinx\Vivado\2022.2"
if not exist "%VIVADO_ROOT%\settings64.bat" (echo Set VIVADO_ROOT & exit /b 1)
call "%VIVADO_ROOT%\settings64.bat"

set "HW=%~dp0.."
if "%HW:~-1%"=="\" set "HW=%HW:~0,-1%"
set "S0=%HW%\stage_0_packages\rtl"
set "S4=%HW%\stage_4_addr_gen"
set "PASS=0"
set "TOTAL=3"

cd /d "%S4%"

REM ── 1/3 tb_addr_gen_input ──
echo.
echo ============================================================
echo  1/3  tb_addr_gen_input
echo ============================================================
call xvlog --sv "%S0%\accel_pkg.sv" "%S0%\desc_pkg.sv" "%S0%\csr_pkg.sv" "%S4%\rtl\addr_gen_input.sv" "%S4%\tb\tb_addr_gen_input.sv"
if errorlevel 1 (echo [FAIL] xvlog & goto :TB2)
call xelab work.tb_addr_gen_input -s sim_agi -timescale 1ns/1ps
if errorlevel 1 (echo [FAIL] xelab & goto :TB2)
call xsim sim_agi -R -log sim_agi.log
findstr /C:"ALL" sim_agi.log | findstr /C:"PASS" >nul && set /a PASS+=1

:TB2
echo.
echo ============================================================
echo  2/3  tb_addr_gen_weight
echo ============================================================
call xvlog --sv "%S0%\accel_pkg.sv" "%S0%\desc_pkg.sv" "%S0%\csr_pkg.sv" "%S4%\rtl\addr_gen_weight.sv" "%S4%\tb\tb_addr_gen_weight.sv"
if errorlevel 1 (echo [FAIL] xvlog & goto :TB3)
call xelab work.tb_addr_gen_weight -s sim_agw -timescale 1ns/1ps
if errorlevel 1 (echo [FAIL] xelab & goto :TB3)
call xsim sim_agw -R -log sim_agw.log
findstr /C:"ALL" sim_agw.log | findstr /C:"PASS" >nul && set /a PASS+=1

:TB3
echo.
echo ============================================================
echo  3/3  tb_addr_gen_output
echo ============================================================
call xvlog --sv "%S0%\accel_pkg.sv" "%S0%\desc_pkg.sv" "%S0%\csr_pkg.sv" "%S4%\rtl\addr_gen_output.sv" "%S4%\tb\tb_addr_gen_output.sv"
if errorlevel 1 (echo [FAIL] xvlog & goto :SUMMARY)
call xelab work.tb_addr_gen_output -s sim_ago -timescale 1ns/1ps
if errorlevel 1 (echo [FAIL] xelab & goto :SUMMARY)
call xsim sim_ago -R -log sim_ago.log
findstr /C:"ALL" sim_ago.log | findstr /C:"PASS" >nul && set /a PASS+=1

:SUMMARY
echo.
echo ============================================================
echo  STAGE 4 SUMMARY:  !PASS! / !TOTAL! testbenches PASSED
echo ============================================================
if !PASS!==!TOTAL! (echo [STAGE 4 PASS]) else (echo [STAGE 4 FAIL])
pause
exit /b 0
