@echo off
setlocal enabledelayedexpansion
REM Stage 3: Compile + run 4 memory testbenches
set "VIVADO_ROOT=C:\Xilinx\Vivado\2022.2"
if not exist "%VIVADO_ROOT%\settings64.bat" (echo Set VIVADO_ROOT & exit /b 1)
call "%VIVADO_ROOT%\settings64.bat"

set "HW=%~dp0.."
if "%HW:~-1%"=="\" set "HW=%HW:~0,-1%"
set "S0=%HW%\stage_0_packages\rtl"
set "S3=%HW%\stage_3_memory"
set "PASS=0"
set "TOTAL=4"

cd /d "%S3%"

REM ── 1/4 tb_glb_input_bank_db ──
echo.
echo ============================================================
echo  1/4  tb_glb_input_bank_db
echo ============================================================
call xvlog --sv "%S0%\accel_pkg.sv" "%S0%\desc_pkg.sv" "%S0%\csr_pkg.sv" "%S3%\rtl\glb_input_bank_db.sv" "%S3%\tb\tb_glb_input_bank_db.sv"
if errorlevel 1 (echo [FAIL] xvlog & goto :TB2)
call xelab work.tb_glb_input_bank_db -s sim_gidb -timescale 1ns/1ps
if errorlevel 1 (echo [FAIL] xelab & goto :TB2)
call xsim sim_gidb -R -log sim_gidb.log
findstr /C:"ALL" sim_gidb.log | findstr /C:"PASS" >nul && set /a PASS+=1

:TB2
echo.
echo ============================================================
echo  2/4  tb_glb_weight_bank
echo ============================================================
call xvlog --sv "%S0%\accel_pkg.sv" "%S0%\desc_pkg.sv" "%S0%\csr_pkg.sv" "%S3%\rtl\glb_weight_bank.sv" "%S3%\tb\tb_glb_weight_bank.sv"
if errorlevel 1 (echo [FAIL] xvlog & goto :TB3)
call xelab work.tb_glb_weight_bank -s sim_gwb -timescale 1ns/1ps
if errorlevel 1 (echo [FAIL] xelab & goto :TB3)
call xsim sim_gwb -R -log sim_gwb.log
findstr /C:"ALL" sim_gwb.log | findstr /C:"PASS" >nul && set /a PASS+=1

:TB3
echo.
echo ============================================================
echo  3/4  tb_glb_output_bank
echo ============================================================
call xvlog --sv "%S0%\accel_pkg.sv" "%S0%\desc_pkg.sv" "%S0%\csr_pkg.sv" "%S3%\rtl\glb_output_bank.sv" "%S3%\tb\tb_glb_output_bank.sv"
if errorlevel 1 (echo [FAIL] xvlog & goto :TB4)
call xelab work.tb_glb_output_bank -s sim_gob -timescale 1ns/1ps
if errorlevel 1 (echo [FAIL] xelab & goto :TB4)
call xsim sim_gob -R -log sim_gob.log
findstr /C:"ALL" sim_gob.log | findstr /C:"PASS" >nul && set /a PASS+=1

:TB4
echo.
echo ============================================================
echo  4/4  tb_metadata_ram
echo ============================================================
call xvlog --sv "%S0%\accel_pkg.sv" "%S0%\desc_pkg.sv" "%S0%\csr_pkg.sv" "%S3%\rtl\metadata_ram.sv" "%S3%\tb\tb_metadata_ram.sv"
if errorlevel 1 (echo [FAIL] xvlog & goto :SUMMARY)
call xelab work.tb_metadata_ram -s sim_meta -timescale 1ns/1ps
if errorlevel 1 (echo [FAIL] xelab & goto :SUMMARY)
call xsim sim_meta -R -log sim_meta.log
findstr /C:"ALL" sim_meta.log | findstr /C:"PASS" >nul && set /a PASS+=1

:SUMMARY
echo.
echo ============================================================
echo  STAGE 3 SUMMARY:  !PASS! / !TOTAL! testbenches PASSED
echo ============================================================
if !PASS!==!TOTAL! (echo [STAGE 3 PASS]) else (echo [STAGE 3 FAIL])
pause
exit /b 0
