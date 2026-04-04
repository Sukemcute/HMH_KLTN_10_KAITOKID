@echo off
setlocal
REM Stage 2: Compile + run PPU testbench
set "VIVADO_ROOT=C:\Xilinx\Vivado\2022.2"
if not exist "%VIVADO_ROOT%\settings64.bat" (echo Set VIVADO_ROOT & exit /b 1)
call "%VIVADO_ROOT%\settings64.bat"

set "HW=%~dp0.."
if "%HW:~-1%"=="\" set "HW=%HW:~0,-1%"
set "S0=%HW%\stage_0_packages\rtl"
set "S2=%HW%\stage_2_ppu"

cd /d "%S2%"

echo ============================================================
echo  Stage 2: tb_ppu (PPU — bias, INT64 mul, half-up, ReLU, clamp)
echo ============================================================
call xvlog --sv "%S0%\accel_pkg.sv" "%S0%\desc_pkg.sv" "%S0%\csr_pkg.sv" "%S2%\rtl\ppu.sv" "%S2%\tb\tb_ppu.sv"
if errorlevel 1 (echo [FAIL] xvlog & pause & exit /b 1)

call xelab work.tb_ppu -s sim_ppu -timescale 1ns/1ps
if errorlevel 1 (echo [FAIL] xelab & pause & exit /b 1)

call xsim sim_ppu -R -log sim_ppu.log
echo.
findstr /C:"PASS" sim_ppu.log | findstr /C:"ALL" >nul
if errorlevel 1 (
  findstr /C:"PASS" sim_ppu.log | findstr /C:"8/8" >nul
  if errorlevel 1 (
    echo [STAGE 2 FAIL] — check sim_ppu.log
  ) else (
    echo [STAGE 2 PASS]
  )
) else (
  echo [STAGE 2 PASS]
)
pause
exit /b 0
