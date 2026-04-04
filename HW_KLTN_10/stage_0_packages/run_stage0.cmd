@echo off
setlocal
REM Stage 0: Compile-only — verify accel_pkg, desc_pkg, csr_pkg parse clean.
set "VIVADO_ROOT=C:\Xilinx\Vivado\2022.2"
if not exist "%VIVADO_ROOT%\settings64.bat" (echo Set VIVADO_ROOT & exit /b 1)
call "%VIVADO_ROOT%\settings64.bat"

set "S0=%~dp0"
if "%S0:~-1%"=="\" set "S0=%S0:~0,-1%"

echo === Stage 0: Compile packages ===
call xvlog --sv "%S0%\rtl\accel_pkg.sv" "%S0%\rtl\desc_pkg.sv" "%S0%\rtl\csr_pkg.sv"
if errorlevel 1 (
  echo [FAIL] Stage 0 — package compile errors
  pause & exit /b 1
)
echo [PASS] Stage 0 — all 3 packages compiled clean
pause
exit /b 0
