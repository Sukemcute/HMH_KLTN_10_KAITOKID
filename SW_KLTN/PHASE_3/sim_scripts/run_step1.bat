@echo off
REM ═══════════════════════════════════════════════════════════════
REM  Run Step 1: Compute Leaf Tests with Vivado Simulator
REM  Usage: run_step1.bat
REM ═══════════════════════════════════════════════════════════════

set PROJ_ROOT=%~dp0..
set PKG=%PROJ_ROOT%\packages
set RTL1=%PROJ_ROOT%\01_compute_leaf\rtl
set TB1=%PROJ_ROOT%\01_compute_leaf\tb

echo.
echo ╔══════════════════════════════════════════════════╗
echo ║  PHASE 3 - Step 1: Compute Leaf Tests            ║
echo ╚══════════════════════════════════════════════════╝
echo.

REM --- TEST 1: dsp_pair_int8 ---
echo === Running: tb_dsp_pair_int8 ===
xvlog -sv %PKG%\accel_pkg.sv %PKG%\desc_pkg.sv %RTL1%\dsp_pair_int8.sv %TB1%\tb_dsp_pair_int8.sv
xelab tb_dsp_pair_int8 -s sim_dsp
xsim sim_dsp -runall
echo.

REM --- TEST 2: pe_unit ---
echo === Running: tb_pe_unit ===
xvlog -sv %PKG%\accel_pkg.sv %PKG%\desc_pkg.sv %RTL1%\dsp_pair_int8.sv %RTL1%\pe_unit.sv %TB1%\tb_pe_unit.sv
xelab tb_pe_unit -s sim_pe
xsim sim_pe -runall
echo.

REM --- TEST 3: column_reduce ---
echo === Running: tb_column_reduce ===
xvlog -sv %PKG%\accel_pkg.sv %PKG%\desc_pkg.sv %RTL1%\column_reduce.sv %TB1%\tb_column_reduce.sv
xelab tb_column_reduce -s sim_colred
xsim sim_colred -runall
echo.

REM --- TEST 4: comparator_tree ---
echo === Running: tb_comparator_tree ===
xvlog -sv %PKG%\accel_pkg.sv %RTL1%\comparator_tree.sv %TB1%\tb_comparator_tree.sv
xelab tb_comparator_tree -s sim_cmptree
xsim sim_cmptree -runall
echo.

REM --- TEST 5: silu_lut ---
echo === Running: tb_silu_lut ===
xvlog -sv %RTL1%\silu_lut.sv %TB1%\tb_silu_lut.sv
xelab tb_silu_lut -s sim_silu
xsim sim_silu -runall
echo.

echo ════════════════════════════════════════
echo   Step 1 Complete. Check results above.
echo ════════════════════════════════════════
pause
