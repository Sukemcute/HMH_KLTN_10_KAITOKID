@echo off
setlocal
REM Stage 100 — HW/SW golden cosim hub (Stage 8 TB + memh).
REM CWD must stay this folder so $readmemh finds vectors/<prim>/*.memh
REM HW root = stage_100\work -> stage_100 -> HW_KLTN_10

set "VIVADO_ROOT=C:\Xilinx\Vivado\2022.2"
if not exist "%VIVADO_ROOT%\settings64.bat" (
  echo Hay sua VIVADO_ROOT trong run_stage100_golden.cmd
  exit /b 1
)
call "%VIVADO_ROOT%\settings64.bat"

set "S100W=%~dp0"
if "%S100W:~-1%"=="\" set "S100W=%S100W:~0,-1%"
set "HW=%S100W%\..\.."
if "%HW:~-1%"=="\" set "HW=%HW:~0,-1%"
set "REPO=%HW%\.."
if "%REPO:~-1%"=="\" set "REPO=%REPO:~0,-1%"
set "TOOLS=%REPO%\FUNCTION\SW_KLTN\tools"

echo === 1) Generate vectors into stage_100\work\vectors\ ===
python "%TOOLS%\cosim_vector_gen.py" --prim all --seed 42
if errorlevel 1 (
  echo [ERROR] cosim_vector_gen.py failed
  exit /b 1
)

echo === 2) Compile and simulate (cwd=%S100W%) ===
cd /d "%S100W%"

set "FLIST=%S100W%\compile_rtl.f"
(
echo %HW%\stage_0_packages\rtl\accel_pkg.sv
echo %HW%\stage_0_packages\rtl\desc_pkg.sv
echo %HW%\stage_0_packages\rtl\csr_pkg.sv
echo %HW%\stage_1_compute\rtl\dsp_pair_int8.sv
echo %HW%\stage_1_compute\rtl\pe_unit.sv
echo %HW%\stage_1_compute\rtl\column_reduce.sv
echo %HW%\stage_1_compute\rtl\comparator_tree.sv
echo %HW%\stage_2_ppu\rtl\ppu.sv
echo %HW%\stage_3_memory\rtl\glb_input_bank_db.sv
echo %HW%\stage_3_memory\rtl\glb_weight_bank.sv
echo %HW%\stage_3_memory\rtl\glb_output_bank.sv
echo %HW%\stage_3_memory\rtl\metadata_ram.sv
echo %HW%\stage_4_addr_gen\rtl\addr_gen_input.sv
echo %HW%\stage_4_addr_gen\rtl\addr_gen_weight.sv
echo %HW%\stage_4_addr_gen\rtl\addr_gen_output.sv
echo %HW%\stage_5_data_movement\rtl\router_cluster_v2.sv
echo %HW%\stage_5_data_movement\rtl\window_gen.sv
echo %HW%\stage_5_data_movement\rtl\swizzle_engine.sv
echo %HW%\stage_6_control\rtl\tile_fsm.sv
echo %HW%\stage_6_control\rtl\shadow_reg_file.sv
echo %HW%\stage_6_control\rtl\compute_sequencer.sv
echo %HW%\stage_7_subcluster\rtl\pe_cluster_v4.sv
echo %HW%\stage_7_subcluster\rtl\subcluster_datapath.sv
echo %HW%\stage_8_primitive_verify\rtl\stage8_pkg.sv
) > "%FLIST%"

REM Xilinx tools on Windows are .bat — must use CALL or this .cmd stops after xvlog.
echo === xvlog (log: "%S100W%\xvlog.log") ===
call xvlog --sv -f "%FLIST%" "%HW%\stage_8_primitive_verify\tb\tb_stage8_subcluster_primitives.sv"
if errorlevel 1 (
  echo [ERROR] xvlog failed — open "%S100W%\xvlog.log"
  pause
  exit /b 1
)

echo === xelab ===
call xelab work.tb_stage8_subcluster_primitives -s sim_stage8 -timescale 1ns/1ps
if errorlevel 1 (
  echo [ERROR] xelab failed
  pause
  exit /b 1
)

echo === xsim (log: "%S100W%\xsim_sim.log") ===
call xsim sim_stage8 -R --testplusarg USE_GOLDEN_VECTORS -log "%S100W%\xsim_sim.log"

echo.
echo === Done. xsim ERRORLEVEL=%ERRORLEVEL% ===
echo Inspect: vectors\^<prim^>\   xvlog.log   xsim_sim.log   xsim.dir\
pause
exit /b %ERRORLEVEL%
