@echo off
setlocal
set "ROOT=%~dp0..\.."
set "SWTOOLS=%ROOT%\FUNCTION\SW_KLTN\tools"
set "GEN=%~dp0generated"
set "VIVADO_ROOT=C:\Xilinx\Vivado\2022.2"
if not exist "%VIVADO_ROOT%\settings64.bat" (
  echo Chinh VIVADO_ROOT trong file nay.
  exit /b 1
)
call "%VIVADO_ROOT%\settings64.bat"

set "HW=%~dp0.."
set "WORKDIR=%HW%\sim_work_stage11_sw"
if not exist "%WORKDIR%" mkdir "%WORKDIR%"

echo === 1) Export SW golden + copy .memh into sim cwd ===
python "%SWTOOLS%\hw_sw_cosim_stage11_rs3.py" --xsim-cwd "%WORKDIR%"
if errorlevel 1 exit /b 1

echo === 2) Compile and sim with USE_SW_GOLDEN ===
cd /d "%WORKDIR%"

xvlog --sv "%HW%\stage_0_packages\rtl\accel_pkg.sv" "%HW%\stage_0_packages\rtl\desc_pkg.sv" "%HW%\stage_0_packages\rtl\csr_pkg.sv" "%HW%\stage_1_compute\rtl\dsp_pair_int8.sv" "%HW%\stage_1_compute\rtl\pe_unit.sv" "%HW%\stage_1_compute\rtl\column_reduce.sv" "%HW%\stage_1_compute\rtl\comparator_tree.sv" "%HW%\stage_2_ppu\rtl\ppu.sv" "%HW%\stage_3_memory\rtl\glb_input_bank_db.sv" "%HW%\stage_3_memory\rtl\glb_weight_bank.sv" "%HW%\stage_3_memory\rtl\glb_output_bank.sv" "%HW%\stage_3_memory\rtl\metadata_ram.sv" "%HW%\stage_4_addr_gen\rtl\addr_gen_input.sv" "%HW%\stage_4_addr_gen\rtl\addr_gen_weight.sv" "%HW%\stage_4_addr_gen\rtl\addr_gen_output.sv" "%HW%\stage_5_data_movement\rtl\router_cluster_v2.sv" "%HW%\stage_5_data_movement\rtl\window_gen.sv" "%HW%\stage_5_data_movement\rtl\swizzle_engine.sv" "%HW%\stage_6_control\rtl\tile_fsm.sv" "%HW%\stage_6_control\rtl\shadow_reg_file.sv" "%HW%\stage_6_control\rtl\compute_sequencer.sv" "%HW%\stage_7_subcluster\rtl\pe_cluster_v4.sv" "%HW%\stage_7_subcluster\rtl\subcluster_datapath.sv" "%HW%\stage_11_block_verify\rtl\stage11_pkg.sv" "%HW%\stage_11_block_verify\tb\tb_stage11_block_verify.sv"
if errorlevel 1 exit /b 1

xelab work.tb_stage11_block_verify -s sim_stage11 -timescale 1ns/1ps
if errorlevel 1 exit /b 1

xsim sim_stage11 -R --testplusarg USE_SW_GOLDEN
exit /b %ERRORLEVEL%
