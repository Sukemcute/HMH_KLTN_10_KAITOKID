@echo off
REM Verify golden data (no Vivado). Run from PHASE_4 after export.
set PHASE4=%~dp0
cd /d "%PHASE4%"

echo.
echo === Step 4A: System readiness (golden files + sizes) ===
python 03_rtl_cosim\verify_system_readiness.py --golden-dir "%PHASE4%02_golden_data" --strict-descriptors
if %errorlevel% neq 0 exit /b 1

echo.
echo === Complex blocks: Upsample + QConcat (pure numpy) ===
python 03_rtl_cosim\verify_complex_blocks.py --upsample --qconcat
if %errorlevel% neq 0 exit /b 1

echo.
echo === Complex blocks: SCDown, QC2f, SPPF, QPSA (needs PyTorch + dill) ===
python 03_rtl_cosim\verify_complex_blocks.py --torch-blocks
REM non-zero exit ok if dill missing — user can pip install dill

echo.
echo === Full model P3/P4/P5 vs golden_*.hex ===
python 03_rtl_cosim\verify_full_model_outputs.py --golden-dir "%PHASE4%02_golden_data"
if %errorlevel% neq 0 (
  echo If import failed: pip install dill
  exit /b 1
)

echo.
echo === Conv layers L0/L1/L3/L17 (optional, slow with --rows 0) ===
python 03_rtl_cosim\verify_conv_layer.py --layer 0 --rows 1
python 03_rtl_cosim\verify_conv_layer.py --layer 1 --rows 1
python 03_rtl_cosim\verify_conv_layer.py --layer 3 --rows 1
python 03_rtl_cosim\verify_conv_layer.py --layer 17 --rows 1

echo.
echo Done. Next: Vivado run_ppu_golden.tcl and tb_golden_check when accel_top is ready.
pause
