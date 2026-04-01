@echo off
REM ═══════════════════════════════════════════════════════════════
REM  PHASE 4: Software-Hardware Co-Simulation — Master Runner
REM ═══════════════════════════════════════════════════════════════
REM  Prerequisites:
REM    1. Python 3.10+ with: pip install torch torchvision opencv-python numpy dill
REM    2. pip install -e Ultralytics-dev  (quantized YOLOv10 fork)
REM    3. Vivado 2022.2+ for RTL simulation
REM    4. img1.jpg in SW_KLTN root
REM ═══════════════════════════════════════════════════════════════

set ROOT=%~dp0..
set PHASE4=%~dp0
set DATA=%PHASE4%02_golden_data
set LBL=%DATA%\layer_by_layer

echo.
echo ================================================================
echo  PHASE 4 — Step 1: Export Golden Data
echo ================================================================
cd /d "%PHASE4%01_export"

echo [1a] Exporting main golden data (input, weights, P3/P4/P5)...
python export_golden_data.py --image "%ROOT%\img1.jpg" --output "%DATA%"
if %errorlevel% neq 0 (
    echo ERROR: export_golden_data.py failed!
    pause
    exit /b 1
)

echo.
echo [1b] Exporting layer-by-layer golden data...
python export_layer_by_layer.py --image "%ROOT%\img1.jpg" --output "%LBL%"
if %errorlevel% neq 0 (
    echo ERROR: export_layer_by_layer.py failed!
    pause
    exit /b 1
)

echo.
echo [1c] Generating descriptors...
python generate_descriptors.py --params "%DATA%\quant_params.json" --output "%DATA%"
if %errorlevel% neq 0 (
    echo ERROR: generate_descriptors.py failed!
    pause
    exit /b 1
)

echo.
echo ================================================================
echo  PHASE 4 — Step 1 Complete
echo ================================================================
echo.
echo  Golden data exported to: %DATA%
echo  Layer-by-layer data in:  %LBL%
echo.
echo  Next steps:
echo    1. Python verify: run PHASE_4\run_phase4_verify.bat  (complex blocks + P3/P4/P5)
echo    2. Vivado: source PHASE_4/03_rtl_cosim/run_ppu_golden.tcl then run_ppu_test 0..17
echo    3. When accel_top ready: tb_golden_check (see COMPLEX_AND_FULL_MODEL.md)
echo    4. cpu_postprocess.py for boxes from hex outputs
echo.
echo  Example postprocessing command:
echo    cd PHASE_4\04_postprocess
echo    python cpu_postprocess.py --hex_dir ..\02_golden_data --image ..\..\img1.jpg --output output_rtl.jpg
echo.
pause
