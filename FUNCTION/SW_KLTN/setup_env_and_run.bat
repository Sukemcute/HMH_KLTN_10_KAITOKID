@echo off
echo Activiting virtual environment...
call .venv\Scripts\activate.bat

echo Installing Jupyter and dill...
pip install jupyter dill

echo Installing ultralytics in editable mode...
cd Ultralytics-dev
pip install -e .
cd ..

echo Setup complete. Starting Jupyter Notebook...
jupyter notebook Quantized_model.ipynb
pause
