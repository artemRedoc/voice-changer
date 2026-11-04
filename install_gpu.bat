@echo off
cd /d "%~dp0"
echo Installing the NVIDIA (CUDA) build of PyTorch into .venv ...
if not exist .venv (
  py -3 -m venv .venv || python -m venv .venv
  .venv\Scripts\python -m pip install --quiet --upgrade pip
  .venv\Scripts\python -m pip install --quiet -r requirements.txt
)
.venv\Scripts\python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
pause
