@echo off
cd /d "%~dp0"
if not exist .venv (
  echo Creating Python environment...
  py -3 -m venv .venv || python -m venv .venv
  .venv\Scripts\python -m pip install --quiet --upgrade pip
  .venv\Scripts\python -m pip install --quiet -r requirements.txt
)
start "" .venv\Scripts\pythonw.exe app.py
