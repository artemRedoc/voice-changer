@echo off
cd /d "%~dp0"
if not exist .venv (
  py -3 -m venv .venv || python -m venv .venv
)
.venv\Scripts\python -m pip install --quiet --upgrade pip
.venv\Scripts\python -m pip install --quiet -r requirements.txt pyinstaller
.venv\Scripts\pyinstaller --noconfirm --clean --onefile --windowed --name VoiceChanger app.py
echo.
echo Built: dist\VoiceChanger.exe
pause
