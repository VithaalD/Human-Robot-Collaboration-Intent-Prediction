@echo off
setlocal
set "intentPython=%INTENT_PYTHON%"
if defined intentPython goto python_selected
set "intentPython=%USERPROFILE%\.venvs\intent-gpu\Scripts\python.exe"
if not exist "%intentPython%" set "intentPython=%USERPROFILE%\anaconda3\envs\DEEPLABCUT\python.exe"
:python_selected
if not exist "%intentPython%" (
    echo Python was not found. Run setup_gpu.ps1 or set INTENT_PYTHON to your Python executable.
    pause
    exit /b 1
)
cd /d "%~dp0"
"%intentPython%" -B "%~dp0intent_monitor.py" %*
if errorlevel 1 pause
