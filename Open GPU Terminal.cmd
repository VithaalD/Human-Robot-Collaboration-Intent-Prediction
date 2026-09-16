@echo off
setlocal
set "intentActivate=%USERPROFILE%\.venvs\intent-gpu\Scripts\activate.bat"
if not exist "%intentActivate%" (
    echo Run setup_gpu.ps1 first.
    pause
    exit /b 1
)
call "%intentActivate%"
cd /d "%~dp0"
echo GPU training environment activated. Run python tools\verify_gpu.py to check it.
cmd /k
