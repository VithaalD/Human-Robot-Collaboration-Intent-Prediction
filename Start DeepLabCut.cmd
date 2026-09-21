@echo off
setlocal
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3\envs\deeplabcut-gpu"
if errorlevel 1 (
    echo Could not activate the DeepLabCut GPU environment.
    pause
    exit /b 1
)
cd /d "%~dp0"
python -m deeplabcut
if errorlevel 1 pause
