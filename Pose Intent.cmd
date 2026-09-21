@echo off
setlocal
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3\envs\deeplabcut-gpu"
if errorlevel 1 (
    echo Install the DeepLabCut GPU environment described in DEEPLABCUT_INTEGRATION.md.
    pause
    exit /b 1
)
cd /d "%~dp0"
if "%~1"=="" (
    python -m pose_intent --help
    pause
) else (
    python -m pose_intent %*
    if errorlevel 1 exit /b 1
)
