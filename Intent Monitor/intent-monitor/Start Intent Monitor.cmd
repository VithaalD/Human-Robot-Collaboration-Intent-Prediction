@echo off
cd /d "%~dp0"
"%USERPROFILE%\anaconda3\envs\DEEPLABCUT\python.exe" -B "%~dp0intent_monitor.py"
if errorlevel 1 pause
