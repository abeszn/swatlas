@echo off
REM Double-click to open the Swatlas dashboard.
REM Keep this window open while you use it - closing it stops the dashboard.

cd /d "%~dp0"
title Swatlas dashboard

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo   ERROR: virtual environment not found at .venv\Scripts\python.exe
  echo   Run this once to create it:
  echo       python -m venv .venv
  echo       .venv\Scripts\python.exe -m pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" scripts\run_dashboard.py %*

echo.
echo   Dashboard stopped.
timeout /t 5 >nul
