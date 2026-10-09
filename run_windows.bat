@echo off
cd /d %~dp0
where python >nul 2>nul
if errorlevel 1 (
  echo Python 3 is required.
  pause
  exit /b 1
)
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo FFmpeg is required and must be in PATH.
  echo Download FFmpeg, add its bin folder to PATH, then run this again.
  pause
  exit /b 1
)
python server.py
pause
