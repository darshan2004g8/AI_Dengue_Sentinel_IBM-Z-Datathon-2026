@echo off
REM ========================================================================
REM Dengue Sentinel: One-Command Runner (Windows Batch)
REM Event Code: SSF2026 (IBM Z Datathon 2026)
REM ========================================================================
echo Starting Dengue Sentinel Master Runner...
python run_all.py %*
pause
