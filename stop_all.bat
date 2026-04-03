@echo off
setlocal enabledelayedexpansion

echo Stopping all services...

REM Kill ngrok processes first
taskkill /F /IM ngrok.exe 2>nul

REM Find and kill process on port 8501 (Streamlit) safely
for /f "tokens=5" %%a in ('netstat -ano ^| find ":8501"') do (
    set PID=%%a
    if defined PID (
        taskkill /PID !PID! /F 2>nul
        echo Killed Streamlit process (PID: !PID!)
    )
)

echo.
echo ========================================
echo All services stopped!
echo ========================================
pause
