@echo off
echo ============================================
echo    Sign Language Detection ML Project
echo ============================================
echo.

cd /d "%~dp0"

echo Checking Python installation...
python --version 2>nul
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.10 or higher.
    pause
    exit /b 1
)

echo.
echo Starting application...
echo.

python main.py

pause
