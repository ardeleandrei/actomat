@echo off
REM Universal offline installer for Python project

REM --- CONFIG ---
set VENV_DIR=venv

REM 1. Check for Python
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    pause
    exit /b 1
)

REM 2. Create venv if not present
if not exist "%VENV_DIR%\Scripts\activate" (
    echo [INFO] Creating virtual environment...
    python -m venv %VENV_DIR%
)

REM 3. Activate venv
call "%VENV_DIR%\Scripts\activate"

REM 4. Upgrade pip (offline, if possible)
python -m pip install --upgrade pip --no-index --find-links=./wheels

REM 5. Install all requirements from wheels
pip install --no-index --find-links=./wheels -r requirements.txt

REM 6. Finished!
echo.
echo [DONE] Offline install finished. To activate venv later, run:
echo     %VENV_DIR%\Scripts\activate
pause
