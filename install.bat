@echo off
REM Universal offline installer for Python project

REM --- CONFIG ---
set VENV_DIR=venv
set TESSERACT_INSTALLER_PATH=.\installers\tesseract-ocr-w64-setup-5.5.0.20241111.exe
set TESSERACT_INSTALL_PATH=C:\Program Files\Tesseract-OCR

REM 1. Check for Python
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    pause
    exit /b 1
)

REM 2. Check if Tesseract is installed
where tesseract >nul 2>nul
if errorlevel 1 (
    echo [INFO] Tesseract is not installed. Installing...
    REM Install Tesseract using the local installer
    if not exist "%TESSERACT_INSTALLER_PATH%" (
        echo [ERROR] Tesseract installer not found at %TESSERACT_INSTALLER_PATH%
        pause
        exit /b 1
    )
    echo [INFO] Installing Tesseract...
    start /wait %TESSERACT_INSTALLER_PATH% /SILENT
    REM Add Tesseract to the PATH environment variable
    echo [INFO] Adding Tesseract to PATH...
    setx PATH "%PATH%;%TESSERACT_INSTALL_PATH%"
) else (
    echo [INFO] Tesseract is already installed.
)

REM 3. Create venv if not present
if not exist "%VENV_DIR%\Scripts\activate" (
    echo [INFO] Creating virtual environment...
    python -m venv %VENV_DIR%
)

REM 4. Activate venv
call "%VENV_DIR%\Scripts\activate"

REM 5. Upgrade pip (offline, if possible)
python -m pip install --upgrade pip --no-index --find-links=./wheels

REM 6. Install all requirements from wheels
pip install --no-index --find-links=./wheels -r requirements.txt

REM 7. Finished!
echo.
echo [DONE] Offline install finished. To activate venv later, run:
echo     %VENV_DIR%\Scripts\activate
pause
