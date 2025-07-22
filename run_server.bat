@echo off
REM Activate virtual environment
call "%~dp0venv\Scripts\activate.bat"
REM Start the FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000
pause
