@echo off
REM ============================================================
REM  Department Kiosk RAG Chatbot - Windows Launcher
REM  Just double-click this file to start everything!
REM ============================================================

echo.
echo ============================================================
echo  DEPARTMENT KIOSK RAG CHATBOT
echo ============================================================
echo.

REM Activate the Python 3.11 virtual environment if it exists
if exist "venv311\Scripts\activate.bat" (
    call venv311\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Check if dependencies are installed
python -c "import fastapi" 2>nul
if errorlevel 1 (
    echo ⚠️  Installing dependencies...
    pip install -r requirements.txt
    echo ✅ Dependencies installed
)

REM Check if Tesseract is available
tesseract --version >nul 2>&1
if errorlevel 1 (
    echo ⚠️  WARNING: Tesseract OCR not found in PATH
    echo    Download from: https://github.com/UB-Mannheim/tesseract/wiki
    echo    Install it and add to PATH for image text extraction
    echo.
)

REM Start the kiosk
echo 🚀 Starting kiosk...
echo.

python start.py

pause