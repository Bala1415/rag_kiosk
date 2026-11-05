#!/bin/bash
# ============================================================
#  Department Kiosk RAG Chatbot - Linux Launcher
#  Run this to start everything (works on Ubuntu, macOS, etc.)
# ============================================================

echo "=================================================="
echo "  DEPARTMENT KIOSK RAG CHATBOT"
echo "=================================================="
echo

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "✅ Virtual environment found"
    source venv/bin/activate
else
    echo "⚠️  No virtual environment found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Check Tesseract
tesseract --version >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "⚠️  WARNING: Tesseract OCR not found in PATH"
    echo "    Install it and add to PATH for image text extraction"
    echo "    Ubuntu: sudo apt-get install tesseract-ocr"
    echo "    Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
    echo
fi

# Start the kiosk
echo "🚀 Starting kiosk..."
echo "   Backend: http://localhost:8000"
echo "   Frontend: http://localhost:8000/frontend/index.html"
echo

python start.py
