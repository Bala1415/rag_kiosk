#!/usr/bin/env python3
"""
Unified Launcher for Department Kiosk RAG Chatbot
==================================================
Starts everything with ONE command:

1. Validates environment (.env, Sarvam API key)
2. Starts FastAPI backend server
3. Ingests data (hard disk + website) ONLY ONCE if vector DB is empty
4. Opens the frontend chat UI in your browser

Usage:
    python start.py

Press Ctrl+C to stop the server.
"""

import os
import sys
import time
import webbrowser
import subprocess
import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


def print_banner():
    print("=" * 60)
    print("🤖  DEPARTMENT KIOSK RAG CHATBOT  —  UNIFIED LAUNCHER")
    print("=" * 60)
    print()


def validate_env():
    """Load .env and report optional provider configuration."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv optional; vars may already be exported

    provider = os.getenv("DEFAULT_PROVIDER", "ollama").lower()
    api_key = os.getenv("SARVAM_API_KEY")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

    if provider == "sarvam" and not api_key:
        print("❌  ERROR: DEFAULT_PROVIDER=sarvam requires SARVAM_API_KEY.")
        return False

    if api_key:
        print("✅  Sarvam API key configured")
    else:
        print("ℹ️  Sarvam API key not configured; Ollama-only mode is available")
    print(f"✅  LLM endpoint: {ollama_url}")
    return True


def check_dependencies():
    """Verify core Python packages are installed."""
    required = ["uvicorn", "chromadb", "sentence_transformers", "fastapi"]
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"❌  Missing dependencies: {', '.join(missing)}")
        print("   Run:  pip install -r requirements.txt")
        print()
        return False

    print("✅  Core dependencies available")
    return True


def start_backend():
    """Launch FastAPI backend as a subprocess."""
    print("🔧  Starting FastAPI backend on http://localhost:8000 ...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app",
         "--host", "0.0.0.0", "--port", "8000"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def wait_for_backend(timeout=40):
    """Poll /health until the backend responds."""
    print("⏳  Waiting for backend to be ready...", end=" ", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get("http://localhost:8000/health", timeout=2)
            if r.status_code == 200:
                print("✅  Ready")
                return True
        except Exception:
            pass
        print(".", end=" ", flush=True)
        time.sleep(1)
    print("⚠️  Backend did not respond in time (continuing anyway)")
    return False


def is_vector_db_populated():
    """Check the collection count through the running backend."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        response.raise_for_status()
        return response.json().get("vector_store_chunks", 0) > 0
    except (requests.RequestException, ValueError):
        return False


def ingest_data():
    """Trigger one-time ingestion of hard disk + website via API."""
    print("\n📥  Running one-time data ingestion...")

    jobs = [
        ("/ingest/harddisk", "Local documents (DATA_DIR)"),
        ("/ingest/website",  "College website (srmvalliammai.ac.in)"),
    ]

    for endpoint, label in jobs:
        try:
            print(f"   📦  Ingesting {label} ...")
            r = requests.post(
                f"http://localhost:8000{endpoint}", timeout=600
            )
            if r.status_code == 200:
                print(f"   ✅  {label} done")
            else:
                print(f"   ⚠️  {label} returned HTTP {r.status_code}")
        except Exception as exc:
            print(f"   ❌  {label} failed: {exc}")

    print("✅  Ingestion complete")


def open_frontend():
    """Open the chat UI in the default browser."""
    frontend_path = PROJECT_ROOT / "frontend" / "index.html"
    if frontend_path.exists():
        print("\n🌐  Opening frontend in browser...")
        webbrowser.open("http://localhost:8000/frontend/index.html")
    else:
        print("\n⚠️  Frontend file not found:", frontend_path)


def main():
    print_banner()

    if not validate_env():
        sys.exit(1)

    if not check_dependencies():
        sys.exit(1)

    # 1. Start backend
    backend_proc = start_backend()

    try:
        # 2. Wait for it to be ready
        wait_for_backend()

        # 3. Ingest only if ChromaDB is empty (runs once)
        if is_vector_db_populated():
            print("\n✅  Vector database already populated — skipping ingestion.")
        else:
            ingest_data()

        # 4. Open chat UI
        open_frontend()

        # 5. Keep running until user interrupts
        print("\n" + "=" * 60)
        print("🎉  KIOSK IS RUNNING!")
        print(f"   Backend API : http://localhost:8000")
        print(f"   API Docs    : http://localhost:8000/docs")
        print(f"   Frontend    : frontend/index.html")
        print("=" * 60)
        print("\nPress Ctrl+C to stop.\n")

        backend_proc.wait()

    except KeyboardInterrupt:
        print("\n🛑  Shutting down...")
        backend_proc.terminate()
        try:
            backend_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_proc.kill()
        print("✅  Stopped successfully.")


if __name__ == "__main__":
    main()