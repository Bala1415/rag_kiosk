# Department Kiosk RAG Chatbot

A lightweight, autonomous department kiosk for your college information. Powered by RAG (Retrieval-Augmented Generation) with Sarvam AI LLM integration.

## 🎯 Purpose

This kiosk provides instant, accurate answers about your department using:
- **College website content** (events, placements, news)
- **Local documents** (PDFs, Word docs, images)
- **AI-powered responses** via Sarvam AI

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11 (recommended; the pinned ChromaDB/Pillow versions do not support Python 3.14)
- Windows 10+ or Ubuntu 20.04+
- **Tesseract OCR** (for image/PDF text extraction):
  - **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki and install
  - **Ubuntu**: `sudo apt-get install tesseract-ocr`

### Installation

```bash
# 1. Clone / navigate to project
cd rag_kiosk

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Jetson: source .venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements-desktop.txt

# 4. Verify installation
pip list
```

### Configuration

Copy `.env.example` to `.env` and edit machine-specific values:

```env
DEFAULT_PROVIDER=ollama
OLLAMA_BASE_URL=http://<AGX_ORIN_IP>:11434
OLLAMA_MODEL=gpt-oss:20b
DATA_DIR=./newsletter
COLLEGE_URL=https://www.srmvalliammai.ac.in/
DEPARTMENT_URL=https://www.srmvalliammai.ac.in/department/department-of-artificial-intelligence-and-data-science
WEBSITE_MAX_PAGES=150
EMBEDDING_DEVICE=auto
STT_PROVIDER=whisper
WHISPER_MODEL=small
WHISPER_LANGUAGE=
WHISPER_TASK=transcribe
WHISPER_DEVICE=auto
WHISPER_COMPUTE_TYPE=auto
SARVAM_STT_LANGUAGE=en-IN
TTS_PROVIDER=sarvam
SARVAM_TTS_LANGUAGE=ta-IN
```

Keep `.env` private; it is ignored by Git. Never commit API keys.

Ollama uses available GPU acceleration by default. Do not set `num_gpu` to
`0` in the request, because that explicitly forces CPU-only inference. For an
NVIDIA GPU, install the current NVIDIA driver and run `ollama serve`; verify
GPU usage with `ollama ps` or `nvidia-smi` while a request is running.

On Windows, install `requirements-desktop.txt`. On Jetson, first install the
PyTorch wheel matching the installed JetPack release, then run:

```bash
pip install -r requirements-jetson.txt
```

For server-side voice input, install the optional Whisper dependency and FFmpeg:

```bash
sudo apt-get install -y ffmpeg
pip install -r requirements-voice.txt
```

The browser records audio and the backend transcribes it with
`faster-whisper`. `WHISPER_MODEL=small` is a quality-oriented default; use
`tiny` or `base` on an 8 GB Nano if memory or latency is a problem. The first
voice request downloads the selected model. Sarvam TTS is used for reliable
Indian-language audio output, so `SARVAM_API_KEY` is required for `/speak` and
`/voice-chat`.

Use `WHISPER_TASK=transcribe` to preserve the spoken language. Use
`WHISPER_TASK=translate` only when you specifically want Whisper to translate
speech into English. Set `SARVAM_STT_LANGUAGE` when Sarvam is used as fallback.

The RAG embedding pipeline also uses CUDA automatically when PyTorch detects
an NVIDIA GPU. On startup it prints the selected device and GPU name. If CUDA
is unavailable, it safely falls back to CPU. Set `EMBEDDING_DEVICE=cuda` to
require CUDA, or `EMBEDDING_DEVICE=cpu` to disable GPU embeddings.

Website ingestion starts with the college homepage and the AI & Data Science
department page, follows same-domain links from every visited HTML page, and
extracts linked PDFs (including department reports, magazines, and
newsletters). `WEBSITE_MAX_PAGES` limits the crawl to avoid an unbounded
site-wide crawl.

### One-Time Initial Setup

Run these commands **once** to build the vector database:

```bash
# Start the backend (will also ingest data on first run)
python run.py
```

**OR run ingestion separately:**

```bash
# Initialize backend
python run.py

# Ingest documents configured by DATA_DIR
POST http://localhost:8000/ingest/harddisk

# Scrape college website
POST http://localhost:8000/ingest/website
```

> 💡 **Tip**: The first run may take 2-5 minutes depending on your document count. Subsequent starts will be fast.

---

## ▶️ Running for Regular Use

After initial setup, daily usage is simple:

### Start the Kiosk

```bash
python run.py
```

Sarvam AI is used first. If its API key is missing or the request fails, the
backend automatically falls back to the local Ollama model configured by
`OLLAMA_MODEL`. Ensure Ollama is running before starting the kiosk:

```bash
ollama serve
```

The backend will:
- Load the existing vector database (no re-ingestion needed)
- Start the FastAPI server on `http://localhost:8000`

Hard-disk ingestion recursively scans every subfolder under `DATA_DIR`.
It indexes text from PDF, Word, Excel, PowerPoint, TXT, Markdown, and image
files (using OCR when available). Videos, ZIP files, installers, and other
binary files are visited but skipped because they do not contain directly
extractable document text.

### Access the Chat

- **Open in browser**: `http://localhost:8000/frontend/index.html`
- **Kiosk mode (Windows)**: `chrome.exe --kiosk http://localhost:8000/frontend/index.html`
- **Kiosk mode (Linux/macOS)**: `google-chrome --kiosk http://localhost:8000/frontend/index.html`

### Regular Chat Workflow

1. Open the frontend URL in the kiosk browser
2. Type your question (e.g., "When is the placement drive?", "Show me the latest newsletter")
3. The bot will:
   - Search the vector database for relevant documents
   - Generate an answer using Sarvam AI
   - Return the response with citations

### Adding New Documents

If you add new PDFs, docs, or images to `F:\Dept activities`:

```bash
# Re-ingest only new/changed files
POST http://localhost:8000/ingest/harddisk
```

> ⚡ **Fast**: The vector store is incremental — only new files are processed.

### Updating Website Content

If the college website changes:

```bash
POST http://localhost:8000/ingest/website
```

---

## 📁 Folder Structure

```
rag_kiosk/
├── .env                    # Configuration (API key, paths)
├── requirements.txt       # Python dependencies
├── run.py                 # Main entry point
├── README.md              # This file
├── vector_store.py        # ChromaDB and vector operations
├── document_parser.py     # Document file parsing (PDF, DOCX, images, text)
├── website_scraper.py     # College website content extractor
├── sarvam_ai.py          # Sarvam AI API client
├── backend/
│   ├── __init__.py
│   └── main.py           # FastAPI backend server
├── frontend/
│   └── index.html        # Chat UI with robot avatar
└── data/
    └── ingest_harddisk.py # Hard disk ingestion script
└── chroma_db/            # Persistent vector DB (auto-created, skip on subsequent runs)
```

---

## ⚡ Key Features for MVP

✅ **Fast Loading**: Zero-latency local embeddings, quick response generation  
✅ **Lightweight**: ~200MB total footprint, minimal memory usage  
✅ **Cross-Platform**: Windows, Ubuntu, and macOS support  
✅ **Touch-Friendly**: Optimized for kiosk touchscreens  
✅ **Offline Capable**: Local embeddings work without internet (except Sarvam AI)  
✅ **Scalable**: Easy to add new document sources  

---

## 🛠 Development Workflow

### 1. Adding New Documents

Simply add new files to `DATA_DIR` — the system automatically re-ingests them on next run.

### 2. Adding New Sources

Update `website_scraper.py` with additional URLs or parsing logic.

### 3. Extending the Chatbot

Modify `backend/main.py` to customize response prompts or add new features.

---

## 📊 Performance Benchmarks (MVP)

| Component | Response Time |
|-----------|---------------|
| Vector Search | 10-50 ms |
| Document Parse | 100-500 ms per file |
| Sarvam AI Response | 1-3 seconds |
| UI Rendering | ~100 ms per message |

---

## 🔄 Troubleshooting

### Common Issues

#### "Missing Environment Variables"

```bash
# Check .env file exists
type .env

# Verify SARVAM_API_KEY is set
echo $SARVAM_API_KEY
```

#### "Python Package Not Found"

```bash
pip install -r requirements.txt
```

#### "Web Scraping Errors"

Check if the college website is accessible and if their `robots.txt` allows scraping.

#### "Tesseract Not Found"

Install Tesseract OCR (see Prerequisites section).

### Debug Endpoints

- **Health Check**: `GET /health`
- **Ingestion Status**: Use `/ingest/harddisk` or `/ingest/website`
- **API Docs**: `GET /docs` (Swagger UI)

---

## 🛡️ Security Considerations

- **API Key**: Never commit your Sarvam API key to version control
- **Environment Variables**: Use `.env` file and add it to `.gitignore`
- **Firewall**: Consider firewall rules for kiosk deployment

---

## 📝 Future Enhancements

1. **Advanced OCR**: Handle more document types
2. **Multi-modal**: Add image analysis
3. **Smart Routing**: Route queries to different knowledge bases
4. **Analytics**: Track usage patterns
5. **Backup**: Vector database backup and restore

---

## 📞 Support

For issues, visit the repository and create an issue, or refer to documentation in the code.

---
*Built with ❤️ for your department*
*Developed using FastAPI, ChromaDB, and modern web technologies.*

## License

This project is licensed for your department use.