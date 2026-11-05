"""
Main FastAPI Backend - RAG Chatbot Server
"""
import asyncio
import os
import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional
import requests

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from document_parser import parse_all_documents
from vector_store import get_vector_store, VectorStore
from website_scraper import scrape_all

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1-performance:latest")
SARVAM_API_ROOT = os.getenv("SARVAM_API_ROOT", "https://api.sarvam.ai")
SARVAM_TTS_LANGUAGE = os.getenv("SARVAM_TTS_LANGUAGE", "ta-IN")
SARVAM_TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "shubh_enhi_ads")
SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v4-flash")

# Initialize FastAPI app
app = FastAPI(
    title="Department Kiosk RAG Chatbot",
    description="Autonomous department kiosk with RAG-based Q&A",
    version="1.0.0"
)

# Add CORS for kiosk deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/frontend",
    StaticFiles(directory=Path(__file__).resolve().parent.parent / "frontend"),
    name="frontend",
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    provider: Literal["ollama", "sarvam"] = "ollama"


class SpeakRequest(BaseModel):
    text: str


class ChatResponse(BaseModel):
    reply: str
    sources: Optional[List[Dict]] = None
    model: str = "sarvam-ai"
    confidence: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    vector_store_chunks: int


def extract_reply(response: Dict) -> str:
    """Extract assistant text from Sarvam's OpenAI-compatible response."""
    choices = response.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            text = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict)
            ).strip()
            if text:
                return text

    replies = response.get("replies", [])
    if replies:
        return replies[0]

    logger.error("Sarvam response contained no assistant text; keys=%s", list(response))
    return "I'm sorry, I couldn't generate a response."


def ollama_chat(messages: List[Dict[str, str]]) -> str:
    """Generate a local fallback response through the Ollama HTTP API."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": 4096},
        },
        timeout=120,
    )
    response.raise_for_status()
    content = response.json().get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama response contained no assistant text")
    return content.strip()


def sarvam_headers(api_key: str) -> Dict[str, str]:
    return {
        "api-subscription-key": api_key,
        "Accept": "application/json",
    }


def transcribe_audio(audio_bytes: bytes, filename: str, content_type: str, api_key: str) -> str:
    response = requests.post(
        f"{SARVAM_API_ROOT}/speech-to-text",
        headers=sarvam_headers(api_key),
        files={"file": (filename, audio_bytes, content_type or "audio/webm")},
        data={"model": "saaras:v4", "language_code": "en-IN"},
        timeout=120,
    )
    if not response.ok:
        logger.error("Speech-to-text failed (%s): %s", response.status_code, response.text[:500])
        raise requests.HTTPError(
            f"Speech-to-text service returned HTTP {response.status_code}: "
            f"{response.text[:200]}",
            response=response,
        )
    transcript = response.json().get("transcript", "")
    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError("Speech-to-text response contained no transcript")
    return transcript.strip()


def synthesize_speech(text: str, api_key: str) -> str:
    response = requests.post(
        f"{SARVAM_API_ROOT}/text-to-speech",
        headers={**sarvam_headers(api_key), "Content-Type": "application/json"},
        json={
            "text": text[:2500],
            "target_language_code": SARVAM_TTS_LANGUAGE,
            "speaker": SARVAM_TTS_SPEAKER,
            "model": SARVAM_TTS_MODEL,
            "pace": 1,
            "speech_sample_rate": 24000,
        },
        timeout=120,
    )
    response.raise_for_status()
    audios = response.json().get("audios", [])
    if not audios:
        raise ValueError("Text-to-speech response contained no audio")
    return "".join(audios)


def generate_reply(
    messages: List[Dict[str, str]],
    api_key: Optional[str],
    provider: Literal["ollama", "sarvam"] = "ollama",
) -> tuple[str, str]:
    """Generate a response using the explicitly selected provider."""
    if provider == "sarvam" and api_key:
        try:
            from sarvam_ai import SarvamClient

            response = SarvamClient(api_key=api_key).chat(
                model="sarvam-105b",
                messages=messages,
                max_tokens=500,
                temperature=0.7,
            )
            reply = extract_reply(response)
            if reply != "I'm sorry, I couldn't generate a response.":
                return reply, "sarvam-105b"
            logger.warning("Sarvam returned no usable text; trying Ollama fallback")
        except Exception as exc:
            logger.warning("Sarvam request failed: %s", exc)
    elif provider == "sarvam":
        logger.warning("SARVAM_API_KEY is not configured")

        return "Sarvam AI is selected, but SARVAM_API_KEY is not configured.", "unavailable"

    if provider == "ollama":
        try:
            return ollama_chat(messages), f"ollama/{OLLAMA_MODEL}"
        except requests.RequestException as exc:
            logger.error("Ollama request failed: %s", exc)
        except (ValueError, KeyError) as exc:
            logger.error("Invalid Ollama response: %s", exc)

    return (
        "I'm sorry, I couldn't generate a response. Please check the selected provider.",
        "unavailable",
    )


# Global state
vector_store: Optional[VectorStore] = None


@app.on_event("startup")
async def startup_event():
    """Initialize vector store on startup."""
    global vector_store
    logger.info("🚀 Starting RAG Chatbot backend...")
    vector_store = get_vector_store()
    logger.info(f"✅ Vector store ready: {vector_store.get_stats()['total_chunks']} chunks loaded")
    logger.info("🔧 Backend ready to accept requests")


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Department Kiosk RAG Chatbot",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "chat": "POST /chat - Send a message and get a response",
            "health": "GET /health - Service health check",
            "ingest_harddisk": "POST /ingest/harddisk - Ingest local files",
            "ingest_website": "POST /ingest/website - Scrape college website",
            "clear": "POST /clear - Clear vector store"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint."""
    if vector_store:
        stats = vector_store.get_stats()
        return HealthResponse(
            status="healthy",
            vector_store_chunks=stats.get("total_chunks", 0)
        )
    return HealthResponse(status="unhealthy", vector_store_chunks=0)


@app.post("/ingest/harddisk", tags=["ingestion"])
async def ingest_harddisk():
    """Ingest all documents from the hard disk data directory."""
    global vector_store
    
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    try:
        logger.info("📥 Starting hard disk ingestion...")
        documents = parse_all_documents()
        
        if not documents:
            return {
                "status": "no_documents",
                "message": "No supported documents found in data directory"
            }
        
        # Add to vector store
        chunks_added = vector_store.add_documents(documents)
        
        return {
            "status": "success",
            "documents_parsed": len(documents),
            "chunks_added": chunks_added,
            "total_chunks": vector_store.get_stats()["total_chunks"]
        }
        
    except Exception as e:
        logger.exception("Hard disk ingestion error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/website", tags=["ingestion"])
async def ingest_website():
    """Scrape college website and add to vector store."""
    global vector_store
    
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    try:
        logger.info("🕷️  Starting website ingestion...")
        pages = scrape_all()
        
        if not pages:
            return {
                "status": "no_pages",
                "message": "No pages found from college website"
            }
        
        # Add to vector store
        chunks_added = vector_store.add_documents(pages)
        
        return {
            "status": "success",
            "pages_scraped": len(pages),
            "chunks_added": chunks_added,
            "total_chunks": vector_store.get_stats()["total_chunks"]
        }
        
    except Exception as e:
        logger.exception("Website ingestion error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest):
    """
    Main chat endpoint - receives user query, retrieves relevant docs,
    generates response using Sarvam AI LLM.
    """
    global vector_store
    
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    if not request.message or request.message.strip() == "":
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        logger.info(f"💬 Processing query: {request.message[:80]}...")
        
        # 1. Query vector store for similar documents
        query_results = vector_store.query(request.message, top_k=5)
        
        if not query_results:
            # No relevant documents found
            system_prompt = """You are a department kiosk assistant. 
            
            I'm sorry, but I don't have information about that in my knowledge base.
            Please ask me about department events, placements, news, or other college-related topics.
            Be helpful but honest about your knowledge limits."""
            
            reply, model = generate_reply(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request.message},
                ],
                os.getenv("SARVAM_API_KEY"),
                request.provider,
            )
            
            return ChatResponse(
                reply=reply,
                sources=[],
                model=model,
            )
        
        # 2. Build context from retrieved documents
        context_parts = []
        sources = []
        
        for i, result in enumerate(query_results):
            content = result.get("content", "")[:300]  # Limit per chunk
            metadata = result.get("metadata", {})
            distance = result.get("distance", 1.0)
            
            context_parts.append(f"[Source {i+1} - {metadata.get('filename', 'unknown')}]\n{content}")
            
            sources.append({
                "source": metadata.get("source", "unknown"),
                "file": metadata.get("filename", "unknown"),
                "type": metadata.get("file_type", "unknown"),
                "distance": round(distance, 4)
            })
        
        context = "\n\n".join(context_parts)
        
        # 3. Build system prompt with context
        system_prompt = f"""You are a helpful department kiosk assistant for SRM Valliammai College. 
        
        Answer the user's question based ONLY on the provided context from college documents.
        
        Guidelines:
        - Provide accurate, concise answers based strictly on the context
        - If the answer is not in the context, say "I'm sorry, but I don't have that information in my college records."
        - Include source references when possible (e.g., "According to the events list...")
        - Be professional and friendly
        - If asked about something outside department scope, politely redirect to department topics
        
        Context from college documents:
        {context}
        
        User question: {request.message}"""
        
        # 4. Call Sarvam AI, with local Ollama fallback
        reply, model = generate_reply(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.message},
            ],
            os.getenv("SARVAM_API_KEY"),
            request.provider,
        )
        
        # 5. Return response with sources
        return ChatResponse(
            reply=reply,
            sources=sources,
            model=model,
            confidence=None  # Could add confidence scoring if needed
        )

    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        # Return graceful error response
        reply = f"I'm sorry, I encountered an error processing your request: {str(e)[:100]}"
        return ChatResponse(
            reply=reply,
            sources=[],
            model="error"
        )


@app.post("/speak", tags=["voice"])
async def speak(request: SpeakRequest):
    """Convert a text reply to base64-encoded WAV audio."""
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Sarvam API key is not configured")
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    try:
        return {
            "audio_base64": synthesize_speech(request.text, api_key),
            "content_type": "audio/wav",
        }
    except requests.RequestException as exc:
        logger.error("Speech synthesis failed: %s", exc)
        raise HTTPException(status_code=502, detail="Speech synthesis failed")
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/voice-chat", tags=["voice"])
async def voice_chat(
    audio: UploadFile = File(...),
    provider: Literal["ollama", "sarvam"] = Form("sarvam"),
):
    """Transcribe microphone audio, answer it, and synthesize the reply."""
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Sarvam API key is required for voice input/output")
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty")

    try:
        transcript = transcribe_audio(
            audio_bytes,
            audio.filename or "recording.webm",
            audio.content_type or "audio/webm",
            api_key,
        )
        query_results = vector_store.query(transcript, top_k=5) if vector_store else []
        context = "\n\n".join(
            f"[Source {i + 1}]\n{result.get('content', '')[:300]}"
            for i, result in enumerate(query_results)
        )
        system_prompt = f"""You are a helpful department kiosk assistant for SRM Valliammai College.
Answer only from this context. If the answer is absent, say you do not have that information.
Context:
{context}"""
        reply, model = generate_reply(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": transcript}],
            api_key,
            provider,
        )
        audio_base64 = synthesize_speech(reply, api_key)
        return {
            "transcript": transcript,
            "reply": reply,
            "model": model,
            "audio_base64": audio_base64,
            "content_type": "audio/wav",
        }
    except requests.RequestException as exc:
        logger.error("Voice request failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/clear", tags=["maintenance"])
async def clear_store():
    """Clear the vector store."""
    global vector_store
    
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    try:
        vector_store.clear()
        return {"status": "cleared", "message": "Vector store has been cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))