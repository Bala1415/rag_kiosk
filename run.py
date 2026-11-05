#!/usr/bin/env python3
"""
Main Entry Point - Initialize and run the RAG Kiosk chatbot.
"""
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Setup logging to stdout for console visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Environment validation
def validate_environment():
    """Validate environment variables and dependencies."""
    required_vars = ["SARVAM_API_KEY"]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set them in .env file or export them as environment variables")
        return False
    
    return True


def main():
    """Main entry point."""
    # Log working directory for troubleshooting
    logger.info(f"📁 Working directory: {os.getcwd()}")
    logger.info(f"🐍 Python version: {sys.version}")
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Show startup banner
    logger.info("=" * 60)
    logger.info("🚀 DEPARTMENT KIOSK RAG CHATBOT STARTING UP")
    logger.info("=" * 60)
    logger.info("📊 This kiosk will:")
    logger.info("   - Scrape college website for general info")
    logger.info("   - Ingest all documents from hard disk")
    logger.info("   - Build a vector database for RAG")
    logger.info("   - Respond to queries using Sarvam AI")
    logger.info("=" * 60)
    
    # Run the application
    try:
        # Import and run FastAPI app
        import uvicorn
        from backend.main import app
        
        # Log startup messages to console
        logger.info("🔧 Starting backend server...")
        logger.info("📖 API documentation available at: http://localhost:8000/docs")
        logger.info("📊 Health check at: http://localhost:8000/health")
        logger.info("🎯 Chat endpoint: POST http://localhost:8000/chat")
        
        uvicorn.run(
            app,
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", 8000)),
            reload=False,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        logger.error("\n💡 Troubleshooting tips:")
        logger.error("1. Check if .env file exists in workspace root")
        logger.error("2. Verify SARVAM_API_KEY is correctly set")
        logger.error("3. Ensure all dependencies are installed: pip install -r requirements.txt")
        logger.error("4. Check if port 8000 is available")
        sys.exit(1)


if __name__ == "__main__":
    main()
