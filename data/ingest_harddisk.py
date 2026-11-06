"""
Document Parser Module - Extracts text from various file formats.
Supports PDF, DOCX, TXT, MD, images with OCR.
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Add parent path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_parser import DocumentParser

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = os.getenv("DATA_DIR", str(PROJECT_ROOT / "newsletter"))
SUPPORTED_EXTENSIONS = {
    '.pdf', '.docx', '.txt', '.md', '.png', '.jpg', '.jpeg',
    '.xlsx', '.xlsm', '.pptx',
}
SUPPORTED_EXTENSIONS_LOWER = {f'.{ext.lower()}' for ext in SUPPORTED_EXTENSIONS}

def collect_files(directory: str) -> List[str]:
    """Recursively find all supported files in directory."""
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in SUPPORTED_EXTENSIONS_LOWER:
                full_path = os.path.join(root, filename)
                # Normalize Windows path for cross-platform compatibility
                relative_path = os.path.relpath(full_path, directory)
                files.append((full_path, relative_path))
    return files


def parse_all_documents(data_dir: str = None) -> List[Dict[str, Any]]:
    """Parse all documents from the data directory."""
    if data_dir is None:
        data_dir = DATA_DIR
    
    if not os.path.exists(data_dir):
        print(f"⚠️  Data directory not found: {data_dir}")
        return []
    
    print(f"🔍 Scanning {data_dir}...")
    files = collect_files(data_dir)
    print(f"   Found {len(files)} supported files")
    
    all_documents = []
    success_count = 0
    skipped_count = 0
    failed_count = 0
    
    for file_path, relative_path in files:
        try:
            doc_type = Path(file_path).suffix.lower()
            
            # Use the shared parser
            content, metadata = DocumentParser.parse(file_path, file_type=doc_type)
            
            if content and content.strip():
                document = {
                    "content": content.strip(),
                    "metadata": {
                        "source": "harddisk",
                        "file_path": relative_path,
                        "file_type": doc_type,
                        "ingested_at": "",
                        "extracted_text": True
                    }
                }
                all_documents.append(document)
                success_count += 1
            else:
                print(f"   ⚠️  No extractable text from: {relative_path}")
                skipped_count += 1
                
        except Exception as e:
            print(f"   ❌ Failed to parse {relative_path}: {e}")
            failed_count += 1
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Successfully parsed: {success_count}")
    print(f"   ⚠️  Skipped (no text): {skipped_count}")
    print(f"   ❌ Failed: {failed_count}")
    
    return all_documents
