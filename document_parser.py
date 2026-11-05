"""
Document Parser Module - Extracts text from various file formats.
Supports PDF, DOCX, TXT, MD, images with OCR.
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# Optional OCR - gracefully handle if tesseract not installed
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import pypdf
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

try:
    from pptx import Presentation
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False


class DocumentParser:
    """Handles text extraction from individual files."""

    @staticmethod
    def parse(file_path: str, file_type: str = None) -> Tuple[str, Dict[str, Any]]:
        """
        Parse a single file and return (content, metadata).
        
        Args:
            file_path: Absolute path to the file
            file_type: File extension (e.g. '.pdf')
        
        Returns:
            Tuple of (extracted_text, metadata_dict)
        """
        if file_type is None:
            file_type = Path(file_path).suffix.lower()
        
        metadata = {
            "filename": Path(file_path).name,
            "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            "ocr_used": False
        }
        
        content = ""
        
        if file_type == ".pdf":
            content = DocumentParser._parse_pdf(file_path, metadata)
        elif file_type == ".docx":
            content = DocumentParser._parse_docx(file_path, metadata)
        elif file_type in (".xlsx", ".xlsm"):
            content = DocumentParser._parse_spreadsheet(file_path, metadata)
        elif file_type == ".pptx":
            content = DocumentParser._parse_presentation(file_path, metadata)
        elif file_type in (".txt", ".md"):
            content = DocumentParser._parse_text(file_path, metadata)
        elif file_type in (".png", ".jpg", ".jpeg"):
            content = DocumentParser._parse_image(file_path, metadata)
        else:
            content = ""
        
        return content, metadata

    @staticmethod
    def _parse_pdf(file_path: str, metadata: Dict[str, Any]) -> str:
        """Extract text from PDF file."""
        if not PDF_AVAILABLE:
            print(f"   WARNING: pypdf not installed, skipping PDF: {file_path}")
            return ""
        
        try:
            text_parts = []
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text() or ""
                        if page_text.strip():
                            text_parts.append(page_text)
                    except Exception as e:
                        print(f"      WARNING: Error on page {page_num + 1}: {e}")
            
            content = "\n".join(text_parts)
            
            # If no text extracted (scanned PDF), try OCR if available
            if not content.strip() and OCR_AVAILABLE:
                metadata["ocr_used"] = True
                print(f"      OCR: PDF has no text layer, attempting OCR: {Path(file_path).name}")
                content = DocumentParser._ocr_pdf_images(file_path)
            
            return content
        except Exception as e:
            print(f"   ERROR: PDF parse error: {e}")
            return ""

    @staticmethod
    def _ocr_pdf_images(file_path: str) -> str:
        """OCR for scanned PDF pages (basic approach)."""
        try:
            import fitz  # PyMuPDF for PDF to image
        except ImportError:
            print("      WARNING: PyMuPDF not installed for OCR on PDFs")
            return ""
        
        try:
            doc = fitz.open(file_path)
            text_parts = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")
                
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(img_data))
                text = pytesseract.image_to_string(img)
                if text.strip():
                    text_parts.append(text)
            
            return "\n".join(text_parts)
        except Exception as e:
            print(f"      ERROR: OCR error: {e}")
            return ""

    @staticmethod
    def _parse_docx(file_path: str, metadata: Dict[str, Any]) -> str:
        """Extract text from Word document."""
        if not DOCX_AVAILABLE:
            print(f"   WARNING: python-docx not installed, skipping DOCX: {file_path}")
            return ""
        
        try:
            doc = docx.Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            content = "\n".join(paragraphs)
            
            # Also extract table content
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells)
                    if row_text.strip():
                        content += "\n" + row_text
            
            return content
        except Exception as e:
            print(f"   ERROR: DOCX parse error: {e}")
            return ""

    @staticmethod
    def _parse_text(file_path: str, metadata: Dict[str, Any]) -> str:
        """Extract text from plain text or markdown file."""
        try:
            encodings = ["utf-8", "latin-1", "cp1252"]
            for enc in encodings:
                try:
                    with open(file_path, "r", encoding=enc) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            print(f"   WARNING: Could not decode text file: {file_path}")
            return ""
        except Exception as e:
            print(f"   ERROR: Text parse error: {e}")
            return ""

    @staticmethod
    def _parse_spreadsheet(file_path: str, metadata: Dict[str, Any]) -> str:
        """Extract cell values from Excel workbooks."""
        if not OPENPYXL_AVAILABLE:
            print(f"   WARNING: openpyxl not installed, skipping spreadsheet: {file_path}")
            return ""

        try:
            workbook = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
            rows = []
            for sheet in workbook.worksheets:
                rows.append(f"[Sheet: {sheet.title}]")
                for row in sheet.iter_rows(values_only=True):
                    values = [str(value).strip() for value in row if value is not None]
                    if values:
                        rows.append(" | ".join(values))
            workbook.close()
            return "\n".join(rows)
        except Exception as e:
            print(f"   ERROR: Spreadsheet parse error: {e}")
            return ""

    @staticmethod
    def _parse_presentation(file_path: str, metadata: Dict[str, Any]) -> str:
        """Extract text from PowerPoint slides."""
        if not PPTX_AVAILABLE:
            print(f"   WARNING: python-pptx not installed, skipping presentation: {file_path}")
            return ""

        try:
            presentation = Presentation(file_path)
            text_parts = []
            for slide_number, slide in enumerate(presentation.slides, start=1):
                slide_text = []
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        slide_text.append(shape.text.strip())
                if slide_text:
                    text_parts.append(f"[Slide {slide_number}]\n" + "\n".join(slide_text))
            return "\n\n".join(text_parts)
        except Exception as e:
            print(f"   ERROR: Presentation parse error: {e}")
            return ""

    @staticmethod
    def _parse_image(file_path: str, metadata: Dict[str, Any]) -> str:
        """OCR text from image files."""
        if not OCR_AVAILABLE:
            print(f"   WARNING: OCR not available (pytesseract/Pillow), skipping image: {file_path}")
            return ""
        
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            if text.strip():
                metadata["ocr_used"] = True
            return text
        except Exception as e:
            print(f"   ERROR: Image OCR error: {e}")
            return ""


def collect_files(directory: str) -> List[str]:
    """Recursively find all supported files in directory."""
    supported = {
        '.pdf', '.docx', '.txt', '.md',
        '.png', '.jpg', '.jpeg',
        '.xlsx', '.xlsm', '.pptx',
    }
    files = []
    
    if not os.path.exists(directory):
        return files
    
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in supported:
                full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(full_path, directory)
                files.append((full_path, relative_path))
    
    return files


def parse_all_documents(data_dir: str = None) -> List[Dict[str, Any]]:
    """Parse all documents from the data directory."""
    if data_dir is None:
        data_dir = os.getenv("DATA_DIR", "F:\\Dept activities")
    
    if not os.path.exists(data_dir):
        print(f"WARNING: Data directory not found: {data_dir}")
        return []
    
    print(f"Scanning {data_dir}...")
    files = collect_files(data_dir)
    print(f"   Found {len(files)} supported files")
    
    all_documents = []
    success_count = 0
    skipped_count = 0
    failed_count = 0
    
    for file_path, relative_path in files:
        try:
            doc_type = Path(file_path).suffix.lower()
            content, metadata = DocumentParser.parse(file_path, file_type=doc_type)
            
            if content and content.strip():
                document = {
                    "content": content.strip(),
                    "metadata": {
                        "source": "harddisk",
                        "file_path": relative_path,
                        "file_type": doc_type,
                        "filename": metadata.get("filename", ""),
                        "ocr_used": metadata.get("ocr_used", False),
                        "file_size": metadata.get("file_size", 0),
                        "ingested_at": ""
                    }
                }
                all_documents.append(document)
                success_count += 1
            else:
                print(f"   WARNING: No extractable text from: {relative_path}")
                skipped_count += 1
                
        except Exception as e:
            print(f"   ERROR: Failed to parse {relative_path}: {e}")
            failed_count += 1
    
    print("\nSummary:")
    print(f"   Successfully parsed: {success_count}")
    print(f"   Skipped (no text): {skipped_count}")
    print(f"   Failed: {failed_count}")
    
    return all_documents