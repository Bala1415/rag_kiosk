"""
Vector Store Module - Handles ChromaDB operations for the RAG system.
"""
import os
import chromadb
import torch
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Configuration
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "department_kb")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto").lower()


def select_embedding_device() -> str:
    """Select the configured embedding device, falling back to CPU safely."""
    if EMBEDDING_DEVICE in {"cpu", "cuda"}:
        if EMBEDDING_DEVICE == "cuda" and not torch.cuda.is_available():
            print("CUDA was requested but is unavailable; using CPU.")
            return "cpu"
        return EMBEDDING_DEVICE

    return "cuda" if torch.cuda.is_available() else "cpu"


class VectorStore:
    """Manages ChromaDB vector database for document storage and retrieval."""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        self.embedding_device = select_embedding_device()
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=self.embedding_device,
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Vector store initialized at {CHROMA_DB_PATH}")
        print(f"   Collection: {COLLECTION_NAME}")
        print(f"   Embedding model: {EMBEDDING_MODEL}")
        print(f"   Embedding device: {self.embedding_device}")
        if self.embedding_device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= CHUNK_SIZE:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = end - CHUNK_OVERLAP
        return chunks
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of dicts with keys: 'content', 'metadata'
                      metadata should include: source, file_path, file_type, etc.
        Returns:
            Number of chunks added
        """
        if not documents:
            return 0
        
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for doc in documents:
            content = doc.get("content", "").strip()
            if not content:
                continue
            
            metadata = doc.get("metadata", {})
            chunks = self._chunk_text(content)
            source_id = (
                metadata.get("file_path")
                or metadata.get("url")
                or metadata.get("source")
                or "unknown"
            )
            safe_source_id = str(source_id).replace("\\", "/")

            for i, chunk in enumerate(chunks):
                chunk_id = f"{safe_source_id}_{i}"
                all_chunks.append(chunk)
                all_metadatas.append({**metadata, "chunk_index": i})
                all_ids.append(chunk_id)
        
        if not all_chunks:
            return 0
        
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        batch_size = 128
        for start in range(0, len(all_chunks), batch_size):
            end = min(start + batch_size, len(all_chunks))
            batch_chunks = all_chunks[start:end]
            embeddings = self.embedding_model.encode(
                batch_chunks,
                show_progress_bar=True,
            ).tolist()
            self.collection.upsert(
                documents=batch_chunks,
                embeddings=embeddings,
                metadatas=all_metadatas[start:end],
                ids=all_ids[start:end],
            )

        print(f"Added {len(all_chunks)} chunks to vector store")
        return len(all_chunks)
    
    def query(self, query_text: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar documents.
        
        Returns:
            List of dicts with keys: content, metadata, distance
        """
        if top_k is None:
            top_k = TOP_K_RESULTS
        
        query_embedding = self.embedding_model.encode([query_text]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                formatted.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                })
        
        return formatted
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL
        }
    
    def clear(self):
        """Clear all documents from the collection."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print("Vector store cleared")


# Singleton instance
_vector_store_instance: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create the singleton VectorStore instance."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance