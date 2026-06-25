"""
Dependency wiring for the API layer.

Builds the RAG service graph once and exposes it to the routes via FastAPI's
dependency-injection system. Centralizing construction here keeps the routes
thin and makes the wiring easy to follow.
"""

from functools import lru_cache

from app.config import get_settings
from app.services.document_loader import DocumentLoader
from app.services.chunker import TextChunker
from app.services.embeddings import GeminiEmbeddingService
from app.services.vector_store import VectorStore
from app.services.rag_service import RagService


@lru_cache
def get_rag_service() -> RagService:
    """
    Builds (once) and returns the fully-wired RAG service.

    The lru_cache ensures a single shared instance — and therefore a single
    ChromaDB client and Gemini configuration — across all requests.

    Returns:
        The shared RagService instance.
    """
    settings = get_settings()

    loader = DocumentLoader()
    chunker = TextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    embeddings = GeminiEmbeddingService(
        api_key=settings.gemini_api_key,
        model=settings.gemini_embed_model,
    )
    store = VectorStore(
        persist_dir=settings.chroma_dir,
        collection_name=settings.collection_name,
    )

    return RagService(
        loader=loader,
        chunker=chunker,
        embeddings=embeddings,
        store=store,
        chat_model=settings.gemini_chat_model,
        top_k=settings.top_k,
    )
