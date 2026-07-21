"""Dependency wiring for the API layer."""

from functools import lru_cache

from app.config import get_settings
from app.services.chunker import TextChunker
from app.services.document_loader import DocumentLoader
from app.services.embeddings import GeminiEmbeddingService
from app.services.rag_service import RagService
from app.services.vector_store import VectorStore


@lru_cache
def get_rag_service() -> RagService:
    settings = get_settings()
    return RagService(
        loader=DocumentLoader(max_pdf_pages=settings.max_pdf_pages),
        chunker=TextChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap),
        embeddings=GeminiEmbeddingService(api_key=settings.gemini_api_key, model=settings.gemini_embed_model, dimensions=settings.gemini_embed_dimensions),
        store=VectorStore(persist_dir=settings.chroma_dir, collection_name=settings.collection_name),
        api_key=settings.gemini_api_key,
        chat_model=settings.gemini_chat_model,
        top_k=settings.top_k,
        min_similarity=settings.min_similarity,
    )
