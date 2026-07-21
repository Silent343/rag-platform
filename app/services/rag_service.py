"""RAG orchestration: ingestion, retrieval and grounded generation."""

import hashlib
import uuid

from google import genai
<<<<<<< HEAD
=======
from google.genai import types
>>>>>>> 3970a87806e1d9393c5cf7ec6f3b7e0d2c5f37b5

from app.errors import ProviderError
from app.services.chunker import TextChunker
from app.services.document_loader import DocumentLoader
from app.services.embeddings import GeminiEmbeddingService
from app.services.vector_store import VectorStore

_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions using ONLY the context provided below. The context is a set of excerpts from the user's documents.

Rules:
- Answer strictly from the context. Do not use outside knowledge.
- If the answer is not contained in the context, reply exactly: "I could not find the answer in the provided documents."
- Be concise and cite the filename and page number used.

Context:
{context}

Question: {question}

Answer:"""

_NOT_FOUND = "I could not find the answer in the provided documents."


class RagService:
    """High-level Retrieval-Augmented Generation service."""

    def __init__(self, loader: DocumentLoader, chunker: TextChunker, embeddings: GeminiEmbeddingService, store: VectorStore, api_key: str, chat_model: str, top_k: int, min_similarity: float) -> None:
        self._loader = loader
        self._chunker = chunker
        self._embeddings = embeddings
        self._store = store
        self._client = genai.Client(api_key=api_key)
        self._chat_model = chat_model
        self._top_k = top_k
        self._min_similarity = min_similarity

    def ingest(self, filename: str, content: bytes) -> dict:
        content_hash = hashlib.sha256(content).hexdigest()
        existing = self._store.find_document_by_hash(content_hash)
        if existing:
            return {"document_id": existing["document_id"], "filename": existing["filename"], "chunks_created": 0, "already_indexed": True}

        chunks = [
            {"text": chunk, "page_number": page_number}
            for page_number, page_text in self._loader.extract_pages(filename, content)
            for chunk in self._chunker.split(page_text)
        ]
        if not chunks:
            raise ValueError("No text could be extracted from the document.")
        try:
            vectors = self._embeddings.embed_documents([chunk["text"] for chunk in chunks])
        except Exception as exc:
            raise ProviderError("Gemini could not index this document. Please try again.") from exc

        document_id = str(uuid.uuid4())
        self._store.add_chunks(document_id, filename, content_hash, chunks, vectors)
        return {"document_id": document_id, "filename": filename, "chunks_created": len(chunks), "already_indexed": False}

    def answer(self, question: str) -> dict:
        try:
            query_vector = self._embeddings.embed_query(question)
        except Exception as exc:
            raise ProviderError("Gemini could not process the question. Please try again.") from exc

        matches = [match for match in self._store.query(query_vector, self._top_k) if match["score"] >= self._min_similarity]
        if not matches:
<<<<<<< HEAD
            return {"answer": _NOT_FOUND, "sources": []}
=======
            return {
                "answer": "I could not find the answer in the provided documents.",
                "sources": [],
            }

        context = self._build_context(matches)
        prompt = _PROMPT_TEMPLATE.format(context=context, question=question)

        response = self._embeddings._client.models.generate_content(
            model=self._chat_model,
            contents=prompt,
        )
>>>>>>> 3970a87806e1d9393c5cf7ec6f3b7e0d2c5f37b5

        try:
            response = self._client.models.generate_content(model=self._chat_model, contents=_PROMPT_TEMPLATE.format(context=self._build_context(matches), question=question))
        except Exception as exc:
            raise ProviderError("Gemini could not generate an answer. Please try again.") from exc
        if not response.text:
            raise ProviderError("Gemini returned an empty answer. Please try again.")
        return {"answer": response.text.strip(), "sources": matches}

    @staticmethod
    def _build_context(matches: list[dict]) -> str:
        return "\n\n---\n\n".join(f"[Source: {match['filename']}, page {match['page_number']}]\n{match['text']}" for match in matches)
