"""
RAG orchestration: ties together loading, chunking, embedding, storage and
generation into the two high-level operations the API exposes — ingest a
document, and answer a question grounded in the stored documents.
"""

import uuid

from google import genai
from google.genai import types

from app.services.document_loader import DocumentLoader
from app.services.chunker import TextChunker
from app.services.embeddings import GeminiEmbeddingService
from app.services.vector_store import VectorStore


# Prompt template that instructs the model to answer ONLY from the provided
# context and to say so when the answer is not present — the anti-hallucination
# core of a RAG system.
_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions using \
ONLY the context provided below. The context is a set of excerpts from the \
user's documents.

Rules:
- Answer strictly from the context. Do not use outside knowledge.
- If the answer is not contained in the context, reply exactly: \
"I could not find the answer in the provided documents."
- Be concise and cite the source filenames you used.

Context:
{context}

Question: {question}

Answer:"""


class RagService:
    """High-level Retrieval-Augmented Generation service."""

    def __init__(
        self,
        loader: DocumentLoader,
        chunker: TextChunker,
        embeddings: GeminiEmbeddingService,
        store: VectorStore,
        chat_model: str,
        top_k: int,
    ) -> None:
        """
        Args:
            loader: Extracts text from uploaded files.
            chunker: Splits text into chunks.
            embeddings: Produces embeddings via Gemini.
            store: Persists and searches chunk embeddings.
            chat_model: Gemini model id used for answer generation.
            top_k: Number of chunks to retrieve per query.
        """
        self._loader = loader
        self._chunker = chunker
        self._embeddings = embeddings
        self._store = store
        self._chat_model = chat_model
        self._top_k = top_k

    def ingest(self, filename: str, content: bytes) -> dict:
        """
        Ingests one document: extract → chunk → embed → store.

        Args:
            filename: The original file name.
            content: The raw file bytes.

        Returns:
            A dict with ``document_id``, ``filename`` and ``chunks_created``.

        Raises:
            ValueError: If the file yields no extractable text.
        """
        text = self._loader.extract_text(filename, content)
        chunks = self._chunker.split(text)
        if not chunks:
            raise ValueError("No text could be extracted from the document.")

        document_id = str(uuid.uuid4())
        vectors = self._embeddings.embed_documents(chunks)
        self._store.add_chunks(document_id, filename, chunks, vectors)

        return {
            "document_id": document_id,
            "filename": filename,
            "chunks_created": len(chunks),
        }

    def answer(self, question: str) -> dict:
        """
        Answers a question grounded in the stored documents.

        Args:
            question: The user's question.

        Returns:
            A dict with ``answer`` (str) and ``sources`` (list of chunk dicts).
        """
        query_vector = self._embeddings.embed_query(question)
        matches = self._store.query(query_vector, self._top_k)

        if not matches:
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

        return {"answer": response.text.strip(), "sources": matches}

    @staticmethod
    def _build_context(matches: list[dict]) -> str:
        """
        Formats retrieved chunks into a labelled context block for the prompt.

        Args:
            matches: The retrieved chunks.

        Returns:
            A string with each chunk prefixed by its source filename.
        """
        blocks = []
        for match in matches:
            blocks.append(f"[Source: {match['filename']}]\n{match['text']}")
        return "\n\n---\n\n".join(blocks)
