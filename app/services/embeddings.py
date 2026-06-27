"""
Embeddings: turns text into vectors using the Gemini embedding model.

Embeddings let us measure semantic similarity: the question and the chunks that
answer it land close together in vector space, even when they share no words.
"""

from google import genai
from google.genai import types
import google.generativeai as genai


class GeminiEmbeddingService:
    """Generates text embeddings via the Gemini API."""

    # Task types tell Gemini how the embedding will be used, which improves
    # retrieval quality (documents and queries are embedded differently).
    _DOCUMENT_TASK = "retrieval_document"
    _QUERY_TASK = "retrieval_query"

    def __init__(self, api_key: str, model: str) -> None:
        """
        Args:
            api_key: The Gemini API key.
            model: The embedding model id (e.g. ``text-embedding-004``).
        """
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embeds a batch of document chunks for storage.

        Args:
            texts: The chunk texts to embed.

        Returns:
            One embedding vector per input text, in the same order.
        """
        vectors: list[list[float]] = []
        for text in texts:
            result = self._client.models.embed_content(
                model=self._model,
                contents=text,
                config=types.EmbedContentConfig(task_type=self._DOCUMENT_TASK),
            )
            vectors.append(result.embeddings[0].values)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """
        Embeds a single user query for similarity search.

        Args:
            text: The query text.

        Returns:
            The query's embedding vector.
        """
        result = self._client.models.embed_content(
            model=self._model,
            contents=text,
            config=types.EmbedContentConfig(task_type=self._QUERY_TASK),
        )
        return result.embeddings[0].values
