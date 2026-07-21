"""Embeddings: turns text into vectors using the Gemini embedding model."""

<<<<<<< HEAD
from google import genai
from google.genai import types
=======
Embeddings let us measure semantic similarity: the question and the chunks that
answer it land close together in vector space, even when they share no words.
"""

from google import genai
from google.genai import types
import google.generativeai as genai
>>>>>>> 3970a87806e1d9393c5cf7ec6f3b7e0d2c5f37b5


class GeminiEmbeddingService:
    """Generates RAG embeddings via the current Google GenAI SDK."""

    # Gemini Embedding 2 uses task instructions in the text rather than the
    # legacy ``task_type`` parameter. Keep the formats complementary so Chroma
    # retrieves document chunks for question-answering queries.
    _DOCUMENT_PREFIX = "title: document | text: "
    _QUERY_PREFIX = "task: question answering | query: "

<<<<<<< HEAD
    def __init__(self, api_key: str, model: str, dimensions: int) -> None:
=======
    def __init__(self, api_key: str, model: str) -> None:
        """
        Args:
            api_key: The Gemini API key.
            model: The embedding model id (e.g. ``text-embedding-004``).
        """
>>>>>>> 3970a87806e1d9393c5cf7ec6f3b7e0d2c5f37b5
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._config = types.EmbedContentConfig(output_dimensionality=dimensions)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
<<<<<<< HEAD
        """Embeds document chunks, one vector per ChromaDB item."""
        return [self._embed(self._DOCUMENT_PREFIX + text) for text in texts]
=======
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
>>>>>>> 3970a87806e1d9393c5cf7ec6f3b7e0d2c5f37b5

    def embed_query(self, text: str) -> list[float]:
        """Embeds a question using the complementary RAG query format."""
        return self._embed(self._QUERY_PREFIX + text)

<<<<<<< HEAD
    def _embed(self, text: str) -> list[float]:
        result = self._client.models.embed_content(
            model=self._model,
            contents=text,
            config=self._config,
        )
        return list(result.embeddings[0].values)
=======
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
>>>>>>> 3970a87806e1d9393c5cf7ec6f3b7e0d2c5f37b5
