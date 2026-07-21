"""Embeddings: turns text into vectors using the Gemini embedding model."""

from google import genai
from google.genai import types


class GeminiEmbeddingService:
    """Generates RAG embeddings via the current Google GenAI SDK."""

    _DOCUMENT_PREFIX = "title: document | text: "
    _QUERY_PREFIX = "task: question answering | query: "

    def __init__(self, api_key: str, model: str, dimensions: int) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._config = types.EmbedContentConfig(output_dimensionality=dimensions)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds document chunks, one vector per ChromaDB item."""
        return [self._embed(self._DOCUMENT_PREFIX + text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embeds a question using the complementary RAG query format."""
        return self._embed(self._QUERY_PREFIX + text)

    def _embed(self, text: str) -> list[float]:
        result = self._client.models.embed_content(
            model=self._model,
            contents=text,
            config=self._config,
        )
        return list(result.embeddings[0].values)
