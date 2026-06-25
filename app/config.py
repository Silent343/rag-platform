"""
Application configuration.

Loads settings from environment variables (and an optional .env file) so that
secrets like the Gemini API key never live in source control.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Strongly-typed application settings.

    Attributes:
        gemini_api_key: API key for Google Gemini (embeddings + generation).
        gemini_chat_model: Model id used to generate answers.
        gemini_embed_model: Model id used to create embeddings.
        chroma_dir: Filesystem path where ChromaDB persists its data.
        collection_name: Name of the ChromaDB collection holding chunks.
        chunk_size: Target size of each text chunk, in characters.
        chunk_overlap: Overlap between consecutive chunks, in characters.
        top_k: How many chunks to retrieve as context for each query.
        cors_origins: Comma-separated origins allowed to call the API.
    """

    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-2.0-flash"
    gemini_embed_model: str = "text-embedding-004"

    chroma_dir: str = "./chroma_data"
    collection_name: str = "documents"

    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4

    cors_origins: str = "http://localhost:4200"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    def cors_origin_list(self) -> list[str]:
        """
        Parses the comma-separated CORS origins into a list.

        Returns:
            A list of allowed origin strings.
        """
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached singleton of the application settings.

    Using an lru_cache means the .env file is read only once per process.

    Returns:
        The shared Settings instance.
    """
    return Settings()
