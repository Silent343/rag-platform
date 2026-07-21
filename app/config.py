"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings."""

    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-3.5-flash"
    gemini_embed_model: str = "gemini-embedding-2"
    gemini_embed_dimensions: int = 768
    chroma_dir: str = "./chroma_data"
    collection_name: str = "documents_gemini_embedding_2_v2"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4
    min_similarity: float = 0.4
    max_upload_bytes: int = 10 * 1024 * 1024
    max_pdf_pages: int = 100
    max_question_chars: int = 2000
    rate_limit_requests: int = 30
    rate_limit_window_seconds: int = 60
    cors_origins: str = "http://localhost:4200"
    enable_docs: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

