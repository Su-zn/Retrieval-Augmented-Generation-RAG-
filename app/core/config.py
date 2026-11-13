"""Application configuration loaded from environment variables / `.env`."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "development", "docker", "staging", "production", "test"]


class Settings(BaseSettings):
    """Strongly typed runtime configuration.

    Values are read (in order of precedence) from constructor arguments,
    environment variables, and the local `.env` file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Conversational RAG & Interview Booking API"
    app_version: str = "0.1.0"
    environment: Environment = "local"
    debug: bool = False
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"

    # Relational database
    database_url: str = "sqlite+aiosqlite:///./app.db"
    database_echo: bool = False

    # Redis chat memory
    redis_url: str = "redis://localhost:6379/0"
    chat_history_ttl_seconds: int = Field(default=86_400, gt=0)
    chat_history_max_messages: int = Field(default=20, gt=0)

    # Qdrant vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "documents"
    embedding_dimension: int = Field(default=1536, gt=0)

    # LLM & embeddings provider
    llm_api_key: str = ""
    llm_base_url: str | None = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = Field(default=64, gt=0)

    # Retrieval
    rag_top_k: int = Field(default=5, gt=0, le=50)

    # Uploads
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)

    @field_validator("qdrant_api_key", "llm_base_url", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        """Treat empty/whitespace-only environment values as unset."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("qdrant_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @property
    def is_sqlite(self) -> bool:
        """Whether the configured relational database is SQLite."""
        return self.database_url.startswith("sqlite")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings instance."""
    return Settings()


settings: Settings = get_settings()
