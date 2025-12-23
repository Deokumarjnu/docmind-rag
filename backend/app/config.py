"""Configuration settings for DocMind RAG application."""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "DocMind RAG"
    app_version: str = "1.0.0"
    debug: bool = False

    # API Keys
    openai_api_key: str = ""
    langsmith_api_key: Optional[str] = None

    # LangSmith Tracing
    langsmith_project: str = "docmind-rag"
    langchain_tracing_v2: bool = False

    # Vector Store (Qdrant)
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "docmind_documents"

    # Redis (for Celery)
    redis_url: str = "redis://localhost:6379/0"

    # Document Processing
    chunk_size: int = 700
    chunk_overlap: int = 100
    max_parallel_workers: int = 8

    # Embedding Model
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072

    # LLM Settings
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096

    # Vision Model
    vision_model: str = "gpt-4o"

    # Upload Settings
    max_upload_size_mb: int = 100
    allowed_extensions: list[str] = ["pdf", "docx", "html", "txt"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()

