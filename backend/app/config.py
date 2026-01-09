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
    # For local: use qdrant_host + qdrant_port
    # For Qdrant Cloud: use qdrant_url + qdrant_api_key
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_url: Optional[str] = None  # Qdrant Cloud URL (e.g., https://xxx.qdrant.io)
    qdrant_api_key: Optional[str] = None  # Qdrant Cloud API key
    qdrant_collection_name: str = "docmind_documents"

    # Redis (for Celery + Semantic Cache)
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_db: int = 1  # Separate DB for semantic cache
    
    # Neo4j Knowledge Graph
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    
    # PostgreSQL (Conversation Memory)
    postgres_url: str = "postgresql://docmind:docmind@localhost:5432/docmind"
    
    # Semantic Cache Settings
    cache_similarity_threshold: float = 0.92  # Cosine similarity for cache hit
    cache_ttl_seconds: int = 86400  # 24 hours
    cache_enabled: bool = True

    # Document Processing
    # Chunk size: 1000 chars (~250 tokens) - good balance of context and precision
    # Overlap: 200 chars (20%) - prevents information loss at boundaries
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_parallel_workers: int = 8
    
    # Large PDF optimizations
    vision_batch_size: int = 10          # Process vision pages in batches
    max_vision_pages: int = 100          # Limit vision API calls for cost control
    embedding_batch_size: int = 100      # Batch size for embedding API calls
    use_vision_for_images: bool = True   # Toggle vision LLM for images/charts

    # Embedding Model
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072

    # LLM Settings (GPT-5.2: 400K context, better reasoning, fewer hallucinations)
    # Note: gpt-4o-mini is used for fast validation tasks (query expansion, reranking)
    llm_model: str = "gpt-5.2"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 8192  # GPT-5.2 supports up to 128K output
    fast_llm_model: str = "gpt-4o-mini"  # For quick validation/expansion tasks

    # Vision Model (GPT-5.2: improved CharXiv reasoning 93.9% vs 75.7%)
    vision_model: str = "gpt-5.2"

    # Upload Settings
    max_upload_size_mb: int = 100
    allowed_extensions: list[str] = [
        "pdf", "docx", "doc", "html", "txt",  # Documents
        "csv", "xlsx", "xls",                  # Spreadsheets
        "json", "jsonl",                       # JSON formats
        "md", "markdown",                      # Markdown
    ]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()

