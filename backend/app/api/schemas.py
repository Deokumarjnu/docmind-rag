"""Pydantic schemas for API request/response models."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    """Content type classification for document pages."""

    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CHART = "chart"
    CODE = "code"
    HANDWRITING = "handwriting"
    MIXED = "mixed"


class ExtractionMethod(str, Enum):
    """Extraction method used for document processing."""

    PYMUPDF = "pymupdf"
    UNSTRUCTURED_ELEMENTS = "unstructured_elements"
    UNSTRUCTURED_AUTO = "unstructured_auto"
    OCR = "ocr"
    VISION_LLM = "vision_llm"


class DocumentMetadata(BaseModel):
    """Metadata for a processed document."""

    source: str
    page: int
    content_type: ContentType
    extraction_method: ExtractionMethod
    language: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    document_title: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class UploadRequest(BaseModel):
    """Request model for document upload."""

    filename: str
    content_type: str


class UploadResponse(BaseModel):
    """Response model for document upload."""

    task_id: str
    filename: str
    status: str
    message: str


class UploadProgressResponse(BaseModel):
    """Response model for upload progress."""

    task_id: str
    status: str
    progress: float
    current_page: Optional[int] = None
    total_pages: Optional[int] = None
    message: Optional[str] = None
    result: Optional[dict[str, Any]] = None


class QueryRequest(BaseModel):
    """Request model for RAG query."""

    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    filters: Optional[dict[str, Any]] = None
    include_sources: bool = True


class SourceDocument(BaseModel):
    """Source document returned with query response."""

    content: str
    page: int
    source: str
    content_type: ContentType
    relevance_score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Response model for RAG query."""

    answer: str
    sources: list[SourceDocument] = Field(default_factory=list)
    query: str
    enhanced_query: Optional[str] = None
    is_valid: bool = True
    retry_count: int = 0


class DocumentInfo(BaseModel):
    """Information about a stored document."""

    document_id: str
    filename: str
    total_chunks: int
    pages: int
    created_at: Optional[datetime] = None
    content_types: list[ContentType] = Field(default_factory=list)
    language: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Response model for listing documents."""

    documents: list[DocumentInfo]
    total: int


class DeleteDocumentResponse(BaseModel):
    """Response model for document deletion."""

    document_id: str
    deleted_chunks: int
    status: str

