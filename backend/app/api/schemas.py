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
    # Additional types from vision processor
    DIAGRAM = "diagram"
    PHOTO = "photo"
    GRAPH = "graph"
    FLOWCHART = "flowchart"
    OTHER = "other"


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
    conversation_id: Optional[str] = None  # For multi-turn context
    document_id: Optional[str] = None  # Scope to specific document
    use_cache: bool = True  # Enable semantic caching
    use_knowledge_graph: bool = True  # Enable multi-hop reasoning


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
    conversation_id: Optional[str] = None
    cache_hit: bool = False  # Whether response was from cache
    graph_entities: Optional[list[dict]] = None  # Related entities from KG


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


# Conversation schemas
class CreateConversationRequest(BaseModel):
    """Request to create a new conversation."""
    
    user_id: Optional[str] = None
    document_id: Optional[str] = None
    title: Optional[str] = None


class ConversationResponse(BaseModel):
    """Response with conversation details."""
    
    id: str
    title: Optional[str]
    document_id: Optional[str]
    message_count: int
    created_at: datetime
    updated_at: datetime


class MessageResponse(BaseModel):
    """Response for a single message."""
    
    id: str
    role: str
    content: str
    timestamp: datetime
    sources: Optional[list[dict]] = None


class ConversationListResponse(BaseModel):
    """Response for listing conversations."""
    
    conversations: list[ConversationResponse]
    total: int


# Cache and Graph stats
class CacheStatsResponse(BaseModel):
    """Cache statistics response."""
    
    hits: int
    misses: int
    sets: int
    hit_rate: str
    total_requests: int


class GraphStatsResponse(BaseModel):
    """Knowledge graph statistics response."""
    
    entity_count: int
    relationship_count: int
    status: str

