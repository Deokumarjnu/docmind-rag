"""Vector store initialization and management using Qdrant."""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import settings

logger = logging.getLogger(__name__)


def get_embedding_model() -> Embeddings:
    """Get the configured embedding model."""
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        openai_api_key=settings.openai_api_key,
    )


def get_qdrant_client() -> QdrantClient:
    """Get a Qdrant client instance.
    
    Supports both local Qdrant and Qdrant Cloud.
    - Local: Uses qdrant_host + qdrant_port
    - Cloud: Uses qdrant_url + qdrant_api_key
    """
    # Use Qdrant Cloud if URL is provided
    if settings.qdrant_url:
        return QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
    
    # Otherwise use local Qdrant
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )


def ensure_collection_exists(
    client: QdrantClient,
    collection_name: Optional[str] = None,
) -> None:
    """
    Ensure the Qdrant collection exists, create if not.
    
    Args:
        client: Qdrant client
        collection_name: Collection name (defaults to config value)
    """
    collection_name = collection_name or settings.qdrant_collection_name
    
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if collection_name not in collection_names:
        logger.info(f"Creating collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=settings.embedding_dimensions,
                distance=Distance.COSINE,
            ),
        )
    else:
        logger.debug(f"Collection exists: {collection_name}")


def get_vector_store(
    collection_name: Optional[str] = None,
    embeddings: Optional[Embeddings] = None,
) -> QdrantVectorStore:
    """
    Get a LangChain Qdrant vector store instance.
    
    Args:
        collection_name: Qdrant collection name
        embeddings: Embedding model (defaults to OpenAI)
        
    Returns:
        QdrantVectorStore instance
    """
    collection_name = collection_name or settings.qdrant_collection_name
    embeddings = embeddings or get_embedding_model()
    
    client = get_qdrant_client()
    ensure_collection_exists(client, collection_name)
    
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )


async def add_documents(
    documents: list[Document],
    collection_name: Optional[str] = None,
) -> list[str]:
    """
    Add documents to the vector store.
    
    Args:
        documents: List of documents to add
        collection_name: Target collection name
        
    Returns:
        List of document IDs
    """
    vector_store = get_vector_store(collection_name)
    
    # Add documents and return IDs
    ids = await vector_store.aadd_documents(documents)
    logger.info(f"Added {len(documents)} documents to {collection_name or settings.qdrant_collection_name}")
    
    return ids


def add_documents_sync(
    documents: list[Document],
    collection_name: Optional[str] = None,
) -> list[str]:
    """
    Add documents to the vector store (sync version).
    
    Args:
        documents: List of documents to add
        collection_name: Target collection name
        
    Returns:
        List of document IDs
    """
    vector_store = get_vector_store(collection_name)
    
    # Add documents and return IDs
    ids = vector_store.add_documents(documents)
    logger.info(f"Added {len(documents)} documents to {collection_name or settings.qdrant_collection_name}")
    
    return ids


def similarity_search(
    query: str,
    k: int = 5,
    collection_name: Optional[str] = None,
    filter_dict: Optional[dict] = None,
) -> list[Document]:
    """
    Perform similarity search in the vector store.
    
    Args:
        query: Search query
        k: Number of results to return
        collection_name: Target collection name
        filter_dict: Optional metadata filters
        
    Returns:
        List of matching documents
    """
    vector_store = get_vector_store(collection_name)
    
    if filter_dict:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        conditions = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in filter_dict.items()
        ]
        qdrant_filter = Filter(must=conditions)
        results = vector_store.similarity_search(query, k=k, filter=qdrant_filter)
    else:
        results = vector_store.similarity_search(query, k=k)
    
    return results


def similarity_search_with_score(
    query: str,
    k: int = 5,
    collection_name: Optional[str] = None,
) -> list[tuple[Document, float]]:
    """
    Perform similarity search with relevance scores.
    
    Args:
        query: Search query
        k: Number of results to return
        collection_name: Target collection name
        
    Returns:
        List of (document, score) tuples
    """
    vector_store = get_vector_store(collection_name)
    return vector_store.similarity_search_with_score(query, k=k)

