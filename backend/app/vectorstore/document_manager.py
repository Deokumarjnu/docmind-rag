"""Document manager for CRUD operations on vector store.

This module provides document lifecycle management including
deletion, updates, and incremental additions.
"""

import logging
from typing import Optional

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.config import settings
from app.vectorstore.store import (
    get_qdrant_client,
    get_vector_store,
    add_documents_sync,
)

logger = logging.getLogger(__name__)


class DocumentManager:
    """
    Document manager for vector store operations.
    
    Provides methods to:
    - Delete documents by source/ID
    - Update existing documents
    - Add pages incrementally
    - List all documents
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize document manager.
        
        Args:
            collection_name: Vector store collection name
        """
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.client = get_qdrant_client()

    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks belonging to a document.
        
        Args:
            document_id: Document identifier (usually source path)
            
        Returns:
            Number of points deleted
        """
        # Count before deletion
        count_before = self._count_by_source(document_id)
        
        # Delete by source filter
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
        )
        
        # Count after deletion
        count_after = self._count_by_source(document_id)
        deleted = count_before - count_after
        
        logger.info(f"Deleted {deleted} chunks for document: {document_id}")
        return deleted

    def update_document(
        self,
        document_id: str,
        new_chunks: list[Document],
    ) -> int:
        """
        Update document by replacing all chunks.
        
        Args:
            document_id: Document identifier
            new_chunks: New document chunks
            
        Returns:
            Number of new chunks added
        """
        # Delete old chunks
        self.delete_document(document_id)
        
        # Ensure all chunks have the correct source
        for chunk in new_chunks:
            chunk.metadata["source"] = document_id
        
        # Add new chunks
        add_documents_sync(new_chunks, self.collection_name)
        
        logger.info(f"Updated document {document_id} with {len(new_chunks)} chunks")
        return len(new_chunks)

    def add_pages(
        self,
        document_id: str,
        page_chunks: list[Document],
    ) -> int:
        """
        Add specific pages to an existing document.
        
        Args:
            document_id: Document identifier
            page_chunks: Chunks from new pages
            
        Returns:
            Number of chunks added
        """
        # Ensure source is set
        for chunk in page_chunks:
            chunk.metadata["source"] = document_id
        
        # Add chunks
        add_documents_sync(page_chunks, self.collection_name)
        
        logger.info(f"Added {len(page_chunks)} chunks to document: {document_id}")
        return len(page_chunks)

    def delete_pages(
        self,
        document_id: str,
        page_numbers: list[int],
    ) -> int:
        """
        Delete specific pages from a document.
        
        Args:
            document_id: Document identifier
            page_numbers: Page numbers to delete
            
        Returns:
            Number of chunks deleted
        """
        deleted = 0
        
        for page_num in page_numbers:
            # Delete chunks for this page
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=document_id)
                        ),
                        FieldCondition(
                            key="page",
                            match=MatchValue(value=page_num)
                        )
                    ]
                )
            )
            deleted += 1  # Approximate
        
        logger.info(f"Deleted pages {page_numbers} from document: {document_id}")
        return deleted

    def list_documents(self) -> list[dict]:
        """
        List all unique documents in the collection.
        
        Returns:
            List of document info dictionaries
        """
        # Scroll through all points to get unique sources
        sources = {}
        offset = None
        
        while True:
            results, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                offset=offset,
                with_payload=["source", "page", "content_type"],
            )
            
            for point in results:
                source = point.payload.get("source", "unknown")
                page = point.payload.get("page", 0)
                content_type = point.payload.get("content_type", "text")
                
                if source not in sources:
                    sources[source] = {
                        "document_id": source,
                        "chunk_count": 0,
                        "pages": set(),
                        "content_types": set(),
                    }
                
                sources[source]["chunk_count"] += 1
                sources[source]["pages"].add(page)
                sources[source]["content_types"].add(content_type)
            
            if offset is None:
                break
        
        # Format results
        documents = []
        for source, info in sources.items():
            documents.append({
                "document_id": source,
                "filename": source.split("/")[-1] if "/" in source else source,
                "total_chunks": info["chunk_count"],
                "pages": len(info["pages"]),
                "content_types": list(info["content_types"]),
            })
        
        return documents

    def get_document_info(self, document_id: str) -> Optional[dict]:
        """
        Get information about a specific document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document info or None if not found
        """
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=1000,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=document_id)
                    )
                ]
            ),
            with_payload=["source", "page", "content_type", "chunk_index"],
        )
        
        if not results:
            return None
        
        pages = set()
        content_types = set()
        
        for point in results:
            pages.add(point.payload.get("page", 0))
            content_types.add(point.payload.get("content_type", "text"))
        
        return {
            "document_id": document_id,
            "filename": document_id.split("/")[-1] if "/" in document_id else document_id,
            "total_chunks": len(results),
            "pages": sorted(list(pages)),
            "content_types": list(content_types),
        }

    def _count_by_source(self, source: str) -> int:
        """Count chunks for a source document."""
        try:
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=1,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=source)
                        )
                    ]
                ),
                with_payload=False,
            )
            
            # This is approximate - full count requires aggregation
            return len(results)
        except Exception:
            return 0

    def search_by_content_type(
        self,
        content_type: str,
        query: str,
        k: int = 5,
    ) -> list[Document]:
        """
        Search within specific content type.
        
        Args:
            content_type: Filter by content type
            query: Search query
            k: Number of results
            
        Returns:
            List of matching documents
        """
        vector_store = get_vector_store(self.collection_name)
        
        results = vector_store.similarity_search(
            query,
            k=k,
            filter=Filter(
                must=[
                    FieldCondition(
                        key="content_type",
                        match=MatchValue(value=content_type)
                    )
                ]
            )
        )
        
        return results

