"""Hybrid retriever combining dense embeddings and BM25 sparse search.

This module implements hybrid retrieval for better accuracy,
combining semantic search with keyword matching.
"""

import logging
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field

from app.vectorstore.store import get_vector_store, similarity_search_with_score
from app.config import settings

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    Simple BM25 retriever for sparse keyword matching.
    """

    def __init__(self, documents: Optional[list[Document]] = None):
        """
        Initialize BM25 retriever.
        
        Args:
            documents: Documents to index
        """
        self.documents = documents or []
        self.bm25 = None
        self._tokenized_corpus = []
        
        if documents:
            self._index_documents(documents)

    def _index_documents(self, documents: list[Document]) -> None:
        """Index documents for BM25 search."""
        try:
            from rank_bm25 import BM25Okapi
            
            self._tokenized_corpus = [
                self._tokenize(doc.page_content)
                for doc in documents
            ]
            self.bm25 = BM25Okapi(self._tokenized_corpus)
            self.documents = documents
            
        except ImportError:
            logger.warning("rank-bm25 not installed, BM25 retrieval disabled")

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        return text.lower().split()

    def search(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        """
        Search documents using BM25.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of (document, score) tuples
        """
        if not self.bm25 or not self.documents:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]
        
        return [
            (self.documents[i], scores[i])
            for i in top_indices
            if scores[i] > 0
        ]

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the index."""
        self.documents.extend(documents)
        self._index_documents(self.documents)


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining dense vector search and BM25 sparse search.
    
    Uses Reciprocal Rank Fusion (RRF) to combine results from both methods.
    """

    dense_weight: float = Field(default=0.6, description="Weight for dense retrieval")
    sparse_weight: float = Field(default=0.4, description="Weight for sparse retrieval")
    k: int = Field(default=5, description="Number of documents to retrieve")
    collection_name: Optional[str] = Field(default=None)
    bm25_retriever: Optional[Any] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        k: int = 5,
        collection_name: Optional[str] = None,
        documents: Optional[list[Document]] = None,
        **kwargs
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            dense_weight: Weight for dense (semantic) retrieval
            sparse_weight: Weight for sparse (BM25) retrieval
            k: Number of documents to retrieve
            collection_name: Vector store collection name
            documents: Documents for BM25 indexing
        """
        super().__init__(
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            k=k,
            collection_name=collection_name,
            **kwargs
        )
        
        if documents:
            self.bm25_retriever = BM25Retriever(documents)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """
        Retrieve relevant documents using hybrid search.
        
        Args:
            query: Search query
            run_manager: Callback manager
            
        Returns:
            List of relevant documents
        """
        # Dense retrieval
        dense_results = similarity_search_with_score(
            query,
            k=self.k * 2,  # Get more for fusion
            collection_name=self.collection_name,
        )
        
        # Sparse retrieval - ensure BM25 is initialized
        sparse_results = []
        if self.bm25_retriever is None:
            # Try to initialize BM25 from vector store
            self.ensure_bm25_initialized()
        
        if self.bm25_retriever:
            sparse_results = self.bm25_retriever.search(query, k=self.k * 2)
        
        # Combine using RRF
        combined = self._reciprocal_rank_fusion(
            dense_results,
            sparse_results,
        )
        
        return combined[:self.k]

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[tuple[Document, float]],
        sparse_results: list[tuple[Document, float]],
        k_constant: int = 60,
    ) -> list[Document]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            k_constant: RRF constant (typically 60)
            
        Returns:
            Combined and reranked documents
        """
        doc_scores: dict[str, dict] = {}
        
        # Score dense results
        for rank, (doc, score) in enumerate(dense_results):
            doc_id = self._get_doc_id(doc)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "score": 0}
            
            rrf_score = self.dense_weight / (k_constant + rank + 1)
            doc_scores[doc_id]["score"] += rrf_score
        
        # Score sparse results
        for rank, (doc, score) in enumerate(sparse_results):
            doc_id = self._get_doc_id(doc)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "score": 0}
            
            rrf_score = self.sparse_weight / (k_constant + rank + 1)
            doc_scores[doc_id]["score"] += rrf_score
        
        # Sort by combined score
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        return [item["doc"] for item in sorted_docs]

    def _get_doc_id(self, doc: Document) -> str:
        """Generate a unique ID for a document."""
        # Use source and page as ID, or hash content
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", 0)
        chunk_idx = doc.metadata.get("chunk_index", 0)
        
        return f"{source}:{page}:{chunk_idx}"

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to BM25 index."""
        if self.bm25_retriever is None:
            self.bm25_retriever = BM25Retriever(documents)
        else:
            self.bm25_retriever.add_documents(documents)

    def rebuild_bm25_from_store(self, limit: int = 10000) -> int:
        """
        Rebuild BM25 index from all documents in the vector store.
        
        This ensures the sparse retriever is synced with the dense retriever.
        
        Args:
            limit: Maximum number of documents to index
            
        Returns:
            Number of documents indexed
        """
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter
        
        from app.vectorstore.store import get_qdrant_client
        from app.config import settings
        
        client = get_qdrant_client()
        collection = self.collection_name or settings.qdrant_collection_name
        
        try:
            # Scroll through all documents
            documents = []
            offset = None
            
            while len(documents) < limit:
                results, offset = client.scroll(
                    collection_name=collection,
                    limit=min(1000, limit - len(documents)),
                    offset=offset,
                    with_payload=True,
                )
                
                for point in results:
                    payload = point.payload or {}
                    content = payload.get("page_content", "")
                    if content:
                        doc = Document(
                            page_content=content,
                            metadata={
                                k: v for k, v in payload.items() 
                                if k != "page_content"
                            }
                        )
                        documents.append(doc)
                
                if offset is None:
                    break
            
            if documents:
                self.bm25_retriever = BM25Retriever(documents)
                logger.info(f"Rebuilt BM25 index with {len(documents)} documents")
            
            return len(documents)
            
        except Exception as e:
            logger.error(f"Failed to rebuild BM25 index: {e}")
            return 0

    def ensure_bm25_initialized(self) -> bool:
        """
        Ensure BM25 index is initialized from vector store.
        
        Returns:
            True if BM25 is ready
        """
        if self.bm25_retriever is None or not self.bm25_retriever.documents:
            count = self.rebuild_bm25_from_store()
            return count > 0
        return True


def create_hybrid_retriever(
    collection_name: Optional[str] = None,
    documents: Optional[list[Document]] = None,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
    k: int = 5,
) -> HybridRetriever:
    """
    Create a hybrid retriever instance.
    
    Args:
        collection_name: Vector store collection
        documents: Documents for BM25
        dense_weight: Weight for dense retrieval
        sparse_weight: Weight for sparse retrieval
        k: Number of results
        
    Returns:
        HybridRetriever instance
    """
    return HybridRetriever(
        collection_name=collection_name,
        documents=documents,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        k=k,
    )

