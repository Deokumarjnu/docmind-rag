"""Cross-encoder reranking for improved precision.

This module uses cross-encoder models to rerank retrieved documents
for better precision in the final results.
"""

import logging
from typing import Optional

from langchain_core.documents import Document

from app.config import settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder reranker for improved retrieval precision.
    
    Uses sentence-transformers cross-encoder models to score
    query-document pairs more accurately than bi-encoders.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model name for cross-encoder
            device: Device to use (cpu, cuda, mps)
        """
        self.model_name = model_name
        self.model = None
        self.device = device
        
        self._load_model()

    def _load_model(self) -> None:
        """Load the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            
            self.model = CrossEncoder(
                self.model_name,
                device=self.device,
            )
            logger.info(f"Loaded cross-encoder model: {self.model_name}")
            
        except ImportError:
            logger.warning(
                "sentence-transformers not installed, reranking disabled. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: Optional[int] = None,
    ) -> list[tuple[Document, float]]:
        """
        Rerank documents using cross-encoder.
        
        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            List of (document, score) tuples, sorted by score
        """
        if not self.model or not documents:
            return [(doc, 0.0) for doc in documents]
        
        # Create query-document pairs
        pairs = [(query, doc.page_content) for doc in documents]
        
        try:
            # Score all pairs
            scores = self.model.predict(pairs)
            
            # Combine documents with scores
            doc_scores = list(zip(documents, scores))
            
            # Sort by score (descending)
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k if specified
            if top_k:
                doc_scores = doc_scores[:top_k]
            
            return doc_scores
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return [(doc, 0.0) for doc in documents]

    def rerank_with_threshold(
        self,
        query: str,
        documents: list[Document],
        threshold: float = 0.0,
        top_k: Optional[int] = None,
    ) -> list[Document]:
        """
        Rerank and filter documents by score threshold.
        
        Args:
            query: Search query
            documents: Documents to rerank
            threshold: Minimum score threshold
            top_k: Maximum documents to return
            
        Returns:
            List of documents passing threshold
        """
        ranked = self.rerank(query, documents, top_k)
        
        return [
            doc for doc, score in ranked
            if score >= threshold
        ]


class LLMReranker:
    """
    LLM-based reranker using GPT to score relevance.
    
    More expensive but can be more accurate for complex queries.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize LLM reranker.
        
        Args:
            model_name: LLM model to use
        """
        from langchain_openai import ChatOpenAI
        
        self.model = ChatOpenAI(
            model=model_name or settings.fast_llm_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:
        """
        Rerank documents using LLM scoring.
        
        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            List of (document, score) tuples
        """
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a relevance scorer. Given a query and a document,
score how relevant the document is to answering the query.

Score from 0 to 10:
- 0-2: Not relevant at all
- 3-4: Slightly relevant, mentions similar topics
- 5-6: Moderately relevant, contains some useful information
- 7-8: Highly relevant, directly addresses the query
- 9-10: Perfectly relevant, completely answers the query

Reply with ONLY the numeric score."""),
            ("user", "Query: {query}\n\nDocument:\n{document}\n\nScore:"),
        ])
        
        chain = prompt | self.model | StrOutputParser()
        
        scored = []
        for doc in documents:
            try:
                score_str = chain.invoke({
                    "query": query,
                    "document": doc.page_content[:2000],  # Truncate long docs
                })
                score = float(score_str.strip()) / 10.0  # Normalize to 0-1
                scored.append((doc, score))
            except Exception as e:
                logger.warning(f"Failed to score document: {e}")
                scored.append((doc, 0.5))  # Default middle score
        
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored[:top_k]


def create_reranker(
    reranker_type: str = "cross_encoder",
    model_name: Optional[str] = None,
) -> CrossEncoderReranker | LLMReranker:
    """
    Create a reranker instance.
    
    Args:
        reranker_type: "cross_encoder" or "llm"
        model_name: Model name to use
        
    Returns:
        Reranker instance
    """
    if reranker_type == "llm":
        return LLMReranker(model_name)
    else:
        return CrossEncoderReranker(model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2")

