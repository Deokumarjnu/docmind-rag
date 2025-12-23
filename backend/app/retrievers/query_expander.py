"""Query expansion and enhancement for better retrieval.

This module implements query rewriting, expansion, and enhancement
for handling fuzzy, ambiguous, or acronym-heavy queries.
"""

import logging
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app.config import settings
from app.ingestion.acronym_expander import AcronymExpander

logger = logging.getLogger(__name__)


class QueryExpander:
    """
    Query expander for improving retrieval accuracy.
    
    Performs:
    - Acronym expansion
    - Query rewriting for clarity
    - Multi-query generation for broader coverage
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        acronym_dict: Optional[dict[str, str]] = None,
    ):
        """
        Initialize query expander.
        
        Args:
            model_name: LLM model to use
            acronym_dict: Dictionary of acronyms to expand
        """
        self.model = ChatOpenAI(
            model=model_name or "gpt-4o-mini",
            temperature=0.3,
            api_key=settings.openai_api_key,
        )
        self.acronym_expander = AcronymExpander()
        
        if acronym_dict:
            self.acronym_expander.acronym_dict.update(acronym_dict)

    def expand_acronyms(self, query: str) -> str:
        """
        Expand acronyms in query.
        
        Args:
            query: Original query
            
        Returns:
            Query with acronyms expanded
        """
        return self.acronym_expander.expand_query(query)

    def rewrite_query(self, query: str) -> str:
        """
        Rewrite query for better retrieval.
        
        Args:
            query: Original query
            
        Returns:
            Rewritten query
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query rewriter for a document search system.
Rewrite the user's query to be clearer and more specific for retrieval.

Guidelines:
- Expand abbreviations and acronyms
- Make implicit concepts explicit
- Use synonyms that might appear in documents
- Keep the rewritten query concise (1-2 sentences)
- Preserve the original intent

Reply with ONLY the rewritten query, nothing else."""),
            ("user", "Original query: {query}"),
        ])
        
        chain = prompt | self.model | StrOutputParser()
        
        try:
            rewritten = chain.invoke({"query": query})
            return rewritten.strip()
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}")
            return query

    def generate_sub_queries(self, query: str, num_queries: int = 3) -> list[str]:
        """
        Generate multiple sub-queries for broader coverage.
        
        Args:
            query: Original query
            num_queries: Number of sub-queries to generate
            
        Returns:
            List of sub-queries
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a query decomposer for a document search system.
Break down the user's query into {num_queries} simpler, more specific sub-queries.

Each sub-query should:
- Focus on one aspect of the original question
- Be self-contained and searchable
- Use different phrasings or synonyms
- Together cover the full intent of the original query

Reply with exactly {num_queries} queries, one per line, without numbering."""),
            ("user", "Original query: {query}"),
        ])
        
        chain = prompt | self.model | StrOutputParser()
        
        try:
            result = chain.invoke({"query": query})
            sub_queries = [q.strip() for q in result.strip().split('\n') if q.strip()]
            return sub_queries[:num_queries]
        except Exception as e:
            logger.warning(f"Sub-query generation failed: {e}")
            return [query]

    def enhance_query(
        self,
        query: str,
        expand_acronyms: bool = True,
        rewrite: bool = True,
    ) -> dict[str, str | list[str]]:
        """
        Fully enhance a query with all available methods.
        
        Args:
            query: Original query
            expand_acronyms: Whether to expand acronyms
            rewrite: Whether to rewrite the query
            
        Returns:
            Dictionary with enhanced query variants
        """
        result = {
            "original": query,
            "expanded": query,
            "rewritten": query,
            "sub_queries": [query],
        }
        
        # Expand acronyms
        if expand_acronyms:
            result["expanded"] = self.expand_acronyms(query)
        
        # Rewrite for clarity
        if rewrite:
            result["rewritten"] = self.rewrite_query(result["expanded"])
        
        return result


class HypotheticalDocumentEmbedder:
    """
    HyDE: Generate hypothetical document for better semantic search.
    
    Creates a hypothetical answer document, embeds it, and uses
    that embedding for retrieval (better than query embedding alone).
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize HyDE.
        
        Args:
            model_name: LLM model to use
        """
        self.model = ChatOpenAI(
            model=model_name or "gpt-4o-mini",
            temperature=0.7,
            api_key=settings.openai_api_key,
        )

    def generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document that would answer the query.
        
        Args:
            query: User query
            
        Returns:
            Hypothetical document text
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert document writer. Given a question,
write a short paragraph that would be a perfect answer to that question.
Write as if you're writing content that would appear in a document or article.
Be specific and informative. Do not include any preamble or explanation."""),
            ("user", "Question: {query}\n\nAnswer paragraph:"),
        ])
        
        chain = prompt | self.model | StrOutputParser()
        
        try:
            return chain.invoke({"query": query}).strip()
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")
            return query


def create_query_expander(
    model_name: Optional[str] = None,
    acronym_dict: Optional[dict[str, str]] = None,
) -> QueryExpander:
    """
    Create a query expander instance.
    
    Args:
        model_name: LLM model to use
        acronym_dict: Acronym dictionary
        
    Returns:
        QueryExpander instance
    """
    return QueryExpander(model_name, acronym_dict)

