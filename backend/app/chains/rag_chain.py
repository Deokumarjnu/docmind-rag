"""Basic RAG chain for simple query-answer workflows.

This module provides a simpler RAG chain without the full
agentic self-correction loop for faster responses.
"""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from app.config import settings
from app.vectorstore.store import similarity_search, get_vector_store

logger = logging.getLogger(__name__)


def format_docs(docs: list[Document]) -> str:
    """Format documents for the prompt."""
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")
        formatted.append(
            f"[Document {i+1} - Source: {source}, Page: {page}]\n{doc.page_content}"
        )
    return "\n\n".join(formatted)


def create_rag_chain(
    model_name: Optional[str] = None,
    k: int = 5,
    collection_name: Optional[str] = None,
):
    """
    Create a basic RAG chain.
    
    Args:
        model_name: LLM model to use
        k: Number of documents to retrieve
        collection_name: Vector store collection
        
    Returns:
        Runnable RAG chain
    """
    # Get vector store
    vector_store = get_vector_store(collection_name)
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    
    # Create LLM
    llm = ChatOpenAI(
        model=model_name or settings.llm_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on provided documents.

Guidelines:
- Answer based ONLY on the provided documents
- If the documents don't contain enough information, say so
- Cite sources using [Source: document name, page X] notation
- Be concise but complete"""),
        ("user", """Context documents:
{context}

Question: {question}

Answer based on the documents above:"""),
    ])
    
    # Build chain
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


def create_rag_chain_with_sources(
    model_name: Optional[str] = None,
    k: int = 5,
    collection_name: Optional[str] = None,
):
    """
    Create a RAG chain that returns sources with the answer.
    
    Args:
        model_name: LLM model to use
        k: Number of documents to retrieve
        collection_name: Vector store collection
        
    Returns:
        Function that returns answer and sources
    """
    # Get vector store
    vector_store = get_vector_store(collection_name)
    
    # Create LLM
    llm = ChatOpenAI(
        model=model_name or settings.llm_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on provided documents.

Guidelines:
- Answer based ONLY on the provided documents
- If the documents don't contain enough information, say so
- Be concise but complete
- Reference specific information from documents when relevant"""),
        ("user", """Context documents:
{context}

Question: {question}

Answer:"""),
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    def invoke_with_sources(query: str) -> dict:
        """Invoke chain and return answer with sources."""
        # Retrieve documents
        docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
        docs = [doc for doc, score in docs_with_scores]
        
        # Format for prompt
        context = format_docs(docs)
        
        # Generate answer
        answer = chain.invoke({
            "context": context,
            "question": query,
        })
        
        # Format sources
        sources = [
            {
                "content": doc.page_content[:300],
                "page": doc.metadata.get("page"),
                "source": doc.metadata.get("source"),
                "content_type": doc.metadata.get("content_type"),
                "relevance_score": float(score),
            }
            for doc, score in docs_with_scores
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "query": query,
        }
    
    return invoke_with_sources


class SimpleRAG:
    """
    Simple RAG wrapper for quick query-answer.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        k: int = 5,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize simple RAG.
        
        Args:
            model_name: LLM model
            k: Number of documents
            collection_name: Vector store collection
        """
        self.k = k
        self.collection_name = collection_name
        self.chain_with_sources = create_rag_chain_with_sources(
            model_name, k, collection_name
        )

    def query(self, question: str) -> dict:
        """
        Query the RAG system.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and sources
        """
        return self.chain_with_sources(question)

    def __call__(self, question: str) -> str:
        """
        Query and return just the answer.
        
        Args:
            question: User question
            
        Returns:
            Answer string
        """
        result = self.query(question)
        return result["answer"]

