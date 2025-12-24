"""LangGraph Agentic RAG with self-correction.

This module implements a stateful RAG workflow using LangGraph,
with query enhancement, retrieval validation, and answer validation.
"""

import logging
from typing import Annotated, Any, Literal, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from app.config import settings
from app.retrievers.hybrid_retriever import HybridRetriever
from app.retrievers.reranker import CrossEncoderReranker
from app.retrievers.query_expander import QueryExpander
from app.vectorstore.store import similarity_search_with_score

logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    """State for the RAG agent workflow."""
    
    query: str
    enhanced_query: str
    documents: list[Document]
    answer: str
    sources: list[dict]
    is_relevant: bool
    is_valid: bool
    retry_count: int
    max_retries: int


def create_enhance_query_node():
    """Create the query enhancement node."""
    
    query_expander = QueryExpander()
    
    def enhance_query(state: RAGState) -> RAGState:
        """Enhance the query for better retrieval."""
        query = state["query"]
        
        # Expand acronyms and rewrite
        enhanced = query_expander.enhance_query(
            query,
            expand_acronyms=True,
            rewrite=True,
        )
        
        state["enhanced_query"] = enhanced["rewritten"]
        logger.debug(f"Enhanced query: {state['enhanced_query']}")
        
        return state
    
    return enhance_query


def create_retrieve_documents_node(
    retriever: Optional[HybridRetriever] = None,
    k: int = 5,
):
    """Create the document retrieval node."""
    
    def retrieve_documents(state: RAGState) -> RAGState:
        """Retrieve relevant documents."""
        query = state.get("enhanced_query") or state["query"]
        
        if retriever:
            documents = retriever.invoke(query)
        else:
            # Fallback to direct vector search
            results = similarity_search_with_score(query, k=k * 2)
            documents = [doc for doc, score in results]
        
        state["documents"] = documents[:k]
        logger.debug(f"Retrieved {len(state['documents'])} documents")
        
        return state
    
    return retrieve_documents


def create_validate_retrieval_node():
    """Create the retrieval validation node."""
    
    model = ChatOpenAI(
        model=settings.fast_llm_model,  # Use fast model for validation
        temperature=0,
        api_key=settings.openai_api_key,
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a relevance evaluator. Given a query and retrieved documents,
determine if the documents are relevant enough to answer the query.

Reply with ONLY 'relevant' or 'not_relevant'."""),
        ("user", """Query: {query}

Documents:
{documents}

Are these documents relevant to answer the query?"""),
    ])
    
    chain = prompt | model | StrOutputParser()
    
    def validate_retrieval(state: RAGState) -> RAGState:
        """Validate if retrieved documents are relevant."""
        documents = state.get("documents", [])
        
        if not documents:
            state["is_relevant"] = False
            return state
        
        # Format documents for evaluation
        docs_text = "\n\n".join([
            f"Document {i+1}:\n{doc.page_content[:500]}..."
            for i, doc in enumerate(documents[:3])
        ])
        
        try:
            result = chain.invoke({
                "query": state["query"],
                "documents": docs_text,
            })
            
            state["is_relevant"] = "relevant" in result.lower()
            
        except Exception as e:
            logger.warning(f"Retrieval validation failed: {e}")
            state["is_relevant"] = True  # Assume relevant on error
        
        return state
    
    return validate_retrieval


def create_generate_answer_node():
    """Create the answer generation node."""
    
    model = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on provided documents.

Guidelines:
- Answer based ONLY on the provided documents
- If the documents don't contain enough information, say so
- Cite sources using [Source: page X] notation
- Be concise but complete
- If there are multiple perspectives, present them"""),
        ("user", """Question: {query}

Documents:
{documents}

Answer the question based on the documents above."""),
    ])
    
    chain = prompt | model | StrOutputParser()
    
    def generate_answer(state: RAGState) -> RAGState:
        """Generate answer from retrieved documents."""
        documents = state.get("documents", [])
        
        if not documents:
            state["answer"] = "I couldn't find any relevant information to answer your question."
            state["sources"] = []
            return state
        
        # Format documents with source info
        docs_text = "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'unknown')}, "
            f"Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
            for doc in documents
        ])
        
        try:
            answer = chain.invoke({
                "query": state["query"],
                "documents": docs_text,
            })
            
            state["answer"] = answer
            
            # Extract sources
            state["sources"] = [
                {
                    "content": doc.page_content[:300],
                    "page": doc.metadata.get("page"),
                    "source": doc.metadata.get("source"),
                    "content_type": doc.metadata.get("content_type"),
                }
                for doc in documents
            ]
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            state["answer"] = f"Error generating answer: {e}"
            state["sources"] = []
        
        return state
    
    return generate_answer


def create_validate_answer_node():
    """Create the answer validation node."""
    
    model = ChatOpenAI(
        model=settings.fast_llm_model,  # Use fast model for validation
        temperature=0,
        api_key=settings.openai_api_key,
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an answer quality evaluator. 
Evaluate if the answer properly addresses the question and is grounded in the sources.

Check for:
1. Does the answer address the question?
2. Is the answer supported by the provided documents?
3. Are there obvious errors or hallucinations?

Reply with ONLY 'valid' or 'invalid'."""),
        ("user", """Question: {query}

Answer: {answer}

Is this a valid, well-grounded answer?"""),
    ])
    
    chain = prompt | model | StrOutputParser()
    
    def validate_answer(state: RAGState) -> RAGState:
        """Validate the generated answer."""
        try:
            result = chain.invoke({
                "query": state["query"],
                "answer": state.get("answer", ""),
            })
            
            state["is_valid"] = "valid" in result.lower()
            
        except Exception as e:
            logger.warning(f"Answer validation failed: {e}")
            state["is_valid"] = True  # Assume valid on error
        
        return state
    
    return validate_answer


def should_retry_retrieval(state: RAGState) -> Literal["generate", "enhance"]:
    """Determine if we should retry retrieval."""
    if state.get("is_relevant", True):
        return "generate"
    
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    
    if retry_count < max_retries:
        state["retry_count"] = retry_count + 1
        return "enhance"
    
    return "generate"


def should_retry_generation(state: RAGState) -> Literal[END, "retrieve"]:
    """Determine if we should retry generation."""
    if state.get("is_valid", True):
        return END
    
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    
    if retry_count < max_retries:
        state["retry_count"] = retry_count + 1
        return "retrieve"
    
    return END


def create_rag_agent(
    retriever: Optional[HybridRetriever] = None,
    k: int = 5,
    max_retries: int = 2,
) -> StateGraph:
    """
    Create a LangGraph RAG agent with self-correction.
    
    Args:
        retriever: Optional custom retriever
        k: Number of documents to retrieve
        max_retries: Maximum retry attempts
        
    Returns:
        Compiled LangGraph workflow
    """
    # Create nodes
    enhance_query = create_enhance_query_node()
    retrieve_documents = create_retrieve_documents_node(retriever, k)
    validate_retrieval = create_validate_retrieval_node()
    generate_answer = create_generate_answer_node()
    validate_answer = create_validate_answer_node()
    
    # Build the graph
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("enhance", enhance_query)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("validate_retrieval", validate_retrieval)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("validate_answer", validate_answer)
    
    # Set entry point
    workflow.set_entry_point("enhance")
    
    # Add edges
    workflow.add_edge("enhance", "retrieve")
    workflow.add_edge("retrieve", "validate_retrieval")
    
    # Conditional edge: retry retrieval if not relevant
    workflow.add_conditional_edges(
        "validate_retrieval",
        should_retry_retrieval,
        {
            "generate": "generate",
            "enhance": "enhance",
        }
    )
    
    workflow.add_edge("generate", "validate_answer")
    
    # Conditional edge: retry generation if not valid
    workflow.add_conditional_edges(
        "validate_answer",
        should_retry_generation,
        {
            END: END,
            "retrieve": "retrieve",
        }
    )
    
    # Compile
    return workflow.compile()


class AgenticRAG:
    """
    Wrapper class for the agentic RAG system.
    """

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        k: int = 5,
        max_retries: int = 2,
        use_reranker: bool = True,
    ):
        """
        Initialize agentic RAG.
        
        Args:
            retriever: Custom retriever
            k: Number of documents to retrieve
            max_retries: Maximum retry attempts
            use_reranker: Whether to use cross-encoder reranking
        """
        self.retriever = retriever
        self.k = k
        self.max_retries = max_retries
        
        self.reranker = None
        if use_reranker:
            try:
                self.reranker = CrossEncoderReranker()
            except Exception as e:
                logger.warning(f"Reranker initialization failed: {e}")
        
        self.agent = create_rag_agent(retriever, k, max_retries)

    def invoke(self, query: str) -> dict:
        """
        Process a query through the RAG pipeline.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with answer and sources
        """
        initial_state: RAGState = {
            "query": query,
            "enhanced_query": "",
            "documents": [],
            "answer": "",
            "sources": [],
            "is_relevant": False,
            "is_valid": False,
            "retry_count": 0,
            "max_retries": self.max_retries,
        }
        
        # Run the agent
        result = self.agent.invoke(initial_state)
        
        # Apply reranking if enabled
        if self.reranker and result.get("documents"):
            reranked = self.reranker.rerank(
                result["query"],
                result["documents"],
                top_k=self.k,
            )
            result["documents"] = [doc for doc, score in reranked]
        
        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "query": result.get("query", query),
            "enhanced_query": result.get("enhanced_query", ""),
            "is_valid": result.get("is_valid", True),
            "retry_count": result.get("retry_count", 0),
        }

    async def ainvoke(self, query: str) -> dict:
        """Async version of invoke."""
        # LangGraph supports async invocation
        initial_state: RAGState = {
            "query": query,
            "enhanced_query": "",
            "documents": [],
            "answer": "",
            "sources": [],
            "is_relevant": False,
            "is_valid": False,
            "retry_count": 0,
            "max_retries": self.max_retries,
        }
        
        result = await self.agent.ainvoke(initial_state)
        
        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "query": result.get("query", query),
            "enhanced_query": result.get("enhanced_query", ""),
            "is_valid": result.get("is_valid", True),
            "retry_count": result.get("retry_count", 0),
        }


def create_agentic_rag(
    retriever: Optional[HybridRetriever] = None,
    k: int = 5,
    max_retries: int = 2,
    use_reranker: bool = True,
) -> AgenticRAG:
    """
    Create an agentic RAG instance.
    
    Args:
        retriever: Custom retriever
        k: Number of documents
        max_retries: Maximum retries
        use_reranker: Whether to use reranking
        
    Returns:
        AgenticRAG instance
    """
    return AgenticRAG(
        retriever=retriever,
        k=k,
        max_retries=max_retries,
        use_reranker=use_reranker,
    )

