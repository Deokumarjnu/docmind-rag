"""API routes for DocMind RAG."""

import logging
import time
import uuid
from typing import Annotated, AsyncIterator, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    CacheStatsResponse,
    ContentType,
    ConversationListResponse,
    ConversationResponse,
    CreateConversationRequest,
    DeleteDocumentResponse,
    DocumentInfo,
    DocumentListResponse,
    GraphStatsResponse,
    MessageResponse,
    QueryRequest,
    QueryResponse,
    SourceDocument,
    UploadProgressResponse,
    UploadResponse,
)
from app.api.upload import router as upload_router
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Include upload routes
router.include_router(upload_router, tags=["upload"])


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the document store using RAG with caching and multi-turn context.
    
    Returns an AI-generated answer with source citations.
    
    Args:
        request: Query request with question and options
        
    Returns:
        Answer with sources and metadata
    """
    from app.chains.agentic_rag import create_agentic_rag
    
    start_time = time.time()
    cache_hit = False
    conversation_id = request.conversation_id
    graph_entities = None
    
    try:
        # Check semantic cache first
        if request.use_cache:
            try:
                from app.cache import get_semantic_cache
                cache = get_semantic_cache()
                cached = await cache.get(request.query, request.document_id)
                
                if cached:
                    cache_hit = True
                    logger.info(f"Cache hit for query: {request.query[:50]}...")
                    
                    # Store in conversation if provided
                    if conversation_id:
                        await _store_conversation_turn(
                            conversation_id,
                            request.query,
                            cached.get("answer", ""),
                        )
                    
                    return QueryResponse(
                        answer=cached.get("answer", ""),
                        sources=[],  # Sources not cached
                        query=request.query,
                        enhanced_query=cached.get("enhanced_query"),
                        is_valid=True,
                        cache_hit=True,
                        conversation_id=conversation_id,
                    )
            except Exception as e:
                logger.warning(f"Cache lookup failed: {e}")
        
        # Get knowledge graph context for multi-hop reasoning
        if request.use_knowledge_graph:
            try:
                from app.graph import KnowledgeGraphStore
                kg_store = KnowledgeGraphStore()
                graph_result = await kg_store.hybrid_search(
                    request.query,
                    entity_limit=5,
                    context_hops=2,
                )
                graph_entities = graph_result.get("seed_entities", [])
                logger.debug(f"KG found {len(graph_entities)} entities")
            except Exception as e:
                logger.debug(f"KG lookup skipped: {e}")
        
        # Get conversation context for multi-turn
        conversation_context = []
        if conversation_id:
            try:
                from app.memory import get_conversation_store
                store = get_conversation_store()
                conversation_context = await store.get_conversation_context(
                    conversation_id,
                    max_messages=6,
                )
            except Exception as e:
                logger.warning(f"Conversation context lookup failed: {e}")
        
        # Create agentic RAG with context
        rag = create_agentic_rag(
            k=request.top_k,
            use_reranker=True,
            conversation_context=conversation_context,
            graph_context=graph_entities,
        )
        
        # Process query
        result = await rag.ainvoke(request.query)
        
        # Format sources
        sources = []
        if request.include_sources:
            for src in result.get("sources", []):
                ct_value = src.get("content_type") or "text"
                try:
                    content_type = ContentType(ct_value)
                except ValueError:
                    content_type = ContentType.OTHER
                
                sources.append(SourceDocument(
                    content=src.get("content", ""),
                    page=src.get("page", 0),
                    source=src.get("source", "unknown"),
                    content_type=content_type,
                    relevance_score=src.get("relevance_score", 0.0),
                    metadata=src,
                ))
        
        answer = result.get("answer", "")
        
        # Cache the result
        if request.use_cache and not cache_hit:
            try:
                from app.cache import get_semantic_cache
                cache = get_semantic_cache()
                await cache.set(
                    request.query,
                    {
                        "answer": answer,
                        "enhanced_query": result.get("enhanced_query"),
                    },
                    request.document_id,
                )
            except Exception as e:
                logger.warning(f"Cache set failed: {e}")
        
        # Store in conversation
        if conversation_id:
            await _store_conversation_turn(conversation_id, request.query, answer)
        
        latency_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Query processed in {latency_ms}ms")
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            query=request.query,
            enhanced_query=result.get("enhanced_query"),
            is_valid=result.get("is_valid", True),
            retry_count=result.get("retry_count", 0),
            conversation_id=conversation_id,
            cache_hit=cache_hit,
            graph_entities=graph_entities,
        )
        
    except Exception as e:
        logger.exception(f"Query processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {e}",
        )


async def _store_conversation_turn(
    conversation_id: str,
    query: str,
    answer: str,
) -> None:
    """Store a query-answer pair in conversation history."""
    try:
        from app.memory import get_conversation_store, MessageRole
        store = get_conversation_store()
        
        # Add user message
        await store.add_message(conversation_id, MessageRole.USER, query)
        # Add assistant message
        await store.add_message(conversation_id, MessageRole.ASSISTANT, answer)
    except Exception as e:
        logger.warning(f"Failed to store conversation turn: {e}")


@router.post("/query/simple", response_model=QueryResponse)
async def query_documents_simple(request: QueryRequest):
    """
    Query using simple RAG (faster, no self-correction).
    
    Use this for quick queries where speed is more important than accuracy.
    """
    from app.chains.rag_chain import SimpleRAG
    
    try:
        rag = SimpleRAG(k=request.top_k)
        result = rag.query(request.query)
        
        sources = []
        if request.include_sources:
            for src in result.get("sources", []):
                # Handle None or invalid content_type
                ct_value = src.get("content_type") or "text"
                try:
                    content_type = ContentType(ct_value)
                except ValueError:
                    content_type = ContentType.OTHER
                
                sources.append(SourceDocument(
                    content=src.get("content", ""),
                    page=src.get("page", 0),
                    source=src.get("source", "unknown"),
                    content_type=content_type,
                    relevance_score=src.get("relevance_score", 0.0),
                    metadata=src,
                ))
        
        return QueryResponse(
            answer=result.get("answer", ""),
            sources=sources,
            query=request.query,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {e}",
        )


@router.post("/query/stream")
async def query_documents_stream(request: QueryRequest):
    """
    Query the document store with streaming response.
    
    Returns a Server-Sent Events stream for real-time answer generation.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    
    from app.vectorstore.store import similarity_search
    
    async def generate() -> AsyncIterator[str]:
        try:
            # Retrieve documents
            docs = similarity_search(request.query, k=request.top_k)
            
            if not docs:
                yield "data: No relevant documents found.\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # Format documents
            context = "\n\n".join([
                f"[Source: {d.metadata.get('source', 'unknown')}, "
                f"Page {d.metadata.get('page', 'N/A')}]\n{d.page_content}"
                for d in docs
            ])
            
            # Create streaming model
            model = ChatOpenAI(
                model=settings.llm_model,
                temperature=0,
                streaming=True,
                api_key=settings.openai_api_key,
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Answer the question based on the provided documents.
Cite sources using [Source: name, page X] notation."""),
                ("user", f"Documents:\n{context}\n\nQuestion: {request.query}"),
            ])
            
            # Stream response
            async for chunk in model.astream(prompt.format_messages()):
                if chunk.content:
                    yield f"data: {chunk.content}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: Error: {e}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """
    List all documents in the vector store.
    
    Returns:
        List of documents with metadata
    """
    from app.vectorstore.document_manager import DocumentManager
    
    try:
        manager = DocumentManager()
        documents = manager.list_documents()
        
        doc_infos = []
        for doc in documents:
            # Handle content types - map unknown types to appropriate enum values
            content_types = []
            for ct in doc.get("content_types", ["text"]):
                try:
                    content_types.append(ContentType(ct))
                except ValueError:
                    # Map unknown content types to closest match
                    content_types.append(ContentType.OTHER)
            doc_infos.append(DocumentInfo(
                document_id=doc["document_id"],
                filename=doc["filename"],
                total_chunks=doc["total_chunks"],
                pages=doc["pages"],
                created_at=doc.get("created_at"),
                content_types=content_types,
                language=doc.get("language"),
            ))
        
        return DocumentListResponse(
            documents=doc_infos,
            total=len(doc_infos),
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {e}",
        )


@router.get("/documents/{document_id}")
async def get_document(document_id: str):
    """
    Get information about a specific document.
    
    Args:
        document_id: Document identifier
        
    Returns:
        Document information
    """
    from app.vectorstore.document_manager import DocumentManager
    
    try:
        manager = DocumentManager()
        info = manager.get_document_info(document_id)
        
        if not info:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}",
            )
        
        return info
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get document: {e}",
        )


@router.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
async def delete_document(document_id: str):
    """
    Delete a document and all its chunks from the vector store.
    
    Args:
        document_id: Document identifier
        
    Returns:
        Deletion confirmation
    """
    from app.vectorstore.document_manager import DocumentManager
    
    try:
        manager = DocumentManager()
        deleted_chunks = manager.delete_document(document_id)
        
        return DeleteDocumentResponse(
            document_id=document_id,
            deleted_chunks=deleted_chunks,
            status="deleted",
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {e}",
        )


@router.post("/search")
async def search_documents(
    query: str,
    k: int = 5,
    content_type: str = None,
):
    """
    Direct vector search without LLM generation.
    
    Useful for debugging or when only document retrieval is needed.
    
    Args:
        query: Search query
        k: Number of results
        content_type: Optional content type filter
        
    Returns:
        List of matching document chunks
    """
    from app.vectorstore.store import similarity_search_with_score
    from app.vectorstore.document_manager import DocumentManager
    
    try:
        if content_type:
            manager = DocumentManager()
            results = manager.search_by_content_type(content_type, query, k)
            return {
                "query": query,
                "results": [
                    {
                        "content": doc.page_content[:500],
                        "metadata": doc.metadata,
                    }
                    for doc in results
                ],
            }
        else:
            results = similarity_search_with_score(query, k=k)
            return {
                "query": query,
                "results": [
                    {
                        "content": doc.page_content[:500],
                        "score": float(score),
                        "metadata": doc.metadata,
                    }
                    for doc, score in results
                ],
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {e}",
        )


# ============================================================================
# Conversation Management Endpoints
# ============================================================================

@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(request: CreateConversationRequest):
    """
    Create a new conversation for multi-turn RAG.
    
    Args:
        request: Conversation creation request
        
    Returns:
        Created conversation details
    """
    from app.memory import get_conversation_store
    
    try:
        store = get_conversation_store()
        conversation = await store.create_conversation(
            user_id=request.user_id,
            document_id=request.document_id,
            title=request.title,
        )
        
        return ConversationResponse(
            id=conversation.id,
            title=conversation.title,
            document_id=conversation.document_id,
            message_count=0,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create conversation: {e}",
        )


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    user_id: Optional[str] = None,
    document_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """
    List conversations with summaries.
    
    Args:
        user_id: Filter by user
        document_id: Filter by document
        limit: Maximum results
        offset: Pagination offset
        
    Returns:
        List of conversation summaries
    """
    from app.memory import get_conversation_store
    
    try:
        store = get_conversation_store()
        summaries = await store.list_conversations(
            user_id=user_id,
            document_id=document_id,
            limit=limit,
            offset=offset,
        )
        
        return ConversationListResponse(
            conversations=[
                ConversationResponse(
                    id=s.id,
                    title=s.title,
                    document_id=s.document_id,
                    message_count=s.message_count,
                    created_at=s.created_at,
                    updated_at=s.updated_at,
                )
                for s in summaries
            ],
            total=len(summaries),
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list conversations: {e}",
        )


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """
    Get a conversation with all messages.
    
    Args:
        conversation_id: Conversation ID
        
    Returns:
        Conversation with messages
    """
    from app.memory import get_conversation_store
    
    try:
        store = get_conversation_store()
        conversation = await store.get_conversation(conversation_id)
        
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation not found: {conversation_id}",
            )
        
        return {
            "id": conversation.id,
            "title": conversation.title,
            "document_id": conversation.document_id,
            "messages": [
                {
                    "id": m.id,
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                    "sources": m.sources,
                }
                for m in conversation.messages
            ],
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get conversation: {e}",
        )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation and all its messages.
    
    Args:
        conversation_id: Conversation ID
        
    Returns:
        Deletion confirmation
    """
    from app.memory import get_conversation_store
    
    try:
        store = get_conversation_store()
        deleted = await store.delete_conversation(conversation_id)
        
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation not found: {conversation_id}",
            )
        
        return {"status": "deleted", "conversation_id": conversation_id}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete conversation: {e}",
        )


# ============================================================================
# Cache and Knowledge Graph Stats Endpoints
# ============================================================================

@router.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """
    Get semantic cache statistics.
    
    Returns:
        Cache hit/miss rates and totals
    """
    from app.cache import get_semantic_cache
    
    try:
        cache = get_semantic_cache()
        stats = await cache.get_stats()
        
        return CacheStatsResponse(
            hits=stats.get("hits", 0),
            misses=stats.get("misses", 0),
            sets=stats.get("sets", 0),
            hit_rate=stats.get("hit_rate", "0%"),
            total_requests=stats.get("total_requests", 0),
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get cache stats: {e}",
        )


@router.delete("/cache")
async def clear_cache(document_id: Optional[str] = None):
    """
    Clear semantic cache.
    
    Args:
        document_id: Optional, clear only for specific document
        
    Returns:
        Number of entries cleared
    """
    from app.cache import get_semantic_cache
    
    try:
        cache = get_semantic_cache()
        invalidated = await cache.invalidate(document_id)
        
        return {"status": "cleared", "entries_invalidated": invalidated}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {e}",
        )


@router.get("/graph/stats", response_model=GraphStatsResponse)
async def get_graph_stats():
    """
    Get knowledge graph statistics.
    
    Returns:
        Entity and relationship counts
    """
    from app.graph import KnowledgeGraphStore
    
    try:
        store = KnowledgeGraphStore()
        stats = await store.get_stats()
        
        return GraphStatsResponse(
            entity_count=stats.get("entity_count", 0),
            relationship_count=stats.get("relationship_count", 0),
            status="healthy",
        )
        
    except Exception as e:
        return GraphStatsResponse(
            entity_count=0,
            relationship_count=0,
            status=f"error: {e}",
        )


@router.post("/graph/search")
async def search_graph(
    query: str,
    hops: int = 2,
    entity_limit: int = 5,
):
    """
    Search the knowledge graph with multi-hop expansion.
    
    Args:
        query: Search query
        hops: Number of relationship hops
        entity_limit: Maximum seed entities
        
    Returns:
        Entities and expanded context
    """
    from app.graph import KnowledgeGraphStore
    
    try:
        store = KnowledgeGraphStore()
        result = await store.hybrid_search(
            query=query,
            entity_limit=entity_limit,
            context_hops=hops,
        )
        
        return {
            "query": query,
            "seed_entities": result.get("seed_entities", []),
            "expanded_context": result.get("expanded_context", []),
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Graph search failed: {e}",
        )
