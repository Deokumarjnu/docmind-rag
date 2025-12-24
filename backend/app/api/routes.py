"""API routes for DocMind RAG."""

import uuid
from typing import Annotated, AsyncIterator

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    ContentType,
    DeleteDocumentResponse,
    DocumentInfo,
    DocumentListResponse,
    QueryRequest,
    QueryResponse,
    SourceDocument,
    UploadProgressResponse,
    UploadResponse,
)
from app.api.upload import router as upload_router
from app.config import settings

router = APIRouter()

# Include upload routes
router.include_router(upload_router, tags=["upload"])


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the document store using RAG.
    
    Returns an AI-generated answer with source citations.
    
    Args:
        request: Query request with question and options
        
    Returns:
        Answer with sources and metadata
    """
    from app.chains.agentic_rag import create_agentic_rag
    
    try:
        # Create agentic RAG
        rag = create_agentic_rag(
            k=request.top_k,
            use_reranker=True,
        )
        
        # Process query
        result = await rag.ainvoke(request.query)
        
        # Format sources
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
            enhanced_query=result.get("enhanced_query"),
            is_valid=result.get("is_valid", True),
            retry_count=result.get("retry_count", 0),
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {e}",
        )


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
