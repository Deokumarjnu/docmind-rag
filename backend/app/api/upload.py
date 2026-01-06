"""Upload endpoints with async processing and progress tracking."""

import os
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.schemas import UploadProgressResponse, UploadResponse
from app.config import settings
from app.workers.tasks import (
    process_document_task,
    process_document_simple_task,
    get_task_status,
)

router = APIRouter()

# Directory for temporary uploads - use shared volume in Docker
# Falls back to /tmp for local development
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/app/uploads"))
if not UPLOAD_DIR.exists():
    UPLOAD_DIR = Path(tempfile.gettempdir()) / "docmind_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: Annotated[UploadFile, File(description="Document to upload")],
    use_deep_agents: bool = True,
):
    """
    Upload a document for processing and indexing.
    
    Supports PDF, DOCX, HTML, and TXT files.
    Large documents are processed asynchronously with progress tracking.
    
    Args:
        file: The document file to upload
        use_deep_agents: Whether to use deep agent orchestration (slower but smarter)
        
    Returns:
        Upload response with task ID for tracking
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    extension = file.filename.split(".")[-1].lower()
    if extension not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{extension}' not allowed. Allowed: {settings.allowed_extensions}",
        )

    # Read and check file size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File size {size_mb:.1f}MB exceeds maximum {settings.max_upload_size_mb}MB",
        )

    # Generate unique document ID
    document_id = f"{uuid.uuid4()}_{file.filename}"
    
    # Save file temporarily
    file_path = UPLOAD_DIR / document_id
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(contents)

    # Queue processing task
    try:
        if use_deep_agents:
            task = process_document_task.delay(
                str(file_path),
                document_id,
            )
        else:
            task = process_document_simple_task.delay(
                str(file_path),
                document_id,
            )
        
        return UploadResponse(
            task_id=task.id,
            filename=file.filename,
            status="processing",
            message="Document upload initiated. Processing will begin shortly.",
        )
        
    except Exception as e:
        # Clean up on failure
        if file_path.exists():
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to queue document processing: {e}",
        )


@router.get("/upload/status/{task_id}", response_model=UploadProgressResponse)
async def get_upload_status(task_id: str):
    """
    Get the processing status of an uploaded document.
    
    Poll this endpoint to track progress for large document uploads.
    
    Args:
        task_id: The task ID returned from the upload endpoint
        
    Returns:
        Current processing status with progress information
    """
    try:
        status = get_task_status(task_id)
        
        return UploadProgressResponse(
            task_id=task_id,
            status=status.get("status", "unknown"),
            progress=status.get("progress", 0),
            current_page=status.get("current_page"),
            total_pages=status.get("total_pages"),
            message=status.get("message"),
            result=status.get("result"),
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get task status: {e}",
        )


@router.post("/upload/sync", response_model=dict)
async def upload_document_sync(
    file: Annotated[UploadFile, File(description="Document to upload")],
):
    """
    Upload and process a document synchronously.
    
    This endpoint processes the document immediately and waits for completion.
    Use for smaller documents or when immediate results are needed.
    
    WARNING: May timeout for large documents. Use async upload for large files.
    """
    from app.ingestion.parallel_processor import process_pdf_parallel
    from app.vectorstore.store import add_documents_sync
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    extension = file.filename.split(".")[-1].lower()
    if extension != "pdf":
        raise HTTPException(
            status_code=400,
            detail="Sync upload currently only supports PDF files",
        )

    # Read file
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    
    if size_mb > 10:  # Limit sync processing to 10MB
        raise HTTPException(
            status_code=400,
            detail=f"File too large for sync processing ({size_mb:.1f}MB). Use async upload for files > 10MB.",
        )

    # Save temporarily
    document_id = f"{uuid.uuid4()}_{file.filename}"
    file_path = UPLOAD_DIR / document_id
    
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(contents)

    try:
        # Process synchronously
        chunks = process_pdf_parallel(str(file_path))
        
        # Add source metadata
        for chunk in chunks:
            chunk.metadata["source"] = document_id
        
        # Store in vector database
        add_documents_sync(chunks)
        
        # Clean up
        os.remove(file_path)
        
        return {
            "status": "completed",
            "document_id": document_id,
            "filename": file.filename,
            "total_chunks": len(chunks),
        }
        
    except Exception as e:
        # Clean up on failure
        if file_path.exists():
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {e}",
        )

