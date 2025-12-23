"""Celery tasks for async document processing.

This module provides background tasks for processing large documents
with progress tracking and error handling.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from celery import shared_task, current_task

from app.workers.celery_app import celery_app
from app.agents.orchestrator import DocumentProcessingOrchestrator
from app.vectorstore.store import add_documents_sync
from app.ingestion.content_chunker import chunk_documents

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="process_document")
def process_document_task(
    self,
    pdf_path: str,
    document_id: str,
    user_id: Optional[str] = None,
):
    """
    Async task to process a PDF document.
    
    Args:
        pdf_path: Path to the PDF file
        document_id: Unique document identifier
        user_id: Optional user identifier
        
    Returns:
        Processing result dictionary
    """
    logger.info(f"Starting document processing: {document_id}")
    
    try:
        # Initialize orchestrator
        orchestrator = DocumentProcessingOrchestrator()
        
        def update_progress(current: int, total: int):
            """Update task progress."""
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': current,
                    'total': total,
                    'percent': int((current / total) * 100),
                    'status': f'Processing page {current} of {total}',
                }
            )
        
        # Process document
        documents = orchestrator.process_document_parallel(
            pdf_path,
            on_progress=update_progress,
        )
        
        # Chunk documents
        self.update_state(
            state='PROGRESS',
            meta={
                'status': 'Chunking documents...',
                'percent': 90,
            }
        )
        chunks = chunk_documents(documents)
        
        # Add source metadata
        for chunk in chunks:
            chunk.metadata["source"] = document_id
            if user_id:
                chunk.metadata["user_id"] = user_id
        
        # Store in vector database
        self.update_state(
            state='PROGRESS',
            meta={
                'status': 'Storing in vector database...',
                'percent': 95,
            }
        )
        add_documents_sync(chunks)
        
        result = {
            'status': 'completed',
            'document_id': document_id,
            'total_pages': len(documents),
            'total_chunks': len(chunks),
            'errors': [],
        }
        
        logger.info(f"Completed processing: {document_id}, {len(chunks)} chunks")
        return result
        
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        self.update_state(
            state='FAILURE',
            meta={
                'status': 'failed',
                'error': str(e),
            }
        )
        raise


@celery_app.task(bind=True, name="process_document_simple")
def process_document_simple_task(
    self,
    pdf_path: str,
    document_id: str,
):
    """
    Simpler document processing without full orchestration.
    
    Uses parallel processor directly without agent coordination.
    """
    from app.ingestion.parallel_processor import process_pdf_parallel
    
    logger.info(f"Starting simple processing: {document_id}")
    
    try:
        def update_progress(current: int, total: int):
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': current,
                    'total': total,
                    'percent': int((current / total) * 100),
                    'status': f'Processing page {current} of {total}',
                }
            )
        
        # Process
        chunks = process_pdf_parallel(pdf_path, on_progress=update_progress)
        
        # Add source
        for chunk in chunks:
            chunk.metadata["source"] = document_id
        
        # Store
        add_documents_sync(chunks)
        
        return {
            'status': 'completed',
            'document_id': document_id,
            'total_chunks': len(chunks),
        }
        
    except Exception as e:
        logger.error(f"Simple processing failed: {e}")
        raise


@celery_app.task(name="delete_document")
def delete_document_task(document_id: str):
    """
    Async task to delete a document.
    
    Args:
        document_id: Document identifier
        
    Returns:
        Deletion result
    """
    from app.vectorstore.document_manager import DocumentManager
    
    manager = DocumentManager()
    deleted = manager.delete_document(document_id)
    
    return {
        'status': 'deleted',
        'document_id': document_id,
        'deleted_chunks': deleted,
    }


def get_task_status(task_id: str) -> dict:
    """
    Get the status of a task.
    
    Args:
        task_id: Celery task ID
        
    Returns:
        Task status dictionary
    """
    from celery.result import AsyncResult
    
    task = AsyncResult(task_id, app=celery_app)
    
    if task.state == 'PENDING':
        return {
            'status': 'pending',
            'progress': 0,
            'message': 'Task is waiting to start',
        }
    
    elif task.state == 'PROGRESS':
        info = task.info or {}
        return {
            'status': 'processing',
            'progress': info.get('percent', 0),
            'current_page': info.get('current', 0),
            'total_pages': info.get('total', 0),
            'message': info.get('status', 'Processing...'),
        }
    
    elif task.state == 'SUCCESS':
        return {
            'status': 'completed',
            'progress': 100,
            'result': task.result,
        }
    
    elif task.state == 'FAILURE':
        return {
            'status': 'failed',
            'progress': 0,
            'error': str(task.info) if task.info else 'Unknown error',
        }
    
    else:
        return {
            'status': task.state.lower(),
            'progress': 0,
        }

