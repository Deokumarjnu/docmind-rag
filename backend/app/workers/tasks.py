"""Celery tasks for async document processing.

This module provides background tasks for processing large documents
with progress tracking and error handling.
"""

import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import Optional

# Suppress matplotlib font warnings (common on macOS)
warnings.filterwarnings("ignore", message=".*font.*", category=UserWarning)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

from celery import shared_task, current_task

from app.workers.celery_app import celery_app
from app.agents.orchestrator import DocumentProcessingOrchestrator
from app.vectorstore.store import add_documents_sync
from app.ingestion.content_chunker import chunk_documents

logger = logging.getLogger(__name__)


def extract_knowledge_graph_sync(chunks, document_id: str) -> dict:
    """
    Extract knowledge graph from document chunks synchronously.
    
    Args:
        chunks: Document chunks to process
        document_id: Document identifier
        
    Returns:
        Dictionary with extraction statistics
    """
    import asyncio
    
    async def _extract():
        from app.graph import KnowledgeGraphExtractor, KnowledgeGraphStore
        
        extractor = KnowledgeGraphExtractor()
        store = KnowledgeGraphStore()
        await store.initialize()
        
        total_entities = 0
        total_relations = 0
        
        # Process a sample of chunks to avoid API overload
        sample_chunks = chunks[:20] if len(chunks) > 20 else chunks
        
        for chunk in sample_chunks:
            try:
                result = extractor.extract_from_document(chunk)
                stats = await store.add_extraction_results(result, document_id)
                total_entities += stats.get("entities_added", 0)
                total_relations += stats.get("relationships_added", 0)
            except Exception as e:
                logger.debug(f"Chunk extraction failed: {e}")
                continue
        
        return {
            "total_entities": total_entities,
            "total_relations": total_relations,
        }
    
    # Run async function in sync context
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(_extract())


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
                'percent': 90,
            }
        )
        add_documents_sync(chunks)
        
        # Extract knowledge graph (async in background)
        kg_entities = 0
        kg_relations = 0
        try:
            self.update_state(
                state='PROGRESS',
                meta={
                    'status': 'Extracting knowledge graph...',
                    'percent': 95,
                }
            )
            kg_result = extract_knowledge_graph_sync(chunks[:50], document_id)
            kg_entities = kg_result.get("total_entities", 0)
            kg_relations = kg_result.get("total_relations", 0)
            logger.info(f"KG extracted: {kg_entities} entities, {kg_relations} relations")
        except Exception as e:
            logger.warning(f"KG extraction failed (non-critical): {e}")
        
        result = {
            'status': 'completed',
            'document_id': document_id,
            'total_pages': len(documents),
            'total_chunks': len(chunks),
            'kg_entities': kg_entities,
            'kg_relations': kg_relations,
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
    from app.vectorstore.store import get_vector_store
    
    logger.info(f"Starting simple processing: {document_id}")
    
    try:
        def update_progress(current: int, total: int):
            # Page processing is 0-50% of total progress
            percent = int((current / total) * 50)
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': current,
                    'total': total,
                    'percent': percent,
                    'status': f'Processing page {current} of {total}',
                }
            )
        
        # Process pages (0-50%)
        chunks = process_pdf_parallel(pdf_path, on_progress=update_progress)
        
        # Add source
        for chunk in chunks:
            chunk.metadata["source"] = document_id
        
        # Store with progress updates (50-100%)
        self.update_state(
            state='PROGRESS',
            meta={
                'percent': 55,
                'status': f'Embedding {len(chunks)} chunks...',
            }
        )
        
        # Get vector store and add in batches with progress
        vector_store = get_vector_store()
        batch_size = 50
        total_chunks = len(chunks)
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            vector_store.add_documents(batch)
            
            # Update progress (50-100%)
            batch_progress = min(i + batch_size, total_chunks)
            percent = 50 + int((batch_progress / total_chunks) * 50)
            self.update_state(
                state='PROGRESS',
                meta={
                    'percent': percent,
                    'status': f'Stored {batch_progress}/{total_chunks} chunks',
                }
            )
        
        logger.info(f"Added {total_chunks} documents to vector store")
        
        return {
            'status': 'completed',
            'document_id': document_id,
            'total_chunks': total_chunks,
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

