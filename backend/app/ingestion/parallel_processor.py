"""Parallel processing for large PDF documents.

This module enables efficient processing of large documents
using ThreadPoolExecutor for concurrent page processing.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

from langchain_core.documents import Document

from app.config import settings
from app.ingestion.page_classifier import classify_page, PageType
from app.ingestion.adaptive_extractor import AdaptiveExtractor
from app.ingestion.content_chunker import chunk_document
from app.ingestion.error_handler import (
    safe_extract_page,
    validate_documents,
    ExtractionErrorCollector,
)

logger = logging.getLogger(__name__)


def process_page(
    page: Document,
    pdf_path: Path,
    extractor: AdaptiveExtractor,
) -> list[Document]:
    """
    Process a single page: classify, extract, and chunk.
    
    Args:
        page: Page document
        pdf_path: Path to PDF
        extractor: Adaptive extractor instance
        
    Returns:
        List of processed chunks
    """
    # Classify page
    page_type = classify_page(page)
    page.metadata["page_type"] = page_type.value
    
    # Extract with optimal strategy
    extracted = extractor.extract_page(page, pdf_path, page_type)
    
    # Chunk the extracted content
    chunks = []
    for doc in extracted:
        chunks.extend(chunk_document(doc))
    
    return chunks


def process_pdf_parallel(
    pdf_path: str | Path,
    max_workers: Optional[int] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> list[Document]:
    """
    Process a PDF with parallel page processing.
    
    Args:
        pdf_path: Path to PDF file
        max_workers: Maximum parallel workers (defaults to config)
        on_progress: Progress callback (current, total)
        
    Returns:
        List of processed document chunks
    """
    pdf_path = Path(pdf_path)
    max_workers = max_workers or settings.max_parallel_workers
    
    extractor = AdaptiveExtractor()
    
    # Load all pages
    pages = extractor.load_pdf_pages(pdf_path)
    total_pages = len(pages)
    
    logger.info(f"Processing {total_pages} pages from {pdf_path.name}")
    
    all_chunks = []
    error_collector = ExtractionErrorCollector()
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all pages
        future_to_page = {
            executor.submit(process_page, page, pdf_path, extractor): page
            for page in pages
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_page):
            page = future_to_page[future]
            page_num = page.metadata.get("page", 0)
            
            try:
                chunks = future.result()
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"Failed to process page {page_num}: {e}")
                error_collector.add_error(page_num, str(e))
                
                # Create error placeholder
                all_chunks.append(Document(
                    page_content=f"[Page {page_num} processing failed]",
                    metadata={
                        "source": str(pdf_path),
                        "page": page_num,
                        "extraction_error": True,
                    }
                ))
            
            completed += 1
            if on_progress:
                on_progress(completed, total_pages)
    
    # Validate all documents
    validated_chunks = validate_documents(all_chunks)
    
    # Log summary
    logger.info(
        f"Processed {total_pages} pages into {len(validated_chunks)} chunks. "
        f"Errors: {len(error_collector.errors)}"
    )
    
    return validated_chunks


def process_pages_batch(
    pages: list[Document],
    pdf_path: str | Path,
    batch_size: int = 10,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> list[Document]:
    """
    Process pages in batches for memory efficiency.
    
    Args:
        pages: List of pages to process
        pdf_path: Path to PDF
        batch_size: Pages per batch
        on_progress: Progress callback
        
    Returns:
        List of processed chunks
    """
    pdf_path = Path(pdf_path)
    extractor = AdaptiveExtractor()
    
    all_chunks = []
    total_pages = len(pages)
    
    for i in range(0, total_pages, batch_size):
        batch = pages[i:i + batch_size]
        
        for page in batch:
            try:
                chunks = process_page(page, pdf_path, extractor)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to process page: {e}")
        
        if on_progress:
            on_progress(min(i + batch_size, total_pages), total_pages)
    
    return all_chunks


class ParallelProcessor:
    """
    Parallel document processor for large PDFs.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        batch_size: int = 10,
    ):
        """
        Initialize parallel processor.
        
        Args:
            max_workers: Maximum parallel workers
            batch_size: Pages per batch (for batch mode)
        """
        self.max_workers = max_workers or settings.max_parallel_workers
        self.batch_size = batch_size
        self.extractor = AdaptiveExtractor()

    def process(
        self,
        pdf_path: str | Path,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> list[Document]:
        """
        Process a PDF with parallel execution.
        
        Args:
            pdf_path: Path to PDF
            on_progress: Progress callback
            
        Returns:
            List of processed chunks
        """
        return process_pdf_parallel(
            pdf_path,
            max_workers=self.max_workers,
            on_progress=on_progress,
        )

    def process_sync(
        self,
        pdf_path: str | Path,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> list[Document]:
        """
        Process a PDF synchronously (for debugging).
        
        Args:
            pdf_path: Path to PDF
            on_progress: Progress callback
            
        Returns:
            List of processed chunks
        """
        pdf_path = Path(pdf_path)
        pages = self.extractor.load_pdf_pages(pdf_path)
        
        all_chunks = []
        for i, page in enumerate(pages):
            try:
                chunks = process_page(page, pdf_path, self.extractor)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to process page {i}: {e}")
            
            if on_progress:
                on_progress(i + 1, len(pages))
        
        return all_chunks

