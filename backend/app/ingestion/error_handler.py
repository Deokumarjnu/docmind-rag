"""Error handling for document ingestion with graceful degradation.

This module provides robust error handling for page extraction,
with retries and fallback strategies for problematic pages.
"""

import logging
from pathlib import Path
from typing import Callable, Optional

from langchain_core.documents import Document

from app.ingestion.page_classifier import PageType

logger = logging.getLogger(__name__)


class PageExtractionError(Exception):
    """Custom exception for page extraction failures."""

    def __init__(self, page_num: int, message: str, original_error: Optional[Exception] = None):
        self.page_num = page_num
        self.message = message
        self.original_error = original_error
        super().__init__(f"Page {page_num}: {message}")


def safe_extract_page(
    extract_fn: Callable,
    page: Document,
    pdf_path: Path,
    max_retries: int = 2,
) -> list[Document]:
    """
    Extract page content with error handling and retries.
    
    Args:
        extract_fn: Extraction function to call
        page: Document page to extract
        pdf_path: Path to source PDF
        max_retries: Maximum retry attempts
        
    Returns:
        List of extracted documents
    """
    page_num = page.metadata.get("page", "unknown")
    
    for attempt in range(max_retries + 1):
        try:
            return extract_fn(page, pdf_path)
            
        except Exception as e:
            logger.warning(
                f"Extraction failed for page {page_num}, "
                f"attempt {attempt + 1}/{max_retries + 1}: {e}"
            )
            
            if attempt < max_retries:
                # Try fallback extraction on retry
                try:
                    return fallback_extraction(page, pdf_path)
                except Exception:
                    continue
    
    # All attempts failed - create placeholder
    logger.error(f"All extraction attempts failed for page {page_num}")
    return [create_error_placeholder(page)]


def fallback_extraction(page: Document, pdf_path: Path) -> list[Document]:
    """
    Fallback extraction using simple PyMuPDF.
    
    Args:
        page: Document page
        pdf_path: Path to PDF
        
    Returns:
        List of extracted documents
    """
    try:
        import fitz
        
        page_num = page.metadata.get("page", 0)
        doc = fitz.open(str(pdf_path))
        pdf_page = doc[page_num]
        text = pdf_page.get_text()
        doc.close()
        
        return [Document(
            page_content=text,
            metadata={
                **page.metadata,
                "extraction_method": "pymupdf_fallback",
            }
        )]
        
    except Exception as e:
        logger.error(f"Fallback extraction failed: {e}")
        raise


def create_error_placeholder(page: Document) -> Document:
    """
    Create placeholder document for failed extraction.
    
    Args:
        page: Original page document
        
    Returns:
        Placeholder document with error metadata
    """
    return Document(
        page_content="[Content extraction failed for this page. "
                    "Page may contain complex formatting or be corrupted.]",
        metadata={
            **page.metadata,
            "extraction_error": True,
            "extraction_status": "failed",
            "content_type": PageType.TEXT.value,
        }
    )


def validate_extracted_content(doc: Document) -> Document:
    """
    Validate extraction quality and add warnings.
    
    Args:
        doc: Extracted document
        
    Returns:
        Document with quality metadata
    """
    content = doc.page_content
    
    # Check for garbage characters (common in failed OCR)
    garbage_ratio = sum(1 for c in content if ord(c) > 127) / max(len(content), 1)
    if garbage_ratio > 0.3:
        doc.metadata["quality_warning"] = "high_garbage_ratio"
        logger.warning(f"High garbage ratio ({garbage_ratio:.2f}) in extracted content")
    
    # Check for extremely short content (may indicate failure)
    if len(content.strip()) < 10:
        doc.metadata["quality_warning"] = "very_short_content"
    
    # Check for repetitive characters (OCR failure pattern)
    if has_repetitive_pattern(content):
        doc.metadata["quality_warning"] = "repetitive_pattern"
    
    return doc


def has_repetitive_pattern(text: str, threshold: float = 0.5) -> bool:
    """
    Detect repetitive character patterns indicating OCR failure.
    
    Args:
        text: Text to analyze
        threshold: Frequency threshold for repetition
        
    Returns:
        True if repetitive pattern detected
    """
    if len(text) < 20:
        return False
    
    from collections import Counter
    char_counts = Counter(text.lower())
    if not char_counts:
        return False
    
    most_common_ratio = char_counts.most_common(1)[0][1] / len(text)
    return most_common_ratio > threshold


def validate_documents(documents: list[Document]) -> list[Document]:
    """
    Validate all documents and add quality warnings.
    
    Args:
        documents: List of documents to validate
        
    Returns:
        List of validated documents
    """
    validated = []
    for doc in documents:
        validated_doc = validate_extracted_content(doc)
        validated.append(validated_doc)
    
    # Log summary
    warnings = sum(1 for d in validated if "quality_warning" in d.metadata)
    if warnings:
        logger.warning(f"{warnings}/{len(validated)} documents have quality warnings")
    
    return validated


class ExtractionErrorCollector:
    """
    Collector for tracking extraction errors during batch processing.
    """

    def __init__(self):
        self.errors: list[dict] = []
        self.warnings: list[dict] = []

    def add_error(self, page_num: int, error: str, error_type: str = "extraction"):
        """Add an error record."""
        self.errors.append({
            "page": page_num,
            "error": error,
            "type": error_type,
        })

    def add_warning(self, page_num: int, warning: str, warning_type: str = "quality"):
        """Add a warning record."""
        self.warnings.append({
            "page": page_num,
            "warning": warning,
            "type": warning_type,
        })

    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        """Check if any warnings occurred."""
        return len(self.warnings) > 0

    def get_summary(self) -> dict:
        """Get summary of errors and warnings."""
        return {
            "total_errors": len(self.errors),
            "total_warnings": len(self.warnings),
            "error_pages": [e["page"] for e in self.errors],
            "warning_pages": [w["page"] for w in self.warnings],
        }

