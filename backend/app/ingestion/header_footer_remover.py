"""Header and footer detection and removal.

This module detects repetitive headers and footers across pages
and removes them to clean up document content.
"""

import logging
import re
from collections import Counter
from typing import Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def detect_headers_footers(
    pages: list[Document],
    threshold: float = 0.7,
) -> tuple[Optional[str], Optional[str]]:
    """
    Detect repetitive headers and footers across pages.
    
    Args:
        pages: List of page documents
        threshold: Frequency threshold (0-1) for pattern detection
        
    Returns:
        Tuple of (header_pattern, footer_pattern)
    """
    if len(pages) < 3:
        return None, None
    
    first_lines = []
    last_lines = []
    
    for page in pages:
        lines = page.page_content.strip().split('\n')
        if lines:
            # Get first few lines for header detection
            first_lines.extend([normalize_line(line) for line in lines[:3] if line.strip()])
            # Get last few lines for footer detection
            last_lines.extend([normalize_line(line) for line in lines[-3:] if line.strip()])
    
    header_pattern = find_common_pattern(first_lines, threshold)
    footer_pattern = find_common_pattern(last_lines, threshold)
    
    return header_pattern, footer_pattern


def find_common_pattern(lines: list[str], threshold: float) -> Optional[str]:
    """
    Find patterns appearing above threshold frequency.
    
    Args:
        lines: List of normalized lines
        threshold: Frequency threshold
        
    Returns:
        Common pattern string or None
    """
    if not lines:
        return None
    
    counter = Counter(lines)
    total = len(lines)
    
    for pattern, count in counter.most_common(3):
        if pattern and count / total >= threshold:
            return pattern
    
    return None


def normalize_line(line: str) -> str:
    """
    Normalize line by removing variable parts.
    
    Removes page numbers, dates, and other variable content
    to enable pattern matching.
    """
    # Remove page numbers (various formats)
    line = re.sub(r'\b(page\s*)?\d+\s*(of\s*\d+)?\b', '', line, flags=re.IGNORECASE)
    line = re.sub(r'\bp\.\s*\d+\b', '', line, flags=re.IGNORECASE)
    line = re.sub(r'^\s*\d+\s*$', '', line)
    
    # Remove dates (various formats)
    line = re.sub(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', '', line)
    line = re.sub(r'\d{4}[/-]\d{1,2}[/-]\d{1,2}', '', line)
    line = re.sub(
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
        '',
        line,
        flags=re.IGNORECASE
    )
    
    # Remove timestamps
    line = re.sub(r'\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM)?', '', line, flags=re.IGNORECASE)
    
    # Normalize whitespace
    line = re.sub(r'\s+', ' ', line)
    
    return line.strip()


def remove_headers_footers(
    page_content: str,
    header_pattern: Optional[str],
    footer_pattern: Optional[str],
    max_lines_to_check: int = 5,
) -> str:
    """
    Remove detected headers and footers from page content.
    
    Args:
        page_content: Page text content
        header_pattern: Detected header pattern
        footer_pattern: Detected footer pattern
        max_lines_to_check: Maximum lines to check at start/end
        
    Returns:
        Cleaned page content
    """
    lines = page_content.split('\n')
    
    # Remove headers (from beginning)
    if header_pattern:
        lines_to_remove = 0
        for i, line in enumerate(lines[:max_lines_to_check]):
            normalized = normalize_line(line)
            if matches_pattern(normalized, header_pattern):
                lines_to_remove = i + 1
        
        if lines_to_remove > 0:
            lines = lines[lines_to_remove:]
    
    # Remove footers (from end)
    if footer_pattern:
        lines_to_remove = 0
        for i, line in enumerate(reversed(lines[-max_lines_to_check:])):
            normalized = normalize_line(line)
            if matches_pattern(normalized, footer_pattern):
                lines_to_remove = i + 1
        
        if lines_to_remove > 0:
            lines = lines[:-lines_to_remove]
    
    return '\n'.join(lines)


def matches_pattern(text: str, pattern: str) -> bool:
    """
    Check if text matches a pattern (with fuzzy matching).
    
    Args:
        text: Text to check
        pattern: Pattern to match
        
    Returns:
        True if text matches pattern
    """
    if not pattern or not text:
        return False
    
    # Exact match
    if text == pattern:
        return True
    
    # Check if pattern is substring
    if pattern in text or text in pattern:
        return True
    
    # Fuzzy match using word overlap
    pattern_words = set(pattern.lower().split())
    text_words = set(text.lower().split())
    
    if not pattern_words or not text_words:
        return False
    
    overlap = len(pattern_words & text_words)
    max_words = max(len(pattern_words), len(text_words))
    
    return overlap / max_words >= 0.8


def process_pages(pages: list[Document]) -> list[Document]:
    """
    Process pages to remove headers and footers.
    
    Args:
        pages: List of page documents
        
    Returns:
        List of cleaned page documents
    """
    # Detect patterns
    header_pattern, footer_pattern = detect_headers_footers(pages)
    
    if header_pattern:
        logger.info(f"Detected header pattern: {header_pattern[:50]}...")
    if footer_pattern:
        logger.info(f"Detected footer pattern: {footer_pattern[:50]}...")
    
    if not header_pattern and not footer_pattern:
        return pages
    
    # Remove from each page
    cleaned_pages = []
    for page in pages:
        cleaned_content = remove_headers_footers(
            page.page_content,
            header_pattern,
            footer_pattern,
        )
        
        cleaned_page = Document(
            page_content=cleaned_content,
            metadata={
                **page.metadata,
                "headers_footers_removed": True,
            }
        )
        cleaned_pages.append(cleaned_page)
    
    return cleaned_pages


class HeaderFooterRemover:
    """
    Header and footer remover for document processing.
    """

    def __init__(self, threshold: float = 0.7):
        """
        Initialize remover.
        
        Args:
            threshold: Frequency threshold for pattern detection
        """
        self.threshold = threshold
        self.header_pattern: Optional[str] = None
        self.footer_pattern: Optional[str] = None

    def detect(self, pages: list[Document]) -> None:
        """Detect header/footer patterns from pages."""
        self.header_pattern, self.footer_pattern = detect_headers_footers(
            pages, self.threshold
        )

    def remove(self, page: Document) -> Document:
        """Remove headers/footers from a single page."""
        cleaned_content = remove_headers_footers(
            page.page_content,
            self.header_pattern,
            self.footer_pattern,
        )
        
        return Document(
            page_content=cleaned_content,
            metadata={
                **page.metadata,
                "headers_footers_removed": True,
            }
        )

    def process(self, pages: list[Document]) -> list[Document]:
        """Detect patterns and remove from all pages."""
        self.detect(pages)
        return [self.remove(page) for page in pages]

