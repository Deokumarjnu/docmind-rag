"""Multi-page table detection and merging.

This module handles tables that span multiple pages, detecting
continuation patterns and merging them into complete units.
"""

import logging
import re
from typing import Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def detect_table_continuation(pages: list[Document]) -> list[Document]:
    """
    Detect and merge tables that span multiple pages.
    
    Args:
        pages: List of page documents to process
        
    Returns:
        List of documents with multi-page tables merged
    """
    merged_tables = []
    current_table: Optional[Document] = None
    non_table_docs = []
    
    for i, page in enumerate(pages):
        # Check if this page contains table content
        if not _is_table_document(page):
            # Finalize any pending table
            if current_table:
                merged_tables.append(current_table)
                current_table = None
            non_table_docs.append(page)
            continue
        
        # This is a table page
        if current_table is None:
            # Start new table
            current_table = page
        elif _is_table_continuation(current_table, page):
            # Merge with current table
            current_table = merge_tables(current_table, page)
            logger.debug(f"Merged table from page {page.metadata.get('page')} with previous")
        else:
            # Different table - finalize current and start new
            merged_tables.append(current_table)
            current_table = page
    
    # Finalize any remaining table
    if current_table:
        merged_tables.append(current_table)
    
    # Combine all documents in order
    result = []
    table_idx = 0
    non_table_idx = 0
    
    for page in pages:
        if _is_table_document(page):
            if table_idx < len(merged_tables):
                # Add merged table (skip if already added)
                merged = merged_tables[table_idx]
                if merged not in result:
                    result.append(merged)
                # Skip pages that were merged
                if page.metadata.get("page") in merged.metadata.get("spans_pages", [page.metadata.get("page")]):
                    if page.metadata.get("page") == merged.metadata.get("spans_pages", [page.metadata.get("page")])[-1]:
                        table_idx += 1
        else:
            if non_table_idx < len(non_table_docs):
                result.append(non_table_docs[non_table_idx])
                non_table_idx += 1
    
    return merged_tables + non_table_docs


def _is_table_document(doc: Document) -> bool:
    """Check if document contains table content."""
    content_type = doc.metadata.get("content_type", "")
    element_type = doc.metadata.get("category", "")
    
    if content_type == "table" or element_type == "Table":
        return True
    
    # Heuristic check for table patterns
    text = doc.page_content
    if _has_table_indicators(text):
        return True
    
    return False


def _has_table_indicators(text: str) -> bool:
    """Check for table indicators in text."""
    # Markdown table pattern
    if re.search(r'\|.*\|.*\|', text):
        return True
    
    # Tab-separated values
    lines = text.split('\n')
    tab_lines = sum(1 for line in lines if line.count('\t') >= 2)
    if tab_lines >= 3:
        return True
    
    # Multiple numbers aligned
    number_patterns = len(re.findall(r'\d+\s{2,}\d+', text))
    if number_patterns >= 3:
        return True
    
    return False


def _is_table_continuation(prev_table: Document, curr_table: Document) -> bool:
    """
    Check if current table is a continuation of previous.
    
    Heuristics:
    1. Column headers match
    2. Table starts at top of page
    3. "(continued)" marker present
    """
    if prev_table is None:
        return False
    
    # Check for continuation markers
    curr_text = curr_table.page_content.lower()
    if any(marker in curr_text for marker in ["continued", "cont'd", "(cont.)", "...continued"]):
        return True
    
    # Check if column headers match
    prev_headers = extract_headers(prev_table)
    curr_headers = extract_headers(curr_table)
    
    if prev_headers and curr_headers and prev_headers == curr_headers:
        # Check if table starts near top of page
        y_position = curr_table.metadata.get("y_position", 0)
        if y_position < 100:  # Near top of page
            return True
    
    # Check for consecutive pages
    prev_page = prev_table.metadata.get("page", -1)
    curr_page = curr_table.metadata.get("page", -1)
    
    if curr_page == prev_page + 1:
        # Same column count suggests continuation
        prev_cols = _estimate_column_count(prev_table.page_content)
        curr_cols = _estimate_column_count(curr_table.page_content)
        
        if prev_cols > 0 and prev_cols == curr_cols:
            return True
    
    return False


def extract_headers(table: Document) -> list[str]:
    """
    Extract column headers from a table document.
    
    Args:
        table: Table document
        
    Returns:
        List of header strings
    """
    text = table.page_content.strip()
    lines = text.split('\n')
    
    if not lines:
        return []
    
    # First non-empty line is often the header
    header_line = lines[0].strip()
    
    # Try to split by common separators
    if '|' in header_line:
        headers = [h.strip() for h in header_line.split('|') if h.strip()]
    elif '\t' in header_line:
        headers = [h.strip() for h in header_line.split('\t') if h.strip()]
    else:
        # Try splitting by multiple spaces
        headers = [h.strip() for h in re.split(r'\s{2,}', header_line) if h.strip()]
    
    return headers


def _estimate_column_count(text: str) -> int:
    """Estimate number of columns in table text."""
    lines = text.strip().split('\n')
    if not lines:
        return 0
    
    # Use first few lines to estimate
    col_counts = []
    for line in lines[:5]:
        if '|' in line:
            cols = len([c for c in line.split('|') if c.strip()])
        elif '\t' in line:
            cols = len([c for c in line.split('\t') if c.strip()])
        else:
            cols = len(re.split(r'\s{2,}', line))
        col_counts.append(cols)
    
    if col_counts:
        return max(set(col_counts), key=col_counts.count)
    return 0


def merge_tables(table1: Document, table2: Document) -> Document:
    """
    Merge two table fragments into one.
    
    Args:
        table1: First table (beginning)
        table2: Second table (continuation)
        
    Returns:
        Merged table document
    """
    # Get content without duplicate headers
    content2 = _remove_headers(table2.page_content)
    merged_content = table1.page_content.rstrip() + "\n" + content2.lstrip()
    
    # Track pages that were merged
    page1 = table1.metadata.get("page", 0)
    page2 = table2.metadata.get("page", 0)
    
    existing_spans = table1.metadata.get("spans_pages", [page1])
    if isinstance(existing_spans, list):
        spans = existing_spans + [page2]
    else:
        spans = [page1, page2]
    
    return Document(
        page_content=merged_content,
        metadata={
            **table1.metadata,
            "spans_pages": spans,
            "content_type": "table",
            "merged_table": True,
            "page": page1,  # Keep first page as reference
        }
    )


def _remove_headers(text: str) -> str:
    """Remove header row from table text."""
    lines = text.strip().split('\n')
    
    if len(lines) <= 1:
        return text
    
    # Check if first line looks like a header
    first_line = lines[0]
    second_line = lines[1] if len(lines) > 1 else ""
    
    # If second line is a separator (like |---|---|), skip both
    if re.match(r'^[\s|\-:]+$', second_line):
        return '\n'.join(lines[2:])
    
    # If first line has similar structure to data rows, keep it
    # Otherwise, assume it's a header and skip
    if _looks_like_header(first_line, lines[1:]):
        return '\n'.join(lines[1:])
    
    return text


def _looks_like_header(first_line: str, other_lines: list[str]) -> bool:
    """Check if first line appears to be a header row."""
    if not other_lines:
        return False
    
    # Headers often contain text while data rows have numbers
    first_numbers = len(re.findall(r'\d+', first_line))
    first_letters = len(re.findall(r'[a-zA-Z]+', first_line))
    
    other_numbers = sum(len(re.findall(r'\d+', line)) for line in other_lines[:3])
    
    # If first line has more letters than numbers and others have more numbers
    if first_letters > first_numbers and other_numbers > first_numbers * 2:
        return True
    
    return False


class TableMerger:
    """
    Table merger for processing documents with multi-page tables.
    """

    def __init__(self, max_pages_to_merge: int = 10):
        """
        Initialize table merger.
        
        Args:
            max_pages_to_merge: Maximum pages to merge into single table
        """
        self.max_pages_to_merge = max_pages_to_merge

    def process(self, documents: list[Document]) -> list[Document]:
        """
        Process documents and merge multi-page tables.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List with multi-page tables merged
        """
        return detect_table_continuation(documents)

