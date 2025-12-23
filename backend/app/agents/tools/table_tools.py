"""Tools for Table Specialist subagent.

These tools enable the table specialist to extract, detect,
and merge tables from documents.
"""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.tools import tool

from app.ingestion.table_merger import (
    extract_headers,
    merge_tables,
    _is_table_continuation,
    _has_table_indicators,
)

logger = logging.getLogger(__name__)


@tool
def extract_table_structure(content: str) -> dict:
    """
    Extract table structure from page content.
    
    Args:
        content: Page text content
        
    Returns:
        Dictionary with table structure information
    """
    doc = Document(page_content=content)
    headers = extract_headers(doc)
    
    has_table = _has_table_indicators(content)
    
    lines = content.strip().split('\n')
    
    return {
        "has_table": has_table,
        "headers": headers,
        "row_count": len(lines),
        "header_count": len(headers),
    }


@tool
def detect_table_continuation(prev_content: str, curr_content: str) -> bool:
    """
    Detect if current page's table continues from previous page.
    
    Args:
        prev_content: Previous page content
        curr_content: Current page content
        
    Returns:
        True if tables should be merged
    """
    prev_doc = Document(page_content=prev_content, metadata={"page": 0})
    curr_doc = Document(page_content=curr_content, metadata={"page": 1})
    
    return _is_table_continuation(prev_doc, curr_doc)


@tool
def merge_multipage_tables(table1_content: str, table2_content: str) -> str:
    """
    Merge two table fragments into one.
    
    Args:
        table1_content: First table content (beginning)
        table2_content: Second table content (continuation)
        
    Returns:
        Merged table content
    """
    doc1 = Document(page_content=table1_content, metadata={"page": 0})
    doc2 = Document(page_content=table2_content, metadata={"page": 1})
    
    merged = merge_tables(doc1, doc2)
    return merged.page_content


@tool
def format_table_to_markdown(content: str) -> str:
    """
    Format table content to clean markdown.
    
    Args:
        content: Raw table content
        
    Returns:
        Markdown-formatted table
    """
    lines = content.strip().split('\n')
    if not lines:
        return content
    
    # Try to detect and normalize separators
    formatted_lines = []
    
    for line in lines:
        # Convert tab-separated to pipe-separated
        if '\t' in line and '|' not in line:
            cells = line.split('\t')
            line = '| ' + ' | '.join(cell.strip() for cell in cells) + ' |'
        
        formatted_lines.append(line)
    
    # Add separator after header if not present
    if len(formatted_lines) >= 2:
        first_line = formatted_lines[0]
        second_line = formatted_lines[1]
        
        if '|' in first_line and not second_line.startswith('|--'):
            # Count columns
            col_count = first_line.count('|') - 1
            separator = '|' + '---|' * col_count
            formatted_lines.insert(1, separator)
    
    return '\n'.join(formatted_lines)

