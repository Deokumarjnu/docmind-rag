"""Tools for Text Specialist subagent.

These tools enable the text specialist to extract and clean
plain text content from documents.
"""

import logging
import re
from typing import Optional

from langchain_core.tools import tool

from app.ingestion.header_footer_remover import (
    detect_headers_footers,
    remove_headers_footers,
)
from app.ingestion.cross_page_merger import (
    is_text_continuation,
    get_trailing_incomplete_text,
    get_leading_incomplete_text,
)

logger = logging.getLogger(__name__)


@tool
def extract_clean_text(content: str) -> str:
    """
    Extract and clean text from page content.
    
    Removes noise, normalizes whitespace, and cleans up formatting.
    
    Args:
        content: Raw page content
        
    Returns:
        Cleaned text content
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', content)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[.]{4,}', '...', text)
    text = re.sub(r'[-]{3,}', '--', text)
    text = re.sub(r'[_]{3,}', '__', text)
    
    return text.strip()


@tool
def remove_headers_footers_tool(
    content: str,
    header_pattern: Optional[str] = None,
    footer_pattern: Optional[str] = None,
) -> str:
    """
    Remove repetitive headers and footers from page content.
    
    Args:
        content: Page content
        header_pattern: Known header pattern to remove
        footer_pattern: Known footer pattern to remove
        
    Returns:
        Content with headers/footers removed
    """
    return remove_headers_footers(content, header_pattern, footer_pattern)


@tool
def detect_section_headers(content: str) -> list[dict]:
    """
    Detect section headers in page content.
    
    Args:
        content: Page content
        
    Returns:
        List of detected headers with positions
    """
    headers = []
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        if not line:
            continue
        
        # Check for numbered sections
        if re.match(r'^[\d.]+\s+\w', line):
            headers.append({
                "text": line,
                "line": i,
                "type": "numbered",
            })
            continue
        
        # Check for all caps headers
        if line.isupper() and 10 < len(line) < 100:
            headers.append({
                "text": line,
                "line": i,
                "type": "caps",
            })
            continue
        
        # Check for title case short lines
        if (len(line.split()) <= 8 
            and not line.endswith('.')
            and line[0].isupper()):
            # Check if mostly capitalized words
            words = line.split()
            cap_count = sum(1 for w in words if w and w[0].isupper())
            if cap_count / len(words) >= 0.7:
                headers.append({
                    "text": line,
                    "line": i,
                    "type": "title_case",
                })
    
    return headers


@tool
def handle_cross_page_text(
    prev_page_content: str,
    curr_page_content: str,
) -> dict:
    """
    Handle text that continues across page boundaries.
    
    Args:
        prev_page_content: Previous page content
        curr_page_content: Current page content
        
    Returns:
        Dictionary with continuation info and merged text if applicable
    """
    from langchain_core.documents import Document
    
    prev_doc = Document(page_content=prev_page_content)
    curr_doc = Document(page_content=curr_page_content)
    
    is_continuation = is_text_continuation(prev_doc, curr_doc)
    
    result = {
        "is_continuation": is_continuation,
        "trailing_text": None,
        "leading_text": None,
        "merged_text": None,
    }
    
    if is_continuation:
        trailing = get_trailing_incomplete_text(prev_doc)
        leading = get_leading_incomplete_text(curr_doc)
        
        result["trailing_text"] = trailing
        result["leading_text"] = leading
        
        if trailing and leading:
            result["merged_text"] = trailing + " " + leading
    
    return result

