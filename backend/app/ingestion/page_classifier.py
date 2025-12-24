"""Page type classifier for adaptive document processing.

This module classifies document pages into content types to determine
the optimal extraction strategy for each page.
"""

import re
from enum import Enum
from typing import Optional

from langchain_core.documents import Document


class PageType(str, Enum):
    """Classification of document page content types."""

    TEXT = "text"           # Pure text content
    TABLE = "table"         # Contains tables
    IMAGE = "image"         # Image-only or scanned page
    CHART = "chart"         # Contains charts, graphs, or diagrams
    CODE = "code"           # Contains programming code
    HANDWRITING = "handwriting"  # Contains handwritten text
    MIXED = "mixed"         # Mixed content types


def classify_page(page: Document) -> PageType:
    """
    Classify a page based on its content to determine extraction strategy.
    
    Args:
        page: Document object with page_content and metadata
        
    Returns:
        PageType indicating the dominant content type
    """
    text = page.page_content.strip()
    
    # Check for minimal text (likely image/scanned page)
    if len(text) < 30:
        return PageType.IMAGE
    
    # Check for code content
    if _is_code_content(text):
        return PageType.CODE
    
    # Check for table content
    if _is_table_content(text):
        return PageType.TABLE
    
    # Check for chart/diagram indicators
    if _is_chart_content(text):
        return PageType.CHART
    
    # Check for mixed layout
    if _has_mixed_layout(text):
        return PageType.MIXED
    
    # Default to text
    return PageType.TEXT


def _is_code_content(text: str) -> bool:
    """
    Detect if text contains programming code.
    
    Uses pattern matching for common code indicators across multiple languages.
    """
    code_indicators = [
        (r'def\s+\w+\s*\(', 2),           # Python function
        (r'function\s+\w+\s*\(', 2),       # JavaScript function
        (r'class\s+\w+[\s:{]', 2),         # Class definition
        (r'import\s+[\w.{},\s]+', 1),      # Import statements
        (r'from\s+\w+\s+import', 2),       # Python imports
        (r'const\s+\w+\s*=', 1),           # JS const
        (r'let\s+\w+\s*=', 1),             # JS let
        (r'var\s+\w+\s*=', 1),             # JS var
        (r'^\s{4,}\S', 1),                 # Significant indentation
        (r'if\s*\(.*\)\s*\{', 1),          # If statements (C-style)
        (r'for\s*\(.*\)\s*\{', 1),         # For loops (C-style)
        (r'=>', 1),                         # Arrow functions
        (r'console\.log\(', 1),            # JS logging
        (r'print\(', 1),                   # Python print
        (r'return\s+\w', 1),               # Return statements
        (r'async\s+(def|function)', 2),    # Async functions
        (r'await\s+\w', 1),                # Await expressions
        (r'\}\s*else\s*\{', 1),            # Else blocks
        (r'try\s*[:{]', 1),                # Try blocks
        (r'except\s+\w', 1),               # Python except
        (r'catch\s*\(', 1),                # JS catch
        (r'public\s+class', 2),            # Java class
        (r'private\s+\w+', 1),             # Java/C# private
        (r'fn\s+\w+', 2),                  # Rust function
        (r'func\s+\w+', 2),                # Go function
    ]
    
    score = sum(
        weight for pattern, weight in code_indicators
        if re.search(pattern, text, re.MULTILINE)
    )
    return score >= 3


def _is_table_content(text: str) -> bool:
    """
    Detect if text contains table structures.
    
    Looks for common table indicators like pipe characters, 
    aligned columns, and table-related keywords.
    """
    # Check for markdown-style tables
    if re.search(r'\|.*\|.*\|', text):
        return True
    
    # Check for tab-separated values (multiple tabs per line)
    lines = text.split('\n')
    tab_lines = sum(1 for line in lines if line.count('\t') >= 2)
    if tab_lines >= 3:
        return True
    
    # Check for aligned columns (consistent spacing patterns)
    if _has_aligned_columns(text):
        return True
    
    # Check for table-related keywords
    table_keywords = ['table', 'column', 'row', 'header', 'total', 'subtotal']
    keyword_count = sum(
        1 for kw in table_keywords
        if kw in text.lower()
    )
    if keyword_count >= 2:
        # Additional check for numeric data patterns
        if re.search(r'\d+\s+\d+\s+\d+', text):
            return True
    
    return False


def _has_aligned_columns(text: str) -> bool:
    """
    Detect if text has aligned columns (common in tables).
    
    Checks for consistent spacing patterns across multiple lines.
    """
    lines = [line for line in text.split('\n') if line.strip()]
    if len(lines) < 3:
        return False
    
    # Check for consistent multi-space gaps (column separators)
    space_patterns = []
    for line in lines[:10]:  # Check first 10 lines
        # Find positions of 2+ consecutive spaces
        positions = [m.start() for m in re.finditer(r'\s{2,}', line)]
        if positions:
            space_patterns.append(tuple(positions[:5]))  # First 5 gaps
    
    if len(space_patterns) < 3:
        return False
    
    # Check if patterns are similar across lines (columns aligned)
    reference = space_patterns[0]
    matches = sum(
        1 for pattern in space_patterns[1:]
        if _similar_patterns(reference, pattern)
    )
    
    return matches >= len(space_patterns) // 2


def _similar_patterns(p1: tuple, p2: tuple, tolerance: int = 3) -> bool:
    """Check if two position patterns are similar within tolerance."""
    if len(p1) != len(p2):
        return False
    return all(abs(a - b) <= tolerance for a, b in zip(p1, p2))


def _is_chart_content(text: str) -> bool:
    """
    Detect if text likely describes charts, graphs, or diagrams.
    
    This is a heuristic check; actual chart detection requires image analysis.
    """
    chart_keywords = [
        'figure', 'chart', 'graph', 'diagram', 'plot',
        'axis', 'x-axis', 'y-axis', 'legend', 'scale',
        'bar chart', 'pie chart', 'line graph', 'scatter',
        'histogram', 'flowchart', 'workflow'
    ]
    
    text_lower = text.lower()
    keyword_matches = sum(1 for kw in chart_keywords if kw in text_lower)
    
    # Also check for figure references
    figure_refs = len(re.findall(r'(?:fig(?:ure)?\.?\s*\d+|chart\s*\d+)', text_lower))
    
    return keyword_matches >= 2 or figure_refs >= 1


def _has_mixed_layout(text: str) -> bool:
    """
    Detect if page has mixed content that needs special handling.
    
    Looks for indicators of complex layouts combining text, 
    images, tables, etc.
    """
    indicators = 0
    
    # Check for image placeholders/references
    if re.search(r'\[image\]|\[figure\]|\[photo\]', text, re.IGNORECASE):
        indicators += 1
    
    # Check for footnotes or endnotes
    if re.search(r'^\s*\d+\.\s|^\s*\[\d+\]', text, re.MULTILINE):
        indicators += 1
    
    # Check for multiple sections with headers
    header_count = len(re.findall(r'^[A-Z][^.!?]*:?\s*$', text, re.MULTILINE))
    if header_count >= 2:
        indicators += 1
    
    # Check for sidebars or callout boxes (often indented blocks)
    indented_blocks = len(re.findall(r'^\s{8,}\S', text, re.MULTILINE))
    if indented_blocks >= 3:
        indicators += 1
    
    # Check for bullet points mixed with paragraphs
    bullets = len(re.findall(r'^\s*[•\-\*]\s', text, re.MULTILINE))
    paragraphs = len(re.findall(r'\n\n[A-Z]', text))
    if bullets >= 2 and paragraphs >= 2:
        indicators += 1
    
    return indicators >= 2


def get_extraction_strategy(page_type: PageType) -> str:
    """
    Get the recommended extraction strategy for a page type.
    
    Args:
        page_type: The classified page type
        
    Returns:
        Strategy name: 'pymupdf', 'unstructured_elements', 
                      'unstructured_auto', 'ocr', or 'vision_llm'
    """
    strategies = {
        PageType.TEXT: "pymupdf",
        PageType.TABLE: "unstructured_elements",
        PageType.IMAGE: "vision_llm",  # Use vision LLM for images
        PageType.CHART: "vision_llm",  # Use vision LLM for charts/diagrams/figures
        PageType.CODE: "pymupdf",
        PageType.HANDWRITING: "vision_llm",
        PageType.MIXED: "unstructured_auto",
    }
    return strategies.get(page_type, "pymupdf")


def classify_and_annotate_page(page: Document) -> Document:
    """
    Classify a page and add classification metadata.
    
    Args:
        page: Document object to classify
        
    Returns:
        Document with added metadata for page_type and extraction_strategy
    """
    page_type = classify_page(page)
    strategy = get_extraction_strategy(page_type)
    
    page.metadata["page_type"] = page_type.value
    page.metadata["extraction_strategy"] = strategy
    
    return page

