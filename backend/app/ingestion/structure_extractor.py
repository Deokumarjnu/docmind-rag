"""Document structure extraction for navigation and context.

This module extracts hierarchical structure from documents including
table of contents, chapters, and sections.
"""

import logging
import re
from typing import Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def extract_document_structure(pages: list[Document]) -> dict:
    """
    Extract document hierarchy: TOC, chapters, sections.
    
    Args:
        pages: List of page documents
        
    Returns:
        Dictionary with document structure
    """
    structure = {
        "title": None,
        "toc": [],
        "chapters": [],
        "sections": [],
    }
    
    if not pages:
        return structure
    
    # Try to find title (usually first page, large font or prominent position)
    structure["title"] = extract_title(pages[0])
    
    # Look for Table of Contents
    toc_page = find_toc_page(pages)
    if toc_page:
        structure["toc"] = parse_toc(toc_page)
    
    # Extract section headers from all pages
    for page in pages:
        headers = extract_headers_from_page(page)
        for header in headers:
            if header["level"] == 1:
                structure["chapters"].append(header)
            else:
                structure["sections"].append(header)
    
    return structure


def extract_title(page: Document) -> Optional[str]:
    """
    Extract document title from first page.
    
    Args:
        page: First page document
        
    Returns:
        Document title or None
    """
    lines = page.page_content.strip().split('\n')
    
    for line in lines[:10]:
        line = line.strip()
        
        if not line:
            continue
        
        # Skip common non-title lines
        skip_patterns = [
            r'^\d+$',  # Just a number (page number)
            r'^page\s+\d+',  # Page X
            r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$',  # Date
            r'^table of contents$',
        ]
        
        if any(re.match(p, line, re.IGNORECASE) for p in skip_patterns):
            continue
        
        # Title criteria: relatively short, not a full sentence
        if 10 < len(line) < 200 and not line.endswith('.'):
            return line
    
    return None


def find_toc_page(pages: list[Document]) -> Optional[Document]:
    """
    Find the table of contents page.
    
    Args:
        pages: List of page documents
        
    Returns:
        TOC page document or None
    """
    toc_patterns = [
        r'table\s+of\s+contents',
        r'contents',
        r'^toc$',
        r'index',
    ]
    
    # Check first few pages
    for page in pages[:10]:
        text = page.page_content.lower()
        
        for pattern in toc_patterns:
            if re.search(pattern, text):
                # Verify it looks like a TOC
                if _looks_like_toc(page.page_content):
                    return page
    
    return None


def _looks_like_toc(text: str) -> bool:
    """
    Check if text looks like a table of contents.
    
    Args:
        text: Page text
        
    Returns:
        True if appears to be TOC
    """
    lines = text.strip().split('\n')
    
    # TOC typically has multiple numbered entries with page numbers
    numbered_lines = 0
    page_ref_lines = 0
    
    for line in lines:
        # Check for numbered sections
        if re.match(r'^\s*\d+\.?\s+\w', line):
            numbered_lines += 1
        
        # Check for page references (dots or numbers at end)
        if re.search(r'\.{3,}\s*\d+\s*$|\s{2,}\d+\s*$', line):
            page_ref_lines += 1
    
    return numbered_lines >= 3 or page_ref_lines >= 3


def parse_toc(toc_page: Document) -> list[dict]:
    """
    Parse table of contents into structured entries.
    
    Args:
        toc_page: TOC page document
        
    Returns:
        List of TOC entries
    """
    entries = []
    lines = toc_page.page_content.strip().split('\n')
    
    for line in lines:
        entry = _parse_toc_line(line)
        if entry:
            entries.append(entry)
    
    return entries


def _parse_toc_line(line: str) -> Optional[dict]:
    """
    Parse a single TOC line.
    
    Args:
        line: TOC line
        
    Returns:
        TOC entry dict or None
    """
    line = line.strip()
    if not line:
        return None
    
    # Try to extract: number, title, page
    # Pattern: "1.2.3 Title ..... 42"
    match = re.match(
        r'^(\d+(?:\.\d+)*\.?)\s+(.+?)\s*\.{0,}(\d+)?\s*$',
        line
    )
    
    if match:
        number, title, page = match.groups()
        return {
            "number": number.rstrip('.'),
            "title": title.strip().rstrip('.'),
            "page": int(page) if page else None,
            "level": number.count('.') + 1,
        }
    
    # Try simpler pattern: "Title ... 42"
    match = re.match(r'^(.+?)\s*\.{3,}\s*(\d+)\s*$', line)
    if match:
        title, page = match.groups()
        return {
            "number": None,
            "title": title.strip(),
            "page": int(page),
            "level": 1,
        }
    
    return None


def extract_headers_from_page(page: Document) -> list[dict]:
    """
    Extract section headers from a page.
    
    Args:
        page: Page document
        
    Returns:
        List of header entries
    """
    headers = []
    lines = page.page_content.strip().split('\n')
    page_num = page.metadata.get("page", 0)
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        if is_likely_header(line):
            level = determine_header_level(line)
            headers.append({
                "text": clean_header_text(line),
                "level": level,
                "page": page_num,
                "line": i,
            })
    
    return headers


def is_likely_header(line: str) -> bool:
    """
    Check if a line is likely a section header.
    
    Args:
        line: Text line
        
    Returns:
        True if likely a header
    """
    line = line.strip()
    
    if not line or len(line) > 200:
        return False
    
    # Numbered sections: "1.2.3 Title"
    if re.match(r'^[\d.]+\s+\w', line):
        return True
    
    # All caps header (at least 3 words)
    if line.isupper() and len(line.split()) >= 2:
        return True
    
    # Title case and short
    words = line.split()
    if len(words) >= 2 and len(words) <= 10:
        # Check if most words are capitalized
        cap_words = sum(1 for w in words if w and w[0].isupper())
        if cap_words / len(words) >= 0.7 and not line.endswith('.'):
            return True
    
    # Common header patterns
    header_patterns = [
        r'^chapter\s+\d+',
        r'^section\s+\d+',
        r'^part\s+\d+',
        r'^appendix\s+[a-z]',
        r'^\d+\.\s+[A-Z]',
    ]
    
    for pattern in header_patterns:
        if re.match(pattern, line, re.IGNORECASE):
            return True
    
    return False


def determine_header_level(line: str) -> int:
    """
    Determine the hierarchy level of a header.
    
    Args:
        line: Header line
        
    Returns:
        Level number (1 = top level)
    """
    line = line.strip()
    
    # Check for numbered sections
    match = re.match(r'^(\d+(?:\.\d+)*)', line)
    if match:
        return match.group(1).count('.') + 1
    
    # Chapter is level 1
    if re.match(r'^chapter\s+', line, re.IGNORECASE):
        return 1
    
    # All caps is usually level 1 or 2
    if line.isupper():
        return 1
    
    # Part/Appendix is level 1
    if re.match(r'^(part|appendix)\s+', line, re.IGNORECASE):
        return 1
    
    # Section is level 2
    if re.match(r'^section\s+', line, re.IGNORECASE):
        return 2
    
    # Default to level 2
    return 2


def clean_header_text(line: str) -> str:
    """
    Clean header text for storage.
    
    Args:
        line: Raw header line
        
    Returns:
        Cleaned header text
    """
    line = line.strip()
    
    # Remove leading numbers
    line = re.sub(r'^[\d.]+\s*', '', line)
    
    # Remove trailing punctuation
    line = line.rstrip('.:')
    
    return line.strip()


def add_structure_to_chunks(
    chunks: list[Document],
    structure: dict,
) -> list[Document]:
    """
    Enrich chunks with structural context.
    
    Args:
        chunks: List of chunk documents
        structure: Extracted document structure
        
    Returns:
        Chunks with added structure metadata
    """
    for chunk in chunks:
        page_num = chunk.metadata.get("page", 0)
        
        # Find containing chapter/section
        chapter = find_containing_section(page_num, structure.get("chapters", []))
        section = find_containing_section(page_num, structure.get("sections", []))
        
        chunk.metadata.update({
            "chapter": chapter,
            "section": section,
            "document_title": structure.get("title"),
        })
    
    return chunks


def find_containing_section(
    page_num: int,
    sections: list[dict],
) -> Optional[str]:
    """
    Find which section a page belongs to.
    
    Args:
        page_num: Page number
        sections: List of section headers
        
    Returns:
        Section title or None
    """
    if not sections:
        return None
    
    # Find the last section that starts before or on this page
    containing = None
    for section in sections:
        section_page = section.get("page", 0)
        if section_page <= page_num:
            containing = section.get("text")
    
    return containing


class StructureExtractor:
    """
    Document structure extractor.
    """

    def __init__(self):
        """Initialize structure extractor."""
        self.structure: dict = {}

    def extract(self, pages: list[Document]) -> dict:
        """
        Extract structure from pages.
        
        Args:
            pages: List of page documents
            
        Returns:
            Document structure dictionary
        """
        self.structure = extract_document_structure(pages)
        return self.structure

    def enrich_chunks(self, chunks: list[Document]) -> list[Document]:
        """
        Add structure context to chunks.
        
        Args:
            chunks: List of chunk documents
            
        Returns:
            Enriched chunks
        """
        return add_structure_to_chunks(chunks, self.structure)

