"""Acronym and abbreviation detection and expansion.

This module detects acronym definitions in documents and expands
them for better retrieval.
"""

import logging
import re
from typing import Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def extract_acronyms_from_document(pages: list[Document]) -> dict[str, str]:
    """
    Extract acronym definitions from document itself.
    
    Detects patterns like:
    - "Machine Learning (ML)"
    - "ML (Machine Learning)"
    
    Args:
        pages: List of page documents
        
    Returns:
        Dictionary of acronym -> expansion
    """
    acronym_dict = {}
    
    # Pattern: "Full Form (ACRONYM)"
    pattern1 = r'([A-Z][a-zA-Z\s]+)\s*\(([A-Z]{2,})\)'
    
    # Pattern: "ACRONYM (Full Form)"
    pattern2 = r'([A-Z]{2,})\s*\(([A-Za-z\s]+)\)'
    
    for page in pages:
        text = page.page_content
        
        # Match "Full Form (ACRONYM)"
        for match in re.finditer(pattern1, text):
            full_form, acronym = match.groups()
            full_form = full_form.strip()
            acronym = acronym.strip()
            
            # Validate: acronym should be initials of full form
            if _validate_acronym(acronym, full_form):
                acronym_dict[acronym] = full_form
        
        # Match "ACRONYM (Full Form)"
        for match in re.finditer(pattern2, text):
            acronym, full_form = match.groups()
            full_form = full_form.strip()
            acronym = acronym.strip()
            
            if _validate_acronym(acronym, full_form):
                acronym_dict[acronym] = full_form
    
    logger.info(f"Extracted {len(acronym_dict)} acronym definitions")
    return acronym_dict


def _validate_acronym(acronym: str, full_form: str) -> bool:
    """
    Validate that acronym matches the full form.
    
    Args:
        acronym: The acronym (e.g., "ML")
        full_form: The full form (e.g., "Machine Learning")
        
    Returns:
        True if acronym appears valid for full form
    """
    if len(acronym) < 2:
        return False
    
    words = full_form.split()
    if len(words) < 2:
        return False
    
    # Check if acronym is initials of words
    initials = "".join(w[0].upper() for w in words if w and w[0].isalpha())
    if acronym.upper() == initials:
        return True
    
    # Allow partial matches (some words may be skipped)
    if len(acronym) <= len(words):
        return True
    
    return False


def expand_acronyms_in_text(text: str, acronym_dict: dict[str, str]) -> str:
    """
    Expand acronyms in text for better retrieval.
    
    Args:
        text: Document text
        acronym_dict: Dictionary of acronym -> expansion
        
    Returns:
        Text with acronyms expanded
    """
    if not acronym_dict:
        return text
    
    for acronym, expansion in acronym_dict.items():
        # Replace standalone acronyms with expanded form
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(acronym) + r'\b'
        
        # Check if it's not already expanded
        expanded_pattern = f"{acronym} ({expansion})"
        if expanded_pattern not in text:
            replacement = f"{acronym} ({expansion})"
            text = re.sub(pattern, replacement, text)
    
    return text


def expand_acronyms_in_query(query: str, acronym_dict: dict[str, str]) -> str:
    """
    Expand acronyms in user query for better matching.
    
    Args:
        query: User query
        acronym_dict: Dictionary of acronym -> expansion
        
    Returns:
        Query with acronyms expanded
    """
    if not acronym_dict:
        return query
    
    words = query.split()
    expanded = []
    
    for word in words:
        # Clean word for matching
        clean_word = re.sub(r'[^\w]', '', word).upper()
        
        if clean_word in acronym_dict:
            # Keep original word format but add expansion
            expanded.append(f"{word} ({acronym_dict[clean_word]})")
        else:
            expanded.append(word)
    
    return " ".join(expanded)


def create_acronym_expansion_documents(
    acronym_dict: dict[str, str],
    source: str = "glossary",
) -> list[Document]:
    """
    Create documents for acronym definitions for indexing.
    
    This helps retrieval when users search for either the
    acronym or the full form.
    
    Args:
        acronym_dict: Dictionary of acronym -> expansion
        source: Source identifier
        
    Returns:
        List of acronym definition documents
    """
    documents = []
    
    for acronym, expansion in acronym_dict.items():
        doc = Document(
            page_content=f"{acronym}: {expansion}. {acronym} stands for {expansion}.",
            metadata={
                "source": source,
                "content_type": "acronym_definition",
                "acronym": acronym,
                "expansion": expansion,
            }
        )
        documents.append(doc)
    
    return documents


# Common technical acronyms (fallback if not found in document)
COMMON_ACRONYMS = {
    "AI": "Artificial Intelligence",
    "ML": "Machine Learning",
    "NLP": "Natural Language Processing",
    "API": "Application Programming Interface",
    "SDK": "Software Development Kit",
    "SQL": "Structured Query Language",
    "REST": "Representational State Transfer",
    "JSON": "JavaScript Object Notation",
    "XML": "Extensible Markup Language",
    "HTML": "HyperText Markup Language",
    "CSS": "Cascading Style Sheets",
    "HTTP": "HyperText Transfer Protocol",
    "HTTPS": "HyperText Transfer Protocol Secure",
    "DNS": "Domain Name System",
    "TCP": "Transmission Control Protocol",
    "IP": "Internet Protocol",
    "LLM": "Large Language Model",
    "RAG": "Retrieval Augmented Generation",
    "GPU": "Graphics Processing Unit",
    "CPU": "Central Processing Unit",
    "RAM": "Random Access Memory",
    "SSD": "Solid State Drive",
    "AWS": "Amazon Web Services",
    "GCP": "Google Cloud Platform",
    "CLI": "Command Line Interface",
    "GUI": "Graphical User Interface",
    "IDE": "Integrated Development Environment",
    "ORM": "Object-Relational Mapping",
    "CRUD": "Create Read Update Delete",
    "MVP": "Minimum Viable Product",
    "KPI": "Key Performance Indicator",
    "ROI": "Return on Investment",
    "ETL": "Extract Transform Load",
    "CI": "Continuous Integration",
    "CD": "Continuous Deployment",
    "VCS": "Version Control System",
}


class AcronymExpander:
    """
    Acronym expander for document processing.
    """

    def __init__(self, use_common_acronyms: bool = True):
        """
        Initialize acronym expander.
        
        Args:
            use_common_acronyms: Whether to include common technical acronyms
        """
        self.acronym_dict: dict[str, str] = {}
        self.use_common_acronyms = use_common_acronyms
        
        if use_common_acronyms:
            self.acronym_dict.update(COMMON_ACRONYMS)

    def extract_from_documents(self, documents: list[Document]) -> dict[str, str]:
        """Extract acronyms from documents."""
        extracted = extract_acronyms_from_document(documents)
        self.acronym_dict.update(extracted)
        return self.acronym_dict

    def expand_text(self, text: str) -> str:
        """Expand acronyms in text."""
        return expand_acronyms_in_text(text, self.acronym_dict)

    def expand_query(self, query: str) -> str:
        """Expand acronyms in query."""
        return expand_acronyms_in_query(query, self.acronym_dict)

    def process_documents(self, documents: list[Document]) -> list[Document]:
        """
        Process documents: extract acronyms and expand them.
        
        Args:
            documents: List of documents
            
        Returns:
            Documents with expanded acronyms
        """
        # Extract acronyms
        self.extract_from_documents(documents)
        
        # Expand in each document
        processed = []
        for doc in documents:
            expanded_content = self.expand_text(doc.page_content)
            processed.append(Document(
                page_content=expanded_content,
                metadata={
                    **doc.metadata,
                    "acronyms_expanded": True,
                }
            ))
        
        return processed

