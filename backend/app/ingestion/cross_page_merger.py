"""Cross-page text continuity detection and merging.

This module handles text that continues across page boundaries,
detecting incomplete sentences and merging them properly.
"""

import logging
import re
from typing import Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def detect_and_merge_cross_page_text(pages: list[Document]) -> list[Document]:
    """
    Merge text that continues across page boundaries.
    
    Args:
        pages: List of page documents in order
        
    Returns:
        List of documents with cross-page text merged
    """
    if len(pages) < 2:
        return pages
    
    merged_pages = []
    
    for i, page in enumerate(pages):
        content = page.page_content
        is_merged = False
        
        if i > 0 and is_text_continuation(pages[i - 1], page):
            # Get text fragments to merge
            prev_trailing = get_trailing_incomplete_text(pages[i - 1])
            curr_leading = get_leading_incomplete_text(page)
            
            if prev_trailing and curr_leading:
                # Merge the text fragments
                merged_text = prev_trailing + " " + curr_leading
                
                # Update current page content
                content = merged_text + content[len(curr_leading):]
                
                # Remove trailing from previous page
                prev_content = merged_pages[-1].page_content
                merged_pages[-1] = Document(
                    page_content=prev_content[:-len(prev_trailing)].rstrip(),
                    metadata=merged_pages[-1].metadata,
                )
                
                is_merged = True
                logger.debug(f"Merged text across pages {i - 1} and {i}")
        
        merged_pages.append(Document(
            page_content=content,
            metadata={
                **page.metadata,
                "cross_page_merged": is_merged,
            }
        ))
    
    return merged_pages


def is_text_continuation(prev_page: Document, curr_page: Document) -> bool:
    """
    Check if current page continues from previous.
    
    Heuristics:
    1. Previous page ends mid-sentence
    2. Current page starts with lowercase
    3. Previous page ends with hyphenated word
    4. Previous page ends with continuation words
    
    Args:
        prev_page: Previous page document
        curr_page: Current page document
        
    Returns:
        True if continuation detected
    """
    prev_text = prev_page.page_content.strip()
    curr_text = curr_page.page_content.strip()
    
    if not prev_text or not curr_text:
        return False
    
    # Previous page ends with hyphenated word (word split across pages)
    if prev_text.rstrip().endswith('-'):
        return True
    
    # Previous page ends mid-sentence (no terminal punctuation)
    ends_incomplete = prev_text[-1] not in '.!?:;"\''
    
    # Current page starts with lowercase (strong continuation indicator)
    starts_lowercase = curr_text[0].islower()
    
    if ends_incomplete and starts_lowercase:
        return True
    
    # Previous ends with comma or conjunction
    continuation_endings = [',', ';', ' and', ' or', ' but', ' the', ' a', ' an', ' that', ' which']
    for ending in continuation_endings:
        if prev_text.rstrip().endswith(ending):
            return True
    
    return False


def get_trailing_incomplete_text(page: Document, max_chars: int = 500) -> str:
    """
    Get the incomplete sentence/paragraph at end of page.
    
    Args:
        page: Page document
        max_chars: Maximum characters to look back
        
    Returns:
        Trailing incomplete text
    """
    text = page.page_content.strip()
    
    # Handle hyphenated words
    if text.endswith('-'):
        words = text.split()
        if words:
            return words[-1]  # Return the hyphenated word fragment
        return ""
    
    # Find last complete sentence
    search_start = max(0, len(text) - max_chars)
    search_text = text[search_start:]
    
    for i in range(len(search_text) - 1, -1, -1):
        if search_text[i] in '.!?':
            trailing = search_text[i + 1:].strip()
            if trailing:
                return trailing
            break
    
    return ""


def get_leading_incomplete_text(page: Document, max_chars: int = 500) -> str:
    """
    Get the incomplete text at start of page (continuation from previous).
    
    Args:
        page: Page document
        max_chars: Maximum characters to look forward
        
    Returns:
        Leading incomplete text
    """
    text = page.page_content.strip()
    search_text = text[:max_chars]
    
    # If starts with lowercase, find the first complete sentence end
    if text and text[0].islower():
        for i, char in enumerate(search_text):
            if char in '.!?':
                return search_text[:i + 1]
    
    # Handle continuation of hyphenated word
    words = text.split()
    if words and not words[0][0].isupper():
        return words[0]
    
    return ""


def merge_hyphenated_words(pages: list[Document]) -> list[Document]:
    """
    Specifically handle words split by hyphenation across pages.
    
    Args:
        pages: List of page documents
        
    Returns:
        List with hyphenated words merged
    """
    if len(pages) < 2:
        return pages
    
    result = []
    
    for i, page in enumerate(pages):
        content = page.page_content
        
        if i > 0:
            prev_content = result[-1].page_content
            
            # Check if previous page ends with hyphen
            if prev_content.rstrip().endswith('-'):
                # Get the word fragment from previous page
                prev_words = prev_content.rstrip()[:-1].split()
                if prev_words:
                    word_start = prev_words[-1]
                    
                    # Get the word fragment from current page
                    curr_words = content.split()
                    if curr_words:
                        word_end = curr_words[0]
                        
                        # Merge the word
                        merged_word = word_start + word_end
                        
                        # Update previous page
                        new_prev = prev_content[:-len(word_start) - 1].rstrip()
                        result[-1] = Document(
                            page_content=new_prev,
                            metadata=result[-1].metadata,
                        )
                        
                        # Update current page
                        content = merged_word + " " + " ".join(curr_words[1:])
        
        result.append(Document(
            page_content=content,
            metadata=page.metadata,
        ))
    
    return result


class CrossPageMerger:
    """
    Cross-page text merger for handling continuations.
    """

    def __init__(self, merge_hyphenated: bool = True):
        """
        Initialize merger.
        
        Args:
            merge_hyphenated: Whether to merge hyphenated words
        """
        self.merge_hyphenated = merge_hyphenated

    def process(self, pages: list[Document]) -> list[Document]:
        """
        Process pages to merge cross-page text.
        
        Args:
            pages: List of page documents
            
        Returns:
            List with merged text
        """
        result = detect_and_merge_cross_page_text(pages)
        
        if self.merge_hyphenated:
            result = merge_hyphenated_words(result)
        
        return result

