"""Text Specialist subagent for plain text handling.

This agent specializes in extracting and cleaning plain text
content from documents.
"""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.config import settings
from app.agents.tools.text_tools import (
    extract_clean_text,
    remove_headers_footers_tool,
    detect_section_headers,
    handle_cross_page_text,
)

logger = logging.getLogger(__name__)

TEXT_SPECIALIST_PROMPT = """You are a Text Extraction Specialist.

Your job is to:
1. Extract clean text from pages
2. Remove repetitive headers and footers
3. Identify section and chapter headers
4. Handle text that continues across pages
5. Preserve paragraph structure

When processing text:
- Clean up formatting artifacts
- Maintain logical paragraph breaks
- Identify section structure
- Handle cross-page continuations properly"""


class TextSpecialist:
    """
    Text Specialist agent for processing plain text content.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize text specialist.
        
        Args:
            model_name: LLM model name to use
        """
        self.model = ChatOpenAI(
            model=model_name or "gpt-4o-mini",  # Faster for text processing
            temperature=0,
            api_key=settings.openai_api_key,
        )
        self.tools = [
            extract_clean_text,
            remove_headers_footers_tool,
            detect_section_headers,
            handle_cross_page_text,
        ]
        
        # Detected patterns for header/footer removal
        self.header_pattern: Optional[str] = None
        self.footer_pattern: Optional[str] = None

    def clean_text(self, content: str) -> str:
        """
        Clean text content.
        
        Args:
            content: Raw text
            
        Returns:
            Cleaned text
        """
        return extract_clean_text.invoke({"content": content})

    def detect_patterns(self, pages: list[Document]) -> None:
        """
        Detect header/footer patterns from pages.
        
        Args:
            pages: List of page documents
        """
        from app.ingestion.header_footer_remover import detect_headers_footers
        
        self.header_pattern, self.footer_pattern = detect_headers_footers(pages)
        
        if self.header_pattern:
            logger.info(f"Detected header pattern: {self.header_pattern[:50]}...")
        if self.footer_pattern:
            logger.info(f"Detected footer pattern: {self.footer_pattern[:50]}...")

    def process_page(self, page: Document) -> Document:
        """
        Process a page with plain text content.
        
        Args:
            page: Page document
            
        Returns:
            Processed document
        """
        content = page.page_content
        
        # Remove headers/footers if patterns detected
        if self.header_pattern or self.footer_pattern:
            content = remove_headers_footers_tool.invoke({
                "content": content,
                "header_pattern": self.header_pattern,
                "footer_pattern": self.footer_pattern,
            })
        
        # Clean text
        content = self.clean_text(content)
        
        # Detect section headers
        headers = detect_section_headers.invoke({"content": content})
        
        return Document(
            page_content=content,
            metadata={
                **page.metadata,
                "content_type": "text",
                "section_headers": [h["text"] for h in headers],
                "processed_by": "text_specialist",
                "headers_removed": bool(self.header_pattern or self.footer_pattern),
            }
        )

    def handle_continuation(
        self,
        prev_page: Document,
        curr_page: Document,
    ) -> tuple[Document, Document]:
        """
        Handle text continuation between pages.
        
        Args:
            prev_page: Previous page
            curr_page: Current page
            
        Returns:
            Updated (prev_page, curr_page) tuple
        """
        result = handle_cross_page_text.invoke({
            "prev_page_content": prev_page.page_content,
            "curr_page_content": curr_page.page_content,
        })
        
        if result.get("is_continuation"):
            trailing = result.get("trailing_text", "")
            leading = result.get("leading_text", "")
            merged = result.get("merged_text", "")
            
            if trailing and leading and merged:
                # Update previous page
                prev_content = prev_page.page_content
                if trailing:
                    prev_content = prev_content[:-len(trailing)].rstrip()
                
                # Update current page
                curr_content = curr_page.page_content
                if leading:
                    curr_content = merged + curr_content[len(leading):]
                
                prev_page = Document(
                    page_content=prev_content,
                    metadata=prev_page.metadata,
                )
                curr_page = Document(
                    page_content=curr_content,
                    metadata={
                        **curr_page.metadata,
                        "cross_page_merged": True,
                    }
                )
        
        return prev_page, curr_page

    def process_pages(self, pages: list[Document]) -> list[Document]:
        """
        Process multiple pages with text content.
        
        Args:
            pages: List of page documents
            
        Returns:
            Processed documents
        """
        # Detect patterns first
        self.detect_patterns(pages)
        
        # Process each page
        processed = [self.process_page(page) for page in pages]
        
        # Handle cross-page continuations
        if len(processed) > 1:
            result = [processed[0]]
            for i in range(1, len(processed)):
                prev, curr = self.handle_continuation(result[-1], processed[i])
                result[-1] = prev
                result.append(curr)
            return result
        
        return processed

