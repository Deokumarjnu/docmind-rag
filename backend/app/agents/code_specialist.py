"""Code Specialist subagent for code detection and chunking.

This agent specializes in detecting programming code, identifying
languages, and chunking code by logical units.
"""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.config import settings
from app.agents.tools.code_tools import (
    detect_programming_language_tool,
    extract_code_blocks,
    parse_with_ast,
    chunk_by_functions,
)
from app.ingestion.code_chunker import detect_code_block

logger = logging.getLogger(__name__)

CODE_SPECIALIST_PROMPT = """You are a Code Extraction Specialist.

Your job is to:
1. Identify code blocks in the document
2. Detect the programming language
3. Extract code while preserving formatting
4. Split code by logical units (functions, classes)
5. Add language metadata for syntax highlighting

When processing code:
- Preserve exact formatting and indentation
- Keep comments and docstrings
- Group related code together
- Identify function/class boundaries"""


class CodeSpecialist:
    """
    Code Specialist agent for processing code content.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize code specialist.
        
        Args:
            model_name: LLM model name to use
        """
        self.model = ChatOpenAI(
            model=model_name or "gpt-4o-mini",  # Faster for code parsing
            temperature=0,
            api_key=settings.openai_api_key,
        )
        self.tools = [
            detect_programming_language_tool,
            extract_code_blocks,
            parse_with_ast,
            chunk_by_functions,
        ]

    def has_code(self, content: str) -> bool:
        """
        Check if content contains code.
        
        Args:
            content: Text content
            
        Returns:
            True if code detected
        """
        return detect_code_block(content)

    def detect_language(self, content: str) -> str:
        """
        Detect programming language of code.
        
        Args:
            content: Code content
            
        Returns:
            Language name
        """
        return detect_programming_language_tool.invoke({"content": content})

    def process_page(self, page: Document) -> list[Document]:
        """
        Process a page containing code content.
        
        Args:
            page: Page document
            
        Returns:
            List of code chunk documents
        """
        content = page.page_content
        
        if not self.has_code(content):
            return [page]
        
        # Detect language
        language = self.detect_language(content)
        
        # Extract code blocks
        blocks = extract_code_blocks.invoke({"content": content})
        
        if not blocks:
            return [page]
        
        # Chunk by functions
        chunks = chunk_by_functions.invoke({
            "code": content,
            "language": language,
        })
        
        result = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    **page.metadata,
                    "content_type": "code",
                    "language": language,
                    "chunk_index": i,
                    "processed_by": "code_specialist",
                }
            )
            result.append(doc)
        
        return result

    def parse_code(self, content: str, language: str) -> dict:
        """
        Parse code and extract structure.
        
        Args:
            content: Code content
            language: Programming language
            
        Returns:
            Parsed structure
        """
        return parse_with_ast.invoke({
            "code": content,
            "language": language,
        })

    def process_pages(self, pages: list[Document]) -> list[Document]:
        """
        Process multiple pages for code content.
        
        Args:
            pages: List of page documents
            
        Returns:
            Processed documents
        """
        result = []
        for page in pages:
            processed = self.process_page(page)
            result.extend(processed)
        return result

