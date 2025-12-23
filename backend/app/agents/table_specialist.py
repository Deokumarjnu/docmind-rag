"""Table Specialist subagent for table extraction and merging.

This agent specializes in identifying table boundaries, detecting
table continuations, and converting tables to clean formats.
"""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import settings
from app.agents.tools.table_tools import (
    extract_table_structure,
    detect_table_continuation,
    merge_multipage_tables,
    format_table_to_markdown,
)

logger = logging.getLogger(__name__)

TABLE_SPECIALIST_PROMPT = """You are a Table Extraction Specialist.

Your job is to:
1. Identify table boundaries on the page
2. Detect if tables continue from previous pages
3. Merge multi-page tables into complete units
4. Convert tables to clean markdown format
5. Preserve column headers and row relationships

When processing a page:
- First analyze the content structure
- Identify table regions
- Check for continuation indicators ("continued", matching headers at top)
- Extract and format tables properly

Always preserve the semantic meaning of the data."""


class TableSpecialist:
    """
    Table Specialist agent for processing table content.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize table specialist.
        
        Args:
            model_name: LLM model name to use
        """
        self.model = ChatOpenAI(
            model=model_name or settings.llm_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )
        self.tools = [
            extract_table_structure,
            detect_table_continuation,
            merge_multipage_tables,
            format_table_to_markdown,
        ]

    def process_page(self, page: Document) -> Document:
        """
        Process a page containing table content.
        
        Args:
            page: Page document
            
        Returns:
            Processed document with table extraction
        """
        content = page.page_content
        
        # Extract table structure
        structure = extract_table_structure.invoke({"content": content})
        
        if structure.get("has_table"):
            # Format to markdown
            formatted = format_table_to_markdown.invoke({"content": content})
            
            return Document(
                page_content=formatted,
                metadata={
                    **page.metadata,
                    "content_type": "table",
                    "table_headers": structure.get("headers", []),
                    "processed_by": "table_specialist",
                }
            )
        
        return page

    def check_continuation(
        self,
        prev_page: Document,
        curr_page: Document,
    ) -> bool:
        """
        Check if current page's table continues from previous.
        
        Args:
            prev_page: Previous page
            curr_page: Current page
            
        Returns:
            True if continuation detected
        """
        return detect_table_continuation.invoke({
            "prev_content": prev_page.page_content,
            "curr_content": curr_page.page_content,
        })

    def merge_tables(
        self,
        table1: Document,
        table2: Document,
    ) -> Document:
        """
        Merge two table documents.
        
        Args:
            table1: First table
            table2: Second table (continuation)
            
        Returns:
            Merged table document
        """
        merged_content = merge_multipage_tables.invoke({
            "table1_content": table1.page_content,
            "table2_content": table2.page_content,
        })
        
        return Document(
            page_content=merged_content,
            metadata={
                **table1.metadata,
                "merged_table": True,
                "spans_pages": [
                    table1.metadata.get("page", 0),
                    table2.metadata.get("page", 1),
                ],
            }
        )

    def process_pages(self, pages: list[Document]) -> list[Document]:
        """
        Process multiple pages, handling table continuations.
        
        Args:
            pages: List of page documents
            
        Returns:
            Processed documents
        """
        result = []
        current_table: Optional[Document] = None
        
        for page in pages:
            processed = self.process_page(page)
            
            if processed.metadata.get("content_type") == "table":
                if current_table and self.check_continuation(current_table, processed):
                    current_table = self.merge_tables(current_table, processed)
                else:
                    if current_table:
                        result.append(current_table)
                    current_table = processed
            else:
                if current_table:
                    result.append(current_table)
                    current_table = None
                result.append(processed)
        
        if current_table:
            result.append(current_table)
        
        return result

