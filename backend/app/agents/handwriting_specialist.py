"""Handwriting Specialist subagent for handwritten text extraction.

This agent specializes in detecting and transcribing handwritten
text using vision LLM capabilities.
"""

import base64
import logging
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.config import settings
from app.agents.tools.vision_tools import (
    detect_handwritten_regions,
    transcribe_handwriting,
)
from app.ingestion.vision_processor import render_page_to_image
from app.ingestion.handwriting_extractor import is_low_quality_ocr

logger = logging.getLogger(__name__)

HANDWRITING_SPECIALIST_PROMPT = """You are a Handwriting Transcription Specialist.

Your job is to:
1. Identify handwritten regions on pages
2. Accurately transcribe handwritten text
3. Mark unclear portions with [unclear: ...]
4. Preserve structure like lists and diagrams
5. Note any annotations or margin notes

When transcribing:
- Maintain the original structure and formatting
- Indicate uncertainty with brackets
- Preserve meaningful line breaks
- Note any diagrams or symbols"""


class HandwritingSpecialist:
    """
    Handwriting Specialist agent for processing handwritten content.
    """

    def __init__(self, model_name: Optional[str] = None, dpi: int = 200):
        """
        Initialize handwriting specialist.
        
        Args:
            model_name: Vision model to use
            dpi: DPI for page rendering
        """
        self.model = ChatOpenAI(
            model=model_name or settings.vision_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )
        self.dpi = dpi
        self.tools = [
            detect_handwritten_regions,
            transcribe_handwriting,
        ]

    def needs_vision_extraction(self, page: Document) -> bool:
        """
        Check if page needs vision-based extraction.
        
        Args:
            page: Page document with OCR text
            
        Returns:
            True if OCR quality is low and vision needed
        """
        return is_low_quality_ocr(page.page_content)

    def detect_handwriting(self, image_bytes: bytes) -> bool:
        """
        Detect if image contains handwriting.
        
        Args:
            image_bytes: Page image bytes
            
        Returns:
            True if handwriting detected
        """
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        return detect_handwritten_regions.invoke({"image_base64": image_b64})

    def transcribe(self, image_bytes: bytes) -> str:
        """
        Transcribe handwritten text from image.
        
        Args:
            image_bytes: Page image bytes
            
        Returns:
            Transcribed text
        """
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        return transcribe_handwriting.invoke({"image_base64": image_b64})

    def process_page(
        self,
        page: Document,
        pdf_path: Optional[str | Path] = None,
    ) -> Document:
        """
        Process a page for handwriting extraction.
        
        Args:
            page: Page document
            pdf_path: Path to source PDF (for image rendering)
            
        Returns:
            Processed document
        """
        # Check if we need vision extraction
        if not self.needs_vision_extraction(page):
            return page
        
        if not pdf_path:
            logger.warning("PDF path required for handwriting extraction")
            return page
        
        try:
            page_num = page.metadata.get("page", 0)
            
            # Render page to image
            image_bytes = render_page_to_image(pdf_path, page_num, self.dpi)
            
            # Check for handwriting
            if not self.detect_handwriting(image_bytes):
                page.metadata["quality_warning"] = "low_ocr_not_handwriting"
                return page
            
            # Transcribe
            transcribed = self.transcribe(image_bytes)
            
            return Document(
                page_content=transcribed,
                metadata={
                    **page.metadata,
                    "content_type": "handwriting",
                    "extraction_method": "vision_llm_handwriting",
                    "processed_by": "handwriting_specialist",
                    "original_ocr": page.page_content[:500],
                }
            )
            
        except Exception as e:
            logger.error(f"Handwriting extraction failed: {e}")
            page.metadata["extraction_error"] = str(e)
            return page

    def process_pages(
        self,
        pages: list[Document],
        pdf_path: str | Path,
    ) -> list[Document]:
        """
        Process multiple pages for handwriting.
        
        Args:
            pages: List of page documents
            pdf_path: Path to PDF
            
        Returns:
            Processed documents
        """
        return [self.process_page(page, pdf_path) for page in pages]

