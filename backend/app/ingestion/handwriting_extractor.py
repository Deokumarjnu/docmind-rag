"""Vision LLM-based handwriting extraction.

This module uses GPT-5.2 vision capabilities to accurately transcribe
handwritten text, which is far superior to traditional OCR.
"""

import base64
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from app.config import settings
from app.ingestion.vision_processor import render_page_to_image

logger = logging.getLogger(__name__)


def get_vision_model() -> ChatOpenAI:
    """Get the configured vision model."""
    return ChatOpenAI(
        model=settings.vision_model,
        max_tokens=4096,
        api_key=settings.openai_api_key,
    )


def get_detection_model() -> ChatOpenAI:
    """Get a faster model for detection tasks."""
    return ChatOpenAI(
        model=settings.fast_llm_model,
        max_tokens=50,
        api_key=settings.openai_api_key,
    )


def detect_handwritten_content(image_bytes: bytes) -> bool:
    """
    Detect if image contains handwriting using vision model.
    
    Args:
        image_bytes: Image as bytes
        
    Returns:
        True if handwriting detected
    """
    model = get_detection_model()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Does this image contain handwritten text (not printed/typed)? Reply with only 'yes' or 'no'.",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            },
        ]
    )
    
    try:
        response = model.invoke([message])
        return "yes" in response.content.lower()
    except Exception as e:
        logger.warning(f"Handwriting detection failed: {e}")
        return False


def extract_handwritten_text(image_bytes: bytes) -> Optional[str]:
    """
    Use GPT-4o to transcribe handwritten text.
    
    This is far superior to traditional OCR for handwriting.
    
    Args:
        image_bytes: Image as bytes
        
    Returns:
        Transcribed text or None on failure
    """
    model = get_vision_model()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": """Carefully transcribe ALL handwritten text in this image.

Instructions:
1. Preserve the original structure, paragraphs, and formatting
2. If text is unclear, provide your best interpretation with [unclear: ...] notation
3. Include any lists, bullet points, or numbered items
4. Note any diagrams, drawings, or symbols as [diagram: description]
5. Maintain line breaks where they appear meaningful
6. If there are annotations or margin notes, include them with [margin: ...]

Transcribe the handwritten content:""",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            },
        ]
    )
    
    try:
        response = model.invoke([message])
        return response.content
    except Exception as e:
        logger.error(f"Handwriting extraction failed: {e}")
        return None


def is_low_quality_ocr(text: str) -> bool:
    """
    Detect if OCR quality is poor (likely handwriting or scan issues).
    
    Args:
        text: OCR output text
        
    Returns:
        True if quality appears poor
    """
    if not text or len(text.strip()) < 10:
        return True
    
    # High ratio of non-alphanumeric characters = OCR failure
    alnum_chars = sum(1 for c in text if c.isalnum() or c.isspace())
    if alnum_chars / max(len(text), 1) < 0.7:
        return True
    
    # Many very short "words" = OCR gibberish
    words = text.split()
    if words:
        short_word_ratio = sum(1 for w in words if len(w) <= 2) / len(words)
        if short_word_ratio > 0.5:
            return True
    
    # Check for repetitive character patterns
    if has_repetitive_pattern(text):
        return True
    
    # Low average word length
    if words:
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 3:
            return True
    
    return False


def has_repetitive_pattern(text: str, threshold: float = 0.4) -> bool:
    """
    Detect repetitive characters indicating OCR failure.
    
    Args:
        text: Text to analyze
        threshold: Frequency threshold
        
    Returns:
        True if repetitive pattern detected
    """
    if len(text) < 20:
        return False
    
    char_counts = Counter(text.lower().replace(' ', ''))
    if not char_counts:
        return False
    
    most_common_ratio = char_counts.most_common(1)[0][1] / len(text.replace(' ', ''))
    return most_common_ratio > threshold


def extract_page_with_handwriting_support(
    page: Document,
    pdf_path: str | Path,
) -> Document:
    """
    Hybrid approach: OCR first, fall back to vision LLM for handwriting.
    
    Args:
        page: Page document with initial OCR text
        pdf_path: Path to PDF file
        
    Returns:
        Document with best available text extraction
    """
    page_num = page.metadata.get("page", 0)
    ocr_text = page.page_content
    
    # Check OCR quality
    if is_low_quality_ocr(ocr_text):
        logger.info(f"Low quality OCR detected on page {page_num}, checking for handwriting")
        
        try:
            # Render page to image
            page_image = render_page_to_image(pdf_path, page_num)
            
            # Check if it contains handwriting
            if detect_handwritten_content(page_image):
                logger.info(f"Handwriting detected on page {page_num}, using vision LLM")
                
                # Extract with vision LLM
                vision_text = extract_handwritten_text(page_image)
                
                if vision_text:
                    return Document(
                        page_content=vision_text,
                        metadata={
                            **page.metadata,
                            "extraction_method": "vision_llm_handwriting",
                            "ocr_text": ocr_text[:500] if ocr_text else None,
                            "content_type": "handwriting",
                        }
                    )
            else:
                # Low quality but not handwriting - scan issue
                logger.warning(f"Low quality content on page {page_num}, not handwriting")
                page.metadata["quality_warning"] = "low_ocr_quality"
                
        except Exception as e:
            logger.error(f"Handwriting extraction failed for page {page_num}: {e}")
    
    return page


def batch_process_with_handwriting(
    pages: list[Document],
    pdf_path: str | Path,
    parallel: bool = True,
) -> list[Document]:
    """
    Process multiple pages with handwriting detection.
    
    Args:
        pages: List of page documents
        pdf_path: Path to PDF
        parallel: Whether to process in parallel
        
    Returns:
        List of processed documents
    """
    from concurrent.futures import ThreadPoolExecutor
    
    def process_page(page: Document) -> Document:
        return extract_page_with_handwriting_support(page, pdf_path)
    
    if parallel:
        # Limit workers for API rate limits
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_page, pages))
    else:
        results = [process_page(page) for page in pages]
    
    return results


class HandwritingExtractor:
    """
    Handwriting extractor with vision LLM support.
    """

    def __init__(self, dpi: int = 200):
        """
        Initialize extractor.
        
        Args:
            dpi: DPI for page rendering
        """
        self.dpi = dpi

    def detect(self, pdf_path: str | Path, page_num: int) -> bool:
        """Detect if page contains handwriting."""
        try:
            image_bytes = render_page_to_image(pdf_path, page_num, self.dpi)
            return detect_handwritten_content(image_bytes)
        except Exception:
            return False

    def extract(
        self,
        pdf_path: str | Path,
        page_num: int,
        metadata: Optional[dict] = None,
    ) -> Document:
        """Extract handwritten text from page."""
        metadata = metadata or {"source": str(pdf_path), "page": page_num}
        
        try:
            image_bytes = render_page_to_image(pdf_path, page_num, self.dpi)
            text = extract_handwritten_text(image_bytes)
            
            if text:
                return Document(
                    page_content=text,
                    metadata={
                        **metadata,
                        "extraction_method": "vision_llm_handwriting",
                        "content_type": "handwriting",
                    }
                )
        except Exception as e:
            logger.error(f"Handwriting extraction failed: {e}")
        
        return Document(
            page_content="[Handwriting extraction failed for this page]",
            metadata={
                **metadata,
                "extraction_error": True,
                "content_type": "handwriting",
            }
        )

