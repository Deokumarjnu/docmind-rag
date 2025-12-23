"""Adaptive document extractor that routes pages to optimal extraction methods.

This module implements dynamic extraction based on page classification,
using different strategies for text, tables, images, and mixed content.
"""

import logging
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from app.ingestion.page_classifier import PageType, classify_page, get_extraction_strategy

logger = logging.getLogger(__name__)


class AdaptiveExtractor:
    """
    Adaptive document extractor that chooses optimal extraction method per page.
    
    Supports:
    - PyMuPDF for fast text extraction
    - Unstructured with elements mode for tables
    - Unstructured with OCR for image/scanned pages
    - Unstructured auto for mixed content
    """

    def __init__(self):
        """Initialize the adaptive extractor."""
        self._pymupdf_available = self._check_pymupdf()
        self._unstructured_available = self._check_unstructured()

    def _check_pymupdf(self) -> bool:
        """Check if PyMuPDF is available."""
        try:
            import fitz
            return True
        except ImportError:
            logger.warning("PyMuPDF not available, falling back to alternatives")
            return False

    def _check_unstructured(self) -> bool:
        """Check if Unstructured is available."""
        try:
            from unstructured.partition.pdf import partition_pdf
            return True
        except ImportError:
            logger.warning("Unstructured not available for advanced extraction")
            return False

    def load_pdf_pages(self, pdf_path: str | Path) -> list[Document]:
        """
        Load PDF pages using PyMuPDF for initial text extraction.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects, one per page
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        try:
            from langchain_community.document_loaders import PyMuPDFLoader
            loader = PyMuPDFLoader(str(pdf_path))
            return loader.load()
        except Exception as e:
            logger.error(f"Failed to load PDF with PyMuPDF: {e}")
            raise

    def extract_page(
        self,
        page: Document,
        pdf_path: str | Path,
        page_type: Optional[PageType] = None,
    ) -> list[Document]:
        """
        Extract content from a page using the optimal strategy.
        
        Args:
            page: Document object with initial page content
            pdf_path: Path to the source PDF
            page_type: Optional pre-classified page type
            
        Returns:
            List of Document objects with extracted content
        """
        pdf_path = Path(pdf_path)
        
        # Classify if not provided
        if page_type is None:
            page_type = classify_page(page)
        
        page_num = page.metadata.get("page", 0)
        strategy = get_extraction_strategy(page_type)
        
        logger.debug(f"Page {page_num}: type={page_type.value}, strategy={strategy}")
        
        # Route to appropriate extractor
        if strategy == "pymupdf":
            return self._extract_with_pymupdf(page, page_type)
        elif strategy == "unstructured_elements":
            return self._extract_with_unstructured_elements(pdf_path, page_num, page_type)
        elif strategy == "ocr":
            return self._extract_with_ocr(pdf_path, page_num, page_type)
        elif strategy == "unstructured_auto":
            return self._extract_with_unstructured_auto(pdf_path, page_num, page_type)
        elif strategy == "vision_llm":
            # Use vision LLM for handwriting extraction
            return self._extract_with_vision_llm(page, pdf_path, page_type)
        else:
            logger.warning(f"Unknown strategy {strategy}, falling back to pymupdf")
            return self._extract_with_pymupdf(page, page_type)

    def _extract_with_pymupdf(
        self,
        page: Document,
        page_type: PageType,
    ) -> list[Document]:
        """
        Use existing PyMuPDF extraction (already done during load).
        
        Simply annotates the page with extraction metadata.
        """
        page.metadata["extraction_method"] = "pymupdf"
        page.metadata["content_type"] = page_type.value
        return [page]

    def _extract_with_unstructured_elements(
        self,
        pdf_path: Path,
        page_num: int,
        page_type: PageType,
    ) -> list[Document]:
        """
        Extract with Unstructured in elements mode for structured content.
        
        Best for tables and structured layouts.
        """
        if not self._unstructured_available:
            logger.warning("Unstructured not available, using PyMuPDF fallback")
            return self._fallback_extraction(pdf_path, page_num, page_type)

        try:
            from langchain_community.document_loaders import UnstructuredPDFLoader
            
            loader = UnstructuredPDFLoader(
                str(pdf_path),
                mode="elements",
                strategy="hi_res",
            )
            all_elements = loader.load()
            
            # Filter to target page
            page_elements = [
                doc for doc in all_elements
                if doc.metadata.get("page_number", 0) == page_num + 1
            ]
            
            # Add metadata
            for doc in page_elements:
                doc.metadata["extraction_method"] = "unstructured_elements"
                doc.metadata["content_type"] = page_type.value
                doc.metadata["page"] = page_num
            
            return page_elements if page_elements else self._fallback_extraction(pdf_path, page_num, page_type)
            
        except Exception as e:
            logger.error(f"Unstructured elements extraction failed: {e}")
            return self._fallback_extraction(pdf_path, page_num, page_type)

    def _extract_with_ocr(
        self,
        pdf_path: Path,
        page_num: int,
        page_type: PageType,
    ) -> list[Document]:
        """
        Extract with OCR for image-based or scanned pages.
        """
        if not self._unstructured_available:
            logger.warning("Unstructured not available for OCR, using PyMuPDF fallback")
            return self._fallback_extraction(pdf_path, page_num, page_type)

        try:
            from langchain_community.document_loaders import UnstructuredPDFLoader
            
            loader = UnstructuredPDFLoader(
                str(pdf_path),
                mode="single",
                strategy="ocr_only",
            )
            all_docs = loader.load()
            
            # Try to find content for target page
            for doc in all_docs:
                if doc.metadata.get("page_number", 0) == page_num + 1:
                    doc.metadata["extraction_method"] = "ocr"
                    doc.metadata["content_type"] = page_type.value
                    doc.metadata["page"] = page_num
                    return [doc]
            
            # If no page match, return with warning
            if all_docs:
                doc = all_docs[0]
                doc.metadata["extraction_method"] = "ocr"
                doc.metadata["content_type"] = page_type.value
                doc.metadata["page"] = page_num
                doc.metadata["warning"] = "page_extraction_uncertain"
                return [doc]
            
            return self._fallback_extraction(pdf_path, page_num, page_type)
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return self._fallback_extraction(pdf_path, page_num, page_type)

    def _extract_with_unstructured_auto(
        self,
        pdf_path: Path,
        page_num: int,
        page_type: PageType,
    ) -> list[Document]:
        """
        Extract with Unstructured auto strategy for mixed content.
        """
        if not self._unstructured_available:
            logger.warning("Unstructured not available, using PyMuPDF fallback")
            return self._fallback_extraction(pdf_path, page_num, page_type)

        try:
            from langchain_community.document_loaders import UnstructuredPDFLoader
            
            loader = UnstructuredPDFLoader(
                str(pdf_path),
                mode="elements",
                strategy="auto",
            )
            all_elements = loader.load()
            
            # Filter to target page
            page_elements = [
                doc for doc in all_elements
                if doc.metadata.get("page_number", 0) == page_num + 1
            ]
            
            # Add metadata
            for doc in page_elements:
                doc.metadata["extraction_method"] = "unstructured_auto"
                doc.metadata["content_type"] = page_type.value
                doc.metadata["page"] = page_num
            
            return page_elements if page_elements else self._fallback_extraction(pdf_path, page_num, page_type)
            
        except Exception as e:
            logger.error(f"Unstructured auto extraction failed: {e}")
            return self._fallback_extraction(pdf_path, page_num, page_type)

    def _extract_with_vision_llm(
        self,
        page: Document,
        pdf_path: Path,
        page_type: PageType,
    ) -> list[Document]:
        """
        Extract with vision LLM for handwriting or low-quality OCR pages.
        """
        try:
            from app.ingestion.handwriting_extractor import extract_page_with_handwriting_support
            
            result = extract_page_with_handwriting_support(page, pdf_path)
            result.metadata["extraction_method"] = "vision_llm"
            result.metadata["content_type"] = page_type.value
            return [result]
            
        except Exception as e:
            logger.error(f"Vision LLM extraction failed: {e}")
            return self._extract_with_pymupdf(page, page_type)

    def _fallback_extraction(
        self,
        pdf_path: Path,
        page_num: int,
        page_type: PageType,
    ) -> list[Document]:
        """
        Fallback extraction using basic PyMuPDF.
        """
        try:
            import fitz
            
            doc = fitz.open(str(pdf_path))
            page = doc[page_num]
            text = page.get_text()
            
            return [Document(
                page_content=text,
                metadata={
                    "source": str(pdf_path),
                    "page": page_num,
                    "extraction_method": "pymupdf_fallback",
                    "content_type": page_type.value,
                }
            )]
            
        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
            return [Document(
                page_content="[Content extraction failed for this page]",
                metadata={
                    "source": str(pdf_path),
                    "page": page_num,
                    "extraction_method": "failed",
                    "content_type": page_type.value,
                    "extraction_error": True,
                }
            )]


def process_pdf_adaptively(pdf_path: str | Path) -> list[Document]:
    """
    Process a PDF with adaptive extraction based on page content.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of Document objects with content from all pages
    """
    extractor = AdaptiveExtractor()
    
    # Load pages with PyMuPDF for initial classification
    pages = extractor.load_pdf_pages(pdf_path)
    
    all_documents = []
    for page in pages:
        # Classify page
        page_type = classify_page(page)
        
        # Extract with optimal strategy
        extracted = extractor.extract_page(page, pdf_path, page_type)
        all_documents.extend(extracted)
    
    return all_documents

