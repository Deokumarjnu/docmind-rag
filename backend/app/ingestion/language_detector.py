"""Multi-language detection and handling.

This module detects document languages and provides appropriate
embedding models and text direction handling.
"""

import logging
from typing import Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def detect_document_language(
    pages: list[Document],
    sample_size: int = 10,
) -> list[tuple[str, float]]:
    """
    Detect primary language(s) of document.
    
    Args:
        pages: List of page documents
        sample_size: Number of pages to sample
        
    Returns:
        List of (language_code, probability) tuples
    """
    try:
        from langdetect import detect_langs
        
        # Sample text from pages
        sample_text = " ".join([
            page.page_content[:500]
            for page in pages[:sample_size]
        ])
        
        if len(sample_text.strip()) < 20:
            return [("en", 1.0)]
        
        languages = detect_langs(sample_text)
        return [(lang.lang, lang.prob) for lang in languages]
        
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        return [("en", 1.0)]


def detect_page_language(page: Document) -> str:
    """
    Detect language of a single page.
    
    Args:
        page: Page document
        
    Returns:
        Language code (e.g., 'en', 'zh', 'ar')
    """
    try:
        from langdetect import detect
        
        text = page.page_content
        if len(text.strip()) < 20:
            return "en"
        
        return detect(text)
        
    except Exception:
        return "en"


def get_embedding_model_for_language(lang_code: str) -> str:
    """
    Return appropriate embedding model based on language.
    
    Args:
        lang_code: ISO language code
        
    Returns:
        Embedding model name
    """
    # OpenAI's text-embedding-3-large has good multilingual support
    multilingual_models = {
        "default": "text-embedding-3-large",
        "zh": "text-embedding-3-large",  # Chinese
        "ja": "text-embedding-3-large",  # Japanese
        "ko": "text-embedding-3-large",  # Korean
        "ar": "text-embedding-3-large",  # Arabic (RTL)
        "he": "text-embedding-3-large",  # Hebrew (RTL)
        "fa": "text-embedding-3-large",  # Persian/Farsi (RTL)
        "ur": "text-embedding-3-large",  # Urdu (RTL)
        "ru": "text-embedding-3-large",  # Russian
        "de": "text-embedding-3-large",  # German
        "fr": "text-embedding-3-large",  # French
        "es": "text-embedding-3-large",  # Spanish
        "pt": "text-embedding-3-large",  # Portuguese
        "hi": "text-embedding-3-large",  # Hindi
    }
    
    return multilingual_models.get(lang_code, multilingual_models["default"])


def handle_rtl_languages(
    text: str,
    lang_code: str,
) -> tuple[str, dict]:
    """
    Handle right-to-left languages.
    
    Args:
        text: Document text
        lang_code: Language code
        
    Returns:
        Tuple of (text, metadata_dict)
    """
    rtl_languages = ["ar", "he", "fa", "ur", "yi", "ps"]
    
    if lang_code in rtl_languages:
        return text, {
            "text_direction": "rtl",
            "language": lang_code,
        }
    
    return text, {
        "text_direction": "ltr",
        "language": lang_code,
    }


def annotate_documents_with_language(
    documents: list[Document],
) -> list[Document]:
    """
    Add language metadata to documents.
    
    Args:
        documents: List of documents
        
    Returns:
        Documents with language metadata
    """
    # Detect primary language from sample
    languages = detect_document_language(documents)
    primary_lang = languages[0][0] if languages else "en"
    
    logger.info(f"Detected primary language: {primary_lang}")
    
    rtl_languages = ["ar", "he", "fa", "ur", "yi", "ps"]
    
    annotated = []
    for doc in documents:
        # Optionally detect per-page language for multilingual docs
        page_lang = primary_lang
        
        _, lang_metadata = handle_rtl_languages(doc.page_content, page_lang)
        
        annotated.append(Document(
            page_content=doc.page_content,
            metadata={
                **doc.metadata,
                **lang_metadata,
                "primary_language": primary_lang,
            }
        ))
    
    return annotated


class LanguageDetector:
    """
    Language detector for document processing.
    """

    def __init__(self):
        """Initialize language detector."""
        self.primary_language: Optional[str] = None
        self.detected_languages: list[tuple[str, float]] = []

    def detect(self, documents: list[Document]) -> str:
        """
        Detect primary language from documents.
        
        Args:
            documents: List of documents
            
        Returns:
            Primary language code
        """
        self.detected_languages = detect_document_language(documents)
        self.primary_language = (
            self.detected_languages[0][0] 
            if self.detected_languages 
            else "en"
        )
        return self.primary_language

    def get_embedding_model(self) -> str:
        """Get appropriate embedding model for detected language."""
        lang = self.primary_language or "en"
        return get_embedding_model_for_language(lang)

    def is_rtl(self) -> bool:
        """Check if primary language is RTL."""
        rtl_languages = ["ar", "he", "fa", "ur", "yi", "ps"]
        return self.primary_language in rtl_languages

    def annotate(self, documents: list[Document]) -> list[Document]:
        """Add language metadata to documents."""
        if not self.primary_language:
            self.detect(documents)
        return annotate_documents_with_language(documents)

