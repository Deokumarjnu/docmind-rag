"""Content-aware chunking strategies for different document types.

This module implements intelligent chunking that adapts to content type:
- Tables are kept intact (no splitting)
- Text uses semantic chunking with overlap
- Code uses AST-based or logical unit chunking
- OCR text uses smaller chunks due to noise
"""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.ingestion.page_classifier import PageType

logger = logging.getLogger(__name__)


# Text splitter configurations for different content types
TEXT_SPLITTER_NORMAL = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)

TEXT_SPLITTER_SMALL = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)

TEXT_SPLITTER_CODE = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\n\n\n", "\n\n", "\n", " ", ""],
    length_function=len,
)


def chunk_document(doc: Document) -> list[Document]:
    """
    Chunk a document based on its content type.
    
    Args:
        doc: Document to chunk
        
    Returns:
        List of chunked documents
    """
    content_type = doc.metadata.get("content_type", "text")
    
    # Route to appropriate chunker
    if content_type == PageType.TABLE.value or content_type == "table":
        return chunk_table(doc)
    elif content_type == PageType.CODE.value or content_type == "code":
        return chunk_code(doc)
    elif content_type == PageType.IMAGE.value or content_type == "image":
        return chunk_ocr_text(doc)
    elif content_type == PageType.HANDWRITING.value or content_type == "handwriting":
        return chunk_ocr_text(doc)
    else:
        return chunk_text(doc)


def chunk_table(doc: Document) -> list[Document]:
    """
    Keep tables intact without splitting.
    
    Tables lose meaning when split, so we preserve them as single chunks.
    """
    # Tables should not be split
    doc.metadata["chunk_method"] = "table_intact"
    return [doc]


def chunk_text(doc: Document) -> list[Document]:
    """
    Chunk regular text with semantic splitting.
    
    Uses RecursiveCharacterTextSplitter with standard chunk size.
    """
    if len(doc.page_content.strip()) < 50:
        doc.metadata["chunk_method"] = "text_small"
        return [doc]
    
    chunks = TEXT_SPLITTER_NORMAL.split_documents([doc])
    
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_method"] = "text_recursive"
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)
    
    return chunks


def chunk_code(doc: Document) -> list[Document]:
    """
    Chunk code content while preserving logical units.
    
    Uses larger chunk sizes to keep functions/classes together.
    """
    if len(doc.page_content.strip()) < 100:
        doc.metadata["chunk_method"] = "code_small"
        return [doc]
    
    chunks = TEXT_SPLITTER_CODE.split_documents([doc])
    
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_method"] = "code_recursive"
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)
    
    return chunks


def chunk_ocr_text(doc: Document) -> list[Document]:
    """
    Chunk OCR text with smaller sizes due to noise.
    
    OCR text is often noisier, so smaller chunks help isolate errors.
    """
    if len(doc.page_content.strip()) < 30:
        doc.metadata["chunk_method"] = "ocr_small"
        return [doc]
    
    chunks = TEXT_SPLITTER_SMALL.split_documents([doc])
    
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_method"] = "ocr_recursive"
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)
    
    return chunks


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Chunk a list of documents based on their content types.
    
    Args:
        documents: List of documents to chunk
        
    Returns:
        List of chunked documents
    """
    all_chunks = []
    
    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
    
    logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks")
    return all_chunks


class ContentAwareChunker:
    """
    Content-aware chunking processor.
    
    Configurable chunker that adapts to different content types
    and allows customization of chunk sizes.
    """

    def __init__(
        self,
        text_chunk_size: int = 700,
        text_chunk_overlap: int = 100,
        code_chunk_size: int = 1500,
        code_chunk_overlap: int = 200,
        ocr_chunk_size: int = 400,
        ocr_chunk_overlap: int = 50,
    ):
        """
        Initialize the content-aware chunker.
        
        Args:
            text_chunk_size: Chunk size for regular text
            text_chunk_overlap: Overlap for text chunks
            code_chunk_size: Chunk size for code
            code_chunk_overlap: Overlap for code chunks
            ocr_chunk_size: Chunk size for OCR text
            ocr_chunk_overlap: Overlap for OCR chunks
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=text_chunk_size,
            chunk_overlap=text_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        
        self.code_splitter = RecursiveCharacterTextSplitter(
            chunk_size=code_chunk_size,
            chunk_overlap=code_chunk_overlap,
            separators=["\n\n\n", "\n\n", "\n", " ", ""],
        )
        
        self.ocr_splitter = RecursiveCharacterTextSplitter(
            chunk_size=ocr_chunk_size,
            chunk_overlap=ocr_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, doc: Document) -> list[Document]:
        """
        Chunk a document based on content type.
        
        Args:
            doc: Document to chunk
            
        Returns:
            List of chunked documents
        """
        content_type = doc.metadata.get("content_type", "text")
        
        if content_type in ["table", PageType.TABLE.value]:
            return [doc]
        
        elif content_type in ["code", PageType.CODE.value]:
            if len(doc.page_content.strip()) < 100:
                return [doc]
            chunks = self.code_splitter.split_documents([doc])
            for i, c in enumerate(chunks):
                c.metadata["chunk_index"] = i
            return chunks
        
        elif content_type in ["image", "handwriting", "ocr", 
                              PageType.IMAGE.value, PageType.HANDWRITING.value]:
            if len(doc.page_content.strip()) < 30:
                return [doc]
            chunks = self.ocr_splitter.split_documents([doc])
            for i, c in enumerate(chunks):
                c.metadata["chunk_index"] = i
            return chunks
        
        else:
            if len(doc.page_content.strip()) < 50:
                return [doc]
            chunks = self.text_splitter.split_documents([doc])
            for i, c in enumerate(chunks):
                c.metadata["chunk_index"] = i
            return chunks

    def chunk_all(self, documents: list[Document]) -> list[Document]:
        """
        Chunk all documents in a list.
        
        Args:
            documents: List of documents
            
        Returns:
            List of all chunks
        """
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk(doc))
        return all_chunks

