"""Vision model integration for charts, graphs, and diagrams.

This module uses GPT-5.2 vision model to generate
text descriptions of visual content for better retrieval.
GPT-5.2 offers improved CharXiv reasoning (93.9% vs 75.7% on GPT-4o).
"""

import base64
import logging
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from app.config import settings

logger = logging.getLogger(__name__)


def get_vision_model() -> ChatOpenAI:
    """Get the configured vision model."""
    return ChatOpenAI(
        model=settings.vision_model,
        max_tokens=1024,
        api_key=settings.openai_api_key,
    )


def render_page_to_image(pdf_path: str | Path, page_num: int, dpi: int = 200) -> bytes:
    """
    Render PDF page to image bytes.
    
    Args:
        pdf_path: Path to PDF file
        page_num: Page number (0-indexed)
        dpi: Resolution for rendering
        
    Returns:
        PNG image bytes
    """
    try:
        import fitz
        
        doc = fitz.open(str(pdf_path))
        page = doc[page_num]
        
        # Higher DPI for better recognition
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        
        image_bytes = pix.tobytes("png")
        doc.close()
        
        return image_bytes
        
    except Exception as e:
        logger.error(f"Failed to render page {page_num} to image: {e}")
        raise


def classify_visual_element(image_bytes: bytes) -> str:
    """
    Classify type of visual element in image.
    
    Args:
        image_bytes: Image as bytes
        
    Returns:
        Classification: chart, graph, diagram, flowchart, photo, or other
    """
    model = get_vision_model()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Classify this image as one of: chart, graph, diagram, flowchart, photo, table, or other. Reply with just the classification word.",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            },
        ]
    )
    
    try:
        response = model.invoke([message])
        classification = response.content.strip().lower()
        
        valid_types = ["chart", "graph", "diagram", "flowchart", "photo", "table", "other"]
        if classification in valid_types:
            return classification
        
        # Try to match partial
        for vtype in valid_types:
            if vtype in classification:
                return vtype
        
        return "other"
        
    except Exception as e:
        logger.warning(f"Visual classification failed: {e}")
        return "other"


def describe_visual_element(
    image_bytes: bytes,
    element_type: str = "chart",
) -> Optional[str]:
    """
    Use vision model to describe charts, graphs, diagrams.
    
    Args:
        image_bytes: Image as bytes
        element_type: Type of visual element
        
    Returns:
        Text description of the visual element
    """
    model = ChatOpenAI(
        model=settings.vision_model,
        max_tokens=2048,
        api_key=settings.openai_api_key,
    )
    
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    
    prompts = {
        "chart": """Extract and describe ALL content from this document page. Include:

1. FIGURE/CHART IDENTIFICATION:
   - Figure number and title (e.g., "Figure 1-1: AI Progress Timeline")
   - Chart type (bar, pie, line, timeline, etc.)

2. VISUAL CONTENT DESCRIPTION:
   - All labels, axes, and legends
   - All data points, values, and categories shown
   - Timeline periods or date ranges if present
   - Any visual progression or flow

3. KEY INFORMATION:
   - Main message or conclusion the figure conveys
   - All text visible in the figure
   - Relationships between elements

4. SURROUNDING TEXT:
   - Any captions or descriptions near the figure
   - Section headers visible on the page

Be thorough - this description will be used for search and retrieval.""",
        
        "graph": """Extract and describe ALL content from this graph. Include:

1. IDENTIFICATION: Graph title, figure number, and any labels
2. AXES: What is measured on each axis, units, and scale
3. DATA: All data points, trends, and patterns visible
4. RELATIONSHIPS: What the graph shows about relationships between variables
5. KEY INSIGHTS: Main conclusions that can be drawn
6. ALL TEXT: Every piece of text visible in and around the graph

Be thorough - this description will be used for search and retrieval.""",
        
        "diagram": """Extract and describe ALL content from this diagram. Include:

1. IDENTIFICATION: Diagram title, figure number
2. COMPONENTS: All elements, boxes, shapes, and their labels
3. RELATIONSHIPS: How components connect or relate to each other
4. FLOW: Any sequence, progression, or timeline shown
5. CATEGORIES: Any groupings or classifications shown
6. ALL TEXT: Every piece of text visible in the diagram
7. KEY MESSAGE: What the diagram is trying to communicate

Be thorough - this description will be used for search and retrieval.""",
        
        "flowchart": """Extract and describe this flowchart step by step. Include:

1. TITLE: Flowchart name and figure number
2. START: Starting point(s)
3. STEPS: Each step in order with exact text
4. DECISIONS: All decision points and their conditions
5. BRANCHES: Different paths and outcomes
6. END: Final outcomes or endpoints
7. OVERALL PROCESS: Summary of what the flowchart represents

Be thorough - this description will be used for search and retrieval.""",
        
        "table": """Extract ALL data from this table. Include:

1. TABLE IDENTIFICATION: Table number, title, caption
2. HEADERS: All column and row headers exactly as shown
3. DATA: All cell values, preserving the structure
4. TOTALS: Any summary rows or calculations
5. NOTES: Any footnotes or annotations
6. CONTEXT: What the table represents

Format the data clearly so it can be searched and retrieved.""",
        
        "photo": """Describe this image in detail. Include:

1. MAIN SUBJECT: What the image shows
2. TEXT: Any visible text, labels, or captions
3. CONTEXT: Setting, background, and relevant details
4. FIGURES: If this is a figure from a document, include figure number
5. RELEVANCE: What information this image conveys

Be thorough - this description will be used for search and retrieval.""",
    }
    
    prompt = prompts.get(element_type, prompts["chart"])
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
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
        logger.error(f"Visual description failed: {e}")
        return None


def process_visual_page(
    pdf_path: str | Path,
    page_num: int,
    page_metadata: dict,
) -> Document:
    """
    Process a page containing visual content.
    
    Args:
        pdf_path: Path to PDF
        page_num: Page number
        page_metadata: Existing page metadata
        
    Returns:
        Document with visual content description
    """
    try:
        # Render page to image
        image_bytes = render_page_to_image(pdf_path, page_num)
        
        # Classify visual type
        visual_type = classify_visual_element(image_bytes)
        logger.info(f"Page {page_num} classified as: {visual_type}")
        
        # Get description
        description = describe_visual_element(image_bytes, visual_type)
        
        if description:
            content = f"[{visual_type.upper()}]\n{description}"
        else:
            content = f"[{visual_type.upper()}] Visual content on this page could not be described."
        
        return Document(
            page_content=content,
            metadata={
                **page_metadata,
                "page": page_num,
                "content_type": visual_type,
                "extraction_method": "vision_llm",
                "visual_type": visual_type,
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to process visual page {page_num}: {e}")
        return Document(
            page_content=f"[Visual content extraction failed for this page]",
            metadata={
                **page_metadata,
                "page": page_num,
                "extraction_error": True,
                "content_type": "chart",
            }
        )


class VisionProcessor:
    """
    Vision processor for handling visual document content.
    """

    def __init__(self, dpi: int = 200):
        """
        Initialize vision processor.
        
        Args:
            dpi: DPI for page rendering
        """
        self.dpi = dpi

    def process_page(
        self,
        pdf_path: str | Path,
        page_num: int,
        metadata: Optional[dict] = None,
    ) -> Document:
        """Process a single page with visual content."""
        return process_visual_page(
            pdf_path,
            page_num,
            metadata or {},
        )

    def process_pages(
        self,
        pdf_path: str | Path,
        page_nums: list[int],
        base_metadata: Optional[dict] = None,
    ) -> list[Document]:
        """Process multiple pages with visual content."""
        base_metadata = base_metadata or {"source": str(pdf_path)}
        
        documents = []
        for page_num in page_nums:
            doc = self.process_page(pdf_path, page_num, base_metadata)
            documents.append(doc)
        
        return documents

