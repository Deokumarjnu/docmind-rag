"""Chart/Diagram Specialist subagent for visual content analysis.

This agent specializes in analyzing and describing charts, graphs,
diagrams, and flowcharts using vision models.
"""

import base64
import logging
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.config import settings
from app.agents.tools.vision_tools import (
    classify_visual_type,
    describe_chart,
    extract_data_points,
    describe_diagram_flow,
)
from app.ingestion.vision_processor import render_page_to_image

logger = logging.getLogger(__name__)

CHART_SPECIALIST_PROMPT = """You are a Visual Content Specialist.

Your job is to:
1. Classify visual content (chart, graph, diagram, flowchart)
2. Generate detailed text descriptions
3. Extract key data points and trends from charts
4. Describe relationships and flows in diagrams
5. Make visual content searchable via text

When analyzing visuals:
- Describe what the visual represents
- Extract specific data points when possible
- Note trends, patterns, and relationships
- Make the content understandable without seeing the image"""


class ChartSpecialist:
    """
    Chart/Diagram Specialist agent for processing visual content.
    """

    def __init__(self, model_name: Optional[str] = None, dpi: int = 200):
        """
        Initialize chart specialist.
        
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
            classify_visual_type,
            describe_chart,
            extract_data_points,
            describe_diagram_flow,
        ]

    def classify(self, image_bytes: bytes) -> str:
        """
        Classify the type of visual content.
        
        Args:
            image_bytes: Image bytes
            
        Returns:
            Visual type classification
        """
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        return classify_visual_type.invoke({"image_base64": image_b64})

    def describe(self, image_bytes: bytes, visual_type: str) -> str:
        """
        Generate description of visual content.
        
        Args:
            image_bytes: Image bytes
            visual_type: Type of visual
            
        Returns:
            Text description
        """
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        
        if visual_type in ["chart", "graph"]:
            return describe_chart.invoke({"image_base64": image_b64})
        elif visual_type in ["diagram", "flowchart"]:
            return describe_diagram_flow.invoke({"image_base64": image_b64})
        else:
            return describe_chart.invoke({"image_base64": image_b64})

    def process_page(
        self,
        page: Document,
        pdf_path: str | Path,
    ) -> Document:
        """
        Process a page containing visual content.
        
        Args:
            page: Page document
            pdf_path: Path to source PDF
            
        Returns:
            Processed document with visual description
        """
        try:
            page_num = page.metadata.get("page", 0)
            
            # Render page to image
            image_bytes = render_page_to_image(pdf_path, page_num, self.dpi)
            
            # Classify visual type
            visual_type = self.classify(image_bytes)
            logger.info(f"Page {page_num} classified as: {visual_type}")
            
            # Generate description
            description = self.describe(image_bytes, visual_type)
            
            content = f"[{visual_type.upper()}]\n{description}"
            
            return Document(
                page_content=content,
                metadata={
                    **page.metadata,
                    "content_type": visual_type,
                    "visual_type": visual_type,
                    "extraction_method": "vision_llm",
                    "processed_by": "chart_specialist",
                }
            )
            
        except Exception as e:
            logger.error(f"Visual processing failed for page: {e}")
            return Document(
                page_content="[Visual content could not be processed]",
                metadata={
                    **page.metadata,
                    "extraction_error": str(e),
                    "content_type": "chart",
                }
            )

    def process_pages(
        self,
        pages: list[Document],
        pdf_path: str | Path,
    ) -> list[Document]:
        """
        Process multiple pages for visual content.
        
        Args:
            pages: List of page documents
            pdf_path: Path to PDF
            
        Returns:
            Processed documents
        """
        return [self.process_page(page, pdf_path) for page in pages]

