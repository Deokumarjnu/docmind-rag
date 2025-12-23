"""Tools for visual content analysis.

These tools enable the chart and handwriting specialists
to analyze and describe visual content.
"""

import base64
import logging
from typing import Optional

from langchain_core.tools import tool

from app.ingestion.vision_processor import (
    classify_visual_element,
    describe_visual_element,
)
from app.ingestion.handwriting_extractor import (
    detect_handwritten_content,
    extract_handwritten_text,
)

logger = logging.getLogger(__name__)


@tool
def classify_visual_type(image_base64: str) -> str:
    """
    Classify the type of visual element in an image.
    
    Args:
        image_base64: Base64-encoded image
        
    Returns:
        Classification: chart, graph, diagram, flowchart, photo, table, or other
    """
    try:
        image_bytes = base64.b64decode(image_base64)
        return classify_visual_element(image_bytes)
    except Exception as e:
        logger.error(f"Visual classification failed: {e}")
        return "other"


@tool
def describe_chart(image_base64: str) -> str:
    """
    Generate a detailed description of a chart or graph.
    
    Args:
        image_base64: Base64-encoded image of chart
        
    Returns:
        Text description of the chart
    """
    try:
        image_bytes = base64.b64decode(image_base64)
        description = describe_visual_element(image_bytes, "chart")
        return description or "Unable to describe chart"
    except Exception as e:
        logger.error(f"Chart description failed: {e}")
        return f"Chart description failed: {e}"


@tool
def extract_data_points(image_base64: str) -> dict:
    """
    Extract key data points from a chart or graph.
    
    Args:
        image_base64: Base64-encoded image
        
    Returns:
        Dictionary with extracted data points
    """
    try:
        image_bytes = base64.b64decode(image_base64)
        
        # Get description which includes data points
        description = describe_visual_element(image_bytes, "chart")
        
        return {
            "description": description,
            "extraction_method": "vision_llm",
        }
    except Exception as e:
        logger.error(f"Data extraction failed: {e}")
        return {"error": str(e)}


@tool
def describe_diagram_flow(image_base64: str) -> str:
    """
    Describe a diagram or flowchart step by step.
    
    Args:
        image_base64: Base64-encoded image
        
    Returns:
        Text description of the diagram/flowchart
    """
    try:
        image_bytes = base64.b64decode(image_base64)
        description = describe_visual_element(image_bytes, "flowchart")
        return description or "Unable to describe diagram"
    except Exception as e:
        logger.error(f"Diagram description failed: {e}")
        return f"Diagram description failed: {e}"


@tool
def detect_handwritten_regions(image_base64: str) -> bool:
    """
    Detect if an image contains handwritten text.
    
    Args:
        image_base64: Base64-encoded image
        
    Returns:
        True if handwriting detected
    """
    try:
        image_bytes = base64.b64decode(image_base64)
        return detect_handwritten_content(image_bytes)
    except Exception as e:
        logger.error(f"Handwriting detection failed: {e}")
        return False


@tool
def transcribe_handwriting(image_base64: str) -> str:
    """
    Transcribe handwritten text from an image.
    
    Args:
        image_base64: Base64-encoded image
        
    Returns:
        Transcribed text
    """
    try:
        image_bytes = base64.b64decode(image_base64)
        transcription = extract_handwritten_text(image_bytes)
        return transcription or "Unable to transcribe handwriting"
    except Exception as e:
        logger.error(f"Handwriting transcription failed: {e}")
        return f"Transcription failed: {e}"

