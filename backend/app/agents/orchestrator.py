"""Deep Agent Orchestrator for document processing coordination.

This is the main orchestrator that coordinates specialized subagents
for intelligent document processing based on content type.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.config import settings
from app.ingestion.page_classifier import PageType, classify_page
from app.ingestion.adaptive_extractor import AdaptiveExtractor
from app.agents.table_specialist import TableSpecialist
from app.agents.code_specialist import CodeSpecialist
from app.agents.handwriting_specialist import HandwritingSpecialist
from app.agents.chart_specialist import ChartSpecialist
from app.agents.text_specialist import TextSpecialist

logger = logging.getLogger(__name__)

ORCHESTRATOR_PROMPT = """You are the Document Processing Orchestrator.

For each page in a document:
1. Analyze the content type (text, table, code, handwriting, chart, mixed)
2. Delegate to the appropriate specialist subagent
3. For mixed pages, coordinate multiple subagents
4. Aggregate results and ensure consistency
5. Track processing state for resume capability

Your goal is efficient, accurate document processing that preserves
all content types while making them searchable."""


class DocumentProcessingOrchestrator:
    """
    Deep Agent orchestrator for document processing.
    
    Coordinates specialized subagents based on content classification:
    - Table Specialist: Tables and structured data
    - Code Specialist: Programming code
    - Handwriting Specialist: Handwritten text
    - Chart Specialist: Charts, graphs, diagrams
    - Text Specialist: Plain text content
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_workers: int = 4,
    ):
        """
        Initialize orchestrator with subagents.
        
        Args:
            model_name: LLM model for orchestration
            max_workers: Maximum parallel workers
        """
        self.model = ChatOpenAI(
            model=model_name or settings.llm_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )
        self.max_workers = max_workers
        
        # Initialize subagents
        self.table_specialist = TableSpecialist()
        self.code_specialist = CodeSpecialist()
        self.handwriting_specialist = HandwritingSpecialist()
        self.chart_specialist = ChartSpecialist()
        self.text_specialist = TextSpecialist()
        
        # Extractor for loading pages
        self.extractor = AdaptiveExtractor()
        
        # Processing state for resume capability
        self.processing_state: dict = {}

    def classify_and_route(self, page: Document) -> dict:
        """
        Classify page content and determine routing.
        
        Args:
            page: Page document
            
        Returns:
            Routing decision with content types and tasks
        """
        page_type = classify_page(page)
        
        routing = {
            "page": page.metadata.get("page", 0),
            "content_types": [],
            "primary_type": page_type.value,
            "agents": [],
        }
        
        if page_type == PageType.TABLE:
            routing["content_types"].append("table")
            routing["agents"].append("table_specialist")
        
        elif page_type == PageType.CODE:
            routing["content_types"].append("code")
            routing["agents"].append("code_specialist")
        
        elif page_type == PageType.IMAGE:
            # Image could be chart or handwriting
            routing["content_types"].append("visual")
            routing["agents"].append("chart_specialist")
        
        elif page_type == PageType.CHART:
            routing["content_types"].append("chart")
            routing["agents"].append("chart_specialist")
        
        elif page_type == PageType.HANDWRITING:
            routing["content_types"].append("handwriting")
            routing["agents"].append("handwriting_specialist")
        
        elif page_type == PageType.MIXED:
            # Mixed content - may need multiple specialists
            routing["content_types"].append("mixed")
            routing["agents"].append("text_specialist")
            # Check for specific content types
            if self.code_specialist.has_code(page.page_content):
                routing["agents"].append("code_specialist")
        
        else:  # TEXT
            routing["content_types"].append("text")
            routing["agents"].append("text_specialist")
        
        return routing

    def process_page(
        self,
        page: Document,
        pdf_path: Path,
        routing: Optional[dict] = None,
    ) -> list[Document]:
        """
        Process a single page using appropriate specialist.
        
        Args:
            page: Page document
            pdf_path: Path to source PDF
            routing: Optional pre-computed routing
            
        Returns:
            List of processed documents
        """
        if routing is None:
            routing = self.classify_and_route(page)
        
        agents = routing.get("agents", ["text_specialist"])
        page_num = page.metadata.get("page", 0)
        
        results = []
        
        for agent_name in agents:
            try:
                if agent_name == "table_specialist":
                    result = self.table_specialist.process_page(page)
                    results.append(result)
                
                elif agent_name == "code_specialist":
                    chunks = self.code_specialist.process_page(page)
                    results.extend(chunks)
                
                elif agent_name == "handwriting_specialist":
                    result = self.handwriting_specialist.process_page(page, pdf_path)
                    results.append(result)
                
                elif agent_name == "chart_specialist":
                    result = self.chart_specialist.process_page(page, pdf_path)
                    results.append(result)
                
                else:  # text_specialist
                    result = self.text_specialist.process_page(page)
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Agent {agent_name} failed on page {page_num}: {e}")
                # Fallback to text specialist
                results.append(page)
        
        # If no results (shouldn't happen), return original
        if not results:
            results.append(page)
        
        return results

    async def process_document(
        self,
        pdf_path: str | Path,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> list[Document]:
        """
        Process a complete document with intelligent orchestration.
        
        Args:
            pdf_path: Path to PDF file
            on_progress: Progress callback (current, total)
            
        Returns:
            List of processed document chunks
        """
        pdf_path = Path(pdf_path)
        
        # Load pages
        pages = self.extractor.load_pdf_pages(pdf_path)
        total_pages = len(pages)
        
        logger.info(f"Processing {total_pages} pages from {pdf_path.name}")
        
        # Initialize processing state
        self.processing_state = {
            "pdf_path": str(pdf_path),
            "total_pages": total_pages,
            "completed_pages": 0,
            "status": "processing",
        }
        
        # Detect patterns for text specialist
        self.text_specialist.detect_patterns(pages)
        
        all_results = []
        
        for i, page in enumerate(pages):
            try:
                # Classify and route
                routing = self.classify_and_route(page)
                
                # Process with appropriate specialist
                results = self.process_page(page, pdf_path, routing)
                all_results.extend(results)
                
                # Update state
                self.processing_state["completed_pages"] = i + 1
                
                if on_progress:
                    on_progress(i + 1, total_pages)
                    
            except Exception as e:
                logger.error(f"Failed to process page {i}: {e}")
                # Add error placeholder
                all_results.append(Document(
                    page_content=f"[Page {i} processing failed: {e}]",
                    metadata={
                        "source": str(pdf_path),
                        "page": i,
                        "extraction_error": True,
                    }
                ))
        
        # Post-process: handle table merging
        all_results = self.table_specialist.process_pages(all_results)
        
        self.processing_state["status"] = "completed"
        
        logger.info(
            f"Completed processing {total_pages} pages "
            f"into {len(all_results)} documents"
        )
        
        return all_results

    def process_document_sync(
        self,
        pdf_path: str | Path,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> list[Document]:
        """
        Synchronous version of document processing.
        
        Args:
            pdf_path: Path to PDF file
            on_progress: Progress callback
            
        Returns:
            List of processed documents
        """
        import asyncio
        return asyncio.run(self.process_document(pdf_path, on_progress))

    def process_document_parallel(
        self,
        pdf_path: str | Path,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> list[Document]:
        """
        Process document with parallel page processing.
        
        Args:
            pdf_path: Path to PDF file
            on_progress: Progress callback
            
        Returns:
            List of processed documents
        """
        pdf_path = Path(pdf_path)
        
        # Load pages
        pages = self.extractor.load_pdf_pages(pdf_path)
        total_pages = len(pages)
        
        # Detect patterns
        self.text_specialist.detect_patterns(pages)
        
        all_results = []
        completed = 0
        
        def process_single_page(page: Document) -> list[Document]:
            routing = self.classify_and_route(page)
            return self.process_page(page, pdf_path, routing)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(process_single_page, page): i
                for i, page in enumerate(pages)
            }
            
            # Collect results in order
            results_by_index = {}
            
            from concurrent.futures import as_completed
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results_by_index[idx] = future.result()
                except Exception as e:
                    logger.error(f"Page {idx} processing failed: {e}")
                    results_by_index[idx] = [Document(
                        page_content=f"[Page {idx} failed]",
                        metadata={"page": idx, "error": str(e)},
                    )]
                
                completed += 1
                if on_progress:
                    on_progress(completed, total_pages)
            
            # Combine in order
            for i in range(total_pages):
                all_results.extend(results_by_index.get(i, []))
        
        return all_results

    def get_processing_state(self) -> dict:
        """Get current processing state."""
        return self.processing_state.copy()

    def resume_processing(
        self,
        pdf_path: str | Path,
        from_page: int,
    ) -> list[Document]:
        """
        Resume processing from a specific page.
        
        Args:
            pdf_path: Path to PDF
            from_page: Page to resume from
            
        Returns:
            Processed documents from resume point
        """
        pdf_path = Path(pdf_path)
        pages = self.extractor.load_pdf_pages(pdf_path)
        
        # Process only remaining pages
        remaining = pages[from_page:]
        
        results = []
        for page in remaining:
            routing = self.classify_and_route(page)
            page_results = self.process_page(page, pdf_path, routing)
            results.extend(page_results)
        
        return results

