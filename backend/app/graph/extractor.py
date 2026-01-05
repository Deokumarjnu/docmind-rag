"""Knowledge Graph entity and relationship extraction using LLM."""

import json
import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import settings
from app.graph.schema import (
    EntitySchema,
    EntityType,
    RelationSchema,
    RelationType,
    KNOWLEDGE_GRAPH_SCHEMA,
)

logger = logging.getLogger(__name__)


class KnowledgeGraphExtractor:
    """
    Extract entities and relationships from text using LLM.
    
    Uses a structured schema to prevent hallucinations and ensure
    consistent graph structure.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the extractor.
        
        Args:
            model_name: LLM model to use for extraction
        """
        self.model = ChatOpenAI(
            model=model_name or settings.fast_llm_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", KNOWLEDGE_GRAPH_SCHEMA["extraction_prompt"]),
            ("user", """Extract entities and relationships from this text:

{text}

Remember: Only extract explicitly stated facts. Return valid JSON."""),
        ])
        
        self.chain = self.prompt | self.model

    def extract_from_text(self, text: str) -> dict:
        """
        Extract entities and relationships from text.
        
        Args:
            text: Input text to extract from
            
        Returns:
            Dictionary with 'entities' and 'relationships' lists
        """
        if len(text.strip()) < 50:
            return {"entities": [], "relationships": []}
        
        try:
            # Truncate very long text
            if len(text) > 4000:
                text = text[:4000] + "..."
            
            result = self.chain.invoke({"text": text})
            content = result.content
            
            # Parse JSON from response
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            parsed = json.loads(content.strip())
            
            # Validate and convert entities
            entities = []
            for e in parsed.get("entities", []):
                try:
                    entity_type = EntityType(e.get("type", "Concept"))
                    entities.append(EntitySchema(
                        name=e["name"],
                        type=entity_type,
                        description=e.get("description"),
                    ))
                except (ValueError, KeyError) as err:
                    logger.debug(f"Skipping invalid entity: {e}, error: {err}")
            
            # Validate and convert relationships
            relationships = []
            entity_names = {e.name for e in entities}
            
            for r in parsed.get("relationships", []):
                try:
                    # Only add relationships where both entities exist
                    if r["source"] in entity_names and r["target"] in entity_names:
                        relation_type = RelationType(r.get("type", "RELATED_TO"))
                        relationships.append(RelationSchema(
                            source=r["source"],
                            target=r["target"],
                            relation_type=relation_type,
                        ))
                except (ValueError, KeyError) as err:
                    logger.debug(f"Skipping invalid relationship: {r}, error: {err}")
            
            return {
                "entities": entities,
                "relationships": relationships,
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return {"entities": [], "relationships": []}
        except Exception as e:
            logger.error(f"Knowledge graph extraction failed: {e}")
            return {"entities": [], "relationships": []}

    def extract_from_document(self, doc: Document) -> dict:
        """
        Extract knowledge graph from a LangChain Document.
        
        Args:
            doc: Document to process
            
        Returns:
            Extraction results with document metadata
        """
        result = self.extract_from_text(doc.page_content)
        
        # Add document context
        result["source"] = doc.metadata.get("source", "unknown")
        result["page"] = doc.metadata.get("page", 0)
        result["chunk_id"] = doc.metadata.get("chunk_id", "")
        
        return result

    def extract_from_documents(
        self,
        documents: list[Document],
        batch_size: int = 10,
    ) -> list[dict]:
        """
        Extract knowledge graph from multiple documents.
        
        Args:
            documents: List of documents
            batch_size: Number of documents to process at once
            
        Returns:
            List of extraction results
        """
        results = []
        
        for i, doc in enumerate(documents):
            try:
                result = self.extract_from_document(doc)
                results.append(result)
                
                if (i + 1) % batch_size == 0:
                    logger.info(f"Extracted KG from {i + 1}/{len(documents)} documents")
                    
            except Exception as e:
                logger.error(f"Failed to extract from document {i}: {e}")
                results.append({
                    "entities": [],
                    "relationships": [],
                    "error": str(e),
                })
        
        logger.info(f"Completed KG extraction from {len(documents)} documents")
        return results

