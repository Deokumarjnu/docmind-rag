"""Knowledge Graph schema definitions.

Defines strict entity and relationship types to prevent
LLM hallucinations in graph extraction.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EntityType(str, Enum):
    """Allowed entity types for Knowledge Graph."""
    
    CONCEPT = "Concept"
    TECHNOLOGY = "Technology"
    ORGANIZATION = "Organization"
    PERSON = "Person"
    PRODUCT = "Product"
    PROCESS = "Process"
    METRIC = "Metric"
    LOCATION = "Location"
    EVENT = "Event"
    DOCUMENT = "Document"


class RelationType(str, Enum):
    """Allowed relationship types for Knowledge Graph."""
    
    # Hierarchical
    IS_A = "IS_A"
    PART_OF = "PART_OF"
    CONTAINS = "CONTAINS"
    
    # Associative
    RELATED_TO = "RELATED_TO"
    USES = "USES"
    IMPLEMENTS = "IMPLEMENTS"
    DEPENDS_ON = "DEPENDS_ON"
    
    # Actions
    CREATED_BY = "CREATED_BY"
    PRODUCES = "PRODUCES"
    TRANSFORMS = "TRANSFORMS"
    
    # Temporal
    PRECEDED_BY = "PRECEDED_BY"
    FOLLOWED_BY = "FOLLOWED_BY"
    
    # Comparative
    SIMILAR_TO = "SIMILAR_TO"
    DIFFERENT_FROM = "DIFFERENT_FROM"
    IMPROVES = "IMPROVES"
    
    # Document relations
    MENTIONED_IN = "MENTIONED_IN"
    DEFINED_IN = "DEFINED_IN"


@dataclass
class EntitySchema:
    """Schema for an entity in the Knowledge Graph."""
    
    name: str
    type: EntityType
    description: Optional[str] = None
    properties: Optional[dict] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for Neo4j."""
        return {
            "name": self.name,
            "type": self.type.value,
            "description": self.description or "",
            **(self.properties or {}),
        }


@dataclass
class RelationSchema:
    """Schema for a relationship in the Knowledge Graph."""
    
    source: str  # Source entity name
    target: str  # Target entity name
    relation_type: RelationType
    properties: Optional[dict] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for Neo4j."""
        return {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type.value,
            **(self.properties or {}),
        }


# Schema definition for LLM extraction prompts
KNOWLEDGE_GRAPH_SCHEMA = {
    "entity_types": [e.value for e in EntityType],
    "relation_types": [r.value for r in RelationType],
    "extraction_prompt": """Extract entities and relationships from the text.

ENTITY TYPES (use only these):
- Concept: Abstract ideas, theories, methodologies
- Technology: Software, hardware, frameworks, tools
- Organization: Companies, institutions, groups
- Person: Individual people
- Product: Specific products or services
- Process: Procedures, workflows, algorithms
- Metric: Measurements, statistics, KPIs
- Location: Places, regions
- Event: Conferences, releases, incidents
- Document: Papers, reports, specifications

RELATIONSHIP TYPES (use only these):
- IS_A: Type/subtype relationship
- PART_OF: Component relationship
- CONTAINS: Container relationship
- RELATED_TO: General association
- USES: Utilization relationship
- IMPLEMENTS: Implementation relationship
- DEPENDS_ON: Dependency relationship
- CREATED_BY: Creator relationship
- PRODUCES: Output relationship
- SIMILAR_TO: Similarity relationship
- IMPROVES: Enhancement relationship
- MENTIONED_IN: Document reference
- DEFINED_IN: Definition source

OUTPUT FORMAT (JSON):
{
  "entities": [
    {"name": "Entity Name", "type": "EntityType", "description": "Brief description"}
  ],
  "relationships": [
    {"source": "Entity1", "target": "Entity2", "type": "RELATION_TYPE"}
  ]
}

Be precise. Only extract clearly stated facts. Do not infer or hallucinate.""",
}

