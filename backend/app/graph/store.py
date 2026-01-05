"""Knowledge Graph storage and retrieval operations."""

import logging
from typing import Optional

from app.graph.client import Neo4jClient, get_neo4j_client
from app.graph.schema import EntitySchema, RelationSchema

logger = logging.getLogger(__name__)


class KnowledgeGraphStore:
    """
    Store and query the Knowledge Graph in Neo4j.
    
    Provides methods for adding entities/relationships and
    performing graph-based retrieval for multi-hop reasoning.
    """

    def __init__(self, client: Optional[Neo4jClient] = None):
        """
        Initialize the store.
        
        Args:
            client: Optional Neo4j client (uses singleton if not provided)
        """
        self.client = client or get_neo4j_client()

    async def initialize(self) -> None:
        """Initialize the store and create indexes."""
        await self.client.connect()
        await self.client.create_indexes()

    async def add_entity(
        self,
        entity: EntitySchema,
        document_id: Optional[str] = None,
        chunk_id: Optional[str] = None,
    ) -> dict:
        """
        Add or merge an entity to the graph.
        
        Args:
            entity: Entity to add
            document_id: Source document ID
            chunk_id: Source chunk ID
            
        Returns:
            Query result
        """
        query = """
        MERGE (e:Entity {name: $name})
        ON CREATE SET 
            e.type = $type,
            e.description = $description,
            e.created_at = datetime()
        ON MATCH SET
            e.type = COALESCE($type, e.type),
            e.description = COALESCE($description, e.description),
            e.updated_at = datetime()
        """
        
        # Link to document/chunk if provided
        if document_id:
            query += """
            WITH e
            MERGE (d:Document {document_id: $document_id})
            MERGE (e)-[:MENTIONED_IN]->(d)
            """
        
        if chunk_id:
            query += """
            WITH e
            MERGE (c:Chunk {chunk_id: $chunk_id})
            MERGE (e)-[:EXTRACTED_FROM]->(c)
            """
        
        query += " RETURN e.name as name"
        
        params = {
            **entity.to_dict(),
            "document_id": document_id,
            "chunk_id": chunk_id,
        }
        
        return await self.client.run_write_query(query, params)

    async def add_relationship(self, relation: RelationSchema) -> dict:
        """
        Add a relationship between entities.
        
        Args:
            relation: Relationship to add
            
        Returns:
            Query result
        """
        query = f"""
        MATCH (source:Entity {{name: $source}})
        MATCH (target:Entity {{name: $target}})
        MERGE (source)-[r:{relation.relation_type.value}]->(target)
        ON CREATE SET r.created_at = datetime()
        RETURN type(r) as relation_type
        """
        
        return await self.client.run_write_query(
            query,
            {"source": relation.source, "target": relation.target}
        )

    async def add_extraction_results(
        self,
        extraction: dict,
        document_id: Optional[str] = None,
    ) -> dict:
        """
        Add entities and relationships from extraction results.
        
        Args:
            extraction: Extraction results from KnowledgeGraphExtractor
            document_id: Source document ID
            
        Returns:
            Summary of added items
        """
        entities_added = 0
        relations_added = 0
        
        chunk_id = extraction.get("chunk_id")
        
        # Add entities first
        for entity in extraction.get("entities", []):
            try:
                await self.add_entity(entity, document_id, chunk_id)
                entities_added += 1
            except Exception as e:
                logger.warning(f"Failed to add entity {entity.name}: {e}")
        
        # Add relationships
        for relation in extraction.get("relationships", []):
            try:
                await self.add_relationship(relation)
                relations_added += 1
            except Exception as e:
                logger.warning(f"Failed to add relationship: {e}")
        
        return {
            "entities_added": entities_added,
            "relationships_added": relations_added,
        }

    async def search_entities(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict]:
        """
        Full-text search for entities.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            Matching entities
        """
        cypher = """
        CALL db.index.fulltext.queryNodes("entity_search", $query)
        YIELD node, score
        RETURN node.name as name, 
               node.type as type, 
               node.description as description,
               score
        ORDER BY score DESC
        LIMIT $limit
        """
        
        return await self.client.run_query(cypher, {"query": query, "limit": limit})

    async def get_entity_context(
        self,
        entity_name: str,
        hops: int = 2,
        limit: int = 20,
    ) -> list[dict]:
        """
        Get multi-hop context for an entity.
        
        This is the key method for multi-hop reasoning - it retrieves
        related entities and relationships up to N hops away.
        
        Args:
            entity_name: Starting entity name
            hops: Number of relationship hops
            limit: Maximum relationships to return
            
        Returns:
            List of paths from the entity
        """
        cypher = f"""
        MATCH path = (start:Entity {{name: $name}})-[*1..{hops}]-(related:Entity)
        WITH path, related,
             [r IN relationships(path) | type(r)] as rel_types,
             [n IN nodes(path) | n.name] as node_names
        RETURN DISTINCT
            related.name as entity,
            related.type as type,
            related.description as description,
            rel_types,
            node_names,
            length(path) as distance
        ORDER BY distance ASC
        LIMIT $limit
        """
        
        return await self.client.run_query(
            cypher,
            {"name": entity_name, "limit": limit}
        )

    async def hybrid_search(
        self,
        query: str,
        entity_limit: int = 5,
        context_hops: int = 2,
    ) -> dict:
        """
        Perform hybrid search: text search + multi-hop expansion.
        
        Args:
            query: Search query
            entity_limit: Number of seed entities
            context_hops: Hops for context expansion
            
        Returns:
            Dictionary with entities and context
        """
        # Step 1: Find seed entities via text search
        seed_entities = await self.search_entities(query, limit=entity_limit)
        
        # Step 2: Expand each seed entity with multi-hop context
        all_context = []
        for entity in seed_entities:
            context = await self.get_entity_context(
                entity["name"],
                hops=context_hops,
            )
            all_context.extend(context)
        
        return {
            "seed_entities": seed_entities,
            "expanded_context": all_context,
        }

    async def get_document_entities(
        self,
        document_id: str,
    ) -> list[dict]:
        """
        Get all entities mentioned in a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            List of entities
        """
        cypher = """
        MATCH (e:Entity)-[:MENTIONED_IN]->(d:Document {document_id: $doc_id})
        RETURN e.name as name, e.type as type, e.description as description
        """
        
        return await self.client.run_query(cypher, {"doc_id": document_id})

    async def get_stats(self) -> dict:
        """Get graph statistics."""
        cypher = """
        MATCH (e:Entity)
        WITH count(e) as entity_count
        MATCH ()-[r]->()
        RETURN entity_count, count(r) as relationship_count
        """
        
        results = await self.client.run_query(cypher)
        if results:
            return results[0]
        return {"entity_count": 0, "relationship_count": 0}

