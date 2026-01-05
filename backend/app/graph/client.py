"""Neo4j client for Knowledge Graph operations."""

import logging
from typing import Any, Optional
from contextlib import asynccontextmanager

from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import ServiceUnavailable

from app.config import settings

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Async Neo4j client with connection pooling.
    
    Provides methods for running Cypher queries and managing
    the Knowledge Graph for multi-hop reasoning.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        Initialize Neo4j client.
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
        """
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        self._driver: Optional[AsyncDriver] = None

    async def connect(self) -> None:
        """Establish connection to Neo4j."""
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
            )
            # Verify connectivity
            try:
                await self._driver.verify_connectivity()
                logger.info(f"Connected to Neo4j at {self.uri}")
            except ServiceUnavailable as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                raise

    async def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")

    @asynccontextmanager
    async def session(self, database: str = "neo4j"):
        """Get an async session context manager."""
        if self._driver is None:
            await self.connect()
        
        session = self._driver.session(database=database)
        try:
            yield session
        finally:
            await session.close()

    async def run_query(
        self,
        query: str,
        parameters: Optional[dict] = None,
        database: str = "neo4j",
    ) -> list[dict[str, Any]]:
        """
        Run a Cypher query and return results.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            database: Target database
            
        Returns:
            List of result records as dictionaries
        """
        async with self.session(database) as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records

    async def run_write_query(
        self,
        query: str,
        parameters: Optional[dict] = None,
        database: str = "neo4j",
    ) -> dict[str, Any]:
        """
        Run a write query within a transaction.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            database: Target database
            
        Returns:
            Query summary
        """
        async with self.session(database) as session:
            result = await session.run(query, parameters or {})
            summary = await result.consume()
            return {
                "nodes_created": summary.counters.nodes_created,
                "relationships_created": summary.counters.relationships_created,
                "properties_set": summary.counters.properties_set,
            }

    async def create_indexes(self) -> None:
        """Create necessary indexes for Knowledge Graph."""
        indexes = [
            # Entity indexes
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX document_id IF NOT EXISTS FOR (d:Document) ON (d.document_id)",
            "CREATE INDEX chunk_id IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_id)",
            # Full-text search
            """CREATE FULLTEXT INDEX entity_search IF NOT EXISTS 
               FOR (e:Entity) ON EACH [e.name, e.description]""",
        ]
        
        for index_query in indexes:
            try:
                await self.run_write_query(index_query)
                logger.debug(f"Created index: {index_query[:50]}...")
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")

    async def health_check(self) -> bool:
        """Check if Neo4j is healthy."""
        try:
            result = await self.run_query("RETURN 1 as health")
            return len(result) > 0
        except Exception as e:
            logger.error(f"Neo4j health check failed: {e}")
            return False


# Singleton instance
_neo4j_client: Optional[Neo4jClient] = None


def get_neo4j_client() -> Neo4jClient:
    """Get or create Neo4j client singleton."""
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient()
    return _neo4j_client

