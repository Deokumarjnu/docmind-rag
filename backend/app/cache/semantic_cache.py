"""Semantic cache using Redis for similarity-based query caching.

This module provides significant cost savings on frequently asked questions
by caching responses and matching semantically similar queries.
"""

import hashlib
import json
import logging
import time
from typing import Optional

import numpy as np
import redis.asyncio as redis
from langchain_openai import OpenAIEmbeddings

from app.config import settings

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Redis-based semantic cache for RAG responses.
    
    Uses embedding similarity to match cached responses to new queries,
    even when the exact wording differs.
    
    Architecture:
    - Query embeddings stored as Redis hashes
    - Response data stored as JSON strings
    - Similarity computed using cosine distance
    - TTL-based expiration for cache entries
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        ttl_seconds: Optional[int] = None,
    ):
        """
        Initialize the semantic cache.
        
        Args:
            redis_url: Redis connection URL
            similarity_threshold: Minimum similarity for cache hit (0-1)
            ttl_seconds: Time-to-live for cache entries
        """
        self.redis_url = redis_url or settings.redis_url.replace(
            "/0", f"/{settings.redis_cache_db}"
        )
        self.similarity_threshold = similarity_threshold or settings.cache_similarity_threshold
        self.ttl_seconds = ttl_seconds or settings.cache_ttl_seconds
        
        self._redis: Optional[redis.Redis] = None
        self._embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            api_key=settings.openai_api_key,
        )
        
        # Keys
        self.embedding_key = "docmind:cache:embeddings"
        self.response_key_prefix = "docmind:cache:response:"
        self.stats_key = "docmind:cache:stats"

    async def connect(self) -> None:
        """Establish Redis connection."""
        if self._redis is None:
            self._redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            logger.info(f"Connected to Redis cache at {self.redis_url}")

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None

    def _get_query_hash(self, query: str) -> str:
        """Generate a unique hash for a query."""
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]

    async def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text."""
        return await self._embeddings.aembed_query(text)

    def _cosine_similarity(
        self,
        vec1: list[float],
        vec2: list[float],
    ) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    async def get(
        self,
        query: str,
        document_id: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Look up a cached response for a query.
        
        Args:
            query: User query
            document_id: Optional document scope
            
        Returns:
            Cached response or None if no match found
        """
        if not settings.cache_enabled:
            return None
            
        await self.connect()
        
        try:
            start_time = time.time()
            
            # Get query embedding
            query_embedding = await self._get_embedding(query)
            
            # Get all cached embeddings
            cached_embeddings = await self._redis.hgetall(self.embedding_key)
            
            if not cached_embeddings:
                await self._increment_stat("misses")
                return None
            
            # Find best match
            best_match_key = None
            best_similarity = 0.0
            
            for cache_key, embedding_json in cached_embeddings.items():
                # Filter by document_id if specified
                if document_id:
                    key_parts = cache_key.split(":")
                    if len(key_parts) > 1 and key_parts[1] != document_id:
                        continue
                
                cached_embedding = json.loads(embedding_json)
                similarity = self._cosine_similarity(query_embedding, cached_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_key = cache_key
            
            # Check if similarity threshold is met
            if best_similarity >= self.similarity_threshold:
                response_key = f"{self.response_key_prefix}{best_match_key}"
                cached_response = await self._redis.get(response_key)
                
                if cached_response:
                    await self._increment_stat("hits")
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Cache HIT (similarity: {best_similarity:.3f}, "
                        f"time: {elapsed*1000:.1f}ms)"
                    )
                    
                    response = json.loads(cached_response)
                    response["cache_hit"] = True
                    response["similarity"] = best_similarity
                    return response
            
            await self._increment_stat("misses")
            logger.debug(f"Cache MISS (best similarity: {best_similarity:.3f})")
            return None
            
        except Exception as e:
            logger.error(f"Cache lookup error: {e}")
            return None

    async def set(
        self,
        query: str,
        response: dict,
        document_id: Optional[str] = None,
    ) -> bool:
        """
        Cache a response for a query.
        
        Args:
            query: User query
            response: Response data to cache
            document_id: Optional document scope
            
        Returns:
            True if cached successfully
        """
        if not settings.cache_enabled:
            return False
            
        await self.connect()
        
        try:
            # Generate cache key
            query_hash = self._get_query_hash(query)
            cache_key = f"{query_hash}:{document_id or 'global'}"
            
            # Get and store embedding
            query_embedding = await self._get_embedding(query)
            await self._redis.hset(
                self.embedding_key,
                cache_key,
                json.dumps(query_embedding),
            )
            
            # Store response with TTL
            response_key = f"{self.response_key_prefix}{cache_key}"
            response_data = {
                **response,
                "cached_at": time.time(),
                "original_query": query,
            }
            await self._redis.setex(
                response_key,
                self.ttl_seconds,
                json.dumps(response_data),
            )
            
            await self._increment_stat("sets")
            logger.debug(f"Cached response for query: {query[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    async def invalidate(
        self,
        document_id: Optional[str] = None,
    ) -> int:
        """
        Invalidate cache entries.
        
        Args:
            document_id: If specified, only invalidate for this document
            
        Returns:
            Number of entries invalidated
        """
        await self.connect()
        
        try:
            if document_id:
                # Remove entries for specific document
                cached_embeddings = await self._redis.hgetall(self.embedding_key)
                invalidated = 0
                
                for cache_key in cached_embeddings:
                    if f":{document_id}" in cache_key:
                        await self._redis.hdel(self.embedding_key, cache_key)
                        await self._redis.delete(
                            f"{self.response_key_prefix}{cache_key}"
                        )
                        invalidated += 1
                
                return invalidated
            else:
                # Clear all cache
                await self._redis.delete(self.embedding_key)
                keys = await self._redis.keys(f"{self.response_key_prefix}*")
                if keys:
                    await self._redis.delete(*keys)
                return len(keys) if keys else 0
                
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
            return 0

    async def _increment_stat(self, stat_name: str) -> None:
        """Increment a statistics counter."""
        try:
            await self._redis.hincrby(self.stats_key, stat_name, 1)
        except Exception:
            pass

    async def get_stats(self) -> dict:
        """Get cache statistics."""
        await self.connect()
        
        try:
            stats = await self._redis.hgetall(self.stats_key)
            hits = int(stats.get("hits", 0))
            misses = int(stats.get("misses", 0))
            sets = int(stats.get("sets", 0))
            
            total_requests = hits + misses
            hit_rate = hits / total_requests if total_requests > 0 else 0
            
            return {
                "hits": hits,
                "misses": misses,
                "sets": sets,
                "hit_rate": f"{hit_rate:.1%}",
                "total_requests": total_requests,
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}


# Singleton instance
_semantic_cache: Optional[SemanticCache] = None


def get_semantic_cache() -> SemanticCache:
    """Get or create semantic cache singleton."""
    global _semantic_cache
    if _semantic_cache is None:
        _semantic_cache = SemanticCache()
    return _semantic_cache

