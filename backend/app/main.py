"""FastAPI main application with LangServe integration."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes import router as api_router

logger = logging.getLogger(__name__)

# Set up LangSmith tracing if configured
if settings.langchain_tracing_v2 and settings.langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project


async def init_knowledge_graph():
    """Initialize Neo4j Knowledge Graph."""
    try:
        from app.graph import get_neo4j_client, KnowledgeGraphStore
        
        client = get_neo4j_client()
        await client.connect()
        
        store = KnowledgeGraphStore(client)
        await store.initialize()
        
        logger.info("✅ Neo4j Knowledge Graph initialized")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Neo4j initialization failed (optional): {e}")
        return False


async def init_semantic_cache():
    """Initialize Redis Semantic Cache."""
    try:
        from app.cache import get_semantic_cache
        
        cache = get_semantic_cache()
        await cache.connect()
        
        logger.info("✅ Redis Semantic Cache initialized")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Semantic cache initialization failed (optional): {e}")
        return False


async def init_conversation_store():
    """Initialize PostgreSQL Conversation Store."""
    try:
        from app.memory import get_conversation_store
        
        store = get_conversation_store()
        await store.initialize()
        
        logger.info("✅ PostgreSQL Conversation Store initialized")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Conversation store initialization failed (optional): {e}")
        return False


async def cleanup_services():
    """Cleanup all services on shutdown."""
    try:
        from app.graph import get_neo4j_client
        client = get_neo4j_client()
        await client.close()
    except Exception:
        pass
    
    try:
        from app.cache import get_semantic_cache
        cache = get_semantic_cache()
        await cache.close()
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    # Startup
    print(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    
    # Initialize services (non-blocking, optional)
    await init_knowledge_graph()
    await init_semantic_cache()
    await init_conversation_store()
    
    yield
    
    # Shutdown
    print(f"👋 Shutting down {settings.app_name}")
    await cleanup_services()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Enterprise Document Intelligence Platform with RAG",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Check services status
    services = {"qdrant": "unknown", "redis": "unknown", "neo4j": "unknown", "postgres": "unknown"}
    
    try:
        from app.graph import get_neo4j_client
        client = get_neo4j_client()
        services["neo4j"] = "healthy" if await client.health_check() else "unhealthy"
    except Exception:
        services["neo4j"] = "unavailable"
    
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "services": services,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )

