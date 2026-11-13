"""FastAPI application factory, lifespan wiring, and health probe."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.database import close_db, init_db
from app.core.redis_client import close_redis, init_redis
from app.core.vector_store import close_qdrant, init_qdrant

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open infrastructure clients on startup and release them on shutdown."""
    logger.info(
        "Starting %s v%s (env=%s).",
        settings.app_name,
        settings.app_version,
        settings.environment,
    )

    await init_db()
    await init_redis()
    await init_qdrant()

    logger.info("Startup complete. OpenAPI docs available at /docs.")
    try:
        yield
    finally:
        await close_qdrant()
        await close_redis()
        await close_db()
        logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Document ingestion, conversational RAG over Qdrant, Redis-backed "
            "chat memory, and LLM interview booking."
        ),
        debug=settings.debug,
        lifespan=lifespan,
    )
    application.include_router(api_router, prefix=settings.api_v1_prefix)

    @application.get("/health", tags=["health"], summary="Liveness probe")
    async def health() -> dict[str, str]:
        """Return a lightweight liveness status for orchestrators."""
        return {
            "status": "ok",
            "app": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
        }

    return application


app: FastAPI = create_app()
