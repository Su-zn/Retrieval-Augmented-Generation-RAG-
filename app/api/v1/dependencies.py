"""Composable FastAPI dependencies for endpoint service wiring."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.redis_client import get_redis_dependency
from app.core.vector_store import get_qdrant
from app.services.booking_service import BookingService
from app.services.embedder import EmbeddingService, get_embedding_service
from app.services.memory_service import MemoryService
from app.services.rag_service import RagService


def get_memory_service(
    redis: Annotated[Redis, Depends(get_redis_dependency)],
) -> MemoryService:
    """Build session-scoped chat memory on the shared Redis client."""
    return MemoryService(redis)


def get_booking_service(
    session: Annotated[AsyncSession, Depends(get_db)],
) -> BookingService:
    """Build the booking service on the request-scoped database session."""
    return BookingService(session)


def get_rag_service(
    memory: Annotated[MemoryService, Depends(get_memory_service)],
    embedder: Annotated[EmbeddingService, Depends(get_embedding_service)],
    qdrant: Annotated[AsyncQdrantClient, Depends(get_qdrant)],
    booking_service: Annotated[BookingService, Depends(get_booking_service)],
) -> RagService:
    """Compose the RAG service from its infrastructure collaborators."""
    return RagService(
        memory=memory,
        embedder=embedder,
        qdrant=qdrant,
        booking_service=booking_service,
    )
