"""POST /api/v1/chat — validate the request and delegate to the RAG service."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.redis_client import get_redis_dependency
from app.core.vector_store import get_qdrant
from app.models.schemas import ChatRequest, ChatResponse
from app.services.booking_service import BookingPersistenceError, BookingService
from app.services.embedder import EmbeddingError, EmbeddingService, get_embedding_service
from app.services.memory_service import MemoryService, MemoryStoreError
from app.services.rag_service import LLMError, RagService, RetrievalError

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question about the ingested documents",
    description=(
        "Load the session's Redis-backed history, reformulate follow-up questions "
        "into standalone queries, retrieve the top matching chunks from Qdrant, "
        "and generate a grounded answer with source attribution."
    ),
    responses={
        422: {"description": "Empty session_id or message."},
        502: {"description": "Embedding or LLM provider failure."},
        503: {"description": "Redis, Qdrant, or relational database unavailable."},
    },
)
async def chat_endpoint(
    payload: ChatRequest,
    redis: Annotated[Redis, Depends(get_redis_dependency)],
    qdrant: Annotated[AsyncQdrantClient, Depends(get_qdrant)],
    session: Annotated[AsyncSession, Depends(get_db)],
    embedder: Annotated[EmbeddingService, Depends(get_embedding_service)],
) -> ChatResponse:
    """Answer one conversational turn, optionally booking an interview."""
    service = RagService(
        memory=MemoryService(redis),
        embedder=embedder,
        qdrant=qdrant,
        booking_service=BookingService(session),
    )

    try:
        result = await service.answer(session_id=payload.session_id, question=payload.message)
    except MemoryStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except RetrievalError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except BookingPersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except EmbeddingError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    return ChatResponse(
        session_id=payload.session_id,
        answer=result.answer,
        sources=result.sources,
        booking=result.booking,
    )
