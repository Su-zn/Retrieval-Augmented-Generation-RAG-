"""POST /api/v1/chat — validate the request and delegate to the RAG service."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.v1.dependencies import get_rag_service
from app.models.schemas import ChatRequest, ChatResponse
from app.services.booking_service import BookingPersistenceError
from app.services.embedder import EmbeddingError
from app.services.memory_service import MemoryStoreError
from app.services.rag_service import LLMError, RagService, RetrievalError

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question or book an interview",
    description=(
        "Load the session's Redis-backed history, reformulate follow-up questions "
        "into standalone queries, retrieve the top matching chunks from Qdrant, "
        "generate a grounded answer with source attribution, and execute the "
        "`book_interview` tool when the user supplies complete booking details."
    ),
    responses={
        422: {"description": "Empty session_id or message."},
        502: {"description": "Embedding or LLM provider failure."},
        503: {"description": "Redis, Qdrant, or relational database unavailable."},
    },
)
async def chat_endpoint(
    payload: ChatRequest,
    service: Annotated[RagService, Depends(get_rag_service)],
) -> ChatResponse:
    """Answer one conversational turn, optionally booking an interview."""
    try:
        result = await service.answer(session_id=payload.session_id, question=payload.message)
    except (MemoryStoreError, RetrievalError, BookingPersistenceError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except (EmbeddingError, LLMError) as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    return ChatResponse(
        session_id=payload.session_id,
        answer=result.answer,
        sources=result.sources,
        booking=result.booking,
    )
