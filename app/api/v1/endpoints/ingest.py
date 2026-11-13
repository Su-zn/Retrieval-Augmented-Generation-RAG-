"""POST /api/v1/ingest — validate the upload and delegate to the ingestion service."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.vector_store import get_qdrant
from app.models.schemas import ChunkingStrategy, IngestionResponse
from app.services.embedder import EmbeddingError, EmbeddingService, get_embedding_service
from app.services.extractor import ExtractionError
from app.services.ingestion_service import (
    MetadataPersistenceError,
    VectorStoreError,
    ingest_document,
)

router = APIRouter()


@router.post(
    "/ingest",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a PDF or TXT document",
    description=(
        "Extract text, split it with the selected strategy, generate embeddings, "
        "store vectors in Qdrant, and persist document metadata in SQL."
    ),
    responses={
        400: {"description": "Unsupported, corrupt, or empty document."},
        413: {"description": "Upload exceeds MAX_UPLOAD_BYTES."},
        502: {"description": "Embedding provider failure."},
        503: {"description": "Qdrant or relational database unavailable."},
    },
)
async def ingest_document_endpoint(
    file: Annotated[UploadFile, File(description="PDF (.pdf) or plain-text (.txt) file.")],
    session: Annotated[AsyncSession, Depends(get_db)],
    qdrant: Annotated[AsyncQdrantClient, Depends(get_qdrant)],
    embedder: Annotated[EmbeddingService, Depends(get_embedding_service)],
    chunking_strategy: Annotated[
        ChunkingStrategy,
        Form(description="Chunking strategy: 'fixed' windows or 'semantic' structure-aware."),
    ] = "semantic",
) -> IngestionResponse:
    """Ingest one uploaded document and return its indexing summary."""
    try:
        data = await file.read(settings.max_upload_bytes + 1)
    finally:
        await file.close()

    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File exceeds the maximum upload size of "
                f"{settings.max_upload_bytes} bytes."
            ),
        )

    try:
        result = await ingest_document(
            filename=file.filename,
            content_type=file.content_type,
            data=data,
            chunking_strategy=chunking_strategy,
            session=session,
            qdrant=qdrant,
            embedder=embedder,
        )
    except ExtractionError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except EmbeddingError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except MetadataPersistenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    return IngestionResponse(
        document_id=result.document_id,
        filename=result.filename,
        chunking_strategy=result.chunking_strategy,
        chunks_indexed=result.chunks_indexed,
    )
