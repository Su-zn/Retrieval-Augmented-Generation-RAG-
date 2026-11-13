"""Orchestration for the document ingestion pipeline."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import AsyncQdrantClient, models
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.domain import DocumentMetadata
from app.models.schemas import ChunkingStrategy
from app.services.chunker import build_chunker
from app.services.embedder import EmbeddingError, EmbeddingService
from app.services.extractor import ExtractionError, extract_document

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Raised when Qdrant rejects or cannot accept the chunk vectors."""


class MetadataPersistenceError(Exception):
    """Raised when document metadata cannot be persisted relationally."""


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Outcome of a successful ingestion."""

    document_id: uuid.UUID
    filename: str
    chunking_strategy: ChunkingStrategy
    chunks_indexed: int


def _point_id(document_id: uuid.UUID, chunk_index: int) -> str:
    """Deterministic, namespaced UUID for a chunk point."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"rag://documents/{document_id}/{chunk_index}"))


async def ingest_document(
    *,
    filename: str | None,
    content_type: str | None,
    data: bytes,
    chunking_strategy: ChunkingStrategy,
    session: AsyncSession,
    qdrant: AsyncQdrantClient,
    embedder: EmbeddingService,
) -> IngestionResult:
    """Extract, chunk, embed, index, and persist one uploaded document.

    Vectors are upserted before the metadata commit. If the commit fails, the
    already-written points are deleted on a best-effort basis so Qdrant does not
    retain vectors for an unknown document.
    """
    extracted = await extract_document(filename=filename, content_type=content_type, data=data)

    safe_filename = Path(filename or "upload").name
    if len(safe_filename) > 512:
        raise ExtractionError("The filename is too long (maximum 512 characters).")

    chunker = build_chunker(chunking_strategy)
    chunks = await asyncio.to_thread(chunker.chunk, extracted.text)
    if not chunks:
        raise ExtractionError("No chunks could be produced from the uploaded document.")

    vectors = await embedder.embed_texts(chunks)
    if len(vectors) != len(chunks):
        raise EmbeddingError(
            f"Embedding count mismatch: {len(vectors)} vectors for {len(chunks)} chunks."
        )

    document_id = uuid.uuid4()
    point_ids = [_point_id(document_id, index) for index in range(len(chunks))]
    points = [
        models.PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "document_id": str(document_id),
                "chunk_index": index,
                "text": chunk,
                "filename": safe_filename,
                "chunking_strategy": chunking_strategy,
            },
        )
        for index, (point_id, vector, chunk) in enumerate(
            zip(point_ids, vectors, chunks, strict=True)
        )
    ]

    metadata = DocumentMetadata(
        id=document_id,
        filename=safe_filename,
        content_type=extracted.content_type,
        chunking_strategy=chunking_strategy,
        chunk_count=len(chunks),
    )
    session.add(metadata)

    try:
        await session.flush()
    except SQLAlchemyError as exc:
        await session.rollback()
        raise MetadataPersistenceError("Failed to persist document metadata.") from exc

    try:
        await qdrant.upsert(
            collection_name=settings.qdrant_collection,
            points=points,
            wait=True,
        )
    except Exception as exc:  # noqa: BLE001 - Qdrant client raises transport-specific errors
        await session.rollback()
        logger.warning("Qdrant upsert failed for document %s: %s", document_id, exc)
        raise VectorStoreError("Failed to index document chunks in the vector store.") from exc

    try:
        await session.commit()
    except SQLAlchemyError as exc:
        await session.rollback()
        try:
            await qdrant.delete(
                collection_name=settings.qdrant_collection,
                points_selector=models.PointIdsList(points=point_ids),
                wait=True,
            )
        except Exception as cleanup_exc:  # noqa: BLE001 - cleanup must not mask the original error
            logger.warning(
                "Failed to roll back Qdrant points for document %s: %s", document_id, cleanup_exc
            )
        raise MetadataPersistenceError("Failed to persist document metadata.") from exc

    logger.info(
        "Ingested document %s ('%s', strategy=%s, chunks=%d).",
        document_id,
        safe_filename,
        chunking_strategy,
        len(chunks),
    )

    return IngestionResult(
        document_id=document_id,
        filename=safe_filename,
        chunking_strategy=chunking_strategy,
        chunks_indexed=len(chunks),
    )
