"""Integration tests for the document ingestion endpoint."""

from __future__ import annotations

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import settings
from app.models.domain import DocumentMetadata
from app.services.embedder import EmbeddingError
from tests.conftest import ApiContext


async def _document_count(session_factory: async_sessionmaker[AsyncSession]) -> int:
    async with session_factory() as session:
        count = await session.scalar(select(func.count()).select_from(DocumentMetadata))
    return int(count or 0)


class TestIngestionSuccess:
    async def test_txt_semantic_ingestion_persists_metadata_and_vectors(
        self,
        api: ApiContext,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        response = await api.client.post(
            "/api/v1/ingest",
            files={
                "file": (
                    "handbook.txt",
                    b"Leave Policy\n\nEmployees receive 25 days of paid annual leave.",
                    "text/plain",
                )
            },
            data={"chunking_strategy": "semantic"},
        )

        assert response.status_code == 201, response.text
        body = response.json()
        assert body["filename"] == "handbook.txt"
        assert body["chunking_strategy"] == "semantic"
        assert body["chunks_indexed"] >= 1
        assert len(api.qdrant.upserted) == body["chunks_indexed"]

        point = api.qdrant.upserted[0]
        assert point.payload is not None
        assert point.payload["document_id"] == body["document_id"]
        assert point.payload["filename"] == "handbook.txt"
        assert point.payload["chunk_index"] == 0
        assert isinstance(point.vector, list) and len(point.vector) == 8
        assert api.embedder.inputs

        assert await _document_count(session_factory) == 1

    async def test_fixed_strategy_is_accepted(
        self,
        api: ApiContext,
    ) -> None:
        response = await api.client.post(
            "/api/v1/ingest",
            files={
                "file": (
                    "long.txt",
                    (b"Section body text. " * 200),
                    "text/plain",
                )
            },
            data={"chunking_strategy": "fixed"},
        )

        assert response.status_code == 201, response.text
        assert response.json()["chunking_strategy"] == "fixed"
        assert response.json()["chunks_indexed"] > 1


class TestIngestionValidation:
    async def test_unsupported_extension_is_rejected(self, api: ApiContext) -> None:
        response = await api.client.post(
            "/api/v1/ingest",
            files={"file": ("notes.md", b"# Heading", "text/markdown")},
        )

        assert response.status_code == 400
        assert ".pdf" in response.json()["detail"]
        assert api.embedder.inputs == []

    async def test_mime_mismatch_is_rejected(self, api: ApiContext) -> None:
        response = await api.client.post(
            "/api/v1/ingest",
            files={"file": ("notes.txt", b"plain text", "application/pdf")},
        )

        assert response.status_code == 400

    async def test_empty_file_is_rejected(self, api: ApiContext) -> None:
        response = await api.client.post(
            "/api/v1/ingest",
            files={"file": ("empty.txt", b"   \n", "text/plain")},
        )

        assert response.status_code == 400
        assert "no extractable text" in response.json()["detail"].lower()

    async def test_invalid_strategy_returns_422(self, api: ApiContext) -> None:
        response = await api.client.post(
            "/api/v1/ingest",
            files={"file": ("notes.txt", b"text", "text/plain")},
            data={"chunking_strategy": "magic"},
        )

        assert response.status_code == 422

    async def test_oversized_upload_returns_413(
        self,
        api: ApiContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings, "max_upload_bytes", 16)
        response = await api.client.post(
            "/api/v1/ingest",
            files={"file": ("large.txt", b"x" * 17, "text/plain")},
        )

        assert response.status_code == 413
        assert "maximum upload size" in response.json()["detail"]


class TestIngestionFailures:
    async def test_embedding_failure_returns_502_without_metadata(
        self,
        api: ApiContext,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        api.embedder.error = EmbeddingError("provider down")

        response = await api.client.post(
            "/api/v1/ingest",
            files={"file": ("notes.txt", b"Leave policy text.", "text/plain")},
        )

        assert response.status_code == 502
        assert "provider down" in response.json()["detail"]
        assert await _document_count(session_factory) == 0
        assert api.qdrant.upserted == []

    async def test_vector_store_failure_returns_503_and_rolls_back(
        self,
        api: ApiContext,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        api.qdrant.upsert_error = RuntimeError("qdrant down")

        response = await api.client.post(
            "/api/v1/ingest",
            files={"file": ("notes.txt", b"Leave policy text.", "text/plain")},
        )

        assert response.status_code == 503
        assert "vector store" in response.json()["detail"].lower()
        assert await _document_count(session_factory) == 0
