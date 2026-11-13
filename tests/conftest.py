"""Shared fixtures and test doubles for the API test suite."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import fakeredis.aioredis
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from qdrant_client import models
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from app.core.database import Base, get_db
from app.core.redis_client import get_redis_dependency
from app.core.vector_store import get_qdrant
from app.main import create_app
from app.services.embedder import get_embedding_service
from app.services.rag_service import RagService


class FakeEmbedder:
    """Deterministic embedder that records inputs and can simulate outages."""

    def __init__(self, dimension: int = 8, error: Exception | None = None) -> None:
        self.dimension = dimension
        self.error = error
        self.inputs: list[list[str]] = []

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.inputs.append(list(texts))
        if self.error is not None:
            raise self.error
        return [[0.01] * self.dimension for _ in texts]


class FakeQdrant:
    """In-memory stand-in for the small Qdrant surface the app uses."""

    def __init__(self) -> None:
        self.upserted: list[models.PointStruct] = []
        self.deleted: list[object] = []
        self.query_results: list[models.ScoredPoint] = []
        self.upsert_error: Exception | None = None
        self.query_error: Exception | None = None
        self.last_query: dict[str, Any] | None = None

    async def upsert(
        self,
        *,
        collection_name: str,
        points: list[models.PointStruct],
        wait: bool = True,
    ) -> None:
        if self.upsert_error is not None:
            raise self.upsert_error
        self.upserted.extend(points)

    async def delete(
        self,
        *,
        collection_name: str,
        points_selector: object,
        wait: bool = True,
    ) -> None:
        self.deleted.append(points_selector)

    async def query_points(self, **kwargs: Any) -> SimpleNamespace:
        self.last_query = kwargs
        if self.query_error is not None:
            raise self.query_error
        return SimpleNamespace(points=list(self.query_results))


class FakeCompletions:
    """Records chat-completion calls and returns a caller-supplied response."""

    def __init__(self, handler: Callable[[dict[str, Any]], Any]) -> None:
        self.handler = handler
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.handler(kwargs)


class FakeLLMClient:
    """Minimal AsyncOpenAI-compatible client exposing chat.completions."""

    def __init__(self, handler: Callable[[dict[str, Any]], Any]) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(handler))


LLMHandler = Callable[[dict[str, Any]], Any]
LLMInstaller = Callable[[LLMHandler], FakeLLMClient]


@dataclass
class ApiContext:
    """HTTP client plus the fake infrastructure wired into the app."""

    client: AsyncClient
    app: FastAPI
    qdrant: FakeQdrant
    embedder: FakeEmbedder


def text_response(content: str) -> SimpleNamespace:
    """Build an OpenAI-shaped completion carrying plain text."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=None),
            )
        ]
    )


def tool_call_response(
    name: str,
    arguments: str,
    *,
    call_id: str = "call_1",
) -> SimpleNamespace:
    """Build an OpenAI-shaped completion requesting one tool call."""
    call = SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="", tool_calls=[call]),
            )
        ]
    )


def scripted_handler(
    responses: list[SimpleNamespace],
) -> Callable[[dict[str, Any]], SimpleNamespace]:
    """Return a handler that serves ``responses`` in order."""
    queue = list(responses)

    def handler(_: dict[str, Any]) -> SimpleNamespace:
        if not queue:
            raise AssertionError("Fake LLM received more calls than scripted responses.")
        return queue.pop(0)

    return handler


def make_point(
    *,
    text: str = "Employees receive 25 days of paid annual leave.",
    filename: str = "sample.txt",
    chunk_index: int = 0,
    score: float = 0.9,
    document_id: UUID | None = None,
) -> models.ScoredPoint:
    """Build a Qdrant scored point with the Phase 3 payload shape."""
    return models.ScoredPoint(
        id=str(uuid4()),
        version=1,
        score=score,
        payload={
            "document_id": str(document_id or uuid4()),
            "filename": filename,
            "chunk_index": chunk_index,
            "text": text,
        },
    )


@pytest_asyncio.fixture
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """Yield an isolated in-memory SQLite session factory per test."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    try:
        yield factory
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def fake_redis() -> AsyncIterator[Any]:
    """Yield a fresh in-memory Redis implementation with list/pipeline support."""
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    await client.flushall()
    try:
        yield client
    finally:
        await client.aclose()


@pytest_asyncio.fixture
async def api(
    session_factory: async_sessionmaker[AsyncSession],
    fake_redis: Any,
) -> AsyncIterator[ApiContext]:
    """Yield an ASGI test client with database/Redis/Qdrant/embedding overrides.

    The app lifespan is intentionally not run: the fixtures replace every
    external dependency, so tests stay hermetic and fast.
    """
    app = create_app()
    qdrant = FakeQdrant()
    embedder = FakeEmbedder()

    async def override_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_redis_dependency] = lambda: fake_redis
    app.dependency_overrides[get_qdrant] = lambda: qdrant
    app.dependency_overrides[get_embedding_service] = lambda: embedder

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield ApiContext(client=client, app=app, qdrant=qdrant, embedder=embedder)


@pytest.fixture
def install_llm(monkeypatch: pytest.MonkeyPatch) -> LLMInstaller:
    """Return an installer that replaces ``RagService``'s provider client."""

    def install(handler: LLMHandler) -> FakeLLMClient:
        client = FakeLLMClient(handler)
        monkeypatch.setattr(RagService, "_get_client", lambda self: client)
        return client

    return install
