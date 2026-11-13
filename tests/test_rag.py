"""Tests for multi-turn RAG, reformulation, retrieval, and error mapping."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from openai import APIConnectionError
from qdrant_client import models

from app.services.embedder import EmbeddingError
from app.services.memory_service import MemoryService
from app.services.rag_service import (
    NO_CONTEXT_ANSWER,
    LLMError,
    RagService,
    RetrievalError,
)
from tests.conftest import (
    ApiContext,
    FakeEmbedder,
    FakeLLMClient,
    FakeQdrant,
    LLMInstaller,
    make_point,
    scripted_handler,
    text_response,
)


def connection_error() -> APIConnectionError:
    """Build a realistic provider connection failure."""
    return APIConnectionError(request=httpx.Request("POST", "http://test/v1"))


def build_service(
    *,
    handler: Any,
    fake_redis: Any,
    qdrant: FakeQdrant | None = None,
    embedder: FakeEmbedder | None = None,
) -> tuple[RagService, FakeQdrant, FakeEmbedder]:
    """Build a RagService without booking so the plain RAG path is exercised."""
    qdrant = qdrant or FakeQdrant()
    embedder = embedder or FakeEmbedder()
    service = RagService(
        memory=MemoryService(fake_redis),
        embedder=embedder,  # type: ignore[arg-type]
        qdrant=qdrant,  # type: ignore[arg-type]
        client=FakeLLMClient(handler),  # type: ignore[arg-type]
        temperature=0.0,
        top_k=3,
    )
    return service, qdrant, embedder


class TestRetrieval:
    async def test_grounded_answer_returns_attributed_sources(self, fake_redis: Any) -> None:
        point = make_point(
            text="Employees receive 25 days of paid annual leave.",
            filename="handbook.txt",
            chunk_index=4,
            score=0.83,
        )
        qdrant = FakeQdrant()
        qdrant.query_results = [point]
        service, qdrant, embedder = build_service(
            handler=scripted_handler([text_response("25 days per year. [1]")]),
            fake_redis=fake_redis,
            qdrant=qdrant,
        )

        result = await service.answer(
            session_id="sess-rag", question="How much annual leave do I get?"
        )

        assert result.answer == "25 days per year. [1]"
        assert result.booking is None
        assert result.standalone_query == "How much annual leave do I get?"
        assert len(result.sources) == 1
        source = result.sources[0]
        assert source.filename == "handbook.txt"
        assert source.chunk_index == 4
        assert source.score == pytest.approx(0.83)
        assert embedder.inputs == [["How much annual leave do I get?"]]
        assert qdrant.last_query is not None
        assert qdrant.last_query["limit"] == 3
        assert qdrant.last_query["with_payload"] is True

    async def test_empty_retrieval_returns_refusal_without_calling_the_llm(
        self, fake_redis: Any
    ) -> None:
        llm = FakeLLMClient(scripted_handler([]))
        service = RagService(
            memory=MemoryService(fake_redis),
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
            qdrant=FakeQdrant(),  # type: ignore[arg-type]
            client=llm,  # type: ignore[arg-type]
        )

        result = await service.answer(
            session_id="sess-empty", question="Who won the 1998 World Cup?"
        )

        assert result.answer == NO_CONTEXT_ANSWER
        assert result.sources == []
        assert result.booking is None
        assert llm.chat.completions.calls == []
        assert await fake_redis.llen("chat:history:sess-empty") == 2

    async def test_malformed_points_are_skipped(self, fake_redis: Any) -> None:
        qdrant = FakeQdrant()
        qdrant.query_results = [
            models.ScoredPoint(id="missing-text", version=1, score=0.9, payload={}),
            models.ScoredPoint(
                id="bad-document-id",
                version=1,
                score=0.8,
                payload={"document_id": "not-a-uuid", "text": "text"},
            ),
            make_point(text="  ", score=0.7),
        ]
        service, _, _ = build_service(
            handler=scripted_handler([]), fake_redis=fake_redis, qdrant=qdrant
        )

        result = await service.answer(session_id="sess-bad", question="Anything?")

        assert result.sources == []
        assert result.answer == NO_CONTEXT_ANSWER

    async def test_qdrant_failure_raises_retrieval_error(self, fake_redis: Any) -> None:
        qdrant = FakeQdrant()
        qdrant.query_error = RuntimeError("qdrant down")
        service, _, _ = build_service(
            handler=scripted_handler([]), fake_redis=fake_redis, qdrant=qdrant
        )

        with pytest.raises(RetrievalError, match="vector store"):
            await service.answer(session_id="sess-qdrant", question="Question?")


class TestReformulation:
    async def test_history_rewrites_the_search_query(self, fake_redis: Any) -> None:
        memory = MemoryService(fake_redis)
        await memory.append_exchange(
            "sess-multi",
            "How much annual leave do employees get?",
            "Employees get 25 days.",
        )

        calls: list[dict[str, Any]] = []

        def handler(kwargs: dict[str, Any]) -> Any:
            calls.append(kwargs)
            if len(calls) == 1:
                return text_response("employee paid annual leave allowance")
            return text_response("They get 25 days. [1]")

        qdrant = FakeQdrant()
        qdrant.query_results = [make_point()]
        llm = FakeLLMClient(handler)
        service = RagService(
            memory=memory,
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
            qdrant=qdrant,  # type: ignore[arg-type]
            client=llm,  # type: ignore[arg-type]
        )

        result = await service.answer(session_id="sess-multi", question="And part-time staff?")

        assert result.standalone_query == "employee paid annual leave allowance"
        assert len(llm.chat.completions.calls) == 2
        reformulation_messages = llm.chat.completions.calls[0]["messages"]
        assert "How much annual leave do employees get?" in str(reformulation_messages)
        assert llm.chat.completions.calls[1]["messages"][0]["role"] == "system"
        assert any(
            message["role"] == "assistant" and message["content"] == "Employees get 25 days."
            for message in llm.chat.completions.calls[1]["messages"]
        )
        assert len(await memory.get_history("sess-multi")) == 4

    async def test_reformulation_failure_falls_back_to_the_question(self, fake_redis: Any) -> None:
        memory = MemoryService(fake_redis)
        await memory.append_exchange("sess-fallback", "Earlier question", "Earlier answer")

        calls: list[dict[str, Any]] = []

        def handler(kwargs: dict[str, Any]) -> Any:
            calls.append(kwargs)
            if len(calls) == 1:
                raise connection_error()
            return text_response("Answer. [1]")

        qdrant = FakeQdrant()
        qdrant.query_results = [make_point()]
        service = RagService(
            memory=memory,
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
            qdrant=qdrant,  # type: ignore[arg-type]
            client=FakeLLMClient(handler),  # type: ignore[arg-type]
        )

        result = await service.answer(session_id="sess-fallback", question="Follow-up question?")

        assert result.standalone_query == "Follow-up question?"
        assert result.answer == "Answer. [1]"

    async def test_blank_reformulation_falls_back_to_the_question(self, fake_redis: Any) -> None:
        memory = MemoryService(fake_redis)
        await memory.append_exchange("sess-blank", "Earlier question", "Earlier answer")

        service = RagService(
            memory=memory,
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
            qdrant=FakeQdrant(),  # type: ignore[arg-type]
            client=FakeLLMClient(  # type: ignore[arg-type]
                scripted_handler([text_response("   "), text_response("ok")])
            ),
        )

        result = await service.answer(session_id="sess-blank", question="Another question?")

        assert result.standalone_query == "Another question?"


class TestCompletionFailures:
    async def test_llm_connection_failure_raises_llm_error(self, fake_redis: Any) -> None:
        def handler(_: dict[str, Any]) -> Any:
            raise connection_error()

        qdrant = FakeQdrant()
        qdrant.query_results = [make_point()]
        service, _, _ = build_service(handler=handler, fake_redis=fake_redis, qdrant=qdrant)

        with pytest.raises(LLMError, match="LLM provider request failed"):
            await service.answer(session_id="sess-llm", question="Question?")

    async def test_empty_completion_raises_llm_error(self, fake_redis: Any) -> None:
        qdrant = FakeQdrant()
        qdrant.query_results = [make_point()]
        service, _, _ = build_service(
            handler=scripted_handler([text_response("  ")]),
            fake_redis=fake_redis,
            qdrant=qdrant,
        )

        with pytest.raises(LLMError, match="empty response"):
            await service.answer(session_id="sess-empty-llm", question="Question?")

    async def test_empty_embedding_result_raises_retrieval_error(self, fake_redis: Any) -> None:
        class EmptyEmbedder(FakeEmbedder):
            async def embed_texts(self, texts: list[str]) -> list[list[float]]:
                return []

        service, _, _ = build_service(
            handler=scripted_handler([]),
            fake_redis=fake_redis,
            embedder=EmptyEmbedder(),
        )

        with pytest.raises(RetrievalError, match="embed the search query"):
            await service.answer(session_id="sess-embed", question="Question?")


class TestChatErrorMapping:
    async def test_embedding_failure_returns_502(self, api: ApiContext) -> None:
        api.embedder.error = EmbeddingError("provider down")

        response = await api.client.post(
            "/api/v1/chat",
            json={"session_id": "sess-502", "message": "Question?"},
        )

        assert response.status_code == 502
        assert "provider down" in response.json()["detail"]

    async def test_llm_failure_returns_502(
        self, api: ApiContext, install_llm: LLMInstaller
    ) -> None:
        def handler(_: dict[str, Any]) -> Any:
            raise connection_error()

        install_llm(handler)
        api.qdrant.query_results = [make_point()]

        response = await api.client.post(
            "/api/v1/chat",
            json={"session_id": "sess-llm-502", "message": "Question?"},
        )

        assert response.status_code == 502
        assert "LLM provider request failed" in response.json()["detail"]

    async def test_qdrant_failure_returns_503(self, api: ApiContext) -> None:
        api.qdrant.query_error = RuntimeError("qdrant down")

        response = await api.client.post(
            "/api/v1/chat",
            json={"session_id": "sess-qdrant-503", "message": "Question?"},
        )

        assert response.status_code == 503
        assert "vector store" in response.json()["detail"].lower()

    async def test_redis_failure_returns_503(self, api: ApiContext) -> None:
        from redis.exceptions import ConnectionError as RedisConnectionError

        from app.core.redis_client import get_redis_dependency

        class BrokenRedis:
            async def lrange(self, *_: Any, **__: Any) -> list[str]:
                raise RedisConnectionError("redis down")

        api.app.dependency_overrides[get_redis_dependency] = lambda: BrokenRedis()

        response = await api.client.post(
            "/api/v1/chat",
            json={"session_id": "sess-redis-503", "message": "Question?"},
        )

        assert response.status_code == 503
        assert "temporarily unavailable" in response.json()["detail"]
