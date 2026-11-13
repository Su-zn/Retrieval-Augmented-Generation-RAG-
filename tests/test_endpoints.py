"""Tests for application wiring, docs, validation, and chat endpoint behavior."""

from __future__ import annotations

from typing import Any

import pytest

from app.api.v1.dependencies import get_booking_service
from app.services.booking_service import BookingPersistenceError
from app.services.rag_service import NO_CONTEXT_ANSWER
from tests.conftest import (
    ApiContext,
    LLMInstaller,
    make_point,
    scripted_handler,
    text_response,
    tool_call_response,
)


class TestApplicationWiring:
    async def test_health_probe_reports_ok(self, api: ApiContext) -> None:
        response = await api.client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["version"]
        assert body["environment"]

    async def test_swagger_ui_is_served(self, api: ApiContext) -> None:
        response = await api.client.get("/docs")

        assert response.status_code == 200
        assert "swagger" in response.text.lower()

    async def test_openapi_lists_every_route(self, api: ApiContext) -> None:
        response = await api.client.get("/openapi.json")

        assert response.status_code == 200
        paths = response.json()["paths"]
        assert "/health" in paths
        assert "/api/v1/ingest" in paths
        assert "/api/v1/chat" in paths
        chat_schema = response.json()["components"]["schemas"]["ChatResponse"]
        assert set(chat_schema["properties"]) == {
            "session_id",
            "answer",
            "sources",
            "booking",
        }

    async def test_unknown_route_returns_404(self, api: ApiContext) -> None:
        response = await api.client.get("/api/v1/nope")
        assert response.status_code == 404


class TestChatValidation:
    @pytest.mark.parametrize(
        "payload",
        [
            {"session_id": "sess", "message": "   "},
            {"session_id": "   ", "message": "Hello"},
            {"session_id": "sess"},
            {"message": "Hello"},
        ],
    )
    async def test_invalid_requests_return_422(
        self, api: ApiContext, payload: dict[str, str]
    ) -> None:
        response = await api.client.post("/api/v1/chat", json=payload)
        assert response.status_code == 422

    async def test_ingest_without_a_file_returns_422(self, api: ApiContext) -> None:
        response = await api.client.post("/api/v1/ingest")
        assert response.status_code == 422


class TestChatTurn:
    async def test_grounded_turn_returns_answer_and_sources(
        self, api: ApiContext, install_llm: LLMInstaller, fake_redis: Any
    ) -> None:
        install_llm(scripted_handler([text_response("Employees receive 25 days. [1]")]))
        api.qdrant.query_results = [make_point(filename="handbook.txt", chunk_index=2, score=0.91)]

        response = await api.client.post(
            "/api/v1/chat",
            json={
                "session_id": "sess-endpoint",
                "message": "How much annual leave do employees get?",
            },
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["session_id"] == "sess-endpoint"
        assert body["answer"] == "Employees receive 25 days. [1]"
        assert body["booking"] is None
        assert len(body["sources"]) == 1
        assert body["sources"][0]["filename"] == "handbook.txt"
        assert body["sources"][0]["chunk_index"] == 2
        assert body["sources"][0]["score"] == pytest.approx(0.91)
        assert await fake_redis.llen("chat:history:sess-endpoint") == 2

    async def test_empty_retrieval_returns_deterministic_refusal(
        self, api: ApiContext, install_llm: LLMInstaller
    ) -> None:
        llm = install_llm(scripted_handler([text_response("Paris is the capital.")]))
        api.qdrant.query_results = []

        response = await api.client.post(
            "/api/v1/chat",
            json={"session_id": "sess-no-context", "message": "Capital of France?"},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["answer"] == NO_CONTEXT_ANSWER
        assert body["sources"] == []
        assert body["booking"] is None
        assert llm.chat.completions.calls[0]["tools"] is not None

    async def test_booking_persistence_failure_returns_503(
        self, api: ApiContext, install_llm: LLMInstaller
    ) -> None:
        install_llm(scripted_handler([tool_call_response("book_interview", "{}")]))

        class FailingBookingService:
            async def book(self, **_: Any) -> Any:
                raise BookingPersistenceError("Failed to persist the interview booking.")

        api.app.dependency_overrides[get_booking_service] = FailingBookingService
        api.qdrant.query_results = []

        response = await api.client.post(
            "/api/v1/chat",
            json={"session_id": "sess-db-503", "message": "Book an interview."},
        )

        assert response.status_code == 503
        assert "Failed to persist" in response.json()["detail"]

    async def test_unknown_tool_is_reported_to_the_model(
        self, api: ApiContext, install_llm: LLMInstaller
    ) -> None:
        llm = install_llm(
            scripted_handler(
                [
                    tool_call_response("delete_everything", "{}"),
                    text_response("I can only book interviews."),
                ]
            )
        )
        api.qdrant.query_results = []

        response = await api.client.post(
            "/api/v1/chat",
            json={"session_id": "sess-unknown-tool", "message": "Do something else."},
        )

        assert response.status_code == 200, response.text
        assert response.json()["booking"] is None
        assert response.json()["answer"] == NO_CONTEXT_ANSWER
        tool_messages = [
            message
            for message in llm.chat.completions.calls[1]["messages"]
            if message["role"] == "tool"
        ]
        assert tool_messages
        assert "Unknown tool" in tool_messages[0]["content"]
