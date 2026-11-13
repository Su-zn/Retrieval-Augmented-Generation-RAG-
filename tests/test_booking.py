"""Tests for booking validation, persistence, and the tool-call loop."""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.domain import InterviewBooking
from app.services.booking_service import (
    BookingPersistenceError,
    BookingService,
    BookingValidationError,
)
from tests.conftest import (
    ApiContext,
    LLMInstaller,
    scripted_handler,
    text_response,
    tool_call_response,
)

FUTURE_DATE = (dt.date.today() + dt.timedelta(days=30)).isoformat()


def booking_arguments(**overrides: str) -> dict[str, str]:
    """Return a valid booking payload with optional field overrides."""
    values = {
        "name": "Jane Doe",
        "email": "jane.doe@example.com",
        "date": FUTURE_DATE,
        "time": "15:30",
    }
    values.update(overrides)
    return values


def booking_arguments_json(**overrides: str) -> str:
    """Return a valid booking payload serialised as tool-call JSON."""
    return json.dumps(booking_arguments(**overrides))


class BrokenSession:
    """Session double whose commit always fails."""

    def __init__(self) -> None:
        self.rolled_back = False

    def add(self, _: object) -> None:
        return None

    async def commit(self) -> None:
        raise SQLAlchemyError("database down")

    async def rollback(self) -> None:
        self.rolled_back = True

    async def refresh(self, _: object) -> None:
        return None


async def _booking_count(session_factory: async_sessionmaker[AsyncSession]) -> int:
    async with session_factory() as session:
        count = await session.scalar(select(func.count()).select_from(InterviewBooking))
    return int(count or 0)


class TestBookingValidation:
    def test_valid_arguments_are_parsed(self) -> None:
        parsed = BookingService.parse_arguments(booking_arguments_json())
        assert parsed.name == "Jane Doe"
        assert parsed.date == FUTURE_DATE

    @pytest.mark.parametrize(
        ("arguments", "match"),
        [
            ('{"email":"jane@example.com"}', "Invalid booking details"),
            (booking_arguments_json(email="not-an-email"), "email"),
            (booking_arguments_json(date="2030-02-30"), "date"),
            (booking_arguments_json(time="25:00"), "time"),
            (
                booking_arguments_json(date=(dt.date.today() - dt.timedelta(days=1)).isoformat()),
                "today or later",
            ),
            ("not json", "Invalid booking details"),
        ],
    )
    def test_invalid_arguments_are_rejected(self, arguments: str, match: str) -> None:
        with pytest.raises(BookingValidationError, match=match):
            BookingService.parse_arguments(arguments)


class TestBookingPersistence:
    async def test_valid_booking_is_persisted(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        async with session_factory() as session:
            service = BookingService(session)
            booking = await service.book(
                session_id="sess-db",
                arguments=booking_arguments_json(),
                conversation_text="Book jane.doe@example.com for an interview.",
            )

            assert booking.session_id == "sess-db"
            assert booking.candidate_name == "Jane Doe"
            assert str(booking.candidate_email) == "jane.doe@example.com"
            assert booking.interview_date == dt.date.fromisoformat(FUTURE_DATE)
            assert booking.interview_time == dt.time(15, 30)
            assert booking.created_at is not None

        assert await _booking_count(session_factory) == 1

    async def test_invented_email_is_rejected_without_insert(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        async with session_factory() as session:
            service = BookingService(session)
            with pytest.raises(BookingValidationError, match="supplied by the user"):
                await service.book(
                    session_id="sess-db",
                    arguments=booking_arguments_json(email="invented@example.com"),
                    conversation_text="Please book an interview for me.",
                )

        assert await _booking_count(session_factory) == 0

    async def test_commit_failure_maps_to_persistence_error(self) -> None:
        session = BrokenSession()
        service = BookingService(session)  # type: ignore[arg-type]

        with pytest.raises(BookingPersistenceError, match="Failed to persist"):
            await service.book(
                session_id="sess-db",
                arguments=booking_arguments_json(),
                conversation_text="Book jane.doe@example.com.",
            )
        assert session.rolled_back is True


class TestBookingToolLoop:
    async def test_tool_call_persists_booking_and_returns_it(
        self,
        api: ApiContext,
        fake_redis: Any,
        install_llm: LLMInstaller,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        llm = install_llm(
            scripted_handler(
                [
                    tool_call_response("book_interview", booking_arguments_json()),
                    text_response("Your interview is booked. [1]"),
                ]
            )
        )
        api.qdrant.query_results = []
        message = (
            "Book an interview for Jane Doe, jane.doe@example.com, " f"on {FUTURE_DATE} at 15:30."
        )

        response = await api.client.post(
            "/api/v1/chat",
            json={"session_id": "book-success", "message": message},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["answer"] == "Your interview is booked. [1]"
        assert body["booking"] is not None
        assert body["booking"]["session_id"] == "book-success"
        assert body["booking"]["candidate_name"] == "Jane Doe"
        assert body["booking"]["candidate_email"] == "jane.doe@example.com"
        assert body["booking"]["interview_date"] == FUTURE_DATE
        assert await _booking_count(session_factory) == 1
        assert await fake_redis.llen("chat:history:book-success") == 2

        calls = llm.chat.completions.calls
        assert len(calls) == 2
        assert calls[0]["tools"] is not None
        assert "tools" not in calls[1]
        assert any(message["role"] == "tool" for message in calls[1]["messages"])

    async def test_incomplete_tool_arguments_do_not_book(
        self,
        api: ApiContext,
        install_llm: LLMInstaller,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        install_llm(
            scripted_handler(
                [
                    tool_call_response("book_interview", "{}"),
                    text_response("Please provide your full name, email, date, and time."),
                ]
            )
        )
        api.qdrant.query_results = []

        response = await api.client.post(
            "/api/v1/chat",
            json={"session_id": "book-incomplete", "message": "Book me an interview."},
        )

        assert response.status_code == 200, response.text
        assert response.json()["booking"] is None
        assert "Please provide" in response.json()["answer"]
        assert await _booking_count(session_factory) == 0

    async def test_invented_contact_details_do_not_book(
        self,
        api: ApiContext,
        install_llm: LLMInstaller,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        install_llm(
            scripted_handler(
                [
                    tool_call_response(
                        "book_interview",
                        booking_arguments_json(email="invented@example.com"),
                    ),
                    text_response("Please provide the email address to use."),
                ]
            )
        )
        api.qdrant.query_results = []

        response = await api.client.post(
            "/api/v1/chat",
            json={"session_id": "book-invented", "message": "Book an interview next month."},
        )

        assert response.status_code == 200, response.text
        assert response.json()["booking"] is None
        assert await _booking_count(session_factory) == 0
