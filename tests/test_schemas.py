"""Pydantic schema tests for chat input and booking tool arguments."""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from app.models.schemas import (
    ChatRequest,
    InterviewBookingToolArgs,
    SourceChunk,
)


class TestChatRequest:
    def test_values_are_stripped(self) -> None:
        request = ChatRequest(session_id="  sess-1  ", message="  Hello  ")
        assert request.session_id == "sess-1"
        assert request.message == "Hello"

    @pytest.mark.parametrize("field", ["session_id", "message"])
    def test_blank_values_are_rejected(self, field: str) -> None:
        values = {"session_id": "sess-1", "message": "Hello"}
        values[field] = "   "
        with pytest.raises(ValidationError, match="must not be blank"):
            ChatRequest(**values)


class TestInterviewBookingToolArgs:
    def test_valid_arguments_convert_to_db_values(self) -> None:
        args = InterviewBookingToolArgs(
            name="Jane Doe",
            email="jane.doe@example.com",
            date="2030-06-15",
            time="15:30",
        )
        values = args.as_booking_values()

        assert values["candidate_name"] == "Jane Doe"
        assert values["candidate_email"] == "jane.doe@example.com"
        assert values["interview_date"] == dt.date(2030, 6, 15)
        assert values["interview_time"] == dt.time(15, 30)

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("name", "   ", "must not be blank"),
            ("email", "not-an-email", "valid email"),
            ("date", "2030-02-30", "valid calendar date"),
            ("date", "15/06/2030", "String should match pattern"),
            ("time", "25:00", "String should match pattern"),
            ("time", "3:30 PM", "String should match pattern"),
        ],
    )
    def test_invalid_values_are_rejected(self, field: str, value: str, match: str) -> None:
        values = {
            "name": "Jane Doe",
            "email": "jane.doe@example.com",
            "date": "2030-06-15",
            "time": "15:30",
        }
        values[field] = value
        with pytest.raises(ValidationError, match=match):
            InterviewBookingToolArgs(**values)


class TestSourceChunk:
    def test_score_bounds_are_enforced(self) -> None:
        with pytest.raises(ValidationError):
            SourceChunk(
                document_id="00000000-0000-0000-0000-000000000000",
                chunk_index=0,
                text="chunk",
                score=1.1,
            )
