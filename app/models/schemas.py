"""Pydantic v2 schemas for API contracts and LLM tool arguments."""

from __future__ import annotations

import datetime as dt
from typing import Literal, TypedDict
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

ChunkingStrategy = Literal["fixed", "semantic"]


class BookingValues(TypedDict):
    """DB-ready field mapping used to construct an ``InterviewBooking`` row."""

    candidate_name: str
    candidate_email: str
    interview_date: dt.date
    interview_time: dt.time


class IngestionResponse(BaseModel):
    """Response returned by ``POST /api/v1/ingest``."""

    document_id: UUID
    filename: str
    chunking_strategy: ChunkingStrategy
    chunks_indexed: int = Field(ge=0)
    message: str = "Document ingested and indexed successfully."


class ChatRequest(BaseModel):
    """Request body for ``POST /api/v1/chat``."""

    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=8_000)

    @field_validator("session_id", "message")
    @classmethod
    def _strip_and_require_content(cls, value: str) -> str:
        """Normalise whitespace and reject blank values."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned


class SourceChunk(BaseModel):
    """A retrieved Qdrant chunk, attributed to its source document."""

    document_id: UUID
    filename: str | None = None
    chunk_index: int = Field(ge=0)
    text: str
    score: float = Field(ge=-1.0, le=1.0, description="Cosine similarity score.")


class InterviewBookingRead(BaseModel):
    """Serialised interview booking returned to API clients."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    session_id: str
    candidate_name: str
    candidate_email: EmailStr
    interview_date: dt.date
    interview_time: dt.time
    created_at: dt.datetime


class ChatResponse(BaseModel):
    """Response returned by ``POST /api/v1/chat``."""

    session_id: str
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    booking: InterviewBookingRead | None = None


class InterviewBookingToolArgs(BaseModel):
    """Validated arguments for the ``book_interview`` LLM tool.

    The LLM emits JSON, so date/time stay strings with strict format patterns;
    the validators additionally reject well-formed but impossible values
    (for example ``2026-02-30`` or ``25:00``).
    """

    name: str = Field(min_length=1, max_length=200, description="Candidate full name.")
    email: EmailStr = Field(description="Candidate email address.")
    date: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Interview date in YYYY-MM-DD format.",
    )
    time: str = Field(
        pattern=r"^([01]\d|2[0-3]):[0-5]\d$",
        description="Interview time in HH:MM 24-hour format.",
    )

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned

    @field_validator("date")
    @classmethod
    def _validate_calendar_date(cls, value: str) -> str:
        try:
            dt.date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("must be a valid calendar date (YYYY-MM-DD)") from exc
        return value

    @field_validator("time")
    @classmethod
    def _validate_clock_time(cls, value: str) -> str:
        try:
            dt.time.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("must be a valid 24-hour time (HH:MM)") from exc
        return value

    def as_booking_values(self) -> BookingValues:
        """Return DB-ready values (parsed date/time) for ``InterviewBooking``."""
        return {
            "candidate_name": self.name,
            "candidate_email": str(self.email),
            "interview_date": dt.date.fromisoformat(self.date),
            "interview_time": dt.time.fromisoformat(self.time),
        }
