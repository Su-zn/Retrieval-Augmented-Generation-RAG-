"""Interview booking via a validated OpenAI-compatible tool call."""

from __future__ import annotations

import datetime as dt
import logging

from openai.types.chat import ChatCompletionToolParam
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.domain import InterviewBooking
from app.models.schemas import InterviewBookingRead, InterviewBookingToolArgs

logger = logging.getLogger(__name__)

BOOK_INTERVIEW_TOOL_NAME = "book_interview"

BOOK_INTERVIEW_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": BOOK_INTERVIEW_TOOL_NAME,
        "description": (
            "Book a confirmed interview slot. Only call this after the user has "
            "provided all four details: full name, email address, date in "
            "YYYY-MM-DD format, and time in HH:MM 24-hour format. Never invent or "
            "guess missing details; ask the user for them instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Candidate full name.",
                },
                "email": {
                    "type": "string",
                    "description": "Candidate email address.",
                },
                "date": {
                    "type": "string",
                    "description": "Interview date in YYYY-MM-DD format.",
                },
                "time": {
                    "type": "string",
                    "description": "Interview start time in HH:MM 24-hour format.",
                },
            },
            "required": ["name", "email", "date", "time"],
            "additionalProperties": False,
        },
    },
}


class BookingValidationError(Exception):
    """Raised when tool arguments are malformed, incomplete, or invalid."""


class BookingPersistenceError(Exception):
    """Raised when a validated booking cannot be persisted."""


class BookingService:
    """Validate tool arguments and persist interview bookings."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    def parse_arguments(arguments: str) -> InterviewBookingToolArgs:
        """Parse and validate raw JSON arguments emitted by the LLM.

        Raises:
            BookingValidationError: on malformed JSON, schema violations, or a
                date in the past.
        """
        try:
            parsed = InterviewBookingToolArgs.model_validate_json(arguments)
        except ValidationError as exc:
            raise BookingValidationError(_format_validation_error(exc)) from exc

        if dt.date.fromisoformat(parsed.date) < dt.date.today():
            raise BookingValidationError(
                "Invalid booking details: date: Interview date must be today or later."
            )
        return parsed

    async def book(
        self,
        *,
        session_id: str,
        arguments: str,
        conversation_text: str | None = None,
    ) -> InterviewBookingRead:
        """Validate ``arguments`` and insert one booking row.

        ``conversation_text`` (the user's turns) lets the service reject invented
        contact details instead of trusting a model that fills in placeholders.

        Raises:
            BookingValidationError: if the arguments are not a valid booking.
            BookingPersistenceError: if the insert/commit fails.
        """
        parsed = self.parse_arguments(arguments)

        if (
            conversation_text is not None
            and str(parsed.email).lower() not in conversation_text.lower()
        ):
            raise BookingValidationError(
                "Invalid booking details: email: the email address must be supplied "
                "by the user, not invented."
            )

        booking = InterviewBooking(
            session_id=session_id,
            **parsed.as_booking_values(),
        )
        self._session.add(booking)

        try:
            await self._session.commit()
            await self._session.refresh(booking)
        except SQLAlchemyError as exc:
            await self._session.rollback()
            logger.warning("Failed to persist interview booking: %s", exc)
            raise BookingPersistenceError("Failed to persist the interview booking.") from exc

        logger.info(
            "Booked interview %s for session %s (%s on %s at %s).",
            booking.id,
            session_id,
            parsed.name,
            parsed.date,
            parsed.time,
        )
        return InterviewBookingRead.model_validate(booking)


def _format_validation_error(exc: ValidationError) -> str:
    """Render Pydantic errors as a compact message the LLM can act on."""
    details: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "arguments"
        details.append(f"{location}: {error['msg']}")
    joined = "; ".join(details)
    return f"Invalid booking details: {joined}"
