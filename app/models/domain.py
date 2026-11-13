"""SQLAlchemy ORM models for document metadata and interview bookings.

The models use portable SQLAlchemy 2.0 types so the same schema runs on both
SQLite (zero-friction local evaluation) and PostgreSQL (Docker Compose).
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, time

from sqlalchemy import Date, DateTime, Integer, String, Time, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class DocumentMetadata(Base):
    """Relational record describing one ingested document."""

    __tablename__ = "document_metadata"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str] = mapped_column(String(128), nullable=False)
    chunking_strategy: Mapped[str] = mapped_column(String(32), nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        return (
            f"DocumentMetadata(id={self.id!r}, filename={self.filename!r}, "
            f"chunking_strategy={self.chunking_strategy!r}, chunk_count={self.chunk_count!r})"
        )


class InterviewBooking(Base):
    """A confirmed interview slot collected through LLM tool calling."""

    __tablename__ = "interview_bookings"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    session_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    candidate_name: Mapped[str] = mapped_column(String(200), nullable=False)
    candidate_email: Mapped[str] = mapped_column(String(320), nullable=False)
    interview_date: Mapped[date] = mapped_column(Date, nullable=False)
    interview_time: Mapped[time] = mapped_column(Time, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        return (
            f"InterviewBooking(id={self.id!r}, session_id={self.session_id!r}, "
            f"candidate_email={self.candidate_email!r}, "
            f"interview_date={self.interview_date!r}, interview_time={self.interview_time!r})"
        )
