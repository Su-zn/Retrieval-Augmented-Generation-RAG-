"""Async SQLAlchemy engine, session factory, and FastAPI dependency."""

import logging
from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import StaticPool

from app.core.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base shared by every ORM model."""


def _engine_options(database_url: str) -> dict[str, Any]:
    """Build dialect-specific engine options.

    SQLite is used for zero-friction local runs; PostgreSQL in Docker Compose.
    In-memory SQLite requires a ``StaticPool`` so all sessions share one
    connection.
    """
    if database_url.startswith("sqlite"):
        options: dict[str, Any] = {"connect_args": {"check_same_thread": False}}
        if ":memory:" in database_url:
            options["poolclass"] = StaticPool
        return options
    return {"pool_pre_ping": True, "pool_size": 10, "max_overflow": 20}


engine: AsyncEngine = create_async_engine(
    settings.database_url,
    echo=settings.database_echo,
    **_engine_options(settings.database_url),
)

SessionFactory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a transaction-scoped async session."""
    async with SessionFactory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create all tables registered on :class:`Base`.

    ORM models are imported here so that they are registered on the shared
    metadata before ``create_all`` runs.
    """
    from app.models import domain  # noqa: F401 - registers ORM models on Base.metadata

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    logger.info("Database schema ready (%s).", engine.url.render_as_string(hide_password=True))


async def close_db() -> None:
    """Dispose the engine and release all pooled connections."""
    await engine.dispose()
    logger.info("Database connections closed.")
