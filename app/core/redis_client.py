"""Async Redis client lifecycle for conversational memory."""

import logging
from collections.abc import AsyncIterator

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.config import settings

logger = logging.getLogger(__name__)

_redis_client: Redis | None = None


def get_redis() -> Redis:
    """Return the process-wide Redis client, or raise if uninitialised."""
    if _redis_client is None:
        raise RuntimeError("Redis client has not been initialised. Call init_redis() first.")
    return _redis_client


async def get_redis_dependency() -> AsyncIterator[Redis]:
    """FastAPI dependency that exposes the shared Redis client."""
    yield get_redis()


async def init_redis() -> None:
    """Create the Redis client and verify connectivity.

    A failed ping is logged rather than fatal so the API still starts and
    exposes its OpenAPI docs; endpoints requiring memory surface the outage
    as a structured 503 instead.
    """
    global _redis_client
    if _redis_client is None:
        _redis_client = Redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            health_check_interval=30,
        )

    try:
        await _redis_client.ping()
        logger.info("Connected to Redis at %s.", settings.redis_url)
    except RedisError as exc:
        logger.warning("Redis is not reachable at startup: %s", exc)


async def close_redis() -> None:
    """Close the Redis connection pool."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None
        logger.info("Redis connection closed.")
