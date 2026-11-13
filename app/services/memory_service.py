"""Redis-backed conversational memory with a sliding window and TTL."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Literal, cast

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.config import settings

logger = logging.getLogger(__name__)

ChatRole = Literal["user", "assistant"]

_VALID_ROLES: frozenset[str] = frozenset({"user", "assistant"})


class MemoryStoreError(Exception):
    """Raised when chat history cannot be read from or written to Redis."""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One turn stored in a session's conversation history."""

    role: ChatRole
    content: str


def history_key(session_id: str) -> str:
    """Return the Redis list key for a chat session."""
    return f"chat:history:{session_id}"


class MemoryService:
    """Read and append bounded conversation history stored in Redis.

    Each session is a Redis list of JSON-encoded ``ChatMessage`` objects. Every
    append trims the list to the most recent ``max_messages`` entries and
    refreshes the TTL, so history is both bounded and self-expiring.
    """

    def __init__(
        self,
        client: Redis,
        *,
        max_messages: int | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        self._client = client
        self._max_messages = max_messages or settings.chat_history_max_messages
        self._ttl_seconds = ttl_seconds or settings.chat_history_ttl_seconds

    async def get_history(self, session_id: str) -> list[ChatMessage]:
        """Return the stored messages in chronological order.

        Malformed entries are skipped (and logged) rather than failing the whole
        request, so one bad write cannot poison a session.

        Raises:
            MemoryStoreError: if Redis is unreachable or rejects the read.
        """
        try:
            entries = await cast(
                Awaitable[list[str]],
                self._client.lrange(history_key(session_id), 0, -1),
            )
        except RedisError as exc:
            logger.warning("Failed to read chat history for session %s: %s", session_id, exc)
            raise MemoryStoreError("Chat memory is temporarily unavailable.") from exc

        messages: list[ChatMessage] = []
        for entry in entries:
            message = _decode_message(entry)
            if message is not None:
                messages.append(message)

        return messages[-self._max_messages :]

    async def append_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Append a user/assistant pair, trim to the window, and refresh TTL.

        Raises:
            MemoryStoreError: if Redis is unreachable or rejects the write.
        """
        key = history_key(session_id)
        payloads = [
            _encode_message(ChatMessage(role="user", content=user_message)),
            _encode_message(ChatMessage(role="assistant", content=assistant_message)),
        ]

        try:
            async with self._client.pipeline(transaction=True) as pipe:
                pipe.rpush(key, *payloads)
                pipe.ltrim(key, -self._max_messages, -1)
                pipe.expire(key, self._ttl_seconds)
                await pipe.execute()
        except RedisError as exc:
            logger.warning("Failed to append chat history for session %s: %s", session_id, exc)
            raise MemoryStoreError("Chat memory is temporarily unavailable.") from exc

    async def clear(self, session_id: str) -> None:
        """Delete a session's history.

        Raises:
            MemoryStoreError: if Redis is unreachable or rejects the delete.
        """
        try:
            await self._client.delete(history_key(session_id))
        except RedisError as exc:
            logger.warning("Failed to clear chat history for session %s: %s", session_id, exc)
            raise MemoryStoreError("Chat memory is temporarily unavailable.") from exc


def _encode_message(message: ChatMessage) -> str:
    """Serialise one message for storage in a Redis list."""
    return json.dumps({"role": message.role, "content": message.content}, ensure_ascii=False)


def _decode_message(entry: object) -> ChatMessage | None:
    """Parse one stored list entry, returning ``None`` when it is malformed."""
    if not isinstance(entry, str):
        logger.warning("Skipping non-text chat history entry of type %s.", type(entry).__name__)
        return None

    try:
        payload = json.loads(entry)
    except json.JSONDecodeError:
        logger.warning("Skipping malformed chat history entry.")
        return None

    if not isinstance(payload, dict):
        logger.warning("Skipping chat history entry that is not an object.")
        return None

    role = payload.get("role")
    content = payload.get("content")
    if role not in _VALID_ROLES or not isinstance(content, str):
        logger.warning("Skipping chat history entry with an invalid role/content pair.")
        return None

    return ChatMessage(role=cast(ChatRole, role), content=content)
