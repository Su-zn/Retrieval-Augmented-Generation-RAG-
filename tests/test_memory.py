"""Tests for Redis-backed chat memory."""

from __future__ import annotations

from typing import Any

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.services.memory_service import (
    ChatMessage,
    MemoryService,
    MemoryStoreError,
    history_key,
)


class BrokenRedis:
    """Redis double whose reads and writes always fail."""

    async def lrange(self, *_: Any, **__: Any) -> list[str]:
        raise RedisConnectionError("redis down")

    def pipeline(self, *_: Any, **__: Any) -> Any:
        raise RedisConnectionError("redis down")

    async def delete(self, *_: Any, **__: Any) -> int:
        raise RedisConnectionError("redis down")


class TestHistoryKey:
    def test_key_is_namespaced_by_session(self) -> None:
        assert history_key("abc") == "chat:history:abc"


class TestMemoryService:
    async def test_new_session_has_empty_history(self, fake_redis: Any) -> None:
        memory = MemoryService(fake_redis)
        assert await memory.get_history("new-session") == []

    async def test_exchange_is_stored_in_order(self, fake_redis: Any) -> None:
        memory = MemoryService(fake_redis, max_messages=20, ttl_seconds=60)
        await memory.append_exchange("sess", "question", "answer")

        history = await memory.get_history("sess")
        assert history == [
            ChatMessage(role="user", content="question"),
            ChatMessage(role="assistant", content="answer"),
        ]

    async def test_sliding_window_keeps_newest_messages(self, fake_redis: Any) -> None:
        memory = MemoryService(fake_redis, max_messages=4, ttl_seconds=60)
        await memory.append_exchange("sess", "q1", "a1")
        await memory.append_exchange("sess", "q2", "a2")
        await memory.append_exchange("sess", "q3", "a3")

        history = await memory.get_history("sess")
        assert [(message.role, message.content) for message in history] == [
            ("user", "q2"),
            ("assistant", "a2"),
            ("user", "q3"),
            ("assistant", "a3"),
        ]
        assert len(await fake_redis.lrange(history_key("sess"), 0, -1)) == 4

    async def test_ttl_is_set_and_refreshed(self, fake_redis: Any) -> None:
        memory = MemoryService(fake_redis, max_messages=20, ttl_seconds=120)
        await memory.append_exchange("sess", "q", "a")

        ttl = await fake_redis.ttl(history_key("sess"))
        assert 0 < ttl <= 120

    async def test_sessions_are_isolated(self, fake_redis: Any) -> None:
        memory = MemoryService(fake_redis)
        await memory.append_exchange("one", "q1", "a1")
        await memory.append_exchange("two", "q2", "a2")

        assert [message.content for message in await memory.get_history("one")] == ["q1", "a1"]
        assert [message.content for message in await memory.get_history("two")] == ["q2", "a2"]

    async def test_malformed_entries_are_skipped(self, fake_redis: Any) -> None:
        memory = MemoryService(fake_redis)
        await memory.append_exchange("sess", "q", "a")
        await fake_redis.rpush(
            history_key("sess"),
            "not-json",
            '{"role":"system","content":"ignored"}',
            '{"role":"user"}',
        )

        history = await memory.get_history("sess")
        assert [(message.role, message.content) for message in history] == [
            ("user", "q"),
            ("assistant", "a"),
        ]

    async def test_clear_deletes_the_session(self, fake_redis: Any) -> None:
        memory = MemoryService(fake_redis)
        await memory.append_exchange("sess", "q", "a")
        await memory.clear("sess")

        assert await memory.get_history("sess") == []
        assert not await fake_redis.exists(history_key("sess"))

    async def test_read_failure_maps_to_memory_store_error(self) -> None:
        memory = MemoryService(BrokenRedis())  # type: ignore[arg-type]
        with pytest.raises(MemoryStoreError, match="temporarily unavailable"):
            await memory.get_history("sess")

    async def test_write_failure_maps_to_memory_store_error(self) -> None:
        memory = MemoryService(BrokenRedis())  # type: ignore[arg-type]
        with pytest.raises(MemoryStoreError, match="temporarily unavailable"):
            await memory.append_exchange("sess", "q", "a")

    async def test_clear_failure_maps_to_memory_store_error(self) -> None:
        memory = MemoryService(BrokenRedis())  # type: ignore[arg-type]
        with pytest.raises(MemoryStoreError, match="temporarily unavailable"):
            await memory.clear("sess")
