"""
In-memory session memory and optional per-session query deduplication.

No database — suitable for demo scale; see README for tradeoffs.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from threading import Lock

from src.models import ClassifierResult


@dataclass
class ConversationTurn:
    """One completed exchange in a session (assistant text is a summary, not full SSE)."""

    user: str
    assistant: str
    agent: str
    entities: dict


class SessionStore:
    """
    Thread-safe in-memory conversation store.

    Keys: ``session_id`` (str). Values: ``list[ConversationTurn]`` capped at ``MAX_TURNS``.
    """

    MAX_TURNS = 10

    def __init__(self) -> None:
        self._store: dict[str, list[ConversationTurn]] = {}
        self._lock = asyncio.Lock()

    async def get_turns(self, session_id: str) -> list[ConversationTurn]:
        """Return all turns for session, or empty list."""
        async with self._lock:
            return list(self._store.get(session_id, []))

    async def add_turn(self, session_id: str, turn: ConversationTurn) -> None:
        """Append turn. Trim to ``MAX_TURNS`` (most recent) if needed."""
        async with self._lock:
            turns = self._store.setdefault(session_id, [])
            turns.append(turn)
            if len(turns) > self.MAX_TURNS:
                self._store[session_id] = turns[-self.MAX_TURNS :]

    async def get_prior_user_turns(self, session_id: str) -> list[str]:
        """Return user messages in order, for passing to the classifier."""
        async with self._lock:
            turns = self._store.get(session_id, [])
            return [t.user for t in turns]

    async def get_last_entities(self, session_id: str) -> dict:
        """Return entities from the most recent turn (for entity carryover)."""
        async with self._lock:
            turns = self._store.get(session_id, [])
            if not turns:
                return {}
            return dict(turns[-1].entities)

    async def clear(self, session_id: str) -> None:
        """Remove session (useful for testing)."""
        async with self._lock:
            self._store.pop(session_id, None)


class QueryCache:
    """
    Intra-session identical-query deduplication.

    If the same ``(session_id, query)`` pair is seen within ``TTL`` seconds,
    return a cached ``ClassifierResult``.
    """

    TTL = 60

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], tuple[ClassifierResult, float]] = {}
        self._lock = Lock()

    def _evict_expired(self) -> None:
        """Remove entries older than ``TTL``."""
        now = time.monotonic()
        expired = [k for k, (_, ts) in self._cache.items() if now - ts >= self.TTL]
        for k in expired:
            del self._cache[k]

    def get(self, session_id: str, query: str) -> ClassifierResult | None:
        """Return cached result if within ``TTL``, else ``None``."""
        key = (session_id, query.strip())
        with self._lock:
            self._evict_expired()
            item = self._cache.get(key)
            if item is None:
                return None
            result, ts = item
            if time.monotonic() - ts >= self.TTL:
                del self._cache[key]
                return None
            return result.model_copy(deep=True)

    def set(self, session_id: str, query: str, result: ClassifierResult) -> None:
        """Cache result with current timestamp."""
        key = (session_id, query.strip())
        with self._lock:
            self._evict_expired()
            self._cache[key] = (result.model_copy(deep=True), time.monotonic())


session_store = SessionStore()
query_cache = QueryCache()
