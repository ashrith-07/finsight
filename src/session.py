"""In-memory turns per session and optional (session_id, query) classifier dedupe."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from threading import Lock

from src.models import ClassifierResult


@dataclass
class ConversationTurn:
    user: str
    assistant: str
    agent: str
    entities: dict


class SessionStore:
    MAX_TURNS = 10

    def __init__(self) -> None:
        self._store: dict[str, list[ConversationTurn]] = {}
        self._lock = asyncio.Lock()

    async def get_turns(self, session_id: str) -> list[ConversationTurn]:
        async with self._lock:
            return list(self._store.get(session_id, []))

    async def add_turn(self, session_id: str, turn: ConversationTurn) -> None:
        async with self._lock:
            turns = self._store.setdefault(session_id, [])
            turns.append(turn)
            if len(turns) > self.MAX_TURNS:
                self._store[session_id] = turns[-self.MAX_TURNS :]

    async def get_prior_user_turns(self, session_id: str) -> list[str]:
        async with self._lock:
            turns = self._store.get(session_id, [])
            return [t.user for t in turns]

    async def get_last_entities(self, session_id: str) -> dict:
        async with self._lock:
            turns = self._store.get(session_id, [])
            if not turns:
                return {}
            return turns[-1].entities

    async def clear(self, session_id: str) -> None:
        async with self._lock:
            self._store.pop(session_id, None)


class QueryCache:
    TTL = 60

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], tuple[ClassifierResult, float]] = {}
        self._lock = Lock()

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, (_, ts) in self._cache.items() if now - ts >= self.TTL]
        for k in expired:
            del self._cache[k]

    def get(self, session_id: str, query: str) -> ClassifierResult | None:
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
        key = (session_id, query.strip())
        with self._lock:
            self._evict_expired()
            self._cache[key] = (result.model_copy(deep=True), time.monotonic())


session_store = SessionStore()
query_cache = QueryCache()
