"""Short-term session turns (in-memory) + long-term Agno cross-session memory."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any

from src.logging_config import get_logger
from src.models import ClassifierResult

logger = get_logger("session")


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


class AgnoMemoryManager:
    """Long-term, cross-session user facts via Agno's ``MemoryManager``.

    Default backend is **SQLite** (``.agno_memory.db``) when the optional
    ``sqlalchemy`` dependency is available; otherwise we fall back to Agno's
    file-backed JSON store, and finally to an in-process store so the demo
    keeps working without any external dependency.
    """

    DEFAULT_DB_PATH = os.environ.get("FINSIGHT_MEMORY_DB", ".agno_memory.db")

    def __init__(self, db_path: str | None = None) -> None:
        self._enabled = False
        self._backend = "disabled"
        self._memory: Any = None
        self._db_path = db_path or self.DEFAULT_DB_PATH

        try:
            from agno.memory import MemoryManager  # noqa: WPS433 — lazy on purpose
        except Exception as e:
            logger.warning("Agno memory disabled (import failed): %s", e)
            return

        db = self._build_db()
        if db is None:
            return

        try:
            self._memory = MemoryManager(db=db)
            self._enabled = True
        except Exception as e:
            logger.warning("Agno MemoryManager init failed: %s", e)
            self._memory = None

    def _build_db(self) -> Any:
        # SQLite is the preferred persistent option; ``agno.db.sqlite`` requires
        # ``sqlalchemy`` which is not a hard dependency here.
        try:
            from agno.db.sqlite import SqliteDb

            self._backend = f"sqlite ({self._db_path})"
            return SqliteDb(db_file=self._db_path)
        except TypeError:
            try:
                from agno.db.sqlite import SqliteDb

                self._backend = f"sqlite ({self._db_path})"
                return SqliteDb(db_path=self._db_path)
            except Exception as e:
                logger.warning("Agno SqliteDb fallback failed: %s", e)
        except Exception as e:
            logger.info("Agno SqliteDb unavailable, trying JSON: %s", e)

        try:
            from agno.db.json import JsonDb

            json_path = self._db_path.rsplit(".", 1)[0] + ".json"
            self._backend = f"json ({json_path})"
            return JsonDb(db_path=json_path)
        except Exception as e:
            logger.info("Agno JsonDb unavailable, falling back to in-memory: %s", e)

        try:
            from agno.db.in_memory import InMemoryDb

            self._backend = "in_memory"
            return InMemoryDb()
        except Exception as e:
            logger.warning("Agno memory: no usable db backend (%s)", e)
            return None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def backend(self) -> str:
        return self._backend

    async def add_user_memory(
        self,
        user_id: str,
        conversation: str,
        response: str = "",
    ) -> None:
        """Persist a single fact from the latest turn. ``response`` is kept on the record's ``input``."""
        if not self._enabled or not self._memory or not user_id:
            return
        try:
            from agno.memory import UserMemory

            memory_text = (conversation or "").strip()
            if not memory_text:
                return
            record = UserMemory(
                memory=memory_text[:600],
                user_id=user_id,
                input=(response or "")[:600] or None,
            )
            await asyncio.to_thread(
                self._memory.add_user_memory, memory=record, user_id=user_id,
            )
        except Exception as e:
            logger.warning("Agno memory add failed for user=%s: %s", user_id, e)

    async def get_user_memories(self, user_id: str) -> list[str]:
        if not self._enabled or not self._memory or not user_id:
            return []
        try:
            records = await asyncio.to_thread(
                self._memory.get_user_memories, user_id=user_id,
            )
            return [r.memory for r in (records or []) if getattr(r, "memory", "")]
        except Exception as e:
            logger.warning("Agno memory read failed for user=%s: %s", user_id, e)
            return []

    @staticmethod
    def format_for_prompt(memories: list[str]) -> str:
        if not memories:
            return ""
        facts = "\n".join(f"- {m}" for m in memories[:10])
        return f"\nKnown facts about this user:\n{facts}\n"


session_store = SessionStore()
query_cache = QueryCache()
agno_memory = AgnoMemoryManager()
