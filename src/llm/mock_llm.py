"""Test and CI-friendly mock LLM."""

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import AsyncGenerator

from pydantic import BaseModel

from src.llm.base import LLMClient
from src.models import ClassifierResult


class MockExhaustedError(RuntimeError):
    """Raised when ``complete()`` is called after the mock response queue is empty."""


class MockLLMClient(LLMClient):
    """
    Deterministic LLM double: ``complete`` drains a queue; ``stream`` replays
    fixed text chunks. Safe for CI (no network, no API key).
    """

    def __init__(
        self,
        responses: list[BaseModel | str | list[BaseModel]] | None = None,
        stream_chunks: list[str] | None = None,
    ) -> None:
        self._responses: deque[BaseModel | str | list[BaseModel]] = deque(responses or ())
        self._stream_chunks: deque[str] = deque(stream_chunks or ())
        self._lock = threading.Lock()

    @classmethod
    def for_classifier(cls, result: ClassifierResult) -> MockLLMClient:
        """Build a client with one structured ``ClassifierResult`` queued for ``complete``."""
        return cls(responses=[result])

    async def complete(
        self,
        messages: list[dict],
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> BaseModel | str:
        with self._lock:
            if not self._responses:
                raise MockExhaustedError("Mock LLM response queue is empty")
            item = self._responses.popleft()

        if response_model is not None:
            if isinstance(item, response_model):
                return item
            if isinstance(item, str):
                return response_model.model_validate_json(item)
            if isinstance(item, BaseModel):
                return response_model.model_validate(item.model_dump())
            msg = f"Mock queue item type {type(item)!r} is incompatible with {response_model!r}"
            raise TypeError(msg)

        if isinstance(item, str):
            return item
        if isinstance(item, BaseModel):
            return item.model_dump_json()
        if isinstance(item, list):
            if not item:
                return "[]"
            if all(isinstance(x, BaseModel) for x in item):
                return json.dumps(
                    [x.model_dump() for x in item],
                    ensure_ascii=False,
                )
            return json.dumps(item, ensure_ascii=False, default=str)
        return str(item)

    async def stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> AsyncGenerator[str, None]:
        with self._lock:
            chunks = tuple(self._stream_chunks)
        for chunk in chunks:
            yield chunk
