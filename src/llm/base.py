"""Abstract LLM interface shared by concrete providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from pydantic import BaseModel


class LLMError(Exception):
    """Raised when an OpenAI-backed completion or stream fails after wrapping the SDK error."""


class LLMClient(ABC):
    """Protocol-style ABC for all LLM providers (OpenAI, mocks, etc.)."""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> BaseModel | str:
        """Single structured or unstructured completion call."""

    @abstractmethod
    async def stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> AsyncGenerator[str, None]:
        """Yields text deltas for SSE streaming."""
