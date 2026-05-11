"""Shared LLM interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from pydantic import BaseModel


class LLMError(Exception):
    """Wrapped SDK failure from a provider ``complete`` / ``stream`` call."""


class LLMClient(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> BaseModel | str:
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> AsyncGenerator[str, None]:
        pass
