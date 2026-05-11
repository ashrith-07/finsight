"""LLM factory and exports."""

from __future__ import annotations

import os

from src.llm.base import LLMClient, LLMError
from src.llm.mock_llm import MockExhaustedError, MockLLMClient


def _resolved_model(explicit: str | None) -> str:
    if explicit:
        return explicit
    return (
        os.environ.get("LLM_MODEL", "").strip()
        or os.environ.get("OPENAI_MODEL", "").strip()
        or "gpt-4o-mini"
    )


def get_llm_client(model: str | None = None) -> LLMClient:
    """OpenAI client when ``OPENAI_API_KEY`` is set; else ``MockLLMClient``. Model: arg → ``LLM_MODEL`` → ``OPENAI_MODEL`` → ``gpt-4o-mini``. Never raises."""
    if os.environ.get("OPENAI_API_KEY", "").strip():
        # Avoid importing ``openai`` on the mock-only path (tests / CI).
        from src.llm.openai_llm import OpenAILLMClient

        return OpenAILLMClient(model=_resolved_model(model))
    return MockLLMClient()


__all__ = [
    "LLMClient",
    "LLMError",
    "MockExhaustedError",
    "MockLLMClient",
    "get_llm_client",
]
