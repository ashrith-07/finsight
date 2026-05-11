"""LLM provider abstractions for Valura AI."""

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
    """
    Returns OpenAILLMClient if OPENAI_API_KEY is set in environment.

    Model id: ``model`` arg if provided, else ``LLM_MODEL``, else ``OPENAI_MODEL``,
    defaulting to ``gpt-4o-mini``.

    Falls back to MockLLMClient with an empty response queue otherwise.
    Never raises — always returns a usable client.
    """
    if os.environ.get("OPENAI_API_KEY", "").strip():
        # Lazy import: keeps ``openai`` unloaded when only the mock path is used (e.g. CI).
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
