"""LLM provider abstractions for Valura AI."""

from __future__ import annotations

import os

from src.llm.base import LLMClient, LLMError
from src.llm.mock_llm import MockExhaustedError, MockLLMClient


def get_llm_client(model: str = "gpt-4o-mini") -> LLMClient:
    """
    Returns OpenAILLMClient if OPENAI_API_KEY is set in environment.

    Falls back to MockLLMClient with an empty response queue otherwise.
    Never raises — always returns a usable client.
    """
    if os.environ.get("OPENAI_API_KEY", "").strip():
        # Lazy import: keeps ``openai`` unloaded when only the mock path is used (e.g. CI).
        from src.llm.openai_llm import OpenAILLMClient

        return OpenAILLMClient(model=model)
    return MockLLMClient()


__all__ = [
    "LLMClient",
    "LLMError",
    "MockExhaustedError",
    "MockLLMClient",
    "get_llm_client",
]
