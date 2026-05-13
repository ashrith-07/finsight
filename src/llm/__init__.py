"""LLM factory and exports."""

from __future__ import annotations

import os

from src.llm.agno_model import get_agno_model, get_agno_model_strong
from src.llm.base import LLMClient, LLMError
from src.llm.mock_llm import MockExhaustedError, MockLLMClient, SmartMockLLMClient


def _resolved_model(explicit: str | None) -> str:
    if explicit:
        return explicit
    return (
        os.environ.get("LLM_MODEL", "").strip()
        or os.environ.get("OPENAI_MODEL", "").strip()
        or "gpt-4o-mini"
    )


def get_llm_client(model: str | None = None) -> LLMClient:
    """Provider precedence: ``GROQ_API_KEY`` > ``OPENAI_API_KEY`` > ``SmartMockLLMClient`` (no key)."""
    if os.environ.get("GROQ_API_KEY", "").strip():
        # Lazy import keeps ``groq`` off the test/CI critical path.
        from src.llm.groq_llm import GroqLLMClient

        return GroqLLMClient(model=model)

    if os.environ.get("OPENAI_API_KEY", "").strip():
        from src.llm.openai_llm import OpenAILLMClient

        return OpenAILLMClient(model=_resolved_model(model))

    return SmartMockLLMClient()


__all__ = [
    "LLMClient",
    "LLMError",
    "MockExhaustedError",
    "MockLLMClient",
    "SmartMockLLMClient",
    "get_agno_model",
    "get_agno_model_strong",
    "get_llm_client",
]
