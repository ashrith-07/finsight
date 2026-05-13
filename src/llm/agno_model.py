"""Agno-native ``Model`` instances for ``Agent`` / ``Team`` (separate from ``LLMClient``).

Imports stay lazy so test runs that only use ``MockLLMClient`` avoid loading Agno model modules.
"""

from __future__ import annotations

import os
from typing import Any


def _openai_id_default() -> str:
    return (
        os.environ.get("LLM_MODEL", "").strip()
        or os.environ.get("OPENAI_MODEL", "").strip()
        or "gpt-4o-mini"
    )


def _groq_id_default() -> str:
    mid = os.environ.get("GROQ_MODEL", "").strip() or os.environ.get("LLM_MODEL", "").strip()
    if not mid or mid.lower().startswith("gpt"):
        return "llama-3.1-8b-instant"
    return mid


def get_agno_model() -> Any:
    """Agno model for member agents. Priority: OpenAI key → Groq key → OpenAIChat (no key)."""
    if os.environ.get("OPENAI_API_KEY", "").strip():
        from agno.models.openai import OpenAIChat

        return OpenAIChat(id=_openai_id_default())
    if os.environ.get("GROQ_API_KEY", "").strip():
        from agno.models.groq import Groq

        return Groq(id=_groq_id_default())
    from agno.models.openai import OpenAIChat

    return OpenAIChat(id=_openai_id_default(), api_key="")


def get_agno_model_strong() -> Any:
    """Stronger model for ``Team`` coordination. Falls back to ``get_agno_model()`` with no keys."""
    if os.environ.get("OPENAI_API_KEY", "").strip():
        from agno.models.openai import OpenAIChat

        return OpenAIChat(id="gpt-4o")
    if os.environ.get("GROQ_API_KEY", "").strip():
        from agno.models.groq import Groq

        return Groq(id="llama-3.3-70b-versatile")
    return get_agno_model()


def make_agno_model() -> Any:
    """Backward-compatible alias for code paths that still import ``make_agno_model``."""
    return get_agno_model()


__all__ = ["get_agno_model", "get_agno_model_strong", "make_agno_model"]
