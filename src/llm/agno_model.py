"""Agno-native ``Model`` instances for ``Agent`` / ``Team`` (separate from ``LLMClient``).

Imports stay lazy so test runs that only use ``MockLLMClient`` avoid loading Agno model modules.
"""

from __future__ import annotations

import os
from typing import Any, Optional


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


def get_agno_model() -> Optional[Any]:
    """Agno model for member agents. Priority: OpenAI key → Groq key; ``None`` if neither is set."""
    if os.environ.get("OPENAI_API_KEY", "").strip():
        from agno.models.openai import OpenAIChat

        return OpenAIChat(id=_openai_id_default())
    if os.environ.get("GROQ_API_KEY", "").strip():
        from agno.models.groq import Groq

        return Groq(id=_groq_id_default())
    return None


def get_agno_model_strong() -> Optional[Any]:
    """Model for ``Team`` coordination; ``None`` when no provider key is configured.

    On Groq we default the team lead to the **same** model as member agents
    (``GROQ_MODEL`` / ``llama-3.1-8b-instant``). A separate 70B coordinator plus
    three 8B tool-callers routinely exceeds free-tier TPM and triggers 429s.
    Set ``GROQ_TEAM_MODEL`` to override the coordinator only (e.g.
    ``llama-3.3-70b-versatile``) when your quota allows.
    """
    if os.environ.get("OPENAI_API_KEY", "").strip():
        from agno.models.openai import OpenAIChat

        return OpenAIChat(id="gpt-4o")
    if os.environ.get("GROQ_API_KEY", "").strip():
        from agno.models.groq import Groq

        override = os.environ.get("GROQ_TEAM_MODEL", "").strip()
        if override and not override.lower().startswith("gpt"):
            return Groq(id=override)
        return Groq(id=_groq_id_default())
    return None


def make_agno_model() -> Optional[Any]:
    """Backward-compatible alias for code paths that still import ``make_agno_model``."""
    return get_agno_model()


def agno_allows_structured_output_with_tools() -> bool:
    """Whether Agno may combine ``output_schema`` + ``structured_outputs`` with ``tools``.

    Groq returns HTTP 400: *json mode cannot be combined with tool/function calling*
    when ``response_format`` is set alongside tools. :func:`get_agno_model` picks
    OpenAI when ``OPENAI_API_KEY`` is set, otherwise Groq — only the OpenAI path
    is safe for native structured tool runs.
    """
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


__all__ = [
    "agno_allows_structured_output_with_tools",
    "get_agno_model",
    "get_agno_model_strong",
    "make_agno_model",
]
