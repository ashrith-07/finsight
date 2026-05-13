"""Single source of truth for the Agno ``Model`` instance used by every agent.

Provider precedence: ``GROQ_API_KEY`` > ``OPENAI_API_KEY``. Imports stay lazy so
modules that only need the deterministic ``LLMClient`` path (tests, mock-only
runs) never pull in ``agno.models.openai`` / ``agno.models.groq``.
"""

from __future__ import annotations

import os
from typing import Any


def make_agno_model() -> Any:
    if os.environ.get("GROQ_API_KEY", "").strip():
        from agno.models.groq import Groq

        model_id = (
            os.environ.get("GROQ_MODEL", "").strip()
            or os.environ.get("LLM_MODEL", "").strip()
        )
        if not model_id or model_id.lower().startswith("gpt"):
            model_id = "llama-3.3-70b-versatile"
        return Groq(id=model_id)

    from agno.models.openai import OpenAIChat

    model_id = (
        os.environ.get("OPENAI_MODEL", "").strip()
        or os.environ.get("LLM_MODEL", "").strip()
        or "gpt-4o-mini"
    )
    return OpenAIChat(id=model_id)


__all__ = ["make_agno_model"]
