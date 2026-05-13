"""Groq implementation. Sole module that imports the ``groq`` SDK.

Groq exposes an OpenAI-compatible chat-completions API but does **not** ship the
structured-output ``parse`` helper. We emulate it with native JSON-mode
(``response_format={"type": "json_object"}``) plus the Pydantic model's JSON
schema injected into the system prompt, then validate on the way out.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator

from pydantic import BaseModel, ValidationError

from src.llm.base import LLMClient, LLMError

DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _require_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not str(key).strip():
        raise ValueError(
            "GROQ_API_KEY is not set or is empty. Set it in the environment to use "
            "GroqLLMClient, or use get_llm_client() / SmartMockLLMClient for tests."
        )
    return str(key).strip()


def _resolve_model(explicit: str | None) -> str:
    if explicit:
        return explicit
    env_model = (os.environ.get("GROQ_MODEL", "").strip() or os.environ.get("LLM_MODEL", "").strip())
    # ``LLM_MODEL`` defaulted to ``gpt-...`` historically — guard against it.
    if env_model and not env_model.lower().startswith("gpt"):
        return env_model
    return DEFAULT_MODEL


def _inject_json_schema(messages: list[dict], schema: dict) -> list[dict]:
    """Append schema + JSON keyword to the system message (Groq JSON mode requires the word 'json')."""
    addendum = (
        "\n\nReturn ONLY a valid JSON object matching this schema — no markdown, no prose:\n"
        f"{json.dumps(schema, separators=(',', ':'))}"
    )
    out: list[dict] = []
    sys_seen = False
    for m in messages:
        if not sys_seen and m.get("role") == "system":
            out.append({**m, "content": str(m.get("content") or "") + addendum})
            sys_seen = True
        else:
            out.append(m)
    if not sys_seen:
        out.insert(0, {"role": "system", "content": "You return JSON only." + addendum})
    return out


class GroqLLMClient(LLMClient):
    def __init__(self, model: str | None = None) -> None:
        from groq import AsyncGroq, GroqError

        self._GroqError = GroqError
        self._model = _resolve_model(model)
        self._client = AsyncGroq(api_key=_require_api_key())

    async def complete(
        self,
        messages: list[dict],
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> BaseModel | str:
        try:
            if response_model is not None:
                schema = response_model.model_json_schema()
                msgs = _inject_json_schema(messages, schema)
                completion = await self._client.chat.completions.create(
                    model=self._model,
                    messages=msgs,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                content = completion.choices[0].message.content
                if not content:
                    raise LLMError("Groq structured completion returned empty content")
                try:
                    return response_model.model_validate_json(content)
                except ValidationError as e:
                    raise LLMError(f"Groq JSON did not match {response_model.__name__}: {e}") from e

            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = completion.choices[0].message.content
            if content is None:
                raise LLMError("Groq completion returned empty content")
            return content
        except LLMError:
            raise
        except self._GroqError as e:
            raise LLMError(str(e)) from e

    async def stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> AsyncGenerator[str, None]:
        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            async for event in stream:
                choice = event.choices[0] if event.choices else None
                if choice is None:
                    continue
                delta = getattr(choice.delta, "content", None)
                if delta:
                    yield delta
        except self._GroqError as e:
            raise LLMError(str(e)) from e
