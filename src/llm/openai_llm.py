"""OpenAI implementation — sole module that imports the ``openai`` SDK."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator

from pydantic import BaseModel

from src.llm.base import LLMClient, LLMError


def _require_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not str(key).strip():
        msg = (
            "OPENAI_API_KEY is not set or is empty. Set it in the environment "
            "to use OpenAILLMClient, or use get_llm_client() / MockLLMClient for tests."
        )
        raise ValueError(msg)
    return str(key).strip()


class OpenAILLMClient(LLMClient):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        from openai import AsyncOpenAI, OpenAIError

        self._OpenAIError = OpenAIError
        self._model = model
        self._client = AsyncOpenAI(api_key=_require_api_key())

    async def complete(
        self,
        messages: list[dict],
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> BaseModel | str:
        try:
            if response_model is not None:
                completion = await self._client.chat.completions.parse(
                    model=self._model,
                    messages=messages,
                    response_format=response_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                parsed = completion.choices[0].message.parsed
                if parsed is None:
                    raise LLMError("Structured completion returned no parsed content")
                return parsed

            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = completion.choices[0].message.content
            if content is None:
                raise LLMError("Completion returned empty content")
            return content
        except LLMError:
            raise
        except self._OpenAIError as e:
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
                delta = choice.delta.content
                if delta:
                    yield delta
        except self._OpenAIError as e:
            raise LLMError(str(e)) from e
