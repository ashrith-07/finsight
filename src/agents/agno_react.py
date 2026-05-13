"""Helpers for Agno ``arun`` outputs (structured + loose JSON)."""

from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def run_output_content(resp: Any) -> Any:
    return getattr(resp, "content", resp)


def coerce_pydantic(resp: Any, model_cls: type[T]) -> T | None:
    raw = run_output_content(resp)
    if isinstance(raw, model_cls):
        return raw
    if isinstance(raw, dict):
        try:
            return model_cls.model_validate(raw)
        except Exception:
            logger.debug("coerce_pydantic dict→%s failed", model_cls.__name__, exc_info=True)
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return model_cls.model_validate(data)
        except Exception:
            logger.debug("coerce_pydantic str→%s failed", model_cls.__name__, exc_info=True)
    return None


def coerce_json_dict(resp: Any) -> dict[str, Any] | None:
    raw = run_output_content(resp)
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return None
