"""Mock LLM for tests and no-key runs."""

from __future__ import annotations

import json
import re
import threading
from collections import deque
from collections.abc import AsyncGenerator

from pydantic import BaseModel

from src.agents.stub import StubAgent
from src.llm.base import LLMClient
from src.models import ClassifierResult, Entity

KNOWN_TICKERS = frozenset(
    {
        "AAPL",
        "MSFT",
        "NVDA",
        "GOOGL",
        "META",
        "AMZN",
        "TSLA",
        "AMD",
        "QQQ",
        "ASML",
        "HSBA",
        "VTI",
        "VXUS",
        "BND",
        "VOO",
        "JNJ",
        "PG",
        "KO",
        "VYM",
        "SCHD",
        "TLT",
        "SPY",
    }
)

_TICKER_PATTERN = re.compile(
    r"\b(?:"
    + "|".join(re.escape(t) for t in sorted(KNOWN_TICKERS, key=len, reverse=True))
    + r")\b"
)


class MockExhaustedError(RuntimeError):
    """Queue empty on ``complete``."""


class MockLLMClient(LLMClient):
    def __init__(
        self,
        responses: list[BaseModel | str | list[BaseModel]] | None = None,
        stream_chunks: list[str] | None = None,
    ) -> None:
        self._responses: deque[BaseModel | str | list[BaseModel]] = deque(responses or ())
        self._stream_chunks: deque[str] = deque(stream_chunks or ())
        self._lock = threading.Lock()

    @classmethod
    def for_classifier(cls, result: ClassifierResult) -> MockLLMClient:
        return cls(responses=[result])

    async def complete(
        self,
        messages: list[dict],
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> BaseModel | str:
        with self._lock:
            if not self._responses:
                raise MockExhaustedError("Mock LLM response queue is empty")
            item = self._responses.popleft()

        if response_model is not None:
            if isinstance(item, response_model):
                return item
            if isinstance(item, str):
                return response_model.model_validate_json(item)
            if isinstance(item, BaseModel):
                return response_model.model_validate(item.model_dump())
            msg = f"Mock queue item type {type(item)!r} is incompatible with {response_model!r}"
            raise TypeError(msg)

        if isinstance(item, str):
            return item
        if isinstance(item, BaseModel):
            return item.model_dump_json()
        if isinstance(item, list):
            if not item:
                return "[]"
            if all(isinstance(x, BaseModel) for x in item):
                return json.dumps(
                    [x.model_dump() for x in item],
                    ensure_ascii=False,
                )
            return json.dumps(item, ensure_ascii=False, default=str)
        return str(item)

    async def stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> AsyncGenerator[str, None]:
        with self._lock:
            chunks = tuple(self._stream_chunks)
        for chunk in chunks:
            yield chunk


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content") or "")
    return ""


def _system_prefix(messages: list[dict]) -> str:
    for m in messages:
        if m.get("role") == "system":
            return str(m.get("content") or "")
    return ""


def _is_classifier_prompt(messages: list[dict]) -> bool:
    return "expert financial intent classifier" in _system_prefix(messages)


def _is_observation_prompt(messages: list[dict]) -> bool:
    if "plain-language financial health analyst" in _system_prefix(messages):
        return True
    u = _last_user_text(messages)
    try:
        blob = json.loads(u)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(blob, dict) and "concentration" in blob


def _is_market_research_system(messages: list[dict]) -> bool:
    return "concise market analyst" in _system_prefix(messages)


def _is_market_comparison_prompt(messages: list[dict]) -> bool:
    # Comparison prompt mentions this verbatim; observation prompt does not.
    return "compare the supplied stocks" in _system_prefix(messages).lower()


def _market_comparison_fallback(user_msg: str) -> str:
    try:
        data = json.loads(user_msg)
    except (json.JSONDecodeError, TypeError):
        return "Comparison unavailable in demo mode."
    snaps = data.get("snapshots") if isinstance(data, dict) else None
    if not snaps or len(snaps) < 2:
        return "Comparison unavailable in demo mode."

    def _dir(p: float) -> str:
        return "up" if p >= 0 else "down"

    a, b = snaps[0], snaps[1]
    try:
        return (
            f"{a['ticker']} is {_dir(float(a['day_change_pct']))} "
            f"{abs(float(a['day_change_pct'])):.2f}% today vs "
            f"{b['ticker']} {_dir(float(b['day_change_pct']))} "
            f"{abs(float(b['day_change_pct'])):.2f}% today."
        )
    except (KeyError, TypeError, ValueError):
        return "Comparison unavailable in demo mode."


_EMPTY_OBS_LIST_JSON = '{"observations": []}'


def _extract_tickers(query: str) -> list[str]:
    upper = query.upper()
    found: list[str] = []
    seen: set[str] = set()
    for m in _TICKER_PATTERN.finditer(upper):
        t = m.group(0)
        if t not in seen:
            seen.add(t)
            found.append(t)
    return found


def _route_agent(query: str) -> str:
    q = query.lower()

    portfolio_kw = (
        "portfolio",
        "health check",
        "diversif",
        "concentration",
        "how is my portfolio",
        "holdings",
        "benchmark",
        "am i beating",
    )
    if any(k in q for k in portfolio_kw):
        return "portfolio_health"

    if "compare" in q and " and " in q:
        return "market_research"
    if "how is" in q and "doing" in q:
        return "market_research"
    market_kw = (
        "price of",
        "tell me about",
        "news on",
        "news about",
        "what happened",
        "what is happening",
        "what's happening",
    )
    if any(k in q for k in market_kw):
        return "market_research"

    strat_kw = ("should i buy", "should i sell", "rebalance", "allocation")
    if any(k in q for k in strat_kw):
        return "investment_strategy"

    plan_kw = ("retire", "retirement", "savings rate", "long term goal")
    if any(k in q for k in plan_kw):
        return "financial_planning"

    calc_kw = ("calculate", "compound interest", "mortgage", "future value", "dca")
    if any(k in q for k in calc_kw):
        return "financial_calculator"

    risk_kw = ("risk", "exposure", "what if", "scenario")
    if any(k in q for k in risk_kw):
        return "risk_assessment"

    prod_kw = ("recommend", "which fund", "which etf")
    if any(k in q for k in prod_kw):
        return "product_recommendation"

    pred_kw = ("forecast", "predict", "trend")
    if any(k in q for k in pred_kw):
        return "predictive_analysis"

    support_kw = ("how do i use", "account issue", "support", "help with app")
    if any(k in q for k in support_kw):
        return "customer_support"

    return "general_query"


def _intent_label(query: str, agent: str) -> str:
    q = query.strip()
    if len(q) > 120:
        q = q[:117] + "..."
    return q or f"Route to {agent}"


def _classifier_json(query: str) -> str:
    agent = _route_agent(query)
    tickers = _extract_tickers(query)
    payload = {
        "intent": _intent_label(query, agent),
        "agent": agent,
        "entities": Entity(tickers=tickers).model_dump(exclude_none=True),
        "safety_verdict": "clean",
        "confidence": 0.85,
    }
    return json.dumps(payload, ensure_ascii=False)


def _observation_json() -> str:
    return json.dumps(
        [
            {
                "severity": "info",
                "text": (
                    "Demo mode: connect OPENAI_API_KEY for observations tailored to your metrics."
                ),
            },
            {
                "severity": "warning",
                "text": "Verify figures against your custodian before acting.",
            },
        ],
        ensure_ascii=False,
    )


def _stream_message(agent: str, query: str) -> str:
    if agent == "portfolio_health":
        return "Analyzing your portfolio holdings and computing health metrics..."
    if agent == "market_research":
        return "Researching market data for the requested instruments..."
    if agent == "investment_strategy":
        return "Evaluating your investment strategy based on your profile..."
    if agent == "general_query":
        return "I'm here to help with your financial questions."

    tmpl = StubAgent.NOT_IMPLEMENTED_MESSAGES.get(agent)
    if tmpl:
        short = (query.strip() or "your request")[:80]
        return tmpl.format(entities="your request", intent=short)
    return "Processing your request..."


async def _yield_by_words(text: str) -> AsyncGenerator[str, None]:
    parts = text.split()
    for i, w in enumerate(parts):
        yield w + (" " if i < len(parts) - 1 else "")


class SmartMockLLMClient(LLMClient):
    async def complete(
        self,
        messages: list[dict],
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> BaseModel | str:
        if _is_classifier_prompt(messages):
            raw = _classifier_json(_last_user_text(messages))
            if response_model is not None:
                return response_model.model_validate_json(raw)
            return raw
        if _is_market_research_system(messages):
            if _is_market_comparison_prompt(messages):
                return _market_comparison_fallback(_last_user_text(messages))
            # Empty observation list lets MarketResearchAgent fall back to
            # snapshot-aware defaults without raising/logging an exception.
            if response_model is not None:
                return response_model.model_validate_json(_EMPTY_OBS_LIST_JSON)
            return _EMPTY_OBS_LIST_JSON
        if _is_observation_prompt(messages):
            return _observation_json()
        raw = _classifier_json(_last_user_text(messages))
        if response_model is not None:
            return response_model.model_validate_json(raw)
        return raw

    async def stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> AsyncGenerator[str, None]:
        q = _last_user_text(messages)
        agent = _route_agent(q)
        text = _stream_message(agent, q)
        async for chunk in _yield_by_words(text):
            yield chunk
