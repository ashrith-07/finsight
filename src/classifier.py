"""Intent classifier: one ``LLMClient.complete`` per turn → ``ClassifierResult``. Callers inject the client."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from src.llm.base import LLMClient
from src.models import ClassifierResult, Entity

logger = logging.getLogger(__name__)

# Mirrors fixtures/test_queries/intent_classification.json → agent_taxonomy.
AGENT_TAXONOMY: dict[str, str] = {
    "portfolio_health": (
        "structured assessment of the user's portfolio "
        "(concentration, performance, benchmarking, observations)"
    ),
    "market_research": (
        "factual/recent info about an instrument, sector, or market event"
    ),
    "investment_strategy": (
        "advice/strategy questions: should I buy/sell/rebalance, allocation guidance"
    ),
    "financial_planning": (
        "long-term planning: retirement, goals, savings rate"
    ),
    "financial_calculator": (
        "deterministic numerical computation: DCA returns, mortgage, tax, "
        "future value, FX conversion"
    ),
    "risk_assessment": (
        "risk metrics, exposure analysis, what-if scenarios"
    ),
    "product_recommendation": (
        "recommend specific products/funds matching user profile"
    ),
    "predictive_analysis": (
        "forward-looking analysis: forecasts, trend extrapolation"
    ),
    "customer_support": (
        "platform issues, account questions, how-to-use-app"
    ),
    "general_query": (
        "educational, conversational, definitions, greetings"
    ),
}

_ENTITY_VOCABULARY_TEXT = """
Entity object fields (omit keys not mentioned in the user message; use JSON null only when a scalar is explicitly unknown — prefer omission):
- tickers: array of strings, uppercase, exchange-suffixed where relevant (AAPL, ASML.AS, 7203.T)
- amount: number, in the unit of `currency`
- currency: ISO 4217 string (USD, EUR, GBP, JPY)
- rate: decimal (0.08 for 8%)
- period_years: integer
- frequency: one of: daily, weekly, monthly, yearly
- horizon: string token (6_months, 1_year, 5_years)
- time_period: string token (today, this_week, this_month, this_year)
- topics: array of strings
- sectors: array of strings
- index: string (S&P 500, FTSE 100, NIKKEI 225, MSCI World)
- action: one of: buy, sell, hold, hedge, rebalance
- goal: one of: retirement, education, house, FIRE, emergency_fund
""".strip()


@dataclass
class ClassifierInput:
    query: str
    prior_user_turns: list[str] = field(default_factory=list)
    session_context: dict = field(default_factory=dict)


def _taxonomy_block() -> str:
    lines = ["Agents (return exactly one ``agent`` string from this list):"]
    for agent, desc in AGENT_TAXONOMY.items():
        lines.append(f"- {agent}: {desc}")
    return "\n".join(lines)


def _build_system_message(inp: ClassifierInput) -> str:
    session_blob = ""
    if inp.session_context:
        session_blob = (
            "\nLast known structured entities from this session "
            "(JSON snapshot; may be incomplete or stale):\n"
            f"{json.dumps(inp.session_context, ensure_ascii=False)}\n"
        )

    return f"""You are an expert financial intent classifier for a wealth-management assistant.

{_taxonomy_block()}

{_ENTITY_VOCABULARY_TEXT}
{session_blob}
Respond with ONLY valid JSON — no prose, no markdown fences, no code blocks. Schema:
{{
  "intent": "string describing the user's intent",
  "agent": "one of the 10 valid agent strings listed above",
  "entities": {{ ...only Entity fields that are clearly present... }},
  "safety_verdict": "clean or a short category label for logging",
  "confidence": 0.0
}}

Rules for ``confidence``: 0.0–1.0 reflecting routing certainty.

Critical — follow-up resolution:
- If the current query uses pronouns or vague references ("it", "them", "that stock", "how much do I own"), resolve them using the prior conversation turns in this request.
- Entity carryover: if the query is vague about entities, inherit from the most relevant prior user turn; if the query introduces new explicit entities, prefer those; if the user clearly switches topic, do not carry prior entities.
"""


def _strip_json_fence(text: str) -> str:
    s = text.strip()
    if not s.startswith("```"):
        return s
    s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


_ALLOWED_ENTITY_KEYS = frozenset(Entity.model_fields)


def _coerce_entities(raw: object) -> Entity:
    if not isinstance(raw, dict):
        return Entity()
    filtered = {k: v for k, v in raw.items() if k in _ALLOWED_ENTITY_KEYS}
    try:
        return Entity.model_validate(filtered)
    except ValidationError:
        return Entity()


def _clamp_confidence(raw: object) -> float:
    try:
        c = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, c))


FALLBACK_RESULT = ClassifierResult(
    intent="unknown",
    agent="general_query",
    entities=Entity(),
    safety_verdict="clean",
    confidence=0.0,
)


class IntentClassifier:
    AGENT_TAXONOMY = AGENT_TAXONOMY
    FALLBACK_RESULT = FALLBACK_RESULT

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def _build_messages(self, inp: ClassifierInput) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _build_system_message(inp)},
        ]
        for turn in inp.prior_user_turns:
            messages.append({"role": "user", "content": turn})
            messages.append({"role": "assistant", "content": "[acknowledged]"})
        messages.append({"role": "user", "content": inp.query})
        return messages

    async def classify(
        self,
        query: str,
        prior_user_turns: list[str] | None = None,
        session_context: dict | None = None,
    ) -> ClassifierResult:
        try:
            inp = ClassifierInput(
                query=query,
                prior_user_turns=list(prior_user_turns or ()),
                session_context=dict(session_context or {}),
            )
            messages = self._build_messages(inp)

            raw = await self._llm.complete(
                messages,
                response_model=None,
                temperature=0.0,
                max_tokens=1200,
            )
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError("Classifier LLM returned empty or non-string content")

            payload = json.loads(_strip_json_fence(raw))
            if not isinstance(payload, dict):
                raise TypeError("Classifier JSON root must be an object")

            agent = str(payload.get("agent") or "").strip()
            if agent not in self.AGENT_TAXONOMY:
                agent = "general_query"

            intent = str(payload.get("intent") or "unknown").strip() or "unknown"
            entities = _coerce_entities(payload.get("entities"))
            safety_verdict = str(payload.get("safety_verdict") or "clean").strip() or "clean"
            confidence = _clamp_confidence(payload.get("confidence"))

            return ClassifierResult(
                intent=intent,
                agent=agent,
                entities=entities,
                safety_verdict=safety_verdict,
                confidence=confidence,
            )
        except Exception:
            logger.exception("Intent classification failed; returning fallback result")
            return self.FALLBACK_RESULT


async def classify(
    query: str,
    llm: LLMClient,
    prior_user_turns: list[str] | None = None,
) -> ClassifierResult:
    return await IntentClassifier(llm).classify(query, prior_user_turns)
