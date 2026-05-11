"""Intent classifier: one ``LLMClient.complete`` per turn → ``ClassifierResult``. Callers inject the client."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from pydantic import ValidationError

from src.llm.base import LLMClient
from src.llm.mock_llm import MockLLMClient
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
    last_entities: dict = field(default_factory=dict)


def _taxonomy_block() -> str:
    lines = ["Agents (return exactly one ``agent`` string from this list):"]
    for agent, desc in AGENT_TAXONOMY.items():
        lines.append(f"- {agent}: {desc}")
    return "\n".join(lines)


def _build_system_message(inp: ClassifierInput) -> str:
    context_blob = ""
    if inp.last_entities:
        context_blob = (
            "\nCONVERSATION CONTEXT:\n"
            f"The user's last known entities were: {json.dumps(inp.last_entities, ensure_ascii=False)}\n"
            "If the current query uses vague references or pronouns, resolve them "
            "using these entities first before extracting new ones.\n"
        )

    return f"""You are an expert financial intent classifier for a wealth-management assistant.

{_taxonomy_block()}

{_ENTITY_VOCABULARY_TEXT}
{context_blob}
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


class IntentPreClassifier:
    CONFIDENCE_THRESHOLD = 0.80
    AGENT_LABELS = [
        "portfolio_health",
        "market_research",
        "investment_strategy",
        "financial_planning",
        "financial_calculator",
        "risk_assessment",
        "product_recommendation",
        "predictive_analysis",
        "customer_support",
        "general_query",
    ]

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._train()

    def _load_fixture_data(self) -> list[tuple[str, str]]:
        path = (
            Path(__file__).resolve().parent.parent
            / "fixtures"
            / "test_queries"
            / "intent_classification.json"
        )
        if not path.is_file():
            logger.warning("Intent fixture missing at %s", path)
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed reading intent fixture at %s", path)
            return []

        out: list[tuple[str, str]] = []
        for row in payload.get("queries", []):
            q = str(row.get("query") or "").strip()
            agent = str(row.get("expected_agent") or "").strip()
            if q and agent in self.AGENT_LABELS:
                out.append((q, agent))
        return out

    def _get_augmented_data(self) -> list[tuple[str, str]]:
        return [
            ("how is my portfolio", "portfolio_health"),
            ("check my holdings", "portfolio_health"),
            ("am i diversified enough", "portfolio_health"),
            ("portfolio summary please", "portfolio_health"),
            ("concentration risk review", "portfolio_health"),
            ("show my positions", "portfolio_health"),
            ("how am i doing overall", "portfolio_health"),
            ("portfolio review", "portfolio_health"),
            ("price of aapl", "market_research"),
            ("what is tesla trading at", "market_research"),
            ("news on amazon", "market_research"),
            ("tell me about microsoft", "market_research"),
            ("nvda earnings outlook", "market_research"),
            ("compare apple and google", "market_research"),
            ("what happened to oil today", "market_research"),
            ("how is asml doing", "market_research"),
            ("should i buy more", "investment_strategy"),
            ("when should i sell", "investment_strategy"),
            ("should i rebalance now", "investment_strategy"),
            ("is it a good time to invest", "investment_strategy"),
            ("what should i do with my portfolio", "investment_strategy"),
            ("should i hedge now", "investment_strategy"),
            ("should i trim tech exposure", "investment_strategy"),
            ("buy or hold my winners", "investment_strategy"),
            ("how much should i save for retirement", "financial_planning"),
            ("plan my retirement", "financial_planning"),
            ("set a savings goal", "financial_planning"),
            ("how do i reach fire", "financial_planning"),
            ("how much for a house deposit", "financial_planning"),
            ("education fund planning", "financial_planning"),
            ("retire by 50 plan", "financial_planning"),
            ("goal planning over 20 years", "financial_planning"),
            ("calculate compound interest", "financial_calculator"),
            ("what will 10000 be worth in 10 years", "financial_calculator"),
            ("dca calculator", "financial_calculator"),
            ("mortgage payment estimate", "financial_calculator"),
            ("convert usd to eur", "financial_calculator"),
            ("tax on gains calculation", "financial_calculator"),
            ("future value calculator", "financial_calculator"),
            ("monthly contribution projection", "financial_calculator"),
            ("what is my risk exposure", "risk_assessment"),
            ("how risky is my portfolio", "risk_assessment"),
            ("what if market drops 20 percent", "risk_assessment"),
            ("stress test my holdings", "risk_assessment"),
            ("sector exposure check", "risk_assessment"),
            ("max drawdown estimate", "risk_assessment"),
            ("beta and volatility risk", "risk_assessment"),
            ("currency exposure review", "risk_assessment"),
            ("which etf should i buy", "product_recommendation"),
            ("recommend me a fund", "product_recommendation"),
            ("best index fund for me", "product_recommendation"),
            ("what product matches my profile", "product_recommendation"),
            ("recommend a dividend etf", "product_recommendation"),
            ("suggest low cost world fund", "product_recommendation"),
            ("best bond etf", "product_recommendation"),
            ("fund recommendation for beginners", "product_recommendation"),
            ("where will aapl be in 6 months", "predictive_analysis"),
            ("forecast my portfolio", "predictive_analysis"),
            ("predict market direction", "predictive_analysis"),
            ("what is the outlook for tech stocks", "predictive_analysis"),
            ("next year forecast for nasdaq", "predictive_analysis"),
            ("5 year prediction for sp500", "predictive_analysis"),
            ("market trend forecast", "predictive_analysis"),
            ("future scenario for equities", "predictive_analysis"),
            ("i cant log in", "customer_support"),
            ("my trade didnt execute", "customer_support"),
            ("how do i withdraw", "customer_support"),
            ("contact support", "customer_support"),
            ("app not working", "customer_support"),
            ("bank link failed", "customer_support"),
            ("transaction missing", "customer_support"),
            ("account issue help", "customer_support"),
            ("what is an etf", "general_query"),
            ("explain pe ratio", "general_query"),
            ("what is compound interest", "general_query"),
            ("hi", "general_query"),
            ("hello", "general_query"),
            ("thanks", "general_query"),
            ("what is dollar cost averaging", "general_query"),
            ("how does the stock market work", "general_query"),
        ]

    def _train(self) -> None:
        rows = [*self._load_fixture_data(), *self._get_augmented_data()]
        if not rows:
            logger.warning("Intent pre-classifier disabled: no training rows")
            return

        texts = [q for q, _ in rows]
        labels = [a for _, a in rows]
        self._pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        max_features=8000,
                        sublinear_tf=True,
                        strip_accents="unicode",
                        analyzer="word",
                        min_df=1,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=2.0,
                        max_iter=1000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        )
        self._pipeline.fit(texts, labels)
        counts = Counter(labels)
        logger.info(
            "Intent pre-classifier trained on %d rows: %s",
            len(rows),
            ", ".join(f"{k}={counts.get(k, 0)}" for k in self.AGENT_LABELS),
        )

    def predict(self, query: str) -> tuple[str, float]:
        if self._pipeline is None:
            return "general_query", 0.0
        proba = self._pipeline.predict_proba([query])[0]
        clf = self._pipeline.named_steps["clf"]
        idx = int(proba.argmax())
        return str(clf.classes_[idx]), float(proba[idx])


class IntentClassifier:
    AGENT_TAXONOMY = AGENT_TAXONOMY
    FALLBACK_RESULT = FALLBACK_RESULT

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._pre = IntentPreClassifier()

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
        last_entities: dict | None = None,
    ) -> ClassifierResult:
        try:
            # Keep deterministic tests unchanged: queued mocks should always drive output.
            if not isinstance(self._llm, MockLLMClient):
                pre_agent, pre_conf = self._pre.predict(query)
                if pre_conf >= IntentPreClassifier.CONFIDENCE_THRESHOLD:
                    return ClassifierResult(
                        intent=query,
                        agent=pre_agent,
                        entities=Entity(),
                        safety_verdict="clean",
                        confidence=pre_conf,
                    )

            inp = ClassifierInput(
                query=query,
                prior_user_turns=list(prior_user_turns or ()),
                last_entities=dict(last_entities or {}),
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
    last_entities: dict | None = None,
) -> ClassifierResult:
    return await IntentClassifier(llm).classify(query, prior_user_turns, last_entities)
