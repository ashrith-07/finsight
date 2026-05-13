"""Financial news aggregator: parallel DDG/yfinance fetch, keyword sentiment, LLM digest.

Sub-intent routing inside ``run`` selects ticker / market / sector / economic-calendar /
sentiment-summary flows.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from agno.agent import Agent

from src.llm.base import LLMClient
from src.mcp import web_search_mcp, yfinance_mcp

logger = logging.getLogger(__name__)

INSTRUCTIONS = (
    "You are a financial news synthesiser. Use web-search tools for headlines and "
    "yfinance tools to ground company context. Prefer specific numbers, name "
    "companies explicitly, and surface the 1–2 stories that actually move the thesis."
)

POSITIVE_KEYWORDS = frozenset(
    {
        "surge", "surges", "surged",
        "gain", "gains", "gained",
        "beat", "beats", "beating",
        "rise", "rises", "rising", "rose",
        "growth", "grow", "grew",
        "record", "records",
        "upgrade", "upgrades", "upgraded",
        "bullish", "rally", "rallies", "outperform", "outperforms",
        "raise", "raises", "raised",
        "strong", "strengthens", "strengthened",
    }
)

NEGATIVE_KEYWORDS = frozenset(
    {
        "drop", "drops", "dropped",
        "fall", "falls", "fell", "falling",
        "miss", "misses", "missed",
        "crash", "crashed",
        "decline", "declines", "declined",
        "downgrade", "downgrades", "downgraded",
        "bearish", "tumble", "tumbles", "tumbled",
        "risk", "risks",
        "concern", "concerns",
        "weak", "weakens", "weakened",
        "loss", "losses",
        "lawsuit", "probe", "investigation",
        "warns", "warning",
    }
)


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", str(text or "").lower())


_NEWS_SUB_INTENT_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("economic_events", re.compile(
        r"\b(fed\b|fomc|cpi|inflation\s+data|jobs?\s+report|payrolls?|"
        r"economic\s+(calendar|events)|earnings\s+(calendar|this\s+week|season)|rate\s+decision)\b", re.I)),
    ("sentiment_summary", re.compile(
        r"\b(sentiment|bullish|bearish|mood|how\s+is\s+the\s+market\s+feeling)\b", re.I)),
    ("sector_news", re.compile(
        r"\b(sector|industry|tech\s+(stocks|sector)|healthcare|energy|finance|"
        r"financials|consumer\s+(staples|discretionary)|industrials)\b", re.I)),
    ("market_news", re.compile(
        r"\b(market(s)?\s+(today|news|update)|broad\s+market|overall\s+market)\b", re.I)),
]

_DEFAULT_SECTORS = ["Technology", "Healthcare", "Financials", "Energy", "Consumer"]


def _impact_for(label: str) -> str:
    return {
        "fed_meeting": "high — affects rates, equities, and bonds",
        "earnings": "medium — name-specific moves",
        "macro_data": "high — affects rate-cut expectations",
        "jobs": "high — labour-market read",
    }.get(label, "medium")


def _detect_sub_intent(intent: str, query: str, ticker_count: int) -> str:
    text = f"{intent or ''} {query or ''}"
    for label, rx in _NEWS_SUB_INTENT_RULES:
        if rx.search(text):
            return label
    return "ticker_news" if ticker_count > 0 else "market_news"


class FinancialNewsAgent:
    """Pulls company + market news, scores headlines, returns a deduplicated, summarised digest."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._search = web_search_mcp
        self._yf = yfinance_mcp
        self._agno: Agent | None = None

    # ---------- Agno surface ----------
    def as_agno_agent(self) -> Agent:
        """Lazily build an Agno agent so this capability can also be invoked tool-style."""
        if self._agno is None:
            try:
                from src.llm.agno_model import make_agno_model

                self._agno = Agent(
                    name="financial_news",
                    model=make_agno_model(),
                    tools=[self._search, self._yf],
                    instructions=INSTRUCTIONS,
                )
            except Exception as e:
                logger.warning("Agno agent construction failed for financial_news: %s", e)
                raise
        return self._agno

    @staticmethod
    def _detect_sub_intent(intent: str, query: str, ticker_count: int) -> str:
        return _detect_sub_intent(intent, query, ticker_count)

    # ---------- main pipeline ----------
    async def run(
        self,
        tickers: list[str],
        topics: list[str],
        user: dict,
        intent: str = "",
        query: str = "",
    ) -> dict:
        clean_tickers = self._normalise_tickers(tickers)
        clean_topics = [t.strip() for t in (topics or []) if str(t or "").strip()]
        sub = self._detect_sub_intent(intent, query, len(clean_tickers))

        if sub == "sector_news":
            sectors = clean_topics or _DEFAULT_SECTORS
            payload = await self._sector_news(sectors)
            payload["user_context"] = {
                "name": user.get("name"), "risk_profile": user.get("risk_profile"),
            }
            return payload

        if sub == "economic_events":
            payload = await self._economic_calendar()
            payload["user_context"] = {
                "name": user.get("name"), "risk_profile": user.get("risk_profile"),
            }
            return payload

        if sub == "sentiment_summary":
            payload = await self._sentiment_summary(clean_tickers, clean_topics, user)
            return payload

        # market_news / ticker_news both use the full pipeline.
        names_map = await self._fetch_names(clean_tickers)
        articles = await self._fetch_all_news_parallel(
            clean_tickers, clean_topics, names_map
        )
        scored = [{**a, "sentiment": self._score_sentiment(a.get("title", ""))} for a in articles]
        deduped = self._deduplicate(scored)
        summary = await self._generate_summary(deduped, clean_tickers)

        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for a in deduped:
            key = a.get("sentiment", "neutral")
            sentiment_counts[key] = sentiment_counts.get(key, 0) + 1

        return {
            "sub_intent": sub,
            "tickers": clean_tickers,
            "topics": clean_topics,
            "articles": deduped[:20],
            "sentiment_counts": sentiment_counts,
            "summary": summary,
            "total_results": len(deduped),
            "user_context": {
                "name": user.get("name"),
                "risk_profile": user.get("risk_profile"),
            },
        }

    # ---------- sub-intents ----------
    async def _sector_news(self, sectors: list[str]) -> dict:
        """Top 3 articles per sector with sentiment, all sectors in parallel."""
        async def fetch_one(sector: str) -> tuple[str, list[dict]]:
            try:
                items = await asyncio.to_thread(
                    self._search.search_financial_news, f"{sector} sector stocks", 5
                )
            except Exception as e:
                logger.warning("sector_news fetch failed for %s: %s", sector, e)
                items = []
            scored = [
                {**it, "sentiment": self._score_sentiment(it.get("title", ""))}
                for it in (items or [])
                if isinstance(it, dict)
            ]
            return sector, scored[:3]

        results = await asyncio.gather(*(fetch_one(s) for s in sectors))
        by_sector = {sector: items for sector, items in results}
        all_articles: list[dict] = []
        for sector, items in by_sector.items():
            for it in items:
                all_articles.append({**it, "sector": sector})
        deduped = self._deduplicate(all_articles)

        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for a in deduped:
            sentiment_counts[a.get("sentiment", "neutral")] = sentiment_counts.get(
                a.get("sentiment", "neutral"), 0
            ) + 1

        return {
            "sub_intent": "sector_news",
            "tickers": [],
            "topics": sectors,
            "articles": deduped[:20],
            "sentiment_counts": sentiment_counts,
            "summary": (
                f"Pulled {len(deduped)} articles across {len(sectors)} sector(s). "
                f"Top sectors covered: {', '.join(sectors[:3])}."
            ),
            "total_results": len(deduped),
            "by_sector": {s: items for s, items in by_sector.items()},
        }

    async def _economic_calendar(self) -> dict:
        """Aggregate Fed / earnings / macro data releases via web_search MCP."""
        queries = [
            ("fed_meeting", "next FOMC fed meeting decision rate"),
            ("earnings", "earnings calendar this week major companies"),
            ("macro_data", "upcoming CPI inflation data release"),
            ("jobs", "jobs report nonfarm payrolls release date"),
        ]

        async def one(label: str, q: str) -> tuple[str, list[dict]]:
            try:
                items = await asyncio.to_thread(self._search.search_financial_news, q, 4)
            except Exception as e:
                logger.warning("economic_calendar fetch failed for %s: %s", label, e)
                items = []
            return label, [it for it in (items or []) if isinstance(it, dict)][:3]

        results = await asyncio.gather(*(one(label, q) for label, q in queries))
        events = {label: items for label, items in results}
        flat: list[dict] = []
        for label, items in events.items():
            for it in items:
                flat.append({**it, "category": label, "expected_impact": _impact_for(label)})
        deduped = self._deduplicate(flat)

        return {
            "sub_intent": "economic_events",
            "tickers": [],
            "topics": ["economic calendar"],
            "articles": deduped[:20],
            "sentiment_counts": {"positive": 0, "negative": 0, "neutral": len(deduped)},
            "summary": (
                f"{len(deduped)} upcoming-events articles across Fed, earnings, "
                f"and macro data feeds."
            ),
            "total_results": len(deduped),
            "events_by_category": events,
        }

    async def _sentiment_summary(
        self, tickers: list[str], topics: list[str], user: dict,
    ) -> dict:
        """Aggregate sentiment across all news for given tickers."""
        names_map = await self._fetch_names(tickers)
        articles = await self._fetch_all_news_parallel(tickers, topics, names_map)
        scored = [{**a, "sentiment": self._score_sentiment(a.get("title", ""))} for a in articles]
        deduped = self._deduplicate(scored)

        pos = [a for a in deduped if a.get("sentiment") == "positive"]
        neg = [a for a in deduped if a.get("sentiment") == "negative"]
        neu = [a for a in deduped if a.get("sentiment") == "neutral"]

        total = max(len(deduped), 1)
        sentiment_score = (len(pos) - len(neg)) / total
        if sentiment_score > 0.2:
            overall = "bullish"
        elif sentiment_score < -0.2:
            overall = "bearish"
        else:
            overall = "neutral"

        return {
            "sub_intent": "sentiment_summary",
            "tickers": tickers,
            "topics": topics,
            "overall_sentiment": overall,
            "sentiment_score": round(sentiment_score, 3),
            "sentiment_counts": {
                "positive": len(pos), "negative": len(neg), "neutral": len(neu),
            },
            "top_positive_headline": pos[0]["title"] if pos else None,
            "top_negative_headline": neg[0]["title"] if neg else None,
            "articles": deduped[:10],
            "summary": (
                f"Aggregate sentiment is {overall} (score {sentiment_score:+.2f}) across "
                f"{len(deduped)} headlines."
            ),
            "total_results": len(deduped),
            "user_context": {
                "name": user.get("name"), "risk_profile": user.get("risk_profile"),
            },
        }

    # ---------- fetch helpers ----------
    @staticmethod
    def _normalise_tickers(tickers: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for t in tickers or []:
            s = str(t or "").strip().upper()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    async def _fetch_names(self, tickers: list[str]) -> dict[str, str]:
        async def one(t: str) -> tuple[str, str]:
            try:
                payload = await asyncio.to_thread(self._yf.get_company_fundamentals, t)
                if isinstance(payload, dict):
                    return t, str(payload.get("name") or t)
            except Exception as e:
                logger.warning("news_agent name fetch failed for %s: %s", t, e)
            return t, t

        if not tickers:
            return {}
        pairs = await asyncio.gather(*(one(t) for t in tickers))
        return dict(pairs)

    async def _fetch_all_news_parallel(
        self,
        tickers: list[str],
        topics: list[str],
        names_map: dict[str, str],
    ) -> list[dict]:
        """Single ``asyncio.gather`` covering company news, market analysis, and macro events."""
        tasks: list[asyncio.Future] = []
        labels: list[str] = []

        for t in tickers:
            tasks.append(
                asyncio.to_thread(self._search.search_company_news, names_map.get(t, t), t, 7)
            )
            labels.append(f"company:{t}")
            tasks.append(
                asyncio.to_thread(
                    self._search.search_financial_news,
                    f"{names_map.get(t, t)} {t}",
                    5,
                )
            )
            labels.append(f"news:{t}")

        for tp in topics:
            tasks.append(asyncio.to_thread(self._search.search_market_analysis, tp))
            labels.append(f"topic:{tp}")

        tasks.append(asyncio.to_thread(self._search.get_economic_events))
        labels.append("economic_events")

        if not tickers and not topics:
            tasks.append(
                asyncio.to_thread(
                    self._search.search_financial_news, "stock market today", 5
                )
            )
            labels.append("market_overview")

        results = await asyncio.gather(*tasks, return_exceptions=True)
        flat: list[dict] = []
        for label, res in zip(labels, results):
            if isinstance(res, Exception):
                logger.warning("news_agent fetch '%s' raised: %s", label, res)
                continue
            for item in res or []:
                if not isinstance(item, dict):
                    continue
                flat.append({**item, "_source": label})
        return flat

    # ---------- analysis ----------
    def _score_sentiment(self, headline: str) -> str:
        tokens = _tokenise(headline)
        if not tokens:
            return "neutral"
        pos = sum(1 for tok in tokens if tok in POSITIVE_KEYWORDS)
        neg = sum(1 for tok in tokens if tok in NEGATIVE_KEYWORDS)
        if pos > neg:
            return "positive"
        if neg > pos:
            return "negative"
        return "neutral"

    def _deduplicate(self, articles: list[dict]) -> list[dict]:
        """Drop later articles that share any 5-word window with an earlier title."""
        WINDOW = 5
        kept: list[dict] = []
        seen_grams: set[tuple[str, ...]] = set()

        for art in articles:
            title = art.get("title") or ""
            tokens = _tokenise(title)
            if len(tokens) < WINDOW:
                # Short titles can't form a 5-gram; fall back to whole-title equality.
                key = (" ".join(tokens),)
                if key in seen_grams:
                    continue
                seen_grams.add(key)
                kept.append(art)
                continue

            grams = {tuple(tokens[i : i + WINDOW]) for i in range(len(tokens) - WINDOW + 1)}
            if grams & seen_grams:
                continue
            seen_grams |= grams
            kept.append(art)
        return kept

    async def _generate_summary(
        self,
        articles: list[dict[str, Any]],
        tickers: list[str],
    ) -> str:
        if not articles:
            return "No recent news available for the requested filters."

        bullets = "\n".join(
            f"- ({a.get('sentiment', 'neutral')}) {a.get('title', '')}"
            for a in articles[:15]
        )
        focus = ", ".join(tickers) if tickers else "the broader market"
        system = (
            "You are a market news editor. Summarise these headlines in 3 sentences "
            f"maximum, focusing on what matters most for an investor in {focus}. "
            "Reference companies and numbers where they appear. Plain language. "
            "Return text only — no markdown, no JSON."
        )
        try:
            raw = await self._llm.complete(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": bullets},
                ],
                response_model=None,
                temperature=0.4,
                max_tokens=240,
            )
            text = raw.strip() if isinstance(raw, str) else ""
            return text or self._fallback_summary(articles, tickers)
        except Exception:
            logger.exception("news_agent summary LLM failed; using deterministic fallback")
            return self._fallback_summary(articles, tickers)

    @staticmethod
    def _fallback_summary(articles: list[dict], tickers: list[str]) -> str:
        head = articles[0].get("title") if articles else ""
        focus = ", ".join(tickers) if tickers else "markets"
        if not head:
            return f"No notable headlines on {focus} in the latest pull."
        return f"Top headline on {focus}: {head}. Review the full list for context before acting."


async def run(
    tickers: list[str],
    topics: list[str],
    user: dict,
    llm: LLMClient,
    intent: str = "",
    query: str = "",
) -> dict:
    return await FinancialNewsAgent(llm).run(
        tickers, topics, user, intent=intent, query=query
    )
