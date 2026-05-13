"""Financial news aggregator: parallel DDG/yfinance fetch, keyword sentiment, LLM digest."""

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

    # ---------- main pipeline ----------
    async def run(
        self,
        tickers: list[str],
        topics: list[str],
        user: dict,
    ) -> dict:
        clean_tickers = self._normalise_tickers(tickers)
        clean_topics = [t.strip() for t in (topics or []) if str(t or "").strip()]

        # Fetch company names in parallel so company_news searches are richer.
        names_map = await self._fetch_names(clean_tickers)

        articles = await self._fetch_all_news_parallel(
            clean_tickers, clean_topics, names_map
        )

        scored = []
        for art in articles:
            sentiment = self._score_sentiment(art.get("title", ""))
            scored.append({**art, "sentiment": sentiment})

        deduped = self._deduplicate(scored)

        summary = await self._generate_summary(deduped, clean_tickers)

        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for a in deduped:
            sentiment_counts[a.get("sentiment", "neutral")] = (
                sentiment_counts.get(a.get("sentiment", "neutral"), 0) + 1
            )

        return {
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
) -> dict:
    return await FinancialNewsAgent(llm).run(tickers, topics, user)
