"""DuckDuckGo-backed MCP toolkit for free, key-less news + market search."""

from __future__ import annotations

from agno.tools import Toolkit

from src.logging_config import get_logger

try:
    from ddgs import DDGS  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    from duckduckgo_search import DDGS  # type: ignore[no-redef]

logger = get_logger("mcp.web_search")


def _normalise_news(item: dict) -> dict:
    raw = item.get("body") or item.get("excerpt") or ""
    raw_s = str(raw)
    sn = raw_s[:400] + ("…" if len(raw_s) > 400 else "")
    return {
        "title": (str(item.get("title") or ""))[:200],
        "url": item.get("url") or item.get("href") or "",
        "snippet": sn,
        "published_date": item.get("date") or item.get("published") or "",
        "source": (str(item.get("source") or ""))[:80],
    }


def _normalise_text(item: dict) -> dict:
    raw = item.get("body") or ""
    raw_s = str(raw)
    sn = raw_s[:400] + ("…" if len(raw_s) > 400 else "")
    return {
        "title": (str(item.get("title") or ""))[:200],
        "url": item.get("href") or item.get("url") or "",
        "snippet": sn,
    }


# DDG returns 202/rate-limit errors as plain exceptions; we swallow them and return [].
_MAX_NEWS_RESULTS = 12


def _timelimit_for_days(days_back: int) -> str:
    """Map days → DDG timelimit code (d/w/m/y); pick the smallest window that fits."""
    if days_back <= 1:
        return "d"
    if days_back <= 7:
        return "w"
    if days_back <= 31:
        return "m"
    return "y"


class WebSearchMCPServer(Toolkit):
    """Wraps DuckDuckGo for financial news and market analysis lookups."""

    name = "web_search_mcp"

    def __init__(self) -> None:
        super().__init__(name="web_search_mcp", auto_register=False)
        self.register(self.search_financial_news)
        self.register(self.search_company_news)
        self.register(self.search_market_analysis)
        self.register(self.get_economic_events)

    def search_financial_news(self, query: str | None = None, max_results: int = 5) -> list[dict]:
        """News search; appends 'financial news' so DDG biases towards markets coverage."""
        if not query or not isinstance(query, str) or not query.strip():
            return []
        n = max(1, min(int(max_results or 5), _MAX_NEWS_RESULTS))
        q = f"{query.strip()} financial news".strip()
        try:
            with DDGS() as ddg:
                raw = ddg.news(q, max_results=n) or []
            return [_normalise_news(r) for r in raw]
        except Exception as e:
            logger.warning("web_search_mcp.search_financial_news(%r) failed: %s", q, e)
            return []

    def search_company_news(
        self,
        company_name: str | None = None,
        ticker: str | None = None,
        days_back: int = 7,
    ) -> list[dict]:
        """Recent news for a single company, time-limited via DDG's ``timelimit`` parameter."""
        cn = (company_name or "").strip() if isinstance(company_name, str) else ""
        tk = (ticker or "").strip() if isinstance(ticker, str) else ""
        if not cn and not tk:
            return []
        q = f"{cn} {tk} stock news".strip()
        try:
            with DDGS() as ddg:
                raw = ddg.news(
                    q,
                    max_results=10,
                    timelimit=_timelimit_for_days(days_back),
                ) or []
            return [_normalise_news(r) for r in raw]
        except Exception as e:
            logger.warning("web_search_mcp.search_company_news(%r) failed: %s", q, e)
            return []

    def search_market_analysis(self, topic: str | None = None) -> list[dict]:
        """Free-text web search for analyst commentary / forecasts."""
        if not topic or not isinstance(topic, str) or not topic.strip():
            return []
        q = f"{topic.strip()} market analysis forecast 2025".strip()
        try:
            with DDGS() as ddg:
                raw = ddg.text(q, max_results=5) or []
            return [_normalise_text(r) for r in raw]
        except Exception as e:
            logger.warning("web_search_mcp.search_market_analysis(%r) failed: %s", q, e)
            return []

    def get_economic_events(self) -> list[dict]:
        """Top 5 results for upcoming macro events (Fed, earnings calendar, etc.)."""
        q = "upcoming economic events fed meeting earnings calendar"
        try:
            with DDGS() as ddg:
                raw = ddg.text(q, max_results=5) or []
            return [_normalise_text(r) for r in raw]
        except Exception as e:
            logger.warning("web_search_mcp.get_economic_events failed: %s", e)
            return []
