"""Market research agent: live yfinance snapshots + one LLM pass for plain-English observations.

Sub-intent routing inside ``run`` selects price-only / fundamentals / technical /
options / comparison flows or the default full research.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import yfinance as yf
from pydantic import BaseModel

from src.llm.base import LLMClient
from src.mcp import yfinance_mcp
from src.models import (
    CompanyInfo,
    MarketResearchResult,
    Observation,
    PriceSnapshot,
)

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This data is sourced from public market feeds and is for "
    "informational purposes only. It does not constitute investment "
    "advice or a recommendation to buy or sell any security. "
    "Please consult a qualified financial adviser."
)


class _ObsList(BaseModel):
    observations: list[Observation]


_SUB_INTENT_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("options_activity", re.compile(
        r"\b(options?|put[\s-]?call|implied\s+vol|iv|open\s+interest|expiry)\b", re.I)),
    ("technical_levels", re.compile(
        r"\b(52[\s-]?week|technical|moving\s+average|ma\b|golden\s+cross|"
        r"breakout|support|resistance|near\s+(high|low))\b", re.I)),
    ("fundamentals", re.compile(
        r"\b(fundament|p/?e\b|price[\s-]?to[\s-]?earn|peg|price[\s-]?to[\s-]?book|"
        r"revenue|earnings\s+(growth|per\s+share)|profit\s+margin|"
        r"debt[\s-]?to[\s-]?equity|return\s+on\s+(equity|assets))\b", re.I)),
    ("comparison", re.compile(
        r"\b(compare|comparison|vs\.?|versus|side[\s-]?by[\s-]?side)\b", re.I)),
    ("price_check", re.compile(
        r"\b(price|trading\s+at|how\s+much\s+(is|are)|quote|spot|current\s+price)\b", re.I)),
]


def _detect_sub_intent(intent: str, query: str, ticker_count: int) -> str:
    """Default to comparison whenever 2+ tickers are explicitly given."""
    text = f"{intent or ''} {query or ''}"
    for label, rx in _SUB_INTENT_RULES:
        if rx.search(text):
            return label
    if ticker_count >= 2:
        return "comparison"
    return "full_research"


def _normalize_tickers(tickers: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in tickers or []:
        s = str(t or "").strip().upper()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN check without numpy
        return None
    return f


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


class MarketResearchAgent:
    DISCLAIMER = DISCLAIMER

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._yf = yfinance_mcp

    @staticmethod
    def _detect_sub_intent(intent: str, query: str, ticker_count: int) -> str:
        return _detect_sub_intent(intent, query, ticker_count)

    async def run(
        self,
        tickers: list[str],
        intent: str,
        query: str = "",
    ) -> MarketResearchResult:
        clean = _normalize_tickers(tickers)
        if not clean:
            return self._no_ticker_response(intent)

        sub = self._detect_sub_intent(intent, query, len(clean))
        if sub == "price_check":
            return await self._price_check(clean)
        if sub == "fundamentals":
            return await self._fundamentals_analysis(clean)
        if sub == "technical_levels":
            return await self._technical_levels(clean)
        if sub == "options_activity":
            return await self._options_activity(clean)
        # full_research and comparison both use the rich pipeline; comparison is
        # already auto-triggered by ``_generate_comparison`` when ``len >= 2``.
        return await self._full_research(clean, intent, sub)

    # -------------------- default: full research --------------------
    async def _full_research(
        self, clean: list[str], intent: str, sub: str
    ) -> MarketResearchResult:
        snap_tasks = [self._fetch_snapshot(t) for t in clean]
        info_tasks = [self._fetch_company_info(t) for t in clean]
        snapshots_raw, infos_raw = await asyncio.gather(
            asyncio.gather(*snap_tasks),
            asyncio.gather(*info_tasks),
        )
        snapshots = [s for s in snapshots_raw if s is not None]
        infos = [i for i in infos_raw if i is not None]

        if not snapshots:
            return MarketResearchResult(
                tickers=clean, snapshots=[], company_info=infos,
                observations=[Observation(
                    severity="warning",
                    text="No live quotes were available for the requested ticker(s). "
                         "Please verify the symbol or try again shortly.",
                )],
                comparison_note=None, disclaimer=DISCLAIMER,
                sub_intent=sub, extras=None,
            )

        observations = await self._generate_observations(snapshots, infos, intent, self._llm)
        comparison_note = (
            await self._generate_comparison(snapshots, self._llm) if len(snapshots) >= 2 else None
        )

        return MarketResearchResult(
            tickers=clean, snapshots=snapshots, company_info=infos,
            observations=observations, comparison_note=comparison_note,
            disclaimer=DISCLAIMER, sub_intent=sub, extras=None,
        )

    # -------------------- sub-intents --------------------
    async def _price_check(self, tickers: list[str]) -> MarketResearchResult:
        """Just current price + day change. Skips company info + LLM."""
        snaps_raw = await asyncio.gather(*(self._fetch_snapshot(t) for t in tickers))
        snaps = [s for s in snaps_raw if s is not None]
        if not snaps:
            return self._fetch_failed_response(tickers, "price_check")

        observations = [
            Observation(
                severity="info",
                text=(
                    f"{s.ticker} {s.currency} {s.current_price:.2f} "
                    f"({_direction(s.day_change_pct)} {abs(s.day_change_pct):.2f}% on the day)."
                ),
            )
            for s in snaps[:3]
        ]
        return MarketResearchResult(
            tickers=tickers, snapshots=snaps, company_info=[],
            observations=observations, comparison_note=None,
            disclaimer=DISCLAIMER, sub_intent="price_check", extras=None,
        )

    async def _fundamentals_analysis(self, tickers: list[str]) -> MarketResearchResult:
        """Pull ratios + growth metrics from yfinance via the MCP server."""
        snaps_raw, fundamentals = await asyncio.gather(
            asyncio.gather(*(self._fetch_snapshot(t) for t in tickers)),
            asyncio.gather(*(asyncio.to_thread(self._yf.get_company_fundamentals, t) for t in tickers)),
        )
        snaps = [s for s in snaps_raw if s is not None]

        # Augment yfinance MCP output with metrics that the existing toolkit
        # doesn't expose directly (forward_pe, peg, p/b, growth, ROE).
        async def deep_metrics(ticker: str) -> dict[str, Any]:
            def _blocking() -> dict[str, Any]:
                try:
                    info = yf.Ticker(ticker).info or {}
                    return {
                        "pe_ratio": _safe_float(info.get("trailingPE")),
                        "forward_pe": _safe_float(info.get("forwardPE")),
                        "peg_ratio": _safe_float(info.get("trailingPegRatio") or info.get("pegRatio")),
                        "price_to_book": _safe_float(info.get("priceToBook")),
                        "revenue_growth": _safe_float(info.get("revenueGrowth")),
                        "earnings_growth": _safe_float(info.get("earningsGrowth")),
                        "profit_margin": _safe_float(info.get("profitMargins")),
                        "debt_to_equity": _safe_float(info.get("debtToEquity")),
                        "current_ratio": _safe_float(info.get("currentRatio")),
                        "return_on_equity": _safe_float(info.get("returnOnEquity")),
                        "return_on_assets": _safe_float(info.get("returnOnAssets")),
                    }
                except Exception:
                    logger.exception("fundamentals deep fetch failed for %s", ticker)
                    return {}
            return await asyncio.to_thread(_blocking)

        deep = await asyncio.gather(*(deep_metrics(t) for t in tickers))

        # MCP fundamentals (employee count, sector, description) merged with deep ratios.
        rows = []
        for t, mcp_payload, ratios in zip(tickers, fundamentals, deep):
            mcp = mcp_payload if isinstance(mcp_payload, dict) else {}
            rows.append({
                "ticker": t,
                "name": mcp.get("name"),
                "sector": mcp.get("sector"),
                "industry": mcp.get("industry"),
                "employee_count": mcp.get("employee_count"),
                **ratios,
            })

        observations = []
        for r in rows[:3]:
            pe = r.get("pe_ratio")
            margin = r.get("profit_margin")
            de = r.get("debt_to_equity")
            bits = []
            if pe is not None:
                bits.append(f"P/E {pe:.1f}")
            if margin is not None:
                bits.append(f"profit margin {margin*100:.1f}%")
            if de is not None:
                bits.append(f"D/E {de:.2f}")
            text = (
                f"{r['ticker']} fundamentals — " + ", ".join(bits) + "."
                if bits
                else f"{r['ticker']} fundamentals were not fully reported by the data feed."
            )
            sev = "warning" if (pe is not None and pe > 50) or (de is not None and de > 200) else "info"
            observations.append(Observation(severity=sev, text=text))

        return MarketResearchResult(
            tickers=tickers, snapshots=snaps, company_info=[],
            observations=observations or [Observation(
                severity="info",
                text="Fundamentals fetch returned no data for the requested ticker(s).",
            )],
            comparison_note=None, disclaimer=DISCLAIMER,
            sub_intent="fundamentals", extras={"fundamentals": rows},
        )

    async def _technical_levels(self, tickers: list[str]) -> MarketResearchResult:
        """52w + 50/200-day moving averages + golden-cross flag."""
        snaps_raw = await asyncio.gather(*(self._fetch_snapshot(t) for t in tickers))
        snaps = [s for s in snaps_raw if s is not None]

        async def levels(ticker: str) -> dict[str, Any]:
            def _blocking() -> dict[str, Any]:
                try:
                    info = yf.Ticker(ticker).info or {}
                    current = _safe_float(info.get("currentPrice")) or _safe_float(info.get("regularMarketPrice"))
                    ma50 = _safe_float(info.get("fiftyDayAverage"))
                    ma200 = _safe_float(info.get("twoHundredDayAverage"))
                    h52 = _safe_float(info.get("fiftyTwoWeekHigh"))
                    l52 = _safe_float(info.get("fiftyTwoWeekLow"))
                    return {
                        "ticker": ticker,
                        "current_price": current,
                        "fifty_two_week_high": h52,
                        "fifty_two_week_low": l52,
                        "distance_from_52w_high_pct": (
                            round((h52 - current) / h52 * 100.0, 2)
                            if (current is not None and h52 and h52 > 0) else None
                        ),
                        "distance_from_52w_low_pct": (
                            round((current - l52) / l52 * 100.0, 2)
                            if (current is not None and l52 and l52 > 0) else None
                        ),
                        "ma_50": ma50,
                        "ma_200": ma200,
                        "vs_ma_50_pct": (
                            round((current / ma50 - 1.0) * 100.0, 2)
                            if (current is not None and ma50 and ma50 > 0) else None
                        ),
                        "vs_ma_200_pct": (
                            round((current / ma200 - 1.0) * 100.0, 2)
                            if (current is not None and ma200 and ma200 > 0) else None
                        ),
                        "golden_cross": (
                            ma50 > ma200 if (ma50 is not None and ma200 is not None) else None
                        ),
                    }
                except Exception:
                    logger.exception("technical_levels fetch failed for %s", ticker)
                    return {"ticker": ticker}
            return await asyncio.to_thread(_blocking)

        rows = await asyncio.gather(*(levels(t) for t in tickers))
        observations = []
        for r in rows[:3]:
            cross = r.get("golden_cross")
            d52h = r.get("distance_from_52w_high_pct")
            if cross is True:
                observations.append(Observation(
                    severity="info",
                    text=f"{r['ticker']} 50-day MA above 200-day MA — golden-cross posture (typically bullish).",
                ))
            elif cross is False:
                observations.append(Observation(
                    severity="warning",
                    text=f"{r['ticker']} 50-day MA below 200-day MA — death-cross posture (typically bearish).",
                ))
            if d52h is not None and d52h < 5:
                observations.append(Observation(
                    severity="info",
                    text=f"{r['ticker']} sits within {d52h:.1f}% of its 52-week high — extended territory.",
                ))

        if not observations:
            observations.append(Observation(
                severity="info",
                text="Technical levels were partially unavailable from the data feed.",
            ))
        return MarketResearchResult(
            tickers=tickers, snapshots=snaps, company_info=[],
            observations=observations[:4], comparison_note=None,
            disclaimer=DISCLAIMER, sub_intent="technical_levels",
            extras={"technical_levels": rows},
        )

    async def _options_activity(self, tickers: list[str]) -> MarketResearchResult:
        """Wraps the yfinance MCP options tool for each ticker."""
        snaps_raw, options_raw = await asyncio.gather(
            asyncio.gather(*(self._fetch_snapshot(t) for t in tickers)),
            asyncio.gather(*(asyncio.to_thread(self._yf.get_options_data, t) for t in tickers)),
        )
        snaps = [s for s in snaps_raw if s is not None]
        rows = [
            payload if isinstance(payload, dict) else {"ticker": t, "error": "no data"}
            for t, payload in zip(tickers, options_raw)
        ]

        observations: list[Observation] = []
        for r in rows[:3]:
            if r.get("error"):
                observations.append(Observation(
                    severity="info",
                    text=f"No options data available for {r.get('ticker', 'ticker')}.",
                ))
                continue
            pcr = r.get("put_call_ratio")
            iv = r.get("implied_volatility_avg")
            if pcr is not None:
                sev = "warning" if pcr > 1.2 else ("info" if pcr >= 0.8 else "warning")
                stance = "bearish skew" if pcr > 1.2 else ("bullish skew" if pcr < 0.8 else "balanced")
                observations.append(Observation(
                    severity=sev,
                    text=(
                        f"{r.get('ticker')} options put/call ratio is {pcr:.2f} ({stance})"
                        + (f", avg IV {iv:.2f}." if iv is not None else ".")
                    ),
                ))

        if not observations:
            observations.append(Observation(
                severity="info",
                text="Options data was not available for the requested ticker(s).",
            ))

        return MarketResearchResult(
            tickers=tickers, snapshots=snaps, company_info=[],
            observations=observations[:4], comparison_note=None,
            disclaimer=DISCLAIMER, sub_intent="options_activity",
            extras={"options": rows},
        )

    def _fetch_failed_response(self, tickers: list[str], sub_intent: str) -> MarketResearchResult:
        return MarketResearchResult(
            tickers=tickers, snapshots=[], company_info=[],
            observations=[Observation(
                severity="warning",
                text="No live quotes were available for the requested ticker(s).",
            )],
            comparison_note=None, disclaimer=DISCLAIMER,
            sub_intent=sub_intent, extras=None,
        )

    async def _fetch_snapshot(self, ticker: str) -> PriceSnapshot | None:
        def _blocking() -> tuple[dict, Any]:
            t = yf.Ticker(ticker)
            return (t.info or {}, t.history(period="2d"))

        try:
            info, hist = await asyncio.to_thread(_blocking)
            # yfinance returns ~empty dicts for invalid tickers
            if not info or len(info) < 5:
                return None

            current = _safe_float(info.get("currentPrice")) or _safe_float(
                info.get("regularMarketPrice")
            )
            prev_close = _safe_float(info.get("previousClose")) or _safe_float(
                info.get("regularMarketPreviousClose")
            )
            currency = str(info.get("currency") or "USD").upper()

            # Last-resort price/prev_close from the recent 2-day history.
            if (current is None or prev_close is None) and hist is not None and not hist.empty:
                closes = hist["Close"].dropna()
                if not closes.empty:
                    if current is None:
                        current = float(closes.iloc[-1])
                    if prev_close is None and len(closes) >= 2:
                        prev_close = float(closes.iloc[-2])

            if current is None or prev_close is None or prev_close == 0:
                return None

            day_open = _safe_float(info.get("open")) or _safe_float(
                info.get("regularMarketOpen")
            ) or current
            day_high = _safe_float(info.get("dayHigh")) or _safe_float(
                info.get("regularMarketDayHigh")
            ) or current
            day_low = _safe_float(info.get("dayLow")) or _safe_float(
                info.get("regularMarketDayLow")
            ) or current
            volume = _safe_int(
                info.get("volume") if info.get("volume") is not None
                else info.get("regularMarketVolume")
            )
            market_cap = _safe_float(info.get("marketCap"))
            fifty_two_high = (
                _safe_float(info.get("fiftyTwoWeekHigh")) or current
            )
            fifty_two_low = (
                _safe_float(info.get("fiftyTwoWeekLow")) or current
            )

            day_change_pct = (current - prev_close) / prev_close * 100.0
            if fifty_two_high > 0:
                distance_from_52w_high_pct = (
                    (fifty_two_high - current) / fifty_two_high * 100.0
                )
            else:
                distance_from_52w_high_pct = 0.0

            return PriceSnapshot(
                ticker=ticker,
                current_price=round(current, 4),
                currency=currency,
                day_open=round(day_open, 4),
                day_high=round(day_high, 4),
                day_low=round(day_low, 4),
                previous_close=round(prev_close, 4),
                day_change_pct=round(day_change_pct, 2),
                volume=volume,
                market_cap=market_cap,
                fifty_two_week_high=round(fifty_two_high, 4),
                fifty_two_week_low=round(fifty_two_low, 4),
                distance_from_52w_high_pct=round(distance_from_52w_high_pct, 2),
            )
        except Exception:
            logger.exception("market_research snapshot fetch failed for %s", ticker)
            return None

    async def _fetch_company_info(self, ticker: str) -> CompanyInfo | None:
        def _blocking() -> dict:
            return yf.Ticker(ticker).info or {}

        try:
            info = await asyncio.to_thread(_blocking)
            if not info or len(info) < 5:
                return None
            name = (
                str(info.get("longName") or info.get("shortName") or ticker)
            ).strip() or ticker
            summary_raw = info.get("longBusinessSummary") or ""
            summary = str(summary_raw)[:300] if summary_raw else None
            return CompanyInfo(
                name=name,
                sector=info.get("sector") or None,
                industry=info.get("industry") or None,
                country=info.get("country") or None,
                summary=summary,
            )
        except Exception:
            logger.exception("market_research company info fetch failed for %s", ticker)
            return None

    async def _generate_observations(
        self,
        snapshots: list[PriceSnapshot],
        company_infos: list[CompanyInfo],
        intent: str,
        llm: LLMClient,
    ) -> list[Observation]:
        system = (
            "You are a concise market analyst for a wealth management app.\n"
            "Rules:\n"
            "- Maximum 3 observations\n"
            "- Each observation must reference specific numbers from the data\n"
            "- severity: info for neutral facts, warning for risk signals, "
            "critical for major red flags\n"
            "- Plain language, no jargon\n"
            f"- Focus on what is most relevant to: {intent}\n"
            '- Return JSON: {"observations": [{"severity": "info|warning|critical", "text": "..."}]}'
        )
        payload = {
            "snapshots": [s.model_dump() for s in snapshots],
            "company_info": [c.model_dump() for c in company_infos],
        }
        user_msg = json.dumps(payload, ensure_ascii=False)

        try:
            raw = await llm.complete(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                response_model=_ObsList,
                temperature=0.3,
                max_tokens=600,
            )
            if not isinstance(raw, _ObsList):
                raise TypeError("LLM returned non-_ObsList response")
            obs = raw.observations[:3]
            return obs if obs else _default_observations(snapshots)
        except Exception:
            logger.exception("market_research observation LLM failed; using defaults")
            return _default_observations(snapshots)

    async def _generate_comparison(
        self,
        snapshots: list[PriceSnapshot],
        llm: LLMClient,
    ) -> str:
        system = (
            "You are a concise market analyst. Compare the supplied stocks in "
            "2 sentences maximum. Reference specific numbers from the data. "
            "Plain language, no jargon. Return text only — no markdown, no JSON."
        )
        payload = {"snapshots": [s.model_dump() for s in snapshots]}
        user_msg = json.dumps(payload, ensure_ascii=False)

        try:
            raw = await llm.complete(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                response_model=None,
                temperature=0.3,
                max_tokens=200,
            )
            text = raw if isinstance(raw, str) else ""
            text = text.strip()
            if not text:
                raise ValueError("empty comparison")
            return text
        except Exception:
            logger.exception("market_research comparison LLM failed; using deterministic fallback")
            a, b = snapshots[0], snapshots[1]
            return (
                f"{a.ticker} is {_direction(a.day_change_pct)} {abs(a.day_change_pct):.2f}% today "
                f"vs {b.ticker} {_direction(b.day_change_pct)} {abs(b.day_change_pct):.2f}% today."
            )

    def _no_ticker_response(self, intent: str) -> MarketResearchResult:
        return MarketResearchResult(
            tickers=[],
            snapshots=[],
            company_info=[],
            observations=[
                Observation(
                    severity="info",
                    text=(
                        "Please mention a specific stock ticker or company name, "
                        "for example: 'tell me about AAPL' or 'research Microsoft'"
                    ),
                )
            ],
            comparison_note=None,
            disclaimer=DISCLAIMER,
            sub_intent="full_research",
            extras=None,
        )


def _direction(change_pct: float) -> str:
    return "up" if change_pct >= 0 else "down"


def _default_observations(snapshots: list[PriceSnapshot]) -> list[Observation]:
    if not snapshots:
        return [
            Observation(
                severity="info",
                text=(
                    "Live market data is currently unavailable; please try again in a moment."
                ),
            )
        ]
    s = snapshots[0]
    return [
        Observation(
            severity="info",
            text=(
                f"{s.ticker} is trading at {s.currency} {s.current_price:.2f} "
                f"({_direction(s.day_change_pct)} {abs(s.day_change_pct):.2f}% on the day)."
            ),
        ),
        Observation(
            severity="info" if s.distance_from_52w_high_pct < 15 else "warning",
            text=(
                f"{s.ticker} sits {s.distance_from_52w_high_pct:.1f}% below its 52-week high "
                f"of {s.currency} {s.fifty_two_week_high:.2f}."
            ),
        ),
    ]


async def run(
    tickers: list[str],
    intent: str,
    llm: LLMClient,
    query: str = "",
) -> MarketResearchResult:
    return await MarketResearchAgent(llm).run(tickers, intent, query=query)
