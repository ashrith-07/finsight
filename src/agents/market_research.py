"""Market research agent: live yfinance snapshots + one LLM pass for plain-English observations."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import yfinance as yf
from pydantic import BaseModel

from src.llm.base import LLMClient
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

    async def run(self, tickers: list[str], intent: str) -> MarketResearchResult:
        clean = _normalize_tickers(tickers)
        if not clean:
            return self._no_ticker_response(intent)

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
                tickers=clean,
                snapshots=[],
                company_info=infos,
                observations=[
                    Observation(
                        severity="warning",
                        text=(
                            "No live quotes were available for the requested ticker(s). "
                            "Please verify the symbol or try again shortly."
                        ),
                    )
                ],
                comparison_note=None,
                disclaimer=DISCLAIMER,
            )

        observations = await self._generate_observations(snapshots, infos, intent, self._llm)

        comparison_note: str | None = None
        if len(snapshots) >= 2:
            comparison_note = await self._generate_comparison(snapshots, self._llm)

        return MarketResearchResult(
            tickers=clean,
            snapshots=snapshots,
            company_info=infos,
            observations=observations,
            comparison_note=comparison_note,
            disclaimer=DISCLAIMER,
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
) -> MarketResearchResult:
    return await MarketResearchAgent(llm).run(tickers, intent)
