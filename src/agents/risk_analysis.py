"""Portfolio risk analytics: VaR, drawdown, Sharpe, stress tests, correlations.

Deterministic numpy/scipy pipeline + one LLM pass for plain-English observations.
The Agno ``Agent`` wrapper is built lazily and exposes the yfinance + web-search MCP
toolkits to upstream orchestrators that prefer tool-driven flows.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import numpy as np
from agno.agent import Agent
from pydantic import BaseModel

from src.llm.base import LLMClient
from src.mcp import web_search_mcp, yfinance_mcp
from src.models import Observation

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Risk metrics are estimates based on historical price data and assume past "
    "behaviour is informative for future risk. Actual outcomes can be worse, "
    "particularly in regime shifts. Not investment advice."
)

INSTRUCTIONS = (
    "You are a portfolio risk analyst. Use the yfinance tools to pull historical "
    "prices and the web-search tools to corroborate any unusual signal. Always "
    "anchor commentary to specific numbers (VaR, drawdown, Sharpe, correlation) "
    "and keep the audience as a non-specialist investor."
)


class _ObsList(BaseModel):
    observations: list[Observation]


def _safe_positions(user: dict) -> list[dict]:
    return [p for p in (user.get("positions") or []) if str(p.get("ticker") or "").strip()]


def _portfolio_returns(
    weights: dict[str, float],
    returns_by_ticker: dict[str, np.ndarray],
) -> np.ndarray:
    """Weighted daily portfolio returns clipped to the shortest series across tickers."""
    if not weights or not returns_by_ticker:
        return np.array([])
    series = [(t, returns_by_ticker[t]) for t in weights if t in returns_by_ticker and returns_by_ticker[t].size]
    if not series:
        return np.array([])
    n = min(arr.size for _, arr in series)
    if n < 2:
        return np.array([])
    aligned = np.vstack([arr[-n:] for _, arr in series])
    w = np.array([weights[t] for t, _ in series], dtype=float)
    if w.sum() == 0:
        return np.array([])
    w = w / w.sum()
    return aligned.T @ w


def _portfolio_value_series(
    weights: dict[str, float],
    price_history: dict[str, list[float]],
    portfolio_value: float,
) -> np.ndarray:
    """Re-base weighted prices to ``portfolio_value`` so drawdown is in dollar terms."""
    if not weights or not price_history or portfolio_value <= 0:
        return np.array([])
    aligned: list[np.ndarray] = []
    used: list[str] = []
    for t in weights:
        prices = np.array(price_history.get(t, []), dtype=float)
        prices = prices[~np.isnan(prices)]
        if prices.size >= 2:
            aligned.append(prices)
            used.append(t)
    if not aligned:
        return np.array([])
    n = min(a.size for a in aligned)
    indexed = []
    for arr in aligned:
        a = arr[-n:]
        indexed.append(a / a[0])
    matrix = np.vstack(indexed)
    w = np.array([weights[t] for t in used], dtype=float)
    w = w / w.sum() if w.sum() else w
    return (matrix.T @ w) * portfolio_value


class RiskAnalysisAgent:
    """Computes VaR, drawdown, Sharpe, stress tests and pairwise correlations for a user portfolio."""

    SCENARIOS: dict[str, float] = {
        "2008_financial_crisis": -0.50,
        "2020_covid_crash": -0.34,
        "2022_rate_hike": -0.25,
        "dot_com_bubble": -0.45,
        "mild_correction": -0.10,
    }

    RISK_FREE_RATE = 0.0525
    TRADING_DAYS = 252
    CORR_THRESHOLD = 0.7

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._yf = yfinance_mcp
        self._search = web_search_mcp
        self._agno: Agent | None = None

    # ---------- Agno surface ----------
    def as_agno_agent(self) -> Agent:
        """Lazily build an Agno agent that exposes the same MCP tools to a tool-calling LLM."""
        if self._agno is None:
            try:
                from agno.models.openai import OpenAIChat

                self._agno = Agent(
                    name="risk_analysis",
                    model=OpenAIChat(id="gpt-4o-mini"),
                    tools=[self._yf, self._search],
                    instructions=INSTRUCTIONS,
                )
            except Exception as e:
                logger.warning("Agno agent construction failed for risk_analysis: %s", e)
                raise
        return self._agno

    # ---------- main pipeline ----------
    async def run(self, user: dict) -> dict:
        positions = _safe_positions(user)
        currency = str(user.get("base_currency") or "USD").upper()

        if not positions:
            return self._empty_result(currency, reason="no positions on file")

        price_history = await self._fetch_all_history(positions)
        valid = {t: pr for t, pr in price_history.items() if pr and len(pr) >= 30}
        if not valid:
            return self._empty_result(currency, reason="historical price data unavailable")

        last_price = {t: pr[-1] for t, pr in valid.items()}
        position_values: dict[str, float] = {}
        for p in positions:
            t = str(p.get("ticker") or "").upper()
            if t not in last_price:
                continue
            qty = float(p.get("quantity") or 0.0)
            position_values[t] = qty * last_price[t]

        portfolio_value = float(sum(position_values.values()))
        if portfolio_value <= 0:
            return self._empty_result(currency, reason="portfolio value evaluates to zero")

        weights = {t: v / portfolio_value for t, v in position_values.items()}
        returns_by_ticker = {
            t: np.diff(np.array(pr, dtype=float)) / np.array(pr[:-1], dtype=float)
            for t, pr in valid.items()
            if len(pr) >= 2
        }
        port_returns = _portfolio_returns(weights, returns_by_ticker)
        port_value_series = _portfolio_value_series(weights, valid, portfolio_value)

        var_95 = self._compute_var(port_returns, portfolio_value, confidence=0.95)
        var_99 = self._compute_var(port_returns, portfolio_value, confidence=0.99)
        max_dd = self._compute_max_drawdown(port_value_series.tolist() if port_value_series.size else [])
        sharpe = self._compute_sharpe(port_returns, self.RISK_FREE_RATE)
        stress = self._run_stress_tests(portfolio_value, weights)
        correlations = self._compute_correlations(valid)

        metrics = {
            "currency": currency,
            "portfolio_value": round(portfolio_value, 2),
            "position_weights_pct": {t: round(w * 100, 2) for t, w in weights.items()},
            "var": {
                "method": "historical_simulation",
                "one_day_95": round(var_95, 2),
                "one_day_99": round(var_99, 2),
            },
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio_annualised": round(sharpe, 2),
            "stress_tests": stress,
            "significant_correlations": correlations,
            "lookback_days": int(min((len(pr) for pr in valid.values()), default=0)),
        }

        observations = await self._generate_observations(metrics)
        metrics["observations"] = [o.model_dump() for o in observations]
        metrics["disclaimer"] = DISCLAIMER
        return metrics

    # ---------- data fetch ----------
    async def _fetch_all_history(self, positions: list[dict]) -> dict[str, list[float]]:
        tickers = sorted({str(p.get("ticker") or "").upper() for p in positions if p.get("ticker")})

        async def one(t: str) -> tuple[str, list[float]]:
            try:
                payload = await asyncio.to_thread(
                    self._yf.get_historical_prices, t, "1y", "1d"
                )
                bars = payload.get("bars") if isinstance(payload, dict) else None
                closes = [
                    float(b["close"])
                    for b in (bars or [])
                    if isinstance(b, dict) and b.get("close") is not None
                ]
                return t, closes
            except Exception as e:
                logger.warning("risk_analysis history fetch failed for %s: %s", t, e)
                return t, []

        results = await asyncio.gather(*(one(t) for t in tickers))
        return {t: closes for t, closes in results}

    # ---------- math ----------
    def _compute_var(
        self,
        returns: np.ndarray,
        portfolio_value: float,
        confidence: float = 0.95,
    ) -> float:
        if returns is None or returns.size < 5 or portfolio_value <= 0:
            return 0.0
        sorted_r = np.sort(returns)
        idx = max(int((1.0 - confidence) * sorted_r.size) - 1, 0)
        worst = float(sorted_r[idx])
        return abs(worst) * float(portfolio_value)

    def _compute_max_drawdown(self, prices: list[float]) -> float:
        if not prices or len(prices) < 2:
            return 0.0
        arr = np.array(prices, dtype=float)
        cummax = np.maximum.accumulate(arr)
        # Guard the rare case where the series starts at 0.
        cummax = np.where(cummax == 0, np.nan, cummax)
        dd = (arr - cummax) / cummax
        worst = float(np.nanmin(dd))
        return worst * 100.0

    def _compute_sharpe(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0525,
    ) -> float:
        if returns is None or returns.size < 5:
            return 0.0
        std = float(np.std(returns, ddof=1))
        if std == 0:
            return 0.0
        rf_daily = risk_free_rate / self.TRADING_DAYS
        excess = returns - rf_daily
        return float(np.mean(excess) / std * np.sqrt(self.TRADING_DAYS))

    def _run_stress_tests(
        self,
        portfolio_value: float,
        positions_weights: dict[str, float],
    ) -> list[dict]:
        # ``positions_weights`` is unused at this layer because we apply a market-wide
        # shock; kept in the signature so per-position betas can plug in later.
        _ = positions_weights
        out = []
        for scenario, decline in self.SCENARIOS.items():
            loss = portfolio_value * decline
            out.append(
                {
                    "scenario": scenario,
                    "market_decline_pct": round(decline * 100, 2),
                    "portfolio_loss_usd": round(loss, 2),
                    "portfolio_loss_pct": round(decline * 100, 2),
                    "surviving_value": round(portfolio_value + loss, 2),
                }
            )
        return out

    def _compute_correlations(
        self,
        price_history: dict[str, list[float]],
    ) -> dict:
        tickers = list(price_history.keys())
        if len(tickers) < 2:
            return {}
        n = min(len(price_history[t]) for t in tickers)
        if n < 10:
            return {}

        rets_matrix: list[np.ndarray] = []
        kept: list[str] = []
        for t in tickers:
            arr = np.array(price_history[t][-n:], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.diff(arr) / arr[:-1]
            r = np.where(np.isfinite(r), r, 0.0)
            if r.size and np.std(r) > 0:
                rets_matrix.append(r)
                kept.append(t)
        if len(kept) < 2:
            return {}

        with np.errstate(invalid="ignore"):
            matrix = np.corrcoef(np.vstack(rets_matrix))
        out: dict[str, float] = {}
        for i, a in enumerate(kept):
            for j in range(i + 1, len(kept)):
                r = float(matrix[i, j])
                if np.isfinite(r) and abs(r) > self.CORR_THRESHOLD:
                    out[f"{a}__{kept[j]}"] = round(r, 3)
        return out

    # ---------- LLM observations ----------
    async def _generate_observations(self, metrics: dict[str, Any]) -> list[Observation]:
        system = (
            "You are a portfolio risk analyst. Given these risk metrics, identify the "
            "top 3 risk concerns in plain language.\n"
            "Rules:\n"
            "- Maximum 3 observations\n"
            "- Reference specific numbers (VaR, max drawdown, Sharpe, correlations)\n"
            "- Surface dangerous correlations and stress-test losses, not generic warnings\n"
            "- severity: info / warning / critical\n"
            '- Return JSON: {"observations": [{"severity": "info|warning|critical", "text": "..."}]}'
        )
        try:
            raw = await self._llm.complete(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(metrics, ensure_ascii=False)},
                ],
                response_model=_ObsList,
                temperature=0.3,
                max_tokens=600,
            )
            if not isinstance(raw, _ObsList):
                raise TypeError("LLM returned non-_ObsList response")
            obs = raw.observations[:3]
            return obs if obs else self._default_observations(metrics)
        except Exception:
            logger.exception("risk_analysis observation LLM failed; using defaults")
            return self._default_observations(metrics)

    def _default_observations(self, metrics: dict[str, Any]) -> list[Observation]:
        var95 = metrics.get("var", {}).get("one_day_95", 0)
        max_dd = metrics.get("max_drawdown_pct", 0)
        currency = metrics.get("currency", "USD")
        return [
            Observation(
                severity="info",
                text=(
                    f"1-day 95% VaR is ~{currency} {var95:,.0f}. On a typical bad day you "
                    "should expect a loss in this region; tail days can be materially worse."
                ),
            ),
            Observation(
                severity="warning" if max_dd <= -20 else "info",
                text=(
                    f"Worst peak-to-trough drawdown over the lookback was {max_dd:.1f}%. "
                    "Use it to gauge whether you'd hold through a similar event."
                ),
            ),
        ]

    def _empty_result(self, currency: str, *, reason: str) -> dict:
        return {
            "currency": currency,
            "portfolio_value": 0.0,
            "position_weights_pct": {},
            "var": {"method": "historical_simulation", "one_day_95": 0.0, "one_day_99": 0.0},
            "max_drawdown_pct": 0.0,
            "sharpe_ratio_annualised": 0.0,
            "stress_tests": [],
            "significant_correlations": {},
            "lookback_days": 0,
            "observations": [
                Observation(
                    severity="info",
                    text=f"Risk analysis skipped — {reason}. Add positions or retry once data is available.",
                ).model_dump()
            ],
            "disclaimer": DISCLAIMER,
        }


async def run(user: dict, llm: LLMClient) -> dict:
    return await RiskAnalysisAgent(llm).run(user)
