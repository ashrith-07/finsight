"""Portfolio risk analytics: VaR, drawdown, Sharpe, stress tests, correlations.

Deterministic numpy/scipy pipeline + one LLM pass for plain-English observations.
The Agno ``Agent`` wrapper is built lazily and exposes the yfinance + web-search MCP
toolkits to upstream orchestrators that prefer tool-driven flows.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import numpy as np
import yfinance as yf
from agno.agent import Agent
from pydantic import BaseModel

from src.agents.agno_react import coerce_json_dict
from src.llm.agno_model import get_agno_model
from src.llm.base import LLMClient
from src.mcp import calculator_mcp, portfolio_analytics_mcp, yfinance_mcp
from src.models import Observation

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Risk metrics are estimates based on historical price data and assume past "
    "behaviour is informative for future risk. Actual outcomes can be worse, "
    "particularly in regime shifts. Not investment advice."
)

RISK_REACT_INSTRUCTIONS = [
    "You are a Portfolio Risk Analyst for Valura AI.",
    "Always fetch roughly one year of historical prices (via tools) before estimating VaR.",
    "Run all five stress-test scenarios: 2008 financial crisis, 2020 COVID crash, 2022 rate hikes, dot-com bubble, and a mild correction.",
    "Flag any pairwise position correlation above 0.7 as dangerous.",
    "Compute Sharpe ratio when possible and explain it in plain language.",
    "Report VaR at both 95% and 99% confidence (1-day) when portfolio metrics are computed.",
    "Use tools to pull data; never invent prices or returns.",
]


def _risk_agno_payload_ok(d: dict[str, Any]) -> bool:
    if not isinstance(d, dict) or d.get("error"):
        return False
    if not d.get("disclaimer"):
        return False
    return "portfolio_value" in d or "metrics" in d


class _ObsList(BaseModel):
    observations: list[Observation]


_TICKER_RX = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z]{1,2})?\b")
_RISK_SUB_INTENT_RULES: list[tuple[str, re.Pattern[str]]] = [
    # ``[A-Za-z]{1,5}`` lets the regex stay case-insensitive while still acting
    # as a ticker proxy; the actual ticker resolution happens in
    # ``_extract_focus_ticker`` which checks against held positions.
    ("single_stock_risk", re.compile(
        r"\b(single[\s-]?stock|risk\s+(of|for)\s+[A-Za-z]{1,5}|"
        r"just\s+[A-Za-z]{1,5}\s+risk|how\s+risky\s+is\s+[A-Za-z]{1,5})\b", re.I)),
    ("volatility_analysis", re.compile(
        r"\b(volatil\w*|beta\w*|standard\s+deviation|bollinger|swings?|"
        r"how\s+much\s+do\s+(my|these)\s+stocks?\s+move)\b", re.I)),
    ("correlation", re.compile(
        r"\b(correlat\w*|move\s+together|tied\s+to|relationship\s+between)\b", re.I)),
    ("stress_test", re.compile(
        r"\b(stress\s+test|crash|what\s+if|scenario|drop\s+\d+|tail\s+risk|black\s+swan)\b", re.I)),
    ("var_only", re.compile(
        r"\b(value[\s-]?at[\s-]?risk|var\b|how\s+much\s+can\s+i\s+lose)\b", re.I)),
]


def _detect_sub_intent(intent: str, query: str) -> str:
    text = f"{intent or ''} {query or ''}"
    for label, rx in _RISK_SUB_INTENT_RULES:
        if rx.search(text):
            return label
    return "full_risk"


def _extract_focus_ticker(text: str, known_tickers: list[str]) -> str | None:
    """Pick the first uppercase token that matches a known portfolio holding."""
    if not text or not known_tickers:
        return None
    held = {t.upper() for t in known_tickers}
    for match in _TICKER_RX.finditer(text):
        cand = match.group(0).upper()
        if cand in held:
            return cand
    return None


def _safe_positions(user: dict) -> list[dict]:
    return [p for p in (user.get("positions") or []) if str(p.get("ticker") or "").strip()]


def _safe_beta(info: dict) -> float | None:
    val = info.get("beta")
    try:
        if val is None:
            return None
        f = float(val)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


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
        self._agno_react_checked = False
        self._agno_react: Agent | None = None

    def _ensure_agno_react(self) -> Agent | None:
        if self._agno_react_checked:
            return self._agno_react
        self._agno_react_checked = True
        model = get_agno_model()
        if model is None:
            self._agno_react = None
            return None
        try:
            self._agno_react = Agent(
                name="Portfolio Risk Analyst",
                model=model,
                tools=[yfinance_mcp, portfolio_analytics_mcp, calculator_mcp],
                instructions=RISK_REACT_INSTRUCTIONS,
                markdown=False,
                debug_mode=True,
            )
        except Exception as e:
            logger.warning("Risk Agno Agent construction failed: %s", e)
            self._agno_react = None
        return self._agno_react

    def _build_risk_context(self, user: dict, intent: str, query: str) -> str:
        sub = self._detect_sub_intent(intent, query)
        positions = user.get("positions") or []
        lines = [
            f"Investor: {user.get('name') or 'Unknown'}",
            f"Base currency: {user.get('base_currency') or 'USD'}",
            f"Classifier intent: {intent or '(none)'}",
            f"User query: {query or '(none)'}",
            f"Detected sub-task: {sub}",
            "",
            "Positions (ticker, qty, avg_cost):",
        ]
        if not positions:
            lines.append("  (none)")
        else:
            for p in positions[:40]:
                lines.append(
                    f"  - {p.get('ticker')}: qty={p.get('quantity')} avg_cost={p.get('avg_cost')}"
                )
        lines.append(
            "\nUse tools to build risk metrics, then respond with ONE minified JSON object "
            "matching the shape produced by Valura's risk engine: include disclaimer (string), "
            "currency, portfolio_value, and either metrics (full book) or focused keys "
            "(var, stress_tests, significant_correlations, etc.) plus observations as "
            "list of {severity, text} objects and sub_intent."
        )
        return "\n".join(lines)

    # ---------- Agno surface ----------
    def as_agno_agent(self) -> Agent:
        ag = self._ensure_agno_react()
        if ag is None:
            raise RuntimeError(
                "RiskAnalysisAgent.as_agno_agent requires OPENAI_API_KEY or GROQ_API_KEY."
            )
        return ag

    @staticmethod
    def _detect_sub_intent(intent: str, query: str) -> str:
        return _detect_sub_intent(intent, query)

    # ---------- main pipeline ----------
    async def run(
        self,
        user: dict,
        intent: str = "",
        query: str = "",
    ) -> dict:
        positions = _safe_positions(user)
        currency = str(user.get("base_currency") or "USD").upper()

        if not positions:
            return self._empty_result(currency, reason="no positions on file")

        react = self._ensure_agno_react()
        if react is not None:
            try:
                ctx = self._build_risk_context(user, intent, query)
                resp = await react.arun(ctx, stream=False)
                d = coerce_json_dict(resp)
                if d and _risk_agno_payload_ok(d):
                    return d
            except Exception as e:
                logger.error("Risk Agno react error: %s", e)

        sub = self._detect_sub_intent(intent, query)

        # Compute the shared base once; sub-intents add focused sections.
        base = await self._compute_base_metrics(user, positions, currency)
        if "error" in base:
            return base["error"]

        if sub == "single_stock_risk":
            focus = _extract_focus_ticker(
                f"{intent} {query}", list(base["weights"].keys())
            ) or next(iter(base["weights"]), None)
            if focus is None:
                base["metrics"]["sub_intent"] = sub
                return self._finalise(base["metrics"])
            return await self._single_stock_risk(focus, user, base)

        if sub == "volatility_analysis":
            return await self._volatility_analysis(user, base)

        if sub == "correlation":
            return self._correlation_only(base)

        if sub == "var_only":
            return self._var_only(base)

        if sub == "stress_test":
            return self._stress_test_only(base)

        # full_risk (default) — return everything with LLM observations.
        metrics = base["metrics"]
        metrics["sub_intent"] = "full_risk"
        observations = await self._generate_observations(metrics)
        return self._finalise(metrics, observations)

    async def _compute_base_metrics(
        self, user: dict, positions: list[dict], currency: str,
    ) -> dict[str, Any]:
        price_history = await self._fetch_all_history(positions)
        valid = {t: pr for t, pr in price_history.items() if pr and len(pr) >= 30}
        if not valid:
            return {"error": self._empty_result(currency, reason="historical price data unavailable")}

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
            return {"error": self._empty_result(currency, reason="portfolio value evaluates to zero")}

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

        metrics: dict[str, Any] = {
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
        return {
            "metrics": metrics,
            "weights": weights,
            "position_values": position_values,
            "portfolio_value": portfolio_value,
            "valid": valid,
            "returns_by_ticker": returns_by_ticker,
        }

    def _finalise(self, metrics: dict[str, Any], observations: list[Observation] | None = None) -> dict:
        if observations is None:
            observations = self._default_observations(metrics)
        metrics["observations"] = [o.model_dump() for o in observations]
        metrics["disclaimer"] = DISCLAIMER
        return metrics

    # ---------- focused sub-intents ----------
    def _var_only(self, base: dict) -> dict:
        m = base["metrics"]
        focused = {
            "currency": m["currency"],
            "portfolio_value": m["portfolio_value"],
            "var": m["var"],
            "lookback_days": m["lookback_days"],
            "sub_intent": "var_only",
        }
        var95 = m["var"]["one_day_95"]
        var99 = m["var"]["one_day_99"]
        obs = [
            Observation(
                severity="warning" if var95 > 0.05 * m["portfolio_value"] else "info",
                text=(
                    f"1-day 95% VaR is {m['currency']} {var95:,.0f} — roughly "
                    f"{(var95 / max(m['portfolio_value'], 1) * 100):.2f}% of the book."
                ),
            ),
            Observation(
                severity="info",
                text=(
                    f"At 99% confidence (rare days), the 1-day VaR rises to "
                    f"{m['currency']} {var99:,.0f}."
                ),
            ),
        ]
        return self._finalise(focused, obs)

    def _stress_test_only(self, base: dict) -> dict:
        m = base["metrics"]
        focused = {
            "currency": m["currency"],
            "portfolio_value": m["portfolio_value"],
            "stress_tests": m["stress_tests"],
            "sub_intent": "stress_test",
        }
        worst = min(m["stress_tests"], key=lambda s: s.get("portfolio_loss_pct", 0)) if m["stress_tests"] else {}
        obs = []
        if worst:
            obs.append(Observation(
                severity="critical" if worst.get("portfolio_loss_pct", 0) <= -40 else "warning",
                text=(
                    f"Worst modeled scenario ({worst.get('scenario')}): "
                    f"{worst.get('portfolio_loss_pct'):+.2f}% drawdown → "
                    f"{m['currency']} {abs(worst.get('portfolio_loss_usd', 0)):,.0f} loss."
                ),
            ))
        for s in m["stress_tests"][:2]:
            if s is worst:
                continue
            obs.append(Observation(
                severity="info",
                text=(
                    f"{s.get('scenario')}: {s.get('portfolio_loss_pct'):+.2f}% draw, "
                    f"surviving value {m['currency']} {s.get('surviving_value'):,.0f}."
                ),
            ))
        return self._finalise(focused, obs[:3])

    def _correlation_only(self, base: dict) -> dict:
        m = base["metrics"]
        corrs = m["significant_correlations"]
        focused = {
            "currency": m["currency"],
            "portfolio_value": m["portfolio_value"],
            "significant_correlations": corrs,
            "lookback_days": m["lookback_days"],
            "sub_intent": "correlation",
        }
        obs = []
        if corrs:
            top = max(corrs.items(), key=lambda kv: abs(kv[1]))
            a, b = top[0].split("__")
            obs.append(Observation(
                severity="warning" if abs(top[1]) > 0.85 else "info",
                text=(
                    f"Strongest correlation in the book: {a} ↔ {b} at "
                    f"{top[1]:+.2f} — they tend to move together in stress."
                ),
            ))
            for pair, r in list(corrs.items())[1:3]:
                a2, b2 = pair.split("__")
                obs.append(Observation(
                    severity="info",
                    text=f"{a2} ↔ {b2}: correlation {r:+.2f}.",
                ))
        else:
            obs.append(Observation(
                severity="info",
                text="No pairwise correlations exceeded the 0.7 significance threshold.",
            ))
        return self._finalise(focused, obs[:3])

    async def _volatility_analysis(self, user: dict, base: dict) -> dict:
        """Beta + 30d std dev + Bollinger band position per holding, ranked by volatility."""
        weights = base["weights"]
        valid = base["valid"]

        async def beta_for(ticker: str) -> float | None:
            def _blocking() -> float | None:
                try:
                    return _safe_beta(yf.Ticker(ticker).info or {})
                except Exception:
                    return None
            return await asyncio.to_thread(_blocking)

        betas = await asyncio.gather(*(beta_for(t) for t in weights))
        rows = []
        for (ticker, _w), beta in zip(weights.items(), betas):
            prices = np.array(valid.get(ticker, []), dtype=float)
            if prices.size < 31:
                rows.append({"ticker": ticker, "beta": beta, "stddev_30d_pct": None,
                             "bollinger_position": None, "annualised_vol_pct": None})
                continue
            window = prices[-31:]  # 31 closes → 30 daily returns.
            rets = np.diff(window) / window[:-1]
            stddev_30d = float(np.std(rets, ddof=1)) * 100.0
            ann_vol = stddev_30d * float(np.sqrt(self.TRADING_DAYS))

            # Bollinger band (20-day, 2σ) position: 0=lower band, 1=upper band, 0.5=middle.
            window = prices[-20:] if prices.size >= 20 else prices
            ma = float(np.mean(window))
            sd = float(np.std(window, ddof=1))
            current = float(prices[-1])
            if sd > 0:
                upper, lower = ma + 2 * sd, ma - 2 * sd
                pos = (current - lower) / (upper - lower)
                bollinger_pos = round(max(0.0, min(1.0, pos)), 3)
            else:
                bollinger_pos = None

            rows.append({
                "ticker": ticker,
                "beta": round(beta, 3) if beta is not None else None,
                "stddev_30d_pct": round(stddev_30d, 3),
                "annualised_vol_pct": round(ann_vol, 2),
                "bollinger_position": bollinger_pos,
            })

        rows.sort(
            key=lambda r: r.get("annualised_vol_pct") if r.get("annualised_vol_pct") is not None else -1,
            reverse=True,
        )

        m = base["metrics"]
        focused = {
            "currency": m["currency"],
            "portfolio_value": m["portfolio_value"],
            "volatility_table": rows,
            "sub_intent": "volatility_analysis",
        }

        obs = []
        if rows:
            top = rows[0]
            if top.get("annualised_vol_pct") is not None:
                obs.append(Observation(
                    severity="warning" if top["annualised_vol_pct"] > 35 else "info",
                    text=(
                        f"Most volatile holding: {top['ticker']} — annualised vol "
                        f"{top['annualised_vol_pct']:.1f}%"
                        + (f", beta {top['beta']:.2f}." if top.get("beta") is not None else ".")
                    ),
                ))
            high_beta = [r for r in rows if (r.get("beta") or 0) > 1.5]
            if high_beta:
                names = ", ".join(r["ticker"] for r in high_beta[:3])
                obs.append(Observation(
                    severity="warning",
                    text=f"High-beta names amplifying market swings: {names} (beta > 1.5).",
                ))
        if not obs:
            obs.append(Observation(severity="info", text="Volatility metrics computed; no outliers detected."))
        return self._finalise(focused, obs[:3])

    async def _single_stock_risk(self, ticker: str, user: dict, base: dict) -> dict:
        """Position size, what-if drops, correlation with rest of book, sector contribution."""
        ticker = ticker.upper()
        weights = base["weights"]
        position_values = base["position_values"]
        portfolio_value = base["portfolio_value"]
        valid = base["valid"]

        weight_pct = round(weights.get(ticker, 0.0) * 100, 2)
        position_value = position_values.get(ticker, 0.0)

        what_if = []
        for drop in (0.20, 0.30, 0.50):
            stock_loss = position_value * drop
            new_total = portfolio_value - stock_loss
            what_if.append({
                "stock_drop_pct": round(-drop * 100, 2),
                "stock_loss_usd": round(-stock_loss, 2),
                "portfolio_loss_pct": round(-stock_loss / portfolio_value * 100, 2)
                    if portfolio_value > 0 else 0.0,
                "portfolio_value_after": round(new_total, 2),
            })

        # Correlation with every other holding.
        corrs_with_others: list[dict[str, Any]] = []
        if ticker in valid and len(valid) >= 2:
            n = min(len(valid[ticker]), *(len(v) for v in valid.values()))
            if n >= 10:
                base_arr = np.array(valid[ticker][-n:], dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    base_rets = np.diff(base_arr) / base_arr[:-1]
                base_rets = np.where(np.isfinite(base_rets), base_rets, 0.0)
                for other, prices in valid.items():
                    if other == ticker:
                        continue
                    arr = np.array(prices[-n:], dtype=float)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        rets = np.diff(arr) / arr[:-1]
                    rets = np.where(np.isfinite(rets), rets, 0.0)
                    if rets.size != base_rets.size or np.std(rets) == 0 or np.std(base_rets) == 0:
                        continue
                    r = float(np.corrcoef(base_rets, rets)[0, 1])
                    if np.isfinite(r):
                        corrs_with_others.append({"other_ticker": other, "correlation": round(r, 3)})
        corrs_with_others.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        # Sector contribution lookup (this stock's sector + how much of book it represents).
        def _sector() -> str:
            try:
                return str((yf.Ticker(ticker).info or {}).get("sector") or "Unknown")
            except Exception:
                return "Unknown"
        sector = await asyncio.to_thread(_sector)

        m = base["metrics"]
        focused = {
            "currency": m["currency"],
            "portfolio_value": portfolio_value,
            "focus_ticker": ticker,
            "position_weight_pct": weight_pct,
            "position_value": round(position_value, 2),
            "what_if_drops": what_if,
            "correlations_with_other_holdings": corrs_with_others,
            "sector": sector,
            "sub_intent": "single_stock_risk",
        }

        obs = [
            Observation(
                severity="warning" if weight_pct > 40 else "info",
                text=(
                    f"{ticker} represents {weight_pct:.2f}% of the book "
                    f"(position value ~{m['currency']} {position_value:,.0f})."
                ),
            ),
            Observation(
                severity="warning" if what_if[1]["portfolio_loss_pct"] <= -10 else "info",
                text=(
                    f"If {ticker} falls 30%, the portfolio loses "
                    f"{abs(what_if[1]['portfolio_loss_pct']):.2f}% "
                    f"({m['currency']} {abs(what_if[1]['stock_loss_usd']):,.0f})."
                ),
            ),
        ]
        if corrs_with_others:
            top_corr = corrs_with_others[0]
            obs.append(Observation(
                severity="warning" if abs(top_corr["correlation"]) > 0.8 else "info",
                text=(
                    f"{ticker} most correlated with {top_corr['other_ticker']} "
                    f"({top_corr['correlation']:+.2f}) — they tend to move together."
                ),
            ))
        return self._finalise(focused, obs[:3])

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


async def run(user: dict, llm: LLMClient, intent: str = "", query: str = "") -> dict:
    return await RiskAnalysisAgent(llm).run(user, intent=intent, query=query)
