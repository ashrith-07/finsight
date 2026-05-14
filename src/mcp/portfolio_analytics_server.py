"""Portfolio analytics MCP toolkit: combines yfinance feeds with quant calcs.

Tools degrade to ``{"error": ...}`` on failure so async ``gather`` callers can
keep the rest of the analytics pipeline alive.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import yfinance as yf
from agno.tools import Toolkit

from src.logging_config import get_logger

logger = get_logger("mcp.portfolio_analytics")

RISK_FREE_RATE = 0.0525
TRADING_DAYS = 252
SECTOR_OVERWEIGHT_THRESHOLD = 40.0
COUNTRY_OVERWEIGHT_THRESHOLD = 80.0


def _safe_float(value, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        f = float(value)
    except (TypeError, ValueError):
        return default
    if f != f:
        return default
    return f


def _normalise_weights(weights: list[float]) -> list[float]:
    arr = np.array([_safe_float(w, 0.0) or 0.0 for w in (weights or [])], dtype=float)
    s = arr.sum()
    if s <= 0:
        return arr.tolist()
    return (arr / s).tolist()


def _ticker_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception as e:
        logger.warning("portfolio_analytics_mcp info fetch failed for %s: %s", ticker, e)
        return {}


def _ticker_close_history(ticker: str, period: str = "1y") -> np.ndarray:
    try:
        hist = yf.Ticker(ticker).history(period=period, interval="1d")
        if hist is None or hist.empty:
            return np.array([])
        return hist["Close"].dropna().to_numpy(dtype=float)
    except Exception as e:
        logger.warning("portfolio_analytics_mcp history fetch failed for %s: %s", ticker, e)
        return np.array([])


def _parallel_map(fn, items, max_workers: int = 8):
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        return list(pool.map(fn, items))


class PortfolioAnalyticsMCPServer(Toolkit):
    """Beta, sector/geographic exposure, dividends, attribution, frontier."""

    name = "portfolio_analytics_mcp"

    def __init__(self) -> None:
        super().__init__(name="portfolio_analytics_mcp", auto_register=False)
        self.register(self.efficient_frontier_point)
        self.register(self.portfolio_beta)
        self.register(self.sector_exposure)
        self.register(self.dividend_analysis)
        self.register(self.geographic_exposure)
        self.register(self.performance_attribution)

    def efficient_frontier_point(
        self,
        tickers: list[str],
        weights: list[float],
    ) -> dict:
        """Annualised return, vol, and Sharpe for the given (tickers, weights)."""
        try:
            clean = [str(t).upper() for t in (tickers or []) if str(t).strip()]
            if not clean:
                return {"error": "no tickers provided"}

            histories = _parallel_map(_ticker_close_history, clean)
            usable = [(t, h) for t, h in zip(clean, histories) if h.size >= 30]
            if not usable:
                return {"error": "insufficient price history"}

            n = min(h.size for _, h in usable)
            returns_matrix = []
            kept_tickers: list[str] = []
            for t, h in usable:
                aligned = h[-n:]
                with np.errstate(divide="ignore", invalid="ignore"):
                    rets = np.diff(aligned) / aligned[:-1]
                rets = np.where(np.isfinite(rets), rets, 0.0)
                returns_matrix.append(rets)
                kept_tickers.append(t)

            w_full = _normalise_weights(weights)
            # Weight vector must align with the surviving tickers; fall back to
            # equal-weight if lengths don't match.
            if len(w_full) != len(clean):
                w = np.array([1.0 / len(kept_tickers)] * len(kept_tickers))
            else:
                idx = [clean.index(t) for t in kept_tickers]
                w = np.array([w_full[i] for i in idx], dtype=float)
                if w.sum() > 0:
                    w = w / w.sum()
                else:
                    w = np.array([1.0 / len(kept_tickers)] * len(kept_tickers))

            R = np.vstack(returns_matrix)
            mean_daily = R.mean(axis=1)
            cov_daily = np.cov(R)
            port_mean_daily = float(w @ mean_daily)
            port_var_daily = float(w @ cov_daily @ w)

            ann_return = port_mean_daily * TRADING_DAYS
            ann_vol = float(np.sqrt(max(port_var_daily, 0.0)) * np.sqrt(TRADING_DAYS))
            sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0.0

            return {
                "tickers": kept_tickers,
                "weights": [round(float(x), 4) for x in w],
                "portfolio_expected_return_annual": round(ann_return, 4),
                "portfolio_volatility_annual": round(ann_vol, 4),
                "sharpe_ratio": round(sharpe, 3),
                "is_efficient": bool(sharpe > 1.0),
                "lookback_days": int(n),
                "risk_free_rate": RISK_FREE_RATE,
            }
        except Exception as e:
            logger.warning("portfolio_analytics_mcp.efficient_frontier_point failed: %s", e)
            return {"error": str(e)}

    def portfolio_beta(
        self,
        positions: list[dict[str, Any]],
        benchmark: str = "SPY",
    ) -> dict:
        """Weighted portfolio beta with per-position breakdown."""
        try:
            rows = [p for p in (positions or []) if isinstance(p, dict) and p.get("ticker")]
            if not rows:
                return {"error": "no positions provided"}

            tickers = [str(p["ticker"]).upper() for p in rows]

            def fetch(t: str) -> tuple[str, float | None]:
                info = _ticker_info(t)
                return t, _safe_float(info.get("beta"))

            beta_map = dict(_parallel_map(fetch, tickers))
            total_value = sum(_safe_float(p.get("current_value"), 0.0) or 0.0 for p in rows)

            per_position = []
            weighted_beta = 0.0
            weight_sum = 0.0
            for p in rows:
                t = str(p["ticker"]).upper()
                value = _safe_float(p.get("current_value"), 0.0) or 0.0
                w = (value / total_value) if total_value > 0 else 0.0
                beta = beta_map.get(t)
                if beta is not None:
                    weighted_beta += beta * w
                    weight_sum += w
                per_position.append({
                    "ticker": t,
                    "weight_pct": round(w * 100, 2),
                    "beta": round(beta, 3) if beta is not None else None,
                    "contribution": round(beta * w, 4) if beta is not None else None,
                })

            # Renormalise so missing-beta positions don't deflate the result.
            portfolio_beta = (weighted_beta / weight_sum) if weight_sum > 0 else None

            if portfolio_beta is None:
                interpretation = "unknown"
            elif portfolio_beta > 1.2:
                interpretation = "aggressive"
            elif portfolio_beta < 0.8:
                interpretation = "defensive"
            else:
                interpretation = "moderate"

            return {
                "benchmark": benchmark,
                "portfolio_beta": round(portfolio_beta, 3) if portfolio_beta is not None else None,
                "interpretation": interpretation,
                "per_position_beta": per_position,
                "coverage_pct": round(weight_sum * 100, 2),
            }
        except Exception as e:
            logger.warning("portfolio_analytics_mcp.portfolio_beta failed: %s", e)
            return {"error": str(e)}

    def sector_exposure(
        self,
        tickers: list[str],
        weights: list[float],
    ) -> dict:
        """Sector breakdown with overweight flagging at 40%."""
        try:
            clean = [str(t).upper() for t in (tickers or []) if str(t).strip()]
            if not clean:
                return {"error": "no tickers provided"}

            w_full = _normalise_weights(weights)
            if len(w_full) != len(clean):
                w_full = [1.0 / len(clean)] * len(clean)

            sectors = _parallel_map(
                lambda t: (t, str(_ticker_info(t).get("sector") or "Unknown")), clean
            )

            by_sector: dict[str, float] = {}
            for (t, sector), w in zip(sectors, w_full):
                by_sector[sector] = by_sector.get(sector, 0.0) + (w * 100.0)

            ordered = sorted(by_sector.items(), key=lambda kv: kv[1], reverse=True)
            breakdown = [
                {
                    "sector": sector,
                    "exposure_pct": round(pct, 2),
                    "overweight": pct > SECTOR_OVERWEIGHT_THRESHOLD,
                }
                for sector, pct in ordered
            ]
            overweight = [b for b in breakdown if b["overweight"]]
            return {
                "by_sector": {b["sector"]: b["exposure_pct"] for b in breakdown},
                "breakdown": breakdown,
                "overweight_sectors": overweight,
                "overweight_threshold_pct": SECTOR_OVERWEIGHT_THRESHOLD,
            }
        except Exception as e:
            logger.warning("portfolio_analytics_mcp.sector_exposure failed: %s", e)
            return {"error": str(e)}

    def dividend_analysis(self, positions: list[dict[str, Any]]) -> dict:
        """Annual dividend income, projected monthly income, top payers.

        ``dividendRate`` (annual $/share) anchors the calc when present —
        ``dividendYield`` is only used as a backup since yfinance changed its
        semantics (now percent, was fraction) across versions.
        """
        try:
            rows = [p for p in (positions or []) if isinstance(p, dict) and p.get("ticker")]
            if not rows:
                return {"error": "no positions provided"}

            total_value = sum(_safe_float(p.get("current_value"), 0.0) or 0.0 for p in rows)
            tickers = [str(p["ticker"]).upper() for p in rows]

            def fetch(t: str) -> tuple[str, dict]:
                info = _ticker_info(t)
                return t, {
                    "dividend_yield_raw": _safe_float(info.get("dividendYield")),
                    "dividend_rate": _safe_float(info.get("dividendRate")),
                    "current_price": _safe_float(
                        info.get("currentPrice") or info.get("regularMarketPrice")
                    ),
                }

            div_map = dict(_parallel_map(fetch, tickers))

            per_position = []
            total_annual_income = 0.0
            payers = 0
            for p in rows:
                t = str(p["ticker"]).upper()
                value = _safe_float(p.get("current_value"), 0.0) or 0.0
                qty = _safe_float(p.get("quantity"), 0.0) or 0.0
                meta = div_map.get(t, {})
                rate = meta.get("dividend_rate")
                price = meta.get("current_price")

                # Prefer rate × shares; if shares missing infer from value÷price.
                if rate and qty > 0:
                    annual_income = rate * qty
                elif rate and price and price > 0 and value > 0:
                    annual_income = rate * (value / price)
                else:
                    # Last-resort: normalise yield then apply to value.
                    yr = meta.get("dividend_yield_raw")
                    if yr is None:
                        yield_fraction = None
                    elif yr > 1.0:
                        yield_fraction = yr / 100.0
                    elif yr > 0.30:
                        # Implausibly large for a fraction (>30% yield) — assume
                        # this reading is already percent-units.
                        yield_fraction = yr / 100.0
                    else:
                        yield_fraction = yr
                    annual_income = (
                        yield_fraction * value if (yield_fraction and value) else 0.0
                    )

                if annual_income > 0:
                    payers += 1
                total_annual_income += annual_income
                yield_pct = (annual_income / value * 100) if value > 0 else None
                per_position.append({
                    "ticker": t,
                    "current_value": round(value, 2),
                    "dividend_yield_pct": (
                        round(yield_pct, 3) if yield_pct is not None else None
                    ),
                    "dividend_rate": round(rate, 4) if rate is not None else None,
                    "annual_income": round(annual_income, 2),
                })

            per_position.sort(key=lambda r: r["annual_income"], reverse=True)
            portfolio_yield_pct = (
                round((total_annual_income / total_value) * 100, 3) if total_value > 0 else 0.0
            )

            return {
                "annual_dividend_income": round(total_annual_income, 2),
                "projected_monthly_income": round(total_annual_income / 12.0, 2),
                "portfolio_yield_pct": portfolio_yield_pct,
                "positions_paying_dividends": payers,
                "total_positions": len(rows),
                "per_position": per_position,
                "top_payers": per_position[:3],
            }
        except Exception as e:
            logger.warning("portfolio_analytics_mcp.dividend_analysis failed: %s", e)
            return {"error": str(e)}

    def geographic_exposure(
        self,
        tickers: list[str],
        weights: list[float],
    ) -> dict:
        """Country-of-domicile breakdown with 80% home-bias flag."""
        try:
            clean = [str(t).upper() for t in (tickers or []) if str(t).strip()]
            if not clean:
                return {"error": "no tickers provided"}

            w_full = _normalise_weights(weights)
            if len(w_full) != len(clean):
                w_full = [1.0 / len(clean)] * len(clean)

            countries = _parallel_map(
                lambda t: (t, str(_ticker_info(t).get("country") or "Unknown")), clean
            )

            by_country: dict[str, float] = {}
            for (t, country), w in zip(countries, w_full):
                by_country[country] = by_country.get(country, 0.0) + (w * 100.0)

            ordered = sorted(by_country.items(), key=lambda kv: kv[1], reverse=True)
            top = ordered[0] if ordered else ("Unknown", 0.0)

            return {
                "by_country": {c: round(p, 2) for c, p in ordered},
                "breakdown": [{"country": c, "exposure_pct": round(p, 2)} for c, p in ordered],
                "top_country": {"country": top[0], "exposure_pct": round(top[1], 2)},
                "home_bias_flag": top[1] > COUNTRY_OVERWEIGHT_THRESHOLD,
                "home_bias_threshold_pct": COUNTRY_OVERWEIGHT_THRESHOLD,
            }
        except Exception as e:
            logger.warning("portfolio_analytics_mcp.geographic_exposure failed: %s", e)
            return {"error": str(e)}

    def performance_attribution(
        self,
        positions: list[dict[str, Any]],
        prices: dict[str, float],
    ) -> dict:
        """Per-position contribution to portfolio return, ranked best→worst.

        Each position must carry ``ticker`` and ``avg_cost``, plus ``quantity``
        when inferring weights from holdings. Supply either ``current_price`` on
        the row or a ``prices`` map of ticker (upper-case string) → last price.

        Type hints use ``dict[str, Any]`` / ``dict[str, float]`` so provider tool
        JSON schemas allow dynamic keys (plain ``dict`` / ``list[dict]`` often
        compile to ``additionalProperties: false`` and reject valid calls).
        """
        try:
            rows = [p for p in (positions or []) if isinstance(p, dict) and p.get("ticker")]
            if not rows:
                return {"error": "no positions provided"}

            enriched = []
            total_value = 0.0
            for p in rows:
                t = str(p["ticker"]).upper()
                qty = _safe_float(p.get("quantity"), 0.0) or 0.0
                avg_cost = _safe_float(p.get("avg_cost"), 0.0) or 0.0
                price = _safe_float((prices or {}).get(t)) or _safe_float(p.get("current_price"))
                if price is None or qty <= 0 or avg_cost <= 0:
                    continue
                cur_value = qty * price
                total_value += cur_value
                position_return = (price / avg_cost - 1.0)
                enriched.append({
                    "ticker": t,
                    "quantity": qty,
                    "avg_cost": avg_cost,
                    "current_price": price,
                    "current_value": cur_value,
                    "position_return_pct": position_return * 100,
                })

            if total_value <= 0:
                return {"error": "portfolio value evaluates to zero"}

            attribution = []
            total_contribution = 0.0
            for r in enriched:
                weight = r["current_value"] / total_value
                contribution = (r["position_return_pct"] / 100.0) * weight
                total_contribution += contribution
                attribution.append({
                    "ticker": r["ticker"],
                    "weight_pct": round(weight * 100, 2),
                    "position_return_pct": round(r["position_return_pct"], 2),
                    "contribution_to_return_pct": round(contribution * 100, 4),
                    "current_value": round(r["current_value"], 2),
                })

            attribution.sort(key=lambda x: x["contribution_to_return_pct"], reverse=True)
            best = attribution[0] if attribution else None
            worst = attribution[-1] if len(attribution) >= 2 else None
            helpers = [a for a in attribution if a["contribution_to_return_pct"] > 0]
            hurters = [a for a in attribution if a["contribution_to_return_pct"] < 0]

            return {
                "portfolio_return_pct": round(total_contribution * 100, 4),
                "total_value": round(total_value, 2),
                "attribution": attribution,
                "best_contributor": best,
                "worst_contributor": worst,
                "positive_contributors": len(helpers),
                "negative_contributors": len(hurters),
            }
        except Exception as e:
            logger.warning("portfolio_analytics_mcp.performance_attribution failed: %s", e)
            return {"error": str(e)}
