"""yfinance-backed MCP toolkit. All tools return JSON-safe dicts; failures degrade to ``{"error": ...}``."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import yfinance as yf
from agno.tools import Toolkit

from src.logging_config import get_logger

logger = get_logger("mcp.yfinance")


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def _safe_int(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _truncate(text: str, n: int) -> str:
    s = str(text or "")
    return s[:n]


class YFinanceMCPServer(Toolkit):
    """Wraps yfinance and exposes structured tools an Agno agent can call."""

    name = "yfinance_mcp"

    def __init__(self) -> None:
        # Disable auto-registration: tools we want exposed are listed explicitly below.
        super().__init__(name="yfinance_mcp", auto_register=False)
        self.register(self.get_price_snapshot)
        self.register(self.get_historical_prices)
        self.register(self.get_company_fundamentals)
        self.register(self.get_financial_statements)
        self.register(self.get_options_data)
        self.register(self.screen_stocks)

    def get_price_snapshot(self, ticker: str) -> dict:
        """Live price + key trading metrics for one ticker."""
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            if not info or len(info) < 5:
                return {"ticker": ticker, "error": "no data available"}

            current = _safe_float(
                info.get("currentPrice") or info.get("regularMarketPrice")
            )
            prev_close = _safe_float(
                info.get("previousClose") or info.get("regularMarketPreviousClose")
            )
            day_change_pct: float | None = None
            if current is not None and prev_close not in (None, 0):
                day_change_pct = round((current - prev_close) / prev_close * 100, 4)

            return {
                "ticker": ticker.upper(),
                "currency": str(info.get("currency") or "USD").upper(),
                "current_price": current,
                "previous_close": prev_close,
                "day_change_pct": day_change_pct,
                "volume": _safe_int(
                    info.get("volume") if info.get("volume") is not None
                    else info.get("regularMarketVolume")
                ),
                "market_cap": _safe_float(info.get("marketCap")),
                "fifty_two_week_high": _safe_float(info.get("fiftyTwoWeekHigh")),
                "fifty_two_week_low": _safe_float(info.get("fiftyTwoWeekLow")),
                "pe_ratio": _safe_float(info.get("trailingPE") or info.get("forwardPE")),
                "beta": _safe_float(info.get("beta")),
                "dividend_yield": _safe_float(info.get("dividendYield")),
            }
        except Exception as e:
            logger.warning("yfinance_mcp.get_price_snapshot(%s) failed: %s", ticker, e)
            return {"ticker": ticker, "error": str(e)}

    def get_historical_prices(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> dict:
        """OHLCV bars; ``period`` ∈ {1d,5d,1mo,3mo,6mo,1y,2y,5y}."""
        try:
            hist = yf.Ticker(ticker).history(period=period, interval=interval)
            if hist is None or hist.empty:
                return {"ticker": ticker, "period": period, "bars": [], "error": "no history"}

            bars = []
            for idx, row in hist.iterrows():
                bars.append(
                    {
                        "date": idx.strftime("%Y-%m-%d %H:%M:%S"),
                        "open": _safe_float(row.get("Open")),
                        "high": _safe_float(row.get("High")),
                        "low": _safe_float(row.get("Low")),
                        "close": _safe_float(row.get("Close")),
                        "volume": _safe_int(row.get("Volume")),
                    }
                )
            return {
                "ticker": ticker.upper(),
                "period": period,
                "interval": interval,
                "bars": bars,
            }
        except Exception as e:
            logger.warning("yfinance_mcp.get_historical_prices(%s) failed: %s", ticker, e)
            return {"ticker": ticker, "period": period, "bars": [], "error": str(e)}

    def get_company_fundamentals(self, ticker: str) -> dict:
        """Company-level fundamentals + descriptive metadata."""
        try:
            info = yf.Ticker(ticker).info or {}
            if not info or len(info) < 5:
                return {"ticker": ticker, "error": "no data available"}

            return {
                "ticker": ticker.upper(),
                "name": str(info.get("longName") or info.get("shortName") or ticker),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "country": info.get("country"),
                "employee_count": _safe_int(info.get("fullTimeEmployees")),
                "revenue": _safe_float(info.get("totalRevenue")),
                "earnings": _safe_float(info.get("netIncomeToCommon")),
                "profit_margin": _safe_float(info.get("profitMargins")),
                "debt_to_equity": _safe_float(info.get("debtToEquity")),
                "current_ratio": _safe_float(info.get("currentRatio")),
                "roe": _safe_float(info.get("returnOnEquity")),
                "roa": _safe_float(info.get("returnOnAssets")),
                "description": _truncate(info.get("longBusinessSummary"), 500),
            }
        except Exception as e:
            logger.warning("yfinance_mcp.get_company_fundamentals(%s) failed: %s", ticker, e)
            return {"ticker": ticker, "error": str(e)}

    def get_financial_statements(self, ticker: str) -> dict:
        """Most-recent 4 quarters of revenue, net income, total assets, total debt."""
        try:
            t = yf.Ticker(ticker)
            financials = t.financials
            balance = t.balance_sheet

            def _row_last_n(df, label_candidates: list[str], n: int = 4) -> dict[str, float | None]:
                if df is None or getattr(df, "empty", True):
                    return {}
                # yfinance puts metric labels on the index, periods on columns (most recent first).
                label = next((c for c in label_candidates if c in df.index), None)
                if label is None:
                    return {}
                series = df.loc[label].iloc[:n]
                return {str(period): _safe_float(value) for period, value in series.items()}

            return {
                "ticker": ticker.upper(),
                "revenue": _row_last_n(financials, ["Total Revenue", "TotalRevenue"]),
                "net_income": _row_last_n(financials, ["Net Income", "NetIncome"]),
                "total_assets": _row_last_n(balance, ["Total Assets", "TotalAssets"]),
                "total_debt": _row_last_n(
                    balance, ["Total Debt", "TotalDebt", "Long Term Debt"]
                ),
            }
        except Exception as e:
            logger.warning("yfinance_mcp.get_financial_statements(%s) failed: %s", ticker, e)
            return {"ticker": ticker, "error": str(e)}

    def get_options_data(self, ticker: str) -> dict:
        """Nearest-expiry options chain summary; safe defaults when unsupported."""
        try:
            t = yf.Ticker(ticker)
            expiries = list(t.options or ())
            if not expiries:
                return {"ticker": ticker.upper(), "error": "no options listed"}

            nearest = expiries[0]
            chain = t.option_chain(nearest)
            calls = chain.calls if chain.calls is not None else None
            puts = chain.puts if chain.puts is not None else None

            call_oi = float(calls["openInterest"].dropna().sum()) if calls is not None else 0.0
            put_oi = float(puts["openInterest"].dropna().sum()) if puts is not None else 0.0
            put_call_ratio = round(put_oi / call_oi, 4) if call_oi > 0 else None

            iv_avg = None
            iv_values = []
            if calls is not None and "impliedVolatility" in calls.columns:
                iv_values.extend(calls["impliedVolatility"].dropna().tolist())
            if puts is not None and "impliedVolatility" in puts.columns:
                iv_values.extend(puts["impliedVolatility"].dropna().tolist())
            if iv_values:
                iv_avg = round(sum(iv_values) / len(iv_values), 4)

            return {
                "ticker": ticker.upper(),
                "nearest_expiry": nearest,
                "put_call_ratio": put_call_ratio,
                "implied_volatility_avg": iv_avg,
                "total_open_interest": int(call_oi + put_oi),
                "expiries_available": len(expiries),
            }
        except Exception as e:
            logger.warning("yfinance_mcp.get_options_data(%s) failed: %s", ticker, e)
            return {"ticker": ticker, "error": str(e)}

    def screen_stocks(self, tickers: list[str]) -> list[dict]:
        """Concurrent comparison metrics for a list of tickers, sorted by market cap desc."""
        clean: list[str] = []
        seen: set[str] = set()
        for t in tickers or []:
            s = str(t or "").strip().upper()
            if s and s not in seen:
                seen.add(s)
                clean.append(s)
        if not clean:
            return []

        def _row(sym: str) -> dict:
            snap = self.get_price_snapshot(sym)
            return {
                "ticker": sym,
                "price": snap.get("current_price"),
                "pe_ratio": snap.get("pe_ratio"),
                "market_cap": snap.get("market_cap"),
                "day_change_pct": snap.get("day_change_pct"),
                "volume": snap.get("volume"),
                "beta": snap.get("beta"),
                "error": snap.get("error"),
            }

        with ThreadPoolExecutor(max_workers=min(8, len(clean))) as pool:
            rows = list(pool.map(_row, clean))

        rows.sort(key=lambda r: (r.get("market_cap") or 0.0), reverse=True)
        return rows
