"""yfinance-backed MCP toolkit. All tools return JSON-safe dicts; failures degrade to ``{"error": ...}``."""

from __future__ import annotations

import yfinance as yf
from agno.tools import Toolkit

from src.logging_config import get_logger
from src.ticker_sanitize import normalize_yfinance_ticker
from src.yfinance_throttle import yfinance_pause

logger = get_logger("mcp.yfinance")

# LLM tool payloads must stay small (Groq context + TPM). Full-year dailies are huge.
_MAX_HISTORICAL_BARS = 56


def _invalid_ticker_payload(raw: object) -> dict:
    return {
        "ticker": str(raw or "").strip() or "?",
        "error": "invalid ticker — pass exchange symbols only (e.g. AAPL, MSFT), not names or labels",
    }


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


def _snapshot_from_fast_info(t: yf.Ticker, ticker: str) -> dict | None:
    """Lighter quote path — fewer full ``info`` / crumb issues on congested IPs."""
    try:
        fi = t.fast_info
        if fi is None:
            return None

        def pick(*keys: str):
            for k in keys:
                try:
                    v = fi[k]  # type: ignore[index]
                except Exception:
                    continue
                if v is not None:
                    return v
            return None

        current = _safe_float(pick("last_price", "regular_market_price"))
        if current is None:
            return None
        prev_close = _safe_float(pick("previous_close", "regular_market_previous_close"))
        day_change_pct = None
        if prev_close not in (None, 0):
            day_change_pct = round((current - prev_close) / prev_close * 100, 4)
        return {
            "ticker": str(ticker).upper(),
            "currency": str(pick("currency", "currency_symbol") or "USD").upper(),
            "current_price": current,
            "previous_close": prev_close,
            "day_change_pct": day_change_pct,
            "volume": _safe_int(pick("last_volume", "regular_market_volume")),
            "market_cap": _safe_float(pick("market_cap")),
            "fifty_two_week_high": _safe_float(pick("year_high", "fifty_two_week_high")),
            "fifty_two_week_low": _safe_float(pick("year_low", "fifty_two_week_low")),
            "pe_ratio": _safe_float(pick("pe_ratio", "trailing_pe")),
            "beta": _safe_float(pick("beta")),
            "dividend_yield": _safe_float(pick("dividend_yield")),
        }
    except Exception:
        return None


def _snapshot_from_history(t: yf.Ticker, ticker: str) -> dict | None:
    try:
        hist = t.history(period="5d", interval="1d")
        if hist is None or hist.empty:
            return None
        row = hist.iloc[-1]
        prv = hist.iloc[-2] if len(hist) > 1 else row
        cur = _safe_float(row.get("Close"))
        prev_close = _safe_float(prv.get("Close"))
        if cur is None:
            return None
        day_change_pct = None
        if prev_close not in (None, 0):
            day_change_pct = round((cur - prev_close) / prev_close * 100, 4)
        return {
            "ticker": str(ticker).upper(),
            "currency": "USD",
            "current_price": cur,
            "previous_close": prev_close,
            "day_change_pct": day_change_pct,
            "volume": _safe_int(row.get("Volume")),
            "market_cap": None,
            "fifty_two_week_high": None,
            "fifty_two_week_low": None,
            "pe_ratio": None,
            "beta": None,
            "dividend_yield": None,
        }
    except Exception:
        return None


def _build_snapshot_dict(ticker: str, info: dict) -> dict:
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
        sym = normalize_yfinance_ticker(ticker)
        if sym is None:
            return _invalid_ticker_payload(ticker)
        try:
            t = yf.Ticker(sym)
            yfinance_pause()
            snap = _snapshot_from_fast_info(t, sym)
            if snap is not None and snap.get("current_price") is not None:
                return snap

            yfinance_pause()
            info = t.info or {}
            if info and len(info) >= 5:
                return _build_snapshot_dict(sym, info)

            yfinance_pause()
            snap = _snapshot_from_history(t, sym)
            if snap is not None:
                return snap
            return {"ticker": sym, "error": "no data available"}
        except Exception as e:
            logger.warning("yfinance_mcp.get_price_snapshot(%s) failed: %s", sym, e)
            return {"ticker": sym, "error": str(e)}

    def get_historical_prices(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> dict:
        """OHLCV bars; ``period`` ∈ {1d,5d,1mo,3mo,6mo,1y,2y,5y}."""
        sym = normalize_yfinance_ticker(ticker)
        if sym is None:
            return {"ticker": str(ticker or ""), "period": period, "bars": [], "error": _invalid_ticker_payload(ticker)["error"]}
        yfinance_pause()
        try:
            hist = yf.Ticker(sym).history(period=period, interval=interval)
            if hist is None or hist.empty:
                return {"ticker": sym, "period": period, "bars": [], "error": "no history"}

            bars = []
            for idx, row in hist.iterrows():
                d = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
                bars.append(
                    {
                        "date": d,
                        "open": _safe_float(row.get("Open")),
                        "high": _safe_float(row.get("High")),
                        "low": _safe_float(row.get("Low")),
                        "close": _safe_float(row.get("Close")),
                        "volume": _safe_int(row.get("Volume")),
                    }
                )
            omitted = max(0, len(bars) - _MAX_HISTORICAL_BARS)
            if len(bars) > _MAX_HISTORICAL_BARS:
                bars = bars[-_MAX_HISTORICAL_BARS:]
            return {
                "ticker": sym,
                "period": period,
                "interval": interval,
                "bars": bars,
                "bars_omitted_older": omitted,
            }
        except Exception as e:
            logger.warning("yfinance_mcp.get_historical_prices(%s) failed: %s", sym, e)
            return {"ticker": sym, "period": period, "bars": [], "error": str(e)}

    def get_company_fundamentals(self, ticker: str) -> dict:
        """Company-level fundamentals + descriptive metadata."""
        sym = normalize_yfinance_ticker(ticker)
        if sym is None:
            return _invalid_ticker_payload(ticker)
        yfinance_pause()
        try:
            info = yf.Ticker(sym).info or {}
            if not info or len(info) < 5:
                return {"ticker": sym, "error": "no data available"}

            return {
                "ticker": sym,
                "name": str(info.get("longName") or info.get("shortName") or sym),
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
            logger.warning("yfinance_mcp.get_company_fundamentals(%s) failed: %s", sym, e)
            return {"ticker": sym, "error": str(e)}

    def get_financial_statements(self, ticker: str) -> dict:
        """Most-recent 4 quarters of revenue, net income, total assets, total debt."""
        sym = normalize_yfinance_ticker(ticker)
        if sym is None:
            return _invalid_ticker_payload(ticker)
        yfinance_pause()
        try:
            t = yf.Ticker(sym)
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
                "ticker": sym,
                "revenue": _row_last_n(financials, ["Total Revenue", "TotalRevenue"]),
                "net_income": _row_last_n(financials, ["Net Income", "NetIncome"]),
                "total_assets": _row_last_n(balance, ["Total Assets", "TotalAssets"]),
                "total_debt": _row_last_n(
                    balance, ["Total Debt", "TotalDebt", "Long Term Debt"]
                ),
            }
        except Exception as e:
            logger.warning("yfinance_mcp.get_financial_statements(%s) failed: %s", sym, e)
            return {"ticker": sym, "error": str(e)}

    def get_options_data(self, ticker: str) -> dict:
        """Nearest-expiry options chain summary; safe defaults when unsupported."""
        sym = normalize_yfinance_ticker(ticker)
        if sym is None:
            return _invalid_ticker_payload(ticker)
        yfinance_pause()
        try:
            t = yf.Ticker(sym)
            expiries = list(t.options or ())
            if not expiries:
                return {"ticker": sym, "error": "no options listed"}

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
                "ticker": sym,
                "nearest_expiry": nearest,
                "put_call_ratio": put_call_ratio,
                "implied_volatility_avg": iv_avg,
                "total_open_interest": int(call_oi + put_oi),
                "expiries_available": len(expiries),
            }
        except Exception as e:
            logger.warning("yfinance_mcp.get_options_data(%s) failed: %s", sym, e)
            return {"ticker": sym, "error": str(e)}

    def screen_stocks(self, tickers: list[str]) -> list[dict]:
        """Sequential comparison metrics — parallel Yahoo calls trigger 429 / crumb errors."""
        clean: list[str] = []
        seen: set[str] = set()
        for t in tickers or []:
            sym = normalize_yfinance_ticker(t)
            if sym and sym not in seen:
                seen.add(sym)
                clean.append(sym)
        if not clean:
            return []
        clean = clean[:24]

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

        rows = [_row(sym) for sym in clean]

        rows.sort(key=lambda r: (r.get("market_cap") or 0.0), reverse=True)
        return rows
