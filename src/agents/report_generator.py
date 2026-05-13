"""Report orchestrator: collects live data and dispatches to ``report_mcp`` tools."""

from __future__ import annotations

import asyncio
import logging

from agno.agent import Agent

from src.agents.news_agent import FinancialNewsAgent
from src.agents.risk_analysis import RiskAnalysisAgent
from src.llm.base import LLMClient
from src.mcp import report_mcp, web_search_mcp, yfinance_mcp

logger = logging.getLogger(__name__)

INSTRUCTIONS = (
    "You are a report generator. Use the report tools to render Markdown or PDF "
    "artefacts from structured data. Always include the disclaimer section and "
    "never fabricate numbers — render only what was supplied."
)


class ReportGeneratorAgent:
    """Glue layer: gathers data via the existing analytical agents + MCP, then renders via ``report_mcp``."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._report = report_mcp
        self._yf = yfinance_mcp
        self._search = web_search_mcp
        self._agno: Agent | None = None

    # ---------- Agno surface ----------
    def as_agno_agent(self) -> Agent:
        """Lazily build an Agno agent that can call the report tools directly."""
        if self._agno is None:
            try:
                from src.llm.agno_model import make_agno_model

                self._agno = Agent(
                    name="report_generator",
                    model=make_agno_model(),
                    tools=[self._report, self._yf],
                    instructions=INSTRUCTIONS,
                )
            except Exception as e:
                logger.warning("Agno agent construction failed for report_generator: %s", e)
                raise
        return self._agno

    # ---------- dispatch ----------
    async def run(
        self,
        report_type: str,
        user: dict,
        tickers: list[str] | None = None,
        format: str = "markdown",
    ) -> dict:
        rt = (report_type or "").strip().lower()
        if rt == "portfolio":
            return await self.generate_portfolio_report(user, format)
        if rt == "market":
            return await self.generate_market_report(tickers or [], format)
        if rt == "risk":
            return await self.generate_risk_report(user, format)
        return {
            "error": f"unknown report_type '{report_type}'",
            "supported": ["portfolio", "market", "risk"],
        }

    # ---------- portfolio ----------
    async def generate_portfolio_report(self, user: dict, format: str) -> dict:
        positions = list(user.get("positions") or [])
        currency = str(user.get("base_currency") or "USD").upper()

        priced_positions, current_value, total_cost = await self._price_positions(
            positions, currency
        )
        total_return_pct = (
            ((current_value / total_cost) - 1.0) * 100.0 if total_cost > 0 else 0.0
        )

        top_weight = 0.0
        if current_value > 0 and priced_positions:
            top_weight = max(p["current_value"] for p in priced_positions) / current_value * 100.0

        flag = "low"
        if top_weight > 40:
            flag = "high"
        elif top_weight >= 25:
            flag = "medium"

        portfolio_data = {
            "base_currency": currency,
            "current_value": round(current_value, 2),
            "total_return_pct": round(total_return_pct, 2),
            "benchmark": (user.get("preferences") or {}).get("preferred_benchmark") or "S&P 500",
            "concentration_risk": {
                "flag": flag,
                "top_holding_weight_pct": round(top_weight, 2),
            },
            "positions": priced_positions,
            "observations": [
                {"severity": "info", "text": f"Snapshot taken across {len(priced_positions)} live positions."}
            ],
        }

        return await asyncio.to_thread(
            self._report.generate_portfolio_report,
            user.get("name") or "Investor",
            portfolio_data,
            format,
        )

    async def _price_positions(
        self,
        positions: list[dict],
        currency: str,
    ) -> tuple[list[dict], float, float]:
        if not positions:
            return [], 0.0, 0.0

        async def one(pos: dict) -> dict:
            ticker = str(pos.get("ticker") or "").upper()
            qty = float(pos.get("quantity") or 0.0)
            avg = float(pos.get("avg_cost") or 0.0)
            try:
                snap = await asyncio.to_thread(self._yf.get_price_snapshot, ticker)
                price = snap.get("current_price") if isinstance(snap, dict) else None
            except Exception as e:
                logger.warning("report portfolio price fetch failed for %s: %s", ticker, e)
                price = None
            price = float(price) if price is not None else avg
            return {
                "ticker": ticker,
                "quantity": qty,
                "avg_cost": avg,
                "currency": pos.get("currency") or currency,
                "current_price": price,
                "current_value": qty * price,
                "cost_basis": qty * avg,
            }

        priced = await asyncio.gather(*(one(p) for p in positions))
        cv = float(sum(p["current_value"] for p in priced))
        tc = float(sum(p["cost_basis"] for p in priced))
        return list(priced), cv, tc

    # ---------- market ----------
    async def generate_market_report(self, tickers: list[str], format: str) -> dict:
        clean = []
        seen: set[str] = set()
        for t in tickers or []:
            s = str(t or "").strip().upper()
            if s and s not in seen:
                seen.add(s)
                clean.append(s)

        if not clean:
            return {
                "format": format,
                "error": "no tickers supplied for market report",
            }

        snap_tasks = [asyncio.to_thread(self._yf.get_price_snapshot, t) for t in clean]
        news_tasks = [
            asyncio.to_thread(self._search.search_company_news, t, t, 7) for t in clean
        ]
        snapshots, news_lists = await asyncio.gather(
            asyncio.gather(*snap_tasks, return_exceptions=True),
            asyncio.gather(*news_tasks, return_exceptions=True),
        )

        snaps = [s for s in snapshots if isinstance(s, dict) and not s.get("error")]
        news_flat: list[dict] = []
        for nl in news_lists:
            if isinstance(nl, list):
                news_flat.extend([n for n in nl if isinstance(n, dict)])

        return await asyncio.to_thread(
            self._report.generate_market_report,
            clean,
            snaps,
            news_flat[:20],
            format,
        )

    # ---------- risk ----------
    async def generate_risk_report(self, user: dict, format: str) -> dict:
        risk_metrics = await RiskAnalysisAgent(self._llm).run(user)
        risk_metrics.setdefault("risk_profile", user.get("risk_profile"))
        risk_metrics.setdefault(
            "time_horizon", (user.get("preferences") or {}).get("time_horizon")
        )
        risk_metrics.setdefault(
            "recommendations",
            [
                "Re-test the portfolio after any deposit larger than 10% of current value.",
                "Watch the largest correlation pair — concentrated bets can move together in stress.",
            ],
        )
        return await asyncio.to_thread(
            self._report.generate_risk_report, risk_metrics, format
        )

    # Convenience: news report builder reused by orchestration layers.
    async def collect_news(self, tickers: list[str], topics: list[str], user: dict) -> dict:
        return await FinancialNewsAgent(self._llm).run(tickers, topics, user)


async def run(
    report_type: str,
    user: dict,
    llm: LLMClient,
    tickers: list[str] | None = None,
    format: str = "markdown",
) -> dict:
    return await ReportGeneratorAgent(llm).run(report_type, user, tickers, format)
