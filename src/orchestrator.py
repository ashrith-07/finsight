"""Central coordinator: parallel agent execution, synthesis, fallback to ``AgentRouter``."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable
from typing import Any

from agno.agent import Agent
from agno.team import Team

from src.agents import (
    FinancialNewsAgent,
    MarketResearchAgent,
    PortfolioHealthAgent,
    ReportGeneratorAgent,
    RiskAnalysisAgent,
)
from src.agents.stub import StubAgent
from src.llm.base import LLMClient
from src.models import AgentResponse, ClassifierResult, Entity, PortfolioHealthResult

logger = logging.getLogger("valura.orchestrator")


class ValuraOrchestrator:
    """Coordinates portfolio, market, risk, news, and report flows; delegates unknown agents to ``StubAgent``."""

    PARALLEL_PAIRS = {
        "market_research": ["news_agent"],
        "portfolio_health": ["risk_analysis", "news_agent"],
        "risk_assessment": ["portfolio_health"],
    }

    POST_AGENTS = ["report_generator"]

    _REPORT_MARKET_RX = re.compile(
        r"\b(market\s+research|stock\s+report|equity\s+report|ticker|compare)\b",
        re.I,
    )

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._portfolio = PortfolioHealthAgent(llm)
        self._market = MarketResearchAgent(llm)
        self._risk = RiskAnalysisAgent(llm)
        self._news = FinancialNewsAgent(llm)
        self._report = ReportGeneratorAgent(llm)
        self._stub = StubAgent()
        self._team: Team | None = None
        self._meta_agent: Agent | None = None

    def _ensure_team(self) -> Team:
        if self._team is None:
            from src.llm.agno_model import make_agno_model

            self._team = Team(
                name="valura_team",
                model=make_agno_model(),
                members=[
                    self._risk.as_agno_agent(),
                    self._news.as_agno_agent(),
                    self._report.as_agno_agent(),
                ],
                instructions=(
                    "You coordinate risk analytics, live news, and report rendering for Valura. "
                    "Delegate specialised work to members; never contradict numerical outputs "
                    "computed by tools."
                ),
            )
        return self._team

    def _ensure_meta_agent(self) -> Agent:
        """Thin Agno shell used as the orchestrator's framework anchor (dispatch stays imperative)."""
        if self._meta_agent is None:
            from src.llm.agno_model import make_agno_model

            self._meta_agent = Agent(
                name="valura_orchestrator",
                model=make_agno_model(),
                tools=[],
                instructions=(
                    "You are the Valura orchestrator meta-agent. Runtime routing is handled in "
                    "Python; this object exists for ecosystem tooling compatibility only."
                ),
            )
        return self._meta_agent

    async def run(
        self,
        classifier_result: ClassifierResult,
        user: dict,
        query: str = "",
    ) -> AgentResponse:
        self._ensure_team()
        self._ensure_meta_agent()

        intent = classifier_result.intent or ""
        agent = classifier_result.agent or "general_query"
        entities = classifier_result.entities

        if agent == "report_generator" or self._wants_report(intent):
            payload = await self._run_report_ecosystem(entities, user, intent, query)
            return self._build_agent_response(
                "report_generator",
                intent,
                entities,
                payload,
                message=self._message_for_report(payload),
            )

        if agent == "financial_news":
            payload = await self._run_news_ecosystem(entities, user, intent, query)
            return self._build_agent_response(
                "financial_news",
                intent,
                entities,
                {"market_news": payload},
                message=self._message_for_news(payload),
            )

        if agent == "portfolio_health":
            eco = await self._run_portfolio_ecosystem(user, entities, intent, query)
            if eco.get("portfolio") is None:
                return self._build_agent_response(
                    agent,
                    intent,
                    entities,
                    {"error": eco.get("error") or "portfolio analysis failed"},
                    message="Portfolio analysis failed; partial ecosystem data may still be attached.",
                )
            combined = self._synthesise_portfolio_result(
                eco.get("portfolio"),
                eco.get("risk") or {},
                eco.get("news") or {},
            )
            return self._build_agent_response(
                agent,
                intent,
                entities,
                combined,
                message=self._message_for_portfolio(combined),
            )

        if agent == "market_research":
            tickers = list(entities.tickers or [])
            eco = await self._run_market_research_ecosystem(tickers, intent, user, query)
            combined = self._synthesise_market_result(
                eco.get("market"),
                eco.get("news") or {},
                intent,
            )
            return self._build_agent_response(
                agent,
                intent,
                entities,
                combined,
                message=self._message_for_market(combined),
            )

        if agent == "risk_assessment":
            eco = await self._run_risk_assessment_ecosystem(user, entities, intent, query)
            combined = self._synthesise_risk_assessment_result(eco)
            return self._build_agent_response(
                agent,
                intent,
                entities,
                combined,
                message=self._message_for_risk_ecosystem(combined),
            )

        return await self._stub.run(agent, intent, entities)

    # --- ecosystems ---------------------------------------------------------

    async def _run_portfolio_ecosystem(
        self, user: dict, entities: Entity, intent: str = "", query: str = "",
    ) -> dict[str, Any]:
        positions = user.get("positions") or []
        tickers = [str(p["ticker"]).upper() for p in positions if p.get("ticker")]

        portfolio_res, risk_res, news_res = await asyncio.gather(
            self._safe_run("portfolio_health", self._portfolio.run(user, intent=intent, query=query)),
            self._safe_run("risk_analysis", self._risk.run(user, intent=intent, query=query)),
            self._safe_run(
                "news_agent",
                self._news.run(
                    tickers=tickers, topics=["portfolio", "market"], user=user,
                    intent=intent, query=query,
                ),
            ),
        )

        out: dict[str, Any] = {"portfolio": portfolio_res, "risk": risk_res, "news": news_res}
        if portfolio_res is None:
            out["error"] = "Portfolio analysis failed."
        return out

    async def _run_market_research_ecosystem(
        self, tickers: list[str], intent: str, user: dict, query: str = "",
    ) -> dict[str, Any]:
        market_res, news_res = await asyncio.gather(
            self._safe_run(
                "market_research",
                self._market.run(tickers=tickers, intent=intent, query=query),
            ),
            self._safe_run(
                "news_agent",
                self._news.run(
                    tickers=tickers, topics=[], user=user, intent=intent, query=query,
                ),
            ),
        )
        return {"market": market_res, "news": news_res}

    async def _run_risk_ecosystem(
        self, user: dict, intent: str = "", query: str = "",
    ) -> dict[str, Any]:
        positions = user.get("positions") or []
        tickers = [str(p["ticker"]).upper() for p in positions if p.get("ticker")]
        risk_res, news_res = await asyncio.gather(
            self._safe_run("risk_analysis", self._risk.run(user, intent=intent, query=query)),
            self._safe_run(
                "news_agent",
                self._news.run(
                    tickers=tickers, topics=["market risk", "volatility"],
                    user=user, intent=intent, query=query,
                ),
            ),
        )
        return {"risk": risk_res, "news": news_res}

    async def _run_risk_assessment_ecosystem(
        self, user: dict, entities: Entity, intent: str = "", query: str = "",
    ) -> dict[str, Any]:
        """Risk primary path plus ``portfolio_health`` (see ``PARALLEL_PAIRS``) and news."""
        positions = user.get("positions") or []
        tickers = list(entities.tickers or []) or [
            str(p["ticker"]).upper() for p in positions if p.get("ticker")
        ]

        portfolio_res, risk_res, news_res = await asyncio.gather(
            self._safe_run("portfolio_health", self._portfolio.run(user, intent=intent, query=query)),
            self._safe_run("risk_analysis", self._risk.run(user, intent=intent, query=query)),
            self._safe_run(
                "news_agent",
                self._news.run(
                    tickers=tickers, topics=["market risk", "volatility"],
                    user=user, intent=intent, query=query,
                ),
            ),
        )
        return {"portfolio": portfolio_res, "risk": risk_res, "news": news_res}

    async def _run_news_ecosystem(
        self, entities: Entity, user: dict, intent: str = "", query: str = "",
    ) -> dict[str, Any]:
        tickers = list(entities.tickers or [])
        topics = list(entities.topics or []) or ["markets"]
        res = await self._safe_run(
            "news_agent",
            self._news.run(
                tickers=tickers, topics=topics, user=user, intent=intent, query=query,
            ),
        )
        return res or {}

    async def _run_report_ecosystem(
        self, entities: Entity, user: dict, intent: str, query: str = "",
    ) -> dict[str, Any]:
        report_type = self._infer_report_type(intent, entities, query)
        tickers = list(entities.tickers or [])
        fmt = "markdown"

        pre: dict[str, Any] = {}
        if report_type == "portfolio":
            pre = await self._run_portfolio_ecosystem(user, entities, intent, query)
        elif report_type == "market" and tickers:
            pre = await self._run_market_research_ecosystem(tickers, intent, user, query)
        elif report_type == "risk":
            pre = await self._run_risk_ecosystem(user, intent, query)
        elif report_type == "comparison":
            pre = await self._run_market_research_ecosystem(tickers, intent, user, query)

        report_payload = await self._safe_run(
            "report_generator",
            self._report.run(report_type, user, tickers or None, fmt),
        )

        out: dict[str, Any] = {"report_type": report_type, "report": report_payload or {}}
        if pre:
            out["prefetch"] = pre
        return out

    # --- synthesis ----------------------------------------------------------

    def _synthesise_portfolio_result(
        self,
        portfolio_result: PortfolioHealthResult | dict | None,
        risk_result: dict[str, Any],
        news_result: dict[str, Any],
    ) -> dict[str, Any]:
        ph: dict[str, Any]
        if portfolio_result is None:
            ph = {}
        elif isinstance(portfolio_result, PortfolioHealthResult):
            ph = portfolio_result.model_dump()
        else:
            ph = dict(portfolio_result)

        summary = self._ecosystem_summary_portfolio(ph, risk_result, news_result)
        return {
            "portfolio_health": ph,
            "risk_analysis": risk_result,
            "market_news": news_result,
            "ecosystem_summary": summary,
        }

    def _synthesise_market_result(
        self,
        market_result: Any,
        news_result: dict[str, Any],
        intent: str,
    ) -> dict[str, Any]:
        market_dict: dict[str, Any]
        if market_result is None:
            market_dict = {}
        elif hasattr(market_result, "model_dump"):
            market_dict = market_result.model_dump()
        elif isinstance(market_result, dict):
            market_dict = market_result
        else:
            market_dict = {"raw": str(market_result)}

        snaps = market_dict.get("snapshots") or []
        parts: list[str] = []
        if snaps:
            quotes = ", ".join(
                f"{s.get('ticker')} {s.get('currency', 'USD')} {s.get('current_price'):.2f} "
                f"({s.get('day_change_pct'):+.2f}%)"
                for s in snaps[:3]
                if isinstance(s.get('current_price'), (int, float))
            )
            if quotes:
                parts.append(f"Live tape — {quotes}.")

        comp = market_dict.get("comparison_note")
        if comp:
            parts.append(str(comp).strip())

        articles = (news_result or {}).get("articles") or []
        if articles:
            head = articles[0].get("title") or ""
            parts.append(
                f"Parallel news scan: {len(articles)} articles"
                f"{f' — top headline “{head[:80]}…”' if head else ''}."
            )
        elif snaps:
            parts.append("Parallel news scan returned no headlines (likely rate-limit).")

        summ = " ".join(parts) or f"Market research completed for: {intent}."
        return {
            "market_research": market_dict,
            "market_news": news_result,
            "ecosystem_summary": summ.strip(),
        }

    def _synthesise_risk_assessment_result(self, eco: dict[str, Any]) -> dict[str, Any]:
        portfolio = eco.get("portfolio")
        risk = eco.get("risk") or {}
        news = eco.get("news") or {}

        ph: dict[str, Any]
        if portfolio is None:
            ph = {}
        elif hasattr(portfolio, "model_dump"):
            ph = portfolio.model_dump()
        else:
            ph = dict(portfolio)

        parts: list[str] = []
        var = risk.get("var") or {}
        var95 = var.get("one_day_95")
        var99 = var.get("one_day_99")
        if isinstance(var95, (int, float)) and isinstance(var99, (int, float)):
            parts.append(
                f"1-day VaR: ${var95:,.0f} at 95% / ${var99:,.0f} at 99% confidence."
            )

        sharpe = risk.get("sharpe_ratio_annualised")
        max_dd = risk.get("max_drawdown_pct")
        if isinstance(sharpe, (int, float)) and isinstance(max_dd, (int, float)):
            parts.append(f"Sharpe ≈ {sharpe:.2f}; trailing max drawdown {max_dd:+.2f}%.")

        stress = risk.get("stress_tests") or []
        worst = min(stress, key=lambda s: s.get("portfolio_loss_pct", 0)) if stress else {}
        if worst:
            parts.append(
                f"Worst modeled shock — {worst.get('scenario', 'n/a')}: "
                f"{worst.get('portfolio_loss_pct', 'n/a')}% drawdown."
            )

        arts = news.get("articles") or []
        if arts:
            counts = news.get("sentiment_counts") or {}
            parts.append(
                f"Macro news overlay: {len(arts)} articles "
                f"(pos {counts.get('positive', 0)} / neu {counts.get('neutral', 0)} / "
                f"neg {counts.get('negative', 0)})."
            )

        return {
            "portfolio_health": ph,
            "risk_analysis": risk,
            "market_news": news,
            "ecosystem_summary": " ".join(parts) or "Risk analysis completed.",
        }

    def _ecosystem_summary_portfolio(
        self,
        portfolio: dict[str, Any],
        risk: dict[str, Any],
        news: dict[str, Any],
    ) -> str:
        """Build a numeric, non-redundant summary so it doesn't repeat what observations say."""
        perf = portfolio.get("performance") or {}
        conc = portfolio.get("concentration_risk") or {}
        bench = portfolio.get("benchmark_comparison") or {}

        parts: list[str] = []

        cv = perf.get("current_value_total")
        tr = perf.get("total_return_pct")
        bm = bench.get("benchmark_return_pct")
        bm_name = bench.get("benchmark") or "benchmark"
        if isinstance(cv, (int, float)) and cv > 0:
            tr_str = f"{tr:+.2f}%" if isinstance(tr, (int, float)) else "n/a"
            bm_str = f"{bm:+.2f}%" if isinstance(bm, (int, float)) else "n/a"
            parts.append(
                f"Portfolio value sits at ${cv:,.0f} (return {tr_str} vs {bm_name} {bm_str})."
            )

        flag = (conc.get("flag") or "").lower()
        top = conc.get("top_position_pct")
        if flag in {"medium", "high"} and isinstance(top, (int, float)):
            parts.append(f"Concentration is {flag}: top holding ≈ {top:.1f}% of book.")

        stress = risk.get("stress_tests") or []
        worst = min(stress, key=lambda s: s.get("portfolio_loss_pct", 0)) if stress else {}
        if worst:
            parts.append(
                f"Worst modeled stress scenario ({worst.get('scenario', 'historical shock')}): "
                f"{worst.get('portfolio_loss_pct', 'n/a')}% drawdown."
            )

        arts = news.get("articles") or []
        if arts:
            counts = news.get("sentiment_counts") or {}
            n_total = news.get("total_results") or len(arts)
            sent_summary = (
                f"pos {counts.get('positive', 0)} / neu {counts.get('neutral', 0)} / "
                f"neg {counts.get('negative', 0)}"
            )
            parts.append(f"News overlay: {n_total} articles ({sent_summary}).")

        return " ".join(parts) or "Portfolio analytics completed."

    def _build_agent_response(
        self,
        agent: str,
        intent: str,
        entities: Entity,
        result: dict[str, Any],
        *,
        implemented: bool = True,
        message: str | None = None,
    ) -> AgentResponse:
        return AgentResponse(
            agent=agent,
            implemented=implemented,
            intent=intent,
            entities=entities,
            result=result,
            message=message or f"Completed {agent} orchestration.",
        )

    async def _safe_run(self, name: str, coro: Awaitable[Any]) -> Any:
        start = time.perf_counter()
        try:
            result = await coro
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info("agent=%s status=success duration=%.0fms", name, duration_ms)
            return result
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "agent=%s status=error duration=%.0fms error=%s",
                name,
                duration_ms,
                e,
            )
            return None

    # --- helpers ------------------------------------------------------------

    def _wants_report(self, intent: str) -> bool:
        low = (intent or "").lower()
        return "report" in low or "pdf" in low or "markdown export" in low

    def _infer_report_type(self, intent: str, entities: Entity, query: str = "") -> str:
        low = f"{(intent or '').lower()} {(query or '').lower()}"
        tickers = list(entities.tickers or [])
        if ("compar" in low or " vs " in f" {low} " or "versus" in low) and len(tickers) >= 2:
            return "comparison"
        if "risk" in low or "var" in low or "stress" in low:
            return "risk"
        if self._REPORT_MARKET_RX.search(low) or (tickers and "portfolio" not in low and "report" in low):
            return "market"
        return "portfolio"

    def _message_for_portfolio(self, combined: dict[str, Any]) -> str:
        return str(combined.get("ecosystem_summary") or "").strip() or (
            "Ran portfolio, risk, and news pipelines and merged the ecosystem output."
        )

    def _message_for_market(self, combined: dict[str, Any]) -> str:
        return str(combined.get("ecosystem_summary") or "").strip() or (
            "Merged market research with parallel news scans."
        )

    def _message_for_risk_ecosystem(self, combined: dict[str, Any]) -> str:
        return str(combined.get("ecosystem_summary") or "").strip() or (
            "Combined portfolio context with quantitative risk and news overlays."
        )

    def _message_for_news(self, payload: dict[str, Any]) -> str:
        n = payload.get("total_results") or len(payload.get("articles") or [])
        summary = str(payload.get("summary") or "").strip()
        head = (
            f" Top: “{(payload.get('articles') or [{}])[0].get('title', '')[:80]}…”."
            if (payload.get("articles") or [])
            else ""
        )
        if summary:
            return f"{summary}{head}"
        return f"News enrichment returned {n} deduplicated articles with sentiment tags.{head}"

    def _message_for_report(self, payload: dict[str, Any]) -> str:
        rp = payload.get("report") or {}
        fn = rp.get("filename") or "report"
        path = rp.get("file_path") or ""
        rt = payload.get("report_type", "portfolio")
        # Surface the first non-blank section of the markdown so the chat shows real content.
        preview = ""
        content = rp.get("content")
        if isinstance(content, str) and content.strip():
            for line in content.splitlines():
                if line.strip() and not line.startswith("#"):
                    preview = line.strip()
                    break
        loc = f" Saved to {path}." if path else ""
        head = f"Generated {rt} report ({fn})."
        return f"{head}{loc}{(' ' + preview) if preview else ''}"


__all__ = ["ValuraOrchestrator"]
