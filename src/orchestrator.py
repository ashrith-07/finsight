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
            from agno.models.openai import OpenAIChat

            self._team = Team(
                name="valura_team",
                model=OpenAIChat(id="gpt-4o-mini"),
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
            from agno.models.openai import OpenAIChat

            self._meta_agent = Agent(
                name="valura_orchestrator",
                model=OpenAIChat(id="gpt-4o-mini"),
                tools=[],
                instructions=(
                    "You are the Valura orchestrator meta-agent. Runtime routing is handled in "
                    "Python; this object exists for ecosystem tooling compatibility only."
                ),
            )
        return self._meta_agent

    async def run(self, classifier_result: ClassifierResult, user: dict) -> AgentResponse:
        self._ensure_team()
        self._ensure_meta_agent()

        intent = classifier_result.intent or ""
        agent = classifier_result.agent or "general_query"
        entities = classifier_result.entities

        if agent == "report_generator" or self._wants_report(intent):
            payload = await self._run_report_ecosystem(entities, user, intent)
            return self._build_agent_response(
                "report_generator",
                intent,
                entities,
                payload,
                message=self._message_for_report(payload),
            )

        if agent == "financial_news":
            payload = await self._run_news_ecosystem(entities, user)
            return self._build_agent_response(
                "financial_news",
                intent,
                entities,
                {"market_news": payload},
                message=self._message_for_news(payload),
            )

        if agent == "portfolio_health":
            eco = await self._run_portfolio_ecosystem(user, entities)
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
            eco = await self._run_market_research_ecosystem(tickers, intent, user)
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
            eco = await self._run_risk_assessment_ecosystem(user, entities)
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

    async def _run_portfolio_ecosystem(self, user: dict, entities: Entity) -> dict[str, Any]:
        positions = user.get("positions") or []
        tickers = [str(p["ticker"]).upper() for p in positions if p.get("ticker")]

        portfolio_res, risk_res, news_res = await asyncio.gather(
            self._safe_run("portfolio_health", self._portfolio.run(user)),
            self._safe_run("risk_analysis", self._risk.run(user)),
            self._safe_run(
                "news_agent",
                self._news.run(tickers=tickers, topics=["portfolio", "market"], user=user),
            ),
        )

        out: dict[str, Any] = {"portfolio": portfolio_res, "risk": risk_res, "news": news_res}
        if portfolio_res is None:
            out["error"] = "Portfolio analysis failed."
        return out

    async def _run_market_research_ecosystem(
        self, tickers: list[str], intent: str, user: dict
    ) -> dict[str, Any]:
        market_res, news_res = await asyncio.gather(
            self._safe_run("market_research", self._market.run(tickers=tickers, intent=intent)),
            self._safe_run(
                "news_agent",
                self._news.run(tickers=tickers, topics=[], user=user),
            ),
        )
        return {"market": market_res, "news": news_res}

    async def _run_risk_ecosystem(self, user: dict) -> dict[str, Any]:
        positions = user.get("positions") or []
        tickers = [str(p["ticker"]).upper() for p in positions if p.get("ticker")]
        risk_res, news_res = await asyncio.gather(
            self._safe_run("risk_analysis", self._risk.run(user)),
            self._safe_run(
                "news_agent",
                self._news.run(
                    tickers=tickers,
                    topics=["market risk", "volatility"],
                    user=user,
                ),
            ),
        )
        return {"risk": risk_res, "news": news_res}

    async def _run_risk_assessment_ecosystem(self, user: dict, entities: Entity) -> dict[str, Any]:
        """Risk primary path plus ``portfolio_health`` (see ``PARALLEL_PAIRS``) and news."""
        positions = user.get("positions") or []
        tickers = list(entities.tickers or []) or [
            str(p["ticker"]).upper() for p in positions if p.get("ticker")
        ]

        portfolio_res, risk_res, news_res = await asyncio.gather(
            self._safe_run("portfolio_health", self._portfolio.run(user)),
            self._safe_run("risk_analysis", self._risk.run(user)),
            self._safe_run(
                "news_agent",
                self._news.run(
                    tickers=tickers,
                    topics=["market risk", "volatility"],
                    user=user,
                ),
            ),
        )
        return {"portfolio": portfolio_res, "risk": risk_res, "news": news_res}

    async def _run_news_ecosystem(self, entities: Entity, user: dict) -> dict[str, Any]:
        tickers = list(entities.tickers or [])
        topics = list(entities.topics or []) or ["markets"]
        res = await self._safe_run(
            "news_agent",
            self._news.run(tickers=tickers, topics=topics, user=user),
        )
        return res or {}

    async def _run_report_ecosystem(self, entities: Entity, user: dict, intent: str) -> dict[str, Any]:
        report_type = self._infer_report_type(intent, entities)
        tickers = list(entities.tickers or [])
        fmt = "markdown"

        pre: dict[str, Any] = {}
        if report_type == "portfolio":
            pre = await self._run_portfolio_ecosystem(user, entities)
        elif report_type == "market" and tickers:
            pre = await self._run_market_research_ecosystem(tickers, intent, user)
        elif report_type == "risk":
            pre = await self._run_risk_ecosystem(user)

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

        articles = (news_result or {}).get("articles") or []
        head = articles[0].get("title") if articles else ""
        snap_obs = (market_dict.get("observations") or [{}])[0]
        snap_txt = snap_obs.get("text") if isinstance(snap_obs, dict) else getattr(snap_obs, "text", "")
        summ = (
            f"{snap_txt} "
            f"Parallel news scan returned {len(articles)} articles for context on: {intent}. "
            f"{'Top headline: ' + head if head else 'No headlines matched filters.'}"
        )
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

        obs_ph = (ph.get("observations") or [{}])[0]
        top_p = obs_ph.get("text") if isinstance(obs_ph, dict) else getattr(obs_ph, "text", "")
        stress = risk.get("stress_tests") or []
        worst = min(stress, key=lambda s: s.get("portfolio_loss_pct", 0)) if stress else {}
        arts = (news.get("articles") or [])[:1]
        sent = arts[0].get("sentiment", "neutral") if arts else "neutral"

        summary = (
            f"Portfolio lens: {top_p or 'No headline observation.'} "
            f"Worst modeled shock in the suite: {worst.get('scenario', 'n/a')} "
            f"({worst.get('portfolio_loss_pct', 'n/a')}%). "
            f"Latest macro/news sentiment skew: {sent}."
        )
        return {
            "portfolio_health": ph,
            "risk_analysis": risk,
            "market_news": news,
            "ecosystem_summary": summary.strip(),
        }

    def _ecosystem_summary_portfolio(
        self,
        portfolio: dict[str, Any],
        risk: dict[str, Any],
        news: dict[str, Any],
    ) -> str:
        obs = portfolio.get("observations") or []
        top = ""
        if obs:
            o0 = obs[0]
            top = o0.get("text") if isinstance(o0, dict) else getattr(o0, "text", "")
        stress = risk.get("stress_tests") or []
        worst = min(stress, key=lambda s: s.get("portfolio_loss_pct", 0)) if stress else {}
        arts = news.get("articles") or []
        sent = arts[0].get("sentiment", "neutral") if arts else "neutral"
        hl = arts[0].get("title") if arts else ""

        return (
            f"{top or 'Portfolio analytics completed.'} "
            f"In a severe stress scenario ({worst.get('scenario', 'historical shock')}) "
            f"the modeled portfolio draw approached {worst.get('portfolio_loss_pct', 'n/a')}%. "
            f"Parallel news sentiment on the top story is {sent}"
            f"{f' ({hl[:80]}…)' if hl else ''}."
        )

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

    def _infer_report_type(self, intent: str, entities: Entity) -> str:
        low = (intent or "").lower()
        tickers = list(entities.tickers or [])
        if "risk" in low or "var" in low:
            return "risk"
        if self._REPORT_MARKET_RX.search(low) or (tickers and "portfolio" not in low and "report" in low):
            return "market"
        return "portfolio"

    def _message_for_portfolio(self, combined: dict[str, Any]) -> str:
        ph = combined.get("portfolio_health") or {}
        obs = ph.get("observations") or []
        if obs:
            o0 = obs[0]
            t = o0.get("text") if isinstance(o0, dict) else str(o0)
            return f"Ran portfolio, risk, and news in parallel — primary read: {t[:160]}"
        return "Ran portfolio, risk, and news pipelines and merged the ecosystem output."

    def _message_for_market(self, combined: dict[str, Any]) -> str:
        mr = combined.get("market_research") or {}
        snaps = mr.get("snapshots") or []
        if snaps:
            s0 = snaps[0]
            return (
                f"Merged live quotes with parallel news — lead tape: {s0.get('ticker')} "
                f"{s0.get('day_change_pct')}% on the session."
            )
        return "Merged market research with parallel news scans."

    def _message_for_risk_ecosystem(self, combined: dict[str, Any]) -> str:
        r = combined.get("risk_analysis") or {}
        var95 = (r.get("var") or {}).get("one_day_95")
        return (
            f"Combined portfolio, VaR, stress suite, and macro news — 1-day 95% VaR ≈ {var95}."
            if var95 is not None
            else "Combined portfolio context with quantitative risk and news overlays."
        )

    def _message_for_news(self, payload: dict[str, Any]) -> str:
        n = payload.get("total_results") or len(payload.get("articles") or [])
        return f"News enrichment returned {n} deduplicated articles with sentiment tags."

    def _message_for_report(self, payload: dict[str, Any]) -> str:
        rp = payload.get("report") or {}
        fn = rp.get("filename") or "report"
        return f"Generated {payload.get('report_type', 'portfolio')} report artefact ({fn})."


__all__ = ["ValuraOrchestrator"]
