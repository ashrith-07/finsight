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
from src.models import (
    AgentResponse,
    ClassifierResult,
    Entity,
    ExecutionMetadata,
    PortfolioHealthResult,
)

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

    def _ensure_team(self) -> Team | None:
        if self._team is None:
            from src.llm.agno_model import get_agno_model_strong

            model = get_agno_model_strong()
            if model is None:
                return None
            try:
                self._team = Team(
                    name="valura_team",
                    model=model,
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
            except Exception as e:
                logger.warning("Agno Team initialisation skipped: %s", e)
                return None
        return self._team

    def _ensure_meta_agent(self) -> Agent | None:
        """Thin Agno shell for tooling compatibility; omitted when no LLM API key is configured."""
        if self._meta_agent is None:
            from src.llm.agno_model import get_agno_model

            model = get_agno_model()
            if model is None:
                return None
            self._meta_agent = Agent(
                name="valura_orchestrator",
                model=model,
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
            timings, wall = _pop_timings(payload)
            return self._build_agent_response(
                "report_generator",
                intent,
                entities,
                payload,
                message=self._message_for_report(payload),
                execution_metadata=self._build_exec_metadata(timings, wall),
            )

        if agent == "financial_news":
            eco = await self._run_news_ecosystem(entities, user, intent, query)
            timings, wall = _pop_timings(eco)
            payload = eco.get("payload") or {}
            return self._build_agent_response(
                "financial_news",
                intent,
                entities,
                {"market_news": payload},
                message=self._message_for_news(payload),
                execution_metadata=self._build_exec_metadata(timings, wall),
            )

        if agent == "portfolio_health":
            eco = await self._run_portfolio_ecosystem(user, entities, intent, query)
            timings, wall = _pop_timings(eco)
            if eco.get("portfolio") is None:
                return self._build_agent_response(
                    agent,
                    intent,
                    entities,
                    {"error": eco.get("error") or "portfolio analysis failed"},
                    message="Portfolio analysis failed; partial ecosystem data may still be attached.",
                    execution_metadata=self._build_exec_metadata(timings, wall),
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
                execution_metadata=self._build_exec_metadata(timings, wall),
            )

        if agent == "market_research":
            tickers = list(entities.tickers or [])
            eco = await self._run_market_research_ecosystem(tickers, intent, user, query)
            timings, wall = _pop_timings(eco)
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
                execution_metadata=self._build_exec_metadata(timings, wall),
            )

        if agent == "risk_assessment":
            eco = await self._run_risk_assessment_ecosystem(user, entities, intent, query)
            timings, wall = _pop_timings(eco)
            combined = self._synthesise_risk_assessment_result(eco)
            return self._build_agent_response(
                agent,
                intent,
                entities,
                combined,
                message=self._message_for_risk_ecosystem(combined),
                execution_metadata=self._build_exec_metadata(timings, wall),
            )

        # Stub path — single synthetic timing so the UI's exec_metadata
        # contract is honoured even for not-yet-implemented agents.
        stub_start = time.perf_counter()
        stub_resp = await self._stub.run(agent, intent, entities)
        stub_ms = round((time.perf_counter() - stub_start) * 1000)
        return stub_resp.model_copy(update={
            "execution_metadata": ExecutionMetadata(
                agents_ran=[agent or "general_query"],
                timings={agent or "general_query": stub_ms},
                parallel=False,
                wall_time_ms=stub_ms,
                sequential_time_ms=stub_ms,
                time_saved_ms=0,
            )
        })

    # --- ecosystems ---------------------------------------------------------

    async def _run_portfolio_ecosystem(
        self, user: dict, entities: Entity, intent: str = "", query: str = "",
    ) -> dict[str, Any]:
        positions = user.get("positions") or []
        tickers = [str(p["ticker"]).upper() for p in positions if p.get("ticker")]

        results, timings, wall = await self._run_parallel([
            ("portfolio_health", self._portfolio.run(user, intent=intent, query=query)),
            ("risk_analysis", self._risk.run(user, intent=intent, query=query)),
            ("news_agent", self._news.run(
                tickers=tickers, topics=["portfolio", "market"],
                user=user, intent=intent, query=query,
            )),
        ])

        out: dict[str, Any] = {
            "portfolio": results["portfolio_health"],
            "risk": results["risk_analysis"],
            "news": results["news_agent"],
            "_timings": timings,
            "_wall_ms": wall,
        }
        if results["portfolio_health"] is None:
            out["error"] = "Portfolio analysis failed."
        return out

    async def _run_market_research_ecosystem(
        self, tickers: list[str], intent: str, user: dict, query: str = "",
    ) -> dict[str, Any]:
        results, timings, wall = await self._run_parallel([
            ("market_research", self._market.run(tickers=tickers, intent=intent, query=query)),
            ("news_agent", self._news.run(
                tickers=tickers, topics=[], user=user, intent=intent, query=query,
            )),
        ])
        return {
            "market": results["market_research"],
            "news": results["news_agent"],
            "_timings": timings,
            "_wall_ms": wall,
        }

    async def _run_risk_ecosystem(
        self, user: dict, intent: str = "", query: str = "",
    ) -> dict[str, Any]:
        positions = user.get("positions") or []
        tickers = [str(p["ticker"]).upper() for p in positions if p.get("ticker")]
        results, timings, wall = await self._run_parallel([
            ("risk_analysis", self._risk.run(user, intent=intent, query=query)),
            ("news_agent", self._news.run(
                tickers=tickers, topics=["market risk", "volatility"],
                user=user, intent=intent, query=query,
            )),
        ])
        return {
            "risk": results["risk_analysis"],
            "news": results["news_agent"],
            "_timings": timings,
            "_wall_ms": wall,
        }

    async def _run_risk_assessment_ecosystem(
        self, user: dict, entities: Entity, intent: str = "", query: str = "",
    ) -> dict[str, Any]:
        """Risk primary path plus ``portfolio_health`` (see ``PARALLEL_PAIRS``) and news."""
        positions = user.get("positions") or []
        tickers = list(entities.tickers or []) or [
            str(p["ticker"]).upper() for p in positions if p.get("ticker")
        ]

        results, timings, wall = await self._run_parallel([
            ("portfolio_health", self._portfolio.run(user, intent=intent, query=query)),
            ("risk_analysis", self._risk.run(user, intent=intent, query=query)),
            ("news_agent", self._news.run(
                tickers=tickers, topics=["market risk", "volatility"],
                user=user, intent=intent, query=query,
            )),
        ])
        return {
            "portfolio": results["portfolio_health"],
            "risk": results["risk_analysis"],
            "news": results["news_agent"],
            "_timings": timings,
            "_wall_ms": wall,
        }

    async def _run_news_ecosystem(
        self, entities: Entity, user: dict, intent: str = "", query: str = "",
    ) -> dict[str, Any]:
        tickers = list(entities.tickers or [])
        topics = list(entities.topics or []) or ["markets"]
        results, timings, wall = await self._run_parallel([
            ("news_agent", self._news.run(
                tickers=tickers, topics=topics, user=user, intent=intent, query=query,
            )),
        ])
        return {
            "payload": results["news_agent"] or {},
            "_timings": timings,
            "_wall_ms": wall,
        }

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

        # Render the report serially after prefetch so timings reflect reality.
        report_results, report_timings, report_wall = await self._run_parallel([
            ("report_generator", self._report.run(report_type, user, tickers or None, fmt)),
        ])

        # Merge prefetch + report timings; wall_time_ms is the sum (sequential phases).
        merged_timings: dict[str, int] = {}
        merged_wall = 0.0
        if pre:
            merged_timings.update(pre.get("_timings") or {})
            merged_wall += float(pre.get("_wall_ms") or 0.0)
        merged_timings.update(report_timings)
        merged_wall += report_wall

        out: dict[str, Any] = {
            "report_type": report_type,
            "report": report_results["report_generator"] or {},
            "_timings": merged_timings,
            "_wall_ms": merged_wall,
        }
        if pre:
            # Strip private bookkeeping before exposing prefetch in the result.
            pre_clean = {k: v for k, v in pre.items() if not k.startswith("_")}
            out["prefetch"] = pre_clean
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
        execution_metadata: ExecutionMetadata | None = None,
    ) -> AgentResponse:
        # Strip private bookkeeping (``_timings``/``_wall_ms``) before the result
        # ships to the client — those values live on ``execution_metadata`` instead.
        if isinstance(result, dict):
            clean = {k: v for k, v in result.items() if not str(k).startswith("_")}
        else:
            clean = result
        return AgentResponse(
            agent=agent,
            implemented=implemented,
            intent=intent,
            entities=entities,
            result=clean,
            message=message or f"Completed {agent} orchestration.",
            execution_metadata=execution_metadata,
        )

    async def _safe_run(self, name: str, coro: Awaitable[Any]) -> tuple[Any, float]:
        """Run a coroutine, swallow exceptions, and return (result_or_None, elapsed_ms)."""
        start = time.perf_counter()
        try:
            result = await coro
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info("agent=%s status=success duration=%.0fms", name, duration_ms)
            return result, duration_ms
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "agent=%s status=error duration=%.0fms error=%s",
                name,
                duration_ms,
                e,
            )
            return None, duration_ms

    async def _run_parallel(
        self, items: list[tuple[str, Awaitable[Any]]],
    ) -> tuple[dict[str, Any], dict[str, int], float]:
        """Fan out named coroutines, capture per-task ms + wall-clock ms.

        Returns ``(results_by_name, timings_by_name_ms, wall_time_ms)``.
        Wall time is real awaited time (the parallelism win); timings sum
        gives the sequential estimate.
        """
        wall_start = time.perf_counter()
        gathered = await asyncio.gather(*(self._safe_run(n, c) for n, c in items))
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        results: dict[str, Any] = {}
        timings: dict[str, int] = {}
        for (name, _), (res, ms) in zip(items, gathered):
            results[name] = res
            timings[name] = round(ms)
        return results, timings, wall_ms

    @staticmethod
    def _build_exec_metadata(
        timings: dict[str, int], wall_ms: float,
    ) -> ExecutionMetadata:
        """Pack timings into the ExecutionMetadata that ships to the UI."""
        agents_ran = list(timings.keys())
        seq_ms = int(sum(timings.values()))
        wall = int(round(wall_ms))
        return ExecutionMetadata(
            agents_ran=agents_ran,
            timings=timings,
            parallel=len(agents_ran) > 1,
            wall_time_ms=wall,
            sequential_time_ms=seq_ms,
            time_saved_ms=max(0, seq_ms - wall),
        )

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


def _pop_timings(eco: dict[str, Any]) -> tuple[dict[str, int], float]:
    """Extract and remove the orchestrator's private timing keys from an ecosystem dict."""
    timings = eco.pop("_timings", None) or {}
    wall = float(eco.pop("_wall_ms", 0.0) or 0.0)
    return timings, wall


class ValuraAgnoTeam:
    """Native Agno ``Team`` coordinator with deterministic ``ValuraOrchestrator`` fallback.

    Two teams are created up-front:
      * ``portfolio_team`` runs in ``coordinate`` mode (this Agno build's analogue of
        the requested ``collaborate`` mode) for portfolio / risk-assessment queries.
      * ``research_team`` runs in ``route`` mode for market / news / report queries —
        the team leader picks the single best specialist.

    When no ``OPENAI_API_KEY`` / ``GROQ_API_KEY`` is configured, every call short-circuits
    to ``ValuraOrchestrator`` so existing tests and the no-key demo path still work.
    """

    PORTFOLIO_AGENTS = ("portfolio_health", "risk_assessment")
    RESEARCH_AGENTS = ("market_research", "predictive_analysis")

    def __init__(self, llm: LLMClient | None = None) -> None:
        from src.llm import get_llm_client
        from src.llm.agno_model import get_agno_model, get_agno_model_strong

        self._llm = llm or get_llm_client()
        self._fallback = ValuraOrchestrator(self._llm)
        self._stub = StubAgent()

        model = get_agno_model()
        strong = get_agno_model_strong() or model
        self._available = model is not None
        self._portfolio_team: Team | None = None
        self._research_team: Team | None = None
        self._news_agent: Agent | None = None
        self._report_agent: Agent | None = None

        if not self._available:
            return

        try:
            from src.mcp import (
                calculator_mcp,
                portfolio_analytics_mcp,
                report_mcp,
                web_search_mcp,
                yfinance_mcp,
            )

            portfolio_agent = Agent(
                name="Portfolio Health Analyst",
                role="Analyses portfolio holdings, concentration risk, and performance",
                model=model,
                tools=[yfinance_mcp, portfolio_analytics_mcp, calculator_mcp],
                instructions=[
                    "Analyse portfolio health comprehensively.",
                    "Always fetch live prices for all positions.",
                    "Compute concentration, performance, and benchmark comparison.",
                    "Provide plain-language observations.",
                ],
            )
            risk_agent = Agent(
                name="Risk Analyst",
                role="Computes portfolio risk metrics including VaR and stress tests",
                model=model,
                tools=[yfinance_mcp, portfolio_analytics_mcp],
                instructions=[
                    "Compute Value at Risk at 95% and 99% confidence.",
                    "Run all five stress test scenarios.",
                    "Identify dangerous position correlations above 0.7.",
                    "Calculate Sharpe ratio and explain it plainly.",
                ],
            )
            market_agent = Agent(
                name="Market Research Analyst",
                role="Researches individual stocks and market conditions",
                model=model,
                tools=[yfinance_mcp, web_search_mcp],
                instructions=[
                    "Fetch comprehensive price and fundamental data.",
                    "Search for recent relevant news.",
                    "Compare multiple tickers when provided.",
                    "Highlight the most important metric for investors.",
                ],
            )
            news_agent = Agent(
                name="Financial News Analyst",
                role="Aggregates and analyses financial news with sentiment scoring",
                model=model,
                tools=[web_search_mcp],
                instructions=[
                    "Search for news about all mentioned tickers.",
                    "Score sentiment for each article.",
                    "Identify the most market-moving headlines.",
                    "Summarise in three sentences what matters most.",
                ],
            )
            report_agent = Agent(
                name="Report Generator",
                role="Creates formatted financial reports in PDF or Markdown",
                model=model,
                tools=[report_mcp, yfinance_mcp],
                instructions=[
                    "Generate comprehensive, well-structured reports.",
                    "Always include an executive summary.",
                    "Include data tables for all quantitative information.",
                    "End every report with the regulatory disclaimer.",
                ],
            )

            # ``coordinate`` is this Agno build's collaborate-style mode:
            # the leader fans the task out and synthesises member outputs.
            self._portfolio_team = Team(
                name="Portfolio Analysis Team",
                mode="coordinate",
                model=strong,
                members=[portfolio_agent, risk_agent, news_agent],
                instructions=[
                    "You coordinate a team of financial specialists for Valura AI.",
                    "For portfolio queries every specialist analyses in parallel.",
                    "Synthesise their outputs into one unified response.",
                    "The portfolio analyst handles holdings and performance.",
                    "The risk analyst handles VaR, stress tests, and correlations.",
                    "The news analyst handles recent market news and sentiment.",
                    "Combine insights into a clear, actionable response.",
                    "Always include the regulatory disclaimer.",
                ],
                markdown=True,
                debug_mode=True,
            )
            self._research_team = Team(
                name="Market Research Team",
                mode="route",
                model=strong,
                members=[market_agent, news_agent, report_agent],
                instructions=[
                    "Route financial research queries to the right specialist.",
                    "Market data and stock analysis -> Market Research Analyst.",
                    "News and sentiment queries -> Financial News Analyst.",
                    "Report generation requests -> Report Generator.",
                    "For compare queries route to Market Research Analyst.",
                ],
                debug_mode=True,
            )
            self._news_agent = news_agent
            self._report_agent = report_agent
        except Exception as e:
            logger.warning("ValuraAgnoTeam construction failed: %s", e)
            self._available = False

    async def run(
        self,
        classifier_result: ClassifierResult,
        user: dict,
        query: str = "",
    ) -> AgentResponse:
        if not self._available:
            return await self._fallback.run(classifier_result, user, query=query)

        agent = (classifier_result.agent or "general_query").strip()
        intent = classifier_result.intent or ""
        entities = classifier_result.entities

        try:
            if agent in self.PORTFOLIO_AGENTS:
                return await self._run_portfolio_team(agent, user, intent, entities)
            if agent in self.RESEARCH_AGENTS:
                return await self._run_research_team(agent, intent, entities, user)
            if agent == "financial_news":
                return await self._run_news_direct(intent, entities, user)
            if agent == "report_generator":
                return await self._run_report_direct(intent, entities, user)
            return await self._fallback.run(classifier_result, user, query=query)
        except Exception as e:
            logger.error("AgnoTeam error: %s", e)
            return await self._fallback.run(classifier_result, user, query=query)

    async def _run_portfolio_team(
        self, agent: str, user: dict, intent: str, entities: Entity,
    ) -> AgentResponse:
        assert self._portfolio_team is not None
        prompt = self._build_portfolio_prompt(user, intent)
        start = time.perf_counter()
        response = await self._portfolio_team.arun(prompt, stream=False)
        wall_ms = int((time.perf_counter() - start) * 1000)
        content = self._extract_content(response)
        meta = ExecutionMetadata(
            agents_ran=["portfolio_health", "risk_analysis", "news_agent"],
            timings={"portfolio_team": wall_ms},
            parallel=True,
            wall_time_ms=wall_ms,
            sequential_time_ms=wall_ms,
            time_saved_ms=0,
        )
        return AgentResponse(
            agent=agent,
            implemented=True,
            intent=intent,
            entities=entities,
            result={
                "content": content,
                "team": "portfolio_analysis_team",
                "mode": "coordinate",
                "members": ["portfolio_analyst", "risk_analyst", "news_analyst"],
            },
            message=self._snippet(content),
            execution_metadata=meta,
        )

    async def _run_research_team(
        self, agent: str, intent: str, entities: Entity, user: dict,
    ) -> AgentResponse:
        assert self._research_team is not None
        tickers = list(entities.tickers or [])
        prompt = (
            f"{intent}. Tickers: {', '.join(tickers) if tickers else 'none specified'}. "
            f"User: {user.get('name') or 'investor'} (risk {user.get('risk_profile') or 'unknown'})."
        )
        start = time.perf_counter()
        response = await self._research_team.arun(prompt, stream=False)
        wall_ms = int((time.perf_counter() - start) * 1000)
        content = self._extract_content(response)
        meta = ExecutionMetadata(
            agents_ran=["market_research"],
            timings={"research_team": wall_ms},
            parallel=False,
            wall_time_ms=wall_ms,
            sequential_time_ms=wall_ms,
            time_saved_ms=0,
        )
        return AgentResponse(
            agent="market_research" if agent != "report_generator" else agent,
            implemented=True,
            intent=intent,
            entities=entities,
            result={
                "content": content,
                "team": "market_research_team",
                "mode": "route",
            },
            message=self._snippet(content),
            execution_metadata=meta,
        )

    async def _run_news_direct(
        self, intent: str, entities: Entity, user: dict,
    ) -> AgentResponse:
        assert self._news_agent is not None
        tickers = list(entities.tickers or [])
        prompt = (
            f"News digest for: {intent}. Tickers: "
            f"{', '.join(tickers) if tickers else 'broad market'}."
        )
        start = time.perf_counter()
        response = await self._news_agent.arun(prompt, stream=False)
        wall_ms = int((time.perf_counter() - start) * 1000)
        content = self._extract_content(response)
        meta = ExecutionMetadata(
            agents_ran=["news_agent"],
            timings={"news_agent": wall_ms},
            parallel=False,
            wall_time_ms=wall_ms,
            sequential_time_ms=wall_ms,
            time_saved_ms=0,
        )
        return AgentResponse(
            agent="financial_news",
            implemented=True,
            intent=intent,
            entities=entities,
            result={"content": content},
            message=self._snippet(content),
            execution_metadata=meta,
        )

    async def _run_report_direct(
        self, intent: str, entities: Entity, user: dict,
    ) -> AgentResponse:
        assert self._report_agent is not None
        tickers = list(entities.tickers or [])
        prompt = (
            f"Generate a financial report. Intent: {intent}. Tickers: "
            f"{', '.join(tickers) if tickers else 'portfolio'}. "
            f"User positions: {user.get('positions') or []}."
        )
        start = time.perf_counter()
        response = await self._report_agent.arun(prompt, stream=False)
        wall_ms = int((time.perf_counter() - start) * 1000)
        content = self._extract_content(response)
        meta = ExecutionMetadata(
            agents_ran=["report_generator"],
            timings={"report_generator": wall_ms},
            parallel=False,
            wall_time_ms=wall_ms,
            sequential_time_ms=wall_ms,
            time_saved_ms=0,
        )
        return AgentResponse(
            agent="report_generator",
            implemented=True,
            intent=intent,
            entities=entities,
            result={"content": content},
            message=self._snippet(content),
            execution_metadata=meta,
        )

    def _build_portfolio_prompt(self, user: dict, intent: str) -> str:
        from src.session import agno_memory

        positions = user.get("positions") or []
        if positions:
            pos_text = "\n".join(
                f"- {p.get('ticker')}: {p.get('quantity')} shares, "
                f"avg cost ${p.get('avg_cost')}, bought {p.get('purchased_at')}"
                for p in positions
            )
        else:
            pos_text = "No positions (new investor)"
        bench = (user.get("preferences") or {}).get("preferred_benchmark") or "S&P 500"
        memory_text = agno_memory.format_for_prompt(user.get("_memories") or [])
        return (
            f"{memory_text}"
            f"User: {user.get('name') or 'Unknown'} | Age: {user.get('age')} | "
            f"Risk Profile: {user.get('risk_profile')} | Currency: {user.get('base_currency') or 'USD'}\n\n"
            f"Portfolio Positions:\n{pos_text}\n\n"
            f"Benchmark: {bench}\n\n"
            f"Query: {intent}\n\n"
            "Please provide a comprehensive analysis covering portfolio health, "
            "risk metrics, and relevant market news."
        )

    @staticmethod
    def _extract_content(response: Any) -> Any:
        return getattr(response, "content", response)

    @staticmethod
    def _snippet(content: Any) -> str:
        return str(content or "")[:200] if content is not None else "Team response completed."


__all__ = ["ValuraOrchestrator", "ValuraAgnoTeam"]
