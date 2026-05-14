"""FastAPI app: safety → classify → route → SSE. Default timeout 30s via ``PIPELINE_TIMEOUT``."""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from sse_starlette.sse import EventSourceResponse

from src.agents.market_research import MarketResearchAgent
from src.agents.news_agent import FinancialNewsAgent
from src.agents.portfolio_health import PortfolioHealthAgent
from src.agents.report_generator import ReportGeneratorAgent
from src.agents.risk_analysis import RiskAnalysisAgent
from src.classifier import classify
from src.llm import get_llm_client
from src.logging_config import (
    correlation_id_var,
    get_logger,
    new_correlation_id,
    setup_logging,
)
from src.mcp import calculator_mcp, portfolio_analytics_mcp, yfinance_mcp
from src.models import AgentResponse, ChatRequest, PortfolioHealthResult
from src.orchestrator import ValuraOrchestrator  # noqa: F401  (re-exported for callers)
from src.router import AgentRouter
from src.safety import check as safety_check
from src.session import ConversationTurn, agno_memory, query_cache, session_store

load_dotenv()
setup_logging()


def _pipeline_timeout_s() -> float:
    # 90s default: Agno team coordinate mode runs 3 member agents each making
    # multiple LLM calls, easily exceeding 30s on a free Groq tier. The
    # deterministic fallback still completes in <10s if Agno fails or is cancelled.
    raw = os.environ.get("PIPELINE_TIMEOUT", "90")
    try:
        return float(raw)
    except ValueError:
        return 90.0


PIPELINE_TIMEOUT = _pipeline_timeout_s()

logger = get_logger("api")


# ---------------------------------------------------------------------------
# Metrics — module-level counters updated by /chat, served by /metrics.
# Single asyncio lock keeps increments atomic even under concurrent load.
# Response times use a rolling window so memory stays bounded.
# ---------------------------------------------------------------------------
_METRICS_WINDOW = 1_000
_PARALLEL_AGENTS = frozenset(
    {
        "portfolio_health",
        "market_research",
        "risk_assessment",
        "financial_news",
        "report_generator",
    }
)


class _Metrics:
    def __init__(self) -> None:
        self.total_requests_served: int = 0
        self.requests_per_agent: dict[str, int] = {}
        self.safety_blocks_count: int = 0
        self.parallel_execution_count: int = 0
        self._response_times_ms: list[float] = []
        self._lock = asyncio.Lock()
        self._service_started_at = time.time()

    async def record_request(self, agent: str, elapsed_ms: float) -> None:
        async with self._lock:
            self.total_requests_served += 1
            self.requests_per_agent[agent] = self.requests_per_agent.get(agent, 0) + 1
            if agent in _PARALLEL_AGENTS:
                self.parallel_execution_count += 1
            self._response_times_ms.append(elapsed_ms)
            if len(self._response_times_ms) > _METRICS_WINDOW:
                self._response_times_ms = self._response_times_ms[-_METRICS_WINDOW:]

    async def record_safety_block(self) -> None:
        async with self._lock:
            self.safety_blocks_count += 1

    def snapshot(self) -> dict:
        rts = list(self._response_times_ms)
        avg = round(sum(rts) / len(rts), 2) if rts else 0.0
        p95 = round(sorted(rts)[int(len(rts) * 0.95)], 2) if len(rts) >= 20 else None
        return {
            "uptime_seconds": round(time.time() - self._service_started_at, 1),
            "total_requests_served": self.total_requests_served,
            "requests_per_agent": dict(self.requests_per_agent),
            "average_response_time_ms": avg,
            "p95_response_time_ms": p95,
            "safety_blocks_count": self.safety_blocks_count,
            "parallel_execution_count": self.parallel_execution_count,
            "response_time_window_size": len(rts),
        }


metrics = _Metrics()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()  # idempotent — ensures handlers are reset if uvicorn rewired them.
    logger.info(
        "service_startup",
        extra={
            "event": "lifecycle",
            "phase": "startup",
            "agents": 5,
            "mcp_servers": 5,
            "orchestrator": "ValuraOrchestrator",
            "pipeline_timeout_s": PIPELINE_TIMEOUT,
        },
    )
    yield
    logger.info(
        "service_shutdown",
        extra={"event": "lifecycle", "phase": "shutdown"},
    )


app = FastAPI(
    title="Valura AI Microservice",
    description="AI co-investor agent ecosystem",
    version="0.1.0",
    lifespan=lifespan,
)

# Resolve frontend dir relative to this file so it works regardless of CWD.
_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")

# Generated reports — served raw so the UI can offer a one-click view/download.
_REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
os.makedirs(_REPORTS_DIR, exist_ok=True)
app.mount("/reports", StaticFiles(directory=_REPORTS_DIR), name="reports")


def _sse(event: str, data: str) -> dict[str, str]:
    return {"event": event, "data": data}


_HOUSEKEEPING_OBS_KEYWORDS = (
    "benchmark total return was unavailable",
    "live quote unavailable",
)


def _is_housekeeping(text: str) -> bool:
    """Filter out tooling-warning observations so we don't repeat them in the summary."""
    low = (text or "").lower()
    return any(kw in low for kw in _HOUSEKEEPING_OBS_KEYWORDS)


def _pick_lead_observation(observations: list) -> str:
    """First *substantive* observation text, skipping housekeeping warnings."""
    for o in observations:
        text = o.get("text") if isinstance(o, dict) else getattr(o, "text", "")
        text = str(text or "").strip()
        if text and not _is_housekeeping(text):
            return text
    return ""


def _build_summary(response: AgentResponse) -> str:
    """
    Build the streamed delta text. Rules:
    - Agno Team / Agent path: result is ``{"content": "..."}`` — surface the team's
      synthesised markdown directly.
    - portfolio_health (ecosystem dict): ``ecosystem_summary`` + one substantive
      observation + disclaimer. Never pick the auto-injected "benchmark unavailable"
      obs as the lead — the orchestrator already mentions the benchmark numerically.
    - Legacy single-result ``PortfolioHealthResult``: lead observation + disclaimer.
    - Everything else: ``response.message`` (orchestrator builds these to be self-contained).
    """
    result = response.result

    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    if response.agent == "portfolio_health" and result is not None:
        if isinstance(result, PortfolioHealthResult):
            lead = _pick_lead_observation(result.observations)
            return f"{lead} {result.disclaimer}".strip() if lead else result.disclaimer

        if isinstance(result, dict):
            nested = result.get("portfolio_health")
            if isinstance(nested, dict):
                summ = str(result.get("ecosystem_summary") or "").strip()
                lead = _pick_lead_observation(nested.get("observations") or [])
                disc = str(nested.get("disclaimer") or "")
                pieces = [p for p in (summ, lead, disc) if p]
                if pieces:
                    return " ".join(pieces)
            lead = _pick_lead_observation(result.get("observations") or [])
            disc = str(result.get("disclaimer") or "")
            joined = " ".join(p for p in (lead, disc) if p)
            if joined.strip():
                return joined
    return response.message or "Response ready — see Raw JSON for full details."


@app.post("/chat")
async def chat(request: ChatRequest) -> EventSourceResponse:
    """SSE only: ``delta`` (summary chunks), ``result`` (JSON), optional ``error``, trailing ``done``."""
    llm = get_llm_client()
    router = AgentRouter(llm)

    async def event_generator():
        correlation_id = new_correlation_id()
        correlation_id_var.set(correlation_id)
        t0 = time.monotonic()
        t_safety = 0.0
        t_classify = 0.0
        t_agent = 0.0
        logger.info(
            "chat_request_started",
            extra={
                "event": "chat",
                "phase": "start",
                "session_id": request.session_id,
                "user_id": request.user.get("user_id"),
                "query_length": len(request.query or ""),
            },
        )
        try:
            async with asyncio.timeout(PIPELINE_TIMEOUT):
                t_safety_start = time.monotonic()
                verdict = safety_check(request.query)
                t_safety = time.monotonic() - t_safety_start
                if verdict.blocked:
                    yield _sse("delta", verdict.message)
                    yield _sse("done", "")
                    elapsed = time.monotonic() - t0
                    await metrics.record_safety_block()
                    logger.info(
                        "chat_request_blocked",
                        extra={
                            "event": "chat",
                            "phase": "safety_block",
                            "session_id": request.session_id,
                            "category": verdict.category,
                            "elapsed_s": round(elapsed, 3),
                        },
                    )
                    return

                cached = query_cache.get(request.session_id, request.query)
                t_classify_start = time.monotonic()
                if cached is not None:
                    classifier_result = cached
                else:
                    prior_turns = await session_store.get_prior_user_turns(
                        request.session_id
                    )
                    last_entities = await session_store.get_last_entities(
                        request.session_id
                    )
                    classifier_result = await classify(
                        request.query,
                        llm=llm,
                        prior_user_turns=prior_turns,
                        last_entities=last_entities,
                    )
                    query_cache.set(
                        request.session_id, request.query, classifier_result
                    )
                t_classify = time.monotonic() - t_classify_start

                # Inject long-term Agno memories into the user dict so the
                # orchestrator/team can surface them in the system prompt.
                user_id = str(request.user.get("user_id") or request.session_id)
                user_memories = await agno_memory.get_user_memories(user_id)
                user_with_memory = {**request.user, "_memories": user_memories}

                t_agent_start = time.monotonic()
                agent_response = await router.route(
                    classifier_result, user_with_memory, query=request.query
                )
                t_agent = time.monotonic() - t_agent_start

                # Persist this turn as a long-term memory record (best-effort).
                asyncio.create_task(
                    agno_memory.add_user_memory(
                        user_id=user_id,
                        conversation=request.query,
                        response=agent_response.message,
                    )
                )

                summary = _build_summary(agent_response)
                for word in summary.split():
                    yield _sse("delta", word + " ")
                    await asyncio.sleep(0)

                yield _sse("result", agent_response.model_dump_json())

                assistant_preview = summary[:200]
                await session_store.add_turn(
                    request.session_id,
                    ConversationTurn(
                        user=request.query,
                        assistant=assistant_preview,
                        agent=agent_response.agent,
                        entities=agent_response.entities.model_dump(),
                    ),
                )

                t_total = time.monotonic() - t0
                await metrics.record_request(agent_response.agent, t_total * 1000.0)
                logger.info(
                    "request completed | "
                    f"session={request.session_id} | "
                    f"agent={agent_response.agent} | "
                    f"safety={t_safety*1000:.1f}ms | "
                    f"classify={t_classify*1000:.1f}ms | "
                    f"agent={t_agent*1000:.1f}ms | "
                    f"total={t_total*1000:.1f}ms"
                )
                yield _sse("done", "")

        except TimeoutError:
            logger.error(
                "chat_pipeline_timeout",
                extra={
                    "event": "chat",
                    "phase": "timeout",
                    "session_id": request.session_id,
                    "limit_s": PIPELINE_TIMEOUT,
                },
            )
            yield _sse(
                "error",
                json.dumps(
                    {
                        "code": "timeout",
                        "message": "Request timed out. Please try again.",
                    }
                ),
            )
            yield _sse("done", "")
        except Exception:
            logger.exception(
                "chat_pipeline_unhandled_error",
                extra={
                    "event": "chat",
                    "phase": "error",
                    "session_id": request.session_id,
                },
            )
            yield _sse(
                "error",
                json.dumps(
                    {
                        "code": "internal_error",
                        "message": "An unexpected error occurred.",
                    }
                ),
            )
            yield _sse("done", "")

    return EventSourceResponse(event_generator())


@app.get("/")
async def serve_frontend():
    index_path = os.path.join(_FRONTEND_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return JSONResponse(
        {"status": "ok", "service": "valura-ai", "ui": "frontend/index.html missing"}
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "valura-ai"}


@app.get("/users")
async def list_users() -> dict:
    """Loads the bundled fixture profiles so the frontend can populate its dropdown without hardcoding."""
    fixtures_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "fixtures", "users"
    )
    out: list[dict] = []
    if os.path.isdir(fixtures_dir):
        for name in sorted(os.listdir(fixtures_dir)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(fixtures_dir, name), encoding="utf-8") as fh:
                    profile = json.load(fh)
                out.append(
                    {
                        "id": profile.get("user_id") or name.split(".")[0],
                        "label": (
                            f"{profile.get('name', name)} — "
                            f"{profile.get('risk_profile', 'n/a')}, "
                            f"{len(profile.get('positions') or [])} positions"
                        ),
                        "profile": profile,
                    }
                )
            except Exception:
                logger.exception("failed to load user fixture %s", name)
    return {"users": out}


@app.get("/agents")
async def list_agents() -> dict:
    """Lists agents in the ecosystem with coarse status."""
    return {
        "total_agents": 5,
        "orchestrator": "ValuraOrchestrator",
        "memory": {
            "type": f"Agno Memory ({agno_memory.backend})",
            "enabled": agno_memory.enabled,
            "scope": "cross-session user facts",
            "complement": "in-session turn history via SessionStore",
        },
        "mcp_servers": [
            {
                "name": "yfinance_mcp",
                "description": "Live market data — quotes, fundamentals, history, options",
            },
            {
                "name": "web_search_mcp",
                "description": "Web/news search — DuckDuckGo + Bing fallbacks for market analysis",
            },
            {
                "name": "report_mcp",
                "description": "PDF and Markdown report generation",
            },
            {
                "name": "calculator_mcp",
                "description": "Financial calculations - compound interest, DCA, Black-Scholes, retirement",
            },
            {
                "name": "portfolio_analytics_mcp",
                "description": "Portfolio analytics - beta, sector exposure, dividends, attribution",
            },
        ],
        "agents": [
            {
                "name": "portfolio_health",
                "status": "implemented",
                "description": "Portfolio analysis, concentration risk, benchmark comparison",
                "parallel_with": ["risk_analysis", "news_agent"],
            },
            {
                "name": "market_research",
                "status": "implemented",
                "description": "Live stock data, company fundamentals, price snapshots",
                "parallel_with": ["news_agent"],
            },
            {
                "name": "risk_analysis",
                "status": "implemented",
                "description": "VaR, max drawdown, Sharpe ratio, stress testing",
                "parallel_with": ["portfolio_health"],
            },
            {
                "name": "financial_news",
                "status": "implemented",
                "description": "Real-time news aggregation with sentiment scoring",
                "parallel_with": ["any primary agent"],
            },
            {
                "name": "report_generator",
                "status": "implemented",
                "description": "PDF and Markdown report generation",
                "parallel_with": [],
            },
            {
                "name": "financial_calculator",
                "status": "stub",
                "description": "Coming soon",
                "parallel_with": [],
            },
            {
                "name": "investment_strategy",
                "status": "stub",
                "description": "Coming soon",
                "parallel_with": [],
            },
        ],
    }


# ===========================================================================
# Focused JSON endpoints
# ---------------------------------------------------------------------------
# Each domain endpoint fixes the agent + sub-intent and returns structured
# data (no SSE). Sub-intents are triggered by passing a stable keyword to the
# agent's run() so the existing keyword-based dispatcher routes unambiguously.
# Free-text ``query`` is forwarded so any nuance (e.g. specific ticker focus
# for risk_correlation) still flows through.
# ===========================================================================


class MarketRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tickers: list[str] = Field(default_factory=list)
    user: dict = Field(default_factory=dict)


class NewsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tickers: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    user: dict = Field(default_factory=dict)


class CalcCompoundRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    principal: float
    annual_rate: float
    years: int
    compounds_per_year: int = 12
    monthly_contribution: float = 0.0


class CalcDCARequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    monthly_amount: float
    annual_return_pct: float
    years: int
    initial_investment: float = 0.0


class CalcRetirementRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    current_age: int
    retirement_age: int
    current_savings: float
    monthly_contribution: float
    expected_annual_return: float
    monthly_expenses_in_retirement: float


class CalcOptionsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stock_price: float
    strike_price: float
    time_to_expiry_days: int
    volatility_pct: float
    risk_free_rate: float = 0.0525
    option_type: str = "call"


class ReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user: dict = Field(default_factory=dict)
    format: str = "markdown"


class MarketReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tickers: list[str] = Field(default_factory=list)
    format: str = "markdown"


def _require_positions(user: dict) -> None:
    if not (user or {}).get("positions"):
        raise HTTPException(
            status_code=400,
            detail="user.positions is empty — supply a profile with at least one holding",
        )


def _require_tickers(tickers: list[str]) -> list[str]:
    cleaned = [str(t).strip().upper() for t in (tickers or []) if str(t or "").strip()]
    if not cleaned:
        raise HTTPException(
            status_code=400, detail="tickers list cannot be empty"
        )
    return cleaned


def _result_to_json(result) -> dict:
    """PortfolioHealthResult / MarketResearchResult → plain dict; pass dicts through."""
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if isinstance(result, dict):
        return result
    return {"result": result}


async def _safe_call(coro, label: str):
    """Run a coroutine; convert unhandled errors into a clean 500 response."""
    try:
        return await coro
    except HTTPException:
        raise
    except Exception:
        logger.exception("%s failed", label)
        raise HTTPException(status_code=500, detail=f"{label} failed unexpectedly")


# ---------------------------------------------------------------------------
# Portfolio endpoints
# ---------------------------------------------------------------------------

@app.post("/portfolio/health")
async def portfolio_health(request: ChatRequest) -> dict:
    """Full portfolio health check — concentration, performance, benchmark, observations."""
    _require_positions(request.user)
    agent = PortfolioHealthAgent(get_llm_client())
    result = await _safe_call(
        agent.run(request.user, intent="", query=request.query),
        "portfolio_health",
    )
    return _result_to_json(result)


@app.post("/portfolio/concentration")
async def portfolio_concentration(request: ChatRequest) -> dict:
    """Concentration risk and live sector breakdown."""
    _require_positions(request.user)
    agent = PortfolioHealthAgent(get_llm_client())
    result = await _safe_call(
        agent.run(request.user, intent="concentration diversification analysis", query=request.query),
        "portfolio_concentration",
    )
    return _result_to_json(result)


@app.post("/portfolio/performance")
async def portfolio_performance(request: ChatRequest) -> dict:
    """Returns vs cost basis, best/worst positions, annualised performance."""
    _require_positions(request.user)
    agent = PortfolioHealthAgent(get_llm_client())
    result = await _safe_call(
        agent.run(request.user, intent="performance returns analysis", query=request.query),
        "portfolio_performance",
    )
    return _result_to_json(result)


@app.post("/portfolio/benchmark")
async def portfolio_benchmark(request: ChatRequest) -> dict:
    """Comparison against S&P 500 / QQQ / user's preferred benchmark."""
    _require_positions(request.user)
    agent = PortfolioHealthAgent(get_llm_client())
    result = await _safe_call(
        agent.run(request.user, intent="benchmark comparison vs index", query=request.query),
        "portfolio_benchmark",
    )
    return _result_to_json(result)


@app.post("/portfolio/rebalance")
async def portfolio_rebalance(request: ChatRequest) -> dict:
    """Rebalancing suggestions with exact buy/sell trades."""
    _require_positions(request.user)
    agent = PortfolioHealthAgent(get_llm_client())
    result = await _safe_call(
        agent.run(request.user, intent="rebalance trades suggestion", query=request.query),
        "portfolio_rebalance",
    )
    return _result_to_json(result)


@app.post("/portfolio/tax-loss")
async def portfolio_tax_loss(request: ChatRequest) -> dict:
    """Tax-loss harvesting opportunities + wash-sale-safe replacements."""
    _require_positions(request.user)
    agent = PortfolioHealthAgent(get_llm_client())
    result = await _safe_call(
        agent.run(request.user, intent="tax loss harvesting", query=request.query),
        "portfolio_tax_loss",
    )
    return _result_to_json(result)


@app.post("/portfolio/sector-exposure")
async def portfolio_sector_exposure(request: ChatRequest) -> dict:
    """Sector + geography breakdown straight from portfolio_analytics_mcp."""
    _require_positions(request.user)
    agent = PortfolioHealthAgent(get_llm_client())

    async def _compute() -> dict:
        rows, _, _, warnings = await agent._priced_positions(request.user)
        if not rows:
            return {"sector_exposure": {}, "geographic_exposure": {}, "warnings": warnings}

        total = sum(r["current_value"] for r in rows) or 1.0
        tickers = [r["ticker"] for r in rows]
        weights = [r["current_value"] / total for r in rows]

        sector, geo = await asyncio.gather(
            asyncio.to_thread(portfolio_analytics_mcp.sector_exposure, tickers, weights),
            asyncio.to_thread(portfolio_analytics_mcp.geographic_exposure, tickers, weights),
        )
        return {
            "tickers": tickers,
            "weights": [round(w, 4) for w in weights],
            "sector_exposure": sector,
            "geographic_exposure": geo,
            "warnings": warnings,
        }

    return await _safe_call(_compute(), "portfolio_sector_exposure")


@app.post("/portfolio/dividends")
async def portfolio_dividends(request: ChatRequest) -> dict:
    """Annual dividend income, monthly cash flow estimate, top payers."""
    _require_positions(request.user)
    agent = PortfolioHealthAgent(get_llm_client())

    async def _compute() -> dict:
        rows, _, _, warnings = await agent._priced_positions(request.user)
        if not rows:
            return {"dividend_analysis": {}, "warnings": warnings}
        payload = await asyncio.to_thread(portfolio_analytics_mcp.dividend_analysis, rows)
        return {"dividend_analysis": payload, "warnings": warnings}

    return await _safe_call(_compute(), "portfolio_dividends")


# ---------------------------------------------------------------------------
# Market research endpoints
# ---------------------------------------------------------------------------

@app.post("/market/snapshot")
async def market_snapshot(request: MarketRequest) -> dict:
    """Current price, day change, volume for one or more tickers."""
    tickers = _require_tickers(request.tickers)
    agent = MarketResearchAgent(get_llm_client())
    result = await _safe_call(
        agent.run(tickers, intent="price snapshot quote"),
        "market_snapshot",
    )
    return _result_to_json(result)


@app.post("/market/fundamentals")
async def market_fundamentals(request: MarketRequest) -> dict:
    """P/E, revenue, margins, debt ratios, growth metrics."""
    tickers = _require_tickers(request.tickers)
    agent = MarketResearchAgent(get_llm_client())
    # ``fundament`` (no plural) is what the agent's word-boundary regex matches.
    result = await _safe_call(
        agent.run(tickers, intent="P/E ratio revenue fundament check"),
        "market_fundamentals",
    )
    return _result_to_json(result)


@app.post("/market/technical")
async def market_technical(request: MarketRequest) -> dict:
    """52-week range, moving averages, golden/death cross signals."""
    tickers = _require_tickers(request.tickers)
    agent = MarketResearchAgent(get_llm_client())
    result = await _safe_call(
        agent.run(tickers, intent="technical levels moving average"),
        "market_technical",
    )
    return _result_to_json(result)


@app.post("/market/compare")
async def market_compare(request: MarketRequest) -> dict:
    """Side-by-side comparison of multiple tickers."""
    tickers = _require_tickers(request.tickers)
    if len(tickers) < 2:
        raise HTTPException(
            status_code=400, detail="comparison requires at least 2 tickers"
        )
    agent = MarketResearchAgent(get_llm_client())
    result = await _safe_call(
        agent.run(tickers, intent="compare side by side"),
        "market_compare",
    )
    return _result_to_json(result)


@app.post("/market/options")
async def market_options(request: MarketRequest) -> dict:
    """Options activity — put/call ratio, implied vol, open interest."""
    tickers = _require_tickers(request.tickers)
    agent = MarketResearchAgent(get_llm_client())
    result = await _safe_call(
        agent.run(tickers, intent="options activity implied vol"),
        "market_options",
    )
    return _result_to_json(result)


@app.post("/market/screen")
async def market_screen(request: MarketRequest) -> dict:
    """Stock screener — concurrent metrics for a watchlist, sorted by market cap."""
    tickers = _require_tickers(request.tickers)
    rows = await _safe_call(
        asyncio.to_thread(yfinance_mcp.screen_stocks, tickers),
        "market_screen",
    )
    return {"tickers": tickers, "rows": rows, "count": len(rows or [])}


# ---------------------------------------------------------------------------
# Risk endpoints
# ---------------------------------------------------------------------------

@app.post("/risk/full")
async def risk_full(request: ChatRequest) -> dict:
    """Full risk pack — VaR, drawdown, Sharpe, stress tests, correlations."""
    _require_positions(request.user)
    agent = RiskAnalysisAgent(get_llm_client())
    return await _safe_call(
        agent.run(request.user, intent="", query=request.query),
        "risk_full",
    )


@app.post("/risk/var")
async def risk_var(request: ChatRequest) -> dict:
    """Value at Risk at 95% and 99% confidence."""
    _require_positions(request.user)
    agent = RiskAnalysisAgent(get_llm_client())
    return await _safe_call(
        agent.run(request.user, intent="value at risk var", query=request.query),
        "risk_var",
    )


@app.post("/risk/stress-test")
async def risk_stress_test(request: ChatRequest) -> dict:
    """Portfolio value under five historical crash scenarios."""
    _require_positions(request.user)
    agent = RiskAnalysisAgent(get_llm_client())
    return await _safe_call(
        agent.run(request.user, intent="stress test scenarios", query=request.query),
        "risk_stress_test",
    )


@app.post("/risk/correlation")
async def risk_correlation(request: ChatRequest) -> dict:
    """Pairwise correlations above the 0.7 significance threshold."""
    _require_positions(request.user)
    agent = RiskAnalysisAgent(get_llm_client())
    return await _safe_call(
        agent.run(request.user, intent="correlation between holdings", query=request.query),
        "risk_correlation",
    )


@app.post("/risk/volatility")
async def risk_volatility(request: ChatRequest) -> dict:
    """Beta, 30-day std-dev and Bollinger position per holding."""
    _require_positions(request.user)
    agent = RiskAnalysisAgent(get_llm_client())
    return await _safe_call(
        agent.run(request.user, intent="volatility beta bollinger", query=request.query),
        "risk_volatility",
    )


# ---------------------------------------------------------------------------
# News endpoints
# ---------------------------------------------------------------------------

@app.post("/news/tickers")
async def news_tickers(request: NewsRequest) -> dict:
    """Latest news for specific tickers."""
    tickers = _require_tickers(request.tickers)
    agent = FinancialNewsAgent(get_llm_client())
    return await _safe_call(
        agent.run(tickers, request.topics, request.user, intent="ticker news"),
        "news_tickers",
    )


@app.post("/news/market")
async def news_market(request: NewsRequest) -> dict:
    """Broad market news — indices, macro, headlines."""
    agent = FinancialNewsAgent(get_llm_client())
    return await _safe_call(
        agent.run(
            request.tickers, request.topics, request.user,
            intent="market news today update",
        ),
        "news_market",
    )


@app.post("/news/sentiment")
async def news_sentiment(request: NewsRequest) -> dict:
    """Aggregate bullish/bearish sentiment score for the supplied tickers."""
    if not request.tickers and not request.topics:
        raise HTTPException(
            status_code=400, detail="supply at least one ticker or topic for sentiment"
        )
    agent = FinancialNewsAgent(get_llm_client())
    return await _safe_call(
        agent.run(
            request.tickers, request.topics, request.user,
            intent="news sentiment summary",
        ),
        "news_sentiment",
    )


@app.post("/news/economic-calendar")
async def news_economic_calendar() -> dict:
    """Upcoming Fed / earnings / macro releases — no body required."""
    agent = FinancialNewsAgent(get_llm_client())
    return await _safe_call(
        agent.run([], [], {}, intent="economic calendar events"),
        "news_economic_calendar",
    )


@app.post("/news/sector")
async def news_sector(request: NewsRequest) -> dict:
    """News grouped by sector — uses ``request.sectors`` (or sensible defaults)."""
    sectors = request.sectors or request.topics
    agent = FinancialNewsAgent(get_llm_client())
    return await _safe_call(
        agent.run(
            request.tickers, sectors, request.user,
            intent="sector industry news",
        ),
        "news_sector",
    )


# ---------------------------------------------------------------------------
# Calculator endpoints — direct pass-through to the calculator MCP
# ---------------------------------------------------------------------------

@app.post("/calculate/compound-interest")
async def calculate_compound(request: CalcCompoundRequest) -> dict:
    return calculator_mcp.compound_interest(**request.model_dump())


@app.post("/calculate/dca")
async def calculate_dca(request: CalcDCARequest) -> dict:
    return calculator_mcp.dca_projection(**request.model_dump())


@app.post("/calculate/retirement")
async def calculate_retirement(request: CalcRetirementRequest) -> dict:
    return calculator_mcp.retirement_projection(**request.model_dump())


@app.post("/calculate/options-price")
async def calculate_options(request: CalcOptionsRequest) -> dict:
    return calculator_mcp.options_black_scholes(**request.model_dump())


# ---------------------------------------------------------------------------
# Report endpoints
# ---------------------------------------------------------------------------

@app.post("/report/portfolio")
async def report_portfolio(request: ReportRequest) -> dict:
    """Generate a full portfolio report (markdown or PDF)."""
    _require_positions(request.user)
    agent = ReportGeneratorAgent(get_llm_client())
    return await _safe_call(
        agent.generate_portfolio_report(request.user, request.format),
        "report_portfolio",
    )


@app.post("/report/market")
async def report_market(request: MarketReportRequest) -> dict:
    """Generate a market research report for one or more tickers."""
    tickers = _require_tickers(request.tickers)
    agent = ReportGeneratorAgent(get_llm_client())
    return await _safe_call(
        agent.generate_market_report(tickers, request.format),
        "report_market",
    )


@app.post("/report/risk")
async def report_risk(request: ReportRequest) -> dict:
    """Generate a risk analysis report."""
    _require_positions(request.user)
    agent = ReportGeneratorAgent(get_llm_client())
    return await _safe_call(
        agent.generate_risk_report(request.user, request.format),
        "report_risk",
    )


# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------

@app.get("/metrics")
async def system_metrics() -> dict:
    """Lightweight observability snapshot — counters live in-process."""
    snap = metrics.snapshot()
    snap["logging"] = {
        "format": "structured JSON",
        "correlation_ids": True,
        "example": 'grep correlation_id=req_7f3a2b1c server.log',
    }
    return snap


@app.get("/docs-custom")
async def custom_docs() -> dict:
    """Human-readable API guide grouped by category, with example payloads."""
    sample_user = {
        "user_id": "marcus_webb",
        "name": "Marcus Webb",
        "age": 41,
        "country": "US",
        "base_currency": "USD",
        "risk_profile": "moderate",
        "positions": [
            {"ticker": "AAPL", "quantity": 50, "avg_cost": 150.0, "currency": "USD"},
            {"ticker": "MSFT", "quantity": 30, "avg_cost": 300.0, "currency": "USD"},
        ],
        "preferences": {"preferred_benchmark": "S&P 500"},
    }
    sample_chat = {
        "query": "How is my portfolio doing?",
        "user": sample_user,
        "session_id": "demo-session",
    }

    return {
        "title": "Valura AI API Guide",
        "version": app.version,
        "base_url": "http://localhost:8000",
        "interactive_docs": "/docs",
        "categories": {
            "chat": {
                "description": "Main conversational entry point — SSE-streamed multi-agent pipeline.",
                "endpoints": [
                    {
                        "method": "POST",
                        "path": "/chat",
                        "example": sample_chat,
                        "notes": "SSE stream: delta → result → done.",
                    }
                ],
            },
            "portfolio": {
                "description": "8 portfolio-analysis endpoints, all backed by PortfolioHealthAgent.",
                "endpoints": [
                    {"method": "POST", "path": "/portfolio/health", "example": sample_chat},
                    {"method": "POST", "path": "/portfolio/concentration", "example": sample_chat},
                    {"method": "POST", "path": "/portfolio/performance", "example": sample_chat},
                    {"method": "POST", "path": "/portfolio/benchmark", "example": sample_chat},
                    {"method": "POST", "path": "/portfolio/rebalance", "example": sample_chat},
                    {"method": "POST", "path": "/portfolio/tax-loss", "example": sample_chat},
                    {"method": "POST", "path": "/portfolio/sector-exposure", "example": sample_chat},
                    {"method": "POST", "path": "/portfolio/dividends", "example": sample_chat},
                ],
            },
            "market": {
                "description": "6 market-research endpoints — yfinance MCP + LLM observations.",
                "endpoints": [
                    {
                        "method": "POST",
                        "path": "/market/snapshot",
                        "example": {"tickers": ["AAPL", "MSFT"], "user": sample_user},
                    },
                    {
                        "method": "POST",
                        "path": "/market/fundamentals",
                        "example": {"tickers": ["NVDA"], "user": sample_user},
                    },
                    {
                        "method": "POST",
                        "path": "/market/technical",
                        "example": {"tickers": ["TSLA"], "user": sample_user},
                    },
                    {
                        "method": "POST",
                        "path": "/market/compare",
                        "example": {"tickers": ["AAPL", "MSFT", "GOOGL"], "user": sample_user},
                    },
                    {
                        "method": "POST",
                        "path": "/market/options",
                        "example": {"tickers": ["SPY"], "user": sample_user},
                    },
                    {
                        "method": "POST",
                        "path": "/market/screen",
                        "example": {"tickers": ["AAPL", "MSFT", "NVDA", "AMZN"], "user": sample_user},
                    },
                ],
            },
            "risk": {
                "description": "5 risk-analysis endpoints — VaR, stress, correlation, volatility.",
                "endpoints": [
                    {"method": "POST", "path": "/risk/full", "example": sample_chat},
                    {"method": "POST", "path": "/risk/var", "example": sample_chat},
                    {"method": "POST", "path": "/risk/stress-test", "example": sample_chat},
                    {"method": "POST", "path": "/risk/correlation", "example": sample_chat},
                    {"method": "POST", "path": "/risk/volatility", "example": sample_chat},
                ],
            },
            "news": {
                "description": "5 news endpoints — ticker, market, sentiment, sector, calendar.",
                "endpoints": [
                    {
                        "method": "POST",
                        "path": "/news/tickers",
                        "example": {
                            "tickers": ["AAPL"], "topics": [], "sectors": [], "user": sample_user,
                        },
                    },
                    {
                        "method": "POST",
                        "path": "/news/market",
                        "example": {"tickers": [], "topics": [], "sectors": [], "user": sample_user},
                    },
                    {
                        "method": "POST",
                        "path": "/news/sentiment",
                        "example": {
                            "tickers": ["NVDA", "AAPL"], "topics": [], "sectors": [], "user": sample_user,
                        },
                    },
                    {
                        "method": "POST",
                        "path": "/news/economic-calendar",
                        "example": None,
                        "notes": "No body required.",
                    },
                    {
                        "method": "POST",
                        "path": "/news/sector",
                        "example": {
                            "tickers": [], "topics": [], "sectors": ["Technology", "Energy"],
                            "user": sample_user,
                        },
                    },
                ],
            },
            "calculator": {
                "description": "4 pure-math endpoints — pass-through to calculator_mcp.",
                "endpoints": [
                    {
                        "method": "POST",
                        "path": "/calculate/compound-interest",
                        "example": {
                            "principal": 10000, "annual_rate": 0.07, "years": 20,
                            "compounds_per_year": 12, "monthly_contribution": 500,
                        },
                    },
                    {
                        "method": "POST",
                        "path": "/calculate/dca",
                        "example": {
                            "monthly_amount": 1000, "annual_return_pct": 8.0,
                            "years": 30, "initial_investment": 5000,
                        },
                    },
                    {
                        "method": "POST",
                        "path": "/calculate/retirement",
                        "example": {
                            "current_age": 41, "retirement_age": 65,
                            "current_savings": 250000, "monthly_contribution": 2000,
                            "expected_annual_return": 7.0,
                            "monthly_expenses_in_retirement": 6500,
                        },
                    },
                    {
                        "method": "POST",
                        "path": "/calculate/options-price",
                        "example": {
                            "stock_price": 180, "strike_price": 185,
                            "time_to_expiry_days": 30, "volatility_pct": 28.0,
                            "risk_free_rate": 0.0525, "option_type": "call",
                        },
                    },
                ],
            },
            "report": {
                "description": "3 report endpoints — markdown or PDF artefacts in /reports.",
                "endpoints": [
                    {
                        "method": "POST",
                        "path": "/report/portfolio",
                        "example": {"user": sample_user, "format": "markdown"},
                    },
                    {
                        "method": "POST",
                        "path": "/report/market",
                        "example": {"tickers": ["AAPL", "NVDA"], "format": "markdown"},
                    },
                    {
                        "method": "POST",
                        "path": "/report/risk",
                        "example": {"user": sample_user, "format": "markdown"},
                    },
                ],
            },
            "system": {
                "description": "Service introspection — health, agents, runtime metrics, UI.",
                "endpoints": [
                    {"method": "GET", "path": "/", "notes": "Single-page dashboard."},
                    {"method": "GET", "path": "/health"},
                    {"method": "GET", "path": "/users"},
                    {"method": "GET", "path": "/agents"},
                    {"method": "GET", "path": "/metrics"},
                    {"method": "GET", "path": "/docs-custom"},
                ],
            },
        },
    }
