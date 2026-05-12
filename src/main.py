"""FastAPI app: safety → classify → route → SSE. Default timeout 30s via ``PIPELINE_TIMEOUT``."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse

from src.classifier import classify
from src.llm import get_llm_client
from src.models import AgentResponse, ChatRequest, PortfolioHealthResult
from src.router import AgentRouter
from src.safety import check as safety_check
from src.session import ConversationTurn, query_cache, session_store

load_dotenv()


def _pipeline_timeout_s() -> float:
    raw = os.environ.get("PIPELINE_TIMEOUT", "30")
    try:
        return float(raw)
    except ValueError:
        return 30.0


PIPELINE_TIMEOUT = _pipeline_timeout_s()

logger = logging.getLogger("valura.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level_name, logging.INFO)
    if not logging.root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    else:
        logging.getLogger().setLevel(level)
    logger.info(
        "Valura AI multi-agent ecosystem starting | "
        "agents=5 | mcp_servers=3 | orchestrator=ValuraOrchestrator"
    )
    logger.info(
        "service_startup",
        extra={
            "event": "lifecycle",
            "phase": "startup",
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


def _sse(event: str, data: str) -> dict[str, str]:
    return {"event": event, "data": data}


def _build_summary(response: AgentResponse) -> str:
    if response.agent == "portfolio_health" and response.result is not None:
        result = response.result
        if isinstance(result, PortfolioHealthResult):
            observations = result.observations[:2]
            obs_text = " ".join(o.text for o in observations)
            return f"{obs_text} {result.disclaimer}".strip()
        if isinstance(result, dict):
            nested = result.get("portfolio_health")
            if isinstance(nested, dict):
                observations = nested.get("observations") or []
                parts: list[str] = []
                for o in observations[:2]:
                    if isinstance(o, dict):
                        parts.append(str(o.get("text") or ""))
                    elif hasattr(o, "text"):
                        parts.append(str(getattr(o, "text")))
                obs_text = " ".join(parts)
                disc = str(nested.get("disclaimer") or "")
                summ = str(result.get("ecosystem_summary") or "").strip()
                base = f"{obs_text} {disc}".strip()
                return f"{summ} {base}".strip() if summ else base
            observations = result.get("observations") or []
            parts = []
            for o in observations[:2]:
                if isinstance(o, dict):
                    parts.append(str(o.get("text") or ""))
                elif hasattr(o, "text"):
                    parts.append(str(getattr(o, "text")))
            obs_text = " ".join(parts)
            disc = str(result.get("disclaimer") or "")
            return f"{obs_text} {disc}".strip()
    return response.message


@app.post("/chat")
async def chat(request: ChatRequest) -> EventSourceResponse:
    """SSE only: ``delta`` (summary chunks), ``result`` (JSON), optional ``error``, trailing ``done``."""
    llm = get_llm_client()
    router = AgentRouter(llm)

    async def event_generator():
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

                t_agent_start = time.monotonic()
                agent_response = await router.route(classifier_result, request.user)
                t_agent = time.monotonic() - t_agent_start

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


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "valura-ai"}


@app.get("/agents")
async def list_agents() -> dict:
    """Lists agents in the ecosystem with coarse status."""
    return {
        "total_agents": 5,
        "orchestrator": "ValuraOrchestrator",
        "mcp_servers": ["yfinance_mcp", "web_search_mcp", "report_mcp"],
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
