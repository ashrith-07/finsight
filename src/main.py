"""
Valura AI FastAPI HTTP layer: safety → classifier → router → SSE-only responses.

Pipeline timeout (``PIPELINE_TIMEOUT``) is **30 seconds**: long enough for one
LLM classification plus agent work and token-ish streaming over typical networks,
but short enough that hung upstream calls or stalled generators do not tie up
workers indefinitely (better UX than infinite waits; tune per deployment).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse

from src.classifier import classify
from src.llm import get_llm_client
from src.models import AgentResponse, ChatRequest, PortfolioHealthResult
from src.router import AgentRouter
from src.safety import check as safety_check
from src.session import ConversationTurn, query_cache, session_store

PIPELINE_TIMEOUT = 30  # seconds — see module docstring for rationale

logger = logging.getLogger("valura.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not logging.root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
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
    """
    Plain-language text for streaming deltas.

    Portfolio health: top observations + disclaimer.
    Stubs: ``AgentResponse.message``.
    """
    if response.agent == "portfolio_health" and response.result is not None:
        result = response.result
        if isinstance(result, PortfolioHealthResult):
            observations = result.observations[:2]
            obs_text = " ".join(o.text for o in observations)
            return f"{obs_text} {result.disclaimer}".strip()
        if isinstance(result, dict):
            observations = result.get("observations") or []
            parts: list[str] = []
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
    """
    Full pipeline: safety → classifier → router → SSE stream.

    Streams SSE events:

    - ``delta`` — streamed summary text chunks
    - ``result`` — final structured ``AgentResponse`` JSON
    - ``error`` — client-safe JSON error (no stack traces)
    - ``done`` — stream terminator (always last)
    """
    llm = get_llm_client()
    router = AgentRouter(llm)

    async def event_generator():
        start = time.monotonic()
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
                verdict = safety_check(request.query)
                if verdict.blocked:
                    yield _sse("delta", verdict.message)
                    yield _sse("done", "")
                    elapsed = time.monotonic() - start
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
                if cached is not None:
                    classifier_result = cached
                else:
                    prior_turns = await session_store.get_prior_user_turns(
                        request.session_id
                    )
                    classifier_result = await classify(
                        request.query,
                        llm=llm,
                        prior_user_turns=prior_turns,
                    )
                    query_cache.set(
                        request.session_id, request.query, classifier_result
                    )

                agent_response = await router.route(classifier_result, request.user)

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

                elapsed = time.monotonic() - start
                logger.info(
                    "chat_request_completed",
                    extra={
                        "event": "chat",
                        "phase": "done",
                        "session_id": request.session_id,
                        "agent": agent_response.agent,
                        "implemented": agent_response.implemented,
                        "elapsed_s": round(elapsed, 3),
                    },
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
