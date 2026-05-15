"""Dispatch ``ClassifierResult`` through ``FinsightOrchestrator`` or taxonomy stubs."""

from __future__ import annotations

import os

from src.agents.stub import StubAgent
from src.llm.base import LLMClient
from src.logging_config import get_logger
from src.models import AgentResponse, ClassifierResult
from src.orchestrator import FinsightAgnoTeam, FinsightOrchestrator

logger = get_logger("router")

ORCHESTRATED = frozenset(
    {
        "portfolio_health",
        "market_research",
        "risk_assessment",
        "financial_news",
        "report_generator",
    }
)

# Soft remaps: classifier categories that should reach a concrete agent.
NEWS_INTENTS = frozenset({"predictive_analysis"})

REPORT_INTENTS = frozenset({"financial_planning"})


def _use_agno_multiagent_team() -> bool:
    """Groq free-tier TPM (~6000/min) cannot sustain coordinate-mode teams; prefer orchestrator.

    Set ``FINSIGHT_USE_AGNO_TEAM=1`` to force the Agno team when using Groq only.
    With OpenAI configured, the team is used unless ``FINSIGHT_USE_AGNO_TEAM=0``.
    """
    raw = os.environ.get("FINSIGHT_USE_AGNO_TEAM", "").strip().lower()
    groq_only = bool(os.environ.get("GROQ_API_KEY", "").strip()) and not bool(
        os.environ.get("OPENAI_API_KEY", "").strip()
    )
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return not groq_only


class AgentRouter:
    """Primary path: ``FinsightAgnoTeam``. Falls back to ``FinsightOrchestrator`` automatically when no LLM key is configured."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._orchestrator = FinsightOrchestrator(llm)
        self._team = FinsightAgnoTeam(llm)
        self._stub = StubAgent()

    async def route(
        self,
        classifier_result: ClassifierResult,
        user: dict,
        query: str = "",
    ) -> AgentResponse:
        agent = classifier_result.agent
        intent = classifier_result.intent
        entities = classifier_result.entities

        use_team = _use_agno_multiagent_team()

        async def _dispatch_team(cr: ClassifierResult) -> AgentResponse:
            return await self._team.run(cr, user, query=query)

        async def _dispatch_orchestrator(cr: ClassifierResult) -> AgentResponse:
            return await self._orchestrator.run(cr, user, query=query)

        try:
            if agent in ORCHESTRATED:
                if use_team:
                    return await _dispatch_team(classifier_result)
                logger.info(
                    "router_orchestrator_path",
                    extra={"agent": agent, "reason": "groq_only_default"},
                )
                return await _dispatch_orchestrator(classifier_result)

            if agent in NEWS_INTENTS:
                remapped = classifier_result.model_copy(update={"agent": "financial_news"})
                if use_team:
                    return await _dispatch_team(remapped)
                return await _dispatch_orchestrator(remapped)

            if agent in REPORT_INTENTS:
                remapped = classifier_result.model_copy(update={"agent": "report_generator"})
                if use_team:
                    return await _dispatch_team(remapped)
                return await _dispatch_orchestrator(remapped)

            return await self._stub.run(agent, intent, entities)
        except Exception:
            logger.exception("AgentRouter.route unexpected failure for agent=%s", agent)
            return AgentResponse(
                agent=agent or "general_query",
                implemented=False,
                intent=intent,
                entities=entities,
                result=None,
                message="Something went wrong routing your request. Please try again.",
            )

    async def route_and_wrap(
        self,
        classifier_result: ClassifierResult,
        user: dict,
        query: str = "",
    ) -> AgentResponse:
        response = await self.route(classifier_result, user, query=query)
        if response.result is None and response.agent == "portfolio_health":
            if not (response.message or "").strip():
                response = response.model_copy(
                    update={
                        "message": (
                            "Portfolio analysis encountered an error. Please try again."
                        )
                    }
                )
        return response
