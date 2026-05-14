"""Dispatch ``ClassifierResult`` through ``ValuraOrchestrator`` or taxonomy stubs."""

from __future__ import annotations

from src.agents.stub import StubAgent
from src.llm.base import LLMClient
from src.logging_config import get_logger
from src.models import AgentResponse, ClassifierResult
from src.orchestrator import ValuraAgnoTeam, ValuraOrchestrator

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


class AgentRouter:
    """Primary path: ``ValuraAgnoTeam``. Falls back to ``ValuraOrchestrator`` automatically when no LLM key is configured."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._orchestrator = ValuraOrchestrator(llm)
        self._team = ValuraAgnoTeam(llm)
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

        try:
            if agent in ORCHESTRATED:
                return await self._team.run(classifier_result, user, query=query)

            if agent in NEWS_INTENTS:
                remapped = classifier_result.model_copy(update={"agent": "financial_news"})
                return await self._team.run(remapped, user, query=query)

            if agent in REPORT_INTENTS:
                remapped = classifier_result.model_copy(update={"agent": "report_generator"})
                return await self._team.run(remapped, user, query=query)

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
