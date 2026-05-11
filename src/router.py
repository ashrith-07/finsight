"""Dispatch ``ClassifierResult`` to portfolio health or taxonomy stubs."""

from __future__ import annotations

import logging

from src.agents.portfolio_health import PortfolioHealthAgent
from src.agents.stub import StubAgent
from src.llm.base import LLMClient
from src.models import AgentResponse, ClassifierResult, PortfolioHealthResult

logger = logging.getLogger(__name__)


def _portfolio_summary_message(result: PortfolioHealthResult) -> str:
    if result.observations:
        return result.observations[0].text
    if result.build_guidance:
        return result.build_guidance
    return "Portfolio health analysis is ready below."


class AgentRouter:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._portfolio_health = PortfolioHealthAgent(llm)
        self._stub = StubAgent()

    async def route(
        self,
        classifier_result: ClassifierResult,
        user: dict,
    ) -> AgentResponse:
        agent = classifier_result.agent
        intent = classifier_result.intent
        entities = classifier_result.entities

        try:
            if agent == "portfolio_health":
                try:
                    result = await self._portfolio_health.run(user)
                    return AgentResponse(
                        agent=agent,
                        implemented=True,
                        intent=intent,
                        entities=entities,
                        result=result,
                        message=_portfolio_summary_message(result),
                    )
                except Exception:
                    logger.exception("PortfolioHealthAgent.run failed")
                    return AgentResponse(
                        agent=agent,
                        implemented=True,
                        intent=intent,
                        entities=entities,
                        result=None,
                        message=(
                            "Portfolio analysis encountered an error. Please try again."
                        ),
                    )

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
    ) -> AgentResponse:
        response = await self.route(classifier_result, user)
        if response.result is None and response.agent == "portfolio_health":
            # ``route`` can leave ``message`` blank on the error path
            if not (response.message or "").strip():
                response = response.model_copy(
                    update={
                        "message": (
                            "Portfolio analysis encountered an error. Please try again."
                        )
                    }
                )
        return response
