"""Placeholder agent for taxonomy slots not yet implemented."""

from __future__ import annotations

import logging

from src.models import AgentResponse, Entity

logger = logging.getLogger(__name__)


class StubAgent:
    """
    Returns a structured 'not implemented' response for all agents except portfolio_health.

    Must never raise. Must never return an HTTP error (callers map this to 200 + payload).
    """

    NOT_IMPLEMENTED_MESSAGES = {
        "market_research": (
            "Market Research agent is coming soon. I can see you're asking about {entities} — "
            "our research team is building this capability."
        ),
        "investment_strategy": (
            "Investment Strategy agent is in development. Your query about {intent} has been noted."
        ),
        "financial_planning": (
            "Financial Planning agent is in development. Long-term goal planning is a core part of our roadmap."
        ),
        "financial_calculator": (
            "Financial Calculator agent is coming soon. For now, I can route your calculation request "
            "to a human advisor."
        ),
        "risk_assessment": (
            "Risk Assessment agent is in development. We'll be able to model scenarios for your portfolio soon."
        ),
        "product_recommendation": "Product Recommendation agent is in development.",
        "predictive_analysis": "Predictive Analysis agent is in development.",
        "customer_support": (
            "For platform support, please contact support@valura.ai or visit our help centre."
        ),
        "general_query": (
            "I'm here to help with financial questions. Could you be more specific about what you'd like to know?"
        ),
    }

    async def run(self, agent: str, intent: str, entities: Entity) -> AgentResponse:
        try:
            message_template = self.NOT_IMPLEMENTED_MESSAGES.get(
                agent, "This agent is not yet available."
            )
            entity_summary = self._summarize_entities(entities)
            message = message_template.format(entities=entity_summary, intent=intent)
            return AgentResponse(
                agent=agent,
                implemented=False,
                intent=intent,
                entities=entities,
                result=None,
                message=message,
            )
        except Exception:
            logger.exception("StubAgent.run failed for agent=%s", agent)
            return AgentResponse(
                agent=agent,
                implemented=False,
                intent=intent,
                entities=entities,
                result=None,
                message="This capability is not available yet. Please try again later.",
            )

    def _summarize_entities(self, entities: Entity) -> str:
        parts: list[str] = []
        if entities.tickers:
            parts.append(", ".join(entities.tickers))
        if entities.topics:
            parts.append(", ".join(entities.topics))
        return " and ".join(parts) if parts else "your request"
