"""
Skeleton test for the Portfolio Health agent.

Uses ``MockLLMClient`` for the single observation-generation LLM call.
Live prices come from yfinance (requires network in CI).
"""
import json

import pytest

from src.agents.portfolio_health import run
from src.llm.mock_llm import MockLLMClient


@pytest.mark.asyncio
async def test_portfolio_health_does_not_crash_on_empty_portfolio(load_user):
    """
    user_004 has no positions. Agent must not crash (no LLM call on this path).
    """
    user = load_user("usr_004")
    response = await run(user, MockLLMClient())

    assert response is not None
    data = response.model_dump()
    assert data["disclaimer"]
    assert data["build_guidance"]


@pytest.mark.asyncio
async def test_portfolio_health_flags_concentration(load_user):
    """
    user_003 has ~60% in NVDA. Agent must surface elevated concentration.
    """
    obs = [
        {
            "severity": "warning",
            "text": (
                "NVDA represents a large share of market value — trim if it breaches your single-stock limit."
            ),
        },
        {
            "severity": "info",
            "text": "Overall portfolio value combines five holdings across equities and bonds.",
        },
    ]
    llm = MockLLMClient(responses=[json.dumps(obs)])
    user = load_user("usr_003")
    response = await run(user, llm)

    data = response.model_dump()
    assert data["concentration_risk"]["flag"] == "high"


@pytest.mark.asyncio
async def test_portfolio_health_includes_disclaimer(load_user):
    llm = MockLLMClient(
        responses=[
            json.dumps(
                [
                    {
                        "severity": "info",
                        "text": "Nine equity holdings span mega-cap tech and QQQ with roughly balanced weights.",
                    }
                ]
            )
        ]
    )
    user = load_user("usr_001")
    response = await run(user, llm)
    assert response.disclaimer
    assert "does not constitute investment advice" in response.disclaimer.lower()
