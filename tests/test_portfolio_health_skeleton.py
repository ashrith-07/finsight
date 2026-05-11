"""
Portfolio Health agent tests.

Uses ``MockLLMClient`` for the observation-generation LLM call.
Live prices come from yfinance (requires network in CI).
"""
import pytest

from src.agents.portfolio_health import _ObservationList, run
from src.llm.mock_llm import MockLLMClient
from src.models import Observation


def make_portfolio_mock() -> MockLLMClient:
    obs_list = _ObservationList(
        observations=[
            Observation(severity="warning", text="Test warning observation."),
            Observation(severity="info", text="Test info observation."),
        ]
    )
    return MockLLMClient(responses=[obs_list])


@pytest.mark.asyncio
async def test_portfolio_health_does_not_crash_on_empty_portfolio(load_user):
    """
    user_004 has no positions. Agent must not crash (no LLM call on this path).
    """
    user = load_user("usr_004")
    response = await run(user, make_portfolio_mock())

    assert response is not None
    data = response.model_dump()
    assert data["disclaimer"]
    assert data["build_guidance"]


@pytest.mark.asyncio
async def test_portfolio_health_flags_concentration(load_user):
    """
    user_003 has ~60% in NVDA. Agent must surface elevated concentration.
    """
    mock = make_portfolio_mock()
    user = load_user("usr_003")
    response = await run(user, mock)

    data = response.model_dump()
    assert data["concentration_risk"]["flag"] == "high"


@pytest.mark.asyncio
async def test_portfolio_health_includes_disclaimer(load_user):
    mock = make_portfolio_mock()
    user = load_user("usr_001")
    response = await run(user, mock)
    assert response.disclaimer
    assert "does not constitute investment advice" in response.disclaimer.lower()


@pytest.mark.asyncio
async def test_portfolio_health_multi_currency(load_user):
    """user_006 has EUR, GBP, JPY positions — must not crash and must return USD values."""
    mock = make_portfolio_mock()
    user = load_user("usr_006")
    result = await run(user, llm=mock)
    assert result is not None
    assert result.performance.current_value_total > 0
    assert result.disclaimer
