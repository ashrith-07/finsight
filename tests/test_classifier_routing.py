"""
Classifier routing and entity extraction tests using deterministic mocks.

Entity matcher rules align with fixtures/README.md.
"""
from typing import Any

import pytest

from src.classifier import classify
from src.llm.mock_llm import MockLLMClient
from src.models import ClassifierResult, Entity


def make_mock_for_case(case: dict) -> MockLLMClient:
    """Build a MockLLMClient that returns the expected ClassifierResult for this case."""
    expected_result = ClassifierResult(
        intent=case["expected_agent"],
        agent=case["expected_agent"],
        entities=Entity(**case.get("expected_entities", {})),
        safety_verdict="clean",
        confidence=1.0,
    )
    return MockLLMClient.for_classifier(expected_result)


# ---------------------------------------------------------------------------
# Entity matcher — implements the rules in fixtures/README.md
# ---------------------------------------------------------------------------

def _normalize_ticker(t: str) -> str:
    """Case-fold and drop the exchange suffix (AAPL.US → AAPL)."""
    return t.upper().split(".")[0]


def matches_entities(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """
    Subset match with normalization. `actual` must contain every value in
    `expected`; extra fields and extra values are allowed.
    """
    for field, exp_value in expected.items():
        act_value = actual.get(field)
        if act_value is None:
            return False

        if field == "tickers":
            exp_set = {_normalize_ticker(t) for t in exp_value}
            act_set = {_normalize_ticker(t) for t in act_value}
            if not exp_set.issubset(act_set):
                return False
        elif field in ("topics", "sectors"):
            exp_set = {s.lower() for s in exp_value}
            act_set = {s.lower() for s in act_value}
            if not exp_set.issubset(act_set):
                return False
        elif field in ("amount", "rate"):
            if abs(act_value - exp_value) > abs(exp_value) * 0.05:
                return False
        elif field == "period_years":
            if int(act_value) != int(exp_value):
                return False
        else:
            if str(act_value).lower() != str(exp_value).lower():
                return False
    return True


@pytest.mark.asyncio
async def test_classifier_routing_accuracy(gold_classifier_queries):
    """
    With per-case mocks we expect 100% routing accuracy (parser wiring).
    """
    correct = 0
    for case in gold_classifier_queries:
        mock = make_mock_for_case(case)
        result = await classify(case["query"], llm=mock)
        if result.agent == case["expected_agent"]:
            correct += 1

    accuracy = correct / len(gold_classifier_queries)
    assert accuracy == 1.0, f"Routing accuracy {accuracy:.2%} expected 100% with deterministic mocks"


@pytest.mark.asyncio
async def test_classifier_entity_extraction(gold_classifier_queries):
    """
    Soft signal — not a hard threshold. Reported, not failed on.
    """
    matched = 0
    total_with_entities = 0
    for case in gold_classifier_queries:
        if not case["expected_entities"]:
            continue
        total_with_entities += 1
        mock = make_mock_for_case(case)
        result = await classify(case["query"], llm=mock)
        if matches_entities(result.entities.model_dump(), case["expected_entities"]):
            matched += 1

    rate = matched / total_with_entities if total_with_entities else 0.0
    print(f"\nEntity match rate: {rate:.2%} ({matched}/{total_with_entities})")


@pytest.mark.asyncio
async def test_classifier_handles_followup():
    """Classifier must pass prior turns to LLM and return a valid result."""
    from src.classifier import IntentClassifier

    expected = ClassifierResult(
        intent="portfolio query",
        agent="portfolio_health",
        entities=Entity(tickers=["NVDA"]),
        safety_verdict="clean",
        confidence=0.9,
    )
    mock = MockLLMClient.for_classifier(expected)
    classifier = IntentClassifier(mock)

    result = await classifier.classify(
        query="How much do I own?",
        prior_user_turns=["What's happening with Nvidia this week?"],
    )
    assert result.agent == "portfolio_health"
    assert "NVDA" in result.entities.tickers
