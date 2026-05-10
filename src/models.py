"""
Shared Pydantic models for the Valura AI pipeline.

These types cross-cut safety screening, intent classification, agent dispatch,
structured agent outputs, and the HTTP/SSE boundary. They are the single source
of truth for request/response contracts and internal handoffs.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SafetyVerdict(BaseModel):
    """
    Outcome of the synchronous safety layer before any LLM agent runs.

    When ``blocked`` is true, downstream agents must not execute; the
    ``message`` is shown to the user. ``category`` labels the policy bucket for
    logging and analytics.
    """

    model_config = ConfigDict(extra="forbid")

    blocked: bool
    category: str
    message: str


class Entity(BaseModel):
    """
    Structured slots extracted from the user query by the classifier.

    Used for routing hints, agent prompts, and entity-aware responses. Scalar
    fields are optional when not mentioned; list fields default to empty.
    """

    model_config = ConfigDict(extra="forbid")

    tickers: list[str] = Field(default_factory=list)
    amount: float | None = None
    currency: str | None = None
    rate: float | None = None
    period_years: int | None = None
    frequency: str | None = None
    horizon: str | None = None
    time_period: str | None = None
    topics: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    index: str | None = None
    action: str | None = None
    goal: str | None = None


class ClassifierResult(BaseModel):
    """
    Single structured output from the intent classifier LLM call.

    Ties natural language to an ``agent`` id, populated ``entities``, and an
    informational ``safety_verdict`` string (not authoritative vs. the guard).
    """

    model_config = ConfigDict(extra="forbid")

    intent: str
    agent: str
    entities: Entity
    safety_verdict: str
    confidence: float = 1.0


class ConcentrationRisk(BaseModel):
    """Position sizing risk view used inside portfolio health reporting."""

    model_config = ConfigDict(extra="forbid")

    top_position_pct: float
    top_3_positions_pct: float
    flag: str


class Performance(BaseModel):
    """Aggregate performance figures for the user's portfolio."""

    model_config = ConfigDict(extra="forbid")

    total_return_pct: float
    annualized_return_pct: float
    cost_basis_total: float
    current_value_total: float


class BenchmarkComparison(BaseModel):
    """Relative performance vs. a named benchmark over the same window."""

    model_config = ConfigDict(extra="forbid")

    benchmark: str
    portfolio_return_pct: float
    benchmark_return_pct: float
    alpha_pct: float


class Observation(BaseModel):
    """A single human-readable insight with severity for UI prioritization."""

    model_config = ConfigDict(extra="forbid")

    severity: str
    text: str


class PortfolioHealthResult(BaseModel):
    """
    Canonical structured payload returned by the portfolio health agent.

    Combines risk, performance, benchmark context, narrative observations, and
    compliance copy. ``build_guidance`` is only set when the portfolio is empty
    and the agent should steer the user toward a first allocation.
    """

    model_config = ConfigDict(extra="forbid")

    concentration_risk: ConcentrationRisk
    performance: Performance
    benchmark_comparison: BenchmarkComparison
    observations: list[Observation]
    disclaimer: str
    build_guidance: str | None = None


class AgentResponse(BaseModel):
    """
    Wrapper returned by any specialist agent to the router/streaming layer.

    ``implemented`` distinguishes real logic from stubs. ``result`` may be a
    typed ``PortfolioHealthResult`` or a loose ``dict`` for forward-compatible
    agents; ``message`` is always safe to show as a summary.
    """

    model_config = ConfigDict(extra="forbid")

    agent: str
    implemented: bool
    intent: str
    entities: Entity
    result: PortfolioHealthResult | dict | None
    message: str


class ChatRequest(BaseModel):
    """
    Inbound chat turn from the API: natural language plus embedded user profile.

    ``user`` is passed through as loaded from fixtures or upstream services.
    ``session_id`` scopes conversation memory for follow-ups.
    """

    model_config = ConfigDict(extra="forbid")

    query: str
    user: dict
    session_id: str = "default"


class SSEEvent(BaseModel):
    """
    One Server-Sent Event frame emitted while streaming a reply.

    ``event`` selects client-side handlers; ``data`` is typically JSON text.
    """

    model_config = ConfigDict(extra="forbid")

    event: str
    data: str
