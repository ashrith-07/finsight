"""Shared Pydantic models for API boundaries and internal handoffs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SafetyVerdict(BaseModel):
    """When ``blocked`` is True the pipeline stops before any agent or LLM."""

    model_config = ConfigDict(extra="forbid")

    blocked: bool
    category: str
    message: str


class Entity(BaseModel):
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
    """``safety_verdict`` is informational only; ``src.safety.check`` is authoritative."""

    model_config = ConfigDict(extra="forbid")

    intent: str
    agent: str
    entities: Entity
    safety_verdict: str
    confidence: float = 1.0


class ConcentrationRisk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_position_pct: float
    top_3_positions_pct: float
    flag: str


class Performance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_return_pct: float
    annualized_return_pct: float
    cost_basis_total: float
    current_value_total: float


class BenchmarkComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark: str
    portfolio_return_pct: float
    benchmark_return_pct: float
    alpha_pct: float


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: str
    text: str


class PortfolioHealthResult(BaseModel):
    """``build_guidance`` is set only for zero-position (BUILD) flows."""

    model_config = ConfigDict(extra="forbid")

    concentration_risk: ConcentrationRisk
    performance: Performance
    benchmark_comparison: BenchmarkComparison
    observations: list[Observation]
    disclaimer: str
    build_guidance: str | None = None


class AgentResponse(BaseModel):
    """``implemented`` distinguishes real agents from stubs; ``result`` may be a dict for forwards compat."""

    model_config = ConfigDict(extra="forbid")

    agent: str
    implemented: bool
    intent: str
    entities: Entity
    result: PortfolioHealthResult | dict | None
    message: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    user: dict
    session_id: str = "default"


class SSEEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: str
    data: str
