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
    """``build_guidance`` is set only for zero-position (BUILD) flows.

    ``sub_intent`` and ``extras`` carry sub-intent-specific results
    (e.g. tax-loss replacement candidates, sector concentration breakdowns).
    """

    model_config = ConfigDict(extra="forbid")

    concentration_risk: ConcentrationRisk
    performance: Performance
    benchmark_comparison: BenchmarkComparison
    observations: list[Observation]
    disclaimer: str
    build_guidance: str | None = None
    sub_intent: str | None = None
    extras: dict | None = None


class PriceSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str
    current_price: float
    currency: str
    day_open: float
    day_high: float
    day_low: float
    previous_close: float
    day_change_pct: float
    volume: int
    market_cap: float | None
    fifty_two_week_high: float
    fifty_two_week_low: float
    distance_from_52w_high_pct: float


class CompanyInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    sector: str | None
    industry: str | None
    country: str | None
    summary: str | None


class MarketResearchResult(BaseModel):
    """``sub_intent`` + ``extras`` carry focused payloads (fundamentals,
    technical levels, options, etc.) for narrow sub-intents."""

    model_config = ConfigDict(extra="forbid")

    tickers: list[str]
    snapshots: list[PriceSnapshot]
    company_info: list[CompanyInfo]
    observations: list[Observation]
    comparison_note: str | None = None
    disclaimer: str
    sub_intent: str | None = None
    extras: dict | None = None


class ExecutionMetadata(BaseModel):
    """Per-request execution telemetry — surfaces parallel agent timings to the UI.

    ``timings`` maps agent_name → milliseconds. ``wall_time_ms`` is the total
    awaited time; ``sequential_estimate_ms`` is the naive sum (i.e. what the
    same work would have cost run one-after-another). ``saved_ms`` highlights
    the parallelism win.
    """

    model_config = ConfigDict(extra="forbid")

    agents_ran: list[str] = Field(default_factory=list)
    timings: dict[str, int] = Field(default_factory=dict)
    parallel: bool = False
    wall_time_ms: int = 0
    sequential_estimate_ms: int = 0
    saved_ms: int = 0


class AgentResponse(BaseModel):
    """``implemented`` distinguishes real agents from stubs; ``result`` may be a dict for forwards compat."""

    model_config = ConfigDict(extra="forbid")

    agent: str
    implemented: bool
    intent: str
    entities: Entity
    result: PortfolioHealthResult | MarketResearchResult | dict | None
    message: str
    execution_metadata: ExecutionMetadata | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    user: dict
    session_id: str = "default"


class SSEEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: str
    data: str
