# Finsight AI — Multi-Agent Financial Intelligence Ecosystem

A production-grade FastAPI service that turns one user query into a parallel multi-agent run across portfolio analytics, market research, risk modelling, news sentiment and report generation — streamed back as SSE with per-agent timing telemetry.

> **Defence walkthrough:** [Loom recording](https://www.loom.com/share/0bab3c411f05459c8d6bb514b26ad242)

---

## Architecture Overview

```
                                  ┌────────────────────────────────────────────────┐
                                  │              Browser / API client              │
                                  └─────────────────────┬──────────────────────────┘
                                                        │ POST /chat  (SSE)
                                                        ▼
                       ┌─────────────────────────────────────────────────────────────┐
                       │  FastAPI app  (src/main.py)   ─  38 routes, metrics, /docs  │
                       └─────────────────────────────────────────────────────────────┘
                                                        │
                                                        ▼
                       ┌─────────────────────────────────────────────────────────────┐
                       │  Safety Guard   ─  TF-IDF + Logistic Regression  (~0.003ms) │
                       └─────────────────────────────────────────────────────────────┘
                                          │ pass             │ blocked → SSE refusal
                                          ▼
                       ┌─────────────────────────────────────────────────────────────┐
                       │  Two-stage Intent Classifier                                │
                       │   1. TF-IDF + LR pre-classifier (≥0.80 conf → no LLM call) │
                       │   2. LLM fallback for ambiguous queries  (Groq / OpenAI)   │
                       └─────────────────────────────────────────────────────────────┘
                                                        │
                                                        ▼
                       ┌─────────────────────────────────────────────────────────────┐
                       │  AgentRouter   (src/router.py)                              │
                       │  primary: FinsightAgnoTeam (Agno coordinate/route teams)      │
                       │  fallback: FinsightOrchestrator (deterministic asyncio.gather)│
                       │  → auto-falls-back on missing key, RunStatus.error, or     │
                       │    empty/short-error content from the LLM provider.        │
                       └─────────────────────────────────────────────────────────────┘
                                                        │
        ┌───────────────────┬──────────────────┬────────┴────────┬──────────────────┬───────────────────┐
        ▼                   ▼                  ▼                 ▼                  ▼                   ▼
 ┌────────────────┐  ┌──────────────┐   ┌────────────┐    ┌─────────────┐    ┌──────────────┐
 │ portfolio_health│  │market_research│   │ risk_analysis│  │financial_news│   │report_generator│
 └────────┬───────┘  └──────┬───────┘   └─────┬──────┘    └──────┬──────┘    └──────┬───────┘
          │                 │                 │                  │                  │
          ▼                 ▼                 ▼                  ▼                  ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────┐
 │                              MCP toolkits  (src/mcp/*.py)                                │
 │  yfinance_mcp │ web_search_mcp │ report_mcp │ calculator_mcp │ portfolio_analytics_mcp   │
 └──────────────────────────────────────────────────────────────────────────────────────────┘
                                                        │
                                                        ▼
                       ┌─────────────────────────────────────────────────────────────┐
                       │  SSE stream:  delta · result (+ execution_metadata) · done  │
                       └─────────────────────────────────────────────────────────────┘
```

A `/chat` request flows top-to-bottom: safety check → intent classification → router dispatch → either an Agno `Team.arun` (coordinate or route mode) or the deterministic `asyncio.gather` orchestrator → synthesis → SSE stream with per-agent timings attached as `execution_metadata` on the final `result` event. The Agno Team path *always* degrades to the deterministic orchestrator when no LLM key is configured, when Agno returns `RunStatus.error`, or when the content looks like a connection / rate-limit error — so the demo cannot fail because of an upstream LLM blip. The browser UI in `frontend/index.html` consumes the same stream and renders a live timeline of which agents ran in parallel, their durations and the wall-clock saving.

---

## The 5 Agents

| Agent | Use Cases (Sub-intents) | MCP Servers Used | Runs in Parallel With |
|---|---|---|---|
| **portfolio_health** | `full_health_check`, `concentration_only`, `performance_only`, `benchmark_comparison`, `rebalance_suggestion`, `tax_loss_harvesting` | `yfinance_mcp`, `portfolio_analytics_mcp`, `calculator_mcp` | `risk_analysis`, `financial_news` |
| **market_research** | `price_check`, `full_research`, `comparison`, `fundamentals`, `technical_levels`, `options_activity` | `yfinance_mcp` | `financial_news` |
| **risk_analysis** | `full_risk`, `var_only`, `stress_test`, `correlation`, `single_stock_risk`, `volatility_analysis` | `yfinance_mcp`, `web_search_mcp` | `portfolio_health`, `financial_news` |
| **financial_news** | `ticker_news`, `market_news`, `sector_news`, `economic_events`, `sentiment_summary` | `web_search_mcp`, `yfinance_mcp` | every primary agent |
| **report_generator** | `portfolio`, `market`, `risk`, `comparison` | `report_mcp`, `yfinance_mcp`, `web_search_mcp` | runs serially **after** prefetch ecosystem |

Each agent has its own `_detect_sub_intent()` routine — keyword regexes against the classifier's intent + the raw user query — so a single agent endpoint can route to one of several focused implementations without another LLM round-trip.

---

## The 5 MCP Servers

Built on the **Agno** `Toolkit` base — every tool is registered at construction time and discoverable by both Agno-style tool-calling LLMs and the orchestrator's direct Python dispatch.

| Server | Tools | Used By |
|---|---|---|
| **yfinance_mcp** | `get_price_snapshot`, `get_historical_prices`, `get_company_fundamentals`, `get_financial_statements`, `get_options_data`, `screen_stocks` | `portfolio_health`, `market_research`, `risk_analysis`, `financial_news`, `report_generator` |
| **web_search_mcp** | `search_financial_news`, `search_company_news`, `search_market_analysis`, `get_economic_events` | `financial_news`, `risk_analysis`, `report_generator` |
| **report_mcp** | `generate_portfolio_report`, `generate_market_report`, `generate_risk_report` | `report_generator` |
| **calculator_mcp** | `compound_interest`, `dca_projection`, `options_black_scholes`, `loan_amortisation`, `retirement_projection`, `portfolio_rebalance_trades` | `portfolio_health` (rebalance), direct REST |
| **portfolio_analytics_mcp** | `efficient_frontier_point`, `portfolio_beta`, `sector_exposure`, `dividend_analysis`, `geographic_exposure`, `performance_attribution` | `portfolio_health` (concentration, sector, dividends) |

All servers are instantiated as **singletons** in `src/mcp/__init__.py` and reused across agents — no per-request initialisation cost.

---

## The 38 Endpoints

Every focused REST endpoint runs the same agent code as `/chat` but skips SSE streaming and returns a structured JSON payload. Exhaustive list also lives at **`GET /docs-custom`** (machine-readable) and **`GET /docs`** (Swagger UI).

### Chat (1) — streaming multi-agent pipeline
- `POST /chat` — Server-Sent Events stream (`delta` → `result` → `done`); reads `execution_metadata` for parallel timing telemetry.

### Portfolio (8)
- `POST /portfolio/health` — full health check (concentration + performance + benchmark + analytics extras)
- `POST /portfolio/concentration` — top-position + live sector breakdown
- `POST /portfolio/performance` — returns vs cost basis, best / worst positions
- `POST /portfolio/benchmark` — comparison vs S&P 500 / QQQ / preferred benchmark
- `POST /portfolio/rebalance` — exact buy/sell trades from `calculator_mcp.portfolio_rebalance_trades`
- `POST /portfolio/tax-loss` — tax-loss harvest opportunities + wash-sale-safe replacements
- `POST /portfolio/sector-exposure` — sector + geographic breakdown from `portfolio_analytics_mcp`
- `POST /portfolio/dividends` — annual dividend income + monthly cash flow

### Market (6)
- `POST /market/snapshot` — current price, day change, volume
- `POST /market/fundamentals` — P/E, revenue, margins, debt ratios
- `POST /market/technical` — 52-week range, MAs, golden / death cross
- `POST /market/compare` — side-by-side comparison of 2+ tickers
- `POST /market/options` — options activity, put/call ratio, implied vol
- `POST /market/screen` — concurrent metrics for a watchlist, ranked by market cap

### Risk (5)
- `POST /risk/full` — VaR + drawdown + Sharpe + stress + correlations
- `POST /risk/var` — Value at Risk at 95% and 99% confidence
- `POST /risk/stress-test` — five historical crash scenarios
- `POST /risk/correlation` — pairwise correlations above 0.7
- `POST /risk/volatility` — beta, std-dev, Bollinger position per holding

### News (5)
- `POST /news/tickers` — latest news per ticker (with sentiment tags)
- `POST /news/market` — broad market headlines
- `POST /news/sentiment` — aggregate bullish/bearish score
- `POST /news/economic-calendar` — Fed / earnings / macro releases (no body)
- `POST /news/sector` — news grouped by sector

### Calculator (4)
- `POST /calculate/compound-interest` — compound growth + monthly contributions
- `POST /calculate/dca` — dollar-cost-averaging projection
- `POST /calculate/retirement` — retirement readiness check (4% rule)
- `POST /calculate/options-price` — Black-Scholes price + Greeks

### Report (3)
- `POST /report/portfolio` — markdown / PDF portfolio report → saved under `/reports/`
- `POST /report/market` — market research report for one or more tickers
- `POST /report/risk` — risk analysis report

### System (6)
- `GET /` — single-page dashboard (`frontend/index.html`)
- `GET /health` — liveness probe
- `GET /users` — bundled fixture profiles for the UI selector
- `GET /agents` — agents + MCP servers registry
- `GET /metrics` — live counters (`total_requests_served`, `parallel_execution_count`, `average_response_time_ms`, `requests_per_agent`, …)
- `GET /docs-custom` — human-readable API guide as JSON

---

## Agno Team Coordination

Two native Agno `Team` instances live in `src/orchestrator.py::FinsightAgnoTeam`:

| Team | Mode | Members | Used for |
|---|---|---|---|
| `portfolio_team` | `coordinate` | `portfolio_agent`, `risk_agent`, `news_agent` | `portfolio_health`, `risk_assessment` |
| `research_team` | `route` | `market_agent`, `news_agent`, `report_agent` | `market_research`, `predictive_analysis` |

`coordinate` runs all members in parallel and synthesises their outputs through a strong leader model; `route` lets the leader pick the single best specialist. The `financial_news` and `report_generator` agents are also reachable directly through dedicated single-agent paths for the focused REST endpoints.

Each agent is constructed with a real `model`, `tools` (the MCP toolkits), `role` and concrete `instructions`, and the team coordinator runs on a stronger model (`get_agno_model_strong`). On every Agno call the orchestrator inspects `response.status` and falls back to the deterministic path if it sees `RunStatus.error` or an error-shaped short string (e.g. `"Connection error."`) — guaranteeing the UI always renders a real, numeric response.

---

## Long-term Memory (`AgnoMemoryManager`)

`src/session.py` exposes two complementary stores:

- **`session_store`** — bounded in-memory deque (`MAX_TURNS = 10`) of recent turns, used by the classifier to resolve pronouns / vague references.
- **`agno_memory`** — Agno's built-in `MemoryManager` with **persistent storage** that prefers SQLite, falls back to a JSON directory (`.agno_memory.json/`), then in-memory if neither is installable. Conversations are persisted as `UserMemory` records keyed by `user_id`, and surfaced into the portfolio prompt as `Known facts about this user:` so subsequent turns can build on past intent.

The integration is fire-and-forget: `event_generator` reads memories before the agent runs and stores the new turn as a background task afterwards, so memory writes never block the SSE response.

---

## Structured JSON Logging

`src/logging_config.py` installs a `JSONFormatter` that emits one structured JSON line per log event with a per-request **correlation ID** (`ContextVar`):

```json
{"timestamp":"2026-05-11T13:23:42Z","level":"INFO","correlation_id":"req_a47fc1d2","component":"finsight.orchestrator","message":"agent=portfolio_health status=success duration=2840ms"}
```

The `correlation_id` is generated by `event_generator` and flows automatically through safety, classifier, router, orchestrator, agents and MCP toolkits via the `ContextVar`. To trace a request end-to-end: `grep correlation_id=req_a47fc1d2 server.log`. The same diagnostic info ships in `GET /metrics → "logging"`.

---

## Parallelism

The deterministic orchestrator uses **`asyncio.gather`** to fan out independent agent calls. A small wrapper (`_safe_run`) records each task's wall time, then `_build_exec_metadata` packs the per-task ms, the actual gather wall time and the *would-have-been* sequential sum into the `execution_metadata` field on the `AgentResponse`. The Agno Team path captures a single wall-clock measurement around `team.arun` and reports it with `parallel: true` since coordinate mode fans the work out under the hood.

### What it actually looks like

```python
# src/orchestrator.py
results, timings, wall_ms = await self._run_parallel([
    ("portfolio_health", self._portfolio.run(user, intent=intent, query=query)),
    ("risk_analysis",    self._risk.run(user, intent=intent, query=query)),
    ("news_agent",       self._news.run(tickers=tickers, topics=[...], user=user, ...)),
])
# wall_ms = real awaited time, timings = {agent_name: ms}
```

### Real timing example  (measured live during smoke test)

A `"how is my portfolio doing?"` query for a 2-position profile fans out to **portfolio_health + risk_analysis + financial_news**:

| Agent | Duration |
|---|---|
| portfolio_health | **5045 ms** |
| risk_analysis | **2054 ms** |
| financial_news | **2037 ms** |
| **Sequential sum** | **9136 ms** |
| **Wall (parallel)** | **5046 ms** |
| **Saved** | **4090 ms  (45 % faster)** |

That same `execution_metadata` JSON ships in the SSE result event and the browser UI's "Parallel Execution Timeline" renders it as scaled bars + a saved-ms summary in the accent colour:

```json
{
  "agents_ran": ["portfolio_health", "risk_analysis", "news_agent"],
  "timings": {"portfolio_health": 5045, "risk_analysis": 2054, "news_agent": 2037},
  "parallel": true,
  "wall_time_ms": 5046,
  "sequential_time_ms": 9136,
  "time_saved_ms": 4090
}
```

The wall time is bounded by the *slowest* agent, not the sum — exactly what we want when each task is I/O-bound (LLM call + yfinance round-trip + DDG search). Compute-bound work would not benefit (the asyncio loop is single-threaded), but every agent here spends almost all of its time waiting on the network.

---

## Tech Stack

| Technology | Purpose | Why chosen |
|---|---|---|
| **FastAPI** | HTTP + SSE + OpenAPI | Async-native; Pydantic v2 baked in; `EventSourceResponse` via `sse-starlette` is a one-liner. |
| **Pydantic v2** | Schema validation everywhere | `extra="forbid"` catches drift early; `model_dump_json()` makes SSE serialisation trivial; strict types stop UI ↔ backend contract bugs. |
| **Agno** | Multi-agent + MCP toolkit | Provides `Toolkit` and `Agent` primitives that work with both Groq and OpenAI; lets MCP servers be discoverable to tool-calling LLMs *without* giving up direct Python dispatch. |
| **Groq** (primary) / **OpenAI** (fallback) | LLM inference for classifier + observations | Groq's `llama-3.3-70b-versatile` is fast (sub-second) and free for development; OpenAI is the production fallback. Provider precedence: Groq → OpenAI → `SmartMockLLMClient`. |
| **scikit-learn** | TF-IDF + Logistic Regression | Used in two places: safety guard (~0.003 ms inference) and intent pre-classifier (skips the LLM for ≥0.80 confidence routing). Trains at startup from local fixtures — no network. |
| **yfinance** | Live equity / FX data | Free, global ticker coverage, no signup. Wrapped in `tenacity` for retries. Production swap target: Polygon or IEX. |
| **tenacity** | Exponential-backoff retry | Yahoo Finance is rate-limit-prone; tenacity sits in front of every yfinance call so transient 429s don't kill the request. |
| **scipy** | Black-Scholes Greeks + statistics | Just for `scipy.stats.norm` (options pricing) and a few numerical helpers. |
| **numpy** | Portfolio math | Returns, correlations, drawdowns, stress tests — all vectorised. |
| **duckduckgo-search** | News search MCP | Free, no API key, decent freshness. Bing fallback baked into `web_search_mcp`. |
| **fpdf2** | PDF report rendering | Pure-Python, no system fonts required, produces inspector-friendly artefacts under `reports/`. |
| **pytest** + `pytest-asyncio` | Test runner | First-class coroutine support so the production `async def` agents are tested as-is, no `asyncio.run()` boilerplate. |
| **httpx** | Async HTTP client + TestClient | Used for ad-hoc smoke tests and could back any future webhook MCP. |
| **uvicorn[standard]** | ASGI server | `--reload` in dev; HTTP/1.1 + `httptools` + `uvloop` in prod. |
| **python-dotenv** | `.env` loading | One-line bootstrap; never required in production where env vars come from the orchestrator. |

---

## Setup

```bash
git clone https://github.com/ashrith-07/finsight.git
cd finsight

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# No API key required for the deterministic mock path:
unset GROQ_API_KEY OPENAI_API_KEY    # uses SmartMockLLMClient — no key needed

# Optional: copy the env template and add your Groq (or OpenAI) key for live LLM output
cp .env.example .env

uvicorn src.main:app --reload
```

#### Open

<http://localhost:8000>

The single-page UI lives there. The right panel has three tabs:

- **Ecosystem** — agent cards (live status), MCP servers (last-called), parallel execution timeline, activity log
- **API Explorer** — every endpoint grouped by category with a *Try* button that pre-fills a relevant chat query
- **Live Metrics** — service counters auto-refreshed every 10 s

### Environment variables (all optional)

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | Preferred LLM provider when set |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Override the default Groq model |
| `OPENAI_API_KEY` | — | Used only if `GROQ_API_KEY` is absent |
| `OPENAI_MODEL` / `LLM_MODEL` | `gpt-4o-mini` | OpenAI model id alias |
| `PIPELINE_TIMEOUT` | `30` | Seconds per `/chat` SSE generator |
| `LOG_LEVEL` | `INFO` | Root logging level |
| `APP_ENV` | — | Set to `test` in CI if you branch on it |

If both keys are unset, the app silently falls back to `SmartMockLLMClient` — every endpoint stays functional with deterministic, intent-aware mock responses.

---

## Running Tests

```bash
pytest tests/ -v                     # no API key needed
```

The 10-test suite trains the safety guard and pre-classifier on the bundled fixtures, exercises the full classifier routing matrix, and runs `PortfolioHealthAgent` against a stubbed yfinance — completes in ~9 s.

---

## Architecture Decisions

### Why Agno over LangGraph / CrewAI

Agno's `Agent`, `Team` and `Toolkit` primitives are intentionally thin — closer to "structured glue around an LLM call" than the heavy DAG/state-machine abstractions of LangGraph or the role-playing layer of CrewAI. That's the right shape for this codebase because the **primary path is an Agno `Team` in `coordinate`/`route` mode** but I still want a deterministic Python fallback for resilience. Agno lets both coexist: `FinsightAgnoTeam` constructs two real teams (portfolio + research) with proper member roles, toolkits and instructions, while `FinsightOrchestrator` keeps the imperative `asyncio.gather` path alive for the no-key / provider-down case. Every agent is reachable three ways — through the team, directly via its async `run()` method (used by focused REST endpoints), or as an Agno `Agent` via `as_agno_agent()` for tool-calling LLMs — and all three share the underlying MCP code.

### Why MCP pattern over direct API calls

Each MCP server (`yfinance_mcp`, `web_search_mcp`, `report_mcp`, `calculator_mcp`, `portfolio_analytics_mcp`) is a single class that registers a curated set of stable, well-typed methods. Three benefits:

1. **One swap point per data source.** Replacing yfinance with Polygon means rewriting `yfinance_server.py` only — nothing else moves.
2. **Tool-calling compatibility.** Every method is automatically exposed as an Agno tool, so an LLM can invoke `yfinance_mcp.get_price_snapshot("NVDA")` without bespoke prompt engineering.
3. **Singleton lifecycle.** Servers instantiate once at import, share a thread-pool internally, and amortise authentication / connection costs across requests.

### Why `asyncio.gather` for parallelism

Every fan-out in this codebase is **I/O-bound** — LLM HTTPs, yfinance round-trips, DDG searches. `asyncio.gather` is the right primitive because:

- It's free (no thread pool, no GIL contention) and lives natively inside FastAPI's event loop.
- Each agent's `run()` is already `async def`, so wrapping with `_safe_run` to capture timings is a 5-line helper, not a refactor.
- The wall time becomes the **slowest** agent, not the sum — and we surface that win to the UI via `execution_metadata.time_saved_ms`.

A `multiprocessing` pool would have been wrong (overhead + serialisation cost). A thread pool would have worked but with no upside given the I/O-bound profile and a real downside in the form of per-thread overhead and GIL juggling for the small CPU-bound bits.

### Why TF-IDF + LR over an LLM for the safety guard

The safety guard runs **before** anything else on every request. Latency budget: microseconds. An LLM round-trip would be 100,000× slower (~300 ms vs 3 µs), would **cost money on every blocked query**, and would be non-deterministic across providers. TF-IDF + scikit-learn `LogisticRegression` trains at startup from `fixtures/safety_pairs.json` plus augmented examples, predicts in **~3 µs**, has an inspectable decision boundary, and can be retrained from CI without touching the model graph. The 0.55 confidence threshold is intentionally conservative to err toward not-blocking ambiguous educational queries.

The same reasoning applies to the **two-stage intent classifier**: a TF-IDF + LR pre-classifier handles ≥0.80-confidence routing without ever calling an LLM, cutting cost and tail latency for the common cases (portfolio health, market price checks, news lookups). Anything ambiguous falls through to the LLM — best of both worlds.

### Why in-memory session storage over Redis for the demo

Zero infrastructure dependencies for reviewers and CI: clone, `pip install`, run tests. Access is **O(1)** per session id with a bounded deque (`MAX_TURNS = 10`). For production this is the obvious swap: a Redis-backed store with TTL per session, horizontal replicas and eviction under memory pressure. The async API (`get_prior_user_turns`, `add_turn`, `get_last_entities`) was kept deliberately small so the swap is a one-file change in `src/session.py`, no caller updates required.

Long-term, cross-session memory is a different shape — it's facts about the user (risk tolerance, goals, concerns) that should persist beyond a single browser tab. That's why `AgnoMemoryManager` ships alongside: it uses Agno's native `MemoryManager` with a SQLite → JSON → in-memory backend cascade, so reviewers without `sqlalchemy` installed still get cross-session memory through the JSON fallback. The two stores coexist without overlap.

### Why a deterministic fallback under the Agno Team

LLM providers go down. Networks rate-limit. Free tiers throttle. A demo that depends entirely on `Groq.arun()` returning useful content is one outage away from a blank chat bubble. Every Agno call in `FinsightAgnoTeam` is wrapped in a status + content check: if the team returns `RunStatus.error`, empty content, or a short error-shaped string (`"Connection error."`, `"I'm sorry…"`, `"Rate limit…"`), the request transparently re-routes through `FinsightOrchestrator` which talks directly to yfinance and the analytics MCPs and produces the same `AgentResponse` shape with real numbers. The frontend cannot tell the difference — and that's the point.

---

## Repository Layout

| Path | Role |
|---|---|
| `src/main.py` | FastAPI app, `/chat` SSE pipeline, focused REST endpoints, metrics counters, memory wiring |
| `src/orchestrator.py` | `FinsightAgnoTeam` (Agno coordinate/route teams) + `FinsightOrchestrator` (deterministic fallback) |
| `src/router.py` | Picks the Agno Team primary path, falls back to `FinsightOrchestrator` / `StubAgent` |
| `src/safety.py` | TF-IDF + LR safety guard with a high-precision regex prefilter (insider, manipulation, laundering, guaranteed-returns, fraud) and an `_EDUCATION_RX` whitelist |
| `src/classifier.py` | Two-stage classifier (LR pre-classifier → LLM fallback) with regex overrides for risk / news / portfolio / report queries |
| `src/agents/` | Five primary agents + stub fallback; each agent ships an internal Agno `Agent` exposed via `as_agno_agent()` |
| `src/mcp/` | Five MCP toolkit servers (Agno `Toolkit` subclasses) |
| `src/llm/` | `LLMClient` ABC + Groq, OpenAI, Mock and SmartMock; `agno_model.py` returns Agno-native `OpenAIChat` / `Groq` model objects |
| `src/models.py` | Pydantic schemas (incl. `AgentResponse`, `ExecutionMetadata`) |
| `src/session.py` | In-session deque + query cache + **`AgnoMemoryManager`** (SQLite → JsonDb → InMemoryDb) |
| `src/logging_config.py` | Structured JSON logging with correlation-ID `ContextVar` |
| `frontend/index.html` | Single-page UI with tabbed Ecosystem / API / Metrics panels |
| `fixtures/` | Labelled queries + user profiles used by tests and the UI selector |
| `tests/` | pytest suite (passes without any API key) |
| `reports/` | Generated portfolio / market / risk reports (gitignored runtime output) |
| `.agno_memory.json/` | Persistent long-term memory (gitignored) |
