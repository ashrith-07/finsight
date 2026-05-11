# Valura AI Microservice

Intent routing, portfolio analytics, and streaming responses for Valura’s AI co-investor layer. Single FastAPI service: **one POST endpoint**, **SSE-only** replies, **no OpenAI dependency in CI** (mock LLM fallback).

---

## Defence Video

**[Defence walkthrough (Loom)](https://www.loom.com/share/dda77d0b03b0497ba840a287b02ce007)**

---

## Architecture

A chat turn is handled as a **linear pipeline** before anything hits an agent model for long-form work. First, the **safety guard** evaluates the raw user string synchronously with compiled regex rules (policy buckets such as insider trading, manipulation, AML patterns, etc.). If the query is blocked, the pipeline stops immediately: we stream the refusal as SSE text and close — **no classifier, no external pricing calls, no billing to OpenAI**. That ordering matters because once we’ve invoked third-party models or market data, we’ve already spent latency and money on content we must not produce.

If the query passes safety, the **intent classifier** runs **exactly one** structured LLM call (or a cached duplicate within the same session window). It returns an `agent` slug plus extracted `entities` so downstream code never has to re-parse natural language. The **router** maps that slug to `PortfolioHealthAgent` (fully implemented) or a **stub** for every other taxonomy slot. Only then does the chosen agent run — today that means either portfolio math + yfinance + a **second** LLM call for plain-language observations, or a deterministic stub response.

Finally the HTTP layer streams **SSE**: token-ish **delta** chunks for a short summary line, a **result** event carrying the full structured `AgentResponse` JSON, then **done**. Errors never leak stack traces to the client; they become **error** events with stable codes.

```
  Client POST /chat (JSON body)
           │
           ▼
    ┌──────────────┐     blocked? ──► SSE: delta (refusal) ──► done
    │ Safety guard │                              │
    └──────────────┘                              └── stop
           │ pass
           ▼
    ┌──────────────┐     session query cache hit?
    │ Classifier   │ ◄── replicate ClassifierResult (still route agent)
    │ (1× LLM)     │
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │ AgentRouter  │──► portfolio_health ──► yfinance + metrics + observations LLM
    └──────────────┘──► other agents ───────► StubAgent (no crash)
           │
           ▼
    SSE: delta … ──► result (JSON) ──► persist ConversationTurn ──► done
```

---

## Non-Obvious Decisions

### Why in-memory session storage?

**Zero infrastructure dependencies** for reviewers and classroom CI: clone, `pip install`, run tests. Access is **O(1)** per session id with a bounded deque (`MAX_TURNS = 10`). That’s defensible for demo scale and single-process deployments. For production I’d swap this for **Redis** (TTL per session, horizontal replicas, eviction under memory pressure) or append-only storage if compliance needs audit trails — the async API surface (`get_prior_user_turns`, `add_turn`, …) was kept deliberately small so that swap is mechanical.

### Why a strict `LLMClient` interface?

CI must pass **without** `OPENAI_API_KEY`. A narrow ABC lets tests inject **`MockLLMClient`** with queued responses — no monkeypatching OpenAI. **`openai` is imported in exactly one module** (`src/llm/openai_llm.py`), which keeps greps honest and avoids accidental SDK coupling in agents. Adding Anthropic or Gemini is **one new concrete class** + factory branch; the classifier and portfolio agent stay untouched.

### Why SSE-only (no JSON fallback)?

The assignment requires streaming. Practically, **SSE matches how users perceive LLM latency**: they see the first bytes quickly instead of staring at a spinner until a full JSON blob exists. A JSON-only API would force buffering the summary and the structured payload in lockstep; here we stream a short human-readable prefix, then attach the machine-readable `AgentResponse` as one frame.

### Safety guard design: layered regex vs. LLM

The guard is **pure local regex** with a **per-category allow layer** for educational phrasing (“what is…”, “explain…”, penalties, etc.). Typical checks finish in **sub-millisecond** averages on commodity hardware (see Performance). An LLM-based guard would be slower, cost money per turn, and be harder to regression-test against labeled pairs. **Tradeoff:** brittle edge cases exist — we tuned against `fixtures/test_queries/safety_pairs.json`. Known sensitivities:

- **Ambiguous phrasing** that mixes imperative (“help me…”) with curriculum vocabulary can still trip a block if it hits a Layer‑1 pattern before an allow pattern matches.
- **Guaranteed-return language** is intentionally aggressive; marketing-style questions must include clear definitional cues to bypass.

### Pipeline timeout: 30 seconds

Happy path on this codebase is dominated by **yfinance** round-trips plus **two** LLM calls on portfolio-heavy flows (classifier + observations). Locally that lands **~3–6s** after warm caches; **30s** leaves roughly **6×** slack for slow Yahoo endpoints or transient LLM latency without letting workers hang forever. Override via `PIPELINE_TIMEOUT`.

### Cost per query estimate

Rough order-of-magnitude at published **gpt-4.1**-class pricing (check OpenAI’s current page before quoting externally):


| Step                   | Tokens (indicative) | Notes                                       |
| ---------------------- | ------------------- | ------------------------------------------- |
| Classifier             | ~500 in / ~100 out  | System taxonomy + entity vocab + user query |
| Portfolio observations | ~800 in / ~200 out  | Metrics JSON + profile summary              |

Using current pricing **$2 / 1M input tokens** and **$8 / 1M output tokens**:

- Classifier: `500×2e-6 + 100×8e-6` ≈ **$0.0018**
- Observations: `800×2e-6 + 200×8e-6` ≈ **$0.0032**
- **Total ≈ $0.005** per full **portfolio_health** turn — **well under** the **$0.05** assignment budget (stub agents are cheaper; cache hits skip classifier LLM).

### What I'd do differently with another week

1. **Embedding router** — classify ultra-high-confidence FAQs without an LLM (cost + latency win).
2. **Price cache** — memoize yfinance quotes per ticker with short TTL (seconds/minutes) to cut duplicate downloads within a burst.
3. **Correlation IDs** — propagate `X-Request-ID` through logs across safety, classifier, router, and agent spans.
4. **Rate limits** — token bucket per `user_id` / IP on `/chat` to protect upstream APIs.

---

## Setup

### Requirements

- **Python 3.11+**
- **OpenAI API key** — optional; without it the factory returns **`MockLLMClient`** (empty queue — fine for health checks; use injected mocks in tests).

### Installation

```bash
git clone https://github.com/2CentsCapital/valura-ai-ai-engineer-assignment-ashrith-07
cd valura-ai-ai-engineer-assignment-ashrith-07
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Optionally set OPENAI_API_KEY for live LLM output
```

### Environment Variables


| Variable           | Required | Default       | Description                                                                |
| ------------------ | -------- | ------------- | -------------------------------------------------------------------------- |
| `OPENAI_API_KEY`   | No       | —            | If unset/empty,**`MockLLMClient`** is selected automatically               |
| `LLM_MODEL`        | No       | `gpt-4o-mini` | Chat model id (`OPENAI_MODEL` alias supported for backwards compatibility) |
| `PIPELINE_TIMEOUT` | No       | `30`          | Max seconds for the`/chat` generator (float allowed)                       |
| `LOG_LEVEL`        | No       | `INFO`        | Root log level (`DEBUG`, `INFO`, …)                                       |
| `APP_ENV`          | No       | —            | Set to`test` in CI workflows if you branch on it                           |

See `.env.example` for optional persistence/cache placeholders used in broader designs.

### Running

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Testing (no API key required)

```bash
pytest tests/ -v
```

### Example Request

Build JSON from the bundled US trader fixture and stream SSE:

```bash
python - <<'PY' | curl -sS -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d @-
import json
from pathlib import Path
user = json.loads(Path("fixtures/users/user_001_active_trader_us.json").read_text())
print(json.dumps({
  "query": "how is my portfolio doing",
  "session_id": "demo-session",
  "user": user,
}))
PY
```

---

## Performance Measurements

**Methodology:** micro-benchmarks on a developer laptop (Python 3.13, May 2026) — **not** production SLA data. For HTTP p95 under load, rerun with **wrk**/k6 against `POST /chat` with a representative body.


| Measurement                                 | Result                                                                      | Notes                                  |
| ------------------------------------------- | --------------------------------------------------------------------------- | -------------------------------------- |
| Safety guard                                | **~0.003 ms** avg over **500** calls (`check("how is my portfolio doing")`) | Pure CPU regex path                    |
| Portfolio health (mock LLM + live yfinance) | **~2.5–3.0 s** warm requests; **~5 s** first call after import             | Dominated by Yahoo Finance round-trips |
| Full test suite`pytest tests/ -v`           | **~11 s** wall clock                                                        | Includes network-bound portfolio tests |

**First-byte SSE latency** for `/chat` was not stress-tested with concurrent clients in this submission — recommend measuring after deploying behind your ASGI server of choice.

---

## Library Choices


| Library            | Why                                                                         |
| ------------------ | --------------------------------------------------------------------------- |
| **FastAPI**        | Async-first, native Pydantic v2 models for request validation               |
| **sse-starlette**  | Correct SSE framing (`EventSourceResponse`), ping & disconnect semantics    |
| **yfinance**       | Free equity/FX snapshots without vendor signup; good global ticker coverage |
| **pydantic v2**    | Strict schemas (`extra="forbid"`), fast validation, JSON ergonomics         |
| **pytest-asyncio** | First-class async tests matching production coroutines                      |
| **httpx**          | Async-capable client for integration tests / tooling                        |

---

## Repository Layout


| Path                             | Role                                             |
| -------------------------------- | ------------------------------------------------ |
| `src/main.py`                    | FastAPI app,`/chat` SSE pipeline                 |
| `src/safety.py`                  | Rule-based safety guard                          |
| `src/classifier.py`              | Intent classifier (`IntentClassifier`)           |
| `src/router.py`                  | Agent routing                                    |
| `src/agents/portfolio_health.py` | Full portfolio agent                             |
| `src/agents/stub.py`             | Placeholder responses                            |
| `src/session.py`                 | In-memory sessions + optional query dedupe cache |
| `src/llm/`                       | `LLMClient` ABC, OpenAI + mock implementations   |
| `fixtures/`                      | Labeled queries and user profiles for tests      |
| `tests/`                         | pytest suite (passes without API keys)           |

Assignment brief and rubric context remain in [`ASSIGNMENT.md`](ASSIGNMENT.md).
