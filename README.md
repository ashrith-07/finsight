[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/SHM9MYZJ)
# Valura AI — Team Lead Project Assignment

You have been given access to this repository as part of the Valura AI team lead hiring process.

**Read [`ASSIGNMENT.md`](ASSIGNMENT.md) in full before writing a single line of code.**

---

## What you're building

An AI agent ecosystem that helps a novice investor **build, monitor, grow, and protect** their portfolio. See [`ASSIGNMENT.md`](ASSIGNMENT.md) for the full mission, scope, and constraints.

---

## Setup

**Requirements:** Python 3.11+, an OpenAI API key.

**Persistence is your choice.** Postgres, SQLite, or in-memory — pick one and defend it in your README. `DATABASE_URL` in `.env.example` is optional.

### Session memory (`src/session.py`)

Conversation history uses an **in-memory** `SessionStore` (dict keyed by `session_id`, each value a capped list of `ConversationTurn`). Rationale:

- **Zero extra dependencies** — no Redis driver, no DB URL, no migrations; the demo runs after `pip install` only.
- **Sufficient for demo scale** — bounded retention (`MAX_TURNS = 10` per session) keeps memory predictable for classroom evaluation and local testing.
- **Trivially swappable later** — the store exposes a small async API (`get_turns`, `add_turn`, `get_prior_user_turns`, `get_last_entities`, `clear`). Replacing the dict with Redis or Postgres is a drop-in behind the same methods.

An optional **`QueryCache`** (60s TTL) deduplicates identical `(session_id, query)` classifier calls within a session to shave redundant LLM cost during retries or double-submits.

**Streaming is required.** SSE only. Use `sse-starlette`, FastAPI's `StreamingResponse`, or roll your own — your call.

```bash
git clone <your-classroom-repo-url>
cd <repo-name>

python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt

cp .env.example .env
# Fill in OPENAI_API_KEY
```

Use `gpt-4o-mini` while developing to keep costs down. Evaluation runs against `gpt-4.1`.

---

## Running Tests

```bash
pytest tests/ -v
```

Tests must pass without an `OPENAI_API_KEY` set — mock the LLM. We will run `pytest tests/ -v` on your repo.

---

## Repository Structure

When you submit, your repository must contain:

```
README.md   ← overwrite this with your own (setup, decisions, library choices, video link)
src/        ← all code
tests/      ← all tests, must pass with pytest
```

`fixtures/`, `pytest.ini`, `requirements.txt`, `.env.example`, and `.github/` are part of the scaffold — leave them in place. Do not delete `ASSIGNMENT.md`.

---

## Submission

- Push commits **throughout** your work — we read the git log
- Your `README.md` must:
  - Explain how to run your code
  - List every required environment variable
  - Document the non-obvious decisions you made
  - Link your defence video (≤ 10 min — see `ASSIGNMENT.md`)
- Deadline: **3 days** from the date you accepted this assignment
- Defence video: due within **24 hours** of your final commit

---

## Environment

You self-host everything. We do not provide credentials. See `.env.example` for the variables you'll need.
