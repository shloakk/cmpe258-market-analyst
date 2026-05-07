# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project is a **multi-agent LLM research pipeline** built for CMPE 258 (Deep Learning, Spring 2026) as an Option 2 — LLMs + AI Agent System project. It implements a Scout → Mapper → Critic LangGraph pipeline that accepts a natural language query, retrieves relevant documents via FAISS similarity search, groups results into themes using Claude, and removes unsupported claims before returning a structured JSON response through a FastAPI + vanilla-JS web interface.

## Submission Standards

These rules must be followed in all code produced for this project to meet professor requirements:

- **Every major function must have a docstring** explaining its purpose, parameters, and return value. No exceptions, including helper functions.
- **Every major component must be evaluable.** If you add a new agent or pipeline stage, a corresponding test case or eval metric must exist in `eval/evaluator.py`.
- **No fabricated or placeholder results.** If a component does not work, raise a clear exception — do not silently return dummy data or hardcoded outputs.
- **Code comments must explain *why*, not just *what*.** A comment like `# strip fences before parsing` is not enough; write `# LLM sometimes wraps JSON in markdown fences; strip them to avoid json.loads() failure`.
- **All JSON schemas must be documented inline.** Anywhere a `list[dict]` is produced or consumed, include a comment listing the expected keys (e.g., `# keys: theme_name, companies, rationale, citations`).
- **Commit-ready code only.** Do not leave debugging `print()` statements, commented-out dead code blocks, or `TODO` stubs in finished files.
- **Use type hints on all function signatures.** Prefer `TypedDict` for shared state structures (already established via `ResearchState`).
- **Evaluation evidence is required.** Any claim about model behavior (e.g., "Critic reduces hallucinations") must be backed by a metric printed by `eval/evaluator.py`, not just described in prose.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY and OPENAI_API_KEY

# Must be run before using the pipeline — builds data/index/ from the corpus
python data/scripts/ingest.py
```

## Build / Test Commands

```bash
# ── Local development server (hot reload) ──────────────────────────────────
uvicorn app.main:app --reload

# ── Rebuild the FAISS index (required after any corpus change) ─────────────
python data/scripts/ingest.py

# ── Run the full evaluation suite against the seed query set ───────────────
python eval/evaluator.py

# ── Smoke-test the pipeline end-to-end from the command line ───────────────
python -c "
from pipeline.orchestrator import run_pipeline
import json
print(json.dumps(run_pipeline('map the key players in agentic AI platforms'), indent=2))
"

# ── Run a single agent in isolation (useful for debugging) ─────────────────
python -c "from agents.scout import ScoutAgent; s = ScoutAgent(); print(s.retrieve('LLM evaluation frameworks'))"
python -c "from agents.mapper import MapperAgent; help(MapperAgent.map)"

# ── Check that all imports resolve (catches missing deps before demo) ───────
python -c "from pipeline.orchestrator import run_pipeline; from eval.evaluator import run_eval; print('All imports OK')"
```

## Architecture

The system is a **three-agent LangGraph pipeline** where each agent is a discrete Python class, and the pipeline is a compiled `StateGraph`:

```
Scout → Mapper → Critic
```

**Data flow via `ResearchState` (TypedDict in `pipeline/orchestrator.py`):**
- `query` → input from user
- `retrieved_docs` → populated by Scout (FAISS similarity search results)
- `theme_map` → populated by Mapper (Claude groups companies into themes)
- `reviewed_map` → populated by Critic (Claude removes unsupported claims)

### Agents (`agents/`)

- **`scout.py` — `ScoutAgent`**: Loads `data/index/` (FAISS), embeds the query with `text-embedding-3-small`, returns top-k docs as `list[dict]` with keys `title, source_url, tags, publish_date, snippet, score`.
- **`mapper.py` — `MapperAgent`**: Sends retrieved docs to Claude (`claude-sonnet-4-6`) with a structured prompt. Parses the JSON response into `list[dict]` with keys `theme_name, companies, rationale, citations`.
- **`critic.py` — `CriticAgent`**: Sends the theme map + source docs to Claude. Returns a cleaned map with unsupported companies/themes removed. Uses same JSON schema as Mapper output.

Both Mapper and Critic strip markdown code fences from the LLM response before `json.loads()`.

### Pipeline (`pipeline/orchestrator.py`)

- `build_graph()` compiles the LangGraph `StateGraph`. The compiled graph is cached in a module-level `_pipeline` variable (lazy init on first call to `run_pipeline()`).
- `run_pipeline(query: str) -> list[dict]` is the single public entry point used by both the FastAPI backend and the eval script.

### Data (`data/`)

- **`data/corpus/sample_docs.json`**: Seed corpus. Each record: `{title, description, source_url, tags, publish_date}`. Expand this file (or add more JSON files) when growing the corpus.
- **`data/eval/queries.json`**: Eval set. Each record: `{id, query, gold_entities, gold_themes, gold_snippets}`.
- **`data/index/`**: FAISS index generated by `ingest.py`. Not committed to git; must be regenerated after corpus changes.
- **`data/scripts/ingest.py`**: Reads `sample_docs.json`, creates LangChain `Document` objects, embeds them, and saves the FAISS index to `data/index/`.

### Evaluation (`eval/evaluator.py`)

Three exported functions:
- `entity_precision_recall(gold, pred)` — case-insensitive substring match between entity lists.
- `hallucination_rate(reviewed_map, retrieved_docs)` — fraction of companies in the map not found in any source doc's text.
- `run_eval(pipeline_fn, queries_path)` — iterates the eval set, prints per-query metrics, returns aggregate dict.

### API (`app/`)

- `app/main.py`: FastAPI app. `POST /query` accepts `{query: str}`, calls `run_pipeline()`, returns `{query, themes: [{theme_name, companies, rationale, citations}]}`. `GET /` serves `app/static/index.html`.
- `app/static/index.html`: Vanilla JS single-page UI. Sends fetch to `/query`, renders theme cards with expandable citation lists. Cmd/Ctrl+Enter submits the query.

## Key Design Decisions

- **FAISS index is not committed** — it must be rebuilt with `ingest.py` whenever the corpus changes.
- **Both Mapper and Critic use the same LLM** (`claude-sonnet-4-6`). Swapping to a different model for comparison requires changing the `MODEL` constant in the respective agent file.
- **Agent classes are stateless** — `ScoutAgent` loads the FAISS index at `__init__` time; Mapper and Critic instantiate a new `ChatAnthropic` client each call. For production, these should be singletons.
- **JSON parsing is strict** — if the LLM returns malformed JSON, agents raise `ValueError` with the raw output. The FastAPI handler converts this to a 500 response.
