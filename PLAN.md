# CMPE 258 Project Plan: Multi-Agent Market & Trend Research Analyst

## Current State vs. Proposal

| Area | Proposal Target | Current State | Gap |
|---|---|---|---|
| Corpus | 200–300 docs | 10 docs (`data/corpus/sample_docs.json`) | +190–290 docs |
| Eval set | ≥50 queries with gold labels | 10 queries (`data/eval/queries.json`) | +40 queries |
| Pipeline (Scout→Mapper→Critic) | LangGraph 3-agent | Implemented (`pipeline/orchestrator.py`) | Tracing, robustness |
| Web UI | Market map + citations + 3-LLM comparison | Single-model UI (`app/static/index.html`) | Comparison view missing |
| Multi-LLM | 3 models incl. 1 open-source | Only `claude-sonnet-4-6` | +GPT, +OSS model |
| Metrics | Entity P/R, theme quality, hallucination, cost, latency | P/R + hallucination + cost/latency (`eval/evaluator.py`) | Theme/cluster quality metric |
| Observability | Langfuse | Not integrated | Add Langfuse |

---

## Phase 1 — Data Foundation (Week 1)

### 1.1 Corpus Expansion (target: 200–300 docs)

Build `data/scripts/scrape.py` that pulls from these sources, normalizes to `{title, description, source_url, tags, publish_date}`, and appends to sharded corpus files under `data/corpus/<source>.json`:

- **Startup landing pages** (~80 docs): CrewAI, AutoGen, LangGraph, LlamaIndex, Cognition, Adept, Sierra, Lindy, Decagon, Cresta, Imbue, MultiOn, Fixie, Reworkd, Vellum, Humanloop, etc. — fetch homepage + `/about` + first 1–2 blog posts via `httpx` + `trafilatura`.
- **YC RFS** (~30 docs): scrape `ycombinator.com/rfs` and recent batch pages tagged AI.
- **Newsletters** (~60 docs): Latent Space (Substack RSS), The AI Corner, Import AI, Ben's Bites — RSS → HTML extract.
- **News** (~50 docs): TechCrunch + VentureBeat AI tags via RSS, filtered to 2025–2026.
- **Technical posts** (~40 docs): Anthropic, OpenAI, LangChain, LlamaIndex engineering blogs.

**Tag taxonomy** (enables learnable themes): `multi-agent-orchestration`, `rag`, `observability`, `agent-runtime`, `voice-agents`, `coding-agents`, `vertical-agents`, `eval-tools`, `memory-tools`, `infra`.

Re-run `python data/scripts/ingest.py` after scraping to rebuild the FAISS index. Bump retrieval `k` from current default to 8–12 once corpus grows.

### 1.2 Eval Set Expansion (target: ≥50 queries)

Extend `data/eval/queries.json` with diverse query types:

- **Mapping** (~20): "map agentic AI orchestration frameworks", "group voice-agent startups by vertical"
- **Listing** (~15): "list YC W25 AI infra startups", "list open-source agent eval tools"
- **Comparison** (~10): "compare CrewAI vs AutoGen vs LangGraph"
- **Trend** (~5): "what are 2026 trends in agent memory systems"

For each query: hand-label `gold_entities`, `gold_themes`, `gold_snippets`. Document labeling rubric in `data/eval/RUBRIC.md`. Split work between team members; cross-review 20% overlap for inter-annotator agreement.

---

## Phase 2 — Multi-LLM Support (Week 2)

### 2.1 Model Abstraction

Refactor `agents/mapper.py` and `agents/critic.py` to take a `model_id` constructor arg. Create `agents/llm_client.py` that returns a unified `.invoke(messages) -> (text, usage, latency)` interface for:

- `claude-sonnet-4-6` (Anthropic)
- `gpt-4.1` or `gpt-5` (OpenAI)
- `meta-llama/Llama-3.3-70B-Instruct` via Together AI or Groq (open-source requirement)

### 2.2 Pipeline Parameterization

`run_pipeline` in `pipeline/orchestrator.py` gains a `model: str = "claude"` argument. The `_pipeline` module-level cache becomes a `dict[str, CompiledGraph]` keyed by model. Scout stays model-agnostic (embeddings only).

### 2.3 Robustness Improvements

Both Mapper and Critic currently raise `ValueError` on malformed JSON. Add:

- One retry with "your previous response was not valid JSON, return only JSON" follow-up.
- Pydantic schema validation (`ThemeMap` model) before returning.
- Token-budget guard: truncate `retrieved_docs` snippets to N chars when prompt exceeds model context.

---

## Phase 3 — Evaluation Upgrades (Week 2–3)

### 3.1 Theme/Cluster Quality Metric

Add to `eval/evaluator.py`:

- `theme_coverage(gold_themes, pred_themes)` — fraction of gold themes matched by any predicted theme via cosine similarity of theme names + member-set Jaccard ≥ 0.3.
- `theme_purity(pred_themes)` — average within-theme tag homogeneity using corpus tags.

### 3.2 Per-Model Eval Runner

Extend `run_eval` to loop `for model in ["claude", "gpt", "llama"]`, writing results to `eval/results/<model>_<timestamp>.json` with: per-query metrics, aggregate scores, cost (from token usage × per-model rates), and latency (wall-clock per agent).

### 3.3 Report Generator

Create `eval/report.py` that reads result files and emits:
- `eval/report.md` — comparison tables of all three models
- `eval/report.png` — matplotlib chart of accuracy vs. cost vs. latency

---

## Phase 4 — Observability (Week 3)

### 4.1 Langfuse Integration

Add `langfuse` to `requirements.txt`. In `agents/llm_client.py`, wrap each call with `@observe()`; pass a `trace_id` through `ResearchState` so Scout, Mapper, and Critic share one trace per query. Document `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` in `.env.example`.

### 4.2 Deployment Option

Use Langfuse Cloud free tier for the demo. Document a `docker-compose` self-host snippet in README for graders who cannot access the cloud dashboard.

---

## Phase 5 — UI Comparison View (Week 3–4)

### 5.1 New `/compare` Endpoint

Add `POST /compare` to `app/main.py` with body `{query, models: ["claude","gpt","llama"]}`. Returns `{model: {themes, latency_ms, cost_usd, trace_url}}`. Run the three pipelines concurrently via `asyncio.gather` (each `run_pipeline` wrapped with `asyncio.to_thread`).

### 5.2 Comparison Tab in UI

Add a "Compare models" toggle to `app/static/index.html`: three side-by-side columns each rendering the existing theme-card layout, plus a footer row showing latency / cost / Langfuse trace link per model. Reuse the existing card renderer function.

### 5.3 Eval Dashboard Page

Add `app/static/eval.html` served at `/eval`. Fetches `eval/results/latest.json` and renders the comparison chart and table inline. A run-on-demand button calls `POST /eval/run` (gated behind a `ALLOW_EVAL_RUN` env flag to prevent accidental API spend).

---

## Phase 6 — Polish & Deliverables (Week 4)

- **README rewrite**: architecture diagram (Mermaid), eval results table, demo GIF, quickstart for graders.
- **Dockerfile + docker-compose**: `docker compose up` delivers UI + Langfuse local instance.
- **Demo script** (`docs/demo.md`): 5-minute walkthrough — query → market map → expand citations → toggle comparison → open Langfuse trace → show eval table.
- **Final report PDF**: methodology, dataset stats, eval results, ablation (Critic on/off), limitations.
- **Recorded video** (Loom, 3–5 min): backup for live demo.

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Scraping 200+ pages hits rate limits / robots.txt | Polite scraper (1 req/sec, respect `robots.txt`), cache raw HTML to `data/raw/`, prefer RSS where available |
| Hand-labeling 50 queries is slow | LLM-assisted draft labels, then human review; track inter-annotator agreement on 10-query overlap |
| Llama 70B latency tanks the demo | Use Groq (Llama 3.3 70B at ~500 tok/s); fallback to Llama 3.1 8B if needed |
| Critic over-removes valid claims | Ablation eval: report metrics with Critic on/off; tune prompt to require explicit unsupported-evidence reference |
| Cost blowup running full eval × 3 models × 50 queries | Cache LLM responses keyed on `(model, prompt_hash)` in `eval/cache/`; estimated ≤$15 total |

---

## Timeline

| Week | Phases | Key Deliverables |
|---|---|---|
| 1 | Phase 1 | 200+ doc corpus, 50+ labeled queries, rebuilt FAISS index |
| 2 | Phase 2 + Phase 3.1–3.2 | Multi-LLM pipeline, theme quality metrics, per-model eval |
| 3 | Phase 3.3 + Phase 4 + Phase 5 | Langfuse tracing, eval report generator, comparison UI |
| 4 | Phase 6 | Docker, README, demo script, final report, video |
