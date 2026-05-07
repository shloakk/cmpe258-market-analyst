# Multi-Agent Market & Trend Research Analyst for Emerging AI Startups

A LangGraph-powered multi-agent system that analyzes the agentic AI startup landscape. Users submit natural language queries (e.g., "map the key players in agentic AI platforms") and receive a structured, citation-backed market map produced by three collaborating LLM agents.

---

## Team Members

| Name | Email | SJSU ID |
|---|---|---|
| Shloak Aggarwal | shloak.aggarwla@sjsu.edu | 018189938 |
| Matthew Bernard | matthew.bernard@sjsu.edu | 018230420 |

**Course:** CMPE 258 — Deep Learning  
**Option:** 2 — LLMs + AI Agent System (Evaluation-First)

---

## Dataset

We curate a corpus of **200–300 documents** from the agentic AI / LLM tools space, normalized into a common JSON schema:

```json
{
  "title": "CrewAI: Multi-Agent Orchestration for LLM Applications",
  "description": "...",
  "source_url": "https://crewai.com/blog/...",
  "tags": ["multi-agent", "orchestration", "LLM"],
  "publish_date": "2025-11-01"
}
```

**Sources include:**
- Startup landing pages and technical blog posts (CrewAI, AutoGen, LangGraph-based tools, LlamaIndex, etc.)
- Y Combinator Request for Startups (RFS) posts
- AI infrastructure newsletters: The AI Corner, Latent Space
- News coverage from TechCrunch and VentureBeat (2025–2026)

**Evaluation set:** 50+ hand-labeled queries with gold entities, themes, and supporting snippets — used to measure entity precision/recall, cluster quality, and hallucination rate across models.

---

## Approach

Queries flow through a three-agent LangGraph pipeline:

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  Scout Agent                        │
│  Embeds query → FAISS search        │
│  Returns top-k relevant documents   │
└────────────────┬────────────────────┘
                 │ retrieved_docs
                 ▼
┌─────────────────────────────────────┐
│  Mapper Agent                       │
│  Clusters docs into themes          │
│  (e.g., orchestration, RAG, evals)  │
│  Returns structured market map      │
└────────────────┬────────────────────┘
                 │ theme_map
                 ▼
┌─────────────────────────────────────┐
│  Critic Agent                       │
│  Verifies every claim has a citation│
│  Removes unsupported assertions     │
│  Returns reviewed market map        │
└────────────────┬────────────────────┘
                 │
                 ▼
         Structured Output
         (themes + citations)
```

**Models used:**
- Claude Sonnet (claude-sonnet-4-6) — Mapper and Critic agents
- OpenAI `text-embedding-3-small` — Scout embedding
- One open-source model (planned for comparison view)

**Frameworks:** LangGraph, LangChain, FAISS, FastAPI

---

## Progress

- [x] Project repo initialized with dependency management and environment configuration
- [x] Seed corpus (10 documents) and evaluation query set (10 queries) created
- [x] Document ingestion pipeline: normalizes raw docs to JSON and builds a FAISS vector index
- [x] Scout Agent: embeds queries and retrieves top-k relevant documents
- [x] Mapper Agent: clusters retrieved companies into themes via Claude
- [x] Critic Agent: enforces citations and removes unsupported claims
- [x] LangGraph StateGraph orchestrating all three agents end-to-end
- [x] Evaluation framework: entity precision/recall and hallucination rate metrics
- [x] FastAPI backend with `/query` endpoint and minimal web UI
- [x] Fixed hallucination metric — evaluator now grounds predictions against actual retrieved docs (was always passing an empty list)
- [x] Added `theme_coverage` eval metric using the `gold_themes` field in the evaluation query set
- [x] Per-agent cost and latency tracking — Scout, Mapper, and Critic each report `latency_ms`; LLM agents report `input_tokens`, `output_tokens`, and `cost_usd` (Claude Sonnet pricing)
- [x] Pipeline stats surfaced in API response (`PipelineStats` model with per-agent breakdown and totals)
- [x] Web UI displays a Pipeline Stats panel after each query (latency, cost, token counts per agent)
- [x] Langfuse observability: one trace per query with Scout, Mapper, Critic, and LLM generation spans

---

## Next Steps

- [ ] Expand corpus to 200–300 documents from all planned sources
- [ ] Build and annotate the full 50-query evaluation set with gold labels (currently 10)
- [ ] Integrate a second LLM (e.g., GPT-4o) and an open-source model (e.g., Llama 3) for side-by-side comparison
- [ ] Add side-by-side LLM comparison panel to the web UI
- [ ] Run full eval sweep and report aggregate accuracy, hallucination rate, cost/latency across models

---

## Observability

Langfuse tracing is optional. The app runs normally when Langfuse variables are
blank, but setting them creates one trace per query with nested Scout, Mapper,
Critic, and LLM generation observations.

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

For the demo, use the Langfuse Cloud free tier and open the trace by searching
for the `trace_id` returned from `POST /query`.

If graders cannot access the cloud dashboard, self-host Langfuse with the
official compose file and point this app at `http://localhost:3000`:

```yaml
services:
  langfuse-web:
    image: docker.io/langfuse/langfuse:3
    ports:
      - "3000:3000"
    environment:
      NEXTAUTH_URL: http://localhost:3000
      NEXTAUTH_SECRET: local-dev-secret # change outside local demos
      SALT: local-dev-salt
      ENCRYPTION_KEY: "0000000000000000000000000000000000000000000000000000000000000000"
      DATABASE_URL: postgresql://postgres:postgres@postgres:5432/postgres
      CLICKHOUSE_URL: http://clickhouse:8123
      CLICKHOUSE_MIGRATION_URL: clickhouse://clickhouse:9000
      CLICKHOUSE_USER: clickhouse
      CLICKHOUSE_PASSWORD: clickhouse
      REDIS_HOST: redis
      REDIS_PORT: 6379
      REDIS_AUTH: myredissecret
      LANGFUSE_S3_EVENT_UPLOAD_BUCKET: langfuse
      LANGFUSE_S3_EVENT_UPLOAD_ENDPOINT: http://minio:9000
      LANGFUSE_S3_EVENT_UPLOAD_ACCESS_KEY_ID: minio
      LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY: miniosecret
      LANGFUSE_S3_EVENT_UPLOAD_FORCE_PATH_STYLE: "true"
      LANGFUSE_INIT_ORG_ID: cmpe258
      LANGFUSE_INIT_PROJECT_ID: market-analyst
      LANGFUSE_INIT_PROJECT_PUBLIC_KEY: pk-lf-local
      LANGFUSE_INIT_PROJECT_SECRET_KEY: sk-lf-local

  # Also include the official langfuse-worker, postgres, clickhouse, redis,
  # and minio services from https://github.com/langfuse/langfuse/blob/main/docker-compose.yml.
```

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-local
LANGFUSE_SECRET_KEY=sk-lf-local
LANGFUSE_HOST=http://localhost:3000
```

---

## Project Structure

```
cmpe258-market-analyst/
├── data/
│   ├── corpus/          # Normalized JSON documents
│   ├── eval/            # Evaluation queries with gold labels
│   └── scripts/         # Ingestion and indexing scripts
├── agents/              # Scout, Mapper, Critic agent implementations
├── pipeline/            # LangGraph StateGraph orchestrator
├── eval/                # Evaluation metrics
└── app/                 # FastAPI backend + web UI
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys; set DEFAULT_MODEL to gemini/llama/gpt/claude

# Build the FAISS index from the seed corpus
python data/scripts/ingest.py

# Run the API server
uvicorn app.main:app --reload
# Open http://localhost:8000
```
