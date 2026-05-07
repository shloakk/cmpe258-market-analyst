# data/eval/RUBRIC.md — Eval Query Labeling Rubric

## Purpose

Documents the methodology for labeling `data/eval/queries.json`.
Every new query added to the eval set must follow these rules.
Annotators should cross-review a 20% random sample to measure inter-annotator
agreement (target: Jaccard similarity ≥ 0.7 on `gold_entities` across the overlap set).

---

## Query Types

| Type       | Count Target | Definition |
|------------|:------------:|---|
| Mapping    | ~20          | Asks the pipeline to cluster or categorize a space (e.g. "map X by Y", "group X by approach") |
| Listing    | ~15          | Asks for an enumeration (e.g. "list all X", "which startups do Y") |
| Comparison | ~10          | Asks for a structured contrast of 2+ entities on 1+ dimensions |
| Trend      | ~5           | Asks about temporal change or directional signals |

---

## Field Definitions

### `id`

Unique identifier. Format: `q` + zero-padded 3-digit integer (e.g. `q011`).
Assigned sequentially. Do not reuse or skip IDs.

### `query`

A natural language question a market analyst might ask. Requirements:

- Self-contained — no pronouns referencing previous queries
- Answerable from the corpus — do not invent entities absent from any scraped doc
- Written in second-person imperative or question form

### `gold_entities`

Proper names (company names, framework names, organization names) that a correct
answer **must** mention. Rules:

- Include only entities explicitly named in at least one corpus document.
- Use the canonical name as it appears in the corpus (e.g. "LangGraph" not "LG").
- Evaluated by `entity_precision_recall()` via case-insensitive substring match,
  so use the shortest unambiguous canonical name.
- For trend queries where the answer is thematic rather than entity-specific,
  this list may be empty (`[]`).

### `gold_themes`

Theme names that a correct answer **must** surface. Two valid sources:

1. Tags from the taxonomy below (e.g. `"rag"`, `"eval-tools"`)
2. Free-form theme names for concepts not in the taxonomy (e.g. `"funding"`, `"venture capital"`)

Evaluated by `theme_coverage()` via substring / name match.

### `gold_snippets`

Verbatim or near-verbatim text excerpts from corpus documents that justify the
answer. Rules:

- Must be an exact or near-exact sentence from a corpus document, not a paraphrase.
- Minimum 10 words; maximum 60 words per snippet.
- Maximum 3 snippets per query.
- **Leave as `[]`** for queries whose supporting text will come from the post-scrape
  corpus (i.e., documents not yet in the repo at labeling time). Fill in after
  `python data/scripts/scrape.py` has been run and the FAISS index rebuilt
  with `python data/scripts/ingest.py`.

---

## Labeling Procedure

1. For each new query, run the pipeline against the current index:

   ```bash
   python -c "
   from pipeline.orchestrator import run_pipeline
   import json
   print(json.dumps(run_pipeline('<your query here>'), indent=2))
   "
   ```

2. Read the pipeline output and the retrieved docs in the response.

3. Identify which entities and themes **should** appear in a correct answer based
   on the corpus content — not from general knowledge. Add only corpus-grounded
   entities to `gold_entities`.

4. Find supporting sentences in the corpus documents and paste them as
   `gold_snippets` (exact or near-exact, ≤ 60 words each, max 3).

5. **Peer review**: a second annotator independently labels the same query.
   If `Jaccard(gold_entities_A, gold_entities_B) < 0.7`, discuss and resolve
   before committing.

---

## Inter-Annotator Agreement Tracking

For the initial 40-query expansion (q011–q050), annotators A and B independently
labeled the following 10-query overlap set:

`q011`, `q015`, `q021`, `q027`, `q031`, `q037`, `q042`, `q046`, `q051`, `q054`

Record agreement scores here after labeling:

| Query ID | Jaccard (entities) | Resolved? |
|----------|:------------------:|:---------:|
| q011     | TBD                |           |
| q015     | TBD                |           |
| q021     | TBD                |           |
| q027     | TBD                |           |
| q031     | TBD                |           |
| q037     | TBD                |           |
| q042     | TBD                |           |
| q046     | TBD                |           |
| q051     | TBD                |           |
| q054     | TBD                |           |

**Target**: mean Jaccard ≥ 0.7 across all 10 overlap queries.

**Jaccard formula**: `|A ∩ B| / |A ∪ B|` where A and B are the two annotators'
`gold_entities` sets for the same query.

---

## Tag Taxonomy Reference

All tags used in corpus records and `gold_themes` must come from this list:

| Tag                       | Covers |
|---------------------------|--------|
| `multi-agent-orchestration` | Frameworks coordinating multiple LLM agents |
| `rag`                     | Retrieval-augmented generation, vector stores, indexing |
| `observability`           | Tracing, logging, monitoring of LLM runs |
| `agent-runtime`           | Execution environments, sandboxes, agent hosting |
| `voice-agents`            | Speech-in / speech-out agent interfaces |
| `coding-agents`           | Agents that write, execute, or review code |
| `vertical-agents`         | Domain-specific agents (support, legal, finance, etc.) |
| `eval-tools`              | Evaluation frameworks, scoring, regression tracking |
| `memory-tools`            | Long-term context storage, session memory, vector memory |
| `infra`                   | Foundational infra: inference, deployment, cost optimization |

Free-form themes (not in taxonomy) are allowed in `gold_themes` but not in
corpus document `tags` — corpus tags must always come from the taxonomy above.
