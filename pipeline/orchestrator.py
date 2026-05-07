"""
LangGraph orchestrator: wires Scout → Mapper → Critic into a stateful directed
graph and runs it for a chosen LLM ("claude" / "gpt" / "llama").

A separate compiled graph is built and cached per model. This keeps the
graph compilation overhead a one-time cost per model and lets the agents
hold provider clients in closure (which avoids re-instantiating the FAISS
index and the chat model on every invocation).

Usage:
    from pipeline.orchestrator import run_pipeline, run_pipeline_full
    themes = run_pipeline("map agentic AI platforms", model="claude")
    full = run_pipeline_full("...", model="gpt")  # includes timings + docs
"""

from typing import TypedDict

from langgraph.graph import END, StateGraph

from agents.critic import CriticAgent
from agents.llm_client import ModelId
from agents.mapper import MapperAgent
from agents.scout import ScoutAgent


class ResearchState(TypedDict):
    """Shared state passed between graph nodes.

    Keys:
        query: User query (input).
        retrieved_docs: Populated by Scout. list[dict] with keys title,
            source_url, tags, publish_date, snippet, score.
        theme_map: Populated by Mapper. list[dict] with keys theme_name,
            companies, rationale, citations.
        reviewed_map: Populated by Critic. Same shape as theme_map.
        error: Reserved for future error-propagation; currently always None.
            Errors raised in nodes propagate through LangGraph and surface in
            the FastAPI handler as HTTP 500s.
        timings: Per-agent stats dict. Keys: scout, mapper, critic. Each value
            is a dict with at least latency_ms; LLM agents also include model,
            input_tokens, output_tokens, cost_usd.
    """

    query: str
    retrieved_docs: list[dict]
    theme_map: list[dict]
    reviewed_map: list[dict]
    error: str | None
    timings: dict


def build_graph(model: ModelId = "claude"):
    """Build and compile the Scout → Mapper → Critic graph for one model.

    Scout is model-agnostic (it only uses OpenAI embeddings via the FAISS
    index). Mapper and Critic both bind to ``model`` and are captured via
    closure inside the node functions, so the model selection does not need
    to live in ``ResearchState``. This keeps the state shape stable across
    models and means a single graph instance always uses one model.

    Args:
        model: Short model id passed to MapperAgent and CriticAgent.

    Returns:
        A compiled LangGraph runnable ready to invoke with a ResearchState.
    """
    # Instantiate agents once at compile time; each compiled graph "owns"
    # them. Scout's __init__ reads the FAISS index from disk (heavy), so
    # doing this once per model — not once per query — is a real speedup.
    scout = ScoutAgent()
    mapper = MapperAgent(model=model)
    critic = CriticAgent(model=model)

    def scout_node(state: ResearchState) -> ResearchState:
        docs, stats = scout.run(state["query"])
        timings = {**state.get("timings", {}), "scout": stats}
        return {**state, "retrieved_docs": docs, "timings": timings}

    def mapper_node(state: ResearchState) -> ResearchState:
        theme_map, stats = mapper.run(state["query"], state["retrieved_docs"])
        timings = {**state.get("timings", {}), "mapper": stats}
        return {**state, "theme_map": theme_map, "timings": timings}

    def critic_node(state: ResearchState) -> ResearchState:
        reviewed, stats = critic.run(state["theme_map"], state["retrieved_docs"])
        timings = {**state.get("timings", {}), "critic": stats}
        return {**state, "reviewed_map": reviewed, "timings": timings}

    graph = StateGraph(ResearchState)
    graph.add_node("scout", scout_node)
    graph.add_node("mapper", mapper_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("scout")
    graph.add_edge("scout", "mapper")
    graph.add_edge("mapper", "critic")
    graph.add_edge("critic", END)

    return graph.compile()


# Per-model graph cache. Keyed on ModelId so a service that calls /query for
# multiple models does not pay graph-compilation + FAISS-load cost on every
# request. Populated lazily on first use of each model.
_pipelines: dict[str, object] = {}


def _get_pipeline(model: ModelId):
    """Get or build (and cache) the compiled graph for ``model``."""
    if model not in _pipelines:
        _pipelines[model] = build_graph(model)
    return _pipelines[model]


def run_pipeline(query: str, model: ModelId = "claude") -> list[dict]:
    """Run the Scout → Mapper → Critic pipeline and return only the reviewed map.

    Args:
        query: Natural language market research query.
        model: Short model id selecting which LLM Mapper and Critic use.
            Defaults to "claude" for backwards compatibility with single-model
            callers.

    Returns:
        Verified list of theme dicts. Keys: theme_name, companies, rationale,
        citations.
    """
    return run_pipeline_full(query, model=model)["reviewed_map"]


def run_pipeline_full(query: str, model: ModelId = "claude") -> dict:
    """Run the full pipeline and return the reviewed map plus diagnostics.

    Args:
        query: Natural language market research query.
        model: Short model id ("claude" / "gpt" / "llama").

    Returns:
        Dict with keys:
            reviewed_map: list[dict] of verified themes.
            retrieved_docs: list[dict] of docs returned by Scout (so eval can
                ground its hallucination metric against the same evidence).
            timings: per-agent stats dict (scout / mapper / critic).
    """
    pipeline = _get_pipeline(model)

    initial_state: ResearchState = {
        "query": query,
        "retrieved_docs": [],
        "theme_map": [],
        "reviewed_map": [],
        "error": None,
        "timings": {},
    }
    final_state = pipeline.invoke(initial_state)
    return {
        "reviewed_map": final_state["reviewed_map"],
        "retrieved_docs": final_state["retrieved_docs"],
        "timings": final_state["timings"],
    }
