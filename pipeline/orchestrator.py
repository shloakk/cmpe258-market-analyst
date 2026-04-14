"""
LangGraph orchestrator: wires Scout → Mapper → Critic into a stateful
directed graph. Each node receives the current ResearchState, runs its
agent, and returns updated state fields.

Usage:
    from pipeline.orchestrator import run_pipeline
    result = run_pipeline("map the key players in agentic AI platforms")
    # result is a list of verified theme dicts
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from agents.scout import ScoutAgent
from agents.mapper import MapperAgent
from agents.critic import CriticAgent


class ResearchState(TypedDict):
    query: str
    retrieved_docs: list[dict]
    theme_map: list[dict]
    reviewed_map: list[dict]
    error: str | None


def scout_node(state: ResearchState) -> ResearchState:
    agent = ScoutAgent()
    docs = agent.run(state["query"])
    return {**state, "retrieved_docs": docs}


def mapper_node(state: ResearchState) -> ResearchState:
    agent = MapperAgent()
    theme_map = agent.run(state["query"], state["retrieved_docs"])
    return {**state, "theme_map": theme_map}


def critic_node(state: ResearchState) -> ResearchState:
    agent = CriticAgent()
    reviewed = agent.run(state["theme_map"], state["retrieved_docs"])
    return {**state, "reviewed_map": reviewed}


def build_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("scout", scout_node)
    graph.add_node("mapper", mapper_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("scout")
    graph.add_edge("scout", "mapper")
    graph.add_edge("mapper", "critic")
    graph.add_edge("critic", END)

    return graph.compile()


_pipeline = None


def run_pipeline(query: str) -> list[dict]:
    """
    Run the full Scout → Mapper → Critic pipeline.

    Args:
        query: Natural language market research query.

    Returns:
        Verified list of theme dicts, each with:
        {theme_name, companies, rationale, citations}
    """
    return run_pipeline_full(query)["reviewed_map"]


def run_pipeline_full(query: str) -> dict:
    """
    Run the full Scout → Mapper → Critic pipeline and return the complete result.

    Args:
        query: Natural language market research query.

    Returns:
        Dict with keys:
            reviewed_map: Verified list of theme dicts.
            retrieved_docs: Documents retrieved by the Scout agent.
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = build_graph()

    initial_state: ResearchState = {
        "query": query,
        "retrieved_docs": [],
        "theme_map": [],
        "reviewed_map": [],
        "error": None,
    }
    final_state = _pipeline.invoke(initial_state)
    return {
        "reviewed_map": final_state["reviewed_map"],
        "retrieved_docs": final_state["retrieved_docs"],
    }
