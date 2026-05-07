"""
Mapper Agent: given retrieved documents, uses Groq (Llama 3.3 70B) to cluster
companies into named themes (e.g., multi-agent orchestration, RAG tools, evaluation).
Returns a structured market map with citations for each theme.
"""

import json
import time
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a market research analyst specializing in AI infrastructure and emerging startups.
Given a list of documents about AI companies and tools, your job is to:
1. Identify distinct market themes or categories
2. Group companies/tools under each theme
3. Provide a concise rationale for each theme
4. Cite the specific document (by title) that supports each company's placement

Respond ONLY with valid JSON in the following format:
[
  {
    "theme_name": "string",
    "companies": ["string", ...],
    "rationale": "string",
    "citations": ["document title", ...]
  }
]
"""


class MapperAgent:
    """Clusters retrieved documents into market themes using Claude."""

    def __init__(self) -> None:
        self.llm = ChatGroq(model=MODEL, temperature=0)

    def run(self, query: str, retrieved_docs: list[dict]) -> tuple[list[dict], dict]:
        """
        Args:
            query: The original user query.
            retrieved_docs: Documents returned by the Scout agent.

        Returns:
            Tuple of (theme_map, stats) where stats contains latency and token counts.
            cost_usd is always 0.0 (Groq free tier).
        """
        docs_text = "\n\n".join(
            f"[{i+1}] Title: {d['title']}\n{d['snippet']}"
            for i, d in enumerate(retrieved_docs)
        )
        user_message = (
            f"User query: {query}\n\n"
            f"Retrieved documents:\n{docs_text}\n\n"
            "Based on these documents, produce a structured market map."
        )
        t0 = time.perf_counter()
        response = self.llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ]
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        usage = response.usage_metadata or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        stats = {
            "model": MODEL,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": 0.0,
            "latency_ms": latency_ms,
        }

        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            return json.loads(raw), stats
        except json.JSONDecodeError as e:
            raise ValueError(f"Mapper returned invalid JSON: {e}\nRaw output:\n{raw}")
