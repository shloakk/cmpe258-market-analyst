"""
Mapper Agent: given retrieved documents, uses Claude to cluster companies
into named themes (e.g., multi-agent orchestration, RAG tools, evaluation).
Returns a structured market map with citations for each theme.
"""

import json
import time
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage

MODEL = "claude-sonnet-4-6"
# Claude Sonnet pricing (per token)
_INPUT_COST_PER_TOKEN = 3.0 / 1_000_000
_OUTPUT_COST_PER_TOKEN = 15.0 / 1_000_000

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
        self.llm = ChatAnthropic(model=MODEL, temperature=0)

    def run(self, query: str, retrieved_docs: list[dict]) -> tuple[list[dict], dict]:
        """
        Args:
            query: The original user query.
            retrieved_docs: Documents returned by the Scout agent.

        Returns:
            Tuple of (theme_map, stats) where stats contains latency, token
            counts, and estimated cost.
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
        cost_usd = round(
            input_tokens * _INPUT_COST_PER_TOKEN
            + output_tokens * _OUTPUT_COST_PER_TOKEN,
            6,
        )
        stats = {
            "model": MODEL,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
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
