"""
Critic Agent: reviews the Mapper's theme map against the retrieved documents.
Removes claims or companies that lack grounding in the source documents,
and ensures every remaining entry has a supporting citation.
"""

import json
import time
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a rigorous fact-checker for market research reports.
You will receive:
1. A list of source documents (the evidence)
2. A market map produced by a previous analyst

Your job is to verify that every company and claim in the market map is
supported by at least one of the source documents. For any company or claim
that has NO supporting evidence in the provided documents, remove it from the map.

Rules:
- Do not add new companies or themes not already in the map.
- Do not fabricate citations. Only cite document titles that appear in the source list.
- If an entire theme has no supported companies, remove the theme entirely.
- Keep the same JSON structure as the input map.

Respond ONLY with valid JSON using the same structure as the input:
[
  {
    "theme_name": "string",
    "companies": ["string", ...],
    "rationale": "string",
    "citations": ["document title", ...]
  }
]
"""


class CriticAgent:
    """Verifies the market map against source documents and removes unsupported claims."""

    def __init__(self) -> None:
        self.llm = ChatGroq(model=MODEL, temperature=0)

    def run(self, theme_map: list[dict], retrieved_docs: list[dict]) -> tuple[list[dict], dict]:
        """
        Args:
            theme_map: Output from the Mapper agent.
            retrieved_docs: The original source documents from the Scout agent.

        Returns:
            Tuple of (verified_map, stats) where stats contains latency, token
            counts, and estimated cost.
        """
        docs_text = "\n\n".join(
            f"[{i+1}] Title: {d['title']}\n{d['snippet']}"
            for i, d in enumerate(retrieved_docs)
        )
        map_json = json.dumps(theme_map, indent=2)
        user_message = (
            f"Source documents:\n{docs_text}\n\n"
            f"Market map to verify:\n{map_json}\n\n"
            "Return the verified market map with unsupported entries removed."
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
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            return json.loads(raw), stats
        except json.JSONDecodeError as e:
            raise ValueError(f"Critic returned invalid JSON: {e}\nRaw output:\n{raw}")
