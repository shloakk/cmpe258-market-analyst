"""
Critic Agent: reviews the Mapper's theme map against the retrieved documents.
Removes claims or companies that lack grounding in the source documents,
and ensures every remaining entry has a supporting citation.
"""

import json
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage

MODEL = "claude-sonnet-4-6"

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
        self.llm = ChatAnthropic(model=MODEL, temperature=0)

    def run(self, theme_map: list[dict], retrieved_docs: list[dict]) -> list[dict]:
        """
        Args:
            theme_map: Output from the Mapper agent.
            retrieved_docs: The original source documents from the Scout agent.

        Returns:
            Verified market map with unsupported claims removed.
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
        response = self.llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ]
        )
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Critic returned invalid JSON: {e}\nRaw output:\n{raw}")
