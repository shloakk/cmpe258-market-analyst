"""
Mapper Agent: given retrieved documents, uses Claude to cluster companies
into named themes (e.g., multi-agent orchestration, RAG tools, evaluation).
Returns a structured market map with citations for each theme.
"""

import json
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage

MODEL = "claude-sonnet-4-6"

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

    def run(self, query: str, retrieved_docs: list[dict]) -> list[dict]:
        """
        Args:
            query: The original user query.
            retrieved_docs: Documents returned by the Scout agent.

        Returns:
            List of theme dicts: {theme_name, companies, rationale, citations}.
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
        response = self.llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ]
        )
        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Mapper returned invalid JSON: {e}\nRaw output:\n{raw}")
