"""
Critic Agent: reviews the Mapper's theme map against the retrieved documents
and removes claims/companies/themes that lack grounding in the source text.

The model is configurable (``"claude" | "gpt" | "llama"``) for the multi-LLM
comparison view. By default Critic uses the same model as Mapper to keep the
ablation honest, but the constructor exposes the choice so a downstream caller
can mix models (e.g. cheap Mapper + strong Critic) for cost ablations.
"""

import json

from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm_client import LLMClient, ModelId, truncate_docs_to_budget
from agents.parsers import parse_theme_map_response

# Schema example here MUST match agents/schemas.Theme exactly. The parser
# will reject responses that drift from the four expected keys.
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
    """Verifies a market map against source documents and prunes unsupported claims."""

    def __init__(self, model: ModelId = "claude") -> None:
        """
        Args:
            model: Short model id ("claude", "gpt", or "llama").
        """
        self.client = LLMClient(model=model, temperature=0.0)

    def run(
        self, theme_map: list[dict], retrieved_docs: list[dict]
    ) -> tuple[list[dict], dict]:
        """Verify ``theme_map`` against ``retrieved_docs`` via the configured LLM.

        Args:
            theme_map: Output from MapperAgent. Keys per entry: theme_name,
                companies, rationale, citations.
            retrieved_docs: Original source documents from ScoutAgent. Keys:
                title, source_url, tags, publish_date, snippet, score.

        Returns:
            Tuple of ``(verified_map, stats)``:
                verified_map: list[dict] with the same keys as ``theme_map``
                    but with unsupported entries removed.
                stats: dict with keys model, input_tokens, output_tokens,
                    cost_usd, latency_ms (totals across any JSON-retry).

        Raises:
            ValueError: If the LLM response cannot be parsed even after one
                retry.
        """
        # Same truncation as Mapper so Critic sees the same evidence the
        # Mapper saw. If we let them diverge, Critic could flag a Mapper
        # claim as unsupported just because its (different) truncation
        # dropped the supporting sentence.
        docs = truncate_docs_to_budget(retrieved_docs, self.client.provider_model)

        docs_text = "\n\n".join(
            f"[{i + 1}] Title: {d['title']}\n{d['snippet']}"
            for i, d in enumerate(docs)
        )
        map_json = json.dumps(theme_map, indent=2)
        user_message = (
            f"Source documents:\n{docs_text}\n\n"
            f"Market map to verify:\n{map_json}\n\n"
            "Return the verified market map with unsupported entries removed."
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        verified_map, response = self.client.invoke_with_json_retry(
            messages, parse_theme_map_response
        )
        stats = {
            "model": response.model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
            "latency_ms": response.latency_ms,
        }
        return verified_map, stats
