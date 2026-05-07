"""
Mapper Agent: clusters retrieved documents into market themes using a
configurable LLM (Claude / GPT / Llama).

The model is selected at construction time (``model="claude" | "gpt" | "llama"``)
so the same pipeline can be compiled per model and run side-by-side for the
multi-LLM comparison view without changes to agent code.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from agents.llm_client import LLMClient, ModelId, truncate_docs_to_budget
from agents.parsers import parse_theme_map_response

# The schema example in the prompt MUST match agents/schemas.Theme exactly
# (theme_name / companies / rationale / citations). When updating Theme,
# update this prompt too — the parser will reject responses that drift.
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
    """Clusters retrieved documents into market themes using a configurable LLM."""

    def __init__(self, model: ModelId = "llama") -> None:
        """
        Args:
            model: Short model id ("claude", "gpt", or "llama"). Determines
                which provider/model the LLMClient wraps. Defaults to "llama"
                (Groq, free tier).
        """
        self.client = LLMClient(model=model, temperature=0.0)

    def run(
        self, query: str, retrieved_docs: list[dict]
    ) -> tuple[list[dict], dict]:
        """Cluster ``retrieved_docs`` into themes via the configured LLM.

        Args:
            query: The original user query.
            retrieved_docs: Documents from Scout. Keys: title, source_url,
                tags, publish_date, snippet, score.

        Returns:
            Tuple of ``(theme_map, stats)`` where:
                theme_map: list[dict] with keys theme_name, companies,
                    rationale, citations.
                stats: dict with keys model, input_tokens, output_tokens,
                    cost_usd, latency_ms (totals across any JSON-retry).

        Raises:
            ValueError: If the LLM response cannot be parsed even after one
                retry. The original raw output is included in the message.
        """
        # Shrink snippets if the prompt would overrun the model's context.
        # Critic applies the same truncation independently on the raw
        # retrieved_docs so it sees the same evidence the Mapper saw.
        docs = truncate_docs_to_budget(retrieved_docs, self.client.provider_model)

        docs_text = "\n\n".join(
            f"[{i + 1}] Title: {d['title']}\n{d['snippet']}"
            for i, d in enumerate(docs)
        )
        user_message = (
            f"User query: {query}\n\n"
            f"Retrieved documents:\n{docs_text}\n\n"
            "Based on these documents, produce a structured market map."
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        theme_map, response = self.client.invoke_with_json_retry(
            messages, parse_theme_map_response
        )
        stats = {
            "model": response.model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
            "latency_ms": response.latency_ms,
        }
        return theme_map, stats
