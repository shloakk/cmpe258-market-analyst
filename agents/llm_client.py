"""
Unified LLM client wrapping Anthropic, OpenAI, and Groq behind a single
``invoke()`` interface.

Why a thin custom wrapper instead of using LangChain's BaseChatModel directly?
- We need consistent token-usage extraction across providers; each SDK exposes
  ``usage_metadata`` slightly differently and we want one canonical struct.
- We need a per-model cost table in one place so per-model cost comparisons
  in the eval report stay accurate as pricing changes (and so adding a new
  provider is one entry, not a new code path in every agent).
- It is the natural place to attach uniform JSON-retry logic and (Phase 4)
  Langfuse ``@observe()`` instrumentation without polluting agent code.

Public surface:
- ``ModelId`` — short id used through the pipeline ("claude" / "gpt" / "llama" / "gemini").
- ``LLMResponse`` — provider-agnostic response dataclass.
- ``LLMClient`` — the wrapper itself, with ``invoke`` and
  ``invoke_with_json_retry``.
- ``truncate_docs_to_budget`` — shrinks retrieved-doc snippets to fit a
  model's context window before they are inlined into the prompt.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Literal, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from agents.observability import observe, update_current_generation

ModelId = Literal["claude", "gpt", "llama", "gemini"]

# Maps the short model id used through the pipeline to the provider's concrete
# model name. Adding a new model requires one entry here + one entry in
# _PRICING + (if it's a new provider) one branch in _build_client.
MODEL_REGISTRY: dict[ModelId, str] = {
    "claude": "claude-sonnet-4-6",
    # GPT-5.4 is the comparable-tier model to Claude Sonnet 4.6 as of May 2026.
    # GPT-5.5 exists but is 2x the price for marginal gains; not worth the
    # eval cost blow-up. We can swap by changing this single entry.
    "gpt": "gpt-5.4",
    # Llama 3.3 70B via Groq satisfies the "one open-source model" requirement
    # and Groq's throughput (~280 tok/s) keeps the demo latency tolerable.
    "llama": "llama-3.3-70b-versatile",
    # Google AI Studio (Gemini Developer API).
    # Default to a widely-available Flash model; override with env GEMINI_MODEL
    # if your account exposes different ids (see ListModels).
    "gemini": "gemini-2.0-flash",
}

# Per-1M-token USD pricing. Verified against provider docs on 2026-05-06.
# Update when providers change rates; the per-model cost reported by the
# eval report depends on this table being current.
_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
}

# Conservative input-token budgets per model. Real context windows are larger
# but we leave headroom for the system prompt + completion so a long retrieval
# never causes a mid-stream truncation error from the provider. These are
# applied via ``truncate_docs_to_budget`` before the prompt is built.
_CONTEXT_BUDGETS: dict[str, int] = {
    "claude-sonnet-4-6": 180_000,        # 200k window, 20k headroom for output
    "gpt-5.4": 380_000,                  # 400k window, 20k headroom for output
    "llama-3.3-70b-versatile": 100_000,  # 131k window, 31k headroom for output
    "gemini-2.0-flash": 100_000,         # Conservative; keep prompts stable for JSON.
}


@dataclass
class LLMResponse:
    """Provider-agnostic response from an LLM call.

    Attributes:
        text: Raw response content (markdown fences are NOT stripped here;
            that is the parser's job).
        input_tokens: Prompt token count reported by the provider.
        output_tokens: Completion token count reported by the provider.
        cost_usd: Computed cost using the ``_PRICING`` table.
        latency_ms: Wall-clock duration of the ``invoke()`` call.
        model: Concrete provider model name (e.g. "claude-sonnet-4-6"),
            useful for logs / eval where the short id is ambiguous.
    """

    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    model: str


T = TypeVar("T")


class LLMClient:
    """Thin uniform interface over Anthropic, OpenAI, and Groq chat models."""

    def __init__(self, model: ModelId = "claude", temperature: float = 0.0) -> None:
        """
        Args:
            model: Short id selecting the provider/model. Must be a key of
                ``MODEL_REGISTRY``.
            temperature: Sampling temperature passed through to the provider.
                Defaults to 0 because the agents (Mapper/Critic) require
                deterministic JSON output.

        Raises:
            ValueError: If ``model`` is not a known short id.
        """
        if model not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model id: {model!r}. Available: {list(MODEL_REGISTRY)}"
            )
        self.model_id: ModelId = model
        # Allow overriding concrete provider model via env vars for local experiments.
        # Useful for Gemini where model availability can differ across accounts.
        if model == "gemini":
            self.provider_model = os.getenv("GEMINI_MODEL") or MODEL_REGISTRY[model]
        else:
            self.provider_model = MODEL_REGISTRY[model]
        self.temperature = temperature
        self._client = self._build_client()

    def _build_client(self):
        """Instantiate the provider SDK client for the selected model.

        ``langchain_groq`` is imported lazily so a user without ``GROQ_API_KEY``
        installed (or without the package) can still run the Claude / GPT
        paths without an ImportError at module load.
        """
        if self.model_id == "claude":
            _require_any_env(
                ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"],
                "claude",
            )
            return ChatAnthropic(model=self.provider_model, temperature=self.temperature)
        if self.model_id == "gpt":
            _require_any_env(["OPENAI_API_KEY"], "gpt")
            return ChatOpenAI(model=self.provider_model, temperature=self.temperature)
        if self.model_id == "llama":
            _require_any_env(["GROQ_API_KEY"], "llama")
            from langchain_groq import ChatGroq

            return ChatGroq(model=self.provider_model, temperature=self.temperature)
        if self.model_id == "gemini":
            # Google AI Studio uses the Gemini Developer API. The LangChain wrapper
            # reads the key from the environment (commonly GOOGLE_API_KEY). We also
            # accept GEMINI_API_KEY as an alias to reduce local setup friction.
            from langchain_google_genai import ChatGoogleGenerativeAI

            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                _require_any_env(["GOOGLE_API_KEY", "GEMINI_API_KEY"], "gemini")
            return ChatGoogleGenerativeAI(
                model=self.provider_model,
                temperature=self.temperature,
                google_api_key=api_key,
            )
        raise ValueError(f"No client builder for model id: {self.model_id!r}")

    @observe(
        name="llm-invoke",
        as_type="generation",
        capture_input=False,
        capture_output=False,
    )
    def invoke(self, messages: list[BaseMessage]) -> LLMResponse:
        """Send a message list to the model and return a uniform response struct.

        Args:
            messages: LangChain BaseMessage list (System / Human / AI).

        Returns:
            ``LLMResponse`` with text, token counts, cost, latency, and the
            concrete provider model name.
        """
        update_current_generation(
            model=self.provider_model,
            input=_messages_for_trace(messages),
            metadata={"model_id": self.model_id, "temperature": self.temperature},
        )
        t0 = time.perf_counter()
        response = self._client.invoke(messages)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        # All three providers populate ``usage_metadata`` via LangChain's
        # standard interface; default to 0 if the provider regresses or
        # returns it under a different key, so cost reporting degrades
        # gracefully instead of taking down the pipeline.
        usage = getattr(response, "usage_metadata", None) or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        cost_usd = self._compute_cost(input_tokens, output_tokens)

        # Some providers return content as a list of blocks rather than a string
        # (e.g. Gemini often returns `[{"type":"text","text":"..."}]`). Normalize
        # to plain text so the downstream JSON parser sees a stable shape.
        content = response.content
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                else:
                    parts.append(str(block))
            text = "".join(parts)
        else:
            text = str(content)

        llm_response = LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            model=self.provider_model,
        )
        update_current_generation(
            output=_truncate_for_trace(text),
            usage_details={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
            cost_details={"total_usd": cost_usd},
            metadata={"latency_ms": latency_ms},
        )
        return llm_response

    def invoke_with_json_retry(
        self,
        messages: list[BaseMessage],
        parser: Callable[[str], T],
    ) -> tuple[T, LLMResponse]:
        """Invoke the model and parse the response, retrying once on parse failure.

        On the first parse failure we append the model's bad response and a
        corrective nudge ("respond ONLY with valid JSON matching the schema")
        and re-invoke. We deliberately retry only once because empirically
        capable models recover on the first nudge or never; further retries
        mostly burn cost.

        Args:
            messages: Initial message list to send.
            parser: Function that takes the raw response text and returns the
                parsed value (or raises ``ValueError`` on failure).

        Returns:
            ``(parsed_value, combined_response)``. If a retry happened,
            ``combined_response`` sums tokens / cost / latency from both calls
            so eval cost reporting reflects true cost-per-query including
            retries (otherwise we'd undercount the JSON-fragile models).

        Raises:
            ValueError: If the retry response also fails to parse. The error
                message includes both the original failure and the retry's
                raw output for debugging.
        """
        first = self.invoke(messages)
        try:
            return parser(first.text), first
        except ValueError as first_error:
            retry_messages: list[BaseMessage] = list(messages) + [
                AIMessage(content=first.text),
                HumanMessage(
                    content=(
                        "Your previous response was not valid JSON or did not match "
                        "the required schema. Respond ONLY with valid JSON matching "
                        "the schema. Do not include markdown code fences, prose, or "
                        "any explanation outside the JSON."
                    )
                ),
            ]
            second = self.invoke(retry_messages)
            try:
                parsed = parser(second.text)
            except ValueError as second_error:
                raise ValueError(
                    f"JSON retry also failed.\n"
                    f"First error: {first_error}\n"
                    f"Retry error: {second_error}"
                ) from second_error

            combined = LLMResponse(
                text=second.text,
                input_tokens=first.input_tokens + second.input_tokens,
                output_tokens=first.output_tokens + second.output_tokens,
                cost_usd=round(first.cost_usd + second.cost_usd, 6),
                latency_ms=round(first.latency_ms + second.latency_ms, 1),
                model=second.model,
            )
            return parsed, combined

    def _compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Compute USD cost using the per-1M-token rates in ``_PRICING``.

        Returns 0.0 (rather than raising) for models missing from the table so
        a newly added model can run end-to-end before its pricing is wired in;
        this avoids breaking smoke tests during incremental rollout.
        """
        rates = _PRICING.get(self.provider_model)
        if rates is None:
            return 0.0
        cost = (
            input_tokens * rates["input"] / 1_000_000
            + output_tokens * rates["output"] / 1_000_000
        )
        return round(cost, 6)


def truncate_docs_to_budget(
    docs: list[dict],
    model: str,
    overhead_tokens: int = 2_000,
    chars_per_token: int = 4,
) -> list[dict]:
    """Shrink retrieved-doc snippets so the prompt fits the model's context window.

    Approximates token count with ``chars / 4`` (industry rule of thumb; off
    by 10–20% for English but adequate for budgeting). We truncate each snippet
    proportionally rather than dropping docs entirely because the Critic must
    see the same documents the Mapper saw to verify claims — losing a doc
    would cause the Critic to wrongly flag supported claims as hallucinated.

    Args:
        docs: List of doc dicts with key ``"snippet"`` (other keys are
            preserved verbatim).
        model: Concrete provider model name to look up the budget for. If the
            model has no entry in ``_CONTEXT_BUDGETS``, a conservative default
            of 100k tokens is used.
        overhead_tokens: Estimated tokens consumed by the system prompt,
            user query, and JSON formatting around the docs.
        chars_per_token: Approximation factor (4 chars/token is conservative
            for English text).

    Returns:
        New list of doc dicts with snippets truncated if needed; the input
        list is never mutated. If no truncation is needed, returns the input
        list unchanged (same reference).
    """
    budget_tokens = _CONTEXT_BUDGETS.get(model, 100_000)
    char_budget = max(0, (budget_tokens - overhead_tokens) * chars_per_token)
    total_chars = sum(len(d.get("snippet", "")) for d in docs)
    if total_chars <= char_budget:
        return docs

    # Each doc gets a proportional slice so the smallest doc isn't wiped out
    # entirely. The "[truncated]" marker tells the model not to treat the
    # snippet as complete evidence (matters for Critic correctness).
    ratio = char_budget / total_chars if total_chars else 0.0
    truncated: list[dict] = []
    for d in docs:
        new_d = dict(d)
        snippet = d.get("snippet", "")
        # Floor at 200 chars so each doc retains some signal even when the
        # total corpus dwarfs the context window.
        max_chars = max(200, int(len(snippet) * ratio))
        if len(snippet) > max_chars:
            new_d["snippet"] = snippet[:max_chars] + "... [truncated]"
        truncated.append(new_d)
    return truncated


def _messages_for_trace(messages: list[BaseMessage]) -> list[dict[str, str]]:
    """Convert LangChain messages into a compact trace-safe representation.

    Args:
        messages: Message list about to be sent to a provider.

    Returns:
        list[dict] with keys: role, content.
    """
    return [
        {
            "role": getattr(message, "type", message.__class__.__name__),
            "content": _truncate_for_trace(str(message.content)),
        }
        for message in messages
    ]


def _truncate_for_trace(value: str, max_chars: int = 8_000) -> str:
    """Limit prompt/response text stored in Langfuse observations.

    Args:
        value: Raw text to record.
        max_chars: Maximum number of characters to keep.

    Returns:
        The original text if short enough, otherwise a truncated string with a
        marker explaining that observability storage was capped.
    """
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "... [truncated for trace]"


def _require_any_env(names: list[str], model_id: str) -> None:
    """Raise a clear error when a selected provider has no API key configured.

    Args:
        names: Environment variable names accepted by the provider.
        model_id: Short model id the user selected.

    Raises:
        ValueError: If none of ``names`` is present in the environment.
    """
    if any(os.getenv(name) for name in names):
        return
    joined = " or ".join(names)
    raise ValueError(
        f"Missing API key for model {model_id!r}. Set {joined} in .env, "
        "or use a configured model by setting DEFAULT_MODEL=gemini/llama/gpt."
    )
