"""
Shared parsing logic for LLM responses produced by Mapper and Critic.

Both agents emit the same JSON theme-map schema, so the fence-stripping +
validation flow is identical. Centralizing it here:
- Eliminates the duplicated try/except blocks that lived in each agent.
- Means the JSON-retry layer in LLMClient has one canonical parser to call.
- Lets us tighten validation (e.g. enforce non-empty companies) in one place.
"""

import json

from pydantic import ValidationError

from agents.schemas import ThemeMap


def parse_theme_map_response(raw: str) -> list[dict]:
    """Parse a raw LLM response string into a validated theme map.

    LLMs sometimes wrap JSON in markdown code fences (``` ```json ... ``` ```);
    we strip those before ``json.loads`` to avoid a JSONDecodeError on the
    leading backticks. After parsing, we run the result through the Pydantic
    ``ThemeMap`` schema so missing keys or wrong types are caught at the agent
    boundary instead of crashing downstream code that assumes the dict shape.

    Args:
        raw: Raw text content returned by the LLM.

    Returns:
        list[dict] with keys: theme_name, companies, rationale, citations.

    Raises:
        ValueError: Wraps both ``json.JSONDecodeError`` and
            ``pydantic.ValidationError`` so callers (notably the JSON-retry
            layer in ``LLMClient``) can handle both failure modes uniformly.
            The raw output is included in the message so the retry prompt
            can show the model what it produced.
    """
    cleaned = raw.strip()
    # Strip ```json ... ``` or ``` ... ``` wrappers that some models add despite
    # the "respond ONLY with valid JSON" instruction. We only look at the first
    # fenced block because anything after it would be commentary, which we
    # also want to discard.
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 2:
            cleaned = parts[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM response is not valid JSON: {e}\nRaw output:\n{raw}"
        ) from e

    if not isinstance(parsed, list):
        raise ValueError(
            f"LLM response must be a JSON array of themes, got "
            f"{type(parsed).__name__}.\nRaw output:\n{raw}"
        )

    try:
        return ThemeMap.from_raw(parsed).to_dict_list()
    except ValidationError as e:
        raise ValueError(
            f"LLM response failed schema validation: {e}\nRaw output:\n{raw}"
        ) from e
