"""
Pydantic schemas for the inter-agent data contracts in the pipeline.

Centralizing the Theme / ThemeMap shape here means:
- Mapper output and Critic output are validated against the same schema, so
  a malformed map can never silently propagate to the API or eval layer.
- The FastAPI ThemeResult model and the eval metrics can rely on a single
  source of truth for field names and types.
- Future schema changes (e.g. adding a confidence score) happen in one place.
"""

from pydantic import BaseModel, Field


class Theme(BaseModel):
    """A single market theme produced by Mapper or Critic.

    Attributes:
        theme_name: Short label for the cluster (e.g. "Multi-Agent Orchestration").
        companies: Company / product names placed in this theme.
        rationale: One- or two-sentence explanation of why these companies belong together.
        citations: Document titles from the retrieved corpus that support the placement.
    """

    theme_name: str = Field(min_length=1)
    companies: list[str] = Field(default_factory=list)
    rationale: str = ""
    citations: list[str] = Field(default_factory=list)


class ThemeMap(BaseModel):
    """A full market map: an ordered list of Theme objects.

    Wrapping the list in a model (rather than using ``list[Theme]`` directly)
    gives us a place to attach map-level helpers like ``from_raw`` and
    ``to_dict_list`` without scattering conversion logic across the agents.
    """

    themes: list[Theme]

    @classmethod
    def from_raw(cls, raw: list[dict]) -> "ThemeMap":
        """Validate a raw list[dict] (e.g. from ``json.loads``) into a ThemeMap.

        Args:
            raw: List of dicts with keys ``theme_name``, ``companies``,
                ``rationale``, ``citations``.

        Returns:
            A validated ThemeMap.

        Raises:
            pydantic.ValidationError: If any entry is missing required fields
                or has wrong types. Callers should treat this as a hard failure
                (or trigger a JSON retry against the LLM).
        """
        return cls(themes=[Theme(**t) for t in raw])

    def to_dict_list(self) -> list[dict]:
        """Convert back to the list[dict] shape consumed elsewhere in the pipeline.

        Returns:
            list[dict] with keys: theme_name, companies, rationale, citations.
        """
        return [t.model_dump() for t in self.themes]
