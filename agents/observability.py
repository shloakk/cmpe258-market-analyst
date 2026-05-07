"""Optional Langfuse observability helpers for the research pipeline.

The pipeline should run in local development and CI even when Langfuse keys are
not configured. This module centralizes the conditional Langfuse integration so
agent code can be instrumented without scattering environment checks.
"""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # Environment variables may already be supplied by CI or the shell.
    pass

try:
    from langfuse import get_client, observe as _langfuse_observe, propagate_attributes
except Exception:  # pragma: no cover - exercised only when langfuse is absent
    get_client = None
    _langfuse_observe = None
    propagate_attributes = None


def observe(*args: Any, **kwargs: Any) -> Callable[[F], F] | F:
    """Conditionally apply ``langfuse.observe`` or return a no-op decorator.

    Args:
        *args: Positional decorator arguments.
        **kwargs: Keyword decorator arguments.

    Returns:
        The Langfuse decorator when tracing is configured, otherwise the wrapped
        function unchanged. This keeps local/CI runs quiet when keys are blank.
    """
    if _langfuse_observe is not None and langfuse_enabled():
        return _langfuse_observe(*args, **kwargs)

    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]

    def decorator(func: F) -> F:
        """Return the original function when Langfuse tracing is disabled."""
        return func

    return decorator


def langfuse_enabled() -> bool:
    """Return whether Langfuse tracing is configured for this process.

    Returns:
        True when both required Langfuse API keys are present and the SDK import
        succeeded; otherwise False.
    """
    return bool(
        get_client is not None
        and _langfuse_observe is not None
        and propagate_attributes is not None
        and os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
    )


def create_trace_id(seed: str | None = None) -> str:
    """Create a 32-character trace id for one pipeline run.

    Args:
        seed: Optional deterministic seed. When Langfuse is configured we defer
            to the SDK's trace id generator; otherwise we use a random UUID.

    Returns:
        A lowercase 32-character hex string suitable for Langfuse trace context.
    """
    if langfuse_enabled():
        try:
            return get_client().create_trace_id(seed=seed)
        except Exception:
            # Keep the pipeline usable if Langfuse initialization is unhealthy.
            pass
    return uuid.uuid4().hex


@contextmanager
def pipeline_observation(
    trace_id: str,
    query: str,
    model: str,
) -> Iterator[Any | None]:
    """Open the root Langfuse span for a Scout -> Mapper -> Critic run.

    Args:
        trace_id: Trace id stored in ``ResearchState`` and shared by all spans.
        query: User query, recorded as trace input.
        model: Short model id used by Mapper and Critic.

    Yields:
        The active Langfuse root span when tracing is enabled, otherwise None.
    """
    if not langfuse_enabled():
        yield None
        return

    langfuse = get_client()
    with langfuse.start_as_current_observation(
        as_type="span",
        name="research-pipeline",
        input={"query": query, "model": model},
        trace_context={"trace_id": trace_id},
    ) as root_span:
        with propagate_attributes(
            session_id=trace_id,
            trace_name="market-research-query",
            metadata={"model": model, "pipeline": "scout-mapper-critic"},
        ):
            yield root_span


def update_current_span(**kwargs: Any) -> None:
    """Best-effort update for the active Langfuse span.

    Args:
        **kwargs: Attributes accepted by ``Langfuse.update_current_span``.
    """
    if not langfuse_enabled():
        return
    try:
        get_client().update_current_span(**kwargs)
    except Exception:
        pass


def update_current_generation(**kwargs: Any) -> None:
    """Best-effort update for the active Langfuse generation observation.

    Args:
        **kwargs: Attributes accepted by ``Langfuse.update_current_generation``.
    """
    if not langfuse_enabled():
        return
    try:
        get_client().update_current_generation(**kwargs)
    except Exception:
        pass
